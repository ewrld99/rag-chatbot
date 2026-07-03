"""
app/services/hybrid_retriever.py
----------------------------------
HybridRetriever fuses dense (pgvector) and sparse (PostgreSQL FTS) retrieval
via Reciprocal Rank Fusion, and exposes LangChain-compatible Document output.

Design note
-----------
We intentionally do NOT subclass LangChain BaseRetriever here because
pydantic v2-based BaseRetriever rejects the SQLAlchemy Session field
without complex model_config gymnastics.  Instead we expose:

  - invoke(query)                  — LangChain standard public interface
  - get_relevant_documents(query)  — legacy compat alias
  - retrieve_raw(query)            — returns raw RRFResult list (for /query)

This is fully compatible with how RAGPipeline and the new /query route
call the retriever.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document
from sqlalchemy.orm import Session

from app.core.config import settings
from app.services.dense_retriever import DenseRetriever
from app.services.rrf import ReciprocalRankFusion, RRFResult
from app.services.sparse_retriever import SparseRetriever
from app.services.settings_service import SettingsService

logger = logging.getLogger(__name__)


def _rrf_to_langchain(result: RRFResult) -> Document:
    """Convert an RRFResult into a LangChain Document."""
    return Document(
        page_content=result.text,
        metadata={
            **result.metadata,
            "chunk_id": result.chunk_id,
            "document_id": result.document_id,
            "rrf_score": round(result.rrf_score, 6),
            "dense_rank": result.dense_rank,
            "sparse_rank": result.sparse_rank,
            "retrieval_sources": result.retrieval_sources,
        },
    )


class HybridRetriever:
    """
    Orchestrates dense + sparse retrieval and fuses results with RRF.

    Parameters
    ----------
    db          : SQLAlchemy session
    top_k       : final number of chunks returned to the caller
    dense_top_k : candidate pool size for pgvector search
    sparse_top_k: candidate pool size for FTS search
    rrf_k       : RRF smoothing constant
    """

    def __init__(
        self,
        db: Session,
        top_k: int | None = None,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        rrf_k: int | None = None,
    ) -> None:
        self.db = db
        settings_svc = SettingsService(db)
        self.top_k = top_k if top_k is not None else settings_svc.top_k_final
        self.dense_top_k = dense_top_k if dense_top_k is not None else settings_svc.top_k_dense
        self.sparse_top_k = sparse_top_k if sparse_top_k is not None else settings_svc.top_k_sparse
        self.rrf_k = rrf_k if rrf_k is not None else settings_svc.rrf_k

    # -----------------------------------------------------------------------
    # Core: RRF fusion, returns raw results
    # -----------------------------------------------------------------------
    def retrieve_raw(self, query: str) -> list[RRFResult]:
        """
        Run hybrid retrieval and return raw RRFResult objects.
        Used by the /query endpoint to expose full rank metadata.
        """
        dense_r = DenseRetriever(self.db)
        sparse_r = SparseRetriever(self.db)
        rrf = ReciprocalRankFusion(k=self.rrf_k)

        dense_results = dense_r.retrieve(query, top_k=self.dense_top_k)
        sparse_results = sparse_r.retrieve(query, top_k=self.sparse_top_k)

        logger.debug(
            "HybridRetriever: dense=%d sparse=%d query=%r",
            len(dense_results),
            len(sparse_results),
            query[:80],
        )

        fused = rrf.fuse(dense_results, sparse_results)
        return fused[: self.top_k]

    # -----------------------------------------------------------------------
    # LangChain-compatible public interfaces
    # -----------------------------------------------------------------------
    def invoke(self, query: str, **kwargs: Any) -> list[Document]:
        """LangChain standard retriever interface (v0.2+)."""
        return [_rrf_to_langchain(r) for r in self.retrieve_raw(query)]

    def get_relevant_documents(self, query: str, **kwargs: Any) -> list[Document]:
        """Legacy LangChain alias — delegates to invoke()."""
        return self.invoke(query, **kwargs)
