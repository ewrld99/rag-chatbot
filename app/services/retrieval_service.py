"""
app/services/retrieval_service.py
-----------------------------------
Public façade for the RAG retrieval layer.

The public API (retrieve, retrieve_with_scores, get_context, format_context,
embed_query) is UNCHANGED so RAGPipeline, chat routes, and WebSocket handlers
continue to work without modification.

Internally, retrieval is now delegated to HybridRetriever which fuses
dense (pgvector) and sparse (PostgreSQL FTS) results via Reciprocal Rank
Fusion (RRF).
"""

from __future__ import annotations

from html import escape
import logging
import time
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document
from sqlalchemy.orm import Session

from app.core.logging import format_log_event
from app.services.alias_expansion_service import AliasExpansionService
from app.services.embedding_service import get_embedding
from app.services.grounding_service import GroundingService
from app.services.hybrid_retriever import HybridRetriever
from app.services.reranker_service import RerankService
from app.services.source_service import document_name
from app.services.settings_service import SettingsService


logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Thin façade over HybridRetriever.

    Constructor signature is UNCHANGED:
        RetrievalService(db=session, top_k=5)
    """

    def __init__(
        self,
        db: Session,
        top_k: int = 5,
    ) -> None:
        self.db = db
        self._settings = SettingsService(db)
        self.top_k = top_k if top_k is not None else self._settings.top_k_final
        self._reranker = RerankService()
        self._hybrid = HybridRetriever(
            db=db,
            top_k=self.top_k,
            dense_top_k=self._settings.top_k_dense,
            sparse_top_k=self._settings.top_k_sparse,
            rrf_k=self._settings.rrf_k,
        )

    def set_top_k(self, top_k: int) -> None:
        """Update the final result limit for this request-scoped service."""
        self.top_k = max(1, int(top_k))
        self._hybrid.top_k = self.top_k

    # -----------------------------------------------------------------------
    # 1. Embed Query  (kept for backward compatibility)
    # -----------------------------------------------------------------------
    def embed_query(self, query: str) -> List[float]:
        """Convert user query into embedding vector."""
        return get_embedding(query, self.db)

    # -----------------------------------------------------------------------
    # 2. Retrieve Documents
    # -----------------------------------------------------------------------
    def retrieve(self, query: str, filters: Dict[str, Any] | None = None) -> List[Document]:
        """
        Full hybrid retrieval pipeline → LangChain Documents.

        Calls HybridRetriever which:
          1. Dense  (pgvector cosine)
          2. Sparse (PostgreSQL FTS)
          3. RRF fusion
          4. Reranking (Jina when configured, local lexical fallback otherwise)

        Optional *filters* (keys: 'programme', 'year') narrow results to
        chunks whose JSONB metadata matches. Old chunks without the key
        are still included (IS NULL fallback).
        """
        if self._settings.enable_reranker:
            # Give the reranker the full fused pool from both retrievers.
            # Using only DENSE_TOP_K discards strong sparse-only candidates before
            # Jina ever sees them. The union can be up to DENSE_TOP_K + SPARSE_TOP_K
            # unique chunks after RRF deduplication.
            pool_size = self._settings.top_k_dense + self._settings.top_k_sparse
            raw = self._hybrid.retrieve_raw(
                query,
                filters=filters,
                pool_size=pool_size,
                include_neighbors=False,
            )
            if self._should_skip_neural_rerank(raw):
                reranked = raw[:self.top_k]
                for result in reranked:
                    result.metadata["reranker"] = "skipped_high_confidence"
                    result.metadata["rerank_strategy"] = "rrf_high_confidence"
            else:
                alias_expansions = AliasExpansionService(self.db).get_expansions(query)
                reranked = self._reranker.rerank(
                    query,
                    raw,
                    top_k=self.top_k,
                    alias_expansions=alias_expansions,
                )
            reranked = self._hybrid.expand_neighbor_chunks(reranked)
            from app.services.hybrid_retriever import _rrf_to_langchain
            return [_rrf_to_langchain(r) for r in reranked]
        return self._hybrid.get_relevant_documents(query, filters=filters)

    # -----------------------------------------------------------------------
    # 3. Retrieve with Scores (returns cosine-like rrf_score for compatibility)
    # -----------------------------------------------------------------------
    def retrieve_with_scores(self, query: str, filters: Dict[str, Any] | None = None) -> List[Tuple[Document, float]]:
        """
        Returns (Document, score) pairs.
        Score is the Jina relevance score (if reranker is on) or RRF score.
        """
        if self._settings.enable_reranker:
            started_at = time.perf_counter()
            pool_size = self._settings.top_k_dense + self._settings.top_k_sparse
            raw_started_at = time.perf_counter()
            raw = self._hybrid.retrieve_raw(
                query,
                filters=filters,
                pool_size=pool_size,
                include_neighbors=False,
            )
            candidate_count = len(raw)
            raw_ms = round((time.perf_counter() - raw_started_at) * 1000, 1)
            skip_rerank = self._should_skip_neural_rerank(raw)
            rerank_started_at = time.perf_counter()
            if skip_rerank:
                alias_ms = 0.0
                raw = raw[:self.top_k]
                for result in raw:
                    result.metadata["reranker"] = "skipped_high_confidence"
                    result.metadata["rerank_strategy"] = "rrf_high_confidence"
                rerank_ms = 0.0
            else:
                alias_started_at = time.perf_counter()
                alias_expansions = AliasExpansionService(self.db).get_expansions(query)
                alias_ms = round((time.perf_counter() - alias_started_at) * 1000, 1)
                raw = self._reranker.rerank(
                    query,
                    raw,
                    top_k=self.top_k,
                    alias_expansions=alias_expansions,
                )
                rerank_ms = round((time.perf_counter() - rerank_started_at) * 1000, 1)
            neighbor_started_at = time.perf_counter()
            raw = self._hybrid.expand_neighbor_chunks(raw)
            neighbor_ms = round((time.perf_counter() - neighbor_started_at) * 1000, 1)
            logger.info(
                format_log_event(
                    "Retrieval latency",
                    total_ms=round((time.perf_counter() - started_at) * 1000, 1),
                    alias_ms=alias_ms,
                    raw_hybrid_ms=raw_ms,
                    rerank_ms=rerank_ms,
                    neighbor_ms=neighbor_ms,
                    candidates=candidate_count,
                    returned=len(raw),
                    reranker_enabled=True,
                    rerank_skipped=skip_rerank,
                    query=query[:120],
                )
            )
        else:
            started_at = time.perf_counter()
            raw = self._hybrid.retrieve_raw(query, filters=filters)
            logger.info(
                format_log_event(
                    "Retrieval latency",
                    total_ms=round((time.perf_counter() - started_at) * 1000, 1),
                    returned=len(raw),
                    reranker_enabled=False,
                    query=query[:120],
                )
            )

        output: List[Tuple[Document, float]] = []
        for result in raw:
            doc = Document(
                page_content=result.text,
                metadata={
                    **result.metadata,
                    "document_id": result.document_id,
                    "chunk_id": result.chunk_id,
                    "rrf_score": result.rrf_score,
                    "dense_rank": result.dense_rank,
                    "sparse_rank": result.sparse_rank,
                    "retrieval_sources": result.retrieval_sources,
                },
            )
            score = result.metadata.get("rerank_score", result.rrf_score)
            output.append((doc, score))

        return output

    def _should_skip_neural_rerank(self, raw: List[Any]) -> bool:
        if not self._settings.adaptive_rerank_skip_high_confidence:
            return False
        if len(raw) < self.top_k:
            return False
        top = raw[0] if raw else None
        if top is None:
            return False

        retrieval_sources = set(top.retrieval_sources or [])
        has_hybrid_top = {"dense", "sparse"}.issubset(retrieval_sources)
        dense_rank = top.dense_rank
        sparse_rank = top.sparse_rank
        top_rank_strong = (
            isinstance(dense_rank, int)
            and isinstance(sparse_rank, int)
            and dense_rank <= 3
            and sparse_rank <= 3
        )
        top_rrf = float(top.rrf_score or 0.0)
        return bool(has_hybrid_top and top_rank_strong and top_rrf >= 0.03)

    # -----------------------------------------------------------------------
    # 4. Format Context
    # -----------------------------------------------------------------------
    def format_context(self, documents: List[Document], max_chars: int | None = None) -> str:
        """Convert retrieved documents into structured LLM context.

        max_chars defaults to top_k_final * 2400 so the budget automatically
        scales when the admin increases the number of retrieved chunks.
        """
        if not documents:
            return "No relevant context found."

        # Scale budget with the configured top-k rather than using a hard constant.
        if max_chars is None:
            max_chars = self._settings.top_k_final * 2400

        formatted_chunks: List[str] = []
        current_length = 0
        grounding = GroundingService()
        
        for doc in documents:
            doc_name = escape(document_name(doc.metadata), quote=True)
            document_id = escape(str(doc.metadata.get("document_id", "unknown")), quote=True)
            chunk_index = escape(str(doc.metadata.get("chunk_index", "unknown")), quote=True)
            page_number = escape(str(doc.metadata.get("page_number", "unknown")), quote=True)
            evidence_id = grounding.evidence_id(doc)
            content = escape(doc.page_content.strip(), quote=False)
            chunk_text = (
                f'  <document evidence_id="{evidence_id}" document_id="{document_id}" '
                f'name="{doc_name}" chunk="{chunk_index}" page="{page_number}">\n'
                f'    {content}\n'
                f'  </document>'
            )
            
            if current_length + len(chunk_text) > max_chars:
                overhead = len(chunk_text) - len(content)
                remaining_for_content = max_chars - current_length - overhead - 20 # For "... [TRUNCATED]"

                if remaining_for_content > 0:
                    truncated_content = content[:remaining_for_content] + "... [TRUNCATED]"
                    chunk_text = (
                        f'  <document evidence_id="{evidence_id}" document_id="{document_id}" '
                        f'name="{doc_name}" chunk="{chunk_index}" page="{page_number}">\n'
                        f'    {truncated_content}\n'
                        f'  </document>'
                    )
                    formatted_chunks.append(chunk_text)
                    current_length += len(chunk_text)
                break

            formatted_chunks.append(chunk_text)
            current_length += len(chunk_text)

        if not formatted_chunks:
            return "No relevant context found."

        return "\n".join(formatted_chunks)

    # -----------------------------------------------------------------------
    # 5. Full Pipeline
    # -----------------------------------------------------------------------
    def get_context(self, query: str, filters: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """End-to-end retrieval pipeline — unchanged interface."""
        docs = self.retrieve(query, filters=filters)
        context = self.format_context(docs)
        return {
            "query": query,
            "documents": docs,
            "context": context,
        }
