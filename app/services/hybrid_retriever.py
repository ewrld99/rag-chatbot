"""
app/services/hybrid_retriever.py
----------------------------------
HybridRetriever fuses dense (pgvector) and sparse (PostgreSQL FTS) retrieval
via Reciprocal Rank Fusion, expands neighboring chunks, and exposes
LangChain-compatible Document output.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
import time
from typing import Any
from uuid import UUID

from langchain_core.documents import Document
from sqlalchemy import tuple_
from sqlalchemy.orm import Session, joinedload

from app.core.config import settings
from app.core.logging import format_log_event
from app.db.models import DocumentChunk
from app.db.session import SessionLocal
from app.services.dense_retriever import (
    DenseRetriever,
    _set_embedding_failed,
    dense_embedding_failed,
)
from app.services.rrf import ReciprocalRankFusion, RRFResult
from app.services.sparse_retriever import SparseRetriever
from app.services.settings_service import SettingsService

logger = logging.getLogger(__name__)


NEIGHBOR_WINDOW = 1


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
    Orchestrates dense + sparse retrieval, RRF fusion, and neighbor expansion.
    """

    def __init__(
        self,
        db: Session,
        top_k: int | None = None,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        rrf_k: int | None = None,
        neighbor_window: int = NEIGHBOR_WINDOW,
    ) -> None:
        self.db = db
        self.top_k: int
        self.dense_top_k: int
        self.sparse_top_k: int
        self.rrf_k: int
        self.neighbor_window = max(neighbor_window, 0)

        if any(v is None for v in [top_k, dense_top_k, sparse_top_k, rrf_k]):
            settings_svc = SettingsService(db)
            self.top_k = top_k if top_k is not None else settings_svc.top_k_final
            self.dense_top_k = dense_top_k if dense_top_k is not None else settings_svc.top_k_dense
            self.sparse_top_k = sparse_top_k if sparse_top_k is not None else settings_svc.top_k_sparse
            self.rrf_k = rrf_k if rrf_k is not None else settings_svc.rrf_k
        else:
            assert top_k is not None
            assert dense_top_k is not None
            assert sparse_top_k is not None
            assert rrf_k is not None
            self.top_k = top_k
            self.dense_top_k = dense_top_k
            self.sparse_top_k = sparse_top_k
            self.rrf_k = rrf_k

    def retrieve_raw(
        self,
        query: str,
        filters: dict | None = None,
        pool_size: int | None = None,
        include_neighbors: bool = True,
    ) -> list[RRFResult]:
        """
        Run hybrid retrieval and return raw RRFResult objects.

        The sparse retriever performs raw, normalized, expanded, and fallback
        keyword searches internally. This method fuses dense+sparse candidates
        and then expands neighboring document chunks around selected hits.
        """
        effective_size = pool_size if pool_size is not None else self.top_k

        rrf = ReciprocalRankFusion()

        started_at = time.perf_counter()
        if settings.HYBRID_PARALLEL_RETRIEVAL:
            dense_results, sparse_results, dense_ms, sparse_ms = self._retrieve_parallel(
                query,
                filters,
            )
            retrieval_mode = "parallel"
        else:
            dense_results, sparse_results, dense_ms, sparse_ms = self._retrieve_sequential(
                query,
                filters,
                self.db,
            )
            retrieval_mode = "sequential"

        logger.debug(
            "HybridRetriever: dense=%d sparse=%d query=%r",
            len(dense_results),
            len(sparse_results),
            query[:80],
        )

        # Parallel retrieval propagates the dense worker's failure flag back
        # to this thread, so one check covers both sequential and parallel mode.
        embedding_down = dense_embedding_failed()
        if embedding_down:
            logger.warning(
                "HybridRetriever: running in SPARSE-ONLY degraded mode because "
                "the embedding service is unavailable. Dense retrieval is disabled "
                "for this request. query=%r",
                query[:120],
            )

        fusion_started_at = time.perf_counter()
        fused = rrf.fuse(dense_results, sparse_results, k=self.rrf_k)
        fusion_ms = round((time.perf_counter() - fusion_started_at) * 1000, 1)
        selected = fused[:effective_size]

        # Stamp degraded flag so _assess_retrieval can widen its thresholds.
        if embedding_down:
            for result in selected:
                result.metadata["dense_retrieval_degraded"] = True

        logger.info(
            format_log_event(
                "Hybrid retrieval latency",
                total_ms=round((time.perf_counter() - started_at) * 1000, 1),
                dense_ms=dense_ms,
                sparse_ms=sparse_ms,
                fusion_ms=fusion_ms,
                dense_results=len(dense_results),
                sparse_results=len(sparse_results),
                fused=len(fused),
                selected=len(selected),
                include_neighbors=include_neighbors,
                mode=retrieval_mode,
                degraded=embedding_down,
                query=query[:120],
            )
        )
        if not include_neighbors:
            return selected
        return self.expand_neighbor_chunks(selected)

    def _retrieve_parallel(
        self,
        query: str,
        filters: dict | None,
    ) -> tuple[list[Any], list[Any], float, float]:
        def dense_task() -> tuple[list[Any], float, bool]:
            with SessionLocal() as db:
                started_at = time.perf_counter()
                results = DenseRetriever(db).retrieve(
                    query,
                    top_k=self.dense_top_k,
                    filters=filters,
                )
                failed = dense_embedding_failed()
                return results, round((time.perf_counter() - started_at) * 1000, 1), failed

        def sparse_task() -> tuple[list[Any], float]:
            with SessionLocal() as db:
                started_at = time.perf_counter()
                results = SparseRetriever(db).retrieve(
                    query,
                    top_k=self.sparse_top_k,
                    filters=filters,
                )
                return results, round((time.perf_counter() - started_at) * 1000, 1)

        try:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="hybrid-retrieval") as executor:
                dense_future = executor.submit(dense_task)
                sparse_future = executor.submit(sparse_task)
                dense_results, dense_ms, dense_failed = dense_future.result()
                sparse_results, sparse_ms = sparse_future.result()
            # Propagate this request's result in both directions so a previous
            # failure cannot poison a later request on the coordinator thread.
            _set_embedding_failed(dense_failed)
            return dense_results, sparse_results, dense_ms, sparse_ms
        except Exception as exc:
            logger.warning(
                "HybridRetriever: parallel retrieval failed; retrying sequentially: %s",
                exc,
            )
            return self._retrieve_sequential(query, filters, self.db)

    def _retrieve_sequential(
        self,
        query: str,
        filters: dict | None,
        db: Session,
    ) -> tuple[list[Any], list[Any], float, float]:
        dense_started_at = time.perf_counter()
        dense_results = DenseRetriever(db).retrieve(
            query,
            top_k=self.dense_top_k,
            filters=filters,
        )
        dense_ms = round((time.perf_counter() - dense_started_at) * 1000, 1)
        sparse_started_at = time.perf_counter()
        sparse_results = SparseRetriever(db).retrieve(
            query,
            top_k=self.sparse_top_k,
            filters=filters,
        )
        sparse_ms = round((time.perf_counter() - sparse_started_at) * 1000, 1)
        return dense_results, sparse_results, dense_ms, sparse_ms

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> list[RRFResult]:
        """Compatibility alias returning raw fused RRF results."""
        return self.retrieve_raw(query, filters=filters, pool_size=top_k)

    def invoke(self, query: str, filters: dict | None = None, **kwargs: Any) -> list[Document]:
        """LangChain standard retriever interface."""
        return [_rrf_to_langchain(r) for r in self.retrieve_raw(query, filters=filters)]

    def get_relevant_documents(self, query: str, filters: dict | None = None, **kwargs: Any) -> list[Document]:
        """Legacy LangChain alias."""
        return self.invoke(query, filters=filters, **kwargs)

    def expand_neighbor_chunks(self, results: list[RRFResult]) -> list[RRFResult]:
        """Add adjacent document chunks around selected retrieval hits."""
        return self._expand_neighbor_chunks(results)

    def _expand_neighbor_chunks(self, results: list[RRFResult]) -> list[RRFResult]:
        if self.neighbor_window <= 0 or not results:
            return results

        targets: dict[tuple[UUID, int], tuple[RRFResult, int]] = {}
        original_chunk_ids = {result.chunk_id for result in results}

        for result in results:
            if result.metadata.get("source_type") != "document":
                continue
            chunk_index = result.metadata.get("chunk_index")
            if not isinstance(chunk_index, int):
                continue
            try:
                document_id = UUID(result.document_id)
            except (TypeError, ValueError):
                continue

            for offset in range(-self.neighbor_window, self.neighbor_window + 1):
                if offset == 0:
                    continue
                neighbor_index = chunk_index + offset
                if neighbor_index < 0:
                    continue
                targets[(document_id, neighbor_index)] = (result, offset)

        if not targets:
            return results

        chunks = (
            self.db.query(DocumentChunk)
            .options(joinedload(DocumentChunk.document))
            .filter(tuple_(DocumentChunk.document_id, DocumentChunk.chunk_index).in_(list(targets.keys())))
            .all()
        )

        neighbors_by_parent: dict[str, list[RRFResult]] = {}
        for chunk in chunks:
            parent, offset = targets[(chunk.document_id, chunk.chunk_index)]
            chunk_id = str(chunk.id)
            if chunk_id in original_chunk_ids:
                continue

            metadata = dict(chunk.metadata_ or {})
            metadata.update(
                {
                    "source": chunk.document.filename if chunk.document and chunk.document.filename else "database",
                    "chunk_index": chunk.chunk_index,
                    "source_type": "document",
                    "neighbor_expansion": True,
                    "neighbor_of_chunk_id": parent.chunk_id,
                    "neighbor_offset": offset,
                }
            )
            if chunk.page_number is not None:
                metadata["page_number"] = chunk.page_number
            if chunk.document and chunk.document.title:
                metadata["document_title"] = chunk.document.title
            if chunk.document and chunk.document.source_url:
                metadata["source_url"] = chunk.document.source_url
            if chunk.document:
                metadata["document_status"] = chunk.document.status or "active"
                if chunk.document.content_hash:
                    metadata["content_hash"] = chunk.document.content_hash
                if chunk.document.upload_date:
                    metadata["document_uploaded_at"] = chunk.document.upload_date.isoformat()

            neighbor = RRFResult(
                chunk_id=chunk_id,
                document_id=str(chunk.document_id),
                rrf_score=parent.rrf_score * max(0.0, 0.96 - (abs(offset) * 0.02)),
                dense_rank=parent.dense_rank,
                sparse_rank=parent.sparse_rank,
                retrieval_sources=list(parent.retrieval_sources),
                metadata=metadata,
                text=chunk.chunk_text,
            )
            neighbors_by_parent.setdefault(parent.chunk_id, []).append(neighbor)

        # Keep the reranker's central hits together at the front. Interleaving
        # neighbors after each hit lets low-signal surrounding text consume the
        # LLM's small document budget before later, stronger central hits.
        expanded: list[RRFResult] = []
        seen: set[str] = set()
        for result in results:
            if result.chunk_id not in seen:
                expanded.append(result)
                seen.add(result.chunk_id)

        for result in results:
            neighbors = sorted(
                neighbors_by_parent.get(result.chunk_id, []),
                key=lambda item: (abs(item.metadata.get("neighbor_offset", 0)), item.metadata.get("chunk_index", 0), item.chunk_id),
            )
            for neighbor in neighbors:
                if neighbor.chunk_id in seen:
                    continue
                expanded.append(neighbor)
                seen.add(neighbor.chunk_id)

        return expanded
