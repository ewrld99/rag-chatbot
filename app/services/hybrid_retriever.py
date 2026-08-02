"""
app/services/hybrid_retriever.py
----------------------------------
HybridRetriever fuses dense (pgvector) and sparse (PostgreSQL FTS) retrieval
via Reciprocal Rank Fusion, expands neighboring chunks, and exposes
LangChain-compatible Document output.
"""

from __future__ import annotations

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
from app.db.session import (
    SessionFactory,
    SessionLocal,
    pool_snapshot,
    session_factory_from_session,
    session_scope,
)
from app.services.dense_retriever import DenseRetriever
from app.services.rrf import ReciprocalRankFusion, RRFResult
from app.services.sparse_retriever import SparseRetriever
from app.services.hybrid_executor import RetrievalBusyError, get_hybrid_executor
from app.services.settings_service import RuntimeSettingsSnapshot, SettingsService

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
        db: Session | None = None,
        top_k: int | None = None,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        rrf_k: int | None = None,
        neighbor_window: int = NEIGHBOR_WINDOW,
        *,
        session_factory: SessionFactory | None = None,
        settings_snapshot: RuntimeSettingsSnapshot | None = None,
    ) -> None:
        if session_factory is None:
            session_factory = (
                session_factory_from_session(db) if db is not None else SessionLocal
            )
        self._session_factory = session_factory
        self.top_k: int
        self.dense_top_k: int
        self.sparse_top_k: int
        self.rrf_k: int
        self.neighbor_window = max(neighbor_window, 0)

        if settings_snapshot is None and any(
            value is None for value in [top_k, dense_top_k, sparse_top_k, rrf_k]
        ):
            if db is not None:
                settings_snapshot = SettingsService(db).runtime_snapshot()
            else:
                with session_scope(self._session_factory) as settings_db:
                    settings_snapshot = SettingsService(settings_db).runtime_snapshot()

        if settings_snapshot is not None:
            self.top_k = top_k if top_k is not None else settings_snapshot.top_k_final
            self.dense_top_k = dense_top_k if dense_top_k is not None else settings_snapshot.top_k_dense
            self.sparse_top_k = sparse_top_k if sparse_top_k is not None else settings_snapshot.top_k_sparse
            self.rrf_k = rrf_k if rrf_k is not None else settings_snapshot.rrf_k
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
            dense_results, sparse_results, dense_ms, sparse_ms, embedding_down = self._retrieve_parallel(
                query,
                filters,
            )
            retrieval_mode = "parallel"
        else:
            dense_results, sparse_results, dense_ms, sparse_ms, embedding_down = self._retrieve_sequential(
                query,
                filters,
            )
            retrieval_mode = "sequential"

        logger.debug(
            "HybridRetriever: dense=%d sparse=%d query=%r",
            len(dense_results),
            len(sparse_results),
            query[:80],
        )

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
                **pool_snapshot(),
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
    ) -> tuple[list[Any], list[Any], float, float, bool]:
        def dense_task() -> tuple[list[Any], float, bool]:
            with session_scope(self._session_factory) as db:
                started_at = time.perf_counter()
                outcome = DenseRetriever(db).retrieve_outcome(
                    query,
                    top_k=self.dense_top_k,
                    filters=filters,
                )
                return (
                    outcome.results,
                    round((time.perf_counter() - started_at) * 1000, 1),
                    outcome.embedding_failed,
                )

        def sparse_task() -> tuple[list[Any], float]:
            with session_scope(self._session_factory) as db:
                started_at = time.perf_counter()
                results = SparseRetriever(db).retrieve(
                    query,
                    top_k=self.sparse_top_k,
                    filters=filters,
                )
                return results, round((time.perf_counter() - started_at) * 1000, 1)

        executor = get_hybrid_executor()
        dense_future = None
        sparse_future = None
        try:
            dense_future = executor.submit(dense_task)
            sparse_future = executor.submit(sparse_task)
            dense_results, dense_ms, dense_failed = dense_future.result()
            sparse_results, sparse_ms = sparse_future.result()
            return dense_results, sparse_results, dense_ms, sparse_ms, dense_failed
        except RetrievalBusyError:
            raise
        except Exception as exc:
            for future in (dense_future, sparse_future):
                if future is None:
                    continue
                try:
                    future.result()
                except Exception:
                    pass
            logger.warning(
                "HybridRetriever: parallel retrieval failed; retrying sequentially: %s",
                exc,
            )
            return self._retrieve_sequential(query, filters)

    def _retrieve_sequential(
        self,
        query: str,
        filters: dict | None,
    ) -> tuple[list[Any], list[Any], float, float, bool]:
        dense_started_at = time.perf_counter()
        with session_scope(self._session_factory) as db:
            dense_outcome = DenseRetriever(db).retrieve_outcome(
                query,
                top_k=self.dense_top_k,
                filters=filters,
            )
            dense_results = dense_outcome.results
        dense_ms = round((time.perf_counter() - dense_started_at) * 1000, 1)
        sparse_started_at = time.perf_counter()
        with session_scope(self._session_factory) as db:
            sparse_results = SparseRetriever(db).retrieve(
                query,
                top_k=self.sparse_top_k,
                filters=filters,
            )
        sparse_ms = round((time.perf_counter() - sparse_started_at) * 1000, 1)
        return (
            dense_results,
            sparse_results,
            dense_ms,
            sparse_ms,
            dense_outcome.embedding_failed,
        )

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

        with session_scope(self._session_factory) as db:
            chunks = (
                db.query(DocumentChunk)
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
