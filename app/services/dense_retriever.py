"""
app/services/dense_retriever.py
--------------------------------
Dense (vector) retrieval using pgvector cosine similarity.

Returns the top-N chunks ranked by embedding cosine distance.
Each result is a plain dict so it is easy to pass across service boundaries
without importing LangChain types.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy import or_, text
from sqlalchemy.orm import Session, joinedload

from app.db.models import DocumentChunk, DocumentModel, FAQModel
from app.services.embedding_service import EmbeddingServiceError, get_embedding
from app.services.retrieval_metadata import document_metadata, faq_metadata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thread-local failure flag
#
# Set to True by retrieve() when the embedding service is unavailable.
# HybridRetriever reads this flag to distinguish:
#   - dense_results=[]  because embedding is DOWN  → degraded mode, be lenient
#   - dense_results=[]  because no docs matched    → normal empty result
# ---------------------------------------------------------------------------
_index_capability_lock = threading.Lock()
_index_capability_value = False
_index_capability_checked_at = 0.0
_INDEX_CAPABILITY_TTL_SECONDS = 30.0


def retrievable_hnsw_ready(db: Session, *, force: bool = False) -> bool:
    """Return whether the online backfill and partial HNSW index are ready."""
    global _index_capability_checked_at, _index_capability_value
    now = time.monotonic()
    with _index_capability_lock:
        if not force and now - _index_capability_checked_at < _INDEX_CAPABILITY_TTL_SECONDS:
            return _index_capability_value
        try:
            ready = db.execute(
                text(
                    """
                    SELECT
                        EXISTS (
                            SELECT 1
                            FROM pg_class c
                            JOIN pg_index i ON i.indexrelid = c.oid
                            JOIN pg_am am ON am.oid = c.relam
                            WHERE c.relname = 'idx_document_chunks_embedding_retrievable_hnsw'
                              AND am.amname = 'hnsw'
                              AND i.indisvalid
                              AND i.indisready
                        )
                        AND NOT EXISTS (
                            SELECT 1 FROM document_chunks
                            WHERE is_retrievable IS NULL
                            LIMIT 1
                        )
                    """
                )
            ).scalar_one()
            _index_capability_value = bool(ready)
        except Exception:
            db.rollback()
            _index_capability_value = False
        _index_capability_checked_at = now
        return _index_capability_value


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
class DenseResult:
    """Lightweight result holder for a single dense-retrieval hit."""

    __slots__ = (
        "chunk_id",
        "document_id",
        "similarity_score",
        "text",
        "metadata",
    )

    def __init__(
        self,
        chunk_id: str,
        document_id: str,
        similarity_score: float,
        text: str,
        metadata: dict[str, Any],
    ) -> None:
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.similarity_score = similarity_score
        self.text = text
        self.metadata = metadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "similarity_score": self.similarity_score,
            "text": self.text,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class DenseRetrievalOutcome:
    results: list[DenseResult]
    embedding_failed: bool = False


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
class DenseRetriever:
    """
    Performs approximate-nearest-neighbour search using pgvector.

    The similarity score is **cosine similarity** (1 − cosine_distance),
    so higher = more relevant.
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    def retrieve(self, query: str, top_k: int = 20, filters: dict | None = None) -> list[DenseResult]:
        """Compatibility API returning only result rows."""
        return self.retrieve_outcome(query, top_k=top_k, filters=filters).results

    def retrieve_outcome(
        self,
        query: str,
        top_k: int = 20,
        filters: dict | None = None,
    ) -> DenseRetrievalOutcome:
        """
        Embed *query* and return the *top_k* most similar chunks.

        Parameters
        ----------
        query:   raw user query string
        top_k:   maximum number of results to return
        filters: optional dict with keys 'programme' and/or 'year' to narrow
                 results to chunks whose metadata matches.  Chunks without the
                 metadata key are always included (graceful fallback for older
                 ingested documents).

        Returns
        -------
        List of DenseResult ordered best → worst (descending similarity).
        When embeddings are unavailable, the outcome explicitly marks dense
        retrieval as degraded so worker state cannot leak between requests.
        """
        if not query or not query.strip():
            return DenseRetrievalOutcome([])

        try:
            query_embedding = get_embedding(query, self.db)

            use_retrievable_index = retrievable_hnsw_ready(self.db)
            if use_retrievable_index:
                self.db.execute(text("SET LOCAL hnsw.iterative_scan = 'strict_order'"))

            doc_query = (
                self.db.query(
                    DocumentChunk,
                    DocumentChunk.embedding.cosine_distance(query_embedding).label("dist"),
                )
                .options(joinedload(DocumentChunk.document))
            )
            if use_retrievable_index:
                doc_query = doc_query.filter(DocumentChunk.is_retrievable.is_(True))
            else:
                doc_query = (
                    doc_query.join(DocumentChunk.document)
                    .filter(DocumentModel.status == "active")
                    .filter(or_(DocumentModel.quality_status != "review", DocumentModel.quality_status.is_(None)))
                )

            # Apply optional metadata filters with graceful fallback:
            # chunks missing the metadata key (NULL) are always included so
            # old documents ingested without metadata are not silently dropped.
            if filters:
                programme = filters.get("programme")
                year = filters.get("year")
                if programme:
                    doc_query = doc_query.filter(
                        or_(
                            DocumentChunk.metadata_["programme"].astext == str(programme),
                            DocumentChunk.metadata_["programme"].astext.is_(None),
                        )
                    )
                if year:
                    doc_query = doc_query.filter(
                        or_(
                            DocumentChunk.metadata_["year"].astext == str(year),
                            DocumentChunk.metadata_["year"].astext.is_(None),
                        )
                    )

            rows = doc_query.order_by("dist").limit(top_k).all()

            faq_rows = (
                self.db.query(
                    FAQModel,
                    FAQModel.embedding.cosine_distance(query_embedding).label("dist"),
                )
                .filter(FAQModel.is_active == True)
                .order_by("dist")
                .limit(top_k)
                .all()
            )

        except EmbeddingServiceError as exc:
            self.db.rollback()
            # Always log at WARNING — even a circuit-breaker cooldown is a
            # service degradation that admins need to see in the logs.
            logger.warning(
                "DenseRetriever: embedding service unavailable — dense retrieval "
                "disabled for this request. Sparse (FTS) retrieval will run alone. "
                "Reason: %s",
                exc,
            )
            return DenseRetrievalOutcome([], embedding_failed=True)
        except Exception as exc:
            self.db.rollback()
            logger.exception(
                "DenseRetriever: unexpected error — dense retrieval disabled for "
                "this request. Sparse (FTS) retrieval will run alone. Reason: %s",
                exc,
            )
            raise

        # Merge doc chunks + FAQ rows and sort by distance (ascending = most similar first)
        combined = []
        for chunk, dist in rows:
            combined.append((dist, chunk, "doc"))
        for faq, dist in faq_rows:
            combined.append((dist, faq, "faq"))

        combined.sort(key=lambda item: (float(item[0]), str(item[1].id)))
        combined = combined[:top_k]

        results: list[DenseResult] = []
        for dist, item, type_ in combined:
            similarity_score = round(max(0.0, 1.0 - float(dist)), 6)
            if type_ == "doc":
                chunk = item
                chunk_id = str(chunk.id)
                doc_filename = chunk.document.filename if chunk.document and chunk.document.filename else "database"
                doc_id = str(chunk.document_id)
                metadata = document_metadata(
                    chunk.metadata_,
                    source=doc_filename,
                    chunk_index=chunk.chunk_index,
                    page_number=chunk.page_number,
                    document_title=chunk.document.title if chunk.document else None,
                    source_url=chunk.document.source_url if chunk.document else None,
                    document_status=chunk.document.status if chunk.document else None,
                    content_hash=chunk.document.content_hash if chunk.document else None,
                    document_uploaded_at=chunk.document.upload_date if chunk.document else None,
                    extra={"similarity_score": similarity_score},
                )

                results.append(
                    DenseResult(
                        chunk_id=chunk_id,
                        document_id=doc_id,
                        similarity_score=similarity_score,
                        text=chunk.chunk_text,
                        metadata=metadata,
                    )
                )
            else:
                faq = item
                metadata = faq_metadata(
                    faq.metadata_,
                    faq_id=str(faq.id),
                    category=faq.category,
                    extra={"similarity_score": similarity_score},
                )
                results.append(
                    DenseResult(
                        chunk_id=str(faq.id),
                        document_id=str(faq.id),
                        similarity_score=similarity_score,
                        text=f"{faq.question}\n\n{faq.answer}",
                        metadata=metadata,
                    )
                )

        return DenseRetrievalOutcome(results)
