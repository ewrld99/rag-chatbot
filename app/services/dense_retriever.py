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
from typing import Any

from sqlalchemy import or_
from sqlalchemy.orm import Session, joinedload

from app.db.models import DocumentChunk, FAQModel
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
_dense_state = threading.local()


def dense_embedding_failed() -> bool:
    """Return True if the most recent retrieve() call failed due to embedding."""
    return bool(getattr(_dense_state, "embedding_failed", False))


def _set_embedding_failed(value: bool) -> None:
    _dense_state.embedding_failed = value


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
        When the embedding service is unavailable, returns [] and sets the
        thread-local dense_embedding_failed() flag to True so callers can
        switch to degraded (sparse-only) mode gracefully.
        """
        if not query or not query.strip():
            _set_embedding_failed(False)
            return []

        # Reset failure flag at the start of each call so callers always see
        # the result of THIS request, not a previous one on the same thread.
        _set_embedding_failed(False)

        try:
            query_embedding = get_embedding(query, self.db)

            doc_query = (
                self.db.query(
                    DocumentChunk,
                    DocumentChunk.embedding.cosine_distance(query_embedding).label("dist"),
                )
                .filter(DocumentChunk.document.has(status="active"))
                .options(joinedload(DocumentChunk.document))
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
            _set_embedding_failed(True)
            # Always log at WARNING — even a circuit-breaker cooldown is a
            # service degradation that admins need to see in the logs.
            logger.warning(
                "DenseRetriever: embedding service unavailable — dense retrieval "
                "disabled for this request. Sparse (FTS) retrieval will run alone. "
                "Reason: %s",
                exc,
            )
            return []
        except Exception as exc:
            self.db.rollback()
            _set_embedding_failed(False)
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

        return results
