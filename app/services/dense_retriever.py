"""
app/services/dense_retriever.py
--------------------------------
Dense (vector) retrieval using pgvector cosine similarity.

Returns the top-N chunks ranked by embedding cosine distance.
Each result is a plain dict so it is easy to pass across service boundaries
without importing LangChain types.
"""

from __future__ import annotations

from typing import Any
import logging
from sqlalchemy import or_
from sqlalchemy.orm import Session, joinedload

from app.db.models import DocumentChunk, FAQModel
from app.services.embedding_service import EmbeddingServiceError, get_embedding

logger = logging.getLogger(__name__)

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
        """
        if not query or not query.strip():
            return []

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
            log = logger.debug if exc.provider_cooldown else logger.warning
            log("DenseRetriever embedding unavailable; using sparse retrieval only: %s", exc)
            return []
        except Exception as exc:
            logger.warning("DenseRetriever query failed; using sparse retrieval only: %s", exc)
            return []

        # Merge and sort
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
                metadata = dict(chunk.metadata_ or {})
                metadata.update(
                    {
                        "source": doc_filename,
                        "chunk_index": chunk.chunk_index,
                        "source_type": "document",
                        "similarity_score": similarity_score,
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
                metadata = dict(faq.metadata_ or {})
                metadata.update(
                    {
                        "source": f"FAQ - {faq.category}" if faq.category else "FAQ",
                        "chunk_index": 0,
                        "source_type": "faq",
                        "faq_id": str(faq.id),
                        "category": faq.category,
                        "similarity_score": similarity_score,
                    }
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
