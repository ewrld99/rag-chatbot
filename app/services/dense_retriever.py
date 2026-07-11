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
from app.services.embedding_service import get_embedding

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
        except Exception as exc:
            logger.warning("DenseRetriever query failed: %s", exc)
            return []

        # Merge and sort
        combined = []
        for chunk, dist in rows:
            combined.append((dist, chunk, "doc"))
            
        for faq, dist in faq_rows:
            combined.append((dist, faq, "faq"))
            
        combined.sort(key=lambda x: x[0])
        # Do NOT slice to top_k here — return the full merged pool so RRF
        # has richer candidates from both document chunks and FAQs.

        results: list[DenseResult] = []
        for dist, item, type_ in combined:
            if type_ == "doc":
                chunk = item
                chunk_id = str(chunk.id)
                doc_filename = chunk.document.filename if chunk.document and chunk.document.filename else "database"
                doc_id = str(chunk.document_id)
                
                results.append(
                    DenseResult(
                        chunk_id=chunk_id,
                        document_id=doc_id,
                        similarity_score=round(max(0.0, 1.0 - float(dist)), 6),
                        text=chunk.chunk_text,
                        metadata={
                            "source": doc_filename,
                            "chunk_index": chunk.chunk_index,
                            "source_type": "document"
                        },
                    )
                )
            else:
                faq = item
                results.append(
                    DenseResult(
                        chunk_id=str(faq.id),
                        document_id=str(faq.id),
                        similarity_score=round(max(0.0, 1.0 - float(dist)), 6),
                        text=f"{faq.question}\n\n{faq.answer}",
                        metadata={
                            "source": f"FAQ - {faq.category}" if faq.category else "FAQ",
                            "chunk_index": 0,
                            "source_type": "faq",
                            "faq_id": str(faq.id),
                            "category": faq.category
                        },
                    )
                )

        return results
