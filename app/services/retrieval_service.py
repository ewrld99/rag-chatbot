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

import os
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document
from sqlalchemy.orm import Session

from app.core.config import settings
from app.services.embedding_service import get_embedding
from app.services.hybrid_retriever import HybridRetriever


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
        self.top_k = top_k
        self._hybrid = HybridRetriever(
            db=db,
            top_k=top_k,
            dense_top_k=settings.DENSE_TOP_K,
            sparse_top_k=settings.SPARSE_TOP_K,
            rrf_k=settings.RRF_K,
        )

    # -----------------------------------------------------------------------
    # 1. Embed Query  (kept for backward compatibility)
    # -----------------------------------------------------------------------
    def embed_query(self, query: str) -> List[float]:
        """Convert user query into embedding vector."""
        return get_embedding(query, self.db)

    # -----------------------------------------------------------------------
    # 2. Retrieve Documents
    # -----------------------------------------------------------------------
    def retrieve(self, query: str) -> List[Document]:
        """
        Full hybrid retrieval pipeline → LangChain Documents.

        Calls HybridRetriever which:
          1. Dense  (pgvector cosine)
          2. Sparse (PostgreSQL FTS)
          3. RRF fusion
        """
        return self._hybrid.get_relevant_documents(query)

    # -----------------------------------------------------------------------
    # 3. Retrieve with Scores (returns cosine-like rrf_score for compatibility)
    # -----------------------------------------------------------------------
    def retrieve_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """
        Returns (Document, rrf_score) pairs.
        rrf_score replaces the old cosine distance score.
        """
        raw = self._hybrid.retrieve_raw(query)
        output: List[Tuple[Document, float]] = []

        for result in raw:
            doc = Document(
                page_content=result.text,
                metadata={
                    **result.metadata,
                    "chunk_id": result.chunk_id,
                    "rrf_score": result.rrf_score,
                    "dense_rank": result.dense_rank,
                    "sparse_rank": result.sparse_rank,
                    "retrieval_sources": result.retrieval_sources,
                },
            )
            output.append((doc, result.rrf_score))

        return output

    # -----------------------------------------------------------------------
    # 4. Format Context
    # -----------------------------------------------------------------------
    def format_context(self, documents: List[Document]) -> str:
        """Convert retrieved documents into structured LLM context."""
        if not documents:
            return "No relevant context found."

        formatted_chunks: List[str] = []
        for i, doc in enumerate(documents):
            source = doc.metadata.get("source_url") or doc.metadata.get("source") or "unknown"
            
            # If the source is just a local PDF filename, convert it to a full URL
            if source.endswith(".pdf") and not source.startswith("http"):
                # Clean up path if it includes 'uploads/' or backslashes
                filename = source.split("/")[-1].split("\\")[-1]
                local_path = os.path.join("uploads", filename)
                
                # Double-check that the file actually exists on disk! If someone deleted it
                # manually, we shouldn't feed it to the LLM to avoid broken links.
                if not os.path.exists(local_path):
                    pass # Keep source as filename to avoid broken links
                else:
                    source = f"http://localhost:8000/uploads/{filename}"

            chunk_index = doc.metadata.get("chunk_index", i)
            chunk_text = (
                f'  <document id="[Doc {i+1}]" source="{source}" chunk="{chunk_index}">\n'
                f'    {doc.page_content.strip()}\n'
                f'  </document>'
            )
            formatted_chunks.append(chunk_text)

        return "\n".join(formatted_chunks)

    # -----------------------------------------------------------------------
    # 5. Full Pipeline
    # -----------------------------------------------------------------------
    def get_context(self, query: str) -> Dict[str, Any]:
        """End-to-end retrieval pipeline — unchanged interface."""
        docs = self.retrieve(query)
        context = self.format_context(docs)
        return {
            "query": query,
            "documents": docs,
            "context": context,
        }
