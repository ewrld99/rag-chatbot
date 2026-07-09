"""
app/api/routes/query.py
------------------------
Two standalone endpoints for the Hybrid RAG system:

  POST /api/query   — run hybrid retrieval + LLM generation
  GET  /api/health  — database and index health check
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.schemas.query import (
    HealthResponse,
    IndexStatus,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
    SourceItem,
)
from app.services.generation_service import GenerationService
from app.services.hybrid_retriever import HybridRetriever
from app.services.retrieval_service import RetrievalService

router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# POST /api/query
# ---------------------------------------------------------------------------
@router.post("/query", response_model=QueryResponse)
def query(
    payload: QueryRequest,
    db: Session = Depends(get_db),
):
    """
    Full Hybrid RAG pipeline:

    1. Dense retrieval  (pgvector cosine, top-DENSE_TOP_K)
    2. Sparse retrieval (PostgreSQL FTS, top-SPARSE_TOP_K)
    3. Reciprocal Rank Fusion
    4. LLM generation with the top-K fused chunks as context
    5. Returns answer, sources, retrieved_chunks, and wall-clock latency
    """
    t_start = time.perf_counter()

    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # ── 1. Hybrid retrieval (raw RRFResult objects) ────────────────────────
    hybrid = HybridRetriever(
        db=db,
        top_k=payload.top_k,
        dense_top_k=settings.DENSE_TOP_K,
        sparse_top_k=settings.SPARSE_TOP_K,
        rrf_k=settings.RRF_K,
    )

    try:
        raw_results = hybrid.retrieve_raw(payload.question)
    except Exception as exc:
        logger.exception("Hybrid retrieval failed")
        raise HTTPException(status_code=500, detail=f"Retrieval error: {exc}")

    # ── 2. Build context for LLM ────────────────────────────────────────────
    retrieval_service = RetrievalService(db=db, top_k=payload.top_k)
    docs = hybrid.get_relevant_documents(payload.question)
    context = retrieval_service.format_context(docs)

    # ── 3. LLM generation ───────────────────────────────────────────────────
    try:
        generator = GenerationService()
        if not context or context.strip() == "No relevant context found.":
            answer = generator.DOCUMENT_REFUSAL
        else:
            answer = generator.generate(payload.question, context)
    except Exception as exc:
        logger.exception("LLM generation failed")
        raise HTTPException(status_code=502, detail=f"Generation error: {exc}")

    t_end = time.perf_counter()
    latency_ms = round((t_end - t_start) * 1000, 1)

    # ── 4. Build response ────────────────────────────────────────────────────
    retrieved_chunks = [
        RetrievedChunk(
            chunk_id=r.chunk_id,
            document_id=r.document_id,
            rrf_score=round(r.rrf_score, 6),
            dense_rank=r.dense_rank,
            sparse_rank=r.sparse_rank,
            retrieval_sources=r.retrieval_sources,
            text=r.text,
            metadata=r.metadata,
        )
        for r in raw_results
    ]

    sources = [
        SourceItem(content=doc.page_content, metadata=doc.metadata)
        for doc in docs
    ]

    return QueryResponse(
        answer=answer,
        sources=sources,
        retrieved_chunks=retrieved_chunks,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# GET /api/health
# ---------------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse)
def health(db: Session = Depends(get_db)):
    """
    System health check:

    - database:      can we execute a simple query?
    - vector_index:  does the pgvector HNSW/IVFFlat index exist on documents.embedding?
    - fts_index:     does the GIN index on documents.fts_vector exist?
    """
    db_ok = False
    vector_index = False
    fts_index = False
    detail: str | None = None

    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception as exc:
        detail = f"Database unreachable: {exc}"
        return HealthResponse(
            status="error",
            database=False,
            indexes=IndexStatus(vector_index=False, fts_index=False),
            detail=detail,
        )

    # Check vector index
    try:
        row = db.execute(
            text(
                """
                SELECT COUNT(*) FROM pg_indexes
                WHERE tablename = 'documents'
                  AND indexdef ILIKE '%vector%'
                """
            )
        ).scalar()
        vector_index = bool(row and row > 0)
    except Exception as exc:
        logger.warning("Could not check vector index: %s", exc)

    # Check FTS GIN index
    try:
        row = db.execute(
            text(
                """
                SELECT COUNT(*) FROM pg_indexes
                WHERE tablename = 'documents'
                  AND indexname = 'idx_documents_fts'
                """
            )
        ).scalar()
        fts_index = bool(row and row > 0)
    except Exception as exc:
        logger.warning("Could not check FTS index: %s", exc)

    all_ok = db_ok and vector_index and fts_index
    status = "ok" if all_ok else "degraded"

    if not vector_index:
        detail = (detail or "") + " Vector index missing."
    if not fts_index:
        detail = (detail or "") + " FTS index missing — run init_db()."

    return HealthResponse(
        status=status,
        database=db_ok,
        indexes=IndexStatus(vector_index=vector_index, fts_index=fts_index),
        detail=detail.strip() if detail else None,
    )
