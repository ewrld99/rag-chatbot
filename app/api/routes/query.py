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
from app.services.generation_resilience import GenerationUnavailableError
from app.services.model_router import ModelRouter
from app.services.retrieval_service import RetrievalService
from app.services.settings_service import SettingsService
from app.services.source_service import format_source_records

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

    # 1. Shared retrieval path: hybrid search, neighbor expansion, and reranking.
    retrieval_service = RetrievalService(db=db, top_k=payload.top_k)
    try:
        scored_docs = retrieval_service.retrieve_with_scores(payload.question)
    except Exception as exc:
        logger.exception("Hybrid retrieval failed")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "RETRIEVAL_FAILED",
                "message": "Unable to search the documents right now. Please try again.",
            },
        ) from exc

    docs = [doc for doc, _score in scored_docs]
    context = retrieval_service.format_context(docs)
    safe_sources = format_source_records(docs)

    # ── 3. LLM generation ───────────────────────────────────────────────────
    try:
        generator = GenerationService(
            model_router=ModelRouter(SettingsService(db))
        )
        generator.set_model_preference(payload.model_preference)
        if not context or context.strip() == "No relevant context found.":
            answer = generator.DOCUMENT_REFUSAL
            evidence_ids: list[str] = []
        else:
            generation_result = generator.generate_response(
                payload.question,
                context,
                documents=docs,
            )
            answer = generation_result["answer"]
            evidence_ids = generation_result["evidence_ids"]
    except GenerationUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail=exc.public_payload(sources=safe_sources),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected LLM generation failure")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "GENERATION_FAILED",
                "message": "Unable to generate an answer right now. Please try again.",
                "sources": safe_sources,
            },
        ) from exc

    t_end = time.perf_counter()
    latency_ms = round((t_end - t_start) * 1000, 1)

    # ── 4. Build response ────────────────────────────────────────────────────
    retrieved_chunks = [
        RetrievedChunk(
            chunk_id=str(doc.metadata.get("chunk_id", "")),
            document_id=str(doc.metadata.get("document_id", "")),
            rrf_score=round(float(doc.metadata.get("rrf_score", score)), 6),
            dense_rank=doc.metadata.get("dense_rank"),
            sparse_rank=doc.metadata.get("sparse_rank"),
            retrieval_sources=doc.metadata.get("retrieval_sources", []),
            text=doc.page_content,
            metadata=doc.metadata,
        )
        for doc, score in scored_docs
    ]

    grounded_docs = generator.grounding.filter_documents(docs, evidence_ids)
    sources = [
        SourceItem(content=doc.page_content, metadata=doc.metadata)
        for doc in grounded_docs
    ]

    return QueryResponse(
        answer=answer,
        sources=sources,
        retrieved_chunks=retrieved_chunks,
        latency_ms=latency_ms,
        **generator.model_metadata(),
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
                WHERE tablename IN ('document_chunks', 'faqs')
                  AND indexdef ILIKE '%vector_cosine_ops%'
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
                WHERE tablename IN ('document_chunks', 'faqs')
                  AND indexname IN ('idx_document_chunks_fts', 'idx_faqs_fts')
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
        detail = (detail or "") + " FTS index missing - run migrations/init_db()."

    return HealthResponse(
        status=status,
        database=db_ok,
        indexes=IndexStatus(vector_index=vector_index, fts_index=fts_index),
        detail=detail.strip() if detail else None,
    )
