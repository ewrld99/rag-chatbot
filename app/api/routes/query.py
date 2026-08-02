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

from app.api.deps import get_rag_pipeline, require_admin_user
from app.api.limiter import RateLimiter
from app.core.config import settings
from app.db.models import User
from app.db.session import get_db
from app.schemas.query import (
    HealthResponse,
    IndexStatus,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
    SourceItem,
)
from app.services.generation_resilience import GenerationUnavailableError
from app.services.rag_pipeline import RAGPipeline

router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# POST /api/query
# ---------------------------------------------------------------------------
@router.post("/query", response_model=QueryResponse)
def query(
    payload: QueryRequest,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
    _: User = Depends(require_admin_user),
    __: None = Depends(RateLimiter(limit=20, window=60, scope="admin_query")),
):
    """Run the same RAG pipeline as chat and include admin diagnostics."""
    t_start = time.perf_counter()

    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # ── 3. LLM generation ───────────────────────────────────────────────────
    try:
        rag_pipeline.retrieval_service.set_top_k(payload.top_k)
        result = rag_pipeline.run(
            payload.question,
            model_preference=payload.model_preference,
        )
    except GenerationUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail=exc.public_payload(),
        ) from exc
    except Exception as exc:
        logger.exception("Admin query pipeline failed")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "QUERY_PROCESSING_FAILED",
                "message": "Unable to process the query right now. Please try again.",
            },
        ) from exc

    t_end = time.perf_counter()
    latency_ms = round((t_end - t_start) * 1000, 1)

    # ── 4. Build response ────────────────────────────────────────────────────
    scored_docs = rag_pipeline.last_retrieval()
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

    documents = [doc for doc, _score in scored_docs]
    evidence_ids = result.get("grounding", {}).get("evidence_ids", [])
    grounded_docs = rag_pipeline.generator.grounding.filter_documents(
        documents,
        evidence_ids,
    )
    if not grounded_docs and result.get("sources"):
        source_ids = {
            str(source.get("document_id"))
            for source in result["sources"]
            if source.get("document_id") is not None
        }
        grounded_docs = [
            doc
            for doc in documents
            if str(doc.metadata.get("document_id")) in source_ids
        ]
    sources = [
        SourceItem(content=doc.page_content, metadata=doc.metadata)
        for doc in grounded_docs
    ]

    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        retrieved_chunks=retrieved_chunks,
        latency_ms=latency_ms,
        requested_model=result.get("requested_model", payload.model_preference),
        selected_model=result.get("selected_model"),
        fallback_used=result.get("fallback_used", False),
    )


# ---------------------------------------------------------------------------
# GET /api/health
# ---------------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse)
def health(db: Session = Depends(get_db)):
    """
    System health check:

    - database:      can we execute a simple query?
    - vector_index:  are both cosine HNSW indexes valid and dimension-compatible?
    - fts_index:     are both expected GIN indexes valid and ready?
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
                WITH expected(index_name, table_name, partial_required) AS (
                    VALUES
                        ('idx_document_chunks_embedding_retrievable_hnsw', 'document_chunks', true),
                        ('idx_faqs_embedding', 'faqs', false)
                ), valid_indexes AS (
                    SELECT e.index_name
                    FROM expected e
                    JOIN pg_class idx ON idx.relname = e.index_name
                    JOIN pg_index i ON i.indexrelid = idx.oid
                    JOIN pg_class tbl ON tbl.oid = i.indrelid AND tbl.relname = e.table_name
                    JOIN pg_am am ON am.oid = idx.relam AND am.amname = 'hnsw'
                    WHERE i.indisvalid
                      AND i.indisready
                      AND pg_get_indexdef(idx.oid) ILIKE '%vector_cosine_ops%'
                      AND (
                          NOT e.partial_required
                          OR pg_get_expr(i.indpred, i.indrelid) ILIKE '%is_retrievable IS TRUE%'
                      )
                ), dimensions AS (
                    SELECT COUNT(*) AS compatible
                    FROM pg_attribute a
                    JOIN pg_class t ON t.oid = a.attrelid
                    WHERE t.relname IN ('document_chunks', 'faqs')
                      AND a.attname = 'embedding'
                      AND format_type(a.atttypid, a.atttypmod) = :vector_type
                )
                SELECT
                    (SELECT COUNT(*) FROM valid_indexes) = 2
                    AND (SELECT compatible FROM dimensions) = 2
                    AND NOT EXISTS (
                        SELECT 1 FROM document_chunks
                        WHERE is_retrievable IS NULL
                        LIMIT 1
                    )
                """
            ),
            {"vector_type": f"vector({settings.EMBEDDING_DIMENSION})"},
        ).scalar()
        vector_index = bool(row)
    except Exception as exc:
        logger.warning("Could not check vector index: %s", exc)

    # Check FTS GIN index
    try:
        row = db.execute(
            text(
                """
                SELECT COUNT(*) = 2
                FROM pg_class idx
                JOIN pg_index i ON i.indexrelid = idx.oid
                JOIN pg_am am ON am.oid = idx.relam
                WHERE idx.relname IN ('idx_document_chunks_fts', 'idx_faqs_fts')
                  AND am.amname = 'gin'
                  AND i.indisvalid
                  AND i.indisready
                """
            )
        ).scalar()
        fts_index = bool(row)
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
