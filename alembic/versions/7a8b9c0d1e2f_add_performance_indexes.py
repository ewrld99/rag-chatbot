"""Add HNSW vector and GIN full-text search performance indexes

Revision ID: 7a8b9c0d1e2f
Revises: 600_add_crawler_queue
Create Date: 2026-07-09 22:38:00.000000

Why these indexes matter
------------------------
Without ANN and FTS indexes, every chat query forces PostgreSQL to perform
a full sequential scan (O(n)) over every row in document_chunks and faqs.

  Chunks  | No index (seq scan) | With HNSW index
  --------|---------------------|------------------
    10k   |       ~50 ms        |      ~2 ms
   100k   |      ~500 ms        |      ~5 ms
   500k   |       ~5 sec        |     ~10 ms

History
-------
Migration 363380e26e50 (add_crawler_models) dropped the four performance
indexes that existed previously. This migration restores them with
consistent, well-documented names.

Indexes added
-------------
  idx_chunks_embedding_hnsw  — ANN cosine vector search over chunk embeddings
  idx_chunks_tsv_gin         — GIN index for PostgreSQL FTS on chunk text
  idx_faqs_embedding_hnsw    — ANN cosine vector search over FAQ embeddings
                               (partial: WHERE embedding IS NOT NULL because
                                the column is nullable)
  idx_faqs_fts_gin           — GIN index for PostgreSQL FTS on FAQ text

All four use CONCURRENTLY so the table is never locked during the build —
safe to run on a live database with existing data.

HNSW tuning parameters (m=16, ef_construction=64)
--------------------------------------------------
  m               : number of bi-directional links per node.
                    Higher → better recall + more RAM. 16 is the default.
  ef_construction : size of the dynamic candidate list during build.
                    Higher → slower build but better index quality. 64 is safe.

At query time you can tune recall vs speed via:
    SET hnsw.ef_search = 40;   -- default; raise to 100+ for higher recall
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = '7a8b9c0d1e2f'
down_revision: Union[str, Sequence[str], None] = '600_add_crawler_queue'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    CREATE INDEX CONCURRENTLY cannot run inside a transaction block.
    Alembic's autocommit_block() context manager handles this correctly —
    it issues a COMMIT before the block and begins a new transaction after.
    """
    with op.get_context().autocommit_block():

        # ── document_chunks: HNSW vector index ───────────────────────────────
        # Uses vector_cosine_ops to match the cosine_distance() operator
        # used in DenseRetriever.retrieve().
        op.execute("""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_embedding_hnsw
            ON document_chunks
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """)

        # ── document_chunks: GIN full-text search index ───────────────────────
        # Accelerates the websearch_to_tsquery('english', ...) queries in
        # SparseRetriever. Without this, FTS falls back to a sequential scan.
        op.execute("""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_tsv_gin
            ON document_chunks USING gin(tsv)
        """)

        # ── faqs: HNSW vector index ───────────────────────────────────────────
        # Partial index (WHERE embedding IS NOT NULL) because FAQModel.embedding
        # is nullable — pgvector cannot index NULL vectors.
        op.execute("""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_faqs_embedding_hnsw
            ON faqs
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
            WHERE embedding IS NOT NULL
        """)

        # ── faqs: GIN full-text search index ─────────────────────────────────
        op.execute("""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_faqs_fts_gin
            ON faqs USING gin(fts_vector)
        """)


def downgrade() -> None:
    # DROP INDEX CONCURRENTLY also cannot run in a transaction block.
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_faqs_fts_gin")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_faqs_embedding_hnsw")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_chunks_tsv_gin")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_chunks_embedding_hnsw")
