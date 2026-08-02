from __future__ import annotations

import argparse
import logging

from sqlalchemy import text

from app.db.session import engine


logger = logging.getLogger(__name__)
ONLINE_VERSION = "016_chunk_retrievability_online"

CONCURRENT_STATEMENTS = (
    """
    CREATE INDEX CONCURRENTLY IF NOT EXISTS
        idx_document_chunks_embedding_retrievable_hnsw
    ON document_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE is_retrievable IS TRUE
    """,
    """
    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_chunks_programme_expr
    ON document_chunks ((metadata->>'programme'))
    """,
    """
    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_chunks_year_expr
    ON document_chunks ((metadata->>'year'))
    """,
    "DROP INDEX CONCURRENTLY IF EXISTS idx_document_chunks_document_id",
    "DROP INDEX CONCURRENTLY IF EXISTS ix_document_chunks_document_id",
    "DROP INDEX CONCURRENTLY IF EXISTS ix_document_chunks_id",
    "DROP INDEX CONCURRENTLY IF EXISTS ix_faqs_id",
)

MANAGED_INDEXES = {
    "idx_document_chunks_embedding_retrievable_hnsw",
    "idx_document_chunks_programme_expr",
    "idx_document_chunks_year_expr",
}


def backfill_retrievability(batch_size: int = 1000) -> int:
    total = 0
    while True:
        with engine.begin() as connection:
            result = connection.execute(
                text(
                    """
                    WITH batch AS (
                        SELECT c.id
                        FROM document_chunks c
                        WHERE c.is_retrievable IS NULL
                        ORDER BY c.id
                        FOR UPDATE SKIP LOCKED
                        LIMIT :batch_size
                    )
                    UPDATE document_chunks c
                    SET is_retrievable = (
                        d.status = 'active' AND d.quality_status <> 'review'
                    )
                    FROM batch, documents d
                    WHERE c.id = batch.id
                      AND d.id = c.document_id
                    """
                ),
                {"batch_size": max(1, int(batch_size))},
            )
            updated = int(result.rowcount or 0)
        total += updated
        if updated == 0:
            return total


def build_online_indexes() -> None:
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        invalid_indexes = connection.execute(
            text(
                """
                SELECT c.relname
                FROM pg_class c
                JOIN pg_index i ON i.indexrelid = c.oid
                WHERE c.relname = ANY(:index_names)
                  AND (NOT i.indisvalid OR NOT i.indisready)
                """
            ),
            {"index_names": sorted(MANAGED_INDEXES)},
        ).scalars().all()
        for index_name in invalid_indexes:
            if index_name not in MANAGED_INDEXES:
                continue
            connection.exec_driver_sql(
                f'DROP INDEX CONCURRENTLY IF EXISTS "{index_name}"'
            )
        for statement in CONCURRENT_STATEMENTS:
            connection.exec_driver_sql(statement.strip())


def validate_online_indexes() -> None:
    with engine.connect() as connection:
        pending = connection.execute(
            text(
                "SELECT EXISTS (SELECT 1 FROM document_chunks "
                "WHERE is_retrievable IS NULL LIMIT 1)"
            )
        ).scalar_one()
        if pending:
            raise RuntimeError("document chunk retrievability backfill is incomplete")

        valid = connection.execute(
            text(
                """
                SELECT i.indisvalid AND i.indisready
                FROM pg_class c
                JOIN pg_index i ON i.indexrelid = c.oid
                WHERE c.relname = 'idx_document_chunks_embedding_retrievable_hnsw'
                """
            )
        ).scalar_one_or_none()
        if valid is not True:
            raise RuntimeError("retrievable HNSW index is missing or invalid")


def mark_complete() -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO schema_migrations (version)
                VALUES (:version)
                ON CONFLICT (version) DO NOTHING
                """
            ),
            {"version": ONLINE_VERSION},
        )


def run(batch_size: int = 1000) -> int:
    updated = backfill_retrievability(batch_size=batch_size)
    logger.info("Chunk retrievability backfill complete | updated=%s", updated)
    build_online_indexes()
    validate_online_indexes()
    mark_complete()
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill chunk eligibility and build retrieval indexes online."
    )
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()
    run(batch_size=args.batch_size)


if __name__ == "__main__":
    main()
