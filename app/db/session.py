from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    pool_size=getattr(settings, "DATABASE_POOL_SIZE", 20),
    max_overflow=getattr(settings, "DATABASE_MAX_OVERFLOW", 20),
    pool_pre_ping=True,
    pool_recycle=1800,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def run_sql_migrations(conn) -> None:
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ DEFAULT now() NOT NULL
            )
            """
        )
    )

    applied_versions = set(
        conn.execute(text("SELECT version FROM schema_migrations")).scalars().all()
    )

    if not MIGRATIONS_DIR.exists():
        return

    for migration_path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        version = migration_path.stem
        if version in applied_versions:
            continue

        migration_sql = migration_path.read_text(encoding="utf-8").strip()
        if migration_sql:
            conn.exec_driver_sql(migration_sql)

        conn.execute(
            text("INSERT INTO schema_migrations (version) VALUES (:version)"),
            {"version": version},
        )


def init_db():
    from app.db.models import Base

    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    Base.metadata.create_all(bind=engine)

    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS registration_number TEXT;"))
        conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS programme TEXT;"))
        conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS campus TEXT;"))
        conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS admission_year INTEGER;"))
        conn.execute(text(
            "ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS "
            "turn_context JSONB NOT NULL DEFAULT '{}'::jsonb;"
        ))

    with engine.begin() as conn:
        run_sql_migrations(conn)

    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))
        
        # ── FTS trigger for document_chunks ──────────────────────────────
        conn.execute(text(
            """
            CREATE OR REPLACE FUNCTION document_chunks_fts_update()
            RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN
                -- Combined tsvector: 'simple' preserves Swahili words and exact
                -- English forms; 'english' adds stemmed English lexemes so that
                -- e.g. "register" matches "registration" in queries.
                NEW.tsv := to_tsvector('simple', NEW.chunk_text)
                        || to_tsvector('english', NEW.chunk_text);
                RETURN NEW;
            END;
            $$
            """
        ))

        conn.execute(text(
            """
            DROP TRIGGER IF EXISTS trg_document_chunks_fts ON document_chunks;
            CREATE TRIGGER trg_document_chunks_fts
                BEFORE INSERT OR UPDATE OF chunk_text
                ON document_chunks
                FOR EACH ROW
                EXECUTE FUNCTION document_chunks_fts_update();
            """
        ))
        
        # ensure index exists
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_document_chunks_fts ON document_chunks USING GIN (tsv)"
        ))

        # ── FTS trigger for faqs ──────────────────────────────
        conn.execute(text(
            """
            CREATE OR REPLACE FUNCTION faqs_fts_update()
            RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN
                -- Same bilingual strategy: simple (exact) + english (stemmed)
                NEW.fts_vector := to_tsvector('simple', NEW.question || ' ' || NEW.answer)
                               || to_tsvector('english', NEW.question || ' ' || NEW.answer);
                RETURN NEW;
            END;
            $$
            """
        ))

        conn.execute(text(
            """
            DROP TRIGGER IF EXISTS trg_faqs_fts ON faqs;
            CREATE TRIGGER trg_faqs_fts
                BEFORE INSERT OR UPDATE OF question, answer
                ON faqs
                FOR EACH ROW
                EXECUTE FUNCTION faqs_fts_update();
            """
        ))

        # ── Ensure indexes exist for faqs ─────────────────────────────────────
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_faqs_fts ON faqs USING GIN (fts_vector)"
        ))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_faqs_embedding ON faqs USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
        ))

        # ── Backfill NULL tsvector columns ────────────────────────────────────
        # Rows ingested before the FTS trigger existed have tsv/fts_vector=NULL.
        # A NULL tsvector cannot be indexed by the GIN index, forcing a full
        # sequential scan on every FTS query (~15 second latency per query).
        # This UPDATE is a no-op on subsequent restarts (WHERE tsv IS NULL).
        conn.execute(text(
            """
            UPDATE document_chunks
            SET tsv = to_tsvector('simple', chunk_text)
                   || to_tsvector('english', chunk_text)
            WHERE tsv IS NULL AND chunk_text IS NOT NULL
            """
        ))
        conn.execute(text(
            """
            UPDATE faqs
            SET fts_vector = to_tsvector('simple', question || ' ' || answer)
                          || to_tsvector('english', question || ' ' || answer)
            WHERE fts_vector IS NULL
              AND question IS NOT NULL
              AND answer IS NOT NULL
            """
        ))
