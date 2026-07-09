from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import settings

engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


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
        
        # ensure indexes exist for faqs
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_faqs_fts ON faqs USING GIN (fts_vector)"
        ))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_faqs_embedding ON faqs USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
        ))
