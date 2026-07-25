CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE document_chunks
ADD COLUMN IF NOT EXISTS tsv tsvector;

ALTER TABLE faqs
ADD COLUMN IF NOT EXISTS fts_vector tsvector;

CREATE OR REPLACE FUNCTION document_chunks_fts_update()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    NEW.tsv := to_tsvector('simple', COALESCE(NEW.chunk_text, ''))
            || to_tsvector('english', COALESCE(NEW.chunk_text, ''));
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_document_chunks_fts ON document_chunks;
CREATE TRIGGER trg_document_chunks_fts
    BEFORE INSERT OR UPDATE OF chunk_text
    ON document_chunks
    FOR EACH ROW
    EXECUTE FUNCTION document_chunks_fts_update();

CREATE OR REPLACE FUNCTION faqs_fts_update()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    NEW.fts_vector := to_tsvector(
        'simple',
        COALESCE(NEW.question, '') || ' ' || COALESCE(NEW.answer, '')
    ) || to_tsvector(
        'english',
        COALESCE(NEW.question, '') || ' ' || COALESCE(NEW.answer, '')
    );
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_faqs_fts ON faqs;
CREATE TRIGGER trg_faqs_fts
    BEFORE INSERT OR UPDATE OF question, answer
    ON faqs
    FOR EACH ROW
    EXECUTE FUNCTION faqs_fts_update();

UPDATE document_chunks
SET tsv = to_tsvector('simple', COALESCE(chunk_text, ''))
       || to_tsvector('english', COALESCE(chunk_text, ''))
WHERE tsv IS NULL;

UPDATE faqs
SET fts_vector = to_tsvector(
        'simple',
        COALESCE(question, '') || ' ' || COALESCE(answer, '')
    ) || to_tsvector(
        'english',
        COALESCE(question, '') || ' ' || COALESCE(answer, '')
    )
WHERE fts_vector IS NULL;

CREATE INDEX IF NOT EXISTS idx_document_chunks_fts
ON document_chunks USING GIN (tsv);

CREATE INDEX IF NOT EXISTS idx_faqs_fts
ON faqs USING GIN (fts_vector);

CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id
ON document_chunks (document_id);

CREATE INDEX IF NOT EXISTS idx_document_chunks_metadata_gin
ON document_chunks USING GIN (metadata);

CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding_hnsw
ON document_chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_faqs_embedding
ON faqs USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
