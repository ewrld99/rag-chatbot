ALTER TABLE document_chunks
ADD COLUMN IF NOT EXISTS is_retrievable BOOLEAN NULL;

CREATE OR REPLACE FUNCTION set_document_chunk_retrievability()
RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    eligible BOOLEAN;
BEGIN
    SELECT d.status = 'active' AND d.quality_status <> 'review'
      INTO eligible
      FROM documents d
     WHERE d.id = NEW.document_id;
    NEW.is_retrievable := COALESCE(eligible, FALSE);
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_document_chunk_retrievability ON document_chunks;
CREATE TRIGGER trg_document_chunk_retrievability
BEFORE INSERT OR UPDATE OF document_id
ON document_chunks
FOR EACH ROW
EXECUTE FUNCTION set_document_chunk_retrievability();

CREATE OR REPLACE FUNCTION sync_document_retrievability()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    UPDATE document_chunks
       SET is_retrievable = (NEW.status = 'active' AND NEW.quality_status <> 'review')
     WHERE document_id = NEW.id
       AND is_retrievable IS DISTINCT FROM
           (NEW.status = 'active' AND NEW.quality_status <> 'review');
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_document_retrievability ON documents;
CREATE TRIGGER trg_document_retrievability
AFTER UPDATE OF status, quality_status
ON documents
FOR EACH ROW
WHEN (
    OLD.status IS DISTINCT FROM NEW.status
    OR OLD.quality_status IS DISTINCT FROM NEW.quality_status
)
EXECUTE FUNCTION sync_document_retrievability();

CREATE INDEX IF NOT EXISTS idx_document_chunks_retrievability_pending
ON document_chunks (id)
WHERE is_retrievable IS NULL;
