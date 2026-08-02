ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS storage_state TEXT,
    ADD COLUMN IF NOT EXISTS storage_error TEXT;

UPDATE documents
SET storage_state = CASE
    WHEN file_path IS NULL OR btrim(file_path) = '' THEN 'not_applicable'
    ELSE 'unverified'
END
WHERE storage_state IS NULL;

ALTER TABLE documents
    ALTER COLUMN storage_state SET DEFAULT 'not_applicable',
    ALTER COLUMN storage_state SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'ck_documents_storage_state'
    ) THEN
        ALTER TABLE documents ADD CONSTRAINT ck_documents_storage_state CHECK (
            storage_state IN (
                'not_applicable', 'unverified', 'staged', 'promoting',
                'ready', 'missing', 'delete_pending'
            )
        );
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS file_operations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NULL REFERENCES documents(id) ON DELETE SET NULL,
    operation TEXT NOT NULL CHECK (operation IN ('promote', 'delete')),
    source_path TEXT NOT NULL,
    target_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('held', 'pending', 'processing', 'completed', 'failed', 'cancelled')),
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_file_operations_work
    ON file_operations (status, created_at)
    WHERE status IN ('pending', 'failed');

CREATE INDEX IF NOT EXISTS ix_file_operations_document_id
    ON file_operations (document_id);
