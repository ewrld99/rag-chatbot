CREATE INDEX IF NOT EXISTS idx_document_chunks_document_chunk_index
ON document_chunks (document_id, chunk_index);

INSERT INTO system_settings (key, value, description, category)
VALUES (
    'enable_reranker',
    'true',
    'Enable reranking after fusion; uses Jina when configured and local lexical fallback otherwise',
    'retrieval'
)
ON CONFLICT (key) DO NOTHING;
