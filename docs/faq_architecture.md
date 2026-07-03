# FAQ Management Architecture

This document describes the architectural implementation of the FAQ Management module within the University Hybrid RAG Chatbot.

## Database Schema

The FAQ module introduces the `faqs` table alongside the existing `document_chunks` table:
- **`id`**: UUID primary key.
- **`question` / `answer`**: TEXT fields storing the core content.
- **`category`**: Optional TEXT for organization.
- **`embedding`**: `VECTOR(1024)` column indexed with HNSW (`vector_cosine_ops`) for dense retrieval.
- **`fts_vector`**: `TSVECTOR` column indexed with GIN for sparse (keyword) retrieval.
- **`is_active`**: Boolean flag to enable/disable FAQs without deleting them.

### FTS Trigger
Similar to documents, the `fts_vector` is automatically managed via a PostgreSQL trigger (`trg_faqs_fts`). When an FAQ is inserted or updated, the database natively compiles the English TSVECTOR representation of the question and answer.

## Hybrid Retrieval Integration

The system uses a Reciprocal Rank Fusion (RRF) strategy:
1. **Dense Retriever**: Computes cosine distance between the user's query and `faqs.embedding` (alongside `document_chunks.embedding`). It merges and sorts both streams before passing them back.
2. **Sparse Retriever**: Executes PostgreSQL Full-Text Search against `faqs.fts_vector` and merges the hits with `document_chunks.tsv` search results based on `ts_rank_cd`.
3. **RRF Algorithm**: Takes the merged lists from both retrievers and ranks them identically. The LLM simply sees them as chunks with `"source_type": "faq"`.

## Performance: Async & Batched Operations

During **Bulk Import (CSV/Excel)**:
1. **Async**: The `POST /faqs/import` API accepts a FastAPI `BackgroundTasks` object. The endpoint returns immediately to keep the UI responsive.
2. **Batched Embeddings**: The background worker chunks valid rows into batches of `50`. It sends them collectively to the `get_embeddings` service, eliminating redundant API overhead and avoiding rate limit spikes.

## Security & Auditing

- **Admin Requirement**: All `POST`, `PUT`, `DELETE`, and `POST /import` endpoints require `Depends(require_admin)`, ensuring the decoded JWT belongs to an established admin user.
- **Audit Logging**: Every mutating action is logged into the `audit_logs` table via `log_audit()`:
  - `FAQ_CREATE`
  - `FAQ_UPDATE`
  - `FAQ_DELETE`
  - `FAQ_IMPORT`
  This enables administrators to track system mutations over time.

## Limitations & Best Practices

- **Lengths**: Questions are validated to a max of 500 chars, and answers to 10,000 chars.
- **CSV Format**: Must contain `Question` and `Answer` columns. Optionally `Category`.
