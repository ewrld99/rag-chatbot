"""Re-populate tsv columns with bilingual (simple + english) tsvector

Revision ID: b2c3d4e5f6a7
Revises: 7a8b9c0d1e2f
Create Date: 2026-07-09 22:48:00.000000

What changed in code
--------------------
The FTS triggers in app/db/session.py now build a combined tsvector:

    to_tsvector('simple', text) || to_tsvector('english', text)

Previously they used only 'simple'. The combined tsvector contains BOTH
exact word forms (usable by Swahili queries) AND English-stemmed lexemes
(usable by English queries), enabling bilingual full-text search.

Why a re-population migration is needed
---------------------------------------
The trigger only fires on INSERT or UPDATE. Existing rows already in the
database were built with the old 'simple'-only config. This migration
rebuilds ALL existing tsv/fts_vector columns so the data is consistent
with the new trigger.

After this migration
--------------------
All FTS queries (sparse_retriever.py) use:

    websearch_to_tsquery('simple',  :query)   -- exact match (Swahili / any)
    OR
    websearch_to_tsquery('english', :query)   -- stemmed match (English)

Example matches enabled after this migration
--------------------------------------------
  English: "register"   → matches "registration", "registered"   (stemming)
  English: "fee"        → matches "fees"                         (stemming)
  Swahili: "masomo"     → matches "masomo"                       (exact)
  Swahili: "wanafunzi"  → matches "wanafunzi"                    (exact)
  Swahili: "usajili"    → matches "usajili"                      (exact)
"""
from typing import Sequence, Union

from alembic import op
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, Sequence[str], None] = '7a8b9c0d1e2f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Re-build all stored tsvector values using the combined config.

    These are plain UPDATE statements (not DDL), so they run inside the
    normal transaction and are safe to roll back if something goes wrong.

    On a large corpus this may take a few minutes — it is a full table
    scan. Run during low-traffic hours or a maintenance window.
    """
    conn = op.get_bind()

    # ── document_chunks.tsv ───────────────────────────────────────────────────
    conn.execute(text("""
        UPDATE document_chunks
        SET tsv = to_tsvector('simple',  chunk_text)
               || to_tsvector('english', chunk_text)
        WHERE chunk_text IS NOT NULL
    """))

    # ── faqs.fts_vector ───────────────────────────────────────────────────────
    conn.execute(text("""
        UPDATE faqs
        SET fts_vector = to_tsvector('simple',  question || ' ' || answer)
                      || to_tsvector('english', question || ' ' || answer)
        WHERE question IS NOT NULL
          AND answer   IS NOT NULL
    """))


def downgrade() -> None:
    """
    Revert to the old 'simple'-only tsvector.

    Note: after downgrading you must also revert the trigger functions in
    app/db/session.py and the SQL in sparse_retriever.py manually.
    """
    conn = op.get_bind()

    conn.execute(text("""
        UPDATE document_chunks
        SET tsv = to_tsvector('simple', chunk_text)
        WHERE chunk_text IS NOT NULL
    """))

    conn.execute(text("""
        UPDATE faqs
        SET fts_vector = to_tsvector('simple', question || ' ' || answer)
        WHERE question IS NOT NULL
          AND answer   IS NOT NULL
    """))
