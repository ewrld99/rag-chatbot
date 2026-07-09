import os
import sys
# Add parent directory to path to allow importing app module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sqlalchemy import text
from app.db.session import engine

with engine.begin() as conn:
    conn.execute(text("UPDATE document_chunks SET tsv = to_tsvector('simple', chunk_text)"))
    conn.execute(text("UPDATE faqs SET fts_vector = to_tsvector('simple', question || ' ' || answer)"))
    print("FTS vectors updated successfully.")
