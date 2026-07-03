import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sqlalchemy.orm import Session
from app.db.session import engine
from app.db.models import SystemSetting

DEFAULT_SETTINGS = [
    {"key": "chunk_size", "value": "800", "description": "Size of each text chunk (characters)", "category": "rag"},
    {"key": "chunk_overlap", "value": "100", "description": "Character overlap between consecutive chunks", "category": "rag"},
    {"key": "top_k_dense", "value": "20", "description": "Candidate pool size for dense (vector) retrieval", "category": "retrieval"},
    {"key": "top_k_sparse", "value": "20", "description": "Candidate pool size for sparse (FTS) retrieval", "category": "retrieval"},
    {"key": "top_k_final", "value": "5", "description": "Final number of chunks passed to the LLM", "category": "retrieval"},
    {"key": "rrf_k", "value": "60", "description": "RRF smoothing constant — higher values reduce top-rank influence", "category": "retrieval"},
    {"key": "enable_reranker", "value": "false", "description": "Enable cross-encoder reranking after fusion", "category": "retrieval"},
    {"key": "embedding_model", "value": "jina-embeddings-v2-base-en", "description": "Model used to generate embeddings", "category": "embeddings"},
    {"key": "similarity_metric", "value": "cosine", "description": "Vector similarity metric", "category": "retrieval"},
    {"key": "max_upload_size_mb", "value": "20", "description": "Maximum allowed upload size in megabytes", "category": "upload"},
    {"key": "allowed_extensions", "value": "pdf,docx,txt", "description": "Comma-separated list of allowed file extensions", "category": "upload"},
]

def seed_settings():
    print("Seeding system settings...")
    with Session(engine) as session:
        for setting_data in DEFAULT_SETTINGS:
            existing = session.query(SystemSetting).filter_by(key=setting_data["key"]).first()
            if not existing:
                setting = SystemSetting(
                    key=setting_data["key"],
                    value=setting_data["value"],
                    description=setting_data["description"],
                    category=setting_data["category"]
                )
                session.add(setting)
        session.commit()
    print("Settings seeded successfully.")

if __name__ == "__main__":
    seed_settings()
