
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    from pydantic import BaseSettings
    SettingsConfigDict = None

class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""
    JINA_API_KEY: str = ""
    JINA_API_BASE_URL: str = "https://api.jina.ai/v1"
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/rag_db"
    EMBEDDING_PROVIDER: str = "jina"
    EMBEDDING_MODEL: str = "jina-embeddings-v3"
    EMBEDDING_DIMENSION: int = 1024
    EMBEDDING_BATCH_SIZE: int = 15
    DOCUMENT_CHUNK_SIZE: int = 1800
    DOCUMENT_CHUNK_OVERLAP: int = 120
    ADMIN_USERNAMES: str = "admin"

    # --- Admin seed (used only by scripts/create_admin.py) ---
    ADMIN_USERNAME: str = ""
    ADMIN_PASSWORD: str = ""

    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # --- Hybrid Retrieval ---
    DENSE_TOP_K: int = 20       # candidate pool from pgvector
    SPARSE_TOP_K: int = 20      # candidate pool from FTS
    RRF_K: int = 60             # RRF constant (higher = smoother rank decay)
    HYBRID_TOP_K: int = 5       # final chunks passed to the LLM

    if SettingsConfigDict:
        model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)
    else:
        class Config:
            env_file = ".env"
            case_sensitive = False

settings = Settings()
