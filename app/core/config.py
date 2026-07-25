
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    from pydantic import BaseSettings
    SettingsConfigDict = None

class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""
    JINA_API_KEY: str = ""
    JINA_API_BASE_URL: str = "https://api.jina.ai/v1"
    JINA_AUTH_COOLDOWN_SECONDS: float = 300.0
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
    GROQ_ALLOWED_MODELS: str = (
        "llama-3.3-70b-versatile,"
        "openai/gpt-oss-120b,"
        "qwen/qwen3.6-27b,"
        "openai/gpt-oss-20b,"
        "llama-3.1-8b-instant"
    )
    GROQ_ANSWER_MODEL_ORDER: str = GROQ_ALLOWED_MODELS
    GROQ_UTILITY_MODEL_ORDER: str = (
        "openai/gpt-oss-20b,"
        "llama-3.1-8b-instant,"
        "llama-3.3-70b-versatile,"
        "qwen/qwen3.6-27b,"
        "openai/gpt-oss-120b"
    )
    GROQ_USER_MODEL_SELECTION_ENABLED: bool = True
    GENERATION_MAX_RETRIES: int = 2
    GENERATION_RETRY_BASE_SECONDS: float = 0.5
    GENERATION_RETRY_JITTER_SECONDS: float = 0.25
    GENERATION_MAX_SHORT_RETRY_SECONDS: float = 5.0
    GENERATION_CIRCUIT_FAILURE_THRESHOLD: int = 3
    GENERATION_CIRCUIT_COOLDOWN_SECONDS: float = 30.0
    GENERATION_PERMISSION_COOLDOWN_SECONDS: float = 300.0
    GENERATION_MAX_CIRCUIT_SECONDS: float = 900.0

    COHERE_API_KEY: str = ""

    # --- CORS ---
    # Comma-separated list of allowed frontend origins.
    # Example: ALLOWED_ORIGINS=https://chat.udom.ac.tz,https://www.udom.ac.tz
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    # --- File Uploads & Server ---
    BASE_URL: str = "http://localhost:8000"
    UPLOADS_DIR: str = "uploads"

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
