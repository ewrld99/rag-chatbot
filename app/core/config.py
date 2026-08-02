
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    from pydantic import BaseSettings
    SettingsConfigDict = None

class Settings(BaseSettings):
    TESTING: bool = False
    APP_TIMEZONE: str = "Africa/Nairobi"
    OPENAI_API_KEY: str = ""
    JINA_API_KEY: str = ""
    JINA_API_BASE_URL: str = "https://api.jina.ai/v1"
    JINA_AUTH_COOLDOWN_SECONDS: float = 300.0
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/rag_db"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 20
    DATABASE_POOL_TIMEOUT_SECONDS: float = 10.0
    EMBEDDING_PROVIDER: str = "ollama"
    EMBEDDING_MODEL: str = "bge-m3"
    EMBEDDING_DIMENSION: int = 1024
    EMBEDDING_BATCH_SIZE: int = 15
    QUERY_EMBEDDING_CACHE_MAX_SIZE: int = 2000
    QUERY_EMBEDDING_CACHE_TTL_SECONDS: float = 3600.0
    DOCUMENT_CHUNK_SIZE: int = 1800
    DOCUMENT_CHUNK_OVERLAP: int = 120
    ADMIN_USERNAMES: str = "admin"
    AUTH_SECRET_KEY: str = ""
    AUTH_TOKEN_TTL_SECONDS: int = 60 * 60 * 8
    CONVERSATION_TOKEN_TTL_SECONDS: int = 60 * 60 * 24
    TRUSTED_PROXY_CIDRS: str = ""
    RATE_LIMIT_REDIS_URL: str = ""
    RATE_LIMIT_KEY_PREFIX: str = "rag-chatbot:rate-limit"
    RATE_LIMIT_LOCAL_MAX_KEYS: int = 10000
    RATE_LIMIT_REDIS_COOLDOWN_SECONDS: float = 30.0
    CHAT_MAX_MESSAGE_CHARS: int = 8000
    CHAT_MAX_HISTORY_MESSAGES: int = 20
    WS_MAX_PAYLOAD_BYTES: int = 65536

    # --- Admin seed (used only by scripts/create_admin.py) ---
    ADMIN_USERNAME: str = ""
    ADMIN_PASSWORD: str = ""
    LOG_LEVEL: str = "INFO"

    GEMINI_API_KEY: str = ""
    GEMINI_OPENAI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_OPENAI_BASE_URL: str = "http://localhost:11434/v1/"
    OLLAMA_API_KEY: str = "ollama"

    # Provider-neutral generation settings. Empty values fall back to the
    # legacy GROQ_* names below so existing deployments remain compatible.
    GENERATION_MODEL: str = ""
    GENERATION_ALLOWED_MODELS: str = ""
    GENERATION_ANSWER_MODEL_ORDER: str = ""
    GENERATION_UTILITY_MODEL_ORDER: str = ""

    # Deprecated compatibility aliases; model_provider() performs routing.
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama3.2:3b"
    GROQ_ALLOWED_MODELS: str = (
        "llama3.2:3b,"
        "gemma3:4b,"
        "qwen3.5:4b,"
        "gemini-3.6-flash,"
        "llama-3.3-70b-versatile,"
        "llama-3.1-8b-instant"
    )
    GROQ_ANSWER_MODEL_ORDER: str = (
        "llama3.2:3b,"
        "gemma3:4b,"
        "gemini-3.6-flash,"
        "llama-3.3-70b-versatile,"
        "llama-3.1-8b-instant,"
        "qwen3.5:4b"
    )
    GROQ_UTILITY_MODEL_ORDER: str = (
        "qwen2.5:1.5b,"
        "gemma3:4b,"
        "llama-3.1-8b-instant,"
        "llama-3.3-70b-versatile"
    )
    GROQ_USER_MODEL_SELECTION_ENABLED: bool = True
    GENERATION_MAX_RETRIES: int = 2
    GENERATION_RETRY_BASE_SECONDS: float = 0.5
    GENERATION_RETRY_JITTER_SECONDS: float = 0.25
    GENERATION_MAX_SHORT_RETRY_SECONDS: float = 5.0
    GENERATION_CIRCUIT_FAILURE_THRESHOLD: int = 3
    GENERATION_CIRCUIT_COOLDOWN_SECONDS: float = 30.0
    GENERATION_EMPTY_RESPONSE_COOLDOWN_SECONDS: float = 120.0
    GENERATION_PERMISSION_COOLDOWN_SECONDS: float = 300.0
    GENERATION_MAX_CIRCUIT_SECONDS: float = 900.0
    GENERATION_CONTEXT_MAX_DOCS: int = 6
    GENERATION_CONTEXT_MAX_CHARS: int = 6500
    GENERATION_CONTEXT_MAX_CHARS_PER_DOC: int = 1100
    GENERATION_SIMPLE_CONTEXT_MAX_DOCS: int = 4
    GENERATION_SIMPLE_CONTEXT_MAX_CHARS: int = 4200
    GENERATION_SIMPLE_CONTEXT_MAX_CHARS_PER_DOC: int = 900
    GENERATION_PROCEDURE_CONTEXT_MAX_DOCS: int = 5
    GENERATION_PROCEDURE_CONTEXT_MAX_CHARS: int = 6500
    GENERATION_PROCEDURE_CONTEXT_MAX_CHARS_PER_DOC: int = 1400
    GENERATION_PROCEDURE_SECTION_WINDOW: int = 2
    GENERATION_SIMPLE_MAX_TOKENS: int = 550
    GENERATION_MEDIUM_MAX_TOKENS: int = 800
    GENERATION_COMPLEX_MAX_TOKENS: int = 1100
    GENERATION_VERIFICATION_EVIDENCE_CHARS: int = 1200
    GENERATION_VERIFICATION_BATCH_SIZE: int = 12
    GENERATION_ENABLE_ANSWER_REPAIR: bool = True
    GENERATION_REPAIR_MAX_TOKENS: int = 450
    GENERATION_LIGHTWEIGHT_REPAIR_MAX_TOKENS: int = 500

    COHERE_API_KEY: str = ""

    # --- CORS ---
    # Comma-separated list of allowed frontend origins.
    # Example: ALLOWED_ORIGINS=https://chat.udom.ac.tz,https://www.udom.ac.tz
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    # --- Web Crawler ---
    CRAWLER_ENABLED: bool = False

    # --- File Uploads & Server ---
    BASE_URL: str = "http://localhost:8000"
    UPLOADS_DIR: str = "uploads"
    FILE_RECONCILE_INTERVAL_SECONDS: float = 60.0
    FILE_STAGING_RETENTION_HOURS: int = 24
    FILE_TRASH_RETENTION_HOURS: int = 24

    # --- Hybrid Retrieval ---
    DENSE_TOP_K: int = 20       # candidate pool from pgvector
    SPARSE_TOP_K: int = 20      # candidate pool from FTS
    RRF_K: int = 60             # RRF constant (higher = smoother rank decay)
    HYBRID_TOP_K: int = 5       # final chunks passed to the LLM
    HYBRID_PARALLEL_RETRIEVAL: bool = True
    HYBRID_RETRIEVAL_MAX_WORKERS: int = 8
    HYBRID_RETRIEVAL_MAX_PENDING: int = 64
    HYBRID_RETRIEVAL_QUEUE_TIMEOUT_SECONDS: float = 10.0
    ADAPTIVE_RERANK_SKIP_HIGH_CONFIDENCE: bool = True

    if SettingsConfigDict:
        model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)
    else:
        class Config:
            env_file = ".env"
            case_sensitive = False

settings = Settings()
