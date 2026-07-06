from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session
from app.db.models import SystemSetting, AuditLog
from app.core.config import settings as env_settings

# Keys that require all documents to be reindexed when changed
REINDEX_REQUIRED_KEYS = {"chunk_size", "chunk_overlap", "embedding_model"}

DEFAULT_SETTINGS = [
    {"key": "chunk_size", "value": "800", "description": "Size of each text chunk (characters)", "category": "rag"},
    {"key": "chunk_overlap", "value": "100", "description": "Character overlap between consecutive chunks", "category": "rag"},
    {"key": "top_k_dense", "value": "20", "description": "Candidate pool size for dense (vector) retrieval", "category": "retrieval"},
    {"key": "top_k_sparse", "value": "20", "description": "Candidate pool size for sparse (FTS) retrieval", "category": "retrieval"},
    {"key": "top_k_final", "value": "5", "description": "Final number of chunks passed to the LLM", "category": "retrieval"},
    {"key": "rrf_k", "value": "60", "description": "RRF smoothing constant — higher values reduce top-rank influence", "category": "retrieval"},
    {"key": "enable_reranker", "value": "false", "description": "Enable cross-encoder reranking after fusion", "category": "retrieval"},
    {"key": "embedding_model", "value": env_settings.EMBEDDING_MODEL, "description": "Model used to generate embeddings", "category": "embeddings"},
    {"key": "similarity_metric", "value": "cosine", "description": "Vector similarity metric", "category": "retrieval"},
    {"key": "max_upload_size_mb", "value": "20", "description": "Maximum allowed upload size in megabytes", "category": "upload"},
    {"key": "allowed_extensions", "value": "pdf,docx,txt", "description": "Comma-separated list of allowed file extensions", "category": "upload"},
    {"key": "crawler_allowlist", "value": "www.udom.ac.tz,portal.udom.ac.tz,coi.udom.ac.tz,oas.udom.ac.tz,sr2.udom.ac.tz", "description": "Comma-separated domains allowed for crawling", "category": "crawler"},
    {"key": "crawler_blocklist", "value": "twitter.com,facebook.com,instagram.com,/university_documents/,/login,/logout,/admin,/search,/profile", "description": "Comma-separated domains or paths strictly blocked from crawling", "category": "crawler"},
    {"key": "crawler_max_age_days", "value": "90", "description": "Maximum age of announcements in days (0 to disable)", "category": "crawler"},
]

def seed_default_settings(db: Session):
    """Seed default system settings into the DB if they don't exist."""
    for setting_data in DEFAULT_SETTINGS:
        existing = db.query(SystemSetting).filter_by(key=setting_data["key"]).first()
        if not existing:
            setting = SystemSetting(
                key=setting_data["key"],
                value=setting_data["value"],
                description=setting_data["description"],
                category=setting_data["category"]
            )
            db.add(setting)
    db.commit()



class SettingsService:
    """
    Provides typed, cached access to system settings stored in the database.

    Usage
    -----
    svc = SettingsService(db)
    svc.chunk_size       -> int
    svc.top_k_dense      -> int
    svc.rrf_k            -> int
    svc.embedding_model  -> str

    Cache
    -----
    Settings are loaded from the DB once per instance on first access.
    The full cache is invalidated whenever update() is called.
    """

    def __init__(self, db: Session):
        self.db = db
        self._cache: Optional[Dict[str, str]] = None  # raw string values, None = not loaded

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        """Eagerly load all settings from the DB into the raw string cache."""
        if self._cache is not None:
            return
        rows = self.db.query(SystemSetting).all()
        self._cache = {row.key: row.value for row in rows}

    def _raw(self, key: str) -> Optional[str]:
        """Return the raw string value for a key, or None if not found."""
        self._load_all()
        return self._cache.get(key) if self._cache is not None else None

    def _int(self, key: str, default: int) -> int:
        raw = self._raw(key)
        if raw is None:
            return default
        try:
            return int(raw)
        except (ValueError, TypeError):
            return default

    def _bool(self, key: str, default: bool) -> bool:
        raw = self._raw(key)
        if raw is None:
            return default
        return raw.strip().lower() in ("true", "1", "yes")

    def _str(self, key: str, default: str) -> str:
        raw = self._raw(key)
        return raw if raw is not None else default

    def _list(self, key: str, default: list) -> list:
        raw = self._raw(key)
        if raw is None:
            return default
        return [ext.strip() for ext in raw.split(",") if ext.strip()]

    # ------------------------------------------------------------------
    # Typed properties – RAG
    # ------------------------------------------------------------------

    @property
    def chunk_size(self) -> int:
        return self._int("chunk_size", env_settings.DOCUMENT_CHUNK_SIZE)

    @property
    def chunk_overlap(self) -> int:
        return self._int("chunk_overlap", env_settings.DOCUMENT_CHUNK_OVERLAP)

    # ------------------------------------------------------------------
    # Typed properties – Retrieval
    # ------------------------------------------------------------------

    @property
    def top_k_dense(self) -> int:
        return self._int("top_k_dense", env_settings.DENSE_TOP_K)

    @property
    def top_k_sparse(self) -> int:
        return self._int("top_k_sparse", env_settings.SPARSE_TOP_K)

    @property
    def top_k_final(self) -> int:
        return self._int("top_k_final", env_settings.HYBRID_TOP_K)

    @property
    def rrf_k(self) -> int:
        return self._int("rrf_k", env_settings.RRF_K)

    @property
    def enable_reranker(self) -> bool:
        return self._bool("enable_reranker", False)

    # ------------------------------------------------------------------
    # Typed properties – Embeddings
    # ------------------------------------------------------------------

    @property
    def embedding_model(self) -> str:
        return self._str("embedding_model", env_settings.EMBEDDING_MODEL)

    # ------------------------------------------------------------------
    # Typed properties – Upload
    # ------------------------------------------------------------------

    @property
    def max_upload_size_mb(self) -> int:
        return self._int("max_upload_size_mb", 20)

    @property
    def allowed_extensions(self) -> List[str]:
        return self._list("allowed_extensions", ["pdf", "docx", "txt", "md"])

    # ------------------------------------------------------------------
    # Typed properties – Crawler
    # ------------------------------------------------------------------

    @property
    def crawler_allowlist(self) -> List[str]:
        return self._list("crawler_allowlist", ["www.udom.ac.tz", "portal.udom.ac.tz", "coi.udom.ac.tz", "oas.udom.ac.tz", "sr2.udom.ac.tz"])

    @property
    def crawler_blocklist(self) -> List[str]:
        return self._list("crawler_blocklist", ["twitter.com", "facebook.com", "instagram.com", "/university_documents/", "/login", "/logout", "/admin", "/search", "/profile"])

    @property
    def crawler_max_age_days(self) -> int:
        return self._int("crawler_max_age_days", 90)

    # ------------------------------------------------------------------
    # Legacy generic accessor (kept for backwards compat)
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Generic typed-value accessor. Prefer named properties where possible."""
        raw = self._raw(key)
        if raw is None:
            return default

        if raw.strip().lower() in ("true", "false"):
            return raw.strip().lower() == "true"

        try:
            return int(raw)
        except ValueError:
            pass

        try:
            return float(raw)
        except ValueError:
            pass

        return raw

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def update(self, key: str, value: str, admin_username: str = "System") -> SystemSetting:
        setting = self.db.query(SystemSetting).filter(SystemSetting.key == key).first()
        if not setting:
            raise KeyError(f"Setting '{key}' not found")

        old_value = setting.value
        value = str(value).strip()

        if old_value == value:
            # no change, no need to audit
            return setting

        # Validation
        try:
            if key == "chunk_size":
                if not (500 <= int(value) <= 2000):
                    raise ValueError("chunk_size must be between 500 and 2000")
            elif key == "chunk_overlap":
                if not (0 <= int(value) <= 500):
                    raise ValueError("chunk_overlap must be between 0 and 500")
            elif key == "top_k_dense":
                if not (1 <= int(value) <= 50):
                    raise ValueError("top_k_dense must be between 1 and 50")
            elif key == "top_k_sparse":
                if not (1 <= int(value) <= 50):
                    raise ValueError("top_k_sparse must be between 1 and 50")
            elif key == "top_k_final":
                if not (1 <= int(value) <= 20):
                    raise ValueError("top_k_final must be between 1 and 20")
            elif key == "rrf_k":
                if not (10 <= int(value) <= 200):
                    raise ValueError("rrf_k must be between 10 and 200")
            elif key == "max_upload_size_mb":
                if not (1 <= int(value) <= 100):
                    raise ValueError("max_upload_size_mb must be between 1 and 100")
            elif key == "embedding_model":
                if not value:
                    raise ValueError("embedding_model cannot be empty")
            elif key == "allowed_extensions":
                import re
                value = value.replace(" ", "")
                if not value or not re.match(r"^[a-zA-Z0-9]+(,[a-zA-Z0-9]+)*$", value):
                    raise ValueError(
                        "allowed_extensions must be a comma-separated list (e.g., pdf,docx,txt)"
                    )
            elif key in ("crawler_allowlist", "crawler_blocklist"):
                import re
                value = value.replace(" ", "")
                if value and not re.match(r"^[a-zA-Z0-9.\-_/]+(,[a-zA-Z0-9.\-_/]+)*$", value):
                    raise ValueError(f"{key} must be a comma-separated list of domains or paths")
            elif key == "crawler_max_age_days":
                if not (0 <= int(value) <= 3650):
                    raise ValueError("crawler_max_age_days must be between 0 and 3650")
        except ValueError as e:
            if "invalid literal for int()" in str(e):
                raise ValueError(f"{key} must be an integer")
            raise

        setting.value = value
        
        # Create audit log
        audit = AuditLog(
            admin=admin_username,
            setting_key=key,
            old_value=old_value,
            new_value=value
        )
        self.db.add(audit)
        
        self.db.commit()
        self.db.refresh(setting)

        # Invalidate the full cache so all properties reload on next access
        self._cache = None

        # Mark all documents as needing reindex if a critical key changed
        if key in REINDEX_REQUIRED_KEYS:
            self._mark_all_needs_reindex()

        return setting

    def _mark_all_needs_reindex(self) -> None:
        """Bulk-update all active/indexed documents to needs_reindex status."""
        from app.db.models import DocumentModel
        (
            self.db.query(DocumentModel)
            .filter(DocumentModel.status.in_(["active", "indexed"]))
            .update({"status": "needs_reindex"}, synchronize_session=False)
        )
        self.db.commit()

    # ------------------------------------------------------------------
    # Read-all (for admin list endpoint)
    # ------------------------------------------------------------------

    def get_all(self):
        return (
            self.db.query(SystemSetting)
            .order_by(SystemSetting.category, SystemSetting.key)
            .all()
        )
