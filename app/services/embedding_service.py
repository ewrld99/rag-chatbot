import hashlib
import math
import time
import logging
from typing import List, Union
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from threading import RLock
import httpx
from openai import APIStatusError, OpenAI, RateLimitError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.services.jina_resilience import (
    JinaProviderCooldownError,
    is_jina_account_failure,
    jina_provider_circuit,
)
from app.services.query_normalization import tokenize
from app.services.settings_service import SettingsService
from app.services.tls_service import system_ssl_context

logger = logging.getLogger(__name__)


class EmbeddingServiceError(RuntimeError):
    def __init__(
        self,
        message: str,
        status_code: int = 502,
        *,
        provider_cooldown: bool = False,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.provider_cooldown = provider_cooldown


@dataclass(frozen=True)
class _CachedEmbedding:
    expires_at: float
    embedding: List[float]


_QUERY_EMBEDDING_CACHE = OrderedDict()
_QUERY_EMBEDDING_CACHE_LOCK = RLock()
_EMBEDDING_CLIENT_LOCK = RLock()


@lru_cache(maxsize=4)
def _ollama_client(base_url: str) -> httpx.Client:
    return httpx.Client(base_url=base_url.rstrip("/"), timeout=120.0)


@lru_cache(maxsize=4)
def _openai_embedding_client(
    provider: str,
    api_key: str,
    base_url: str,
) -> OpenAI:
    kwargs = {
        "api_key": api_key,
        "http_client": httpx.Client(
            verify=system_ssl_context(),
            timeout=15.0,
        ),
    }
    if base_url:
        kwargs["base_url"] = base_url
    if provider == "jina":
        kwargs["max_retries"] = 0
    return OpenAI(**kwargs)


def close_embedding_clients() -> None:
    """Close process-wide embedding transports during application shutdown."""
    with _EMBEDDING_CLIENT_LOCK:
        for client in list(_EMBEDDING_HTTP_CLIENTS):
            client.close()
        for client in list(_EMBEDDING_OPENAI_CLIENTS):
            client.close()
        _EMBEDDING_HTTP_CLIENTS.clear()
        _EMBEDDING_OPENAI_CLIENTS.clear()
        _ollama_client.cache_clear()
        _openai_embedding_client.cache_clear()


_EMBEDDING_HTTP_CLIENTS: set[httpx.Client] = set()
_EMBEDDING_OPENAI_CLIENTS: set[OpenAI] = set()


def _shared_ollama_client(base_url: str) -> httpx.Client:
    with _EMBEDDING_CLIENT_LOCK:
        client = _ollama_client(base_url)
        _EMBEDDING_HTTP_CLIENTS.add(client)
        return client


def _shared_openai_client(provider: str, api_key: str, base_url: str = "") -> OpenAI:
    with _EMBEDDING_CLIENT_LOCK:
        client = _openai_embedding_client(provider, api_key, base_url)
        _EMBEDDING_OPENAI_CLIENTS.add(client)
        return client


class EmbeddingService:
    """
    Handles all embedding operations for the RAG system.
    Supports:
    - Single text embedding
    - Batch embedding
    """

    def __init__(self, db: Session):
        settings_svc = SettingsService(db)
        
        # Read from dynamic settings if available, else fallback to env config.
        self.model = settings_svc.embedding_model
        self.provider = self._effective_provider(settings.EMBEDDING_PROVIDER, self.model)
        self.dimension = settings.EMBEDDING_DIMENSION
        self.batch_size = max(settings.EMBEDDING_BATCH_SIZE, 1)
        self.client = None
        logger.info(
            "EmbeddingService configured | provider=%s model=%s dimension=%s",
            self.provider,
            self.model,
            self.dimension,
        )

        if self.provider == "local":
            logger.warning(
                "EmbeddingService: provider='local' uses a hash-based fallback "
                "that produces random vectors. Semantic similarity search will NOT "
                "work correctly. Set EMBEDDING_PROVIDER=ollama, jina, or openai in your .env."
            )
            return

        if self.provider == "ollama":
            self.client = _shared_ollama_client(settings.OLLAMA_BASE_URL)
            return

        if self.provider == "jina":
            if not settings.JINA_API_KEY:
                raise ValueError("JINA_API_KEY is missing in environment variables")
            self.client = _shared_openai_client(
                "jina",
                settings.JINA_API_KEY,
                settings.JINA_API_BASE_URL,
            )
            return

        if self.provider != "openai":
            raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {settings.EMBEDDING_PROVIDER}")

        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is missing in environment variables")

        self.client = _shared_openai_client("openai", settings.OPENAI_API_KEY)

    @staticmethod
    def _effective_provider(configured_provider: str, model: str) -> str:
        provider = (configured_provider or "").strip().lower()
        model_name = (model or "").strip().lower()
        if provider == "local":
            return provider
        if model_name in {"bge-m3", "mxbai-embed-large", "nomic-embed-text", "all-minilm"}:
            return "ollama"
        return provider

    # ---------------------------------------
    # 1. Single Text Embedding
    # ---------------------------------------
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text input.
        """

        clean_text = text.strip() if text else ""
        if not clean_text:
            raise ValueError("Input text for embedding cannot be empty")

        cache_key = self._query_cache_key(clean_text)
        cached = self._get_cached_query_embedding(cache_key)
        if cached is not None:
            return cached

        if self.provider == "local":
            result = self._embed_local(clean_text)
        elif self.provider == "ollama":
            result = self._embed_ollama(clean_text)[0]
        elif self.provider == "jina":
            result = self._embed_jina(clean_text)[0]
        else:
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=clean_text
                )

                embedding = response.data[0].embedding
                self._validate_dimension(embedding)
                result = embedding
            except Exception as e:
                raise self._to_embedding_error(e)

        self._store_query_embedding(cache_key, result)
            
        return list(result)

    def _query_cache_key(self, text: str) -> str:
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        return f"{self.provider}:{self.model}:{self.dimension}:{digest}"

    def _get_cached_query_embedding(self, cache_key: str) -> Union[List[float], None]:
        ttl_seconds = settings.QUERY_EMBEDDING_CACHE_TTL_SECONDS
        max_size = settings.QUERY_EMBEDDING_CACHE_MAX_SIZE
        if ttl_seconds <= 0 or max_size <= 0:
            return None

        now = time.monotonic()
        with _QUERY_EMBEDDING_CACHE_LOCK:
            cached = _QUERY_EMBEDDING_CACHE.get(cache_key)
            if cached is None:
                return None
            if cached.expires_at <= now:
                _QUERY_EMBEDDING_CACHE.pop(cache_key, None)
                return None
            _QUERY_EMBEDDING_CACHE.move_to_end(cache_key)
            return list(cached.embedding)

    def _store_query_embedding(self, cache_key: str, embedding: List[float]) -> None:
        ttl_seconds = settings.QUERY_EMBEDDING_CACHE_TTL_SECONDS
        max_size = settings.QUERY_EMBEDDING_CACHE_MAX_SIZE
        if ttl_seconds <= 0 or max_size <= 0:
            return

        now = time.monotonic()
        expires_at = now + ttl_seconds
        with _QUERY_EMBEDDING_CACHE_LOCK:
            # Prune expired entries from the front of the cache
            while _QUERY_EMBEDDING_CACHE:
                oldest_key, oldest_val = next(iter(_QUERY_EMBEDDING_CACHE.items()))
                if oldest_val.expires_at <= now:
                    _QUERY_EMBEDDING_CACHE.pop(oldest_key, None)
                else:
                    break

            _QUERY_EMBEDDING_CACHE[cache_key] = _CachedEmbedding(
                expires_at=expires_at,
                embedding=list(embedding),
            )
            _QUERY_EMBEDDING_CACHE.move_to_end(cache_key)
            while len(_QUERY_EMBEDDING_CACHE) > max_size:
                _QUERY_EMBEDDING_CACHE.popitem(last=False)

    # ---------------------------------------
    # 2. Batch Embedding (IMPORTANT)
    # ---------------------------------------
    def embed_batch(self, texts: List[str], progress_callback=None) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        progress_callback: Optional callable that receives (current_batch, total_batches)
        """

        if not texts or not isinstance(texts, list):
            raise ValueError("Input must be a non-empty list of texts")

        # Remove empty texts safely
        clean_texts = [t.strip() for t in texts if t and t.strip()]

        if not clean_texts:
            raise ValueError("All input texts are empty")

        if self.provider == "local":
            return [self._embed_local(text) for text in clean_texts]

        if len(clean_texts) > self.batch_size:
            embeddings = []
            total_batches = (len(clean_texts) + self.batch_size - 1) // self.batch_size
            current_batch = 0
            
            for start in range(0, len(clean_texts), self.batch_size):
                current_batch += 1
                if progress_callback:
                    progress_callback(current_batch, total_batches)
                    
                batch = clean_texts[start:start + self.batch_size]
                embeddings.extend(self.embed_batch(batch))
            return embeddings

        if self.provider == "ollama":
            return self._embed_ollama(clean_texts)

        if self.provider == "jina":
            return self._embed_jina(clean_texts)

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=clean_texts
            )

            embeddings = [item.embedding for item in response.data]
            for embedding in embeddings:
                self._validate_dimension(embedding)
            return embeddings

        except Exception as e:
            raise self._to_embedding_error(e)

    def _to_embedding_error(self, error: Exception) -> EmbeddingServiceError:
        if isinstance(error, RateLimitError):
            code = getattr(error, "code", None)
            if code == "insufficient_quota":
                return EmbeddingServiceError(
                    "OpenAI quota exceeded. Check billing, credits, or use an API key with available quota.",
                    status_code=402,
                )

            return EmbeddingServiceError(
                "OpenAI rate limit reached. Wait a moment and try uploading again.",
                status_code=429,
            )

        if isinstance(error, APIStatusError):
            return EmbeddingServiceError(
                f"OpenAI embedding request failed: {error.message}",
                status_code=502,
            )

        return EmbeddingServiceError(f"Embedding service error: {str(error)}")

    def _embed_jina(self, input_value: Union[str, List[str]], retries=3) -> List[List[float]]:
        try:
            jina_provider_circuit.before_call()
        except JinaProviderCooldownError as error:
            raise EmbeddingServiceError(
                "Jina embeddings are temporarily unavailable after an account "
                "authorization or balance failure.",
                status_code=503,
                provider_cooldown=True,
            ) from error

        for attempt in range(retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=input_value,
                    extra_body={
                        "normalized": True,
                        "embedding_type": "float",
                    },
                )
                jina_provider_circuit.record_success()
                break
            except APIStatusError as error:
                code = getattr(error, "code", None)
                if (code == "RATE_TOKEN_LIMIT_EXCEEDED" or error.status_code == 429) and attempt < retries - 1:
                    sleep_time = 2 * (attempt + 1)
                    print(f"Jina API rate limit hit. Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                    continue
                return self._raise_jina_status_error(error)
            except Exception as error:
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                raise EmbeddingServiceError(f"Jina embedding request failed: {str(error)}")

        embeddings = [item.embedding for item in response.data]

        for embedding in embeddings:
            self._validate_dimension(embedding)

        return embeddings

    def _raise_jina_status_error(self, error: APIStatusError):
        message = getattr(error, "message", None) or str(error)
        if is_jina_account_failure(error):
            jina_provider_circuit.record_account_failure()
            raise EmbeddingServiceError(
                "Jina embedding authorization failed or the account balance is "
                "insufficient. Sparse retrieval remains available.",
                status_code=503,
            ) from error
        if "1010" in message:
            message = (
                "Jina rejected the request with error code 1010. "
                "Check that JINA_API_KEY is valid and try again; if it persists, "
                "Jina may be blocking this network/IP."
            )
        raise EmbeddingServiceError(message, status_code=error.status_code)

    def _embed_ollama(self, input_value: Union[str, List[str]]) -> List[List[float]]:
        try:
            response = self.client.post(
                "/api/embed",
                json={
                    "model": self.model,
                    "input": input_value,
                },
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as error:
            raise EmbeddingServiceError(
                f"Ollama embedding request failed: {error.response.text}",
                status_code=502,
            ) from error
        except Exception as error:
            raise EmbeddingServiceError(
                f"Ollama embedding request failed: {str(error)}",
                status_code=502,
            ) from error

        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list) or not embeddings:
            raise EmbeddingServiceError(
                "Ollama embedding response did not include embeddings.",
                status_code=502,
            )

        normalized_embeddings = [list(item) for item in embeddings]
        for embedding in normalized_embeddings:
            self._validate_dimension(embedding)
        return normalized_embeddings

    def _validate_dimension(self, embedding: List[float]) -> None:
        if len(embedding) != self.dimension:
            raise EmbeddingServiceError(
                f"Embedding dimension mismatch: model returned {len(embedding)} values, "
                f"but EMBEDDING_DIMENSION is {self.dimension}."
            )

    def _embed_local(self, text: str) -> List[float]:
        tokens = tokenize(text)
        vector = [0.0] * self.dimension

        features = tokens + [
            f"{tokens[i]} {tokens[i + 1]}"
            for i in range(len(tokens) - 1)
        ]

        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimension
            sign = 1.0 if value & 1 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(item * item for item in vector))
        if norm == 0:
            return vector

        return [item / norm for item in vector]

    # ---------------------------------------
    # 3. Utility: Safe Embedding (Fallback)
    # ---------------------------------------
    def safe_embed(self, text: str) -> Union[List[float], None]:
        """
        Returns embedding or None (prevents pipeline crash).
        Useful in ingestion pipelines.
        """

        try:
            return self.embed(text)
        except Exception:
            return None


# ---------------------------------------
# Convenience Functions (Backward Compatible but require db)
# ---------------------------------------
def get_embedding(text: str, db: Session) -> List[float]:
    return EmbeddingService(db).embed(text)


def get_embeddings(texts: List[str], db: Session, progress_callback=None) -> List[List[float]]:
    return EmbeddingService(db).embed_batch(texts, progress_callback=progress_callback)


def clear_query_embedding_cache() -> None:
    with _QUERY_EMBEDDING_CACHE_LOCK:
        _QUERY_EMBEDDING_CACHE.clear()
