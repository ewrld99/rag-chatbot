import hashlib
import math
import re
import time
from typing import List, Union
from collections import OrderedDict
from openai import APIStatusError, OpenAI, RateLimitError

from app.core.config import settings


class EmbeddingServiceError(RuntimeError):
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


_QUERY_EMBEDDING_CACHE = OrderedDict()
CACHE_MAX_SIZE = 2000


from sqlalchemy.orm import Session
from app.services.settings_service import SettingsService

class EmbeddingService:
    """
    Handles all embedding operations for the RAG system.
    Supports:
    - Single text embedding
    - Batch embedding
    """

    def __init__(self, db: Session):
        settings_svc = SettingsService(db)
        
        # Read from dynamic settings if available, else fallback to env config
        self.provider = settings.EMBEDDING_PROVIDER.lower()
        self.model = settings_svc.embedding_model
        self.dimension = settings.EMBEDDING_DIMENSION
        self.batch_size = max(settings.EMBEDDING_BATCH_SIZE, 1)
        self.client = None

        if self.provider == "local":
            return

        if self.provider == "jina":
            if not settings.JINA_API_KEY:
                raise ValueError("JINA_API_KEY is missing in environment variables")
            self.client = OpenAI(
                api_key=settings.JINA_API_KEY,
                base_url=settings.JINA_API_BASE_URL,
                timeout=15.0
            )
            return

        if self.provider != "openai":
            raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {settings.EMBEDDING_PROVIDER}")

        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is missing in environment variables")

        self.client = OpenAI(api_key=settings.OPENAI_API_KEY, timeout=15.0)

    # ---------------------------------------
    # 1. Single Text Embedding
    # ---------------------------------------
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text input.
        """

        if not text or not text.strip():
            raise ValueError("Input text for embedding cannot be empty")

        cache_key = f"{self.provider}_{self.model}_{hashlib.md5(text.encode('utf-8')).hexdigest()}"
        if cache_key in _QUERY_EMBEDDING_CACHE:
            _QUERY_EMBEDDING_CACHE.move_to_end(cache_key)
            return _QUERY_EMBEDDING_CACHE[cache_key]

        if self.provider == "local":
            result = self._embed_local(text)
        elif self.provider == "jina":
            result = self._embed_jina(text)[0]
        else:
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text
                )

                embedding = response.data[0].embedding
                self._validate_dimension(embedding)
                result = embedding
            except Exception as e:
                raise self._to_embedding_error(e)

        _QUERY_EMBEDDING_CACHE[cache_key] = result
        if len(_QUERY_EMBEDDING_CACHE) > CACHE_MAX_SIZE:
            _QUERY_EMBEDDING_CACHE.popitem(last=False)
            
        return result

    # ---------------------------------------
    # 2. Batch Embedding (IMPORTANT)
    # ---------------------------------------
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.
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
            for start in range(0, len(clean_texts), self.batch_size):
                batch = clean_texts[start:start + self.batch_size]
                embeddings.extend(self.embed_batch(batch))
            return embeddings

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
        if "1010" in message:
            message = (
                "Jina rejected the request with error code 1010. "
                "Check that JINA_API_KEY is valid and try again; if it persists, "
                "Jina may be blocking this network/IP."
            )
        raise EmbeddingServiceError(message, status_code=error.status_code)

    def _validate_dimension(self, embedding: List[float]) -> None:
        if len(embedding) != self.dimension:
            raise EmbeddingServiceError(
                f"Embedding dimension mismatch: model returned {len(embedding)} values, "
                f"but EMBEDDING_DIMENSION is {self.dimension}."
            )

    def _embed_local(self, text: str) -> List[float]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
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


def get_embeddings(texts: List[str], db: Session) -> List[List[float]]:
    return EmbeddingService(db).embed_batch(texts)
