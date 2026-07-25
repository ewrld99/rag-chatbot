from __future__ import annotations

import math
import threading
import time

from app.core.config import settings


class JinaProviderCooldownError(RuntimeError):
    """Raised before a Jina call while its account-level circuit is open."""

    def __init__(self, retry_after: int) -> None:
        super().__init__("Jina is temporarily unavailable after an account authorization or balance failure.")
        self.retry_after = retry_after


class JinaProviderCircuit:
    """Process-local cooldown shared by Jina embeddings and reranking."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._open_until = 0.0

    def before_call(self) -> None:
        with self._lock:
            remaining = self._open_until - time.monotonic()
        if remaining > 0:
            raise JinaProviderCooldownError(max(1, math.ceil(remaining)))

    def record_account_failure(self) -> None:
        cooldown = max(1.0, float(settings.JINA_AUTH_COOLDOWN_SECONDS))
        with self._lock:
            self._open_until = max(self._open_until, time.monotonic() + cooldown)

    def record_success(self) -> None:
        with self._lock:
            self._open_until = 0.0

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._open_until > time.monotonic()

    def reset(self) -> None:
        with self._lock:
            self._open_until = 0.0


def is_jina_account_failure(exc: Exception) -> bool:
    """Identify errors that affect all Jina services using the configured key."""
    response = getattr(exc, "response", None)
    status_code = (
        getattr(exc, "status_code", None)
        or getattr(response, "status_code", None)
    )
    message = str(exc).casefold()
    code = str(getattr(exc, "code", "") or "").casefold()
    markers = (
        "authz_insufficient_balance",
        "insufficient account balance",
        "insufficient balance",
        "invalid api key",
        "authentication",
        "authorization",
        "forbidden",
    )
    return status_code in {401, 402, 403} or any(
        marker in message or marker in code
        for marker in markers
    )


jina_provider_circuit = JinaProviderCircuit()
