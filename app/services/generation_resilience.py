from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import logging
import math
import random
import re
import threading
import time
from typing import Any, TypeVar
from uuid import uuid4

from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
)

from app.core.config import settings


logger = logging.getLogger(__name__)
T = TypeVar("T")


def is_structured_output_error(exc: Exception) -> bool:
    message = str(exc).casefold()
    return any(
        marker in message
        for marker in (
            "failed to generate json",
            "failed to validate json",
            "failed_generation",
        )
    )


class GenerationUnavailableError(RuntimeError):
    """Safe internal exception representing an unavailable generation provider."""

    code = "GENERATION_TEMPORARILY_UNAVAILABLE"

    def __init__(
        self,
        *,
        retry_after: int | None = None,
        error_id: str | None = None,
        sources: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(self.code)
        self.retry_after = retry_after
        self.error_id = error_id or uuid4().hex[:12]
        self.sources = list(sources or [])

    def attach_sources(self, sources: list[dict[str, Any]]) -> "GenerationUnavailableError":
        self.sources = list(sources)
        return self

    def public_payload(
        self,
        *,
        sources: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        safe_sources = list(self.sources if sources is None else sources)
        if safe_sources:
            message = (
                "The answer service is temporarily busy. Relevant UDOM sources "
                "are shown below. Please try again shortly."
            )
        else:
            message = "The answer service is temporarily busy. Please try again shortly."

        payload: dict[str, Any] = {
            "code": self.code,
            "message": message,
            "error_id": self.error_id,
            "sources": safe_sources,
        }
        if self.retry_after is not None:
            payload["retry_after"] = self.retry_after
        return payload


class GenerationCircuitBreaker:
    """Process-local circuit breaker for one provider/model key."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._failures = 0
        self._open_until = 0.0

    def before_call(self) -> None:
        with self._lock:
            now = time.monotonic()
            if self._open_until > now:
                raise GenerationUnavailableError(
                    retry_after=max(1, math.ceil(self._open_until - now))
                )
            if self._open_until:
                self._failures = 0
                self._open_until = 0.0

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._open_until = 0.0

    def record_failure(
        self,
        *,
        retry_after: float | None,
        open_immediately: bool,
    ) -> None:
        with self._lock:
            self._failures += 1
            threshold = max(1, settings.GENERATION_CIRCUIT_FAILURE_THRESHOLD)
            if not open_immediately and self._failures < threshold:
                return

            cooldown = retry_after or settings.GENERATION_CIRCUIT_COOLDOWN_SECONDS
            cooldown = max(1.0, min(cooldown, settings.GENERATION_MAX_CIRCUIT_SECONDS))
            self._open_until = max(self._open_until, time.monotonic() + cooldown)

    def reset(self) -> None:
        with self._lock:
            self._failures = 0
            self._open_until = 0.0


class GenerationResilience:
    """Bounded retries, quota detection, and circuit breaking for Groq calls."""

    def __init__(self) -> None:
        self._circuits: dict[str, GenerationCircuitBreaker] = {}
        self._circuits_lock = threading.Lock()
        self.circuit = self._circuit_for("default")

    def call(
        self,
        operation: str,
        callback: Callable[[], T],
        *,
        circuit_key: str = "default",
        model: str | None = None,
    ) -> T:
        circuit = self._circuit_for(circuit_key)
        circuit.before_call()
        attempt = 0

        while True:
            try:
                result = callback()
                circuit.record_success()
                return result
            except GenerationUnavailableError:
                raise
            except Exception as exc:
                policy = self._policy(exc)
                error_id = uuid4().hex[:12]
                logger.warning(
                    "Generation provider failure | error_id=%s operation=%s "
                    "model=%s attempt=%d retryable=%s retry_after=%s error=%s",
                    error_id,
                    operation,
                    model or "default",
                    attempt + 1,
                    policy["retryable"],
                    policy["retry_after"],
                    exc,
                )

                if policy["retryable"] and attempt < settings.GENERATION_MAX_RETRIES:
                    delay = self._retry_delay(attempt, policy["retry_after"])
                    time.sleep(delay)
                    attempt += 1
                    continue

                if policy["count_failure"]:
                    circuit.record_failure(
                        retry_after=policy["retry_after"],
                        open_immediately=policy["open_immediately"],
                    )
                raise GenerationUnavailableError(
                    retry_after=self._public_retry_after(policy["retry_after"]),
                    error_id=error_id,
                ) from exc

    async def call_async(
        self,
        operation: str,
        callback: Callable[[], Awaitable[T]],
        *,
        circuit_key: str = "default",
        model: str | None = None,
    ) -> T:
        circuit = self._circuit_for(circuit_key)
        circuit.before_call()
        attempt = 0

        while True:
            try:
                result = await callback()
                circuit.record_success()
                return result
            except GenerationUnavailableError:
                raise
            except Exception as exc:
                policy = self._policy(exc)
                error_id = uuid4().hex[:12]
                logger.warning(
                    "Generation provider failure | error_id=%s operation=%s "
                    "model=%s attempt=%d retryable=%s retry_after=%s error=%s",
                    error_id,
                    operation,
                    model or "default",
                    attempt + 1,
                    policy["retryable"],
                    policy["retry_after"],
                    exc,
                )

                if policy["retryable"] and attempt < settings.GENERATION_MAX_RETRIES:
                    delay = self._retry_delay(attempt, policy["retry_after"])
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue

                if policy["count_failure"]:
                    circuit.record_failure(
                        retry_after=policy["retry_after"],
                        open_immediately=policy["open_immediately"],
                    )
                raise GenerationUnavailableError(
                    retry_after=self._public_retry_after(policy["retry_after"]),
                    error_id=error_id,
                ) from exc

    def reset(self) -> None:
        with self._circuits_lock:
            circuits = list(self._circuits.values())
        for circuit in circuits:
            circuit.reset()

    def _circuit_for(self, circuit_key: str) -> GenerationCircuitBreaker:
        with self._circuits_lock:
            circuit = self._circuits.get(circuit_key)
            if circuit is None:
                circuit = GenerationCircuitBreaker()
                self._circuits[circuit_key] = circuit
            return circuit

    def _policy(self, exc: Exception) -> dict[str, Any]:
        retry_after = self._retry_after_seconds(exc)
        message = str(exc).lower()

        if isinstance(exc, (AuthenticationError, PermissionDeniedError)) or (
            getattr(exc, "status_code", 0) in {401, 403}
            and any(
                marker in message
                for marker in (
                    "model_permission_blocked",
                    "permissions_error",
                    "project admin enable this model",
                )
            )
        ):
            return {
                "retryable": False,
                "retry_after": retry_after or settings.GENERATION_PERMISSION_COOLDOWN_SECONDS,
                "open_immediately": True,
                "count_failure": True,
            }

        if isinstance(exc, RateLimitError):
            is_daily_quota = any(
                marker in message
                for marker in ("tokens per day", "requests per day", "tpd", "rpd", "daily quota")
            )
            is_short = (
                retry_after is not None
                and retry_after <= settings.GENERATION_MAX_SHORT_RETRY_SECONDS
            )
            return {
                "retryable": bool(is_short and not is_daily_quota),
                "retry_after": retry_after,
                "open_immediately": bool(is_daily_quota or not is_short),
                "count_failure": True,
            }

        if isinstance(exc, (APIConnectionError, APITimeoutError, InternalServerError)):
            return {
                "retryable": True,
                "retry_after": retry_after,
                "open_immediately": False,
                "count_failure": True,
            }

        if isinstance(exc, APIStatusError) and getattr(exc, "status_code", 0) >= 500:
            return {
                "retryable": True,
                "retry_after": retry_after,
                "open_immediately": False,
                "count_failure": True,
            }

        if is_structured_output_error(exc):
            return {
                "retryable": False,
                "retry_after": None,
                "open_immediately": False,
                "count_failure": False,
            }

        return {
            "retryable": False,
            "retry_after": retry_after,
            "open_immediately": isinstance(exc, AuthenticationError),
            "count_failure": True,
        }

    def _retry_delay(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, settings.GENERATION_MAX_SHORT_RETRY_SECONDS)
        base = settings.GENERATION_RETRY_BASE_SECONDS * (2 ** attempt)
        jitter = random.uniform(0.0, max(0.0, settings.GENERATION_RETRY_JITTER_SECONDS))
        return min(base + jitter, settings.GENERATION_MAX_SHORT_RETRY_SECONDS)

    def _retry_after_seconds(self, exc: Exception) -> float | None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", {}) or {}

        for header in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
            parsed = self._parse_duration(headers.get(header))
            if parsed is not None:
                return parsed

        match = re.search(
            r"try again in\s*(?:(?P<hours>\d+)h)?"
            r"(?:(?P<minutes>\d+)m)?"
            r"(?:(?P<seconds>\d+(?:\.\d+)?)s)?",
            str(exc),
            flags=re.IGNORECASE,
        )
        if not match:
            return None

        hours = float(match.group("hours") or 0)
        minutes = float(match.group("minutes") or 0)
        seconds = float(match.group("seconds") or 0)
        total = (hours * 3600) + (minutes * 60) + seconds
        return total or None

    @staticmethod
    def _parse_duration(value: object) -> float | None:
        if value is None:
            return None
        text = str(value).strip().lower()
        try:
            return max(0.0, float(text))
        except ValueError:
            pass

        match = re.fullmatch(
            r"(?:(?P<hours>\d+)h)?"
            r"(?:(?P<minutes>\d+)m)?"
            r"(?:(?P<seconds>\d+(?:\.\d+)?)s)?",
            text,
        )
        if not match:
            return None
        total = (
            (float(match.group("hours") or 0) * 3600)
            + (float(match.group("minutes") or 0) * 60)
            + float(match.group("seconds") or 0)
        )
        return total or None

    @staticmethod
    def _public_retry_after(value: float | None) -> int | None:
        return max(1, math.ceil(value)) if value is not None else None


generation_resilience = GenerationResilience()
