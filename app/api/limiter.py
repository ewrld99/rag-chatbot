"""Bounded sliding-window rate limiting with Redis and local fallback."""

from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from dataclasses import dataclass
import ipaddress
import logging
from threading import Lock
import time
from typing import Any
from uuid import uuid4

from fastapi import HTTPException, Request

from app.core.config import settings

try:
    from redis.asyncio import Redis
except ImportError:  # pragma: no cover
    Redis = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    retry_after: float = 0.0
    mode: str = "local"


class _BoundedLocalLimiter:
    def __init__(self, max_keys: int) -> None:
        self.max_keys = max(1, int(max_keys))
        self._entries: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = Lock()
        self._checks = 0

    def check(
        self,
        key: str,
        *,
        limit: int,
        window: float,
        now: float | None = None,
    ) -> RateLimitDecision:
        current = time.monotonic() if now is None else float(now)
        cutoff = current - window
        with self._lock:
            timestamps = self._entries.pop(key, deque())
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            if len(timestamps) >= limit:
                retry_after = max(0.001, timestamps[0] + window - current)
                self._entries[key] = timestamps
                self._entries.move_to_end(key)
                decision = RateLimitDecision(False, retry_after, "local")
            else:
                timestamps.append(current)
                self._entries[key] = timestamps
                decision = RateLimitDecision(True, 0.0, "local")

            self._checks += 1
            if self._checks % 128 == 0:
                self._sweep_locked(current)
            while len(self._entries) > self.max_keys:
                self._entries.popitem(last=False)
            return decision

    def _sweep_locked(self, current: float) -> None:
        expired = [
            key
            for key, timestamps in self._entries.items()
            if not timestamps
            or timestamps[-1]
            <= current - (int(key.split(":", 3)[2]) / 1000.0)
        ]
        for key in expired:
            self._entries.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._checks = 0

    def size(self) -> int:
        with self._lock:
            return len(self._entries)


_local_limiter = _BoundedLocalLimiter(settings.RATE_LIMIT_LOCAL_MAX_KEYS)
_redis_client: Any | None = None
_redis_lock = asyncio.Lock()
_redis_cooldown_until = 0.0
_redis_degraded = False

_REDIS_SLIDING_WINDOW = """
local key = KEYS[1]
local cutoff = tonumber(ARGV[1])
local now = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])
local member = ARGV[5]
redis.call('ZREMRANGEBYSCORE', key, '-inf', cutoff)
local count = redis.call('ZCARD', key)
if count >= limit then
  local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
  if oldest[2] then
    return {0, math.max(1, tonumber(oldest[2]) + ttl - now)}
  end
  return {0, ttl}
end
redis.call('ZADD', key, now, member)
redis.call('PEXPIRE', key, ttl)
return {1, 0}
"""


def _trusted_proxy_networks() -> tuple[Any, ...]:
    networks: list[Any] = []
    for raw in settings.TRUSTED_PROXY_CIDRS.split(","):
        value = raw.strip()
        if not value:
            continue
        try:
            networks.append(ipaddress.ip_network(value, strict=False))
        except ValueError:
            logger.warning("Ignoring invalid TRUSTED_PROXY_CIDRS entry: %s", value)
    return tuple(networks)


def _is_trusted_proxy(host: str) -> bool:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return any(address in network for network in _trusted_proxy_networks())


def _get_client_ip(request: Request) -> str:
    peer = request.client.host if request.client else "unknown"
    if not _is_trusted_proxy(peer):
        return peer
    forwarded_for = request.headers.get("X-Forwarded-For")
    if not forwarded_for:
        return peer
    chain = [part.strip() for part in forwarded_for.split(",") if part.strip()]
    chain.append(peer)
    for candidate in reversed(chain):
        if not _is_trusted_proxy(candidate):
            return candidate
    return chain[0] if chain else peer


def _rate_key(scope: str, subject: str, limit: int, window: float) -> str:
    window_ms = max(1, int(window * 1000))
    return f"{scope}:{limit}:{window_ms}:{subject}"


async def _get_redis_client() -> Any | None:
    global _redis_client, _redis_degraded
    if not settings.RATE_LIMIT_REDIS_URL:
        return None
    if time.monotonic() < _redis_cooldown_until:
        return None
    if Redis is None:
        await _mark_redis_failure(RuntimeError("redis package is not installed"))
        return None
    if _redis_client is not None:
        return _redis_client
    async with _redis_lock:
        if _redis_client is None:
            try:
                client = Redis.from_url(
                    settings.RATE_LIMIT_REDIS_URL,
                    encoding="utf-8",
                    decode_responses=True,
                )
                await client.ping()
                _redis_client = client
                if _redis_degraded:
                    logger.info("Redis rate limiter recovered")
                _redis_degraded = False
            except Exception as exc:
                try:
                    await client.aclose()
                except Exception:
                    pass
                await _mark_redis_failure(exc)
                return None
    return _redis_client


async def _mark_redis_failure(error: Exception) -> None:
    global _redis_client, _redis_cooldown_until, _redis_degraded
    client = _redis_client
    _redis_client = None
    _redis_cooldown_until = time.monotonic() + max(
        1.0,
        settings.RATE_LIMIT_REDIS_COOLDOWN_SECONDS,
    )
    if client is not None:
        try:
            await client.aclose()
        except Exception:
            pass
    if not _redis_degraded:
        logger.warning(
            "Redis rate limiter unavailable; using bounded local fallback: %s",
            error,
        )
    _redis_degraded = True


async def check_rate_limit(
    *,
    scope: str,
    subject: str,
    limit: int,
    window: float,
) -> RateLimitDecision:
    normalized_limit = max(1, int(limit))
    normalized_window = max(0.1, float(window))
    key = _rate_key(scope, subject, normalized_limit, normalized_window)
    client = await _get_redis_client()
    if client is not None:
        now_ms = int(time.time() * 1000)
        window_ms = max(1, int(normalized_window * 1000))
        redis_key = f"{settings.RATE_LIMIT_KEY_PREFIX}:{key}"
        try:
            result = await client.eval(
                _REDIS_SLIDING_WINDOW,
                1,
                redis_key,
                now_ms - window_ms,
                now_ms,
                normalized_limit,
                window_ms,
                f"{now_ms}:{uuid4().hex}",
            )
            allowed = bool(int(result[0]))
            retry_after = max(0.0, float(result[1]) / 1000.0)
            return RateLimitDecision(allowed, retry_after, "redis")
        except Exception as exc:
            await _mark_redis_failure(exc)

    decision = _local_limiter.check(
        key,
        limit=normalized_limit,
        window=normalized_window,
    )
    mode = "local_degraded" if settings.RATE_LIMIT_REDIS_URL else "local"
    return RateLimitDecision(decision.allowed, decision.retry_after, mode)


def rate_limiter_status() -> dict[str, Any]:
    configured = bool(settings.RATE_LIMIT_REDIS_URL)
    degraded = configured and (
        _redis_degraded or time.monotonic() < _redis_cooldown_until
    )
    return {
        "mode": "local_degraded" if degraded else "redis" if configured else "local",
        "degraded": degraded,
        "local_identities": _local_limiter.size(),
    }


async def close_rate_limiter() -> None:
    global _redis_client, _redis_cooldown_until, _redis_degraded
    client = _redis_client
    _redis_client = None
    if client is not None:
        await client.aclose()
    _redis_cooldown_until = 0.0
    _redis_degraded = False
    _local_limiter.clear()


class RateLimiter:
    def __init__(
        self,
        limit: int = 20,
        window: float = 60.0,
        scope: str = "http",
    ) -> None:
        self.limit = max(1, int(limit))
        self.window = max(0.1, float(window))
        self.scope = scope.strip() or "http"

    async def __call__(self, request: Request) -> None:
        decision = await check_rate_limit(
            scope=self.scope,
            subject=f"ip:{_get_client_ip(request)}",
            limit=self.limit,
            window=self.window,
        )
        if decision.allowed:
            return
        retry_after = max(1, int(decision.retry_after + 0.999))
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded: max {self.limit} requests per "
                f"{int(self.window)}s. Please wait and try again."
            ),
            headers={"Retry-After": str(retry_after)},
        )
