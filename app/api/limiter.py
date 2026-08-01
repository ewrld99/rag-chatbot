"""Sliding-window rate limiting with trusted proxy and Redis support."""

from __future__ import annotations

import asyncio
from collections import defaultdict
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
except ImportError:  # pragma: no cover - exercised only in minimal installs
    Redis = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)
_request_log: dict[str, list[float]] = defaultdict(list)
_request_log_lock = Lock()
_redis_client: Any | None = None
_redis_lock = asyncio.Lock()

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
  return 0
end
redis.call('ZADD', key, now, member)
redis.call('PEXPIRE', key, ttl)
return 1
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

    # Walk right-to-left, discarding only explicitly trusted proxy hops. The
    # first untrusted address is the originating client.
    chain = [part.strip() for part in forwarded_for.split(",") if part.strip()]
    chain.append(peer)
    for candidate in reversed(chain):
        if not _is_trusted_proxy(candidate):
            return candidate
    return chain[0] if chain else peer


async def _get_redis_client() -> Any | None:
    global _redis_client
    if not settings.RATE_LIMIT_REDIS_URL:
        return None
    if Redis is None:
        raise RuntimeError(
            "RATE_LIMIT_REDIS_URL is configured but the redis package is not installed."
        )
    if _redis_client is not None:
        return _redis_client
    async with _redis_lock:
        if _redis_client is None:
            _redis_client = Redis.from_url(
                settings.RATE_LIMIT_REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
            )
    return _redis_client


async def close_rate_limiter() -> None:
    global _redis_client
    if _redis_client is None:
        return
    await _redis_client.aclose()
    _redis_client = None


class RateLimiter:
    def __init__(self, limit: int = 20, window: float = 60.0) -> None:
        self.limit = max(1, int(limit))
        self.window = max(0.1, float(window))

    async def __call__(self, request: Request) -> None:
        client_ip = _get_client_ip(request)
        redis_client = await _get_redis_client()
        allowed = await self._redis_allowed(redis_client, client_ip)
        if redis_client is None:
            allowed = self._memory_allowed(client_ip)
        if allowed:
            return
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded: max {self.limit} requests per "
                f"{int(self.window)}s. Please wait and try again."
            ),
            headers={"Retry-After": str(max(1, int(self.window)))},
        )

    async def _redis_allowed(self, client: Any | None, client_ip: str) -> bool:
        if client is None:
            return True
        now_ms = int(time.time() * 1000)
        window_ms = max(1, int(self.window * 1000))
        key = f"{settings.RATE_LIMIT_KEY_PREFIX}:{client_ip}"
        try:
            result = await client.eval(
                _REDIS_SLIDING_WINDOW,
                1,
                key,
                now_ms - window_ms,
                now_ms,
                self.limit,
                window_ms,
                f"{now_ms}:{uuid4().hex}",
            )
            return bool(result)
        except Exception:
            logger.exception("Redis rate limiter failed; using process-local fallback")
            return self._memory_allowed(client_ip)

    def _memory_allowed(self, client_ip: str) -> bool:
        now = time.monotonic()
        with _request_log_lock:
            recent = [
                timestamp
                for timestamp in _request_log[client_ip]
                if now - timestamp < self.window
            ]
            if len(recent) >= self.limit:
                _request_log[client_ip] = recent
                return False
            recent.append(now)
            _request_log[client_ip] = recent
            return True
