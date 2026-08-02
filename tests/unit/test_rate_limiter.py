import asyncio

from app.api import limiter as limiter_module
from app.api.limiter import _BoundedLocalLimiter, _rate_key, check_rate_limit
from app.core.config import settings


def test_local_limiter_expires_and_reports_accurate_retry_time():
    limiter = _BoundedLocalLimiter(max_keys=10)

    assert limiter.check("chat:1:10000:user", limit=2, window=10, now=100).allowed
    assert limiter.check("chat:1:10000:user", limit=2, window=10, now=102).allowed
    blocked = limiter.check("chat:1:10000:user", limit=2, window=10, now=103)

    assert blocked.allowed is False
    assert blocked.retry_after == 7
    assert limiter.check("chat:1:10000:user", limit=2, window=10, now=111).allowed


def test_local_limiter_evicts_lru_identities_at_capacity():
    limiter = _BoundedLocalLimiter(max_keys=2)
    limiter.check("scope:1:1000:a", limit=1, window=1, now=1)
    limiter.check("scope:1:1000:b", limit=1, window=1, now=1)
    limiter.check("scope:1:1000:c", limit=1, window=1, now=1)

    assert limiter.size() == 2
    assert limiter.check("scope:1:1000:a", limit=1, window=1, now=1.1).allowed


def test_rate_keys_isolate_endpoint_limit_and_window():
    assert _rate_key("chat", "ip:1", 2, 60) != _rate_key("query", "ip:1", 2, 60)
    assert _rate_key("chat", "ip:1", 2, 60) != _rate_key("chat", "ip:1", 3, 60)
    assert _rate_key("chat", "ip:1", 2, 60) != _rate_key("chat", "ip:1", 2, 30)


def test_redis_failure_enters_cooldown_and_uses_local_fallback(monkeypatch):
    class BrokenRedis:
        @classmethod
        def from_url(cls, *_args, **_kwargs):
            return cls()

        async def ping(self):
            raise ConnectionError("redis unavailable")

        async def aclose(self):
            self.closed = True

    monkeypatch.setattr(settings, "RATE_LIMIT_REDIS_URL", "redis://unavailable")
    monkeypatch.setattr(limiter_module, "Redis", BrokenRedis)
    limiter_module._redis_client = None
    limiter_module._redis_cooldown_until = 0
    limiter_module._redis_degraded = False

    decision = asyncio.run(
        check_rate_limit(scope="test", subject="user:1", limit=2, window=10)
    )

    assert decision.allowed
    assert decision.mode == "local_degraded"
    assert limiter_module.rate_limiter_status()["degraded"] is True

