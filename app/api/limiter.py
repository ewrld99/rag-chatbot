"""
app/api/limiter.py
------------------
Custom sliding-window rate limiter — zero third-party dependencies.

Usage as a FastAPI dependency:

    from fastapi import Depends
    from app.api.limiter import RateLimiter

    @router.post("/")
    async def my_route(body: MySchema, _=Depends(RateLimiter(20, 60))):
        ...

The limiter raises HTTP 429 (Too Many Requests) with a Retry-After header
when the per-IP request count exceeds `limit` within `window` seconds.

⚠️  PRODUCTION WARNING — IN-MEMORY ONLY
This store lives in each Python process. It has two known limitations:

  1. Multi-worker: running `uvicorn --workers N` gives each worker an
     independent store, so a client can make N × limit requests before
     being throttled. To fix this, switch to a shared Redis backend
     (e.g. `slowapi` with `RedisStore`, or `fastapi-limiter`).

  2. Restart reset: counters are cleared whenever the process restarts.

For a single-worker development server both limitations are acceptable.
"""

import logging
import time
from collections import defaultdict
from typing import Dict, List

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)

# Shared in-memory request log: { client_ip: [monotonic_timestamps] }
_request_log: Dict[str, List[float]] = defaultdict(list)


def _get_client_ip(request: Request) -> str:
    """
    Extract the real client IP, honouring X-Forwarded-For when sitting
    behind a reverse proxy (nginx / Caddy / AWS ALB / etc.).
    """
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RateLimiter:
    """
    Sliding-window rate limiter designed as a FastAPI callable dependency.

    Parameters
    ----------
    limit  : maximum requests allowed inside the window
    window : window length in seconds (default 60 s)

    Raises
    ------
    HTTPException 429  when the client exceeds the configured rate.
    """

    def __init__(self, limit: int = 20, window: float = 60.0) -> None:
        self.limit = limit
        self.window = window

    def __call__(self, request: Request) -> None:
        ip = _get_client_ip(request)
        now = time.monotonic()

        # Evict timestamps that have fallen outside the current window
        _request_log[ip] = [t for t in _request_log[ip] if now - t < self.window]

        if len(_request_log[ip]) >= self.limit:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Rate limit exceeded: max {self.limit} requests per "
                    f"{int(self.window)}s. Please wait and try again."
                ),
                headers={"Retry-After": str(int(self.window))},
            )

        _request_log[ip].append(now)
