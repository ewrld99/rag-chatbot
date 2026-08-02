from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import logging
from threading import BoundedSemaphore, Lock
import time
from typing import Any, Callable

from app.core.config import settings


logger = logging.getLogger(__name__)


class RetrievalBusyError(RuntimeError):
    """Raised when bounded retrieval capacity cannot accept more work."""


class BoundedHybridExecutor:
    def __init__(
        self,
        max_workers: int,
        max_pending: int,
        queue_timeout: float,
    ) -> None:
        self.max_workers = max(1, int(max_workers))
        self.max_pending = max(self.max_workers, int(max_pending))
        self.queue_timeout = max(0.1, float(queue_timeout))
        self._slots = BoundedSemaphore(self.max_pending)
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="hybrid-retrieval",
        )

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        started_at = time.perf_counter()
        if not self._slots.acquire(timeout=self.queue_timeout):
            raise RetrievalBusyError(
                "Hybrid retrieval capacity is busy; retry the request shortly."
            )
        wait_ms = round((time.perf_counter() - started_at) * 1000, 1)
        if wait_ms >= 1.0:
            logger.info("Hybrid retrieval queue wait | wait_ms=%s", wait_ms)

        try:
            future = self._executor.submit(fn, *args, **kwargs)
        except Exception:
            self._slots.release()
            raise
        future.add_done_callback(lambda _future: self._slots.release())
        return future

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)


_executor: BoundedHybridExecutor | None = None
_executor_lock = Lock()


def get_hybrid_executor() -> BoundedHybridExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = BoundedHybridExecutor(
                settings.HYBRID_RETRIEVAL_MAX_WORKERS,
                settings.HYBRID_RETRIEVAL_MAX_PENDING,
                settings.HYBRID_RETRIEVAL_QUEUE_TIMEOUT_SECONDS,
            )
        return _executor


def close_hybrid_executor() -> None:
    global _executor
    with _executor_lock:
        executor = _executor
        _executor = None
    if executor is not None:
        executor.shutdown()
