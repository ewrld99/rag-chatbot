import pytest
from threading import Event, get_ident
from starlette.requests import Request

from app.api.limiter import _get_client_ip
from app.core.config import settings
from app.core.security import validate_auth_configuration
from app.schemas.chat import ChatRequest
from app.schemas.query import QueryRequest
from app.services.dense_retriever import (
    DenseRetrievalOutcome,
    DenseRetriever,
)
from app.services.hybrid_retriever import HybridRetriever
from app.services.hybrid_executor import BoundedHybridExecutor, RetrievalBusyError
from app.services.model_catalog import SUPPORTED_MODEL_IDS


def _request(peer: str, forwarded_for: str | None = None) -> Request:
    headers = []
    if forwarded_for:
        headers.append((b"x-forwarded-for", forwarded_for.encode("ascii")))
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": headers,
            "client": (peer, 1234),
            "server": ("testserver", 80),
            "scheme": "http",
        }
    )


def test_forwarded_for_is_ignored_from_untrusted_peer(monkeypatch):
    monkeypatch.setattr(settings, "TRUSTED_PROXY_CIDRS", "")
    request = _request("198.51.100.4", "203.0.113.8")

    assert _get_client_ip(request) == "198.51.100.4"


def test_forwarded_for_uses_first_untrusted_hop(monkeypatch):
    monkeypatch.setattr(settings, "TRUSTED_PROXY_CIDRS", "10.0.0.0/8")
    request = _request("10.0.0.3", "203.0.113.8, 10.0.0.2")

    assert _get_client_ip(request) == "203.0.113.8"


def test_auth_secret_is_required_outside_tests(monkeypatch):
    monkeypatch.setattr(settings, "TESTING", False)
    monkeypatch.setattr(settings, "AUTH_SECRET_KEY", "")

    with pytest.raises(RuntimeError, match="AUTH_SECRET_KEY"):
        validate_auth_configuration()


def test_request_schemas_use_the_catalog_for_answer_models():
    for model in ("auto", *SUPPORTED_MODEL_IDS):
        assert ChatRequest(message="hello", model_preference=model).model_preference == model
        assert QueryRequest(question="hello", model_preference=model).model_preference == model

    with pytest.raises(ValueError):
        ChatRequest(message="hello", model_preference="qwen2.5:1.5b")


def test_parallel_dense_outcome_is_request_scoped(db_session, monkeypatch):
    from app.services import hybrid_retriever as hybrid_module

    outcomes = iter(
        [
            DenseRetrievalOutcome([], embedding_failed=True),
            DenseRetrievalOutcome([], embedding_failed=False),
        ]
    )

    monkeypatch.setattr(
        DenseRetriever,
        "retrieve_outcome",
        lambda *_args, **_kwargs: next(outcomes),
    )
    monkeypatch.setattr(hybrid_module.SparseRetriever, "retrieve", lambda *_args, **_kwargs: [])

    retriever = HybridRetriever(
        db_session,
        top_k=5,
        dense_top_k=5,
        sparse_top_k=5,
        rrf_k=60,
    )
    first = retriever._retrieve_parallel("postponement", None)
    second = retriever._retrieve_parallel("postponement", None)

    assert first[-1] is True
    assert second[-1] is False


def test_bounded_hybrid_executor_rejects_saturated_queue():
    release = Event()
    executor = BoundedHybridExecutor(max_workers=1, max_pending=1, queue_timeout=0.01)
    future = executor.submit(release.wait)
    try:
        with pytest.raises(RetrievalBusyError):
            executor.submit(lambda: None)
    finally:
        release.set()
        future.result(timeout=1)
        executor.shutdown()


def test_sequential_retrieval_owns_distinct_short_sessions(monkeypatch):
    from app.services import hybrid_retriever as hybrid_module

    sessions = []
    calls = []

    class TrackingSession:
        def __init__(self):
            self.closed = False
            sessions.append(self)

        def rollback(self):
            pass

        def close(self):
            self.closed = True

    def record_dense(self, *_args, **_kwargs):
        calls.append(("dense", id(self.db), get_ident()))
        return DenseRetrievalOutcome([])

    def record_sparse(self, *_args, **_kwargs):
        calls.append(("sparse", id(self.db), get_ident()))
        return []

    monkeypatch.setattr(DenseRetriever, "retrieve_outcome", record_dense)
    monkeypatch.setattr(hybrid_module.SparseRetriever, "retrieve", record_sparse)
    retriever = HybridRetriever(
        top_k=5,
        dense_top_k=5,
        sparse_top_k=5,
        rrf_k=60,
        session_factory=TrackingSession,
    )

    retriever._retrieve_sequential("postponement", None)

    assert len(sessions) == 2
    assert calls[0][1] != calls[1][1]
    assert all(session.closed for session in sessions)
