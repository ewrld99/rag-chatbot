import pytest
from starlette.requests import Request

from app.api.limiter import _get_client_ip
from app.core.config import settings
from app.core.security import validate_auth_configuration
from app.schemas.chat import ChatRequest
from app.schemas.query import QueryRequest
from app.services.dense_retriever import (
    DenseRetriever,
    _set_embedding_failed,
    dense_embedding_failed,
)
from app.services.hybrid_retriever import HybridRetriever
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


def test_parallel_dense_success_clears_coordinator_failure_flag(db_session, monkeypatch):
    from app.services import hybrid_retriever as hybrid_module
    from app.services.dense_retriever import _set_embedding_failed as set_worker_state

    def dense_success(_self, *_args, **_kwargs):
        set_worker_state(False)
        return []

    monkeypatch.setattr(DenseRetriever, "retrieve", dense_success)
    monkeypatch.setattr(hybrid_module.SparseRetriever, "retrieve", lambda *_args, **_kwargs: [])
    _set_embedding_failed(True)

    retriever = HybridRetriever(
        db_session,
        top_k=5,
        dense_top_k=5,
        sparse_top_k=5,
        rrf_k=60,
    )
    retriever._retrieve_parallel("postponement", None)

    assert dense_embedding_failed() is False
