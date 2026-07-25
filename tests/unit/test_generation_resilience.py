import httpx
import pytest
from groq import APIConnectionError, PermissionDeniedError, RateLimitError

from app.core.config import settings
from app.services.generation_resilience import (
    GenerationUnavailableError,
    generation_resilience,
)


@pytest.fixture(autouse=True)
def reset_generation_circuit():
    generation_resilience.reset()
    yield
    generation_resilience.reset()


def _rate_limit_error(retry_after: str = "780") -> RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(
        429,
        request=request,
        headers={"retry-after": retry_after},
        json={"error": {"message": "tokens per day limit reached", "code": "rate_limit_exceeded"}},
    )
    return RateLimitError(
        "tokens per day limit reached; provider-secret-detail",
        response=response,
        body=response.json(),
    )


def _permission_error() -> PermissionDeniedError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    body = {
        "error": {
            "message": "The model is blocked at the project level.",
            "type": "permissions_error",
            "code": "model_permission_blocked_project",
        }
    }
    response = httpx.Response(403, request=request, json=body)
    return PermissionDeniedError(
        "model_permission_blocked_project",
        response=response,
        body=body,
    )


def test_daily_quota_is_not_retried_and_opens_circuit():
    calls = 0

    def fail():
        nonlocal calls
        calls += 1
        raise _rate_limit_error()

    with pytest.raises(GenerationUnavailableError) as captured:
        generation_resilience.call("test_daily_quota", fail)

    error = captured.value
    assert calls == 1
    assert error.retry_after == 780
    assert "provider-secret-detail" not in str(error)
    assert "provider-secret-detail" not in str(error.public_payload())

    blocked_calls = 0

    def should_not_run():
        nonlocal blocked_calls
        blocked_calls += 1

    with pytest.raises(GenerationUnavailableError):
        generation_resilience.call("test_open_circuit", should_not_run)
    assert blocked_calls == 0


def test_transient_connection_error_retries_then_succeeds(monkeypatch):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    calls = 0
    monkeypatch.setattr("app.services.generation_resilience.time.sleep", lambda _: None)
    monkeypatch.setattr(settings, "GENERATION_RETRY_JITTER_SECONDS", 0.0)

    def eventually_succeeds():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise APIConnectionError(request=request)
        return "ok"

    assert generation_resilience.call("test_transient", eventually_succeeds) == "ok"
    assert calls == 2


def test_public_payload_changes_when_sources_are_available():
    error = GenerationUnavailableError(retry_after=30, error_id="safe-id")
    payload = error.public_payload(
        sources=[{"document_id": "1", "name": "regulations.pdf", "url": None}]
    )

    assert payload["code"] == "GENERATION_TEMPORARILY_UNAVAILABLE"
    assert payload["retry_after"] == 30
    assert payload["error_id"] == "safe-id"
    assert "sources are shown below" in payload["message"]
    assert payload["sources"][0]["name"] == "regulations.pdf"


def test_open_circuit_for_one_model_does_not_block_another_model():
    calls = {"model_a": 0, "model_b": 0}

    def fail_model_a():
        calls["model_a"] += 1
        raise _rate_limit_error()

    with pytest.raises(GenerationUnavailableError):
        generation_resilience.call(
            "answer",
            fail_model_a,
            circuit_key="groq:model-a",
            model="model-a",
        )

    result = generation_resilience.call(
        "answer",
        lambda: calls.__setitem__("model_b", calls["model_b"] + 1) or "ok",
        circuit_key="groq:model-b",
        model="model-b",
    )

    assert result == "ok"
    assert calls == {"model_a": 1, "model_b": 1}


def test_project_blocked_model_is_quarantined_immediately(monkeypatch):
    monkeypatch.setattr(settings, "GENERATION_PERMISSION_COOLDOWN_SECONDS", 300.0)
    calls = 0

    def fail():
        nonlocal calls
        calls += 1
        raise _permission_error()

    with pytest.raises(GenerationUnavailableError) as captured:
        generation_resilience.call(
            "document_answer_stream",
            fail,
            circuit_key="groq:blocked-model",
            model="blocked-model",
        )

    assert captured.value.retry_after == 300

    with pytest.raises(GenerationUnavailableError):
        generation_resilience.call(
            "document_answer_repair_stream",
            fail,
            circuit_key="groq:blocked-model",
            model="blocked-model",
        )

    assert calls == 1


def test_failed_json_generation_does_not_open_model_circuit(monkeypatch):
    monkeypatch.setattr(settings, "GENERATION_CIRCUIT_FAILURE_THRESHOLD", 1)
    calls = 0

    def fail_json():
        nonlocal calls
        calls += 1
        raise RuntimeError(
            "Failed to generate JSON. See 'failed_generation' for more details."
        )

    with pytest.raises(GenerationUnavailableError):
        generation_resilience.call(
            "document_answer_stream",
            fail_json,
            circuit_key="groq:json-capable-model",
            model="json-capable-model",
        )

    result = generation_resilience.call(
        "student_support_answer",
        lambda: "ok",
        circuit_key="groq:json-capable-model",
        model="json-capable-model",
    )

    assert result == "ok"
    assert calls == 1
