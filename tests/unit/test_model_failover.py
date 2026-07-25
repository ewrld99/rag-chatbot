import pytest

from app.services.generation_resilience import GenerationUnavailableError
from app.services.model_failover import ModelFailoverService
from app.services.model_router import InvalidModelPreference, ModelRouter


class StubSettings:
    generation_user_selection_enabled = True
    generation_allowed_models = [
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
        "qwen/qwen3.6-27b",
        "openai/gpt-oss-20b",
        "llama-3.1-8b-instant",
    ]
    generation_default_model = "llama-3.3-70b-versatile"
    generation_answer_model_order = generation_allowed_models
    generation_utility_model_order = [
        "openai/gpt-oss-20b",
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "qwen/qwen3.6-27b",
        "openai/gpt-oss-120b",
    ]


class PassthroughResilience:
    def call(self, _operation, callback, **_kwargs):
        return callback()


def test_selected_answer_model_is_tried_before_automatic_fallbacks():
    router = ModelRouter(StubSettings())

    assert router.candidates("document_answer", "qwen/qwen3.6-27b") == [
        "qwen/qwen3.6-27b",
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "llama-3.1-8b-instant",
    ]


def test_utility_operations_use_the_low_cost_order():
    router = ModelRouter(StubSettings())

    assert router.candidates("intent_classifier", "qwen/qwen3.6-27b")[0] == (
        "openai/gpt-oss-20b"
    )


def test_disabled_or_unknown_model_cannot_be_selected():
    settings = StubSettings()
    settings.generation_allowed_models = ["openai/gpt-oss-20b"]
    settings.generation_default_model = "openai/gpt-oss-20b"
    router = ModelRouter(settings)

    with pytest.raises(InvalidModelPreference):
        router.normalize_preference("qwen/qwen3.6-27b")


def test_failed_selected_model_falls_back_and_reports_selected_model():
    router = ModelRouter(StubSettings())
    service = ModelFailoverService(
        router,
        resilience=PassthroughResilience(),
    )
    calls = []

    def callback(model):
        calls.append(model)
        if model == "qwen/qwen3.6-27b":
            raise GenerationUnavailableError(retry_after=600)
        return "answer"

    result = service.execute(
        "document_answer",
        "qwen/qwen3.6-27b",
        callback,
    )

    assert result.value == "answer"
    assert result.selected_model == "llama-3.3-70b-versatile"
    assert result.fallback_used is True
    assert calls == ["qwen/qwen3.6-27b", "llama-3.3-70b-versatile"]
