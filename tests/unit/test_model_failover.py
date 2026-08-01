import pytest

from app.services.generation_resilience import GenerationUnavailableError
from app.services.model_failover import ModelFailoverService
from app.services.model_router import InvalidModelPreference, ModelRouter


class StubSettings:
    generation_user_selection_enabled = True
    generation_allowed_models = [
        "gemma3:4b",
        "qwen3.5:4b",
        "gemini-3.6-flash",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ]
    generation_default_model = "gemma3:4b"
    generation_answer_model_order = [
        "gemma3:4b",
        "gemini-3.6-flash",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "qwen3.5:4b",
    ]
    generation_utility_model_order = [
        "qwen2.5:1.5b",
        "gemma3:4b",
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
    ]


class PassthroughResilience:
    def __init__(self):
        self.calls = []

    def call(self, _operation, callback, **_kwargs):
        self.calls.append(_kwargs)
        return callback()


def test_selected_answer_model_is_tried_before_automatic_fallbacks():
    router = ModelRouter(StubSettings())

    assert router.candidates("document_answer", "llama-3.1-8b-instant") == [
        "llama-3.1-8b-instant",
        "gemma3:4b",
        "gemini-3.6-flash",
        "llama-3.3-70b-versatile",
        "qwen3.5:4b",
    ]


@pytest.mark.parametrize(
    "operation",
    [
        "intent_classifier",
        "conversation_relation",
        "query_rewrite",
        "clarification_check",
    ],
)
def test_lightweight_utility_operations_use_the_configured_utility_order(operation):
    router = ModelRouter(StubSettings())

    assert router.candidates(operation, "gemini-3.6-flash") == [
        "qwen2.5:1.5b",
        "gemma3:4b",
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
    ]


def test_grounding_verification_uses_gemma_first():
    router = ModelRouter(StubSettings())

    assert router.candidates("grounding_verification", "auto") == [
        "gemma3:4b",
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "qwen2.5:1.5b",
    ]


def test_stale_gemini_utility_setting_is_filtered_at_runtime():
    settings = StubSettings()
    settings.generation_utility_model_order = [
        "gemini-3.6-flash",
        "llama-3.3-70b-versatile",
    ]

    assert ModelRouter(settings).candidates("grounding_verification", "auto") == [
        "gemma3:4b",
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "qwen2.5:1.5b",
    ]


def test_disabled_unknown_or_utility_only_model_cannot_be_selected():
    settings = StubSettings()
    settings.generation_allowed_models = ["llama-3.1-8b-instant"]
    settings.generation_default_model = "llama-3.1-8b-instant"
    router = ModelRouter(settings)

    with pytest.raises(InvalidModelPreference):
        router.normalize_preference("qwen/qwen3.6-27b")

    with pytest.raises(InvalidModelPreference):
        router.normalize_preference("qwen2.5:1.5b")


def test_failed_selected_model_falls_back_and_reports_selected_model():
    router = ModelRouter(StubSettings())
    resilience = PassthroughResilience()
    service = ModelFailoverService(
        router,
        resilience=resilience,
    )
    calls = []

    def callback(model):
        calls.append(model)
        if model in {"llama-3.1-8b-instant", "qwen3.5:4b", "gemma3:4b"}:
            raise GenerationUnavailableError(retry_after=600)
        return "answer"

    result = service.execute(
        "document_answer",
        "llama-3.1-8b-instant",
        callback,
    )

    assert result.value == "answer"
    assert result.selected_model == "gemini-3.6-flash"
    assert result.fallback_used is True
    assert calls == [
        "llama-3.1-8b-instant",
        "gemma3:4b",
        "gemini-3.6-flash",
    ]
    assert [call["circuit_key"] for call in resilience.calls] == [
        "groq:llama-3.1-8b-instant",
        "ollama:gemma3:4b",
        "gemini:gemini-3.6-flash",
    ]
