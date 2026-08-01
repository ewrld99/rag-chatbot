import pytest

from app.services.generation_service import GenerationService, get_generation_clients


@pytest.fixture(autouse=True)
def reset_generation_client_pool():
    get_generation_clients.cache_clear()
    yield
    get_generation_clients.cache_clear()


def test_generation_clients_disable_provider_auto_retries(monkeypatch):
    calls = {}

    class FakeGroq:
        def __init__(self, **kwargs):
            calls["groq_sync"] = kwargs

    class FakeAsyncGroq:
        def __init__(self, **kwargs):
            calls["groq_async"] = kwargs

    class FakeOpenAI:
        def __init__(self, **kwargs):
            key = (
                "ollama_sync"
                if str(kwargs.get("base_url", "")).startswith("http://localhost:11434")
                else "gemini_sync"
            )
            calls[key] = kwargs

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            key = (
                "ollama_async"
                if str(kwargs.get("base_url", "")).startswith("http://localhost:11434")
                else "gemini_async"
            )
            calls[key] = kwargs

    monkeypatch.setattr("app.services.generation_service.Groq", FakeGroq)
    monkeypatch.setattr("app.services.generation_service.AsyncGroq", FakeAsyncGroq)
    monkeypatch.setattr("app.services.generation_service.OpenAI", FakeOpenAI)
    monkeypatch.setattr("app.services.generation_service.AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr("app.services.generation_service.settings.GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setattr("app.services.generation_service.settings.GROQ_API_KEY", "test-key")

    first = GenerationService()
    second = GenerationService()

    assert first.ollama_client is second.ollama_client
    assert first.async_ollama_client is second.async_ollama_client

    assert calls["gemini_sync"]["max_retries"] == 0
    assert calls["gemini_async"]["max_retries"] == 0
    assert calls["gemini_sync"]["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert calls["ollama_sync"]["max_retries"] == 0
    assert calls["ollama_async"]["max_retries"] == 0
    assert calls["ollama_sync"]["base_url"] == "http://localhost:11434/v1/"
    assert calls["groq_sync"]["max_retries"] == 0
    assert calls["groq_async"]["max_retries"] == 0


def test_missing_gemini_key_fails_clearly_when_gemini_is_default(monkeypatch):
    monkeypatch.setattr("app.services.generation_service.settings.GEMINI_API_KEY", "")
    monkeypatch.setattr("app.services.generation_service.settings.GROQ_MODEL", "gemini-3.6-flash")
    monkeypatch.setattr(
        "app.services.generation_service.settings.GROQ_ALLOWED_MODELS",
        "gemini-3.6-flash,llama-3.3-70b-versatile,llama-3.1-8b-instant",
    )

    try:
        GenerationService()
    except ValueError as exc:
        assert "GEMINI_API_KEY is missing" in str(exc)
    else:
        raise AssertionError("GenerationService should require GEMINI_API_KEY for Gemini default")
