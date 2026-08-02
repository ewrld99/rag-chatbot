import pytest

from app.api.deps import get_rag_pipeline
from app.main import app
from app.services.generation_resilience import GenerationUnavailableError


SAFE_SOURCE = {
    "document_id": "document-1",
    "name": "regulations.pdf",
    "url": "/uploads/regulations.pdf",
}


@pytest.fixture(autouse=True)
def configure_generation_keys(monkeypatch):
    monkeypatch.setattr("app.services.generation_service.settings.GEMINI_API_KEY", "test-gemini-key")


class FailingPipeline:
    def run(self, *_args, **_kwargs):
        raise GenerationUnavailableError(
            retry_after=600,
            error_id="safe-error-id",
            sources=[SAFE_SOURCE],
        )

    async def stream_events(self, *_args, **_kwargs):
        yield {"type": "sources", "sources": [SAFE_SOURCE]}
        raise GenerationUnavailableError(
            retry_after=600,
            error_id="safe-error-id",
            sources=[SAFE_SOURCE],
        )


def test_chat_hides_generation_provider_error(client):
    app.dependency_overrides[get_rag_pipeline] = lambda: FailingPipeline()

    response = client.post("/api/chat/", json={"message": "How is GPA calculated?"})

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["code"] == "GENERATION_TEMPORARILY_UNAVAILABLE"
    assert detail["sources"] == [SAFE_SOURCE]
    assert detail["error_id"] == "safe-error-id"
    assert "temporarily busy" in detail["message"]
    assert "groq" not in response.text.lower()
    assert "429" not in response.text
    assert "tokens per day" not in response.text.lower()


def test_chat_debug_requires_admin(client):
    response = client.post(
        "/api/chat/?debug=true",
        json={"message": "How is GPA calculated?"},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Debug mode requires admin access"


def test_chat_stream_hides_generation_provider_error_and_keeps_sources(client):
    app.dependency_overrides[get_rag_pipeline] = lambda: FailingPipeline()

    response = client.post(
        "/api/chat/stream",
        json={"message": "How is GPA calculated?"},
    )

    assert response.status_code == 200
    assert '"type": "error"' in response.text
    assert '"code": "GENERATION_TEMPORARILY_UNAVAILABLE"' in response.text
    assert '"name": "regulations.pdf"' in response.text
    assert "temporarily busy" in response.text
    assert "groq" not in response.text.lower()
    assert "429" not in response.text
    assert "tokens per day" not in response.text.lower()


def test_model_catalog_exposes_only_the_approved_models(client):
    response = client.get("/api/chat/models")

    assert response.status_code == 200
    body = response.json()
    assert body["default"] == "auto"
    assert [model["id"] for model in body["models"]] == [
        "llama3.2:3b",
        "gemma3:4b",
        "gemini-3.6-flash",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "qwen3.5:4b",
    ]


def test_chat_rejects_a_model_outside_the_allowlist(client):
    response = client.post(
        "/api/chat/",
        json={
            "message": "How is GPA calculated?",
            "model_preference": "unapproved/model",
        },
    )

    assert response.status_code == 422


def test_chat_accepts_gemini_flash_model_preference(client):
    app.dependency_overrides[get_rag_pipeline] = lambda: FailingPipeline()

    response = client.post(
        "/api/chat/",
        json={
            "message": "How is GPA calculated?",
            "model_preference": "gemini-3.6-flash",
        },
    )

    assert response.status_code == 503


def test_chat_rejects_disabled_gpt_oss_120b_model(client):
    response = client.post(
        "/api/chat/",
        json={
            "message": "How is GPA calculated?",
            "model_preference": "openai/gpt-oss-120b",
        },
    )

    assert response.status_code == 422


def test_chat_rejects_disabled_gpt_oss_20b_model(client):
    response = client.post(
        "/api/chat/",
        json={
            "message": "How is GPA calculated?",
            "model_preference": "openai/gpt-oss-20b",
        },
    )

    assert response.status_code == 422


def test_chat_rejects_removed_qwen_model(client):
    response = client.post(
        "/api/chat/",
        json={
            "message": "How is GPA calculated?",
            "model_preference": "qwen/qwen3.6-27b",
        },
    )

    assert response.status_code == 422
