from __future__ import annotations

from app.api.deps import get_rag_pipeline
from app.main import app


class GuestHistoryPipeline:
    def __init__(self) -> None:
        self.history = None

    async def stream_events(
        self,
        _message,
        chat_history=None,
        user_profile=None,
        model_preference="auto",
    ):
        self.history = chat_history
        yield {"type": "stream", "token": "Resolved follow-up."}


def test_guest_sse_passes_recent_history_to_pipeline(client):
    pipeline = GuestHistoryPipeline()
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline

    response = client.post(
        "/api/chat/stream",
        json={
            "message": "How is it truncated?",
            "history": [
                {"role": "user", "content": "How is GPA calculated?"},
                {
                    "role": "assistant",
                    "content": "GPA uses course scores and total credits.",
                },
            ],
        },
    )

    assert response.status_code == 200
    assert "Resolved follow-up." in response.text
    assert pipeline.history == [
        {"role": "user", "content": "How is GPA calculated?"},
        {
            "role": "assistant",
            "content": "GPA uses course scores and total credits.",
        },
    ]
