from __future__ import annotations

import json

from app.api.deps import get_rag_pipeline
from app.main import app
from app.services.conversation_context import ConversationContextResolver


class TrackingPipeline:
    def __init__(self) -> None:
        self.resolver = ConversationContextResolver()
        self.decisions = []

    async def stream_events(self, message, chat_history=None, **_kwargs):
        conversation = self.resolver.resolve(message, chat_history).with_routing(
            "UDOM_DOCUMENT_SEARCH",
            None,
        )
        self.decisions.append(conversation)
        yield {
            "type": "routing",
            "routing": {"intent": "UDOM_DOCUMENT_SEARCH"},
            "conversation": conversation.to_dict(),
        }
        yield {"type": "stream", "token": f"Answer for: {conversation.standalone_query}"}
        yield {"type": "sources", "sources": []}
        yield {
            "type": "model",
            "requested_model": "auto",
            "selected_model": "gemma3:4b",
            "fallback_used": False,
        }


def _events(response):
    return [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


def test_authenticated_sse_follow_up_references_the_previous_user_turn(
    client,
    db_session,
):
    pipeline = TrackingPipeline()
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline

    registration = client.post(
        "/api/auth/register",
        json={"username": "followup-user", "password": "password"},
    ).json()
    headers = {"Authorization": f"Bearer {registration['token']}"}
    session = client.post(
        f"/api/chat/users/{registration['id']}/sessions",
        json={"title": "Follow-up test"},
        headers=headers,
    ).json()
    request_context = {
        "user_id": registration["id"],
        "session_id": session["id"],
    }

    first = client.post(
        "/api/chat/stream",
        json={"message": "Who is the chancellor of UDOM?", **request_context},
        headers=headers,
    )
    second = client.post(
        "/api/chat/stream",
        json={"message": "give the name", **request_context},
        headers=headers,
    )

    first_done = _events(first)[-1]
    second_done = _events(second)[-1]
    assert first_done["conversation"]["relation"] == "new_topic"
    assert second_done["conversation"]["relation"] == "refinement"
    assert second_done["conversation"]["is_follow_up"] is True
    assert second_done["conversation"]["topic_id"] == first_done["conversation"]["topic_id"]

    saved = client.get(
        f"/api/chat/users/{registration['id']}/sessions/{session['id']}",
        headers=headers,
    ).json()
    user_messages = [message for message in saved["messages"] if message["role"] == "user"]
    assert len(user_messages) == 2
    assert second_done["conversation"]["referenced_message_id"] == user_messages[0]["id"]
    assert user_messages[1]["turn_context"]["status"] == "completed"


def test_guest_follow_up_uses_signed_state_and_rejects_a_tampered_token(client):
    pipeline = TrackingPipeline()
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline

    first = client.post(
        "/api/chat/stream",
        json={"message": "Who is the chancellor of UDOM?"},
    )
    first_done = _events(first)[-1]
    token = first_done["conversation_token"]

    second = client.post(
        "/api/chat/stream",
        json={
            "message": "give the name",
            "conversation_token": token,
            "history": [
                {"role": "user", "content": "Who is the chancellor of UDOM?"},
                {"role": "assistant", "content": "The office is described in the regulations."},
            ],
        },
    )
    second_done = _events(second)[-1]

    assert second_done["conversation"]["is_follow_up"] is True
    assert second_done["conversation"]["topic_id"] == first_done["conversation"]["topic_id"]

    tampered = client.post(
        "/api/chat/stream",
        json={"message": "give the name", "conversation_token": f"{token}x"},
    )
    tampered_done = _events(tampered)[-1]
    assert tampered_done["conversation"]["relation"] == "ambiguous"
    assert tampered_done["conversation"]["is_follow_up"] is None
