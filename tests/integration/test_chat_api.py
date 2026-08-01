import json

import pytest
from starlette.websockets import WebSocketDisconnect

from app.db.models import ChatSession, ChatMessage, User
from app.api.deps import get_rag_pipeline
from app.main import app

@pytest.fixture
def test_user(db_session, client):
    response = client.post(
        "/api/auth/register",
        json={"username": "chatuser", "password": "password"}
    )
    return response.json()

def test_create_and_list_chat_sessions(client, db_session, test_user):
    user_id = test_user["id"]
    headers = {"Authorization": f"Bearer {test_user['token']}"}
    
    # Create session
    response = client.post(
        f"/api/chat/users/{user_id}/sessions",
        json={"title": "My first chat"},
        headers=headers,
    )
    assert response.status_code == 201
    session_id = response.json()["id"]
    assert response.json()["title"] == "My first chat"
    
    # List sessions
    response = client.get(f"/api/chat/users/{user_id}/sessions", headers=headers)
    assert response.status_code == 200
    sessions = response.json()
    assert len(sessions) == 1
    assert sessions[0]["id"] == session_id


def test_legacy_websocket_requires_token(client):
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/ws/chat/1"):
            pass
    assert exc_info.value.code == 1008

def test_submit_message_feedback(client, db_session, test_user):
    user_id = test_user["id"]
    headers = {"Authorization": f"Bearer {test_user['token']}"}
    
    # Create a session and message manually for feedback testing
    session = ChatSession(user_id=user_id, title="Test")
    db_session.add(session)
    db_session.commit()
    
    msg = ChatMessage(session_id=session.id, role="assistant", content="Hello")
    db_session.add(msg)
    db_session.commit()
    
    # Submit thumbs up
    response = client.post(
        f"/api/chat/messages/{msg.id}/feedback",
        json={"feedback": 1},
        headers=headers,
    )
    assert response.status_code == 200
    
    db_session.refresh(msg)
    assert msg.feedback == 1
    
    # Submit clear feedback
    response = client.post(
        f"/api/chat/messages/{msg.id}/feedback",
        json={"feedback": 0},
        headers=headers,
    )
    db_session.refresh(msg)
    assert msg.feedback is None


def test_sse_stream_replaces_provisional_answer_with_verified_answer(client):
    class FakePipeline:
        async def stream_events(self, *_args, **_kwargs):
            yield {"type": "stream", "token": "Draft answer"}
            yield {
                "type": "replace",
                "answer": "Verified answer",
                "provisional": False,
            }
            yield {"type": "sources", "sources": []}
            yield {
                "type": "model",
                "requested_model": "auto",
                "selected_model": "gemma3:4b",
                "fallback_used": False,
            }

    app.dependency_overrides[get_rag_pipeline] = lambda: FakePipeline()
    try:
        response = client.post(
            "/api/chat/stream",
            json={"message": "How do I postpone studies?"},
        )
    finally:
        app.dependency_overrides.pop(get_rag_pipeline, None)

    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert response.status_code == 200
    assert [event["type"] for event in events] == ["stream", "replace", "done"]
    assert events[1]["answer"] == "Verified answer"
