import pytest
from app.db.models import ChatSession, ChatMessage, User

@pytest.fixture
def test_user(db_session, client):
    response = client.post(
        "/api/auth/register",
        json={"username": "chatuser", "password": "password"}
    )
    return response.json()

def test_create_and_list_chat_sessions(client, db_session, test_user):
    user_id = test_user["id"]
    
    # Create session
    response = client.post(
        f"/api/chat/users/{user_id}/sessions",
        json={"title": "My first chat"}
    )
    assert response.status_code == 201
    session_id = response.json()["id"]
    assert response.json()["title"] == "My first chat"
    
    # List sessions
    response = client.get(f"/api/chat/users/{user_id}/sessions")
    assert response.status_code == 200
    sessions = response.json()
    assert len(sessions) == 1
    assert sessions[0]["id"] == session_id

def test_submit_message_feedback(client, db_session, test_user):
    user_id = test_user["id"]
    
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
        json={"feedback": 1}
    )
    assert response.status_code == 200
    
    db_session.refresh(msg)
    assert msg.feedback == 1
    
    # Submit clear feedback
    response = client.post(
        f"/api/chat/messages/{msg.id}/feedback",
        json={"feedback": 0}
    )
    db_session.refresh(msg)
    assert msg.feedback is None
