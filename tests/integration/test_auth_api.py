import pytest
from app.db.models import User
from app.core.config import settings
from app.core.security import verify_password

def test_register_user(client, db_session):
    response = client.post(
        "/api/auth/register",
        json={"username": "testuser", "password": "securepassword"}
    )
    
    assert response.status_code == 201
    data = response.json()
    assert data["username"] == "testuser"
    assert "id" in data
    assert data["role"] == "user"
    assert data["token"]
    
    # Verify in DB
    user = db_session.query(User).filter_by(username="testuser").first()
    assert user is not None
    assert verify_password("securepassword", user.password_hash)

def test_register_duplicate_username(client, db_session):
    # Register first time
    client.post(
        "/api/auth/register",
        json={"username": "testuser", "password": "securepassword"}
    )
    
    # Register second time
    response = client.post(
        "/api/auth/register",
        json={"username": "testuser", "password": "anotherpassword"}
    )
    
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]


def test_register_rejects_reserved_admin_username(client, monkeypatch):
    monkeypatch.setattr(settings, "ADMIN_USERNAMES", "reservedadmin")

    response = client.post(
        "/api/auth/register",
        json={"username": "reservedadmin", "password": "securepassword"},
    )

    assert response.status_code == 403
    assert "reserved" in response.json()["detail"]

def test_login_user(client, db_session):
    # Register first
    client.post(
        "/api/auth/register",
        json={"username": "loginuser", "password": "mypassword"}
    )
    
    # Login
    response = client.post(
        "/api/auth/login",
        json={"username": "loginuser", "password": "mypassword"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "loginuser"
    assert "id" in data
    assert data["token"]

def test_login_invalid_password(client, db_session):
    client.post(
        "/api/auth/register",
        json={"username": "loginuser", "password": "mypassword"}
    )
    
    response = client.post(
        "/api/auth/login",
        json={"username": "loginuser", "password": "wrongpassword"}
    )
    
    assert response.status_code == 401
