import pytest
from app.db.models import User, SystemSetting
from app.core.security import hash_password
import base64
import json

def generate_mock_admin_token(username="admin", user_id=1):
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": "admin",
        "iat": 100000,
        "exp": 9999999999
    }
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip('=')
    return f"header.{encoded}.frontend-session"

def test_list_settings(client, db_session):
    # Ensure settings exist
    db_session.add(SystemSetting(key="chunk_size", value="800"))
    db_session.commit()
    
    response = client.get("/api/admin/settings/")
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    
    # Check if chunk_size is in the response
    chunk_setting = next((s for s in data if s["key"] == "chunk_size"), None)
    assert chunk_setting is not None
    assert chunk_setting["value"] == "800"

def test_update_setting_with_admin_token(client, db_session):
    db_session.add(SystemSetting(key="chunk_overlap", value="100"))
    db_session.commit()
    
    token = generate_mock_admin_token("testadmin", 1)
    
    response = client.put(
        "/api/admin/settings/chunk_overlap",
        json={"value": "150"},
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 200
    assert response.json()["value"] == "150"
    
    # Verify DB directly
    setting = db_session.query(SystemSetting).filter_by(key="chunk_overlap").first()
    assert setting.value == "150"
