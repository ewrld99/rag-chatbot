from datetime import datetime, timedelta, timezone
from uuid import uuid4

from app.core.config import settings
from app.core.security import hash_password
from app.db.models import DocumentModel, RetrievalAlias, SystemSetting, User


def create_admin_headers(client, db_session, monkeypatch, username="testadmin"):
    monkeypatch.setattr(settings, "ADMIN_USERNAMES", "testadmin")
    user = db_session.query(User).filter(User.username == username).first()
    if user is None:
        user = User(username=username, password_hash=hash_password("password"))
        db_session.add(user)
        db_session.commit()

    response = client.post(
        "/api/auth/login",
        json={"username": username, "password": "password"},
    )
    assert response.status_code == 200
    token = response.json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_list_settings_requires_admin(client, db_session):
    # Ensure settings exist
    db_session.add(SystemSetting(key="chunk_size", value="800"))
    db_session.commit()
    
    response = client.get("/api/admin/settings/")
    assert response.status_code == 401


def test_list_settings_with_admin_token(client, db_session, monkeypatch):
    db_session.add(SystemSetting(key="chunk_size", value="800"))
    db_session.commit()

    response = client.get("/api/admin/settings/", headers=create_admin_headers(client, db_session, monkeypatch))
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    
    # Check if chunk_size is in the response
    chunk_setting = next((s for s in data if s["key"] == "chunk_size"), None)
    assert chunk_setting is not None
    assert chunk_setting["value"] == "800"

def test_update_setting_rejects_unsigned_frontend_token(client, db_session):
    db_session.add(SystemSetting(key="chunk_overlap", value="100"))
    db_session.commit()

    token = "header.eyJzdWIiOiIxIiwidXNlcm5hbWUiOiJ0ZXN0YWRtaW4iLCJyb2xlIjoiYWRtaW4iLCJleHAiOjk5OTk5OTk5OTl9.frontend-session"

    response = client.put(
        "/api/admin/settings/chunk_overlap",
        json={"value": "150"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401


def test_raw_query_endpoint_requires_admin(client):
    response = client.post("/api/query", json={"question": "How is GPA calculated?"})
    assert response.status_code == 401


def test_update_setting_with_admin_token(client, db_session, monkeypatch):
    db_session.add(SystemSetting(key="chunk_overlap", value="100"))
    db_session.commit()
    
    response = client.put(
        "/api/admin/settings/chunk_overlap",
        json={"value": "150"},
        headers=create_admin_headers(client, db_session, monkeypatch),
    )
    
    assert response.status_code == 200
    assert response.json()["value"] == "150"
    
    # Verify DB directly
    setting = db_session.query(SystemSetting).filter_by(key="chunk_overlap").first()
    assert setting.value == "150"


def test_import_retrieval_aliases_accepts_friendly_acronym_headers(client, db_session, monkeypatch):
    headers = create_admin_headers(client, db_session, monkeypatch)
    csv_content = (
        "Acronym,Full Form,Category\n"
        "IDIT,\"Bachelor of Science in Instructional Design & Information Technology; DM074\",programme\n"
    )

    response = client.post(
        "/api/admin/retrieval-aliases/import",
        headers=headers,
        files={"file": ("aliases.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 200
    assert response.json()["created"] + response.json()["updated"] == 1

    row = db_session.query(RetrievalAlias).filter(RetrievalAlias.term == "idit").first()
    assert row is not None
    assert "bachelor of science in instructional design & information technology" in row.aliases
    assert "dm074" in row.aliases
    assert row.category == "programme"


def test_import_retrieval_aliases_explains_supported_headers(client, db_session, monkeypatch):
    headers = create_admin_headers(client, db_session, monkeypatch)

    response = client.post(
        "/api/admin/retrieval-aliases/import",
        headers=headers,
        files={"file": ("aliases.csv", "Name,Value\nIDIT,Instructional Design\n", "text/csv")},
    )

    assert response.status_code == 400
    assert "Accepted headers" in response.json()["detail"]


def test_list_documents_orders_admin_uploads_before_crawled_pages(client, db_session, monkeypatch):
    headers = create_admin_headers(client, db_session, monkeypatch)
    marker = f"sort-{uuid4()}"
    now = datetime.now(timezone.utc)

    db_session.add_all(
        [
            DocumentModel(
                title=f"{marker}-crawled-newer",
                filename=f"{marker}-crawled-newer",
                category="web",
                source_url=f"https://example.test/{marker}/newer",
                upload_date=now,
                status="active",
            ),
            DocumentModel(
                title=f"{marker}-admin-newest",
                filename=f"{marker}-admin-newest.pdf",
                category="pdf",
                upload_date=now - timedelta(minutes=1),
                status="active",
            ),
            DocumentModel(
                title=f"{marker}-admin-older",
                filename=f"{marker}-admin-older.pdf",
                category="pdf",
                upload_date=now - timedelta(minutes=2),
                status="active",
            ),
        ]
    )
    db_session.commit()

    response = client.get(
        f"/api/admin/documents/?search={marker}&limit=10",
        headers=headers,
    )

    assert response.status_code == 200
    items = response.json()["items"]
    assert [item["title"] for item in items] == [
        f"{marker}-admin-newest",
        f"{marker}-admin-older",
        f"{marker}-crawled-newer",
    ]
    assert items[0]["source_url"] is None
    assert items[-1]["source_url"] == f"https://example.test/{marker}/newer"


def test_list_documents_uses_stable_pagination_for_tied_upload_dates(client, db_session, monkeypatch):
    headers = create_admin_headers(client, db_session, monkeypatch)
    marker = f"stable-page-{uuid4()}"
    uploaded_at = datetime.now(timezone.utc)

    docs = [
        DocumentModel(
            title=f"{marker}-{index}",
            filename=f"{marker}-{index}.txt",
            category="txt",
            upload_date=uploaded_at,
            status="active",
        )
        for index in range(12)
    ]
    db_session.add_all(docs)
    db_session.commit()

    first = client.get(
        f"/api/admin/documents/?search={marker}&page=1&limit=10",
        headers=headers,
    )
    second = client.get(
        f"/api/admin/documents/?search={marker}&page=2&limit=10",
        headers=headers,
    )

    assert first.status_code == 200
    assert second.status_code == 200
    first_ids = {item["id"] for item in first.json()["items"]}
    second_ids = {item["id"] for item in second.json()["items"]}
    assert first_ids
    assert second_ids
    assert first_ids.isdisjoint(second_ids)


def test_list_documents_rejects_page_zero_without_database_error(client, db_session, monkeypatch):
    headers = create_admin_headers(client, db_session, monkeypatch)

    response = client.get(
        "/api/admin/documents/?page=0&limit=10",
        headers=headers,
    )

    assert response.status_code == 422


def test_list_documents_accepts_dashboard_stats_limit(client, db_session, monkeypatch):
    headers = create_admin_headers(client, db_session, monkeypatch)

    response = client.get(
        "/api/admin/documents/?page=1&limit=1000",
        headers=headers,
    )

    assert response.status_code == 200


def test_upload_almanac_strategy_streams_success(client, db_session, monkeypatch, tmp_path):
    headers = create_admin_headers(client, db_session, monkeypatch)
    monkeypatch.setattr("app.api.routes.admin.UPLOAD_DIR", str(tmp_path))
    filename = f"mini-almanac-{uuid4()}.txt"

    class MockEmbedding:
        provider = "local"
        batch_size = 15

        def __init__(self, db, *args, **kwargs):
            self.db = db

        def embed_batch(self, texts):
            return [[0.1] * settings.EMBEDDING_DIMENSION for _text in texts]

    import app.services.embedding_service as embedding_module

    monkeypatch.setattr(embedding_module, "EmbeddingService", MockEmbedding)

    response = client.post(
        "/api/admin/documents/",
        headers=headers,
        data={"strategy": "almanac"},
        files={
            "file": (
                filename,
                "Monday, November 16, 2026 Start of teaching for Semester I Academic Year 2026/27",
                "text/plain",
            )
        },
    )

    assert response.status_code == 200
    assert '"progress": 100' in response.text
    assert '"chunks_stored": 1' in response.text
