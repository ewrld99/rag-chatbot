import os
import sys

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.db.session import engine, get_db
from app.db.models import AuditLog, Base, FAQModel, SystemSetting

# Setup database tables just in case
Base.metadata.create_all(bind=engine)


@pytest.fixture(scope="function")
def db_session():
    """
    Create an isolated SQLAlchemy session for each test.

    The suite may run against a development database that already contains
    seeded settings/FAQs. We clear only the small tables these tests mutate
    inside an outer transaction, then roll everything back after the test.
    """
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(
        bind=connection,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )

    try:
        session.query(AuditLog).delete(synchronize_session=False)
        session.query(FAQModel).delete(synchronize_session=False)
        session.query(SystemSetting).delete(synchronize_session=False)
        session.commit()

        yield session
    finally:
        session.close()
        if transaction.is_active:
            transaction.rollback()
        connection.close()


@pytest.fixture(scope="function")
def client(db_session):
    """
    FastAPI TestClient with overridden database and mocked services.

    Each request gets its own Session object while sharing the same outer
    transaction as the test body. That avoids cross-thread reuse of the
    test Session while keeping request data visible and rollbackable.
    """
    connection = db_session.get_bind()

    def override_get_db():
        request_session = Session(
            bind=connection,
            expire_on_commit=False,
            join_transaction_mode="create_savepoint",
        )
        try:
            yield request_session
        finally:
            request_session.close()

    app.dependency_overrides[get_db] = override_get_db

    test_client = TestClient(app)
    try:
        yield test_client
    finally:
        test_client.close()
        app.dependency_overrides.clear()


@pytest.fixture
def mock_embedding_service(monkeypatch):
    """
    Mocks the EmbeddingService to return deterministic dummy vectors
    so tests don't hit OpenAI/Jina APIs.
    """
    class MockEmbedding:
        def __init__(self, db, *args, **kwargs):
            self.db = db
            self.dimension = 768

        def embed(self, text: str):
            return [0.1] * self.dimension

        def embed_batch(self, texts):
            return [[0.1] * self.dimension for _ in texts]

        def safe_embed(self, text):
            return [0.1] * self.dimension

    from app.services import embedding_service
    monkeypatch.setattr(embedding_service, "EmbeddingService", MockEmbedding)
