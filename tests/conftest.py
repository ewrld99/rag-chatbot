import os
import sys
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.db.session import engine, get_db
from app.api.deps import get_rag_pipeline
from app.db.models import Base

# Setup database tables just in case
Base.metadata.create_all(bind=engine)


@pytest.fixture(scope="function")
def db_session():
    """
    Creates a fresh SQLAlchemy session for a test, wrapped in a transaction.
    After the test completes, the transaction is rolled back, ensuring
    no test data pollutes the development database.
    """
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(scope="function")
def client(db_session):
    """
    FastAPI TestClient with overridden database and mocked services.
    """
    # Override get_db dependency
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    # Clean up overrides
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
