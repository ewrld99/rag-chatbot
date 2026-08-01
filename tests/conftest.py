import os
from pathlib import Path
import re
import sys

import pytest
from dotenv import dotenv_values
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _configure_test_environment() -> str:
    env_values = dotenv_values(PROJECT_ROOT / ".env")
    source_url = (
        os.environ.get("TEST_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or env_values.get("DATABASE_URL")
    )
    if not source_url:
        raise RuntimeError("DATABASE_URL or TEST_DATABASE_URL is required for tests.")

    url = make_url(str(source_url))
    database = str(url.database or "")
    if not database.endswith("_test"):
        database = f"{database}_test"
        url = url.set(database=database)
    if not re.fullmatch(r"[A-Za-z0-9_]+_test", database):
        raise RuntimeError("Tests require a database name ending in '_test'.")

    test_url = url.render_as_string(hide_password=False)
    os.environ["DATABASE_URL"] = test_url
    os.environ["TESTING"] = "1"
    os.environ["AUTH_SECRET_KEY"] = "test-only-auth-secret-do-not-use"
    return test_url


def _ensure_test_database(test_url: str) -> None:
    url = make_url(test_url)
    database = str(url.database)
    admin_url = url.set(database="postgres")
    admin_engine = create_engine(
        admin_url,
        isolation_level="AUTOCOMMIT",
        pool_pre_ping=True,
    )
    try:
        with admin_engine.connect() as connection:
            exists = connection.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": database},
            ).scalar()
            if not exists:
                connection.exec_driver_sql(f'CREATE DATABASE "{database}"')
    except Exception as exc:
        raise RuntimeError(
            f"Unable to create or connect to isolated test database {database!r}. "
            "Set TEST_DATABASE_URL to an existing PostgreSQL test database."
        ) from exc
    finally:
        admin_engine.dispose()

    test_engine = create_engine(test_url, pool_pre_ping=True)
    try:
        with test_engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))
    finally:
        test_engine.dispose()


_TEST_DATABASE_URL = _configure_test_environment()
_ensure_test_database(_TEST_DATABASE_URL)

from app.main import app
from app.core.config import settings
from app.db.session import engine, get_db
from app.db.models import AuditLog, Base, FAQModel, SystemSetting

# Setup database tables just in case
Base.metadata.create_all(bind=engine)
with engine.begin() as connection:
    connection.execute(text(
        "ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS "
        "turn_context JSONB NOT NULL DEFAULT '{}'::jsonb"
    ))


@pytest.fixture(scope="function")
def db_session():
    """
    Create an isolated SQLAlchemy session for each test.

    The suite runs only against the dedicated database configured above.
    Mutations are wrapped in an outer transaction and rolled back per test.
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
        session.flush()

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
            self.dimension = settings.EMBEDDING_DIMENSION

        def embed(self, text: str):
            return [0.1] * self.dimension

        def embed_batch(self, texts):
            return [[0.1] * self.dimension for _ in texts]

        def safe_embed(self, text):
            return [0.1] * self.dimension

    from app.services import embedding_service
    monkeypatch.setattr(embedding_service, "EmbeddingService", MockEmbedding)
