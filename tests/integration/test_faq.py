import pytest
from fastapi.testclient import TestClient
import io
import pandas as pd
from unittest.mock import patch, MagicMock

from app.main import app
from app.db.models import FAQModel, AuditLog
from app.services.faq_service import FAQService
from app.services.hybrid_retriever import HybridRetriever

client = TestClient(app)

# ---------------------------------------------------------
# Fixtures & Mocks
# ---------------------------------------------------------

@pytest.fixture
def mock_db_session():
    return MagicMock()

@pytest.fixture
def mock_admin_token():
    # Helper to generate a mock token for admin
    return "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6ImFkbWluIn0.signature"

# ---------------------------------------------------------
# API Endpoints & Validation Tests
# ---------------------------------------------------------

def test_create_faq_unauthorized():
    response = client.post("/api/admin/faqs/", json={"question": "Q", "answer": "A"})
    assert response.status_code == 401
    assert "Unauthorized" in response.json()["detail"]

@patch("app.api.routes.admin.FAQService.create_faq")
def test_create_faq_authorized(mock_create, mock_admin_token):
    mock_create.return_value = FAQModel(id="123", question="Q", answer="A", category="Test")
    response = client.post(
        "/api/admin/faqs/", 
        json={"question": "Q", "answer": "A", "category": "Test"},
        headers={"Authorization": mock_admin_token}
    )
    assert response.status_code == 200
    assert response.json()["question"] == "Q"

def test_faq_validation_limits(mock_admin_token):
    # Test max length of 500 for question
    long_question = "Q" * 501
    response = client.post(
        "/api/admin/faqs/", 
        json={"question": long_question, "answer": "A"},
        headers={"Authorization": mock_admin_token}
    )
    assert response.status_code == 422 # Pydantic validation error

    # Test max length of 10000 for answer
    long_answer = "A" * 10001
    response = client.post(
        "/api/admin/faqs/", 
        json={"question": "Q", "answer": long_answer},
        headers={"Authorization": mock_admin_token}
    )
    assert response.status_code == 422

# ---------------------------------------------------------
# FAQService Unit Tests
# ---------------------------------------------------------

@patch("app.services.faq_service.get_embedding")
@patch("app.services.faq_service.log_audit")
def test_faq_service_crud(mock_log, mock_get_embedding, mock_db_session):
    mock_get_embedding.return_value = [0.1] * 1024
    svc = FAQService(mock_db_session)
    
    # Test Create
    faq = svc.create_faq("How do I apply?", "Visit the portal.", category="Admissions", admin_username="admin")
    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called()
    mock_log.assert_called_with(mock_db_session, "FAQ_CREATE", "admin", new_value="How do I apply?")
    
    # Test Update
    mock_db_session.query().filter().first.return_value = faq
    updated = svc.update_faq("123", question="New Q", admin_username="admin")
    assert updated.question == "New Q"
    mock_log.assert_called_with(mock_db_session, "FAQ_UPDATE", "admin", old_value="How do I apply?", new_value="New Q")

    # Test Delete
    svc.delete_faq("123", admin_username="admin")
    assert faq.is_active is False
    mock_log.assert_called_with(mock_db_session, "FAQ_DELETE", "admin", old_value="New Q")

@patch("app.services.faq_service.get_embeddings")
@patch("app.services.faq_service.log_audit")
def test_bulk_import_batched(mock_log, mock_get_embeddings, mock_db_session):
    svc = FAQService(mock_db_session)
    mock_get_embeddings.return_value = [[0.1]*1024, [0.2]*1024]
    
    df = pd.DataFrame({
        "Question": ["Q1", "Q2", ""],
        "Answer": ["A1", "A2", "A3"],
        "Category": ["C1", "C2", "C3"]
    })
    
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    
    stats = svc.bulk_import(csv_buffer.getvalue(), "test.csv", admin_username="admin")
    
    assert stats["imported"] == 2
    assert stats["skipped"] == 1 # The empty question
    mock_db_session.add_all.assert_called_once()
    mock_log.assert_called_with(mock_db_session, "FAQ_IMPORT", "admin", new_value="Imported 2 FAQs from test.csv")

# ---------------------------------------------------------
# Hybrid Retrieval Integration
# ---------------------------------------------------------

@patch("app.services.dense_retriever.DenseRetriever.retrieve")
@patch("app.services.sparse_retriever.SparseRetriever.retrieve")
def test_hybrid_retrieval_rrf(mock_sparse, mock_dense, mock_db_session):
    # Mocking that Dense returns an FAQ high up, and Sparse returns a Document
    from app.services.dense_retriever import DenseResult
    from app.services.sparse_retriever import SparseResult
    
    mock_dense.return_value = [
        DenseResult(chunk_id="faq1", document_id="faq1", similarity_score=0.9, text="FAQ Answer", metadata={"source_type": "faq"}),
        DenseResult(chunk_id="doc1", document_id="doc1", similarity_score=0.8, text="Doc Answer", metadata={"source_type": "document"})
    ]
    
    mock_sparse.return_value = [
        SparseResult(chunk_id="doc1", document_id="doc1", fts_score=1.5, text="Doc Answer", metadata={"source_type": "document"}),
        SparseResult(chunk_id="faq1", document_id="faq1", fts_score=0.5, text="FAQ Answer", metadata={"source_type": "faq"})
    ]
    
    retriever = HybridRetriever(mock_db_session)
    results = retriever.retrieve("test query", top_k=2)
    
    # RRF = 1/(k+rank). k=60
    # doc1: dense rank 2 (1/62), sparse rank 1 (1/61) = 0.0325
    # faq1: dense rank 1 (1/61), sparse rank 2 (1/62) = 0.0325
    # Since they tie, the order may vary, but both should be in results
    assert len(results) == 2
    assert any(r.metadata["source_type"] == "faq" for r in results)
    assert any(r.metadata["source_type"] == "document" for r in results)
