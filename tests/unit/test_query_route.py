from langchain_core.documents import Document

from app.api.routes.query import query
from app.schemas.query import QueryRequest


class StubRetrievalService:
    def __init__(self):
        self.top_k = None

    def set_top_k(self, top_k):
        self.top_k = top_k


class StubGrounding:
    def filter_documents(self, documents, evidence_ids):
        return list(documents) if evidence_ids else []


class StubPipeline:
    def __init__(self):
        self.retrieval_service = StubRetrievalService()
        self.generator = type("Generator", (), {"grounding": StubGrounding()})()
        self.calls = []
        document = Document(
            page_content="Grounded regulation text.",
            metadata={
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "rrf_score": 0.75,
                "dense_rank": 1,
                "sparse_rank": 2,
                "retrieval_sources": ["dense", "sparse"],
            },
        )
        self.scored_documents = [(document, 0.75)]

    def run(self, question, model_preference="auto"):
        self.calls.append((question, model_preference))
        return {
            "answer": "Pipeline answer",
            "sources": [{"document_id": "doc-1", "name": "Regulations"}],
            "grounding": {"evidence_ids": ["ev-test"]},
            "requested_model": model_preference,
            "selected_model": "gemma3:4b",
            "fallback_used": False,
        }

    def last_retrieval(self):
        return list(self.scored_documents)


def test_admin_query_delegates_to_rag_pipeline_and_preserves_diagnostics():
    pipeline = StubPipeline()

    response = query(
        QueryRequest(question="What is the regulation?", top_k=7),
        rag_pipeline=pipeline,
        _=object(),
        __=None,
    )

    assert pipeline.calls == [("What is the regulation?", "auto")]
    assert pipeline.retrieval_service.top_k == 7
    assert response.answer == "Pipeline answer"
    assert response.selected_model == "gemma3:4b"
    assert response.retrieved_chunks[0].chunk_id == "chunk-1"
    assert response.sources[0].metadata["document_id"] == "doc-1"
