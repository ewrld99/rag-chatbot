from langchain_core.documents import Document

from app.services.intent_router import IntentDecision
from app.services.rag_pipeline import RAGPipeline


class _ProcedureIntentRouter:
    def classify(self, query, *_args, **_kwargs):
        return IntentDecision(
            intent="UDOM_DOCUMENT_SEARCH",
            confidence=0.94,
            reason="Official procedure requires document evidence.",
            standalone_query=query,
            normalized_query=query.casefold(),
            source="rule:document_search",
        )


class _GroundingFilter:
    @staticmethod
    def filter_documents(documents, _evidence_ids):
        return documents


class _ProcedureGenerator:
    def __init__(self):
        self.called = False
        self.preference = "auto"
        self.grounding = _GroundingFilter()

    def set_model_preference(self, preference):
        self.preference = preference

    def generate_response(self, query, context, _history, **kwargs):
        self.called = True
        assert "postpone" in query
        assert context == "<procedure evidence />"
        assert kwargs["documents"]
        return {
            "answer": "1. Submit the evidence-supported postponement request.",
            "grounding": {
                "status": "grounded",
                "coverage": "full",
                "claim_count": 1,
                "supported_claim_count": 1,
                "evidence_ids": ["ev-procedure"],
                "repaired": False,
            },
            "evidence_ids": ["ev-procedure"],
            "context_used": True,
            "requested_model": self.preference,
            "selected_model": "llama3.2:3b",
            "fallback_used": False,
        }

    def model_metadata(self):
        return {
            "requested_model": self.preference,
            "selected_model": "llama3.2:3b" if self.called else None,
            "fallback_used": False,
        }


class _PolicyGenerator(_ProcedureGenerator):
    def generate_response(self, query, context, _history, **kwargs):
        self.called = True
        assert "dress code" in query
        assert context == "<policy evidence />"
        assert kwargs["documents"]
        return {
            "answer": "The proper dress code includes the evidence-supported items.",
            "grounding": {
                "status": "grounded",
                "coverage": "full",
                "claim_count": 1,
                "supported_claim_count": 1,
                "evidence_ids": ["ev-policy"],
                "repaired": False,
            },
            "evidence_ids": ["ev-policy"],
            "context_used": True,
            "requested_model": self.preference,
            "selected_model": "llama3.2:3b",
            "fallback_used": False,
        }


def test_high_confidence_procedure_uses_model_generation():
    document = Document(
        page_content="12.4 A student shall submit a postponement request through UDOM-SR2.",
        metadata={"source_type": "document", "chunk_id": "procedure-chunk"},
    )
    retrieval = {
        "documents": [document],
        "context": "retrieved procedure",
        "confidence": {
            "sufficient": True,
            "tier": "high",
            "confidence": 1.0,
        },
    }
    pipeline = object.__new__(RAGPipeline)
    pipeline.generator = _ProcedureGenerator()
    pipeline.intent_router = _ProcedureIntentRouter()
    pipeline.document_refusal = "No evidence."
    pipeline._last_scored_documents = []
    pipeline._retrieve_document_context = lambda *_args, **_kwargs: retrieval
    pipeline._generation_context_documents = lambda documents, _query: documents
    pipeline._compact_generation_context = lambda documents, _query: (
        documents,
        "<procedure evidence />",
        "procedure",
    )
    pipeline._direct_curriculum_course_answer = lambda *_args: None
    pipeline._direct_almanac_event_answer = lambda *_args: None
    pipeline._format_sources = lambda documents: [
        {"chunk_id": item.metadata["chunk_id"]} for item in documents
    ]

    result = pipeline.run("What are the procedures to postpone a year of study?")

    assert pipeline.generator.called is True
    assert result["selected_model"] == "llama3.2:3b"
    assert result["grounding"]["status"] == "grounded"
    assert result["answer"].startswith("1.")


def test_high_confidence_policy_list_uses_model_generation():
    document = Document(
        page_content=(
            "APPROPRIATE DRESSING\n"
            "i. Display a student identity card.\n"
            "ii. Wear neat formal or semi-formal clothing."
        ),
        metadata={"source_type": "document", "chunk_id": "policy-chunk"},
    )
    retrieval = {
        "documents": [document],
        "context": "retrieved policy",
        "confidence": {
            "sufficient": True,
            "tier": "high",
            "confidence": 1.0,
        },
    }
    pipeline = object.__new__(RAGPipeline)
    pipeline.generator = _PolicyGenerator()
    pipeline.intent_router = _ProcedureIntentRouter()
    pipeline.document_refusal = "No evidence."
    pipeline._last_scored_documents = []
    pipeline._retrieve_document_context = lambda *_args, **_kwargs: retrieval
    pipeline._generation_context_documents = lambda documents, _query: documents
    pipeline._compact_generation_context = lambda documents, _query: (
        documents,
        "<policy evidence />",
        "complex",
    )
    pipeline._direct_curriculum_course_answer = lambda *_args: None
    pipeline._direct_almanac_event_answer = lambda *_args: None
    pipeline._format_sources = lambda documents: [
        {"chunk_id": item.metadata["chunk_id"]} for item in documents
    ]

    result = pipeline.run("What is the proper UDOM dress code?")

    assert pipeline.generator.called is True
    assert result["selected_model"] == "llama3.2:3b"
    assert result["grounding"]["status"] == "grounded"
    assert "dress code" in result["answer"]
