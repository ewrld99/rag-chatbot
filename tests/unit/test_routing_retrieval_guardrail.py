from app.services.intent_router import IntentDecision
from app.services.rag_pipeline import RAGPipeline


def _support_decision() -> IntentDecision:
    return IntentDecision(
        intent="STUDENT_SUPPORT",
        confidence=0.8,
        reason="Classifier selected general support.",
        standalone_query="Does missing class affect academic standing?",
        normalized_query="missing class affect academic standing",
        source="llm",
    )


class GuardrailRouter:
    def should_probe_documents(self, _query, decision):
        return decision.intent == "STUDENT_SUPPORT"

    def promote_to_document_search(self, decision, _query, confidence):
        return IntentDecision(
            intent="UDOM_DOCUMENT_SEARCH",
            confidence=max(decision.confidence, confidence),
            reason="Strong retrieval evidence.",
            standalone_query=decision.standalone_query,
            normalized_query=decision.normalized_query,
            source="llm+retrieval_guardrail",
        )


def _pipeline_with_confidence(confidence):
    pipeline = object.__new__(RAGPipeline)
    pipeline.intent_router = GuardrailRouter()
    pipeline._last_scored_documents = [("sentinel", 1.0)]
    pipeline._retrieval_query = lambda *_args, **_kwargs: "focused policy question"
    retrieval = {
        "documents": ["official chunk"],
        "context": "official evidence",
        "confidence": confidence,
    }
    pipeline._retrieve_document_context = lambda *_args, **_kwargs: retrieval
    return pipeline, retrieval


def test_strong_hybrid_retrieval_promotes_support_route():
    pipeline, retrieval = _pipeline_with_confidence(
        {
            "sufficient": True,
            "tier": "high",
            "confidence": 0.94,
            "lexical_overlap": 4,
            "matched_sources": ["dense", "sparse"],
            "best_sparse_rank": 1,
        }
    )

    decision, retrieval_query, resolved_retrieval = (
        pipeline._arbitrate_student_support_route(
            _support_decision(),
            "Does missing class affect academic standing?",
            [],
            None,
        )
    )

    assert decision.intent == "UDOM_DOCUMENT_SEARCH"
    assert decision.source.endswith("retrieval_guardrail")
    assert retrieval_query == "focused policy question"
    assert resolved_retrieval is retrieval


def test_weak_retrieval_keeps_support_route():
    pipeline, _retrieval = _pipeline_with_confidence(
        {
            "sufficient": True,
            "tier": "medium",
            "confidence": 0.55,
            "lexical_overlap": 1,
            "matched_sources": ["dense"],
            "best_sparse_rank": None,
        }
    )

    decision, retrieval_query, resolved_retrieval = (
        pipeline._arbitrate_student_support_route(
            _support_decision(),
            "Does missing class affect academic standing?",
            [],
            None,
        )
    )

    assert decision.intent == "STUDENT_SUPPORT"
    assert retrieval_query is None
    assert resolved_retrieval is None
    assert pipeline._last_scored_documents == []
