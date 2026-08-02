import pytest

from app.services.intent_router import IntentRouter


class EvaluationRouter(IntentRouter):
    """Simulate a weak classifier so deterministic safety layers are exercised."""

    def __init__(self):
        super().__init__(db=None, generator=object())

    def _matched_aliases(self, _query):
        return {}

    def _classify_with_llm(self, query, _normalized_query, _aliases, _history):
        return {
            "intent": "STUDENT_SUPPORT",
            "requires_official_evidence": False,
            "confidence": 0.8,
            "reason": "Evaluation classifier defaults to support.",
            "standalone_query": query,
        }


EXPLICIT_POLICY_QUERIES = [
    "Can failing to submit an assignment cause a student to be discontinued?",
    "What are the grounds for discontinuing a student from study?",
    "Could a student be suspended for examination misconduct?",
    "Which conduct is prohibited in University residences?",
    "What penalty applies when a candidate cheats in an examination?",
    "Is an appeal permitted after examination results are released?",
    "What fee must accompany an academic appeal?",
    "Which documents are required for admission?",
    "What is the deadline for registering in a semester?",
    "Who is eligible for a supplementary examination?",
    "What consequences follow from failing all core courses?",
    "What sanctions apply to student misconduct?",
    "Can the University deregister a student for unpaid fees?",
    "When may a candidate be debarred from an examination?",
    "What regulations govern postponement of studies?",
    "According to UDOM policy, how is GPA calculated?",
    "What rules govern acceptable student dress?",
    "Are postgraduate candidates required to publish before graduation?",
    "Can an expelled student submit an appeal?",
    "What eligibility conditions apply to degree classification?",
]


UNCERTAIN_FACTUAL_QUERIES = [
    "When do lectures begin for new students?",
    "Where do students report after arriving on campus?",
    "Who approves a student's change of specialization?",
    "How long does student status remain active after finishing study?",
    "How many attempts does a student receive?",
    "How often are progress reports submitted?",
    "Does missing class affect a student's academic standing?",
    "Is attendance checked during practical sessions?",
    "Are students removed after repeated poor performance?",
    "Will a late submission affect the final result?",
    "Would changing programmes delay graduation?",
    "Could absence from a test affect progression?",
    "What happens after a student completes the final semester?",
    "Which office handles complaints from students?",
    "Whose approval is needed before leaving a programme?",
    "Was the academic calendar changed this year?",
    "Were finalists given additional completion time?",
    "Did the University extend the reporting period?",
    "Do students retain portal access after completing studies?",
    "Can a student repeat a failed assessment?",
]


GENERAL_ADVICE_QUERIES = [
    "How can I submit my assignments on time?",
    "Give me tips for managing examination stress.",
    "Can you help me improve my concentration?",
    "I need advice about balancing study and rest.",
    "Help me create a weekly revision plan.",
    "How can I motivate myself to read every day?",
    "What are some tips for avoiding procrastination?",
    "Help me organize my notes for revision.",
    "How can I study better in a noisy residence?",
    "Coach me on preparing for an oral presentation.",
    "How can I manage my time during examinations?",
    "Please help me cope with academic stress.",
    "How can I improve my essay-writing habits?",
    "Give me advice for working in a study group.",
    "Help me plan my reading around part-time work.",
    "How can I concentrate better during long lectures?",
    "What tips can help me remember what I read?",
    "Help me improve my confidence before a presentation.",
    "How can I manage my workload this semester?",
    "Give me study tips for a difficult subject.",
]


@pytest.mark.parametrize("query", EXPLICIT_POLICY_QUERIES)
def test_explicit_policy_queries_route_directly_to_documents(query):
    router = EvaluationRouter()

    decision = router.classify(query)

    assert decision.intent == "UDOM_DOCUMENT_SEARCH"
    assert router.should_probe_documents(query, decision) is False


@pytest.mark.parametrize("query", UNCERTAIN_FACTUAL_QUERIES)
def test_uncertain_factual_queries_require_retrieval_arbitration(query):
    router = EvaluationRouter()

    decision = router.classify(query)

    if decision.intent == "UDOM_DOCUMENT_SEARCH":
        assert router.should_probe_documents(query, decision) is False
    else:
        assert decision.intent == "STUDENT_SUPPORT"
        assert router.should_probe_documents(query, decision) is True


@pytest.mark.parametrize("query", GENERAL_ADVICE_QUERIES)
def test_general_advice_queries_remain_student_support(query):
    router = EvaluationRouter()

    decision = router.classify(query)

    assert decision.intent == "STUDENT_SUPPORT"
    assert router.should_probe_documents(query, decision) is False


def test_classifier_evidence_flag_overrides_support_topic_label():
    router = EvaluationRouter()
    decision = router._decision_from_payload(
        {
            "intent": "STUDENT_SUPPORT",
            "requires_official_evidence": True,
            "confidence": 0.91,
            "reason": "Institutional consequence requires evidence.",
            "standalone_query": "Can a missed assignment cause removal from study?",
        },
        "Can a missed assignment cause removal from study?",
        "missed assignment cause removal study",
        None,
    )

    assert decision.intent == "UDOM_DOCUMENT_SEARCH"

