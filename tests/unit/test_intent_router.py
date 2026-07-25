from app.services.intent_router import IntentRouter


class FailIfClassifiedRouter(IntentRouter):
    def __init__(self):
        super().__init__(db=None, generator=object())

    def _classify_with_llm(self, query, normalized_query, aliases, chat_history):
        raise AssertionError("Simple conversational turns must not call the LLM classifier.")

    def _matched_aliases(self, query):
        raise AssertionError("Simple conversational turns must not query aliases.")


class StubRouter(IntentRouter):
    def __init__(self, aliases=None, standalone_query="generic student GPA advice"):
        super().__init__(db=None, generator=object())
        self.aliases = aliases or {}
        self.standalone_query = standalone_query

    def _classify_with_llm(self, query, normalized_query, aliases, chat_history):
        return {
            "intent": "STUDENT_SUPPORT",
            "confidence": 0.9,
            "reason": "Stubbed incorrect support classification.",
            "standalone_query": self.standalone_query,
        }

    def _matched_aliases(self, query):
        return self.aliases


def test_gpa_support_classification_is_forced_to_document_search():
    router = StubRouter(
        aliases={
            "gpa": [
                "grade point average",
                "grade point",
                "grading system",
                "course weight",
                "total score",
            ]
        }
    )

    decision = router.classify("how is gpa calculated")

    assert decision.intent == "UDOM_DOCUMENT_SEARCH"
    assert decision.action == "document_search"
    assert "guardrail" in decision.source
    assert decision.standalone_query == "how is gpa calculated"


def test_general_student_support_with_udom_context_stays_support():
    router = StubRouter(standalone_query="how can I manage stress at UDOM?")

    decision = router.classify("how can I manage stress at UDOM?")

    assert decision.intent == "STUDENT_SUPPORT"
    assert decision.action == "student_support"


def test_exact_greeting_uses_rule_without_llm_or_database():
    decision = FailIfClassifiedRouter().classify("Hello!")

    assert decision.intent == "CONVERSATIONAL"
    assert decision.action == "conversational"
    assert decision.confidence == 1.0
    assert decision.source == "rule:conversational"


def test_greeting_with_official_question_routes_to_document_search():
    router = StubRouter(standalone_query="how is GPA calculated at UDOM?")

    decision = router.classify("Hello, how is GPA calculated at UDOM?")

    assert decision.intent == "UDOM_DOCUMENT_SEARCH"
    assert decision.action == "document_search"
    assert "guardrail" in decision.source
