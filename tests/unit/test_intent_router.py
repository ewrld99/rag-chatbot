from app.services.intent_router import IntentRouter


class GreetingClassifierRouter(IntentRouter):
    def __init__(self):
        super().__init__(db=None, generator=object())
        self.classifier_calls = 0
        self.alias_calls = 0

    def _classify_with_llm(self, query, normalized_query, aliases, chat_history):
        self.classifier_calls += 1
        return {
            "intent": "CONVERSATIONAL",
            "confidence": 0.94,
            "reason": "LLM identified a greeting.",
            "standalone_query": query,
        }

    def _matched_aliases(self, query):
        self.alias_calls += 1
        return {}


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


class RecordingGenerator:
    model = "test-model"

    def __init__(self):
        self.kwargs = None

    def _create_utility_completion(self, _operation, **kwargs):
        self.kwargs = kwargs

        class Message:
            content = (
                '{"intent":"STUDENT_SUPPORT","confidence":0.99,'
                '"reason":"Student life advice",'
                '"standalone_query":"university life"}'
            )

        class Choice:
            message = Message()

        class Response:
            choices = [Choice()]

        return Response()


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
    assert decision.source == "rule:document_search"
    assert decision.standalone_query == "how is gpa calculated"


def test_general_student_support_with_udom_context_stays_support():
    router = StubRouter(standalone_query="how can I manage stress at UDOM?")

    decision = router.classify("how can I manage stress at UDOM?")

    assert decision.intent == "STUDENT_SUPPORT"
    assert decision.action == "student_support"


def test_exact_greeting_uses_llm_classifier():
    router = GreetingClassifierRouter()

    decision = router.classify("Hello!")

    assert decision.intent == "CONVERSATIONAL"
    assert decision.action == "conversational"
    assert decision.confidence == 0.94
    assert decision.source == "llm"
    assert router.classifier_calls == 1


def test_social_turn_after_document_topic_uses_llm_classifier_not_follow_up_search():
    router = GreetingClassifierRouter()

    decision = router.classify(
        "Thanks",
        [
            {
                "role": "user",
                "content": "How is GPA calculated at UDOM?",
                "turn_context": {
                    "relation": "new_topic",
                    "is_follow_up": False,
                    "intent": "UDOM_DOCUMENT_SEARCH",
                    "standalone_query": "How is GPA calculated at UDOM?",
                    "status": "completed",
                },
            }
        ],
    )

    assert decision.intent == "CONVERSATIONAL"
    assert decision.source == "llm"
    assert decision.conversation.is_follow_up is True
    assert router.classifier_calls == 1


def test_greeting_with_official_question_routes_to_document_search():
    router = StubRouter(standalone_query="how is GPA calculated at UDOM?")

    decision = router.classify("Hello, how is GPA calculated at UDOM?")

    assert decision.intent == "UDOM_DOCUMENT_SEARCH"
    assert decision.action == "document_search"
    assert decision.source == "rule:document_search"


def test_udom_dressing_code_routes_to_document_search():
    router = StubRouter(standalone_query="what is the proper udom dressing code")

    decision = router.classify("what is the proper udom dressing code")

    assert decision.intent == "UDOM_DOCUMENT_SEARCH"
    assert decision.action == "document_search"
    assert decision.source == "rule:document_search"


def test_llm_classifier_uses_native_json_mode():
    generator = RecordingGenerator()
    router = IntentRouter(db=None, generator=generator)
    router._matched_aliases = lambda _query: {}

    decision = router.classify("Tell me about university life.")

    assert decision.intent == "STUDENT_SUPPORT"
    assert generator.kwargs["response_format"] == {"type": "json_object"}
    assert generator.kwargs["max_tokens"] == 320


def test_classifier_history_text_uses_last_twelve_messages():
    router = IntentRouter(db=None, generator=None)
    history = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"message-{index}"}
        for index in range(14)
    ]

    history_text = router._history_text(history)
    history_lines = history_text.splitlines()

    assert "user: message-0" not in history_lines
    assert "assistant: message-1" not in history_lines
    assert "user: message-2" in history_lines
    assert "assistant: message-13" in history_lines


def test_official_document_terms_skip_llm_classifier():
    class FailIfClassifiedOfficialRouter(StubRouter):
        def _classify_with_llm(self, query, normalized_query, aliases, chat_history):
            raise AssertionError("Official document terms should skip LLM classification.")

    router = FailIfClassifiedOfficialRouter()

    decision = router.classify("How do I postpone a year of study?")

    assert decision.intent == "UDOM_DOCUMENT_SEARCH"
    assert decision.source == "rule:document_search"


def test_discontinued_student_policy_question_routes_to_document_search():
    class FailIfClassifiedOfficialRouter(StubRouter):
        def _classify_with_llm(self, query, normalized_query, aliases, chat_history):
            raise AssertionError("Discontinuation policy must skip LLM classification.")

    router = FailIfClassifiedOfficialRouter()

    decision = router.classify(
        "Can failing to submit an assignment cause a student to be discontinued from study?"
    )

    assert decision.intent == "UDOM_DOCUMENT_SEARCH"
    assert decision.source == "rule:document_search"
    assert decision.standalone_query == (
        "Can failing to submit an assignment cause a student to be discontinued from study?"
    )


def test_swahili_official_question_routes_to_document_search():
    class FailIfClassifiedSwahiliRouter(StubRouter):
        def _classify_with_llm(self, query, normalized_query, aliases, chat_history):
            raise AssertionError("Swahili official terms should skip LLM classification.")

    router = FailIfClassifiedSwahiliRouter()

    decision = router.classify("Nataka kujua utaratibu wa kuahirisha masomo")

    assert decision.intent == "UDOM_DOCUMENT_SEARCH"
    assert decision.source == "rule:document_search"


def test_swahili_cancellation_procedure_routes_to_document_search():
    class FailIfClassifiedSwahiliRouter(StubRouter):
        def _classify_with_llm(self, query, normalized_query, aliases, chat_history):
            raise AssertionError("Swahili postponement/cancellation terms should skip LLM classification.")

    router = FailIfClassifiedSwahiliRouter()

    decision = router.classify("nataka kufahamu hatua za kughairisha mwaka wa masomo")

    assert decision.intent == "UDOM_DOCUMENT_SEARCH"
    assert decision.source == "rule:document_search"


def test_classifier_recovers_truncated_json_prefix():
    router = IntentRouter(db=None, generator=None)

    payload = router._parse_json(
        '{\n'
        '  "intent": "UDOM_DOCUMENT_SEARCH",\n'
        '  "confidence": 0.99,\n'
        '  "reason": "User is requesting official procedures for post'
    )

    assert payload["intent"] == "UDOM_DOCUMENT_SEARCH"
    assert payload["confidence"] == 0.99
    assert payload["reason"] == "User is requesting official procedures for post"
    assert payload["standalone_query"] is None


def test_history_reference_clarify_is_forced_to_document_search():
    class ClarifyRouter(StubRouter):
        def _classify_with_llm(self, query, normalized_query, aliases, chat_history):
            return {
                "intent": "CLARIFY",
                "confidence": 0.72,
                "reason": "Reference needs context.",
                "standalone_query": None,
            }

    router = ClarifyRouter()

    decision = router.classify(
        "when was it start?",
        [
            {
                "role": "user",
                "content": (
                    "when is the end of semester two university examinations "
                    "for degree programmes?"
                ),
            },
            {
                "role": "assistant",
                "content": "Semester II examinations end on 31 July 2026.",
            },
        ],
    )

    assert decision.intent == "UDOM_DOCUMENT_SEARCH"
    assert decision.source == "rule:document_follow_up"
    assert decision.conversation.is_follow_up is True
    assert decision.standalone_query.endswith("provide the exact date")


def test_elliptical_name_follow_up_uses_last_document_question_without_llm():
    class FollowUpRouter(StubRouter):
        def _classify_with_llm(self, *_args, **_kwargs):
            raise AssertionError("A recognized follow-up must not call the LLM classifier.")

    router = FollowUpRouter()
    decision = router.classify(
        "give the name",
        [
            {"role": "user", "content": "who is the chancellor of udom"},
            {"role": "assistant", "content": "The retrieved document does not name the person."},
        ],
    )

    assert decision.intent == "UDOM_DOCUMENT_SEARCH"
    assert decision.source == "rule:document_follow_up"
    assert decision.standalone_query == (
        "who is the chancellor of udom; provide the person's full name"
    )


def test_elliptical_follow_up_without_history_requests_clarification():
    router = StubRouter()

    decision = router.classify("give the name")

    assert decision.intent == "CLARIFY"
    assert decision.source == "rule:unresolved_follow_up"


def test_follow_up_resolution_skips_an_earlier_elliptical_user_turn():
    router = StubRouter()

    decision = router.classify(
        "more details",
        [
            {"role": "user", "content": "what is the UDOM postponement procedure?"},
            {"role": "assistant", "content": "Submit an application."},
            {"role": "user", "content": "give the details"},
            {"role": "assistant", "content": "The documents contain several conditions."},
        ],
    )

    assert decision.intent == "UDOM_DOCUMENT_SEARCH"
    assert decision.standalone_query.startswith("what is the UDOM postponement procedure")


def test_ambiguous_relationship_model_result_is_reused_for_intent_routing():
    class OneCallGenerator:
        model = "auto"

        def __init__(self):
            self.operations = []

        def _create_utility_completion(self, operation, **_kwargs):
            self.operations.append(operation)

            class Message:
                content = (
                    '{"relation":"new_topic","confidence":0.93,'
                    '"intent":"UDOM_DOCUMENT_SEARCH",'
                    '"standalone_query":"UDOM graduation requirements",'
                    '"reason":"Separate official request"}'
                )

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    generator = OneCallGenerator()
    router = IntentRouter(db=None, generator=generator)
    decision = router.classify(
        "graduation requirements",
        [
            {
                "id": 1,
                "role": "user",
                "content": "Who is the chancellor of UDOM?",
                "turn_context": {
                    "relation": "new_topic",
                    "is_follow_up": False,
                    "topic_id": "topic-one",
                    "intent": "UDOM_DOCUMENT_SEARCH",
                    "standalone_query": "Who is the chancellor of UDOM?",
                    "status": "completed",
                },
            }
        ],
    )

    assert generator.operations == ["conversation_relation"]
    assert decision.intent == "UDOM_DOCUMENT_SEARCH"
    assert decision.source == "conversation:model"
    assert decision.standalone_query == "UDOM graduation requirements"
