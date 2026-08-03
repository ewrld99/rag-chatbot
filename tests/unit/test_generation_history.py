from app.services.generation_service import GenerationService
from app.services.intent_router import IntentDecision
from app.services.rag_pipeline import RAGPipeline


class _Message:
    content = "resolved standalone query"


class _Choice:
    message = _Message()


class _Response:
    choices = [_Choice()]


def _service_without_init():
    service = GenerationService.__new__(GenerationService)
    service.model = "test-model"
    return service


def test_query_rewriter_uses_last_twelve_history_messages():
    service = _service_without_init()
    captured = {}

    def fake_create_completion(_operation, **kwargs):
        captured.update(kwargs)
        return _Response()

    service._create_utility_completion = fake_create_completion
    history = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"message-{index}"}
        for index in range(14)
    ]

    assert service.rewrite_query("What about that?", history) == "resolved standalone query"

    prompt = captured["messages"][1]["content"]
    assert "'message-0'" not in prompt
    assert "'message-1'" not in prompt
    assert "'message-2'" in prompt
    assert "'message-13'" in prompt


def test_query_rewriter_corrects_domain_typos_before_fast_path():
    service = _service_without_init()

    def fail_create_completion(*_args, **_kwargs):
        raise AssertionError("English typo correction should not require LLM rewrite.")

    service._create_utility_completion = fail_create_completion

    assert (
        service.rewrite_query("how do i check regstration at udmo")
        == "how do i check registration at UDOM"
    )


def test_query_rewriter_corrects_llm_rewrite_output_typos():
    service = _service_without_init()

    class Message:
        content = "postponment procedure at udmo"

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    service._create_utility_completion = lambda *_args, **_kwargs: Response()

    assert service.rewrite_query("What about that?", [{"role": "user", "content": "postponement"}]) == (
        "postponement procedure at UDOM"
    )


def test_query_rewriter_rewrites_what_about_followup_with_history():
    service = _service_without_init()
    captured = {}

    def fake_create_completion(_operation, **kwargs):
        captured.update(kwargs)

        class Message:
            content = "fee for IDIT students"

        class Choice:
            message = Message()

        class Response:
            choices = [Choice()]

        return Response()

    service._create_utility_completion = fake_create_completion

    assert service.rewrite_query(
        "what about IDIT students",
        [{"role": "user", "content": "what is the fee for a software engineering student?"}],
    ) == "fee for IDIT students"
    assert "software engineering" in captured["messages"][1]["content"]


def test_rag_pipeline_rewrites_history_dependent_router_standalone_query():
    pipeline = object.__new__(RAGPipeline)

    class Generator:
        def is_history_dependent_query(self, query):
            return query.lower().startswith("what about")

        def rewrite_query(self, query, chat_history, user_profile=None):
            return "fee for IDIT students"

    pipeline.generator = Generator()
    decision = IntentDecision(
        intent="UDOM_DOCUMENT_SEARCH",
        confidence=0.8,
        reason="document search",
        standalone_query="what about IDIT students",
        normalized_query="what about idit students",
    )

    assert pipeline._retrieval_query(
        decision,
        "what about IDIT students",
        [{"role": "user", "content": "what is the fee for a software engineering student?"}],
    ) == "fee for IDIT students"


def test_rag_pipeline_uses_deterministic_document_follow_up_query():
    pipeline = object.__new__(RAGPipeline)

    class Generator:
        def is_history_dependent_query(self, _query):
            raise AssertionError("Resolved follow-ups must not need another LLM rewrite.")

    pipeline.generator = Generator()
    decision = IntentDecision(
        intent="UDOM_DOCUMENT_SEARCH",
        confidence=0.95,
        reason="resolved follow-up",
        standalone_query="who is the chancellor of udom; provide the person's full name",
        normalized_query="give name",
        source="rule:document_follow_up",
    )

    retrieval_query = pipeline._retrieval_query(
        decision,
        "give the name",
        [{"role": "user", "content": "who is the chancellor of udom"}],
    )

    assert retrieval_query == decision.standalone_query
    assert pipeline._generation_query(decision, "give the name", retrieval_query) == retrieval_query


def test_rag_pipeline_rewrites_swahili_document_query_for_retrieval():
    pipeline = object.__new__(RAGPipeline)

    class Generator:
        def is_history_dependent_query(self, _query):
            return False

        def rewrite_query(self, query, chat_history, user_profile=None):
            assert query == "nataka kufahamu hatua za kughairisha mwaka wa masomo"
            return "procedure to postpone a year of study"

    pipeline.generator = Generator()
    decision = IntentDecision(
        intent="UDOM_DOCUMENT_SEARCH",
        confidence=0.9,
        reason="document search",
        standalone_query="nataka kufahamu hatua za kughairisha mwaka wa masomo",
        normalized_query="nataka kufahamu hatua za kughairisha mwaka wa masomo",
        source="rule:document_search",
    )

    retrieval_query = pipeline._retrieval_query(
        decision,
        "nataka kufahamu hatua za kughairisha mwaka wa masomo",
    )

    assert retrieval_query == "procedure to postpone a year of study"
