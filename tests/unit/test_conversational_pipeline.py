import asyncio

from app.services.intent_router import IntentDecision
from app.services.rag_pipeline import RAGPipeline


class StubGenerator:
    def __init__(self):
        self.preference = None
        self.calls = []

    def set_model_preference(self, preference):
        self.preference = preference

    def generate_conversational(self, *_args, **_kwargs):
        self.calls.append("generate_conversational")
        return "LLM-generated greeting"

    async def stream_conversational(self, *_args, **_kwargs):
        self.calls.append("stream_conversational")
        yield "LLM-generated greeting"

    def generate_out_of_scope(self, *_args, **_kwargs):
        self.calls.append("generate_out_of_scope")
        return "LLM-generated scope boundary"

    async def stream_out_of_scope(self, *_args, **_kwargs):
        self.calls.append("stream_out_of_scope")
        yield "LLM-generated scope boundary"

    def generate_document_refusal(self, *_args, **_kwargs):
        self.calls.append("generate_document_refusal")
        return "LLM-generated evidence boundary"

    async def stream_document_refusal(self, *_args, **_kwargs):
        self.calls.append("stream_document_refusal")
        yield "LLM-generated evidence boundary"

    def model_metadata(self):
        return {
            "requested_model": self.preference,
            "selected_model": None,
            "fallback_used": False,
        }


class StubIntentRouter:
    def __init__(self, intent="CONVERSATIONAL"):
        self.intent = intent

    def classify(self, query, *_args, **_kwargs):
        return IntentDecision(
            intent=self.intent,
            confidence=1.0,
            reason="Stubbed intent.",
            standalone_query=query,
            normalized_query=query.casefold(),
            source="stub",
        )


class FailRetrieval:
    def retrieve_with_scores(self, *_args, **_kwargs):
        raise AssertionError("Conversational responses must not run retrieval.")


def make_pipeline(intent="CONVERSATIONAL"):
    pipeline = object.__new__(RAGPipeline)
    pipeline.generator = StubGenerator()
    pipeline.intent_router = StubIntentRouter(intent)
    pipeline.retrieval_service = FailRetrieval()
    pipeline._last_scored_documents = []
    return pipeline


def test_greeting_uses_llm_without_retrieval():
    result = make_pipeline().run("Hello!")

    assert result["answer"] == "LLM-generated greeting"
    assert result["sources"] == []
    assert result["context_used"] is False
    assert result["routing"]["intent"] == "CONVERSATIONAL"
    assert result["selected_model"] is None


def test_swahili_greeting_streams_from_llm_without_retrieval():
    async def collect_events():
        return [event async for event in make_pipeline().stream_events("Shikamoo")]

    events = asyncio.run(collect_events())

    assert events[0]["type"] == "routing"
    assert events[1]["type"] == "stream"
    assert events[1]["token"] == "LLM-generated greeting"
    assert events[2]["type"] == "model"
    assert events[2]["selected_model"] is None
    assert events[3] == {"type": "sources", "sources": []}


def test_out_of_scope_uses_llm_refusal_without_retrieval():
    pipeline = make_pipeline("OUT_OF_SCOPE")

    result = pipeline.run("Who won the world cup?")

    assert result["answer"] == "LLM-generated scope boundary"
    assert result["sources"] == []
    assert result["routing"]["intent"] == "OUT_OF_SCOPE"
    assert pipeline.generator.calls == ["generate_out_of_scope"]


def test_empty_document_retrieval_uses_llm_refusal():
    pipeline = make_pipeline("UDOM_DOCUMENT_SEARCH")
    pipeline._retrieve_document_context = lambda *_args, **_kwargs: {
        "documents": [],
        "context": "No relevant context found.",
        "confidence": {
            "sufficient": False,
            "reason": "No chunks were retrieved.",
        },
    }

    result = pipeline.run("Who is the dean of a missing college?")

    assert result["answer"] == "LLM-generated evidence boundary"
    assert result["sources"] == []
    assert result["context_used"] is False
    assert result["routing"]["intent"] == "UDOM_DOCUMENT_SEARCH"
    assert pipeline.generator.calls == ["generate_document_refusal"]
