import asyncio

from app.services.intent_router import IntentDecision
from app.services.rag_pipeline import RAGPipeline


class StubGenerator:
    def __init__(self):
        self.preference = None

    def set_model_preference(self, preference):
        self.preference = preference

    def generate_conversational(self, *_args, **_kwargs):
        raise AssertionError("Static conversational responses must not call the LLM.")

    async def stream_conversational(self, *_args, **_kwargs):
        raise AssertionError("Static conversational responses must not call the LLM.")
        yield

    def model_metadata(self):
        return {
            "requested_model": self.preference,
            "selected_model": None,
            "fallback_used": False,
        }


class StubIntentRouter:
    def classify(self, query, *_args, **_kwargs):
        return IntentDecision(
            intent="CONVERSATIONAL",
            confidence=1.0,
            reason="Exact greeting matched.",
            standalone_query=query,
            normalized_query=query.casefold(),
            source="rule:conversational",
        )


class FailRetrieval:
    def retrieve_with_scores(self, *_args, **_kwargs):
        raise AssertionError("Conversational responses must not run retrieval.")


def make_pipeline():
    pipeline = object.__new__(RAGPipeline)
    pipeline.generator = StubGenerator()
    pipeline.intent_router = StubIntentRouter()
    pipeline.retrieval_service = FailRetrieval()
    return pipeline


def test_static_greeting_skips_generation_and_retrieval():
    result = make_pipeline().run("Hello!")

    assert result["answer"].startswith("Hello!")
    assert result["sources"] == []
    assert result["context_used"] is False
    assert result["routing"]["intent"] == "CONVERSATIONAL"
    assert result["selected_model"] is None


def test_static_swahili_greeting_streams_without_generation_or_retrieval():
    async def collect_events():
        return [event async for event in make_pipeline().stream_events("Shikamoo")]

    events = asyncio.run(collect_events())

    assert events[0]["type"] == "routing"
    assert events[1]["type"] == "stream"
    assert events[1]["token"].startswith("Marahaba!")
    assert events[2]["type"] == "model"
    assert events[2]["selected_model"] is None
    assert events[3] == {"type": "sources", "sources": []}
