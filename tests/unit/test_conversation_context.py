from __future__ import annotations

from app.services.conversation_context import (
    ConversationContextResolver,
    attach_guest_context,
    create_guest_conversation_token,
    decode_guest_conversation_token,
    generation_history,
)
from app.core.config import settings


def _document_history():
    return [
        {
            "id": 41,
            "role": "user",
            "content": "Who is the chancellor of UDOM?",
            "turn_context": {
                "version": 1,
                "relation": "new_topic",
                "is_follow_up": False,
                "confidence": 0.99,
                "topic_id": "topic-chancellor",
                "referenced_message_id": None,
                "intent": "UDOM_DOCUMENT_SEARCH",
                "standalone_query": "Who is the chancellor of UDOM?",
                "source": "rule",
                "reason": "New topic.",
                "status": "completed",
            },
        },
        {
            "id": 42,
            "role": "assistant",
            "content": "The documents describe the office but do not give a name.",
            "turn_context": {
                "topic_id": "topic-chancellor",
                "intent": "UDOM_DOCUMENT_SEARCH",
                "standalone_query": "Who is the chancellor of UDOM?",
                "relation": "new_topic",
                "is_follow_up": False,
                "confidence": 0.99,
                "source": "rule",
                "reason": "New topic.",
                "status": "completed",
            },
        },
    ]


def test_obvious_follow_up_uses_structured_topic_without_model():
    class FailGenerator:
        def _create_utility_completion(self, *_args, **_kwargs):
            raise AssertionError("An obvious follow-up must not call a model.")

    decision = ConversationContextResolver(FailGenerator()).resolve(
        "give the name",
        _document_history(),
    )

    assert decision.relation == "refinement"
    assert decision.is_follow_up is True
    assert decision.topic_id == "topic-chancellor"
    assert decision.referenced_message_id == 41
    assert decision.intent == "UDOM_DOCUMENT_SEARCH"
    assert decision.standalone_query.endswith("provide the person's full name")

    assert ConversationContextResolver(FailGenerator()).resolve(
        "tell me more",
        _document_history(),
    ).is_follow_up is True


def test_complete_question_starts_a_new_topic():
    decision = ConversationContextResolver().resolve(
        "How can I improve my study schedule?",
        _document_history(),
    )

    assert decision.relation == "new_topic"
    assert decision.is_follow_up is False
    assert decision.topic_id != "topic-chancellor"
    assert decision.referenced_message_id is None


def test_correction_and_comparison_keep_the_active_topic():
    resolver = ConversationContextResolver()

    correction = resolver.resolve("I meant semester two", _document_history())
    comparison = resolver.resolve("What about postgraduate students?", _document_history())

    assert correction.relation == "correction"
    assert correction.topic_id == "topic-chancellor"
    assert comparison.relation == "comparison"
    assert comparison.topic_id == "topic-chancellor"


def test_follow_up_chain_uses_the_latest_canonical_query():
    history = _document_history()
    history.append(
        {
            "id": 43,
            "role": "user",
            "content": "give the name",
            "turn_context": {
                **ConversationContextResolver().resolve("give the name", history).to_dict(),
                "status": "completed",
            },
        }
    )

    decision = ConversationContextResolver().resolve("more details", history)

    assert decision.is_follow_up is True
    assert decision.topic_id == "topic-chancellor"
    assert decision.referenced_message_id == 43
    assert "provide the person's full name" in decision.standalone_query


def test_reference_without_a_topic_is_explicitly_ambiguous():
    decision = ConversationContextResolver().resolve("give the name", [])

    assert decision.relation == "ambiguous"
    assert decision.is_follow_up is None
    assert decision.standalone_query is None


def test_short_answer_resolves_a_previous_clarification():
    history = _document_history()
    history[-1]["turn_context"]["status"] = "clarification_requested"

    decision = ConversationContextResolver().resolve("the second one", history)

    assert decision.relation == "clarification_response"
    assert decision.is_follow_up is True
    assert decision.topic_id == "topic-chancellor"


def test_social_turn_does_not_replace_the_active_topic():
    decision = ConversationContextResolver().resolve("thanks", _document_history())

    assert decision.relation == "continuation"
    assert decision.topic_id == "topic-chancellor"
    assert decision.standalone_query == "Who is the chancellor of UDOM?"


def test_guest_conversation_token_round_trip_and_tamper_rejection():
    context = ConversationContextResolver().resolve("Who is the chancellor of UDOM?", []).to_dict()
    token = create_guest_conversation_token(context)

    restored = decode_guest_conversation_token(token)
    assert restored["topic_id"] == context["topic_id"]
    assert restored["standalone_query"] == context["standalone_query"]
    assert decode_guest_conversation_token(f"{token}x") is None


def test_guest_context_is_hidden_from_generation_history():
    context = ConversationContextResolver().resolve("Who is the chancellor of UDOM?", []).to_dict()
    history = attach_guest_context(
        [{"role": "user", "content": "Who is the chancellor of UDOM?"}],
        context,
    )

    assert history[-1]["role"] == "context"
    assert generation_history(history) == [
        {"role": "user", "content": "Who is the chancellor of UDOM?"}
    ]


def test_failed_turns_are_not_sent_to_generation_or_used_as_topic_anchors():
    history = _document_history()
    history.append(
        {
            "id": 99,
            "role": "user",
            "content": "An unrelated failed request",
            "turn_context": {
                "relation": "new_topic",
                "is_follow_up": False,
                "topic_id": "failed-topic",
                "intent": "UDOM_DOCUMENT_SEARCH",
                "standalone_query": "An unrelated failed request",
                "status": "failed",
            },
        }
    )

    decision = ConversationContextResolver().resolve("more details", history)

    assert decision.topic_id == "topic-chancellor"
    assert all(item["content"] != "An unrelated failed request" for item in generation_history(history))


def test_ambiguous_short_query_uses_one_structured_utility_call():
    class ModelResponse:
        calls = 0

        def _create_utility_completion(self, operation, **kwargs):
            self.calls += 1
            assert operation == "conversation_relation"
            assert kwargs["response_format"] == {"type": "json_object"}

            class Message:
                content = (
                    '{"relation":"new_topic","confidence":0.91,'
                    '"intent":"UDOM_DOCUMENT_SEARCH",'
                    '"standalone_query":"UDOM graduation requirements",'
                    '"reason":"Complete separate request"}'
                )

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    generator = ModelResponse()
    decision = ConversationContextResolver(generator).resolve(
        "graduation requirements",
        _document_history(),
    )

    assert generator.calls == 1
    assert decision.relation == "new_topic"
    assert decision.source == "model"
    assert decision.intent == "UDOM_DOCUMENT_SEARCH"


def test_low_confidence_or_malformed_model_output_fails_closed():
    class LowConfidence:
        def _create_utility_completion(self, *_args, **_kwargs):
            class Message:
                content = (
                    '{"relation":"continuation","confidence":0.4,'
                    '"intent":"UDOM_DOCUMENT_SEARCH","standalone_query":null}'
                )

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    class Malformed:
        def _create_utility_completion(self, *_args, **_kwargs):
            class Message:
                content = "not-json"

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    assert ConversationContextResolver(LowConfidence()).resolve(
        "graduation requirements",
        _document_history(),
    ).relation == "ambiguous"
    assert ConversationContextResolver(Malformed()).resolve(
        "fees",
        _document_history(),
    ).relation == "ambiguous"


def test_expired_guest_token_is_rejected(monkeypatch):
    context = ConversationContextResolver().resolve("Who is the chancellor of UDOM?", []).to_dict()
    monkeypatch.setattr(settings, "CONVERSATION_TOKEN_TTL_SECONDS", -1)

    token = create_guest_conversation_token(context)

    assert decode_guest_conversation_token(token) is None
