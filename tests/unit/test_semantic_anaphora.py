from __future__ import annotations

from app.services.conversation_context import ConversationContextResolver
from app.services.intent_router import IntentRouter


def test_lost_certificate_to_id_anaphora_resolution():
    prev = "what should i do if i lost my certificate"
    follow_up = "does this apply for ID as well?"
    resolved = ConversationContextResolver._resolve_query(prev, follow_up, "refinement")

    assert "lost my ID" in resolved
    assert "; follow-up request:" not in resolved


def test_lost_certificate_to_student_id_anaphora():
    prev = "what should i do if i lost my certificate"
    follow_up = "does this apply to student ID?"
    resolved = ConversationContextResolver._resolve_query(prev, follow_up, "refinement")

    assert "lost my student ID" in resolved
    assert "; follow-up request:" not in resolved


def test_chancellor_to_vice_chancellor_anaphora():
    prev = "Who is the chancellor of UDOM?"
    follow_up = "what about vice chancellor?"
    resolved = ConversationContextResolver._resolve_query(prev, follow_up, "comparison")

    assert resolved == "Who is the vice chancellor of UDOM?"


def test_intent_router_delegates_to_semantic_anaphora_resolution():
    prev = "what should i do if i lost my certificate"
    follow_up = "does this apply for ID as well?"
    resolved = IntentRouter._resolve_follow_up_query(prev, follow_up)

    assert "lost my ID" in resolved
    assert "; follow-up request:" not in resolved
