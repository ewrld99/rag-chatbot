from app.services.conversation_service import (
    conversational_kind,
)


def test_english_greeting_is_classified_as_conversational():
    assert conversational_kind("Hello!") == "greeting"


def test_swahili_greeting_is_classified_as_conversational():
    assert conversational_kind("Shikamoo") == "greeting"


def test_thanks_and_acknowledgements_are_conversational():
    assert conversational_kind("Asante sana") == "thanks"
    assert conversational_kind("Okay.") == "acknowledgement"


def test_greeting_plus_question_is_not_treated_as_simple_conversation():
    query = "Hello, how is GPA calculated at UDOM?"

    assert conversational_kind(query) is None
