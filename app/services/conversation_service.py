from __future__ import annotations

import re
from typing import Literal


ConversationKind = Literal[
    "greeting",
    "thanks",
    "farewell",
    "wellbeing",
    "acknowledgement",
]


_ENGLISH_PHRASES: dict[ConversationKind, set[str]] = {
    "greeting": {
        "hi",
        "hello",
        "hey",
        "greetings",
        "good morning",
        "good afternoon",
        "good evening",
    },
    "thanks": {
        "thanks",
        "thank you",
        "thanks a lot",
        "many thanks",
    },
    "farewell": {
        "bye",
        "goodbye",
        "see you",
        "see you later",
    },
    "wellbeing": {
        "how are you",
        "how are you doing",
        "whats up",
    },
    "acknowledgement": {
        "ok",
        "okay",
        "alright",
        "got it",
        "understood",
    },
}

_SWAHILI_PHRASES: dict[ConversationKind, set[str]] = {
    "greeting": {
        "mambo",
        "habari",
        "hujambo",
        "shikamoo",
        "habari za asubuhi",
        "habari za mchana",
        "habari za jioni",
    },
    "thanks": {
        "asante",
        "asante sana",
        "nashukuru",
    },
    "farewell": {
        "kwaheri",
        "tutaonana",
        "baadaye",
    },
    "wellbeing": {
        "hali gani",
        "unaendeleaje",
        "habari yako",
    },
    "acknowledgement": {
        "sawa",
        "nimeelewa",
        "vizuri",
    },
}


def normalize_conversational_text(query: str) -> str:
    text = query.casefold().replace("'", "")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def conversational_kind(query: str) -> ConversationKind | None:
    """Return a category only when the complete query is a simple social turn."""
    normalized = normalize_conversational_text(query)
    for category, phrases in _ENGLISH_PHRASES.items():
        if normalized in phrases:
            return category
    for category, phrases in _SWAHILI_PHRASES.items():
        if normalized in phrases:
            return category
    return None
