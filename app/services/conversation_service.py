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


def static_conversational_response(query: str) -> str | None:
    """Return a quota-free response for an exact greeting or social phrase."""
    normalized = normalize_conversational_text(query)
    category = conversational_kind(query)
    if category is None:
        return None

    is_swahili = any(
        normalized in phrases
        for phrases in _SWAHILI_PHRASES.values()
    )
    if is_swahili:
        if normalized == "shikamoo":
            return "Marahaba! Ninaweza kukusaidia vipi kuhusu UDOM au masuala ya mwanafunzi?"
        return {
            "greeting": "Habari! Ninaweza kukusaidia vipi kuhusu UDOM au masuala ya mwanafunzi?",
            "thanks": "Karibu! Nipo hapa ukihitaji msaada mwingine.",
            "farewell": "Kwaheri! Nakutakia kila la heri katika masomo yako.",
            "wellbeing": "Niko vizuri, asante! Ninaweza kukusaidia nini leo?",
            "acknowledgement": "Sawa. Nipo hapa ukiwa na swali lingine.",
        }[category]

    if normalized == "good morning":
        return "Good morning! How can I help you with UDOM information or student support?"
    if normalized == "good afternoon":
        return "Good afternoon! How can I help you with UDOM information or student support?"
    if normalized == "good evening":
        return "Good evening! How can I help you with UDOM information or student support?"
    return {
        "greeting": "Hello! How can I help you with UDOM information or student support?",
        "thanks": "You're welcome! I'm here whenever you need more help.",
        "farewell": "Goodbye! All the best with your studies.",
        "wellbeing": "I'm doing well, thank you! How can I help you today?",
        "acknowledgement": "Okay. I'm here when you have another question.",
    }[category]
