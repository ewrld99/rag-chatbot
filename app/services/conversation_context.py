"""Session-aware conversation relationship resolution."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import logging
import re
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid4, uuid5

from app.core.config import settings
from app.core.security import create_access_token, decode_access_token
from app.services.conversation_service import conversational_kind
from app.services.query_normalization import clean_query_text, tokenize


logger = logging.getLogger(__name__)

CONVERSATION_RESOLVER_VERSION = 1
CONVERSATION_TOKEN_USE = "guest_conversation"

ConversationRelation = Literal[
    "new_topic",
    "continuation",
    "refinement",
    "correction",
    "comparison",
    "clarification_response",
    "ambiguous",
]

FOLLOW_UP_RELATIONS = {
    "continuation",
    "refinement",
    "correction",
    "comparison",
    "clarification_response",
}

SUPPORTED_INTENTS = {
    "CONVERSATIONAL",
    "STUDENT_SUPPORT",
    "UDOM_DOCUMENT_SEARCH",
    "CLARIFY",
    "OUT_OF_SCOPE",
}

RELATION_PROMPT = """You resolve how the latest user message relates to the active
conversation topic for a University of Dodoma student chatbot.

Return exactly one JSON object with:
{
  "relation": "new_topic | continuation | refinement | correction | comparison | clarification_response | ambiguous",
  "confidence": 0.0,
  "intent": "CONVERSATIONAL | STUDENT_SUPPORT | UDOM_DOCUMENT_SEARCH | CLARIFY | OUT_OF_SCOPE",
  "standalone_query": "A complete query which can be understood without history, or null",
  "reason": "Brief reason"
}

Use follow-up relations only when the latest message depends on the active topic.
Use new_topic for a complete unrelated request. Use ambiguous when choosing would
require guessing. Preserve names, dates, programmes, and other constraints from
both the active topic and latest message. Return JSON only."""


@dataclass(frozen=True)
class ConversationDecision:
    relation: ConversationRelation
    is_follow_up: bool | None
    confidence: float
    topic_id: str
    referenced_message_id: int | None
    intent: str | None
    standalone_query: str | None
    source: str
    reason: str
    version: int = CONVERSATION_RESOLVER_VERSION

    def with_routing(self, intent: str, standalone_query: str | None) -> "ConversationDecision":
        preserve_topic = intent == "CONVERSATIONAL" and self.is_follow_up is True and self.intent
        resolved_intent = (
            self.intent
            if (intent == "CLARIFY" and self.intent) or preserve_topic
            else intent
        )
        resolved_query = self.standalone_query if preserve_topic else standalone_query or self.standalone_query
        return replace(
            self,
            intent=resolved_intent,
            standalone_query=resolved_query,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "relation": self.relation,
            "is_follow_up": self.is_follow_up,
            "confidence": round(self.confidence, 4),
            "topic_id": self.topic_id,
            "referenced_message_id": self.referenced_message_id,
            "intent": self.intent,
            "standalone_query": self.standalone_query,
            "source": self.source,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class _TopicAnchor:
    topic_id: str
    message_id: int | None
    intent: str | None
    standalone_query: str


class ConversationContextResolver:
    """Resolve topic relationships with rules first and one model call at most."""

    _ELLIPTICAL_RE = re.compile(
        r"^\s*(?:"
        r"(?:please\s+)?(?:give|tell|show|state|provide|mention)\s+(?:me\s+)?"
        r"(?:the\s+)?(?:name|names|date|dates|number|numbers|title|titles|details?|"
        r"answer|source|sources)(?:\s+(?:only|instead|exactly))?|"
        r"(?:more|further)\s+(?:details?|information)|"
        r"(?:tell|show|give)(?:\s+me)?\s+more|"
        r"(?:can\s+you\s+)?(?:explain|elaborate)(?:\s+(?:it|that))?"
        r"(?:\s+(?:more|further|on\s+(?:it|that)))?|"
        r"(?:what|who|when|where|which)\s+(?:exactly|then|next|else)|"
        r"(?:why|how|where|when|who|what)|"
        r"(?:what|who)\s+(?:is|are)\s+(?:the\s+|his\s+|her\s+|their\s+)?"
        r"(?:name|date|number|title)|"
        r"(?:nipe\s+jina|jina\s+lake|eleza\s+zaidi|endelea|kwa\s+nini|vipi|lini|wapi|nani)"
        r")\s*[?.!]*\s*$",
        re.IGNORECASE,
    )
    _CORRECTION_RE = re.compile(
        r"^\s*(?:i\s+(?:mean|meant)|sorry[, ]+i\s+(?:mean|meant)|not\s+.+(?:but|,)|"
        r"actually[, ]|rather[, ])",
        re.IGNORECASE,
    )
    _COMPARISON_RE = re.compile(
        r"^\s*(?:what|how)\s+about\b|^\s*(?:and\s+)?(?:for|among)\s+\w+|\bcompared\s+(?:with|to)\b",
        re.IGNORECASE,
    )
    _CONTINUATION_RE = re.compile(
        r"^\s*(?:also|and\s+then|then|next|continue|go\s+on)\b",
        re.IGNORECASE,
    )
    _SHORT_RESPONSE_RE = re.compile(
        r"^\s*(?:yes|no|yeah|nope|correct|the\s+first|the\s+second|first|second|"
        r"both|all|none|that\s+one|this\s+one)\s*[?.!]*\s*$",
        re.IGNORECASE,
    )
    _REFERENCE_TERMS = {
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "they",
        "them",
        "there",
        "same",
        "previous",
        "above",
        "earlier",
        "former",
        "latter",
        "one",
        "ones",
    }
    _SELF_CONTAINED_TIME_REFERENCE_RE = re.compile(
        r"\b(?:this|current|next|last)\s+(?:academic\s+)?"
        r"(?:year|semester|term|month|week|day)\b",
        re.IGNORECASE,
    )

    def __init__(self, generator: Any | None = None) -> None:
        self.generator = generator

    def resolve(
        self,
        query: str,
        chat_history: list[dict[str, Any]] | None = None,
    ) -> ConversationDecision:
        cleaned = clean_query_text(query)
        anchor = self._active_anchor(chat_history)

        if not cleaned:
            return self._decision(
                "ambiguous", None, 1.0, anchor, None, "rule", "Empty query has no resolvable relationship."
            )

        if anchor is None:
            if self._depends_on_history(cleaned):
                return self._decision(
                    "ambiguous",
                    None,
                    0.99,
                    None,
                    None,
                    "rule",
                    "Reference has no topic in the current chat.",
                )
            return self._new_topic(
                cleaned,
                confidence=0.99,
                source="rule",
                intent="CONVERSATIONAL" if conversational_kind(cleaned) else None,
            )

        if conversational_kind(cleaned) is not None:
            return self._follow_up(
                "continuation",
                cleaned,
                anchor,
                1.0,
                "rule",
                "Social turn acknowledges the active topic without replacing it.",
                intent=anchor.intent,
                standalone_query=anchor.standalone_query,
            )

        if self._awaiting_clarification(chat_history) and self._looks_like_clarification_response(cleaned):
            return self._follow_up(
                "clarification_response",
                cleaned,
                anchor,
                0.98,
                "rule",
                "Short answer resolves the previous clarification.",
            )

        if self._CORRECTION_RE.search(cleaned):
            return self._follow_up(
                "correction", cleaned, anchor, 0.98, "rule", "Correction modifies the active topic."
            )
        if self._COMPARISON_RE.search(cleaned):
            rewritten = self._rewrite_comparison_query(cleaned, chat_history)
            if rewritten is None:
                return self._decision(
                    "ambiguous",
                    None,
                    0.70,
                    anchor,
                    None,
                    "fallback",
                    "The comparison target could not be rewritten safely as a standalone query.",
                    intent=anchor.intent,
                )
            return self._follow_up(
                "comparison",
                cleaned,
                anchor,
                0.96,
                "model",
                "Comparison was rewritten as a standalone query using the active topic.",
                standalone_query=rewritten,
            )
        if self._ELLIPTICAL_RE.match(cleaned) or self._has_reference(cleaned):
            return self._follow_up(
                "refinement", cleaned, anchor, 0.97, "rule", "Reference refines the active topic."
            )
        if self._CONTINUATION_RE.search(cleaned):
            return self._follow_up(
                "continuation", cleaned, anchor, 0.94, "rule", "Continuation marker extends the active topic."
            )
        if self._looks_standalone(cleaned):
            return self._new_topic(cleaned, confidence=0.94, source="rule")

        return self._resolve_with_model(cleaned, anchor, chat_history)

    def _resolve_with_model(
        self,
        query: str,
        anchor: _TopicAnchor,
        chat_history: list[dict[str, Any]] | None,
    ) -> ConversationDecision:
        try:
            if self.generator is None or not hasattr(self.generator, "_create_utility_completion"):
                raise RuntimeError("Conversation utility model is unavailable")
            response = self.generator._create_utility_completion(
                "conversation_relation",
                model=getattr(self.generator, "model", "auto"),
                messages=[
                    {"role": "system", "content": RELATION_PROMPT},
                    {"role": "user", "content": self._model_input(query, anchor, chat_history)},
                ],
                temperature=0.0,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            payload = self._parse_model_payload(response.choices[0].message.content)
            relation = str(payload.get("relation") or "ambiguous").strip().lower()
            if relation not in {*FOLLOW_UP_RELATIONS, "new_topic", "ambiguous"}:
                relation = "ambiguous"
            confidence = self._confidence(payload.get("confidence"))
            if confidence < 0.80:
                relation = "ambiguous"

            intent = str(payload.get("intent") or "").strip().upper()
            if intent not in SUPPORTED_INTENTS:
                intent = anchor.intent or ""
            standalone_query = self._clean_optional(payload.get("standalone_query"))
            reason = clean_query_text(str(payload.get("reason") or "Model relationship decision."))[:300]

            if relation == "new_topic":
                decision = self._new_topic(
                    standalone_query or query,
                    confidence=confidence,
                    source="model",
                    intent=intent or None,
                    reason=reason,
                )
            elif relation in FOLLOW_UP_RELATIONS:
                decision = self._follow_up(
                    relation,
                    query,
                    anchor,
                    confidence,
                    "model",
                    reason,
                    intent=intent or anchor.intent,
                    standalone_query=standalone_query,
                )
            else:
                decision = self._decision(
                    "ambiguous", None, confidence, anchor, None, "model", reason, intent=intent or None
                )
        except Exception as exc:
            logger.warning("Conversation relationship model failed; using safe fallback: %s", exc)
            if self._depends_on_history(query) or len(tokenize(query)) <= 3:
                decision = self._decision(
                    "ambiguous",
                    None,
                    0.55,
                    anchor,
                    None,
                    "fallback",
                    "Relationship is unclear without a reliable classifier.",
                )
            else:
                decision = self._new_topic(query, confidence=0.70, source="fallback")

        return decision

    def _rewrite_comparison_query(
        self,
        query: str,
        chat_history: list[dict[str, Any]] | None,
    ) -> str | None:
        rewrite = getattr(self.generator, "rewrite_query", None)
        if not callable(rewrite):
            return None

        try:
            rewritten = clean_query_text(str(rewrite(query, chat_history) or ""))
        except Exception as exc:
            logger.warning("Comparison query rewrite failed: %s", exc)
            return None

        if not rewritten or rewritten.casefold() == clean_query_text(query).casefold():
            return None
        return rewritten[:600]

    def _new_topic(
        self,
        query: str,
        *,
        confidence: float,
        source: str,
        intent: str | None = None,
        reason: str = "Complete request starts a new topic.",
    ) -> ConversationDecision:
        decision = ConversationDecision(
            relation="new_topic",
            is_follow_up=False,
            confidence=confidence,
            topic_id=str(uuid4()),
            referenced_message_id=None,
            intent=intent,
            standalone_query=clean_query_text(query)[:600],
            source=source,
            reason=reason,
        )
        self._log(decision)
        return decision

    def _follow_up(
        self,
        relation: ConversationRelation,
        query: str,
        anchor: _TopicAnchor,
        confidence: float,
        source: str,
        reason: str,
        *,
        intent: str | None = None,
        standalone_query: str | None = None,
    ) -> ConversationDecision:
        resolved = standalone_query or self._resolve_query(anchor.standalone_query, query, relation)
        decision = ConversationDecision(
            relation=relation,
            is_follow_up=True,
            confidence=confidence,
            topic_id=anchor.topic_id,
            referenced_message_id=anchor.message_id,
            intent=intent or anchor.intent,
            standalone_query=resolved[:600],
            source=source,
            reason=reason,
        )
        self._log(decision)
        return decision

    def _decision(
        self,
        relation: ConversationRelation,
        is_follow_up: bool | None,
        confidence: float,
        anchor: _TopicAnchor | None,
        standalone_query: str | None,
        source: str,
        reason: str,
        *,
        intent: str | None = None,
    ) -> ConversationDecision:
        decision = ConversationDecision(
            relation=relation,
            is_follow_up=is_follow_up,
            confidence=confidence,
            topic_id=anchor.topic_id if anchor else str(uuid4()),
            referenced_message_id=anchor.message_id if anchor else None,
            intent=intent or (anchor.intent if anchor else None),
            standalone_query=standalone_query,
            source=source,
            reason=reason,
        )
        self._log(decision)
        return decision

    def _active_anchor(self, chat_history: list[dict[str, Any]] | None) -> _TopicAnchor | None:
        for item in reversed(chat_history or []):
            if item.get("role") not in {"user", "context"}:
                continue
            context = item.get("turn_context")
            if not isinstance(context, dict):
                continue
            standalone = self._clean_optional(context.get("standalone_query"))
            if not standalone or context.get("status") == "failed":
                continue
            if context.get("intent") == "CONVERSATIONAL" and context.get("relation") == "new_topic":
                continue
            message_id = self._message_id(item.get("id") or context.get("message_id"))
            return _TopicAnchor(
                topic_id=str(context.get("topic_id") or self._legacy_topic_id(message_id, standalone)),
                message_id=message_id,
                intent=self._valid_intent(context.get("intent")),
                standalone_query=standalone,
            )

        for item in reversed(chat_history or []):
            if item.get("role") != "user":
                continue
            context = item.get("turn_context")
            if isinstance(context, dict) and context.get("status") == "failed":
                continue
            content = clean_query_text(str(item.get("content") or ""))
            if not content or self._ELLIPTICAL_RE.match(content) or conversational_kind(content):
                continue
            message_id = self._message_id(item.get("id"))
            return _TopicAnchor(
                topic_id=self._legacy_topic_id(message_id, content),
                message_id=message_id,
                intent=None,
                standalone_query=content,
            )
        return None

    def _awaiting_clarification(self, chat_history: list[dict[str, Any]] | None) -> bool:
        for item in reversed(chat_history or []):
            if item.get("role") not in {"assistant", "context"}:
                continue
            context = item.get("turn_context")
            if isinstance(context, dict):
                return (
                    context.get("status") == "clarification_requested"
                    or context.get("intent") == "CLARIFY"
                )
            if item.get("role") == "assistant":
                return False
        return False

    def _model_input(
        self,
        query: str,
        anchor: _TopicAnchor,
        chat_history: list[dict[str, Any]] | None,
    ) -> str:
        rows = []
        for item in (chat_history or [])[-8:]:
            role = item.get("role")
            content = clean_query_text(str(item.get("content") or ""))
            if role in {"user", "assistant"} and content:
                rows.append(f"{role}: {content[:400]}")
        recent_messages = "\n".join(rows) or "None"
        return (
            f"Active intent: {anchor.intent or 'unknown'}\n"
            f"Active standalone topic: {anchor.standalone_query}\n"
            f"Recent messages:\n{recent_messages}\n\n"
            f"Latest user message: {query}\n"
            "Return only the relationship JSON object."
        )

    @classmethod
    def _resolve_query(
        cls,
        previous_query: str,
        follow_up: str,
        relation: ConversationRelation,
    ) -> str:
        previous = previous_query.rstrip(" ?.!;")
        current = clean_query_text(follow_up)
        lowered = current.lower()
        if re.search(r"\b(?:name|names|jina)\b", lowered):
            return f"{previous}; provide the person's full name"
        if re.search(r"\b(?:date|dates|when)\b", lowered):
            return f"{previous}; provide the exact date"
        label = {
            "correction": "correction",
            "comparison": "compare with",
            "clarification_response": "clarification",
        }.get(relation, "follow-up request")
        return f"{previous}; {label}: {current}"

    @classmethod
    def _depends_on_history(cls, query: str) -> bool:
        return bool(
            cls._ELLIPTICAL_RE.match(query)
            or cls._CORRECTION_RE.search(query)
            or cls._COMPARISON_RE.search(query)
            or cls._CONTINUATION_RE.search(query)
            or cls._SHORT_RESPONSE_RE.match(query)
            or cls._has_reference(query)
        )

    @classmethod
    def _has_reference(cls, query: str) -> bool:
        without_self_contained_time = cls._SELF_CONTAINED_TIME_REFERENCE_RE.sub(
            "",
            query,
        )
        return bool(set(tokenize(without_self_contained_time)) & cls._REFERENCE_TERMS)

    @classmethod
    def _looks_like_clarification_response(cls, query: str) -> bool:
        return bool(cls._SHORT_RESPONSE_RE.match(query) or len(tokenize(query)) <= 8)

    @classmethod
    def _looks_standalone(cls, query: str) -> bool:
        tokens = tokenize(query)
        return len(tokens) >= 3 and not cls._depends_on_history(query)

    @staticmethod
    def _legacy_topic_id(message_id: int | None, query: str) -> str:
        identity = str(message_id) if message_id is not None else clean_query_text(query).lower()
        return str(uuid5(NAMESPACE_URL, f"rag-chat-topic:{identity}"))

    @staticmethod
    def _message_id(value: Any) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _valid_intent(value: Any) -> str | None:
        intent = str(value or "").strip().upper()
        return intent if intent in SUPPORTED_INTENTS else None

    @staticmethod
    def _clean_optional(value: Any) -> str | None:
        if value is None:
            return None
        cleaned = clean_query_text(str(value).strip().strip("\"'"))
        if not cleaned or cleaned.lower() in {"null", "none", "n/a"}:
            return None
        return cleaned[:600]

    @staticmethod
    def _confidence(value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _parse_model_payload(raw: str) -> dict[str, Any]:
        cleaned = str(raw or "").strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start < 0 or end <= start:
                raise
            payload = json.loads(cleaned[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("Conversation resolver returned a non-object payload")
        return payload

    @staticmethod
    def _log(decision: ConversationDecision) -> None:
        logger.info(
            "Conversation relation | version=%s relation=%s is_follow_up=%s confidence=%.3f "
            "topic_id=%s referenced_message_id=%s source=%s intent=%s standalone_query=%s",
            decision.version,
            decision.relation,
            decision.is_follow_up,
            decision.confidence,
            decision.topic_id,
            decision.referenced_message_id,
            decision.source,
            decision.intent,
            decision.standalone_query,
        )


def create_guest_conversation_token(context: dict[str, Any] | None) -> str | None:
    safe_context = normalize_turn_context(context)
    if not safe_context:
        return None
    return create_access_token(
        {"token_use": CONVERSATION_TOKEN_USE, "conversation": safe_context},
        ttl_seconds=settings.CONVERSATION_TOKEN_TTL_SECONDS,
    )


def decode_guest_conversation_token(token: str | None) -> dict[str, Any] | None:
    if not token:
        return None
    payload = decode_access_token(token)
    if not payload or payload.get("token_use") != CONVERSATION_TOKEN_USE:
        return None
    return normalize_turn_context(payload.get("conversation"))


def normalize_turn_context(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    relation = str(value.get("relation") or "").strip().lower()
    if relation not in {*FOLLOW_UP_RELATIONS, "new_topic", "ambiguous"}:
        return {}
    topic_id = str(value.get("topic_id") or "").strip()
    standalone = ConversationContextResolver._clean_optional(value.get("standalone_query"))
    if not topic_id or (relation != "ambiguous" and not standalone):
        return {}
    is_follow_up = value.get("is_follow_up")
    if is_follow_up not in {True, False, None}:
        is_follow_up = None
    normalized = {
        "version": CONVERSATION_RESOLVER_VERSION,
        "relation": relation,
        "is_follow_up": is_follow_up,
        "confidence": ConversationContextResolver._confidence(value.get("confidence")),
        "topic_id": topic_id[:80],
        "referenced_message_id": ConversationContextResolver._message_id(value.get("referenced_message_id")),
        "intent": ConversationContextResolver._valid_intent(value.get("intent")),
        "standalone_query": standalone,
        "source": str(value.get("source") or "token")[:40],
        "reason": clean_query_text(str(value.get("reason") or "Restored guest conversation state."))[:300],
    }
    status = str(value.get("status") or "").strip().lower()
    if status in {"pending", "routing_resolved", "completed", "clarification_requested", "failed"}:
        normalized["status"] = status
    return normalized


def attach_guest_context(
    history: list[dict[str, Any]],
    context: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    normalized = normalize_turn_context(context)
    if not normalized:
        return history
    return [*history, {"role": "context", "content": "", "turn_context": normalized}]


def generation_history(history: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    return [
        {"role": str(item["role"]), "content": clean_query_text(str(item["content"]))}
        for item in (history or [])
        if item.get("role") in {"user", "assistant"}
        and not (
            isinstance(item.get("turn_context"), dict)
            and item["turn_context"].get("status") == "failed"
        )
        and clean_query_text(str(item.get("content") or ""))
    ]
