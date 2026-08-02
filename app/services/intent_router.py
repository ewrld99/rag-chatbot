"""LLM intent classifier for the UDOM student support chatbot."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import logging
import re
from typing import Any, Literal

from sqlalchemy.orm import Session

from app.db.session import (
    SessionFactory,
    SessionLocal,
    session_factory_from_session,
    session_scope,
)
from app.services.alias_expansion_service import AliasExpansionService
from app.services.conversation_context import ConversationContextResolver, ConversationDecision
from app.services.conversation_service import conversational_kind
from app.services.query_normalization import clean_query_text, significant_tokens, tokenize

logger = logging.getLogger(__name__)

SupportedIntent = Literal[
    "CONVERSATIONAL",
    "STUDENT_SUPPORT",
    "UDOM_DOCUMENT_SEARCH",
    "CLARIFY",
    "OUT_OF_SCOPE",
]

CLASSIFIER_PROMPT = """You are an intent classifier for a University of Dodoma student
support chatbot.

The chatbot has five supported intents:

CONVERSATIONAL:
The user is greeting the assistant, thanking it, saying goodbye,
acknowledging a response, or making brief social small talk. No
document retrieval is needed.

STUDENT_SUPPORT:
The user is asking for general guidance relevant to university
students, such as studying, revision, concentration, time management,
academic stress, note-taking, assignments, motivation, or university
life. The user wants advice or coaching, and the answer does not require
an authoritative rule or institution-specific fact.

UDOM_DOCUMENT_SEARCH:
The user is asking for official, factual, or institution-specific
information about the University of Dodoma. The answer should be
retrieved from indexed UDOM documents.

CLARIFY:
The request may relate to UDOM or student support, but essential
information is missing and answering would require guessing.

OUT_OF_SCOPE:
The request is unrelated to UDOM and unrelated to university student
learning or academic support.

Rules:
1. Greetings, thanks, goodbyes, acknowledgements, and brief social
   small talk are CONVERSATIONAL.
2. Do not answer general world-knowledge questions unless they are
   directly relevant to university study support.
3. Study strategies and general student advice are STUDENT_SUPPORT.
4. UDOM policies, procedures, programmes, fees, regulations,
   admissions, examinations, GPA, CGPA, grading, raw marks,
   grade points, award classifications, and official requirements are
   UDOM_DOCUMENT_SEARCH.
5. If a known UDOM alias/acronym match is provided, prefer
   UDOM_DOCUMENT_SEARCH unless the user is clearly asking for study advice.
6. Use recent conversation to resolve references such as "it",
   "that regulation", "what about IDIT students", and "the second programme".
7. Choose CLARIFY only when missing information prevents correct routing.
8. When generating the standalone_query, explicitly correct any spelling mistakes or typos (e.g., fix "udmo" to "UDOM").
9. First decide whether a reliable answer requires authoritative evidence. Questions about rules, causes for institutional action, eligibility, deadlines, fees, sanctions, consequences, required steps, or what the University permits require official evidence even when they mention assignments, studying, stress, or other student-support topics.
10. STUDENT_SUPPORT is only for advice or coaching that remains useful without consulting an official UDOM source.
11. When uncertain between STUDENT_SUPPORT and UDOM_DOCUMENT_SEARCH, set requires_official_evidence to true and choose UDOM_DOCUMENT_SEARCH.
12. Return only valid JSON. Keep reason under 12 words.

Contrastive examples:
- "How can I submit assignments on time?" -> STUDENT_SUPPORT, requires_official_evidence=false
- "Can a missed assignment cause discontinuation?" -> UDOM_DOCUMENT_SEARCH, requires_official_evidence=true
- "How can I manage examination stress?" -> STUDENT_SUPPORT, requires_official_evidence=false
- "What happens if I miss an examination?" -> UDOM_DOCUMENT_SEARCH, requires_official_evidence=true
- "Help me improve my study schedule" -> STUDENT_SUPPORT, requires_official_evidence=false
- "How long does student status continue after graduation?" -> UDOM_DOCUMENT_SEARCH, requires_official_evidence=true

Output:
{
  "intent": "CONVERSATIONAL | STUDENT_SUPPORT | UDOM_DOCUMENT_SEARCH | CLARIFY | OUT_OF_SCOPE",
  "requires_official_evidence": true,
  "confidence": 0.0,
  "reason": "Brief reason under 12 words",
  "standalone_query": "Resolved standalone query or null"
}"""


@dataclass(frozen=True)
class IntentDecision:
    intent: SupportedIntent
    confidence: float
    reason: str
    standalone_query: str | None
    normalized_query: str
    source: str = "llm"
    filters: dict[str, Any] = field(default_factory=dict)
    conversation: ConversationDecision | None = None

    @property
    def action(self) -> str:
        return {
            "CONVERSATIONAL": "conversational",
            "STUDENT_SUPPORT": "student_support",
            "UDOM_DOCUMENT_SEARCH": "document_search",
            "CLARIFY": "clarify",
            "OUT_OF_SCOPE": "out_of_scope",
        }[self.intent]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "intent": self.intent,
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "standalone_query": self.standalone_query,
            "normalized_query": self.normalized_query,
            "source": self.source,
            "filters": self.filters,
        }
        if self.conversation is not None:
            payload["conversation"] = self.conversation.to_dict()
        return payload


RoutingDecision = IntentDecision


class IntentRouter:
    """Conversational fast path plus LLM classification and guardrails."""

    _ELLIPTICAL_FOLLOW_UP_RE = re.compile(
        r"^\s*(?:"
        r"(?:and\s+)?(?:what|who|when|where|which)\s+(?:exactly|then|next)|"
        r"(?:what|who)\s+(?:is|are)\s+(?:the\s+|his\s+|her\s+|their\s+)?(?:name|date|number|title)|"
        r"(?:please\s+)?(?:give|tell|show|state|provide|mention)\s+"
        r"(?:me\s+)?(?:the\s+)?(?:name|names|date|dates|number|numbers|"
        r"title|titles|details?|answer|source|sources)(?:\s+(?:only|instead|exactly))?|"
        r"(?:more|further)\s+(?:details?|information)|"
        r"(?:explain|elaborate)(?:\s+(?:more|further|on\s+that))?|"
        r"(?:nipe\s+jina|jina\s+lake|eleza\s+zaidi|endelea)"
        r")\s*[?.!]*\s*$",
        re.IGNORECASE,
    )
    _OFFICIAL_EVIDENCE_REQUEST_RE = re.compile(
        r"\b(?:according\s+to|official|polic(?:y|ies)|regulat\w*|rules?|"
        r"requir\w*|eligib\w*|permit\w*|prohibit\w*|deadlines?|fees?|"
        r"appeals?|penalt\w*|sanction\w*|consequences?|"
        r"discontinu\w*|suspend\w*|expel\w*|deregister\w*|debar\w*)\b",
        re.IGNORECASE,
    )
    _FACTUAL_QUESTION_RE = re.compile(
        r"^\s*(?:(?:what|when|where|who|which|whose)\b|"
        r"(?:is|are|was|were|can|could|does|do|did|will|would|should)\b|"
        r"how\s+(?:long|much|many|often)\b)",
        re.IGNORECASE,
    )
    _ADVICE_REQUEST_RE = re.compile(
        r"\b(?:advice|tips?|coach\w*|motivat\w*|study\s+better|"
        r"improve\s+my|manage\s+my|cope\s+with|help\s+me|"
        r"plan\s+my|organize\s+my|concentrate\s+better)\b",
        re.IGNORECASE,
    )

    UDOM_TERMS = {
        "udom", "dodoma", "university of dodoma", "admission", "admissions",
        "programme", "programmes", "program", "course", "courses", "unit",
        "units", "module", "modules", "fee", "fees", "tuition",
        "registration", "regulation", "regulations", "guideline", "guidelines",
        "examination", "examinations", "exam", "exams", "gpa", "cgpa",
        "grade point average", "continuous assessment", "transcript", "graduation",
        "supplementary", "discontinuation", "undergraduate", "postgraduate",
        "certificate", "diploma", "degree", "student records", "student record",
        "student records system", "student portal", "sr", "sr2", "oas", "tcu",
        "nactvet", "dress", "dressing", "dress code", "dressing code", "attire",
        "uniform", "conduct", "discipline", "student conduct", "code of conduct",
        "postpone", "postponement", "defer", "deferment", "intermission",
        "year of study", "masomo", "mwaka wa masomo", "kozi", "ada",
        "usajili", "udahili", "mtihani", "mitihani", "alama", "matokeo",
        "kuahirisha", "ahirisha", "kusitisha masomo", "rufaa", "mahafali",
        "mavazi", "kanuni za mavazi", "nidhamu", "maadili",
    }
    OFFICIAL_DOCUMENT_TERMS = {
        "admission", "admissions", "programme", "programmes", "program",
        "course", "courses", "fee", "fees", "tuition", "registration",
        "regulation", "regulations", "guideline", "guidelines", "policy",
        "policies", "procedure", "procedures", "requirement", "requirements",
        "examination", "examinations", "exam", "exams", "gpa", "cgpa",
        "grade", "grades", "grading", "grade point", "grade points",
        "grade point average", "raw marks", "marks", "score", "scores",
        "total score", "course weight", "weight", "weights", "credit",
        "credits", "classification", "classifications", "award",
        "calculate", "calculated", "calculation", "compute", "computed",
        "transcript", "graduation", "supplementary", "discontinuation",
        "undergraduate", "postgraduate", "certificate", "diploma", "degree",
        "student records", "student record", "student portal", "sr", "sr2",
        "oas", "tcu", "nactvet", "dress", "dressing", "dress code",
        "dressing code", "attire", "uniform", "conduct", "discipline",
        "student conduct", "code of conduct", "postpone", "postponement",
        "defer", "deferment", "intermission", "year of study",
        "masomo", "mwaka wa masomo", "kozi", "ada", "usajili", "udahili",
        "mtihani", "mitihani", "alama", "matokeo", "kuahirisha", "ahirisha",
        "kusitisha masomo", "rufaa", "mahafali", "mavazi",
        "kanuni za mavazi", "nidhamu", "maadili",
    }

    STUDENT_SUPPORT_TERMS = {
        "study", "studying", "revision", "revise", "concentrate", "concentrating", "concentration", "focus",
        "time", "manage", "management", "time management", "stress", "academic stress", "notes", "note", "note taking",
        "assignment", "assignments", "motivation", "procrastination",
        "exam preparation", "planning", "planner", "productivity", "learning", "read", "reading", "memorize", "memorization", "understand",
        "university life", "student life",
    }
    REFERENCE_TERMS = {
        "it", "this", "that", "these", "those", "they", "them", "there",
        "same", "previous", "above", "earlier", "second", "first", "one", "ones",
    }

    def __init__(
        self,
        db: Session | None = None,
        generator: Any | None = None,
        *,
        session_factory: SessionFactory | None = None,
    ) -> None:
        self._session_factory = session_factory or (
            session_factory_from_session(db) if db is not None else SessionLocal
        )
        self.generator = generator
        self.context_resolver = ConversationContextResolver(generator)

    def classify(
        self,
        query: str,
        chat_history: list[dict[str, str]] | None = None,
        user_profile: dict[str, Any] | None = None,
    ) -> IntentDecision:
        normalized_query = self._lightweight_normalize(query)
        conversation = self.context_resolver.resolve(query, chat_history)

        if not clean_query_text(query):
            return self._attach_conversation(
                self._decision("CLARIFY", 1.0, "The query is empty.", None, normalized_query, "validation"),
                conversation,
            )

        if conversation.relation == "ambiguous":
            source = (
                "rule:unresolved_follow_up"
                if conversation.source == "rule" and conversation.referenced_message_id is None
                else f"conversation:{conversation.source}"
            )
            return self._attach_conversation(
                self._decision(
                    "CLARIFY",
                    max(0.8, conversation.confidence),
                    conversation.reason,
                    None,
                    normalized_query,
                    source,
                ),
                conversation,
            )

        social_turn = conversational_kind(query)
        if social_turn is not None:
            return self._attach_conversation(
                self._decision(
                    "CONVERSATIONAL",
                    1.0,
                    f"Exact {social_turn} phrase matched the conversational fast path.",
                    clean_query_text(query),
                    normalized_query,
                    "rule:conversational",
                ),
                conversation,
            )

        follow_up_decision = self._conversation_follow_up_decision(
            conversation, normalized_query, chat_history, user_profile
        )
        if follow_up_decision is not None:
            return self._attach_conversation(follow_up_decision, conversation)

        if conversation.source == "model" and conversation.intent in {
            "CONVERSATIONAL",
            "STUDENT_SUPPORT",
            "UDOM_DOCUMENT_SEARCH",
            "CLARIFY",
            "OUT_OF_SCOPE",
        }:
            model_decision = self._decision(
                conversation.intent,  # type: ignore[arg-type]
                conversation.confidence,
                conversation.reason,
                conversation.standalone_query,
                normalized_query,
                "conversation:model",
                (
                    self._profile_filters(user_profile)
                    if conversation.intent == "UDOM_DOCUMENT_SEARCH"
                    else {}
                ),
            )
            return self._attach_conversation(model_decision, conversation)

        aliases = self._matched_aliases(query)
        rule_decision = self._rule_document_search_decision(
            query,
            normalized_query,
            aliases,
            chat_history,
            user_profile,
        )
        if rule_decision is not None:
            return self._attach_conversation(rule_decision, conversation)

        try:
            payload = self._classify_with_llm(query, normalized_query, aliases, chat_history)
            decision = self._decision_from_payload(payload, query, normalized_query, user_profile)
        except Exception as exc:
            logger.warning("Intent classifier failed; using fallback routing: %s", exc)
            decision = self._fallback_decision(query, normalized_query, aliases, chat_history, user_profile)

        decision = self._apply_guardrails(decision, query, aliases, chat_history, user_profile)
        return self._attach_conversation(decision, conversation)

    def _conversation_follow_up_decision(
        self,
        conversation: ConversationDecision,
        normalized_query: str,
        chat_history: list[dict[str, Any]] | None,
        user_profile: dict[str, Any] | None,
    ) -> IntentDecision | None:
        if conversation.is_follow_up is not True or not conversation.standalone_query:
            return None

        intent = conversation.intent
        if intent is None and self._history_has_document_signal(chat_history):
            intent = "UDOM_DOCUMENT_SEARCH"

        if intent == "UDOM_DOCUMENT_SEARCH":
            return self._decision(
                "UDOM_DOCUMENT_SEARCH",
                max(0.80, conversation.confidence),
                "Follow-up resolved from structured document-search context.",
                conversation.standalone_query,
                normalized_query,
                "rule:document_follow_up" if conversation.source == "rule" else "conversation:model",
                self._profile_filters(user_profile),
            )
        if intent == "STUDENT_SUPPORT":
            return self._decision(
                "STUDENT_SUPPORT",
                max(0.80, conversation.confidence),
                "Follow-up resolved from structured student-support context.",
                conversation.standalone_query,
                normalized_query,
                "rule:student_support_follow_up" if conversation.source == "rule" else "conversation:model",
            )
        if intent == "CONVERSATIONAL":
            return self._decision(
                "CONVERSATIONAL",
                max(0.80, conversation.confidence),
                "Follow-up resolved from structured conversational context.",
                conversation.standalone_query,
                normalized_query,
                "rule:conversational_follow_up",
            )
        if intent == "OUT_OF_SCOPE":
            return self._decision(
                "OUT_OF_SCOPE",
                max(0.80, conversation.confidence),
                "Follow-up remains outside the supported scope.",
                conversation.standalone_query,
                normalized_query,
                "rule:out_of_scope_follow_up",
            )
        return None

    @staticmethod
    def _attach_conversation(
        decision: IntentDecision,
        conversation: ConversationDecision,
    ) -> IntentDecision:
        resolved_conversation = conversation.with_routing(
            decision.intent,
            decision.standalone_query,
        )
        logger.info(
            "Conversation routing finalized | version=%s relation=%s is_follow_up=%s "
            "confidence=%.3f topic_id=%s referenced_message_id=%s resolver_source=%s "
            "intent=%s standalone_query=%s",
            resolved_conversation.version,
            resolved_conversation.relation,
            resolved_conversation.is_follow_up,
            resolved_conversation.confidence,
            resolved_conversation.topic_id,
            resolved_conversation.referenced_message_id,
            resolved_conversation.source,
            decision.intent,
            resolved_conversation.standalone_query,
        )
        return replace(
            decision,
            conversation=resolved_conversation,
        )

    def _elliptical_follow_up_decision(
        self,
        query: str,
        normalized_query: str,
        chat_history: list[dict[str, str]] | None,
        user_profile: dict[str, Any] | None,
    ) -> IntentDecision | None:
        if not self._is_elliptical_follow_up(query):
            return None

        previous_query = self._last_substantive_user_query(chat_history)
        if previous_query and self._history_has_document_signal(chat_history):
            return self._decision(
                "UDOM_DOCUMENT_SEARCH",
                0.95,
                "Elliptical follow-up resolved from document-search history.",
                self._resolve_follow_up_query(previous_query, query),
                normalized_query,
                "rule:document_follow_up",
                self._profile_filters(user_profile),
            )

        return self._decision(
            "CLARIFY",
            0.9,
            "The short follow-up has no prior topic to resolve.",
            None,
            normalized_query,
            "rule:unresolved_follow_up",
        )

    @classmethod
    def _is_elliptical_follow_up(cls, query: str) -> bool:
        return bool(cls._ELLIPTICAL_FOLLOW_UP_RE.match(clean_query_text(query)))

    @classmethod
    def _last_substantive_user_query(
        cls,
        chat_history: list[dict[str, str]] | None,
    ) -> str | None:
        for item in reversed(chat_history or []):
            if item.get("role") != "user":
                continue
            content = clean_query_text(str(item.get("content", "")))
            if content and not cls._is_elliptical_follow_up(content):
                return content
        return None

    @staticmethod
    def _resolve_follow_up_query(previous_query: str, follow_up: str) -> str:
        cleaned_previous = previous_query.rstrip(" ?.!")
        lowered_follow_up = clean_query_text(follow_up).lower()
        if re.search(r"\b(?:name|names|jina)\b", lowered_follow_up):
            return f"{cleaned_previous}; provide the person's full name"
        if re.search(r"\b(?:date|dates|when)\b", lowered_follow_up):
            return f"{cleaned_previous}; provide the exact date"
        return f"{cleaned_previous}; follow-up request: {clean_query_text(follow_up)}"

    def route(
        self,
        query: str,
        chat_history: list[dict[str, str]] | None = None,
        user_profile: dict[str, Any] | None = None,
    ) -> IntentDecision:
        return self.classify(query, chat_history=chat_history, user_profile=user_profile)

    def should_probe_documents(
        self,
        query: str,
        decision: IntentDecision,
    ) -> bool:
        """Return whether a support answer needs retrieval-based arbitration."""
        if decision.intent != "STUDENT_SUPPORT":
            return False
        if self._requires_official_evidence(query):
            return True
        if self._ADVICE_REQUEST_RE.search(query):
            return False
        return bool(self._FACTUAL_QUESTION_RE.search(query))

    def promote_to_document_search(
        self,
        decision: IntentDecision,
        query: str,
        confidence: float,
    ) -> IntentDecision:
        """Promote an uncertain route after strong official-document retrieval."""
        promoted = self._decision(
            "UDOM_DOCUMENT_SEARCH",
            max(decision.confidence, confidence),
            "Strong UDOM retrieval evidence overrode the general-support route.",
            decision.standalone_query or clean_query_text(query),
            decision.normalized_query,
            f"{decision.source}+retrieval_guardrail",
            decision.filters,
        )
        if decision.conversation is None:
            return promoted
        return self._attach_conversation(promoted, decision.conversation)

    def _rule_document_search_decision(
        self,
        query: str,
        normalized_query: str,
        aliases: dict[str, list[str]],
        chat_history: list[dict[str, str]] | None,
        user_profile: dict[str, Any] | None,
    ) -> IntentDecision | None:
        terms = self._expanded_terms(query, aliases)
        if self._has_reference_without_anchor(terms, bool(chat_history)):
            return None
        explicit_evidence_request = self._requires_official_evidence(query)
        topic_document_match = self._requires_document_search(terms, aliases)
        if self._ADVICE_REQUEST_RE.search(query) and not explicit_evidence_request:
            topic_document_match = False
        if not (explicit_evidence_request or topic_document_match):
            return None
        return self._decision(
            "UDOM_DOCUMENT_SEARCH",
            0.9,
            "Official UDOM terms matched rule-based document routing.",
            clean_query_text(query),
            normalized_query,
            "rule:document_search",
            self._profile_filters(user_profile),
        )

    def _classify_with_llm(
        self,
        query: str,
        normalized_query: str,
        aliases: dict[str, list[str]],
        chat_history: list[dict[str, str]] | None,
    ) -> dict[str, Any]:
        if self.generator is None:
            raise RuntimeError("No generation service available for LLM classification")

        response = self.generator._create_utility_completion(
            "intent_classifier",
            model=self.generator.model,
            messages=[
                {"role": "system", "content": CLASSIFIER_PROMPT},
                {"role": "user", "content": self._classifier_input(query, normalized_query, aliases, chat_history)},
            ],
            temperature=0.0,
            max_tokens=320,
            response_format={"type": "json_object"},
        )
        return self._parse_json(response.choices[0].message.content.strip())

    def _classifier_input(
        self,
        query: str,
        normalized_query: str,
        aliases: dict[str, list[str]],
        chat_history: list[dict[str, str]] | None,
    ) -> str:
        return (
            "Recent conversation:\n"
            f"{self._history_text(chat_history)}\n\n"
            "Latest user query:\n"
            f"{query}\n\n"
            "Lightweight normalized query:\n"
            f"{normalized_query or 'None'}\n\n"
            "Known UDOM alias/acronym matches from the retrieval alias table:\n"
            f"{self._alias_text(aliases)}\n\n"
            "Classify the latest user query and return only the JSON object."
        )

    def _history_text(self, chat_history: list[dict[str, str]] | None, limit: int = 12) -> str:
        rows: list[str] = []
        for item in (chat_history or [])[-limit:]:
            role = item.get("role")
            content = clean_query_text(str(item.get("content", "")))
            if role in {"user", "assistant"} and content:
                rows.append(f"{role}: {content[:500]}")
        return "\n".join(rows) if rows else "None"

    def _alias_text(self, aliases: dict[str, list[str]]) -> str:
        if not aliases:
            return "None"
        chunks = []
        for term in sorted(aliases):
            chunks.append(f"{term}: {', '.join(aliases[term][:6])}")
        return "; ".join(chunks)

    def _decision_from_payload(
        self,
        payload: dict[str, Any],
        query: str,
        normalized_query: str,
        user_profile: dict[str, Any] | None,
    ) -> IntentDecision:
        intent = self._normalize_intent(payload.get("intent"))
        requires_official_evidence = self._optional_bool(
            payload.get("requires_official_evidence")
        )
        if requires_official_evidence is True and intent == "STUDENT_SUPPORT":
            intent = "UDOM_DOCUMENT_SEARCH"
        confidence = self._safe_confidence(payload.get("confidence"), default=0.55)
        reason = clean_query_text(str(payload.get("reason") or "LLM intent classifier decision."))
        standalone_query = self._normalize_standalone_query(payload.get("standalone_query"))
        if intent in {"STUDENT_SUPPORT", "UDOM_DOCUMENT_SEARCH"} and not standalone_query:
            standalone_query = clean_query_text(query)
        filters = self._profile_filters(user_profile) if intent == "UDOM_DOCUMENT_SEARCH" else {}
        return self._decision(intent, confidence, reason, standalone_query, normalized_query, "llm", filters)

    def _fallback_decision(
        self,
        query: str,
        normalized_query: str,
        aliases: dict[str, list[str]],
        chat_history: list[dict[str, str]] | None,
        user_profile: dict[str, Any] | None,
    ) -> IntentDecision:
        terms = self._expanded_terms(query, aliases)
        has_history = bool(chat_history)
        if self._has_reference_without_anchor(terms, has_history):
            return self._decision("CLARIFY", 0.7, "The query appears to refer to previous context that is not available.", None, normalized_query, "fallback")
        if conversational_kind(query) is not None:
            return self._decision("CONVERSATIONAL", 1.0, "Fallback matched a simple conversational phrase.", clean_query_text(query), normalized_query, "fallback")
        if self._has_udom_signal(terms, aliases):
            return self._decision("UDOM_DOCUMENT_SEARCH", 0.65, "Fallback found UDOM-specific terms or registered aliases.", clean_query_text(query), normalized_query, "fallback", self._profile_filters(user_profile))
        if self._has_student_support_signal(terms):
            return self._decision("STUDENT_SUPPORT", 0.6, "Fallback found general student support terms.", clean_query_text(query), normalized_query, "fallback")
        return self._decision("OUT_OF_SCOPE", 0.5, "Fallback found no UDOM or student-support signal.", None, normalized_query, "fallback")

    def _apply_guardrails(
        self,
        decision: IntentDecision,
        query: str,
        aliases: dict[str, list[str]],
        chat_history: list[dict[str, str]] | None,
        user_profile: dict[str, Any] | None,
    ) -> IntentDecision:
        terms = self._expanded_terms(query, aliases)
        has_history = bool(chat_history)

        if (
            has_history
            and terms & self.REFERENCE_TERMS
            and decision.intent != "UDOM_DOCUMENT_SEARCH"
            and self._history_has_document_signal(chat_history)
        ):
            return self._decision(
                "UDOM_DOCUMENT_SEARCH",
                max(decision.confidence, 0.76),
                "Follow-up reference resolved from recent document-search context.",
                decision.standalone_query or clean_query_text(query),
                decision.normalized_query,
                f"{decision.source}+guardrail",
                self._profile_filters(user_profile),
            )

        if self._has_reference_without_anchor(terms, has_history) and decision.intent != "CLARIFY":
            return self._decision(
                "CLARIFY",
                max(decision.confidence, 0.72),
                "The query contains an unresolved reference and needs a focused clarification.",
                None,
                decision.normalized_query,
                f"{decision.source}+guardrail",
            )
        explicit_evidence_request = self._requires_official_evidence(query)
        topic_document_match = self._requires_document_search(terms, aliases)
        if self._ADVICE_REQUEST_RE.search(query) and not explicit_evidence_request:
            topic_document_match = False
        if decision.intent in {"CONVERSATIONAL", "STUDENT_SUPPORT"} and (
            explicit_evidence_request or topic_document_match
        ):
            return self._decision(
                "UDOM_DOCUMENT_SEARCH",
                max(decision.confidence, 0.78),
                "Official UDOM terms, GPA/grading terms, or registered aliases require document search.",
                clean_query_text(query),
                decision.normalized_query,
                f"{decision.source}+guardrail",
                self._profile_filters(user_profile),
            )
        if decision.intent == "CONVERSATIONAL" and self._has_student_support_signal(terms):
            return self._decision(
                "STUDENT_SUPPORT",
                max(decision.confidence, 0.72),
                "The query contains a substantive student-support request beyond small talk.",
                decision.standalone_query or clean_query_text(query),
                decision.normalized_query,
                f"{decision.source}+guardrail",
            )

        if decision.intent == "OUT_OF_SCOPE" and self._has_udom_signal(terms, aliases):
            return self._decision(
                "UDOM_DOCUMENT_SEARCH",
                max(decision.confidence, 0.72),
                "UDOM-specific terms or aliases were present, so the request should use document search.",
                decision.standalone_query or clean_query_text(query),
                decision.normalized_query,
                f"{decision.source}+guardrail",
                self._profile_filters(user_profile),
            )
        if decision.intent == "OUT_OF_SCOPE" and self._has_student_support_signal(terms):
            return self._decision(
                "STUDENT_SUPPORT",
                max(decision.confidence, 0.68),
                "The query asks for general university student support.",
                decision.standalone_query or clean_query_text(query),
                decision.normalized_query,
                f"{decision.source}+guardrail",
            )
        if decision.intent == "UDOM_DOCUMENT_SEARCH":
            return self._decision(
                decision.intent,
                decision.confidence,
                decision.reason,
                decision.standalone_query or clean_query_text(query),
                decision.normalized_query,
                decision.source,
                self._profile_filters(user_profile),
            )
        return decision
    def _lightweight_normalize(self, query: str) -> str:
        normalized = " ".join(significant_tokens(clean_query_text(query)))
        return normalized or clean_query_text(query).lower()

    def _matched_aliases(self, query: str) -> dict[str, list[str]]:
        try:
            with session_scope(self._session_factory) as db:
                return AliasExpansionService(db).get_expansions(query)
        except Exception as exc:
            logger.warning("Could not load retrieval aliases for intent classifier: %s", exc)
            return {}

    def _expanded_terms(self, query: str, aliases: dict[str, list[str]]) -> set[str]:
        terms = set(tokenize(query)) | set(significant_tokens(query))
        for term, values in aliases.items():
            terms.update(tokenize(term))
            for value in values:
                terms.update(tokenize(value))
                phrase = " ".join(tokenize(value))
                if phrase:
                    terms.add(phrase)
        return terms

    def _has_udom_signal(self, terms: set[str], aliases: dict[str, list[str]]) -> bool:
        return bool(aliases) or bool(terms & self.UDOM_TERMS)

    def _requires_document_search(self, terms: set[str], aliases: dict[str, list[str]]) -> bool:
        if aliases:
            return True
        if terms & {"gpa", "cgpa", "sr", "sr2", "oas", "tcu", "nactvet"}:
            return True
        return bool((terms & self.UDOM_TERMS) and (terms & self.OFFICIAL_DOCUMENT_TERMS))

    def _requires_official_evidence(self, query: str) -> bool:
        return bool(self._OFFICIAL_EVIDENCE_REQUEST_RE.search(query or ""))

    def _has_student_support_signal(self, terms: set[str]) -> bool:
        return bool(terms & self.STUDENT_SUPPORT_TERMS)

    def _has_reference_without_anchor(self, terms: set[str], has_history: bool) -> bool:
        if has_history or not (terms & self.REFERENCE_TERMS):
            return False
        non_reference_terms = terms - self.REFERENCE_TERMS
        return len(non_reference_terms) <= 3 and not bool(non_reference_terms & self.UDOM_TERMS)

    def _history_has_document_signal(self, chat_history: list[dict[str, Any]] | None) -> bool:
        if any(
            isinstance(item.get("turn_context"), dict)
            and item["turn_context"].get("intent") == "UDOM_DOCUMENT_SEARCH"
            for item in (chat_history or [])[-12:]
        ):
            return True
        history_text = " ".join(
            clean_query_text(str(item.get("content", "")))
            for item in (chat_history or [])[-12:]
            if item.get("role") in {"user", "assistant"}
        )
        terms = set(tokenize(history_text)) | set(significant_tokens(history_text))
        return bool(terms & (self.UDOM_TERMS | self.OFFICIAL_DOCUMENT_TERMS))

    def _profile_filters(self, user_profile: dict[str, Any] | None) -> dict[str, Any]:
        if not user_profile:
            return {}
        filters: dict[str, Any] = {}
        programme = user_profile.get("programme")
        year = user_profile.get("year_of_study")
        if programme and str(programme).strip():
            filters["programme"] = str(programme).strip()
        if year and str(year).strip().lower() != "unknown":
            filters["year"] = str(year).strip()
        return filters

    def _decision(
        self,
        intent: SupportedIntent,
        confidence: float,
        reason: str,
        standalone_query: str | None,
        normalized_query: str,
        source: str,
        filters: dict[str, Any] | None = None,
    ) -> IntentDecision:
        return IntentDecision(
            intent=intent,
            confidence=max(0.0, min(1.0, confidence)),
            reason=reason[:400],
            standalone_query=standalone_query,
            normalized_query=normalized_query,
            source=source,
            filters=filters or {},
        )

    def _parse_json(self, raw: str) -> dict[str, Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            recovered = self._recover_truncated_json(raw)
            if recovered is not None:
                logger.warning(
                    "Recovered truncated classifier JSON prefix: %s",
                    raw[:120],
                )
                return recovered
        raise ValueError(f"Classifier did not return valid JSON: {raw[:120]}")

    def _recover_truncated_json(self, raw: str) -> dict[str, Any] | None:
        intent_match = re.search(
            r'"intent"\s*:\s*"(?P<intent>CONVERSATIONAL|STUDENT_SUPPORT|UDOM_DOCUMENT_SEARCH|CLARIFY|OUT_OF_SCOPE)"',
            raw,
        )
        if not intent_match:
            return None

        confidence_match = re.search(r'"confidence"\s*:\s*(?P<confidence>0(?:\.\d+)?|1(?:\.0+)?)', raw)
        reason_match = re.search(r'"reason"\s*:\s*"(?P<reason>[^"]*)', raw)
        standalone_match = re.search(r'"standalone_query"\s*:\s*"(?P<query>[^"]*)', raw)

        return {
            "intent": intent_match.group("intent"),
            "confidence": float(confidence_match.group("confidence")) if confidence_match else 0.55,
            "reason": clean_query_text(reason_match.group("reason")) if reason_match else "Recovered truncated classifier output.",
            "standalone_query": clean_query_text(standalone_match.group("query")) if standalone_match else None,
        }

    def _normalize_intent(self, value: Any) -> SupportedIntent:
        cleaned = re.sub(r"[^A-Z_]", "", str(value or "").upper())
        mapping: dict[str, SupportedIntent] = {
            "GREETING": "CONVERSATIONAL",
            "SMALLTALK": "CONVERSATIONAL",
            "SMALL_TALK": "CONVERSATIONAL",
            "GENERAL_STUDENT_SUPPORT": "STUDENT_SUPPORT",
            "STUDENT": "STUDENT_SUPPORT",
            "SUPPORT": "STUDENT_SUPPORT",
            "UNIVERSITY_INFO": "UDOM_DOCUMENT_SEARCH",
            "DOCUMENT_SEARCH": "UDOM_DOCUMENT_SEARCH",
            "UDOM_SEARCH": "UDOM_DOCUMENT_SEARCH",
            "RAG": "UDOM_DOCUMENT_SEARCH",
            "CLARIFICATION": "CLARIFY",
            "OUT_OF_DOMAIN": "OUT_OF_SCOPE",
            "OUTOFSCOPE": "OUT_OF_SCOPE",
        }
        normalized = mapping.get(cleaned, cleaned)
        valid = {"CONVERSATIONAL", "STUDENT_SUPPORT", "UDOM_DOCUMENT_SEARCH", "CLARIFY", "OUT_OF_SCOPE"}
        return normalized if normalized in valid else "UDOM_DOCUMENT_SEARCH"  # type: ignore[return-value]

    def _normalize_standalone_query(self, value: Any) -> str | None:
        if value is None:
            return None
        text = clean_query_text(str(value).strip().strip('"\''))
        if not text or text.lower() in {"null", "none", "n/a", "na"}:
            return None
        return text[:600]

    def _safe_confidence(self, value: Any, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _optional_bool(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        normalized = str(value or "").strip().casefold()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
        return None
