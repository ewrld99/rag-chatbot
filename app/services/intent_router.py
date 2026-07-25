"""LLM intent classifier for the UDOM student support chatbot."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import re
from typing import Any, Literal

from sqlalchemy.orm import Session

from app.services.alias_expansion_service import AliasExpansionService
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
life. The answer does not require official UDOM documents.

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
   "that regulation", and "the second programme".
7. Choose CLARIFY only when missing information prevents correct routing.
8. Return only valid JSON.

Output:
{
  "intent": "CONVERSATIONAL | STUDENT_SUPPORT | UDOM_DOCUMENT_SEARCH | CLARIFY | OUT_OF_SCOPE",
  "confidence": 0.0,
  "reason": "Brief explanation",
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
        return {
            "intent": self.intent,
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "standalone_query": self.standalone_query,
            "normalized_query": self.normalized_query,
            "source": self.source,
            "filters": self.filters,
        }


RoutingDecision = IntentDecision


class IntentRouter:
    """Conversational fast path plus LLM classification and guardrails."""

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
        "nactvet",
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
        "oas", "tcu", "nactvet",
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

    def __init__(self, db: Session, generator: Any | None = None) -> None:
        self.db = db
        self.generator = generator

    def classify(
        self,
        query: str,
        chat_history: list[dict[str, str]] | None = None,
        user_profile: dict[str, Any] | None = None,
    ) -> IntentDecision:
        normalized_query = self._lightweight_normalize(query)

        if not clean_query_text(query):
            return self._decision("CLARIFY", 1.0, "The query is empty.", None, normalized_query, "validation")

        social_turn = conversational_kind(query)
        if social_turn is not None:
            return self._decision(
                "CONVERSATIONAL",
                1.0,
                f"Exact {social_turn} phrase matched the conversational fast path.",
                clean_query_text(query),
                normalized_query,
                "rule:conversational",
            )

        aliases = self._matched_aliases(query)

        try:
            payload = self._classify_with_llm(query, normalized_query, aliases, chat_history)
            decision = self._decision_from_payload(payload, query, normalized_query, user_profile)
        except Exception as exc:
            logger.warning("Intent classifier failed; using fallback routing: %s", exc)
            decision = self._fallback_decision(query, normalized_query, aliases, chat_history, user_profile)

        return self._apply_guardrails(decision, query, aliases, chat_history, user_profile)

    def route(
        self,
        query: str,
        chat_history: list[dict[str, str]] | None = None,
        user_profile: dict[str, Any] | None = None,
    ) -> IntentDecision:
        return self.classify(query, chat_history=chat_history, user_profile=user_profile)

    def _classify_with_llm(
        self,
        query: str,
        normalized_query: str,
        aliases: dict[str, list[str]],
        chat_history: list[dict[str, str]] | None,
    ) -> dict[str, Any]:
        if self.generator is None:
            raise RuntimeError("No generation service available for LLM classification")

        response = self.generator.create_completion(
            "intent_classifier",
            model=self.generator.model,
            messages=[
                {"role": "system", "content": CLASSIFIER_PROMPT},
                {"role": "user", "content": self._classifier_input(query, normalized_query, aliases, chat_history)},
            ],
            temperature=0.0,
            max_tokens=220,
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

    def _history_text(self, chat_history: list[dict[str, str]] | None, limit: int = 6) -> str:
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

        if self._has_reference_without_anchor(terms, has_history) and decision.intent != "CLARIFY":
            return self._decision(
                "CLARIFY",
                max(decision.confidence, 0.72),
                "The query contains an unresolved reference and needs a focused clarification.",
                None,
                decision.normalized_query,
                f"{decision.source}+guardrail",
            )
        if decision.intent in {"CONVERSATIONAL", "STUDENT_SUPPORT"} and self._requires_document_search(terms, aliases):
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
            return AliasExpansionService(self.db).get_expansions(query)
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

    def _has_student_support_signal(self, terms: set[str]) -> bool:
        return bool(terms & self.STUDENT_SUPPORT_TERMS)

    def _has_reference_without_anchor(self, terms: set[str], has_history: bool) -> bool:
        if has_history or not (terms & self.REFERENCE_TERMS):
            return False
        non_reference_terms = terms - self.REFERENCE_TERMS
        return len(non_reference_terms) <= 3 and not bool(non_reference_terms & self.UDOM_TERMS)

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
                return json.loads(match.group(0))
        raise ValueError(f"Classifier did not return valid JSON: {raw[:120]}")

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
