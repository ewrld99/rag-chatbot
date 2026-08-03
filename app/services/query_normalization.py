"""
app/services/query_normalization.py
------------------------------------
Deterministic query normalization and keyword expansion for retrieval.

The sparse retriever can pass database-backed aliases into this module, so
UDOM-specific acronyms do not have to be hardcoded in Python. A small fallback
map remains for resilience when the database table has not been migrated yet.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class QueryVariant:
    label: str
    query: str
    weight: float


MAX_EXPANDED_TERMS = 12
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
SAFE_METADATA_TEXT_KEYS = ("source", "document_title", "source_url", "category")

DOMAIN_TERMS = {
    "udom",
    "dodoma",
    "university",
    "universityofdodoma",
}

SWAHILI_QUERY_TERMS = {
    "ada", "ahirisha", "baada", "barua", "chuo", "eleza", "fanya",
    "hatua", "hati", "hivyo", "jinsi", "kanuni", "kuahirisha",
    "kufahamu", "kufanya", "kughairi", "kughairisha", "kuhusu",
    "kujisajili", "kusitisha", "kutuma", "maadili", "mahafali", "malipo",
    "masomo", "matokeo", "mavazi", "mchakato", "mfumo", "mhitimu",
    "mitihani", "mtihani", "mwanafunzi", "mwaka", "nani", "nataka",
    "nidhamu", "nipe", "nini", "nita", "nitaruhusiwa", "ombeni",
    "ombi", "programu", "rufaa", "sajili", "shahada",
    "taratibu", "utaratibu", "udahili", "usajili", "vipi", "wapi",
    "wastani",
}

QUESTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "by",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "please",
    "should",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "would",
    "you",
    "your",
}

# Bootstrap aliases used only when DB-backed aliases are unavailable. Add large
# vocabularies to the retrieval_aliases table, not here.
FALLBACK_ALIAS_EXPANSIONS: dict[str, list[str]] = {
    "gpa": [
        "gpa",
        "grade point average",
        "grade point",
        "grading system",
        "course weight",
        "total score",
        "wastani wa alama",
        "alama za gpa",
        "alama za kozi",
        "uzito wa kozi",
    ],
    "cgpa": ["cgpa", "cumulative grade point average", "grade point average", "wastani wa jumla wa alama"],
    "ca": ["continuous assessment", "coursework", "alama za kazi za darasani", "alama za coursework"],
    "coursework": ["coursework", "continuous assessment", "ca", "alama za kazi za darasani"],
    "examination": ["examination", "exam", "university examination", "mtihani", "mitihani"],
    "postponement": [
        "postponement",
        "postpone",
        "deferment",
        "defer",
        "intermission",
        "kuahirisha",
        "ahirisha",
        "kughairi",
        "kughairisha",
        "kuahirisha masomo",
        "kughairi masomo",
        "kughairisha mwaka wa masomo",
        "kusitisha masomo",
    ],
    "registration": ["registration", "register", "student registration", "usajili", "kujisajili"],
    "appeal": ["appeal", "appeals", "rufaa", "kata rufaa"],
    "fee": ["fee", "fees", "tuition", "ada", "malipo"],
    "idit": [
        "idit",
        "idt",
        "instructional design and information technology",
        "bachelor of science in instructional design and information technology",
        "bachelor of science in instructional design & information technology",
        "dm074",
    ],
    "admission": ["admission", "admissions", "application", "udahili", "maombi"],
    "graduation": ["graduation", "graduate", "mahafali", "kuhitimu"],
    "transcript": ["transcript", "academic transcript", "nakala ya matokeo", "matokeo"],
    "dress code": ["dress code", "dressing code", "attire", "mavazi", "kanuni za mavazi"],
    "discipline": ["discipline", "conduct", "student conduct", "nidhamu", "maadili"],
    "sr": [
        "sr",
        "sr2",
        "student records",
        "student record",
        "student records system",
        "student registration",
        "student information system",
        "student portal",
        "mfumo wa taarifa za wanafunzi",
        "mfumo wa mwanafunzi",
    ],
    "sr2": [
        "sr2",
        "sr",
        "student records",
        "student records system",
        "student portal",
        "mfumo wa taarifa za wanafunzi",
    ],
    "tcu": ["tanzania commission for universities"],
    "nactvet": ["national council for technical and vocational education and training"],
}

CALCULATION_TERMS = {"calculate", "calculated", "calculating", "calculation", "computed", "computing"}


def clean_query_text(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip())


def tokenize(query: str) -> list[str]:
    return TOKEN_RE.findall(query.lower())


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def likely_swahili_query(query: str) -> bool:
    tokens = set(tokenize(query))
    return bool(tokens & SWAHILI_QUERY_TERMS)


def metadata_search_text(metadata: Mapping[str, object] | None) -> str:
    if not metadata:
        return ""
    return " ".join(str(metadata.get(key, "")) for key in SAFE_METADATA_TEXT_KEYS)


def significant_tokens(query: str) -> list[str]:
    tokens = []
    for token in tokenize(query):
        if token in DOMAIN_TERMS or token in QUESTION_STOPWORDS:
            continue
        if len(token) <= 1:
            continue
        tokens.append(token)
    return _dedupe(tokens)


def normalize_sparse_query(query: str) -> str:
    return " ".join(significant_tokens(query))


def build_sparse_query_variants(
    query: str,
    alias_expansions: Mapping[str, Sequence[str]] | None = None,
    max_expanded_terms: int = MAX_EXPANDED_TERMS,
) -> list[QueryVariant]:
    raw = clean_query_text(query)
    tokens = significant_tokens(raw)
    token_set = set(tokens)
    aliases = _matched_alias_terms(raw, tokens, alias_expansions or FALLBACK_ALIAS_EXPANSIONS)

    variants: list[QueryVariant] = []
    _add_variant(variants, "raw", raw, 1.0)

    normalized = " ".join(tokens)
    _add_variant(variants, "normalized", normalized, 0.98)

    expanded_terms = _limit_terms([*tokens, *aliases], max_expanded_terms)
    if "gpa" in token_set and (token_set & CALCULATION_TERMS):
        expanded_terms = _limit_terms(
            [
                *expanded_terms,
                "calculation",
                "calculated",
                "compute",
                "computed",
                "grade point average",
                "course weight",
                "total score",
            ],
            max_expanded_terms,
        )

    expanded_query = _or_query(expanded_terms)
    _add_variant(variants, "expanded", expanded_query, 0.92)

    fallback_terms = [*tokens, *aliases]
    if token_set & CALCULATION_TERMS:
        fallback_terms.extend(["calculation", "calculated", "computed"])

    fallback_query = _or_query(_limit_terms(fallback_terms, max_expanded_terms))
    _add_variant(variants, "fallback", fallback_query, 0.72)

    return variants


def _matched_alias_terms(
    query: str,
    tokens: list[str],
    alias_expansions: Mapping[str, Sequence[str]],
) -> list[str]:
    token_set = set(tokens)
    normalized_query = _token_phrase(query)
    matched_terms: list[str] = []

    for term, aliases in _normalize_alias_expansions(alias_expansions).items():
        single_word_aliases = {alias for alias in aliases if " " not in alias}
        phrase_aliases = [alias for alias in aliases if " " in alias]
        matched = (
            term in token_set
            or bool(single_word_aliases & token_set)
            or any(_token_phrase(phrase).strip() and _token_phrase(phrase) in normalized_query for phrase in phrase_aliases)
        )
        if matched:
            matched_terms.append(term)
            matched_terms.extend(aliases)

    return _dedupe(matched_terms)


def _normalize_alias_expansions(
    alias_expansions: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for raw_term, raw_aliases in alias_expansions.items():
        term = clean_query_text(str(raw_term)).lower()
        if not term:
            continue
        aliases: list[str] = []
        values = [raw_aliases] if isinstance(raw_aliases, str) else list(raw_aliases or [])
        for value in values:
            alias = clean_query_text(str(value)).lower()
            if alias:
                aliases.append(alias)
        output[term] = _dedupe(aliases)
    return output


def _or_query(terms: list[str]) -> str:
    formatted = []
    for term in _dedupe([item.strip() for item in terms if item and item.strip()]):
        if " " in term:
            formatted.append(f'"{term}"')
        else:
            formatted.append(term)
    return " OR ".join(formatted)


def _add_variant(variants: list[QueryVariant], label: str, query: str, weight: float) -> None:
    query = clean_query_text(query)
    if not query:
        return
    normalized = query.lower()
    if any(existing.query.lower() == normalized for existing in variants):
        return
    variants.append(QueryVariant(label=label, query=query, weight=weight))


def _limit_terms(items: list[str], limit: int) -> list[str]:
    if limit <= 0:
        return []
    return _dedupe(items)[:limit]


def _token_phrase(text: str) -> str:
    phrase = " ".join(tokenize(text))
    return f" {phrase} " if phrase else " "


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    output = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output
