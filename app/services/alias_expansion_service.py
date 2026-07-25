"""
app/services/alias_expansion_service.py
----------------------------------------
Database-backed retrieval vocabulary for acronyms and synonyms.

The service loads active aliases from retrieval_aliases and returns only aliases
that are relevant to the current query. This keeps query expansion cheap even
when the table grows to thousands of UDOM-specific terms.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from sqlalchemy.orm import Session

from app.db.models import RetrievalAlias
from app.services.query_normalization import (
    FALLBACK_ALIAS_EXPANSIONS,
    clean_query_text,
    tokenize,
)

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 300
MAX_ALIASES_PER_MATCH = 12


@dataclass(frozen=True)
class AliasRecord:
    term: str
    aliases: tuple[str, ...]
    category: str | None
    weight: float


class AliasExpansionService:
    """Loads and matches DB-backed retrieval aliases."""

    _cache_expires_at: float = 0.0
    _cache_records: tuple[AliasRecord, ...] = ()

    def __init__(self, db: Session) -> None:
        self.db = db

    def get_expansions(
        self,
        query: str,
        max_aliases_per_match: int = MAX_ALIASES_PER_MATCH,
    ) -> dict[str, list[str]]:
        tokens = set(tokenize(query))
        if not tokens:
            return {}

        normalized_query = self._token_phrase(query)
        expansions = self._fallback_matches(tokens, normalized_query, max_aliases_per_match)

        for record in self._load_records():
            matching_keys = self._matching_keys(record, tokens, normalized_query)
            if not matching_keys:
                continue

            values = [record.term, *record.aliases]
            for key in matching_keys:
                destination = expansions.setdefault(key, [])
                for value in values:
                    if len(destination) >= max_aliases_per_match:
                        break
                    if value.lower() == key.lower():
                        continue
                    if value not in destination:
                        destination.append(value)

        return expansions

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache_expires_at = 0.0
        cls._cache_records = ()

    def _load_records(self) -> tuple[AliasRecord, ...]:
        now = time.monotonic()
        if self.__class__._cache_records and self.__class__._cache_expires_at > now:
            return self.__class__._cache_records

        try:
            rows = (
                self.db.query(RetrievalAlias)
                .filter(RetrievalAlias.is_active == True)
                .order_by(RetrievalAlias.term.asc())
                .all()
            )
        except Exception as exc:
            logger.warning("AliasExpansionService: failed to load retrieval aliases: %s", exc)
            return ()

        records = tuple(self._row_to_record(row) for row in rows)
        self.__class__._cache_records = records
        self.__class__._cache_expires_at = now + CACHE_TTL_SECONDS
        return records

    def _row_to_record(self, row: RetrievalAlias) -> AliasRecord:
        aliases = row.aliases or []
        if not isinstance(aliases, list):
            aliases = []
        clean_aliases = []
        for alias in aliases:
            clean_alias = clean_query_text(str(alias)).lower()
            if clean_alias:
                clean_aliases.append(clean_alias)

        return AliasRecord(
            term=clean_query_text(row.term or "").lower(),
            aliases=tuple(dict.fromkeys(clean_aliases)),
            category=row.category,
            weight=float(row.weight or 1.0),
        )

    def _matching_keys(
        self,
        record: AliasRecord,
        tokens: set[str],
        normalized_query: str,
    ) -> list[str]:
        keys: list[str] = []
        if record.term in tokens:
            keys.append(record.term)

        for alias in record.aliases:
            if " " in alias:
                alias_phrase = self._token_phrase(alias)
                if alias_phrase.strip() and alias_phrase in normalized_query:
                    keys.append(record.term)
                continue
            if alias in tokens:
                keys.append(alias)

        return list(dict.fromkeys(keys))

    def _token_phrase(self, text: str) -> str:
        phrase = " ".join(tokenize(text))
        return f" {phrase} " if phrase else " "

    def _fallback_matches(
        self,
        tokens: set[str],
        normalized_query: str,
        max_aliases_per_match: int,
    ) -> dict[str, list[str]]:
        expansions: dict[str, list[str]] = {}
        for term, aliases in FALLBACK_ALIAS_EXPANSIONS.items():
            normalized_aliases = [clean_query_text(alias).lower() for alias in aliases]
            matched = (
                term in tokens
                or any(alias in tokens for alias in normalized_aliases if " " not in alias)
                or any(self._token_phrase(alias).strip() and self._token_phrase(alias) in normalized_query for alias in normalized_aliases if " " in alias)
            )
            if matched:
                expansions[term] = normalized_aliases[:max_aliases_per_match]
        return expansions
