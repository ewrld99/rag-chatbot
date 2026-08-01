"""
app/services/reranker_service.py
---------------------------------
Reranks RRFResult candidates against the user query.

Uses Jina Reranker when JINA_API_KEY is configured. If Jina is unavailable or
fails, it falls back to a deterministic local lexical reranker. The local
fallback can use DB-backed alias expansions, so terms such as SR and SR2 are
scored as equivalent during fallback reranking.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
import logging
from typing import List

import httpx

from app.core.config import settings
from app.services.jina_resilience import (
    JinaProviderCooldownError,
    is_jina_account_failure,
    jina_provider_circuit,
)
from app.services.query_normalization import metadata_search_text, significant_tokens, token_set, tokenize
from app.services.rrf import RRFResult
from app.services.tls_service import system_ssl_context

logger = logging.getLogger(__name__)

_JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
_JINA_RERANK_MODEL = "jina-reranker-v2-base-multilingual"
_REQUEST_TIMEOUT = 10.0


@lru_cache(maxsize=1)
def get_reranker_client() -> httpx.Client:
    return httpx.Client(
        timeout=_REQUEST_TIMEOUT,
        verify=system_ssl_context(),
    )


def close_reranker_client() -> None:
    if get_reranker_client.cache_info().currsize:
        get_reranker_client().close()
        get_reranker_client.cache_clear()


class RerankService:
    """Blended Jina, lexical, and retrieval-rank reranker."""

    def __init__(self) -> None:
        if settings.JINA_API_KEY:
            self._headers = {
                "Authorization": f"Bearer {settings.JINA_API_KEY}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        else:
            self._headers = None
            logger.warning(
                "RerankService: JINA_API_KEY is not set. "
                "Reranker will use local lexical fallback."
            )

    @property
    def is_available(self) -> bool:
        return self._headers is not None and not jina_provider_circuit.is_open

    def rerank(
        self,
        query: str,
        candidates: List[RRFResult],
        top_k: int = 5,
        alias_expansions: Mapping[str, Sequence[str]] | None = None,
    ) -> List[RRFResult]:
        """Return candidates ordered by multiple relevance signals."""
        if not candidates or top_k <= 0:
            return []

        if not self._headers:
            return self._lexical_rerank(
                query,
                candidates,
                top_k,
                alias_expansions=alias_expansions,
            )

        try:
            jina_provider_circuit.before_call()
        except JinaProviderCooldownError:
            logger.debug(
                "RerankService: skipping Jina during provider cooldown; "
                "using local lexical fallback."
            )
            return self._lexical_rerank(
                query,
                candidates,
                top_k,
                alias_expansions=alias_expansions,
            )

        try:
            payload = {
                "model": _JINA_RERANK_MODEL,
                "query": query,
                "documents": [candidate.text for candidate in candidates],
                # Scores for the full pool are needed to blend Jina with RRF,
                # lexical relevance, and sparse/dense rank signals.
                "top_n": len(candidates),
            }

            response = get_reranker_client().post(
                _JINA_RERANK_URL,
                headers=self._headers,
                json=payload,
            )
            response.raise_for_status()

            jina_provider_circuit.record_success()
            jina_scores: dict[int, float] = {}
            for item in response.json().get("results", []):
                index = int(item["index"])
                if 0 <= index < len(candidates):
                    jina_scores[index] = float(item["relevance_score"])

            if not jina_scores:
                raise ValueError("Jina reranker returned no candidate scores")

            reranked = self._blend_jina_with_retrieval(
                query,
                candidates,
                jina_scores,
                top_k,
                alias_expansions=alias_expansions,
            )

            logger.debug(
                "RerankService: blended %d candidates -> %d for query %r",
                len(candidates),
                len(reranked),
                query[:80],
            )
            return reranked
        except Exception as exc:
            if is_jina_account_failure(exc):
                jina_provider_circuit.record_account_failure()
                logger.warning(
                    "RerankService: Jina authorization failed or its account "
                    "balance is insufficient. Using local lexical fallback."
                )
            else:
                logger.warning(
                    "RerankService: Jina Reranker API call failed (%s). "
                    "Using local lexical fallback.",
                    exc,
                )
            return self._lexical_rerank(
                query,
                candidates,
                top_k,
                alias_expansions=alias_expansions,
            )

    def _blend_jina_with_retrieval(
        self,
        query: str,
        candidates: List[RRFResult],
        jina_scores: Mapping[int, float],
        top_k: int,
        alias_expansions: Mapping[str, Sequence[str]] | None = None,
    ) -> List[RRFResult]:
        """
        Blend semantic reranking with deterministic retrieval evidence.

        A neural reranker is a useful signal, but it must not be able to erase
        an exact FTS match. The strongest sparse candidate is therefore kept as
        an anchor even when Jina assigns it a low score.
        """
        lexical_ranked = self._lexical_rerank(
            query,
            candidates,
            len(candidates),
            alias_expansions=alias_expansions,
        )
        lexical_scores = {
            candidate.chunk_id: float(candidate.metadata.get("rerank_score", 0.0))
            for candidate in lexical_ranked
        }
        max_lexical = max(lexical_scores.values(), default=1.0) or 1.0
        max_rrf = max((candidate.rrf_score for candidate in candidates), default=1.0) or 1.0

        scored: list[tuple[float, int, str, str, RRFResult]] = []
        by_chunk_id: dict[str, tuple[float, int, str, str, RRFResult]] = {}

        for index, candidate in enumerate(candidates):
            jina_score = max(0.0, min(1.0, float(jina_scores.get(index, 0.0))))
            lexical_score = lexical_scores.get(candidate.chunk_id, 0.0)
            lexical_normalized = max(0.0, lexical_score / max_lexical)
            rrf_normalized = max(0.0, candidate.rrf_score / max_rrf)
            rank_signal = max(
                self._reciprocal_rank(candidate.dense_rank),
                self._reciprocal_rank(candidate.sparse_rank),
            )

            blended_score = (
                (0.35 * jina_score)
                + (0.40 * lexical_normalized)
                + (0.15 * rrf_normalized)
                + (0.10 * rank_signal)
            )
            candidate.metadata["jina_rerank_score"] = round(jina_score, 4)
            candidate.metadata["lexical_rerank_score"] = round(lexical_score, 4)
            candidate.metadata["rerank_score"] = round(blended_score, 4)
            candidate.metadata["reranker"] = "jina"
            candidate.metadata["rerank_strategy"] = "blended"

            item = (
                blended_score,
                index,
                candidate.document_id,
                candidate.chunk_id,
                candidate,
            )
            scored.append(item)
            by_chunk_id[candidate.chunk_id] = item

        scored.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
        selected = scored[:top_k]

        sparse_candidates = [
            candidate
            for candidate in candidates
            if candidate.sparse_rank is not None
        ]
        if sparse_candidates and selected:
            sparse_anchor = min(
                sparse_candidates,
                key=lambda candidate: (
                    candidate.sparse_rank,
                    -candidate.rrf_score,
                    candidate.document_id,
                    candidate.chunk_id,
                ),
            )
            if all(item[4].chunk_id != sparse_anchor.chunk_id for item in selected):
                selected[-1] = by_chunk_id[sparse_anchor.chunk_id]
                selected.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))

        return [item[4] for item in selected]

    @staticmethod
    def _reciprocal_rank(rank: int | None) -> float:
        return 1.0 / rank if isinstance(rank, int) and rank > 0 else 0.0

    def _lexical_rerank(
        self,
        query: str,
        candidates: List[RRFResult],
        top_k: int,
        alias_expansions: Mapping[str, Sequence[str]] | None = None,
    ) -> List[RRFResult]:
        """Deterministic local fallback reranker used when Jina is unavailable."""
        query_terms = set(significant_tokens(query))
        if not query_terms:
            query_terms = {term for term in tokenize(query) if len(term) > 2}

        alias_single_terms, alias_phrases = self._alias_terms(alias_expansions or {})
        query_terms.update(alias_single_terms)

        if not query_terms and not alias_phrases:
            return candidates[:top_k]

        scored: list[tuple[float, int, str, str, RRFResult]] = []

        for index, candidate in enumerate(candidates):
            haystack = f"{candidate.text} {metadata_search_text(candidate.metadata)}".lower()
            content_tokens = token_set(haystack)

            overlap = len(query_terms & content_tokens)
            phrase_bonus = sum(2.0 for phrase in alias_phrases if phrase in haystack)

            if "gpa" in query_terms and "grade point average" in haystack:
                phrase_bonus += 3.0
            if {"calculate", "calculated", "calculating", "calculation", "compute", "computed"} & query_terms:
                if any(term in haystack for term in ("calculate", "calculated", "calculating", "calculation", "compute", "computed")):
                    phrase_bonus += 2.0
            if "gpa" in query_terms and "gpa" in content_tokens:
                phrase_bonus += 1.0

            neighbor_penalty = 0.05 if candidate.metadata.get("neighbor_expansion") else 0.0
            lexical_score = candidate.rrf_score + overlap + phrase_bonus - neighbor_penalty
            scored.append((lexical_score, index, candidate.document_id, candidate.chunk_id, candidate))

        scored.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
        for score, _, _, _, candidate in scored[:top_k]:
            candidate.metadata["rerank_score"] = round(score, 4)
            candidate.metadata["reranker"] = "local_lexical"

        return [candidate for _, _, _, _, candidate in scored[:top_k]]

    def _alias_terms(
        self,
        alias_expansions: Mapping[str, Sequence[str]],
    ) -> tuple[set[str], set[str]]:
        single_terms: set[str] = set()
        phrases: set[str] = set()

        for term, aliases in alias_expansions.items():
            values = [term, *list(aliases or [])]
            for value in values:
                tokens = tokenize(str(value))
                if not tokens:
                    continue
                if len(tokens) == 1:
                    single_terms.add(tokens[0])
                else:
                    phrases.add(" ".join(tokens))

        return single_terms, phrases
