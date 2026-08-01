"""
app/services/sparse_retriever.py
----------------------------------
Sparse keyword retrieval using PostgreSQL Full-Text Search.

This retriever searches multiple query variants:
  - raw user query
  - normalized query with domain filler words removed
  - expanded keyword query for acronyms such as GPA
  - relaxed OR fallback query for partial matches
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.services.alias_expansion_service import AliasExpansionService
from app.services.query_normalization import QueryVariant, build_sparse_query_variants
from app.services.retrieval_metadata import document_metadata, faq_metadata

logger = logging.getLogger(__name__)


class SparseResult:
    """Lightweight result holder for a single FTS hit."""

    __slots__ = (
        "chunk_id",
        "document_id",
        "fts_score",
        "text",
        "metadata",
    )

    def __init__(
        self,
        chunk_id: str,
        document_id: str,
        fts_score: float,
        text: str,
        metadata: dict[str, Any],
    ) -> None:
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.fts_score = fts_score
        self.text = text
        self.metadata = metadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "fts_score": self.fts_score,
            "text": self.text,
            "metadata": self.metadata,
        }





class SparseRetriever:
    """
    Keyword retrieval via PostgreSQL FTS.

    websearch_to_tsquery is strict for normal text because terms are ANDed.
    To behave more like a production sparse retriever, this class tries
    normalized, expanded, and fallback variants, then merges them by score.
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    def retrieve(self, query: str, top_k: int = 20, filters: dict | None = None) -> list[SparseResult]:
        if not query or not query.strip():
            return []

        alias_expansions = AliasExpansionService(self.db).get_expansions(query)
        variants = build_sparse_query_variants(query, alias_expansions=alias_expansions)
        if not variants:
            return []

        merged: dict[str, SparseResult] = {}
        matched_queries: dict[str, list[dict[str, Any]]] = {}
        alias_boost_terms = self._alias_boost_terms(alias_expansions)

        for variant_index, variant in enumerate(variants):
            fetch_k = max(top_k * 3, 60) if variant.label in {"expanded", "fallback"} else top_k
            variant_results = self._retrieve_variant(variant, top_k=fetch_k, filters=filters)
            for rank, result in enumerate(variant_results, start=1):
                result.metadata["sparse_query_type"] = variant.label
                result.metadata["sparse_query"] = variant.query
                result.metadata["sparse_query_rank"] = rank
                result.metadata["sparse_query_weight"] = variant.weight
                alias_boost = self._alias_text_boost(result.text, alias_boost_terms)
                if alias_boost:
                    result.fts_score = round(result.fts_score + alias_boost, 6)
                    result.metadata["sparse_alias_boost"] = round(alias_boost, 6)

                matched_queries.setdefault(result.chunk_id, []).append(
                    {
                        "type": variant.label,
                        "query": variant.query,
                        "rank": rank,
                        "weight": variant.weight,
                    }
                )

                existing = merged.get(result.chunk_id)
                if existing is None:
                    merged[result.chunk_id] = result
                    continue

                existing_rank = existing.metadata.get("sparse_query_rank", 10**9)
                existing_variant_index = self._variant_order(
                    existing.metadata.get("sparse_query_type"),
                    variants,
                )
                should_replace = (
                    result.fts_score > existing.fts_score
                    or (
                        result.fts_score == existing.fts_score
                        and (variant_index, rank, result.chunk_id) < (existing_variant_index, existing_rank, existing.chunk_id)
                    )
                )
                if should_replace:
                    merged[result.chunk_id] = result

        for chunk_id, result in merged.items():
            result.metadata["matched_sparse_queries"] = matched_queries.get(chunk_id, [])

        results = sorted(
            merged.values(),
            key=lambda item: (
                -item.fts_score,
                self._variant_order(item.metadata.get("sparse_query_type"), variants),
                item.metadata.get("sparse_query_rank", 10**9),
                item.document_id,
                item.chunk_id,
            ),
        )
        return results[:top_k]

    def _alias_boost_terms(self, alias_expansions: dict[str, list[str]]) -> list[str]:
        generic_keys = {"fee", "fees", "tuition", "registration", "student"}
        terms: list[str] = []
        for key, aliases in alias_expansions.items():
            if key.lower() in generic_keys:
                continue
            for value in [key, *aliases]:
                clean_value = str(value or "").strip().lower()
                if not clean_value or clean_value in terms:
                    continue
                if len(clean_value) >= 4 or any(char.isdigit() for char in clean_value):
                    terms.append(clean_value)
        return terms

    def _alias_text_boost(self, text_value: str, alias_terms: list[str]) -> float:
        if not alias_terms:
            return 0.0
        text_lower = str(text_value or "").lower()
        boost = 0.0
        for term in alias_terms:
            if term in text_lower:
                boost += 0.02 if " " not in term else 0.04
        return min(boost, 0.08)

    def _retrieve_variant(
        self,
        variant: QueryVariant,
        top_k: int,
        filters: dict | None,
    ) -> list[SparseResult]:
        params: dict[str, Any] = {"query": variant.query, "top_k": top_k}
        prog_clause = ""
        year_clause = ""

        if filters:
            programme = filters.get("programme")
            year = filters.get("year")
            if programme:
                prog_clause = "AND (dc.metadata->>'programme' = :programme OR dc.metadata->>'programme' IS NULL)"
                params["programme"] = str(programme)
            if year:
                year_clause = "AND (dc.metadata->>'year' = :year OR dc.metadata->>'year' IS NULL)"
                params["year"] = str(year)

        # Single UNION ALL merges document_chunks and faqs in one DB round-trip
        # instead of two separate queries, halving the number of DB calls from
        # 8 (4 variants × 2 queries) to 4 (4 variants × 1 query).
        union_sql = text(f"""
            WITH search_query AS (
                SELECT
                    websearch_to_tsquery('simple', :query) AS tsq_s,
                    websearch_to_tsquery('english', :query) AS tsq_e
            ),
            doc_hits AS (
                SELECT
                    dc.id::text                                     AS chunk_id,
                    d.id::text                                      AS document_id,
                    d.filename                                      AS source,
                    d.title                                         AS document_title,
                    d.source_url                                    AS source_url,
                    d.status                                        AS document_status,
                    d.content_hash                                  AS content_hash,
                    d.upload_date                                   AS document_uploaded_at,
                    dc.chunk_text                                   AS text,
                    dc.chunk_index                                  AS chunk_index,
                    dc.page_number                                  AS page_number,
                    dc.metadata                                     AS metadata_json,
                    NULL::text                                      AS category,
                    'doc'::text                                     AS row_type,
                    GREATEST(
                        ts_rank_cd(dc.tsv, search_query.tsq_s, 34),
                        ts_rank_cd(dc.tsv, search_query.tsq_e, 34)
                    )                                               AS fts_score
                FROM document_chunks dc
                JOIN documents d ON d.id = dc.document_id
                CROSS JOIN search_query
                WHERE (dc.tsv @@ search_query.tsq_s OR dc.tsv @@ search_query.tsq_e)
                  AND d.status = 'active'
                  {prog_clause}
                  {year_clause}
            ),
            faq_hits AS (
                SELECT
                    f.id::text                                      AS chunk_id,
                    f.id::text                                      AS document_id,
                    COALESCE('FAQ - ' || f.category, 'FAQ')        AS source,
                    NULL::text                                      AS document_title,
                    NULL::text                                      AS source_url,
                    NULL::text                                      AS document_status,
                    NULL::text                                      AS content_hash,
                    NULL::timestamptz                               AS document_uploaded_at,
                    f.question || CHR(10) || CHR(10) || f.answer   AS text,
                    0                                               AS chunk_index,
                    NULL::integer                                   AS page_number,
                    f.metadata                                      AS metadata_json,
                    f.category                                      AS category,
                    'faq'::text                                     AS row_type,
                    GREATEST(
                        ts_rank_cd(f.fts_vector, search_query.tsq_s, 34),
                        ts_rank_cd(f.fts_vector, search_query.tsq_e, 34)
                    )                                               AS fts_score
                FROM faqs f
                CROSS JOIN search_query
                WHERE f.is_active = true
                  AND (f.fts_vector @@ search_query.tsq_s OR f.fts_vector @@ search_query.tsq_e)
            )
            SELECT * FROM (
                SELECT * FROM doc_hits
                UNION ALL
                SELECT * FROM faq_hits
            ) combined
            ORDER BY fts_score DESC, chunk_id ASC
            LIMIT :top_k
        """)

        try:
            all_rows = self.db.execute(union_sql, params).fetchall()
        except Exception as exc:
            self.db.rollback()
            logger.warning(
                "SparseRetriever FTS query failed for %s query %r: %s",
                variant.label,
                variant.query,
                exc,
            )
            return []

        results: list[SparseResult] = []
        for row in all_rows:
            raw_score = float(row.fts_score or 0.0)
            adjusted_score = round(raw_score * variant.weight, 6)
            if row.row_type == "doc":
                metadata = document_metadata(
                    row.metadata_json,
                    source=row.source,
                    chunk_index=row.chunk_index,
                    page_number=row.page_number,
                    document_title=row.document_title,
                    source_url=row.source_url,
                    document_status=row.document_status,
                    content_hash=row.content_hash,
                    document_uploaded_at=row.document_uploaded_at,
                    extra={"fts_score_raw": round(raw_score, 6)},
                )
                results.append(
                    SparseResult(
                        chunk_id=str(row.chunk_id),
                        document_id=str(row.document_id),
                        fts_score=adjusted_score,
                        text=row.text,
                        metadata=metadata,
                    )
                )
            else:
                metadata = faq_metadata(
                    row.metadata_json,
                    faq_id=str(row.document_id),
                    category=row.category,
                    source=row.source,
                    extra={"fts_score_raw": round(raw_score, 6)},
                )
                results.append(
                    SparseResult(
                        chunk_id=str(row.chunk_id),
                        document_id=str(row.document_id),
                        fts_score=adjusted_score,
                        text=row.text,
                        metadata=metadata,
                    )
                )

        return results

    def _variant_order(self, label: Any, variants: list[QueryVariant]) -> int:
        for index, variant in enumerate(variants):
            if variant.label == label:
                return index
        return len(variants)
