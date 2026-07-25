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


_FTS_FAQ_QUERY = text(
    """
    WITH search_query AS (
        SELECT
            websearch_to_tsquery('simple', :query) AS tsq_s,
            websearch_to_tsquery('english', :query) AS tsq_e
    )
    SELECT
        f.id::text                                      AS chunk_id,
        f.id::text                                      AS document_id,
        COALESCE('FAQ - ' || f.category, 'FAQ')         AS source,
        f.question || CHR(10) || CHR(10) || f.answer     AS text,
        0                                               AS chunk_index,
        f.category                                      AS category,
        f.metadata                                      AS metadata_json,
        GREATEST(
            ts_rank_cd(f.fts_vector, search_query.tsq_s, 34),
            ts_rank_cd(f.fts_vector, search_query.tsq_e, 34)
        )                                               AS fts_score
    FROM faqs f
    CROSS JOIN search_query
    WHERE f.is_active = true
      AND (f.fts_vector @@ search_query.tsq_s OR f.fts_vector @@ search_query.tsq_e)
    ORDER BY fts_score DESC, f.id ASC
    LIMIT :top_k
    """
)


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

        for variant_index, variant in enumerate(variants):
            variant_results = self._retrieve_variant(variant, top_k=top_k, filters=filters)
            for rank, result in enumerate(variant_results, start=1):
                result.metadata["sparse_query_type"] = variant.label
                result.metadata["sparse_query"] = variant.query
                result.metadata["sparse_query_rank"] = rank
                result.metadata["sparse_query_weight"] = variant.weight

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

        fts_sql = text(f"""
            WITH search_query AS (
                SELECT
                    websearch_to_tsquery('simple', :query) AS tsq_s,
                    websearch_to_tsquery('english', :query) AS tsq_e
            )
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
            ORDER BY fts_score DESC, dc.id ASC
            LIMIT :top_k
        """)

        try:
            rows = self.db.execute(fts_sql, params).fetchall()
            faq_rows = self.db.execute(
                _FTS_FAQ_QUERY,
                {"query": variant.query, "top_k": top_k},
            ).fetchall()
        except Exception as exc:
            logger.warning(
                "SparseRetriever FTS query failed for %s query %r: %s",
                variant.label,
                variant.query,
                exc,
            )
            return []

        combined = []
        for row in rows:
            combined.append((row.fts_score, row, "doc"))
        for row in faq_rows:
            combined.append((row.fts_score, row, "faq"))

        combined.sort(key=lambda item: (-float(item[0]), str(item[1].chunk_id)))
        combined = combined[:top_k]

        results: list[SparseResult] = []
        for score, row, type_ in combined:
            raw_score = float(score or 0.0)
            adjusted_score = round(raw_score * variant.weight, 6)
            if type_ == "doc":
                metadata = dict(row.metadata_json or {})
                metadata.update(
                    {
                        "source": row.source or "database",
                        "chunk_index": row.chunk_index,
                        "source_type": "document",
                        "fts_score_raw": round(raw_score, 6),
                    }
                )
                if row.page_number is not None:
                    metadata["page_number"] = row.page_number
                if row.document_title:
                    metadata["document_title"] = row.document_title
                if row.source_url:
                    metadata["source_url"] = row.source_url
                metadata["document_status"] = row.document_status or "active"
                if row.content_hash:
                    metadata["content_hash"] = row.content_hash
                if row.document_uploaded_at:
                    metadata["document_uploaded_at"] = row.document_uploaded_at.isoformat()

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
                metadata = dict(row.metadata_json or {})
                metadata.update(
                    {
                        "source": row.source,
                        "chunk_index": 0,
                        "source_type": "faq",
                        "faq_id": str(row.document_id),
                        "category": row.category,
                        "fts_score_raw": round(raw_score, 6),
                    }
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
