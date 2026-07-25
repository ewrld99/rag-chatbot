"""
app/services/rrf.py
--------------------
Reciprocal Rank Fusion (RRF) implementation.

Formula
-------
    score(d) = sum(1 / (k + rank(d, r))) for every retriever r

where rank is 1-based (rank 1 = best result).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from app.services.dense_retriever import DenseResult
from app.services.sparse_retriever import SparseResult


@dataclass
class RRFResult:
    chunk_id: str
    document_id: str
    rrf_score: float
    dense_rank: int | None
    sparse_rank: int | None
    retrieval_sources: list[Literal["dense", "sparse"]]
    metadata: dict[str, Any] = field(default_factory=dict)
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "rrf_score": round(self.rrf_score, 6),
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "retrieval_sources": self.retrieval_sources,
            "metadata": self.metadata,
            "text": self.text,
        }


class ReciprocalRankFusion:
    """
    Fuses dense and sparse ranked lists into one deterministic ranking.
    """

    def fuse(
        self,
        dense_results: list[DenseResult],
        sparse_results: list[SparseResult],
        k: int = 60,
    ) -> list[RRFResult]:
        if k < 0:
            raise ValueError("RRF k must be greater than or equal to 0")

        acc: dict[str, dict[str, Any]] = {}

        def ensure(chunk_id: str, text: str, document_id: str, metadata: dict[str, Any]) -> None:
            if chunk_id not in acc:
                acc[chunk_id] = {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "sparse_rank": None,
                    "retrieval_sources": [],
                    "metadata": dict(metadata or {}),
                    "text": text or "",
                }

        seen_dense: set[str] = set()
        for rank, result in enumerate(dense_results, start=1):
            chunk_id = str(result.chunk_id)
            if chunk_id in seen_dense:
                continue
            seen_dense.add(chunk_id)

            ensure(chunk_id, result.text, str(result.document_id), result.metadata)
            acc[chunk_id]["metadata"].update(result.metadata or {})
            acc[chunk_id]["metadata"]["dense_similarity_score"] = result.similarity_score
            acc[chunk_id]["rrf_score"] += 1.0 / (k + rank)
            acc[chunk_id]["dense_rank"] = rank
            if "dense" not in acc[chunk_id]["retrieval_sources"]:
                acc[chunk_id]["retrieval_sources"].append("dense")

        seen_sparse: set[str] = set()
        for rank, result in enumerate(sparse_results, start=1):
            chunk_id = str(result.chunk_id)
            if chunk_id in seen_sparse:
                continue
            seen_sparse.add(chunk_id)

            ensure(chunk_id, result.text, str(result.document_id), result.metadata)
            acc[chunk_id]["metadata"].update(result.metadata or {})
            acc[chunk_id]["metadata"]["sparse_fts_score"] = result.fts_score
            acc[chunk_id]["rrf_score"] += 1.0 / (k + rank)
            acc[chunk_id]["sparse_rank"] = rank
            if "sparse" not in acc[chunk_id]["retrieval_sources"]:
                acc[chunk_id]["retrieval_sources"].append("sparse")

        merged = sorted(
            acc.values(),
            key=lambda item: (
                -item["rrf_score"],
                item["dense_rank"] if item["dense_rank"] is not None else float("inf"),
                item["sparse_rank"] if item["sparse_rank"] is not None else float("inf"),
                item["document_id"],
                item["chunk_id"],
            ),
        )

        return [
            RRFResult(
                chunk_id=item["chunk_id"],
                document_id=item["document_id"],
                rrf_score=item["rrf_score"],
                dense_rank=item["dense_rank"],
                sparse_rank=item["sparse_rank"],
                retrieval_sources=item["retrieval_sources"],
                metadata=item["metadata"],
                text=item["text"],
            )
            for item in merged
        ]
