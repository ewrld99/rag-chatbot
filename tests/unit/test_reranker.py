import pytest

from app.services.jina_resilience import jina_provider_circuit
from app.services.reranker_service import RerankService
from app.services.rrf import RRFResult


@pytest.fixture(autouse=True)
def reset_jina_circuit():
    jina_provider_circuit.reset()
    yield
    jina_provider_circuit.reset()


class _FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "results": [
                {"index": 0, "relevance_score": 0.95},
                {"index": 1, "relevance_score": 0.05},
            ]
        }


class _FakeClient:
    def __init__(self, *args, **kwargs):
        self.payload = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def post(self, *args, **kwargs):
        self.payload = kwargs["json"]
        assert self.payload["top_n"] == 2
        return _FakeResponse()


def _candidate(
    chunk_id: str,
    text: str,
    rrf_score: float,
    dense_rank: int | None,
    sparse_rank: int | None,
) -> RRFResult:
    sources = []
    if dense_rank is not None:
        sources.append("dense")
    if sparse_rank is not None:
        sources.append("sparse")
    return RRFResult(
        chunk_id=chunk_id,
        document_id=f"document-{chunk_id}",
        rrf_score=rrf_score,
        dense_rank=dense_rank,
        sparse_rank=sparse_rank,
        retrieval_sources=sources,
        text=text,
    )


def test_blended_reranker_preserves_strong_sparse_evidence(monkeypatch):
    from app.services import reranker_service

    monkeypatch.setattr(reranker_service.httpx, "Client", _FakeClient)
    service = RerankService()
    service._headers = {"Authorization": "Bearer test"}

    weak_dense = _candidate(
        "dense",
        "UDOM credit transfer course equivalency and grade conversion.",
        rrf_score=0.020,
        dense_rank=1,
        sparse_rank=None,
    )
    exact_sparse = _candidate(
        "sparse",
        "The GPA is calculated by dividing the total weighted grade point score "
        "by the total course credits.",
        rrf_score=0.016,
        dense_rank=None,
        sparse_rank=1,
    )

    results = service.rerank(
        "How is GPA calculated at UDOM?",
        [weak_dense, exact_sparse],
        top_k=1,
        alias_expansions={
            "gpa": ["grade point average", "course weight", "total score"],
        },
    )

    assert [result.chunk_id for result in results] == ["sparse"]
    assert results[0].metadata["rerank_strategy"] == "blended"
    assert results[0].metadata["jina_rerank_score"] == 0.05


def test_sparse_anchor_survives_even_when_blended_score_is_lower(monkeypatch):
    from app.services import reranker_service

    monkeypatch.setattr(reranker_service.httpx, "Client", _FakeClient)
    service = RerankService()
    service._headers = {"Authorization": "Bearer test"}

    weak_dense = _candidate(
        "dense",
        "A semantically plausible result.",
        rrf_score=0.020,
        dense_rank=1,
        sparse_rank=None,
    )
    sparse_anchor = _candidate(
        "sparse",
        "An exact keyword result.",
        rrf_score=0.010,
        dense_rank=None,
        sparse_rank=1,
    )

    results = service.rerank(
        "A semantically plausible result",
        [weak_dense, sparse_anchor],
        top_k=1,
    )

    assert [result.chunk_id for result in results] == ["sparse"]


def test_jina_cooldown_uses_local_reranker_without_http_call(monkeypatch):
    from app.services import reranker_service

    class FailClient:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("Jina must not be called during cooldown.")

    monkeypatch.setattr(reranker_service.httpx, "Client", FailClient)
    service = RerankService()
    service._headers = {"Authorization": "Bearer test"}
    jina_provider_circuit.record_account_failure()

    result = service.rerank(
        "How is GPA calculated?",
        [
            _candidate(
                "sparse",
                "GPA is calculated from grade points and course credits.",
                rrf_score=0.016,
                dense_rank=None,
                sparse_rank=1,
            )
        ],
        top_k=1,
    )

    assert result[0].metadata["reranker"] == "local_lexical"
