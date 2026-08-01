from types import SimpleNamespace

from langchain_core.documents import Document

from app.services.retrieval_service import RetrievalService


def _service(enabled=True, top_k=5):
    service = object.__new__(RetrievalService)
    service.top_k = top_k
    service._settings = SimpleNamespace(
        adaptive_rerank_skip_high_confidence=enabled,
    )
    return service


def _result(
    *,
    sources=("dense", "sparse"),
    dense_rank=1,
    sparse_rank=1,
    rrf_score=0.032,
):
    return SimpleNamespace(
        retrieval_sources=list(sources),
        dense_rank=dense_rank,
        sparse_rank=sparse_rank,
        rrf_score=rrf_score,
    )


def test_adaptive_rerank_skips_when_top_result_is_strong_hybrid_match():
    service = _service()
    raw = [_result(), *[_result(rrf_score=0.02) for _ in range(5)]]

    assert service._should_skip_neural_rerank(raw) is True


def test_adaptive_rerank_keeps_jina_for_weak_or_single_source_matches():
    service = _service()

    assert service._should_skip_neural_rerank([
        _result(sources=("dense",), dense_rank=1, sparse_rank=None),
        *[_result(rrf_score=0.02) for _ in range(5)],
    ]) is False
    assert service._should_skip_neural_rerank([
        _result(dense_rank=8, sparse_rank=1),
        *[_result(rrf_score=0.02) for _ in range(5)],
    ]) is False
    assert service._should_skip_neural_rerank([
        _result(rrf_score=0.02),
        *[_result(rrf_score=0.02) for _ in range(5)],
    ]) is False


def test_retrieve_skips_alias_lookup_and_reranker_for_strong_match(monkeypatch):
    service = _service()
    raw = [_result(), *[_result(rrf_score=0.02) for _ in range(5)]]
    for index, item in enumerate(raw):
        item.text = f"Strong answer {index}"
        item.document_id = f"doc-{index}"
        item.chunk_id = f"chunk-{index}"
        item.metadata = {}

    class Hybrid:
        def retrieve_raw(self, *_args, **_kwargs):
            return raw

        def expand_neighbor_chunks(self, results):
            return results

    class Reranker:
        def rerank(self, *_args, **_kwargs):
            raise AssertionError("reranker should be skipped")

    def fail_alias_lookup(*_args, **_kwargs):
        raise AssertionError("alias lookup should be skipped")

    monkeypatch.setattr(
        "app.services.retrieval_service.AliasExpansionService",
        fail_alias_lookup,
    )
    service._settings.enable_reranker = True
    service._settings.top_k_dense = 20
    service._settings.top_k_sparse = 20
    service._hybrid = Hybrid()
    service._reranker = Reranker()

    docs = service.retrieve("gpa")

    assert len(docs) == 5
    assert isinstance(docs[0], Document)
    assert docs[0].metadata["reranker"] == "skipped_high_confidence"
