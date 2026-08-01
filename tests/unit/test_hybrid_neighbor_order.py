from app.core.config import settings
from app.db.models import DocumentChunk, DocumentModel
from app.services.hybrid_retriever import HybridRetriever
from app.services.rrf import RRFResult


def test_neighbor_expansion_keeps_ranked_hits_before_neighbors(db_session):
    document = DocumentModel(
        title="Student Rules",
        filename="student-rules.pdf",
        status="active",
    )
    db_session.add(document)
    db_session.flush()

    rows = []
    for index in range(6):
        row = DocumentChunk(
            document_id=document.id,
            chunk_index=index,
            chunk_text=f"Rule chunk {index}",
            embedding=[0.0] * settings.EMBEDDING_DIMENSION,
            metadata_={},
        )
        db_session.add(row)
        rows.append(row)
    db_session.flush()

    def result(index: int, score: float) -> RRFResult:
        return RRFResult(
            chunk_id=str(rows[index].id),
            document_id=str(document.id),
            rrf_score=score,
            dense_rank=index,
            sparse_rank=index,
            retrieval_sources=["dense", "sparse"],
            metadata={
                "source_type": "document",
                "chunk_index": index,
            },
            text=rows[index].chunk_text,
        )

    first = result(1, 0.03)
    second = result(4, 0.02)
    retriever = HybridRetriever(
        db_session,
        top_k=2,
        dense_top_k=2,
        sparse_top_k=2,
        rrf_k=60,
        neighbor_window=1,
    )

    expanded = retriever.expand_neighbor_chunks([first, second])

    assert [item.chunk_id for item in expanded[:2]] == [first.chunk_id, second.chunk_id]
    assert all(item.metadata.get("neighbor_expansion") for item in expanded[2:])

