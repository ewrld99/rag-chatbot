from app.core.config import settings
from app.db.models import DocumentChunk, DocumentModel


def test_document_status_and_quality_sync_chunk_retrievability(db_session):
    document = DocumentModel(
        title="Retrievability trigger test",
        filename="retrievability-trigger.pdf",
        status="active",
        quality_status="passed",
    )
    db_session.add(document)
    db_session.flush()
    chunk = DocumentChunk(
        document_id=document.id,
        chunk_index=0,
        chunk_text="Official regulation text.",
        embedding=[0.0] * settings.EMBEDDING_DIMENSION,
        metadata_={},
    )
    db_session.add(chunk)
    db_session.flush()
    db_session.refresh(chunk)
    assert chunk.is_retrievable is True

    document.quality_status = "review"
    db_session.flush()
    db_session.refresh(chunk)
    assert chunk.is_retrievable is False

    document.quality_status = "overridden"
    document.status = "active"
    db_session.flush()
    db_session.refresh(chunk)
    assert chunk.is_retrievable is True

    document.status = "needs_reindex"
    db_session.flush()
    db_session.refresh(chunk)
    assert chunk.is_retrievable is False
