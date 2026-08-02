from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pytest

from app.db.models import DocumentModel, FileOperation
from app.services.file_storage_service import (
    FileStorageError,
    FileStorageService,
    UploadTooLargeError,
)
from app.services.storage_reconciler import _finish_deletion, _finish_promotion


def test_storage_stages_hashes_and_promotes_atomically(tmp_path):
    storage = FileStorageService(tmp_path)
    staged = storage.stage_stream(BytesIO(b"document body"), "Rules.PDF", max_bytes=100)

    assert staged.sha256 == "21e251f136ac7d866c371c2b426d4bc4b2821470a82e8b20a703b32a3c939136"
    assert Path(staged.source_path).is_file()
    storage.promote(staged.source_path, staged.target_path)
    assert not Path(staged.source_path).exists()
    assert Path(staged.target_path).read_bytes() == b"document body"


def test_oversized_stage_leaves_no_artifact(tmp_path):
    storage = FileStorageService(tmp_path)

    with pytest.raises(UploadTooLargeError):
        storage.stage_stream(BytesIO(b"too large"), "large.txt", max_bytes=3)

    assert list(storage.staging_root.iterdir()) == []


def test_storage_rejects_paths_outside_upload_root(tmp_path):
    storage = FileStorageService(tmp_path / "uploads")

    with pytest.raises(FileStorageError, match="escapes uploads root"):
        storage.resolve(tmp_path / "outside.txt")


def test_promotion_and_delete_operations_are_idempotent(db_session, tmp_path):
    storage = FileStorageService(tmp_path)
    staged = storage.stage_stream(BytesIO(b"policy"), "policy.txt", max_bytes=100)
    document = DocumentModel(
        title="Policy",
        filename=staged.filename,
        file_path=staged.target_path,
        category="txt",
        status="processing",
        indexing_status="processing",
        storage_state="promoting",
    )
    db_session.add(document)
    db_session.flush()
    operation = FileOperation(
        document_id=document.id,
        operation="promote",
        source_path=staged.source_path,
        target_path=staged.target_path,
        status="pending",
    )
    db_session.add(operation)
    db_session.commit()

    _finish_promotion(db_session, storage, operation)
    _finish_promotion(db_session, storage, operation)
    assert document.storage_state == "ready"
    assert document.status == "active"

    delete_operation = FileOperation(
        document_id=document.id,
        operation="delete",
        source_path=staged.target_path,
        target_path=storage.trash_path(document.id, staged.filename),
        status="pending",
    )
    document.status = "delete_pending"
    document.storage_state = "delete_pending"
    db_session.add(delete_operation)
    db_session.commit()
    document_id = document.id

    _finish_deletion(db_session, storage, delete_operation)

    assert db_session.query(DocumentModel).filter(DocumentModel.id == document_id).first() is None
    assert not Path(delete_operation.target_path).exists()
