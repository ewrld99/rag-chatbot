from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
from uuid import UUID

from sqlalchemy import text

from app.core.config import settings
from app.db.models import DocumentModel, FileOperation
from app.db.session import SessionLocal, engine
from app.services.file_storage_service import FileStorageService

logger = logging.getLogger(__name__)
_ADVISORY_LOCK_ID = 742913861
_stop_event: asyncio.Event | None = None
_task: asyncio.Task | None = None


def create_file_operation(
    db,
    *,
    document_id: UUID | None,
    operation: str,
    source_path: str,
    target_path: str,
) -> FileOperation:
    row = FileOperation(
        document_id=document_id,
        operation=operation,
        source_path=source_path,
        target_path=target_path,
        status="pending",
    )
    db.add(row)
    db.flush()
    return row


def reconcile_file_operations(
    limit: int = 100,
    *,
    storage: FileStorageService | None = None,
) -> int:
    storage = storage or FileStorageService()
    with engine.connect() as connection:
        db = SessionLocal()
        lock_acquired = True
        if connection.dialect.name == "postgresql":
            lock_acquired = bool(
                connection.execute(
                    text("SELECT pg_try_advisory_lock(:lock_id)"),
                    {"lock_id": _ADVISORY_LOCK_ID},
                ).scalar()
            )
        if not lock_acquired:
            db.close()
            return 0
        try:
            operations = (
                db.query(FileOperation)
                .filter(FileOperation.status.in_(["pending", "failed", "processing"]))
                .order_by(FileOperation.created_at.asc())
                .with_for_update(skip_locked=True)
                .limit(limit)
                .all()
            )
            processed = 0
            for operation in operations:
                operation.status = "processing"
                operation.attempts += 1
                operation.last_error = None
                db.commit()
                try:
                    if operation.operation == "promote":
                        _finish_promotion(db, storage, operation)
                    elif operation.operation == "delete":
                        _finish_deletion(db, storage, operation)
                    else:
                        raise RuntimeError(f"Unknown file operation: {operation.operation}")
                    processed += 1
                except Exception as exc:
                    db.rollback()
                    current = db.query(FileOperation).filter(FileOperation.id == operation.id).first()
                    if current is not None:
                        current.status = "failed"
                        current.last_error = str(exc)[:2000]
                        document = (
                            db.query(DocumentModel).filter(DocumentModel.id == current.document_id).first()
                            if current.document_id
                            else None
                        )
                        if document is not None:
                            document.storage_state = "missing"
                            document.storage_error = current.last_error
                            document.status = "storage_failed"
                            document.indexing_status = "failed"
                        db.commit()
                    logger.warning("File operation %s failed: %s", operation.id, exc)
            _classify_unverified(db, storage)
            _clean_stale(storage)
            return processed
        finally:
            db.close()
            if connection.dialect.name == "postgresql":
                connection.execute(
                    text("SELECT pg_advisory_unlock(:lock_id)"),
                    {"lock_id": _ADVISORY_LOCK_ID},
                )


def _finish_promotion(db, storage: FileStorageService, operation: FileOperation) -> None:
    storage.promote(operation.source_path, operation.target_path)
    document = (
        db.query(DocumentModel).filter(DocumentModel.id == operation.document_id).first()
        if operation.document_id
        else None
    )
    if document is not None:
        document.file_path = operation.target_path
        document.storage_state = "ready"
        document.storage_error = None
        document.indexing_status = "idle"
        document.status = (
            "quality_review" if document.quality_status == "review" else "active"
        )
    operation.status = "completed"
    operation.last_error = None
    db.commit()


def _finish_deletion(db, storage: FileStorageService, operation: FileOperation) -> None:
    trash_path = operation.target_path
    storage.move_to_trash(operation.source_path, operation.target_path)
    document = (
        db.query(DocumentModel).filter(DocumentModel.id == operation.document_id).first()
        if operation.document_id
        else None
    )
    if document is not None:
        db.delete(document)
    db.commit()
    operation = db.query(FileOperation).filter(FileOperation.id == operation.id).first()
    if operation is not None:
        operation.status = "completed"
        operation.last_error = None
        db.commit()
    storage.remove(trash_path)


def _classify_unverified(db, storage: FileStorageService) -> None:
    rows = (
        db.query(DocumentModel)
        .filter(DocumentModel.storage_state == "unverified")
        .limit(500)
        .all()
    )
    for document in rows:
        if not document.file_path:
            document.storage_state = "not_applicable"
        else:
            try:
                document.storage_state = (
                    "ready" if storage.resolve(document.file_path).is_file() else "missing"
                )
                document.storage_error = (
                    None if document.storage_state == "ready" else "Stored source file is missing"
                )
            except Exception as exc:
                document.storage_state = "missing"
                document.storage_error = str(exc)[:2000]
    db.commit()


def _clean_stale(storage: FileStorageService) -> None:
    now = datetime.now(timezone.utc).timestamp()
    for root, hours in (
        (storage.staging_root, settings.FILE_STAGING_RETENTION_HOURS),
        (storage.trash_root, settings.FILE_TRASH_RETENTION_HOURS),
    ):
        cutoff = now - timedelta(hours=max(hours, 1)).total_seconds()
        for path in root.iterdir():
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
            except OSError:
                logger.debug("Could not clean stale storage artifact %s", path)


async def _reconciler_loop() -> None:
    assert _stop_event is not None
    while not _stop_event.is_set():
        try:
            await asyncio.to_thread(reconcile_file_operations)
        except Exception:
            logger.exception("Storage reconciliation pass failed")
        try:
            await asyncio.wait_for(
                _stop_event.wait(),
                timeout=max(settings.FILE_RECONCILE_INTERVAL_SECONDS, 1.0),
            )
        except asyncio.TimeoutError:
            pass


def start_storage_reconciler() -> None:
    global _stop_event, _task
    if _task is not None and not _task.done():
        return
    _stop_event = asyncio.Event()
    _task = asyncio.create_task(_reconciler_loop(), name="storage-reconciler")


async def stop_storage_reconciler() -> None:
    global _stop_event, _task
    if _task is None:
        return
    assert _stop_event is not None
    _stop_event.set()
    await _task
    _task = None
    _stop_event = None
