import os
import logging
import time
from uuid import UUID, uuid4
from typing import Optional, List
from fastapi import APIRouter, BackgroundTasks, UploadFile, File, HTTPException, Depends, Form, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
import csv
import io
import json

from app.api.deps import require_admin_identity, require_admin_user
from app.core.config import settings
from app.core.time import application_now
from app.services.embedding_service import EmbeddingServiceError, get_embeddings
from app.db.models import DocumentModel, DocumentChunk, FileOperation, RetrievalAlias, User
from app.db.session import get_db, SessionLocal, session_scope
from app.schemas.document import (
    DocumentCreate, 
    DocumentUpdate, 
    DocumentResponse, 
    PaginatedDocumentResponse,
    DocumentQualityResponse,
    DocumentQualityApproval,
    DocumentAuditRequest,
)
from app.services.settings_service import SettingsService
from app.schemas.settings import SystemSettingResponse, SystemSettingUpdate
from app.services.faq_service import FAQService
from app.services.alias_expansion_service import AliasExpansionService
from app.services.almanac_service import build_almanac_chunks
from app.services.curriculum_service import build_curriculum_chunks
from app.services.document_ingestion_service import (
    DocumentIngestionService,
    quality_summary,
)
from app.services.document_quality_service import normalized_document_hash
from app.services.file_storage_service import (
    FileStorageService,
    UploadTooLargeError,
)
from app.services.storage_reconciler import (
    create_file_operation,
    reconcile_file_operations,
)

router = APIRouter()
logger = logging.getLogger(__name__)

UPLOAD_DIR = settings.UPLOADS_DIR
os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_admin_username(current_user: User = Depends(require_admin_user)) -> str:
    return current_user.username


def require_admin(admin_username: str = Depends(get_admin_username)) -> str:
    return admin_username


def serialize_document(document: DocumentModel, include_content: bool = False) -> dict:
    data = {
        "id": str(document.id),
        "title": document.title,
        "filename": document.filename,
        "file_path": document.file_path,
        "category": document.category,
        "source_url": document.source_url,
        "uploaded_by": document.uploaded_by,
        "upload_date": document.upload_date,
        "status": document.status,
        "chunk_count": len(document.chunks),
        "quality_status": document.quality_status or "unchecked",
        "indexing_status": document.indexing_status or "idle",
        "quality_checked_at": document.quality_checked_at,
        "ingestion_version": document.ingestion_version or "1",
        "duplicate_of_document_id": (
            str(document.duplicate_of_document_id)
            if document.duplicate_of_document_id
            else None
        ),
        "quality_summary": quality_summary(document),
        "storage_state": document.storage_state or "not_applicable",
        "storage_error": document.storage_error,
    }
    if include_content:
        data["content"] = "\n\n".join(chunk.chunk_text for chunk in sorted(document.chunks, key=lambda c: c.chunk_index))
    return data


def get_document_or_404(document_id: str | UUID, db: Session) -> DocumentModel:
    try:
        row_id = document_id if isinstance(document_id, UUID) else UUID(document_id)
        document = db.query(DocumentModel).filter(DocumentModel.id == row_id).first()
    except (TypeError, ValueError):
        # fallback to finding by filename
        document = db.query(DocumentModel).filter(DocumentModel.filename == str(document_id)).first()
        
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    return document


def rebuild_document_chunks(
    document: DocumentModel,
    content: str,
    db: Session,
    extra_metadata: dict | None = None,
) -> None:
    ingestion = DocumentIngestionService(db)
    prepared = ingestion.prepare_text(
        content,
        file_type=document.category or "txt",
        current_document_id=document.id,
        base_metadata=extra_metadata,
        extraction_method="manual",
    )
    try:
        embeddings = ingestion.embed(prepared)
    except EmbeddingServiceError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
    except Exception as e:
        logger.exception("Embedding failed while rebuilding document chunks")
        raise HTTPException(status_code=502, detail=f"Embedding service error: {str(e)}")

    ingestion.apply(
        document,
        prepared,
        embeddings,
        persist_review_chunks=True,
    )
    db.expire(document, ["chunks"])


# ---------------------------------------
# Document CRUD
# ---------------------------------------
from sqlalchemy import Text, func, or_
from app.db.models import DocumentChunk

@router.get("/documents/", response_model=PaginatedDocumentResponse)
def list_documents(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=1000),
    search: Optional[str] = None,
    db: Session = Depends(get_db),
    admin_username: str = Depends(require_admin),
):
    query = db.query(DocumentModel, func.count(DocumentChunk.id).label("chunk_count")) \
        .outerjoin(DocumentChunk, DocumentModel.id == DocumentChunk.document_id)
        
    if search:
        search_filter = f"%{search}%"
        query = query.filter(
            (DocumentModel.filename.ilike(search_filter)) | 
            (DocumentModel.title.ilike(search_filter))
        )
        
    query = query.group_by(DocumentModel.id).order_by(
        DocumentModel.source_url.is_(None).desc(),
        DocumentModel.upload_date.desc(),
        DocumentModel.id.desc(),
    )
    
    # Calculate totals
    total = query.count()
    total_pages = max(1, (total + limit - 1) // limit)
    page = min(page, total_pages)
    
    # Apply pagination
    results = query.offset((page - 1) * limit).limit(limit).all()
    
    items = []
    for doc, count in results:
        items.append({
            "id": str(doc.id),
            "title": doc.title,
            "filename": doc.filename,
            "file_path": doc.file_path,
            "category": doc.category,
            "source_url": doc.source_url,
            "uploaded_by": doc.uploaded_by,
            "upload_date": doc.upload_date,
            "status": doc.status,
            "chunk_count": count,
            "quality_status": doc.quality_status or "unchecked",
            "indexing_status": doc.indexing_status or "idle",
            "quality_checked_at": doc.quality_checked_at,
            "ingestion_version": doc.ingestion_version or "1",
            "duplicate_of_document_id": (
                str(doc.duplicate_of_document_id)
                if doc.duplicate_of_document_id
                else None
            ),
            "quality_summary": quality_summary(doc),
            "storage_state": doc.storage_state or "not_applicable",
            "storage_error": doc.storage_error,
        })
        
    return {
        "items": items,
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": total_pages
    }


@router.get("/documents/{document_id:uuid}", response_model=DocumentResponse)
def read_document(
    document_id: UUID,
    db: Session = Depends(get_db),
    admin_username: str = Depends(require_admin),
):
    document = get_document_or_404(document_id, db)
    return serialize_document(document, include_content=True)


@router.post("/documents/manual", response_model=DocumentResponse, status_code=201)
def create_document(
    payload: DocumentCreate,
    db: Session = Depends(get_db),
    admin_username: str = Depends(require_admin),
):
    content = payload.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Content cannot be empty")

    try:
        title = payload.title or f"manual-{uuid4()}"
        document = DocumentModel(
            title=title,
            filename=payload.filename or title,
            category=payload.category or "manual",
            status=payload.status or "active"
        )
        db.add(document)
        db.flush()
        
        rebuild_document_chunks(document, content, db)
        db.commit()
        db.refresh(document)
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.exception("Database insert failed for manually created document")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    return serialize_document(document, include_content=True)


@router.put("/documents/{document_id:uuid}", response_model=DocumentResponse)
def update_document(
    document_id: UUID,
    payload: DocumentUpdate,
    db: Session = Depends(get_db),
    admin_username: str = Depends(require_admin),
):
    document = get_document_or_404(document_id, db)
    changes = payload.dict(exclude_unset=True)

    if "title" in changes: document.title = changes["title"]
    if "filename" in changes: document.filename = changes["filename"]
    if "category" in changes: document.category = changes["category"]
    if "status" in changes: document.status = changes["status"]

    content = changes.get("content")
    if content is not None:
        content = content.strip()
        if not content:
            raise HTTPException(status_code=400, detail="Content cannot be empty")

    try:
        if content is not None:
            rebuild_document_chunks(document, content, db)
            
        db.commit()
        db.refresh(document)
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.exception("Database update failed for document %s", document_id)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    return serialize_document(document, include_content=True)


@router.delete("/documents/{document_id:uuid}")
def delete_document(
    document_id: UUID,
    db: Session = Depends(get_db),
    admin_username: str = Depends(require_admin),
):
    document = get_document_or_404(document_id, db)
    deleted_document_id = str(document.id)
    chunk_count = len(document.chunks)

    try:
        if document.file_path:
            storage = FileStorageService(UPLOAD_DIR)
            document.status = "delete_pending"
            document.storage_state = "delete_pending"
            document.storage_error = None
            create_file_operation(
                db,
                document_id=document.id,
                operation="delete",
                source_path=document.file_path,
                target_path=storage.trash_path(
                    document.id,
                    document.filename or "document",
                ),
            )
            db.commit()
            reconcile_file_operations(storage=storage)
        else:
            db.delete(document)
            db.commit()
    except Exception as e:
        db.rollback()
        logger.exception("Database delete failed for document %s", document_id)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    return {
        "message": "Document deleted successfully",
        "id": deleted_document_id,
        "chunks_deleted": chunk_count,
    }


@router.post("/documents/delete-batch")
def delete_documents_batch(
    document_ids: List[str],
    db: Session = Depends(get_db),
    admin_username: str = Depends(require_admin),
):
    if not document_ids:
        return {"status": "ok", "deleted_count": 0}
        
    try:
        # Convert valid UUID strings
        valid_ids = []
        for doc_id in document_ids:
            try:
                valid_ids.append(UUID(doc_id))
            except ValueError:
                pass
                
        if valid_ids:
            documents = db.query(DocumentModel).filter(DocumentModel.id.in_(valid_ids)).all()
            storage = FileStorageService(UPLOAD_DIR)
            for document in documents:
                if document.file_path:
                    document.status = "delete_pending"
                    document.storage_state = "delete_pending"
                    document.storage_error = None
                    create_file_operation(
                        db,
                        document_id=document.id,
                        operation="delete",
                        source_path=document.file_path,
                        target_path=storage.trash_path(
                            document.id,
                            document.filename or "document",
                        ),
                    )
                else:
                    db.delete(document)
            db.commit()
            reconcile_file_operations(storage=storage)
            return {"status": "ok", "deleted_count": len(documents)}
        return {"status": "ok", "deleted_count": 0}
    except Exception as e:
        db.rollback()
        logger.exception("Database batch delete failed")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


# ---------------------------------------
# Settings CRUD
# ---------------------------------------
@router.get("/settings/", response_model=list[SystemSettingResponse])
def list_settings(db: Session = Depends(get_db), admin_username: str = Depends(require_admin)):
    settings_svc = SettingsService(db)
    return settings_svc.get_all()


@router.put("/settings/{key}", response_model=SystemSettingResponse)
def update_setting(
    key: str, 
    payload: SystemSettingUpdate, 
    db: Session = Depends(get_db),
    admin_username: str = Depends(require_admin)
):
    settings_svc = SettingsService(db)
    try:
        return settings_svc.update(key, payload.value, admin_username=admin_username)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------
# 1. Upload & Ingest Document (FIXED)
# ---------------------------------------
def _prepare_uploaded_document(
    file_path: str,
    file_ext: str,
    strategy: str,
    chunk_meta: dict,
):
    with session_scope(SessionLocal) as db:
        return DocumentIngestionService(db).prepare_file(
            file_path,
            file_ext,
            strategy=strategy,
            base_metadata=chunk_meta,
        )


def _upload_embedding_service():
    from app.services.embedding_service import EmbeddingService

    with session_scope(SessionLocal) as db:
        return EmbeddingService(db)


def _register_staged_upload(staged, title: str, file_ext: str) -> UUID:
    with session_scope(SessionLocal) as db:
        document = DocumentModel(
            title=title,
            filename=staged.filename,
            file_path=staged.target_path,
            category=file_ext,
            content_hash=staged.sha256,
            status="processing",
            indexing_status="processing",
            storage_state="staged",
        )
        db.add(document)
        db.flush()
        create_file_operation(
            db,
            document_id=document.id,
            operation="promote",
            source_path=staged.source_path,
            target_path=staged.target_path,
        )
        db.commit()
        return document.id


def _retain_failed_upload(
    document_id: UUID,
    staged_path: str,
    filename: str,
    error: Exception,
    prepared=None,
) -> None:
    storage = FileStorageService(UPLOAD_DIR)
    retained_path = None
    storage_error = str(error)[:2000]
    try:
        retained_path = storage.retain_failed(staged_path, document_id, filename)
    except Exception as storage_exc:
        storage_error = f"{storage_error}; retaining source failed: {storage_exc}"[:2000]

    with session_scope(SessionLocal) as db:
        document = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
        if document is None:
            return
        if prepared is not None:
            DocumentIngestionService(db).record_quality(document, prepared)
        document.file_path = retained_path or document.file_path
        document.storage_state = "ready" if retained_path else "missing"
        document.storage_error = storage_error
        document.status = "embedding_failed"
        document.indexing_status = "failed"
        db.query(FileOperation).filter(
            FileOperation.document_id == document.id,
            FileOperation.operation == "promote",
            FileOperation.status.in_(["pending", "processing", "failed"]),
        ).update(
            {FileOperation.status: "cancelled", FileOperation.last_error: storage_error},
            synchronize_session=False,
        )
        db.commit()


def _save_completed_upload(
    document_id: UUID,
    prepared,
    embeddings: list[list[float]],
) -> None:
    with session_scope(SessionLocal) as db:
        document = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
        if document is None:
            raise RuntimeError("Staged document record no longer exists")
        DocumentIngestionService(db).apply(
            document,
            prepared,
            embeddings,
            persist_review_chunks=True,
        )
        document.status = "processing"
        document.indexing_status = "processing"
        document.storage_state = "promoting"
        document.storage_error = None
        db.commit()


def _upload_ingestion_generator(
    *,
    document_id: UUID,
    staged_path: str,
    file_ext: str,
    filename: str,
    strategy: str,
    programme: str | None,
    year: int | None,
):
    try:
        yield json.dumps({"progress": 15, "status": "Extracting text..."}) + "\n"
        chunk_meta: dict = {}
        if programme:
            chunk_meta["programme"] = programme.strip()
        if year is not None:
            chunk_meta["year"] = str(year)
        if strategy == "almanac":
            chunk_meta.update(category="academic_calendar", record_type="almanac_event")
        elif strategy == "curriculum":
            chunk_meta.update(category="curriculum", record_type="course")

        try:
            prepared = _prepare_uploaded_document(staged_path, file_ext, strategy, chunk_meta)
        except Exception as exc:
            _retain_failed_upload(document_id, staged_path, filename, exc)
            yield json.dumps({
                "error": f"Error reading file: {str(exc)}",
                "status": "embedding_failed",
            }) + "\n"
            return
        chunks = [chunk.text for chunk in prepared.chunks]
        yield json.dumps({
            "progress": 30,
            "status": "Checking extraction and chunk quality...",
            "quality_status": prepared.quality.status,
            "quality": prepared.quality.report,
        }) + "\n"
        yield json.dumps({
            "progress": 35,
            "status": f"Generating embeddings for {len(chunks)} chunks...",
        }) + "\n"

        embedding_service = _upload_embedding_service()
        batch_size = 15 if embedding_service.provider == "local" else embedding_service.batch_size
        batch_size = max(1, int(batch_size))
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        embeddings: list[list[float]] = []
        for batch_index, start in enumerate(range(0, len(chunks), batch_size), start=1):
            yield json.dumps({
                "progress": 35 + int((batch_index / total_batches) * 50),
                "status": f"Generating embeddings (batch {batch_index}/{total_batches})...",
            }) + "\n"
            try:
                embeddings.extend(
                    embedding_service.embed_batch(chunks[start:start + batch_size])
                )
            except Exception as exc:
                logger.warning("Embedding failed for uploaded document %s: %s", filename, exc)
                try:
                    _retain_failed_upload(
                        document_id,
                        staged_path,
                        filename,
                        exc,
                        prepared,
                    )
                except Exception:
                    logger.exception("Could not save embedding_failed record for %s", filename)
                yield json.dumps({
                    "error": f"Embedding service error: {str(exc)}",
                    "status": "embedding_failed",
                    "message": "The document was saved for retry after embedding recovers.",
                }) + "\n"
                return

        if len(embeddings) != len(chunks):
            yield json.dumps({"error": "Embedding service returned an unexpected number of vectors"}) + "\n"
            return

        yield json.dumps({"progress": 90, "status": "Saving document to database..."}) + "\n"
        _save_completed_upload(document_id, prepared, embeddings)
        reconcile_file_operations(storage=FileStorageService(UPLOAD_DIR))
        with session_scope(SessionLocal) as db:
            document = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
            if document is None or document.storage_state != "ready":
                detail = document.storage_error if document is not None else "document disappeared"
                raise RuntimeError(f"File promotion did not complete: {detail}")
        yield json.dumps({
            "progress": 100,
            "status": "Quality review required" if prepared.quality.blocks_activation else "Done",
            "filename": filename,
            "chunks_stored": len(chunks),
            "quality_status": prepared.quality.status,
            "quality": prepared.quality.report,
        }) + "\n"
    except Exception as exc:
        logger.exception("Error during document ingestion generator")
        yield json.dumps({"error": f"Internal Server Error: {str(exc)}"}) + "\n"


@router.post("/documents/")
def upload_document(
    file: UploadFile = File(...),
    strategy: str = Form("auto"),
    programme: Optional[str] = Form(None, description="Programme this document belongs to, e.g. 'BSc Computer Science'"),
    year: Optional[int] = Form(None, description="Year of study this document applies to, e.g. 2"),
    admin_username: str = Depends(require_admin_identity),
):
    """
    Streaming ingestion pipeline:
    - Save file
    - Extract text
    - Chunk text
    - Generate embeddings (with progress)
    - Store in PostgreSQL (pgvector)
    Yields NDJSON progress updates.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a name")

    filename = os.path.basename(file.filename).lower()
    if not filename.endswith((".pdf", ".docx", ".txt", ".md")):
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, TXT, and MD allowed")
    
    file_ext = filename.split(".")[-1]
    name_without_ext = os.path.splitext(filename)[0]

    with session_scope(SessionLocal) as db:
        max_upload_bytes = SettingsService(db).max_upload_size_mb * 1024 * 1024
    storage = FileStorageService(UPLOAD_DIR)
    try:
        staged = storage.stage_stream(file.file, filename, max_bytes=max_upload_bytes)
    except UploadTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"File staging error: {str(exc)}") from exc

    try:
        document_id = _register_staged_upload(staged, name_without_ext, file_ext)
    except Exception:
        storage.remove(staged.source_path)
        raise
    return StreamingResponse(
        _upload_ingestion_generator(
            document_id=document_id,
            staged_path=staged.source_path,
            file_ext=file_ext,
            filename=staged.filename,
            strategy=strategy,
            programme=programme,
            year=year,
        ),
        media_type="application/x-ndjson",
    )

# ---------------------------------------
# Reindex All
# ---------------------------------------
@router.post("/documents/reindex-all")
def reindex_all_documents(
    background_tasks: BackgroundTasks,
    force: bool = False,
    db: Session = Depends(get_db),
    admin_username: str = Depends(require_admin),
):
    """
    Queue a background re-ingestion job for documents.
    If force is True, reindexes all documents. Otherwise, only those marked needs_reindex.
    """
    query = db.query(DocumentModel)
    if not force:
        query = query.filter(DocumentModel.status == "needs_reindex")

    doc_ids = [
        str(row.id)
        for row in query.with_entities(DocumentModel.id).yield_per(500)
    ]

    if not doc_ids:
        return {"queued": 0, "message": "No documents require reindexing."}

    query.update({DocumentModel.indexing_status: "processing"}, synchronize_session=False)
    db.commit()

    background_tasks.add_task(_quality_reindex_documents_task, doc_ids)

    return {"queued": len(doc_ids), "message": f"{len(doc_ids)} document(s) queued for reindexing."}

@router.post("/documents/{document_id:uuid}/reindex")
def reindex_single_document(
    document_id: UUID,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    admin_username: str = Depends(require_admin),
):
    """
    Queue a background re-ingestion job for a specific document.
    """
    doc = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    doc.indexing_status = "processing"
    db.commit()

    background_tasks.add_task(_quality_reindex_documents_task, [str(doc.id)])

    return {"message": f"Document '{doc.filename}' queued for reindexing."}


def _quality_reindex_documents_task(doc_ids: list[str]) -> None:
    """Reindex through the shared quality gate without deleting a valid old index."""
    from app.db.session import SessionLocal

    with SessionLocal() as db:
        for raw_document_id in doc_ids:
            try:
                document_id = UUID(raw_document_id)
                document = db.query(DocumentModel).filter(
                    DocumentModel.id == document_id
                ).first()
                if document is None:
                    logger.warning("Reindex: document %s not found, skipping", raw_document_id)
                    continue

                document.indexing_status = "processing"
                db.commit()

                rows = (
                    db.query(DocumentChunk)
                    .filter(DocumentChunk.document_id == document.id)
                    .order_by(DocumentChunk.chunk_index.asc())
                    .all()
                )
                first_metadata = dict(rows[0].metadata_ or {}) if rows else {}
                strategy = "auto"
                if first_metadata.get("record_type") == "almanac_event":
                    strategy = "almanac"
                elif (
                    first_metadata.get("category") == "curriculum"
                    and first_metadata.get("record_type") == "course"
                ):
                    strategy = "curriculum"

                base_metadata = {
                    key: value
                    for key, value in first_metadata.items()
                    if key in {"programme", "year", "source_url"}
                }
                ingestion = DocumentIngestionService(db)
                target_file_path = document.file_path
                if not target_file_path and document.filename:
                    target_file_path = os.path.join(UPLOAD_DIR, document.filename)

                if target_file_path and os.path.exists(target_file_path):
                    file_type = target_file_path.rsplit(".", 1)[-1].lower()
                    prepared = ingestion.prepare_file(
                        target_file_path,
                        file_type,
                        strategy=strategy,
                        current_document_id=document.id,
                        base_metadata=base_metadata,
                    )
                elif rows:
                    prepared = ingestion.prepare_existing_chunks(
                        rows,
                        file_type=document.category or "txt",
                        strategy=strategy,
                        current_document_id=document.id,
                    )
                else:
                    prepared = ingestion.prepare_text(
                        "",
                        file_type=document.category or "txt",
                        strategy=strategy,
                        current_document_id=document.id,
                        extraction_method="missing_source",
                    )

                prepared = ingestion.preserve_override(document, prepared)

                if prepared.quality.blocks_activation:
                    ingestion.record_quality(document, prepared)
                    document.status = "quality_review"
                    document.indexing_status = "idle"
                    db.commit()
                    logger.warning(
                        "Reindex: document %s requires quality review; existing chunks retained",
                        raw_document_id,
                    )
                    continue

                embeddings = ingestion.embed(prepared)
                ingestion.apply(document, prepared, embeddings)
                db.commit()
                logger.info(
                    "Reindex: document %s completed (%d chunks, quality=%s)",
                    raw_document_id,
                    len(prepared.chunks),
                    prepared.quality.status,
                )
            except Exception as exc:
                logger.exception("Reindex: document %s failed: %s", raw_document_id, exc)
                db.rollback()
                try:
                    document = db.query(DocumentModel).filter(
                        DocumentModel.id == UUID(raw_document_id)
                    ).first()
                    if document is not None:
                        document.indexing_status = "failed"
                        if not document.chunks:
                            document.status = "embedding_failed"
                        db.commit()
                except Exception:
                    db.rollback()


def _legacy_reindex_documents_task(doc_ids: list[str]) -> None:
    """Background task: re-chunk and re-embed each document."""
    from app.db.session import SessionLocal

    db = SessionLocal()
    try:
        settings_svc = SettingsService(db)
        chunk_size = settings_svc.chunk_size
        chunk_overlap = settings_svc.chunk_overlap

        for doc_id in doc_ids:
            try:
                from uuid import UUID as _UUID
                document = db.query(DocumentModel).filter(
                    DocumentModel.id == _UUID(doc_id)
                ).first()

                if not document:
                    logger.warning("Reindex: document %s not found, skipping", doc_id)
                    continue

                first_chunk = (
                    db.query(DocumentChunk.metadata_)
                    .filter(DocumentChunk.document_id == document.id)
                    .order_by(DocumentChunk.chunk_index)
                    .first()
                )
                first_metadata = dict(first_chunk.metadata_ or {}) if first_chunk else {}
                is_almanac_document = first_metadata.get("record_type") == "almanac_event"
                is_curriculum_document = (
                    first_metadata.get("category") == "curriculum"
                    and first_metadata.get("record_type") == "course"
                )
                preserved_chunk_meta = {
                    key: first_metadata[key]
                    for key in ("programme", "year")
                    if first_metadata.get(key)
                }

                # Try to reload from the original file if it still exists
                target_file_path = document.file_path
                if not target_file_path and document.filename:
                    import os
                    target_file_path = os.path.join(UPLOAD_DIR, document.filename)

                content = None
                fallback_chunk_rows = None
                if target_file_path and target_file_path.endswith((".pdf", ".docx", ".txt", ".md")):
                    import os
                    if os.path.exists(target_file_path):
                        try:
                            file_ext = target_file_path.rsplit(".", 1)[-1].lower()
                            content = load_document(target_file_path, file_ext)
                        except Exception as e:
                            logger.error(f"Reindex failed reading file {target_file_path}: {e}")
                    else:
                        logger.error(f"Reindex failed: file not found on disk at {target_file_path}")

                if not content:
                    chunk_rows = (
                        db.query(DocumentChunk.chunk_text, DocumentChunk.metadata_)
                        .filter(DocumentChunk.document_id == document.id)
                        .order_by(DocumentChunk.chunk_index)
                        .yield_per(200)
                    )
                    fallback_chunk_rows = [
                        (row.chunk_text, dict(row.metadata_ or {}))
                        for row in chunk_rows
                        if row.chunk_text
                    ]
                    content_parts = [chunk_text for chunk_text, _metadata in fallback_chunk_rows]
                    if content_parts:
                        content = "\n\n".join(content_parts)

                if not content:
                    logger.warning("Reindex: document %s content could not be loaded, failing safely to avoid corruption", doc_id)
                    document.status = "failed"
                    db.commit()
                    continue

                if not content or not content.strip():
                    document.status = "failed"
                    db.commit()
                    continue

                new_chunk_metadata: list[dict] | None = None
                if is_almanac_document:
                    almanac_chunks = build_almanac_chunks(content)
                    if almanac_chunks:
                        new_chunks = [chunk for chunk, _metadata in almanac_chunks]
                        new_chunk_metadata = [
                            {**preserved_chunk_meta, **metadata}
                            for _chunk, metadata in almanac_chunks
                        ]
                    elif fallback_chunk_rows:
                        new_chunks = [chunk for chunk, _metadata in fallback_chunk_rows]
                        new_chunk_metadata = [
                            {
                                "category": "academic_calendar",
                                "record_type": "almanac_event",
                                **preserved_chunk_meta,
                                **metadata,
                            }
                            for _chunk, metadata in fallback_chunk_rows
                        ]
                    else:
                        logger.warning("Reindex: almanac parser found no event rows for document %s", doc_id)
                        document.status = "failed"
                        db.commit()
                        continue
                elif is_curriculum_document:
                    curriculum_chunks = build_curriculum_chunks(content)
                    if curriculum_chunks:
                        new_chunks = [chunk for chunk, _metadata in curriculum_chunks]
                        new_chunk_metadata = [
                            {**preserved_chunk_meta, **metadata}
                            for _chunk, metadata in curriculum_chunks
                        ]
                    elif fallback_chunk_rows:
                        new_chunks = [chunk for chunk, _metadata in fallback_chunk_rows]
                        new_chunk_metadata = [
                            {
                                "category": "curriculum",
                                "record_type": "course",
                                **preserved_chunk_meta,
                                **metadata,
                            }
                            for _chunk, metadata in fallback_chunk_rows
                        ]
                    else:
                        logger.warning("Reindex: curriculum parser found no course rows for document %s", doc_id)
                        document.status = "failed"
                        db.commit()
                        continue
                else:
                    new_chunks = split_text(content, chunk_size=chunk_size, overlap=chunk_overlap)

                if not new_chunks:
                    document.status = "failed"
                    db.commit()
                    continue

                # ── SAFE ATOMIC REINDEX ──────────────────────────────────
                # Embed FIRST. Only delete old chunks after we confirm the
                # new embeddings are ready. This prevents the document from
                # being left with zero chunks if embedding fails mid-task.
                try:
                    new_embeddings = get_embeddings(new_chunks, db)
                except Exception as emb_exc:
                    logger.warning(
                        "Reindex: embedding failed for document %s — keeping "
                        "existing chunks intact and marking as 'embedding_failed'. "
                        "Retry once the embedding service is available. Reason: %s",
                        doc_id, emb_exc,
                    )
                    document.status = "embedding_failed"
                    db.commit()
                    continue

                if len(new_embeddings) != len(new_chunks):
                    logger.error(
                        "Reindex: embedding count mismatch for document %s "
                        "(%d chunks, %d embeddings) — keeping existing chunks.",
                        doc_id, len(new_chunks), len(new_embeddings),
                    )
                    document.status = "embedding_failed"
                    db.commit()
                    continue

                # Embeddings are ready — now safely replace chunks.
                db.query(DocumentChunk).filter(
                    DocumentChunk.document_id == document.id
                ).delete()

                db.bulk_insert_mappings(
                    DocumentChunk,
                    [
                        {
                            "document_id": document.id,
                            "chunk_text": chunk,
                            "embedding": embedding,
                            "chunk_index": i,
                            "metadata_": new_chunk_metadata[i] if new_chunk_metadata else None,
                        }
                        for i, (chunk, embedding) in enumerate(zip(new_chunks, new_embeddings))
                    ],
                )

                document.status = "active"
                db.commit()
                logger.info("Reindex: document %s completed (%d chunks)", doc_id, len(new_chunks))

            except Exception as exc:
                logger.exception("Reindex: document %s failed: %s", doc_id, exc)
                db.rollback()
                try:
                    document = db.query(DocumentModel).filter(
                        DocumentModel.id == _UUID(doc_id)
                    ).first()
                    if document:
                        document.status = "failed"
                        db.commit()
                except Exception:
                    db.rollback()
    finally:
        db.close()


# ---------------------------------------
# Document Quality Diagnostics
# ---------------------------------------
@router.get(
    "/documents/{document_id:uuid}/quality",
    response_model=DocumentQualityResponse,
)
def get_document_quality(
    document_id: UUID,
    db: Session = Depends(get_db),
    admin_username: str = Depends(require_admin),
):
    document = get_document_or_404(document_id, db)
    return {
        "document_id": str(document.id),
        "lifecycle_status": document.status or "active",
        "indexing_status": document.indexing_status or "idle",
        "quality_status": document.quality_status or "unchecked",
        "quality_checked_at": document.quality_checked_at,
        "ingestion_version": document.ingestion_version or "1",
        "duplicate_of_document_id": (
            str(document.duplicate_of_document_id)
            if document.duplicate_of_document_id
            else None
        ),
        "report": dict(document.quality_report or {}),
    }


@router.post("/documents/quality-audit")
def audit_document_quality(
    payload: DocumentAuditRequest,
    db: Session = Depends(get_db),
    admin_username: str = Depends(require_admin),
):
    query = db.query(DocumentModel)
    if payload.document_ids:
        valid_ids: list[UUID] = []
        for raw_id in payload.document_ids:
            try:
                valid_ids.append(UUID(raw_id))
            except (TypeError, ValueError):
                continue
        query = query.filter(DocumentModel.id.in_(valid_ids))

    documents = query.order_by(
        DocumentModel.upload_date.asc(),
        DocumentModel.id.asc(),
    ).all()
    counts = {"passed": 0, "warning": 0, "review": 0}
    results: list[dict] = []
    ingestion = DocumentIngestionService(db)

    for document in documents:
        rows = sorted(document.chunks, key=lambda chunk: chunk.chunk_index)
        first_metadata = dict(rows[0].metadata_ or {}) if rows else {}
        strategy = "auto"
        if first_metadata.get("record_type") == "almanac_event":
            strategy = "almanac"
        elif (
            first_metadata.get("category") == "curriculum"
            and first_metadata.get("record_type") == "course"
        ):
            strategy = "curriculum"

        target_file_path = document.file_path
        if not target_file_path and document.filename:
            target_file_path = os.path.join(UPLOAD_DIR, document.filename)
        try:
            if target_file_path and os.path.exists(target_file_path):
                prepared = ingestion.prepare_file(
                    target_file_path,
                    target_file_path.rsplit(".", 1)[-1].lower(),
                    strategy=strategy,
                    current_document_id=document.id,
                )
                prepared = ingestion.preserve_override(document, prepared)
                ingestion.record_quality(document, prepared)
                result = prepared.quality
            else:
                result = ingestion.audit_document(document)
            db.commit()
        except Exception as exc:
            db.rollback()
            logger.exception("Quality audit failed for document %s", document.id)
            results.append({
                "document_id": str(document.id),
                "quality_status": "error",
                "error": str(exc),
            })
            continue

        counts[result.status] = counts.get(result.status, 0) + 1
        results.append({
            "document_id": str(document.id),
            "quality_status": result.status,
            "warning_count": result.report.get("warning_count", 0),
            "blocker_count": result.report.get("blocker_count", 0),
            "duplicate_of_document_id": (
                str(result.duplicate_of_document_id)
                if result.duplicate_of_document_id
                else None
            ),
        })

    return {"audited": len(results), "counts": counts, "documents": results}


@router.post(
    "/documents/{document_id:uuid}/quality/approve",
    response_model=DocumentQualityResponse,
)
def approve_document_quality(
    document_id: UUID,
    payload: DocumentQualityApproval,
    db: Session = Depends(get_db),
    admin_username: str = Depends(require_admin),
):
    document = get_document_or_404(document_id, db)
    if not document.chunks:
        raise HTTPException(
            status_code=409,
            detail="The document has no embedded chunks. Correct the source and reindex it before approval.",
        )

    report = dict(document.quality_report or {})
    report["override"] = {
        "approved_by": admin_username,
        "reason": payload.reason.strip(),
        "approved_at": application_now().isoformat(),
        "previous_quality_status": document.quality_status or "unchecked",
    }
    document.quality_report = report
    if not document.content_hash:
        document.content_hash = normalized_document_hash(
            "\n\n".join(
                chunk.chunk_text
                for chunk in sorted(document.chunks, key=lambda row: row.chunk_index)
            )
        )
    document.quality_status = "overridden"
    document.quality_checked_at = application_now()
    document.status = "active"
    document.indexing_status = "idle"
    db.commit()
    db.refresh(document)
    return get_document_quality(document.id, db, admin_username)


# ---------------------------------------
# Broken Document Diagnostics
@router.get("/documents/broken")
def list_broken_documents(db: Session = Depends(get_db), admin_username: str = Depends(require_admin)):
    """
    Return all documents that are likely broken:
      - status is not 'active'
      - OR status is 'active' but they have zero chunks (embedding was never stored)

    Use this to quickly spot documents that uploaded successfully but were
    never indexed (e.g. Jina was down at upload time).
    """
    from sqlalchemy import func as sqlfunc

    # Documents with a non-active status
    non_active = (
        db.query(DocumentModel)
        .filter(DocumentModel.status != "active")
        .all()
    )

    # Documents that are 'active' but have no chunks
    chunk_counts = (
        db.query(DocumentChunk.document_id, sqlfunc.count(DocumentChunk.id).label("cnt"))
        .group_by(DocumentChunk.document_id)
        .subquery()
    )
    active_no_chunks = (
        db.query(DocumentModel)
        .outerjoin(chunk_counts, DocumentModel.id == chunk_counts.c.document_id)
        .filter(
            DocumentModel.status == "active",
            sqlfunc.coalesce(chunk_counts.c.cnt, 0) == 0,
        )
        .all()
    )

    def _row(doc: DocumentModel, reason: str) -> dict:
        return {
            "id": str(doc.id),
            "filename": doc.filename,
            "title": doc.title,
            "status": doc.status,
            "file_path": doc.file_path,
            "storage_state": doc.storage_state or "not_applicable",
            "storage_error": doc.storage_error,
            "upload_date": doc.upload_date.isoformat() if doc.upload_date else None,
            "reason": reason,
        }

    results = [
        _row(doc, f"status={doc.status!r}")
        for doc in non_active
    ] + [
        _row(doc, "active but 0 chunks — embedding was never stored")
        for doc in active_no_chunks
    ]

    return {"broken_count": len(results), "documents": results}


@router.post("/documents/retry-failed")
def retry_failed_documents(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    admin_username: str = Depends(require_admin),
):
    """
    Re-queue all documents with status 'embedding_failed' for reindexing.

    Call this after the embedding service has recovered. The task will
    re-embed each document using its existing file on disk and mark it
    'active' on success or keep it 'embedding_failed' on failure.
    """
    failed_docs = (
        db.query(DocumentModel)
        .filter(DocumentModel.status == "embedding_failed")
        .all()
    )

    if not failed_docs:
        return {"queued": 0, "message": "No documents with status 'embedding_failed' found."}

    for doc in failed_docs:
        doc.indexing_status = "processing"
    db.commit()

    doc_ids = [str(doc.id) for doc in failed_docs]
    background_tasks.add_task(_quality_reindex_documents_task, doc_ids)

    return {
        "queued": len(doc_ids),
        "message": (
            f"{len(doc_ids)} document(s) queued for re-embedding. "
            "Check /documents/broken after a minute to confirm they are now active."
        ),
    }


# ---------------------------------------
# FAQ CRUD
# ---------------------------------------

class FAQCreate(BaseModel):
    question: str = Field(..., max_length=500)
    answer: str = Field(..., max_length=10000)
    category: Optional[str] = None

class FAQUpdate(BaseModel):
    question: Optional[str] = Field(None, max_length=500)
    answer: Optional[str] = Field(None, max_length=10000)
    category: Optional[str] = None
    is_active: Optional[bool] = None

def serialize_faq(faq) -> dict:
    return {
        "id": str(faq.id),
        "question": faq.question,
        "answer": faq.answer,
        "category": faq.category,
        "is_active": faq.is_active,
        "created_at": faq.created_at.isoformat() if faq.created_at else None,
        "updated_at": faq.updated_at.isoformat() if faq.updated_at else None,
    }


@router.get("/faqs/")
def list_faqs(
    skip: int = 0,
    limit: int = 100,
    search: Optional[str] = None,
    category: Optional[str] = None,
    is_active: Optional[bool] = None,
    db: Session = Depends(get_db),
    admin_username: str = Depends(require_admin)
):
    svc = FAQService(db)
    result = svc.get_faqs(skip, limit, search, category, is_active)
    result["items"] = [serialize_faq(f) for f in result["items"]]
    return result

@router.post("/faqs/")
def create_faq(payload: FAQCreate, db: Session = Depends(get_db), admin_username: str = Depends(require_admin)):
    svc = FAQService(db)
    faq = svc.create_faq(payload.question, payload.answer, payload.category, admin_username)
    return serialize_faq(faq)

@router.put("/faqs/{faq_id}")
def update_faq(faq_id: str, payload: FAQUpdate, db: Session = Depends(get_db), admin_username: str = Depends(require_admin)):
    svc = FAQService(db)
    faq = svc.update_faq(faq_id, payload.question, payload.answer, payload.category, payload.is_active, admin_username)
    if not faq:
        raise HTTPException(status_code=404, detail="FAQ not found")
    return serialize_faq(faq)

@router.delete("/faqs/{faq_id}")
def delete_faq(faq_id: str, db: Session = Depends(get_db), admin_username: str = Depends(require_admin)):
    svc = FAQService(db)
    if not svc.delete_faq(faq_id, admin_username):
        raise HTTPException(status_code=404, detail="FAQ not found")
    return {"message": "FAQ soft deleted successfully"}

def _import_faqs_task(content: bytes, filename: str, admin_username: str):
    from app.db.session import SessionLocal
    db = SessionLocal()
    try:
        svc = FAQService(db)
        svc.bulk_import(content, filename, admin_username)
    finally:
        db.close()

@router.post("/faqs/import")
def import_faqs(background_tasks: BackgroundTasks, file: UploadFile = File(...), admin_username: str = Depends(require_admin)):
    content = file.file.read()
    filename = file.filename
    background_tasks.add_task(_import_faqs_task, content, filename, admin_username)
    return {"message": "Import process started in the background."}


# ---------------------------------------
# Retrieval Alias CRUD
# ---------------------------------------

class RetrievalAliasCreate(BaseModel):
    term: str = Field(..., min_length=1, max_length=120)
    aliases: list[str] = Field(default_factory=list, max_length=50)
    category: Optional[str] = Field(None, max_length=120)
    weight: float = Field(1.0, ge=0.1, le=5.0)
    is_active: bool = True


class RetrievalAliasUpdate(BaseModel):
    term: Optional[str] = Field(None, min_length=1, max_length=120)
    aliases: Optional[list[str]] = Field(None, max_length=50)
    category: Optional[str] = Field(None, max_length=120)
    weight: Optional[float] = Field(None, ge=0.1, le=5.0)
    is_active: Optional[bool] = None


def _payload_dict(payload: BaseModel) -> dict:
    if hasattr(payload, "model_dump"):
        return payload.model_dump(exclude_unset=True)
    return payload.dict(exclude_unset=True)


def _normalize_alias_term(term: str) -> str:
    term = " ".join(str(term).strip().lower().split())
    if not term:
        raise HTTPException(status_code=400, detail="Alias term cannot be empty")
    return term


def _normalize_alias_values(values: list[str] | None) -> list[str]:
    normalized: list[str] = []
    seen = set()
    for value in values or []:
        item = " ".join(str(value).strip().lower().split())
        if not item or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return normalized


def _serialize_retrieval_alias(row: RetrievalAlias) -> dict:
    return {
        "id": str(row.id),
        "term": row.term,
        "aliases": row.aliases or [],
        "category": row.category,
        "weight": float(row.weight or 1.0),
        "is_active": bool(row.is_active),
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def _parse_alias_csv_values(raw: str | None) -> list[str]:
    if not raw or not raw.strip():
        return []
    value = raw.strip()
    if value.startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError("aliases JSON must be an array")
        return _normalize_alias_values([str(item) for item in parsed])
    delimiter = ";" if ";" in value else ","
    return _normalize_alias_values(value.split(delimiter))


_ALIAS_CSV_COLUMNS = {
    "term": {
        "term",
        "acronym",
        "abbr",
        "abbreviation",
        "short",
        "short form",
        "short_form",
        "code",
    },
    "aliases": {
        "aliases",
        "alias",
        "meaning",
        "meanings",
        "full form",
        "full_form",
        "full name",
        "full_name",
        "description",
        "expanded",
        "expansion",
        "stands for",
        "stands_for",
    },
    "category": {"category", "type", "group"},
    "weight": {"weight", "score", "priority"},
    "is_active": {"is_active", "active", "enabled", "status"},
}


def _canonical_alias_csv_headers(fieldnames: list[str] | None) -> dict[str, str]:
    header_map: dict[str, str] = {}
    for raw_header in fieldnames or []:
        normalized_header = " ".join(str(raw_header or "").strip().lower().replace("_", " ").split())
        for canonical, aliases in _ALIAS_CSV_COLUMNS.items():
            if normalized_header in aliases and canonical not in header_map:
                header_map[canonical] = raw_header
                break
    return header_map


def _alias_csv_value(row_data: dict[str, str], header_map: dict[str, str], name: str) -> str | None:
    header = header_map.get(name)
    return row_data.get(header) if header else None


def _parse_alias_bool(raw: str | None, default: bool = True) -> bool:
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "active"}


def _parse_alias_weight(raw: str | None) -> float:
    if raw is None or not str(raw).strip():
        return 1.0
    return float(raw)


@router.get("/retrieval-aliases/")
def list_retrieval_aliases(
    skip: int = 0,
    limit: int = 100,
    search: Optional[str] = None,
    category: Optional[str] = None,
    is_active: Optional[bool] = None,
    db: Session = Depends(get_db),
    admin_username: str = Depends(require_admin),
):
    query = db.query(RetrievalAlias)

    if search:
        like = f"%{search.strip().lower()}%"
        query = query.filter(
            or_(
                func.lower(RetrievalAlias.term).like(like),
                func.lower(RetrievalAlias.aliases.cast(Text)).like(like),
            )
        )
    if category:
        query = query.filter(func.lower(RetrievalAlias.category) == category.strip().lower())
    if is_active is not None:
        query = query.filter(RetrievalAlias.is_active == is_active)

    total = query.count()
    rows = (
        query.order_by(func.lower(RetrievalAlias.term).asc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    categories = [
        row[0]
        for row in db.query(RetrievalAlias.category)
        .filter(RetrievalAlias.category.isnot(None))
        .distinct()
        .order_by(RetrievalAlias.category.asc())
        .all()
        if row[0]
    ]

    return {
        "items": [_serialize_retrieval_alias(row) for row in rows],
        "total": total,
        "skip": skip,
        "limit": limit,
        "categories": categories,
    }


@router.post("/retrieval-aliases/", status_code=201)
def create_retrieval_alias(
    payload: RetrievalAliasCreate,
    db: Session = Depends(get_db),
    admin_username: str = Depends(require_admin),
):
    term = _normalize_alias_term(payload.term)
    existing = db.query(RetrievalAlias).filter(func.lower(RetrievalAlias.term) == term).first()
    if existing:
        raise HTTPException(status_code=409, detail="An alias with this term already exists")

    row = RetrievalAlias(
        term=term,
        aliases=_normalize_alias_values(payload.aliases),
        category=_normalize_alias_term(payload.category) if payload.category else None,
        weight=payload.weight,
        is_active=payload.is_active,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    AliasExpansionService.clear_cache()
    return _serialize_retrieval_alias(row)


@router.put("/retrieval-aliases/{alias_id}")
def update_retrieval_alias(
    alias_id: str,
    payload: RetrievalAliasUpdate,
    db: Session = Depends(get_db),
    admin_username: str = Depends(require_admin),
):
    try:
        row_id = UUID(alias_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid alias id")

    row = db.query(RetrievalAlias).filter(RetrievalAlias.id == row_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Alias not found")

    changes = _payload_dict(payload)
    if "term" in changes and changes["term"] is not None:
        term = _normalize_alias_term(changes["term"])
        conflict = (
            db.query(RetrievalAlias)
            .filter(func.lower(RetrievalAlias.term) == term, RetrievalAlias.id != row.id)
            .first()
        )
        if conflict:
            raise HTTPException(status_code=409, detail="An alias with this term already exists")
        row.term = term
    if "aliases" in changes and changes["aliases"] is not None:
        row.aliases = _normalize_alias_values(changes["aliases"])
    if "category" in changes:
        row.category = _normalize_alias_term(changes["category"]) if changes["category"] else None
    if "weight" in changes and changes["weight"] is not None:
        row.weight = changes["weight"]
    if "is_active" in changes and changes["is_active"] is not None:
        row.is_active = changes["is_active"]

    db.commit()
    db.refresh(row)
    AliasExpansionService.clear_cache()
    return _serialize_retrieval_alias(row)


@router.delete("/retrieval-aliases/{alias_id}")
def delete_retrieval_alias(
    alias_id: str,
    db: Session = Depends(get_db),
    admin_username: str = Depends(require_admin),
):
    try:
        row_id = UUID(alias_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid alias id")

    row = db.query(RetrievalAlias).filter(RetrievalAlias.id == row_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Alias not found")

    db.delete(row)
    db.commit()
    AliasExpansionService.clear_cache()
    return {"message": "Alias deleted successfully", "id": alias_id}


@router.post("/retrieval-aliases/import")
def import_retrieval_aliases(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    admin_username: str = Depends(require_admin),
):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    content = file.file.read()
    try:
        text_content = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="CSV must be UTF-8 encoded")

    reader = csv.DictReader(io.StringIO(text_content))
    header_map = _canonical_alias_csv_headers(reader.fieldnames)
    required = {"term", "aliases"}
    missing = required - set(header_map)
    if missing:
        accepted = "term/acronym/abbreviation and aliases/full form/meaning"
        raise HTTPException(
            status_code=400,
            detail=f"Missing CSV columns: {', '.join(sorted(missing))}. Accepted headers include: {accepted}",
        )

    created = 0
    updated = 0
    skipped = 0
    errors: list[str] = []

    for line_number, row_data in enumerate(reader, start=2):
        try:
            term = _normalize_alias_term(_alias_csv_value(row_data, header_map, "term") or "")
            aliases = _parse_alias_csv_values(_alias_csv_value(row_data, header_map, "aliases"))
            category_raw = _alias_csv_value(row_data, header_map, "category")
            category = _normalize_alias_term(category_raw) if category_raw and category_raw.strip() else None
            weight = _parse_alias_weight(_alias_csv_value(row_data, header_map, "weight"))
            is_active = _parse_alias_bool(_alias_csv_value(row_data, header_map, "is_active"), default=True)
        except Exception as exc:
            errors.append(f"Row {line_number}: {exc}")
            skipped += 1
            continue

        existing = db.query(RetrievalAlias).filter(func.lower(RetrievalAlias.term) == term).first()
        if existing:
            existing.aliases = aliases
            existing.category = category
            existing.weight = weight
            existing.is_active = is_active
            updated += 1
        else:
            db.add(
                RetrievalAlias(
                    term=term,
                    aliases=aliases,
                    category=category,
                    weight=weight,
                    is_active=is_active,
                )
            )
            created += 1

    db.commit()
    AliasExpansionService.clear_cache()
    return {"created": created, "updated": updated, "skipped": skipped, "errors": errors[:25]}


# -----------------------------------------------
# CRAWLER ROUTES
# -----------------------------------------------

from app.db.models import CrawlerJob

@router.get("/crawler/status")
def get_crawler_status(db: Session = Depends(get_db), admin_username: str = Depends(require_admin)):
    jobs = db.query(CrawlerJob).all()
    job_dict = {job.job_type: job for job in jobs}
    
    status = {"enabled": settings.CRAWLER_ENABLED}
    for jtype in ["full", "announcements"]:
        if jtype in job_dict:
            j = job_dict[jtype]
            display_status = j.status
            if j.status == "idle" and j.crawled_count:
                display_status = "finished" if j.max_pages and j.crawled_count >= j.max_pages else "stopped"
            status[jtype] = {
                "status": display_status,
                "raw_status": j.status,
                "crawled": j.crawled_count,
                "max": j.max_pages,
                "current_url": j.current_url or "",
                "last_run": j.last_run.isoformat() if j.last_run else None
            }
        else:
            status[jtype] = {
                "status": "idle", "crawled": 0, "max": 0, "current_url": "", "last_run": None
            }
    return status

def _require_crawler_enabled():
    if not settings.CRAWLER_ENABLED:
        raise HTTPException(status_code=503, detail="Crawler is disabled.")

@router.post("/crawler/cancel/{job_type}")
def cancel_crawler(
    job_type: str,
    db: Session = Depends(get_db),
    admin_username: str = Depends(require_admin)
):
    if job_type not in ["full", "announcements"]:
        raise HTTPException(status_code=400, detail="Invalid job type")
        
    job = db.query(CrawlerJob).filter_by(job_type=job_type).first()
    if job:
        job.status = "cancelled"
        
    from app.db.models import CrawlerQueue
    db.query(CrawlerQueue).filter_by(job_type=job_type).delete()
    db.commit()
    
    return {"message": f"{job_type} crawler cancellation requested."}

def _trigger_full_crawler_task(reset: bool = True):
    from app.core.scheduler import run_full_crawler
    import asyncio
    asyncio.run(run_full_crawler(reset=reset))

def _trigger_announcement_crawler_task():
    from app.core.scheduler import run_announcement_crawler
    import asyncio
    asyncio.run(run_announcement_crawler())

@router.post("/crawler/trigger-full")
def trigger_full_crawler(
    background_tasks: BackgroundTasks,
    admin_username: str = Depends(require_admin)
):
    _require_crawler_enabled()
    background_tasks.add_task(_trigger_full_crawler_task, True)
    return {"message": "Fresh full crawler started in the background."}

@router.post("/crawler/resume-full")
def resume_full_crawler(
    background_tasks: BackgroundTasks,
    admin_username: str = Depends(require_admin)
):
    _require_crawler_enabled()
    background_tasks.add_task(_trigger_full_crawler_task, False)
    return {"message": "Full crawler resume started in the background."}

@router.post("/crawler/trigger-announcements")
def trigger_announcement_crawler(
    background_tasks: BackgroundTasks,
    admin_username: str = Depends(require_admin)
):
    _require_crawler_enabled()
    background_tasks.add_task(_trigger_announcement_crawler_task)
    return {"message": "Announcement crawler started in the background."}
