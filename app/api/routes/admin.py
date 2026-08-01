import os
import shutil
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

from app.api.deps import require_admin_user
from app.utils.loaders import load_document
from app.utils.chunking import split_text
from app.core.config import settings
from app.services.embedding_service import EmbeddingServiceError, get_embeddings
from app.db.models import DocumentModel, DocumentChunk, RetrievalAlias, User
from app.db.session import get_db, SessionLocal
from app.schemas.document import (
    DocumentCreate, 
    DocumentUpdate, 
    DocumentResponse, 
    PaginatedDocumentResponse
)
from app.services.settings_service import SettingsService
from app.schemas.settings import SystemSettingResponse, SystemSettingUpdate
from app.services.faq_service import FAQService
from app.services.alias_expansion_service import AliasExpansionService
from app.services.almanac_service import build_almanac_chunks
from app.services.curriculum_service import build_curriculum_chunks

router = APIRouter()
logger = logging.getLogger(__name__)

UPLOAD_DIR = "uploads"
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
    settings_svc = SettingsService(db)
    chunk_size = settings_svc.chunk_size
    chunk_overlap = settings_svc.chunk_overlap

    chunks = split_text(
        content,
        chunk_size=chunk_size,
        overlap=chunk_overlap,
    )

    if not chunks:
        raise HTTPException(status_code=400, detail="Text could not be chunked")

    try:
        embeddings = get_embeddings(chunks, db)
    except EmbeddingServiceError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
    except Exception as e:
        logger.exception("Embedding failed while rebuilding document chunks")
        raise HTTPException(status_code=502, detail=f"Embedding service error: {str(e)}")

    if len(embeddings) != len(chunks):
        raise HTTPException(
            status_code=502,
            detail="Embedding service returned an unexpected number of vectors",
        )

    # delete old chunks and synchronize ORM session state
    db.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).delete(synchronize_session="fetch")

    new_chunks = [
        DocumentChunk(
            document_id=document.id,
            chunk_index=i,
            chunk_text=chunk,
            embedding=embedding,
            metadata_=extra_metadata or {},
        )
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    db.add_all(new_chunks)
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
    
    chunk_count = len(document.chunks)

    try:
        db.delete(document)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.exception("Database delete failed for document %s", document_id)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    return {
        "message": "Document deleted successfully",
        "id": str(document.id),
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
            deleted_count = db.query(DocumentModel).filter(DocumentModel.id.in_(valid_ids)).delete(synchronize_session=False)
            db.commit()
            return {"status": "ok", "deleted_count": deleted_count}
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
@router.post("/documents/")
def upload_document(
    file: UploadFile = File(...),
    strategy: str = Form("auto"),
    programme: Optional[str] = Form(None, description="Programme this document belongs to, e.g. 'BSc Computer Science'"),
    year: Optional[int] = Form(None, description="Year of study this document applies to, e.g. 2"),
    db: Session = Depends(get_db),
    admin_username: str = Depends(require_admin),
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

    # Pre-save the file directly in the main thread to avoid holding UploadFile open in the generator
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        copy_index = 1
        while os.path.exists(file_path):
            suffix = f"_copy{copy_index}"
            filename = f"{name_without_ext}{suffix}.{file_ext}"
            file_path = os.path.join(UPLOAD_DIR, filename)
            copy_index += 1

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save error: {str(e)}")


    def ingest_generator():
        # Open a fresh DB session that is owned entirely by this generator so it
        # is not closed by FastAPI's request-scoped dependency teardown before
        # the StreamingResponse body finishes streaming.
        with SessionLocal() as gen_db:
            try:
                yield json.dumps({"progress": 15, "status": "Extracting text..."}) + "\n"

                try:
                    text = load_document(file_path, file_ext, strategy)
                except Exception as e:
                    yield json.dumps({"error": f"Error reading file: {str(e)}"}) + "\n"
                    return

                if not text or not text.strip():
                    yield json.dumps({"error": "No text extracted from document"}) + "\n"
                    return

                yield json.dumps({"progress": 25, "status": "Splitting text into chunks..."}) + "\n"

                chunk_metadata: list[dict] | None = None
                if strategy == "almanac":
                    almanac_chunks = build_almanac_chunks(text)
                    chunks = [chunk for chunk, _metadata in almanac_chunks]
                    chunk_metadata = [metadata for _chunk, metadata in almanac_chunks]
                elif strategy == "curriculum":
                    curriculum_chunks = build_curriculum_chunks(text)
                    chunks = [chunk for chunk, _metadata in curriculum_chunks]
                    chunk_metadata = [metadata for _chunk, metadata in curriculum_chunks]
                else:
                    settings_svc = SettingsService(gen_db)
                    chunks = split_text(
                        text,
                        chunk_size=settings_svc.chunk_size,
                        overlap=settings_svc.chunk_overlap,
                    )

                if not chunks:
                    yield json.dumps({"error": "Text could not be chunked"}) + "\n"
                    return

                yield json.dumps({"progress": 35, "status": f"Generating embeddings for {len(chunks)} chunks..."}) + "\n"

                embeddings = []
                batch_size = 15  # default
                from app.services.embedding_service import EmbeddingService
                embedding_svc = EmbeddingService(gen_db)

                # Determine provider batch size
                if embedding_svc.provider != "local":
                    batch_size = embedding_svc.batch_size

                total_batches = (len(chunks) + batch_size - 1) // batch_size

                for i in range(total_batches):
                    start = i * batch_size
                    batch_chunks = chunks[start:start + batch_size]

                    pct = 35 + int(((i + 1) / total_batches) * 50)
                    yield json.dumps({
                        "progress": pct,
                        "status": f"Generating embeddings (batch {i + 1}/{total_batches})..."
                    }) + "\n"

                    try:
                        batch_embeddings = embedding_svc.embed_batch(batch_chunks)
                        embeddings.extend(batch_embeddings)
                    except Exception as e:
                        # Embedding failed: save the document row with status
                        # "embedding_failed" so it shows up in the admin UI and
                        # the admin can retry it once the embedding service recovers.
                        # Without this the file exists on disk but is invisible.
                        logger.warning(
                            "Embedding failed for uploaded document %s: %s — "
                            "saving document record with status='embedding_failed' "
                            "so it can be retried from the admin panel.",
                            filename, str(e),
                        )
                        try:
                            failed_doc = DocumentModel(
                                title=name_without_ext,
                                filename=filename,
                                file_path=file_path,
                                category=file_ext,
                                status="embedding_failed",
                            )
                            gen_db.add(failed_doc)
                            gen_db.commit()
                            yield json.dumps({
                                "error": f"Embedding service error: {str(e)}",
                                "status": "embedding_failed",
                                "message": (
                                    "The document was saved but could not be embedded. "
                                    "It will appear in the admin document list with status "
                                    "'embedding_failed'. Use Retry Failed Documents to "
                                    "re-process it once the embedding service is available."
                                ),
                            }) + "\n"
                        except Exception as db_exc:
                            logger.error(
                                "Could not save embedding_failed record for %s: %s",
                                filename, db_exc,
                            )
                            yield json.dumps({"error": f"Embedding service error: {str(e)}"}) + "\n"
                        return

                if len(embeddings) != len(chunks):
                    yield json.dumps({"error": "Embedding service returned an unexpected number of vectors"}) + "\n"
                    return

                yield json.dumps({"progress": 90, "status": "Saving document to database..."}) + "\n"

                document = DocumentModel(
                    title=name_without_ext,
                    filename=filename,
                    file_path=file_path,
                    category=file_ext,
                    status="active"
                )
                gen_db.add(document)
                gen_db.flush()

                # Build chunk metadata; include programme and year when provided by the uploader.
                chunk_meta: dict = {}
                if programme:
                    chunk_meta["programme"] = programme.strip()
                if year is not None:
                    chunk_meta["year"] = str(year)
                if strategy == "almanac":
                    chunk_meta["category"] = "academic_calendar"
                    chunk_meta["record_type"] = "almanac_event"
                elif strategy == "curriculum":
                    chunk_meta["category"] = "curriculum"
                    chunk_meta["record_type"] = "course"

                gen_db.bulk_insert_mappings(
                    DocumentChunk,
                    [
                        {
                            "document_id": document.id,
                            "chunk_text": chunk,
                            "embedding": embedding,
                            "chunk_index": i,
                            "metadata_": {
                                **chunk_meta,
                                **(chunk_metadata[i] if chunk_metadata else {}),
                            } if (chunk_meta or chunk_metadata) else None,
                        }
                        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
                    ]
                )
                gen_db.commit()

                yield json.dumps({
                    "progress": 100,
                    "status": "Done",
                    "filename": filename,
                    "chunks_stored": len(chunks)
                }) + "\n"

            except Exception as e:
                logger.exception("Error during document ingestion generator")
                yield json.dumps({"error": f"Internal Server Error: {str(e)}"}) + "\n"

    return StreamingResponse(ingest_generator(), media_type="application/x-ndjson")

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

    query.update({DocumentModel.status: "processing"}, synchronize_session=False)
    db.commit()

    background_tasks.add_task(_reindex_documents_task, doc_ids)

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

    doc.status = "processing"
    db.commit()

    background_tasks.add_task(_reindex_documents_task, [str(doc.id)])

    return {"message": f"Document '{doc.filename}' queued for reindexing."}


def _reindex_documents_task(doc_ids: list[str]) -> None:
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
                    target_file_path = os.path.join("uploads", document.filename)

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
        doc.status = "processing"
    db.commit()

    doc_ids = [str(doc.id) for doc in failed_docs]
    background_tasks.add_task(_reindex_documents_task, doc_ids)

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
