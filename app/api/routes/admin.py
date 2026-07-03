import os
import shutil
import logging
import time
from uuid import UUID, uuid4
from typing import Optional, List
from fastapi import APIRouter, BackgroundTasks, UploadFile, File, HTTPException, Depends, Request, Form
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
import base64
import json

from app.utils.loaders import load_document
from app.utils.chunking import split_text
from app.core.config import settings
from app.services.embedding_service import EmbeddingServiceError, get_embeddings
from app.db.models import DocumentModel, DocumentChunk
from app.db.session import get_db
from app.schemas.document import DocumentCreate, DocumentResponse, DocumentUpdate
from app.services.settings_service import SettingsService
from app.schemas.settings import SystemSettingResponse, SystemSettingUpdate
from app.services.faq_service import FAQService
from app.services.timetable_fetcher import TimetableFetcherService, TimetableFetchError

router = APIRouter()
logger = logging.getLogger(__name__)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def serialize_document(document: DocumentModel) -> dict:
    return {
        "id": str(document.id),
        "title": document.title,
        "filename": document.filename,
        "file_path": document.file_path,
        "category": document.category,
        "uploaded_by": document.uploaded_by,
        "upload_date": document.upload_date,
        "status": document.status,
        "chunk_count": len(document.chunks),
        # the frontend edit form expects content
        "content": "\n\n".join(chunk.chunk_text for chunk in sorted(document.chunks, key=lambda c: c.chunk_index)),
    }


def get_document_or_404(document_id: str, db: Session) -> DocumentModel:
    try:
        row_id = UUID(document_id)
        document = db.query(DocumentModel).filter(DocumentModel.id == row_id).first()
    except ValueError:
        # fallback to finding by filename
        document = db.query(DocumentModel).filter(DocumentModel.filename == document_id).first()
        
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    return document


def rebuild_document_chunks(
    document: DocumentModel,
    content: str,
    db: Session,
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

    # delete old chunks
    db.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).delete()

    new_chunks = [
        DocumentChunk(
            document_id=document.id,
            chunk_index=i,
            chunk_text=chunk,
            embedding=embedding,
        )
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    db.add_all(new_chunks)


# ---------------------------------------
# Document CRUD
# ---------------------------------------
@router.get("/documents/", response_model=list[DocumentResponse])
def list_documents(db: Session = Depends(get_db)):
    documents = db.query(DocumentModel).order_by(DocumentModel.upload_date.desc()).all()
    return [serialize_document(doc) for doc in documents]


@router.get("/documents/{document_id}", response_model=DocumentResponse)
def read_document(document_id: str, db: Session = Depends(get_db)):
    document = get_document_or_404(document_id, db)
    return serialize_document(document)


@router.post("/documents/manual", response_model=DocumentResponse, status_code=201)
def create_document(payload: DocumentCreate, db: Session = Depends(get_db)):
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

    return serialize_document(document)


@router.put("/documents/{document_id}", response_model=DocumentResponse)
def update_document(document_id: str, payload: DocumentUpdate, db: Session = Depends(get_db)):
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

    return serialize_document(document)


@router.delete("/documents/{document_id}")
def delete_document(document_id: str, db: Session = Depends(get_db)):
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


def get_admin_username(request: Request) -> str:
    auth = request.headers.get("Authorization")
    if auth and auth.startswith("Bearer "):
        token = auth.split(" ")[1]
        try:
            parts = token.split(".")
            if len(parts) >= 2:
                payload = parts[1]
                payload += "=" * ((4 - len(payload) % 4) % 4)
                decoded = base64.urlsafe_b64decode(payload)
                data = json.loads(decoded)
                return data.get("username", "Unknown Admin")
        except Exception:
            pass
    return "Unknown Admin"

def require_admin(admin_username: str = Depends(get_admin_username)) -> str:
    if admin_username == "Unknown Admin":
        raise HTTPException(status_code=401, detail="Unauthorized: Admin access required")
    return admin_username


# ---------------------------------------
# Settings CRUD
# ---------------------------------------
@router.get("/settings/", response_model=list[SystemSettingResponse])
def list_settings(db: Session = Depends(get_db)):
    settings_svc = SettingsService(db)
    return settings_svc.get_all()


@router.put("/settings/{key}", response_model=SystemSettingResponse)
def update_setting(
    key: str, 
    payload: SystemSettingUpdate, 
    db: Session = Depends(get_db),
    admin_username: str = Depends(get_admin_username)
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
    db: Session = Depends(get_db)
):
    """
    Full ingestion pipeline:
    - Save file
    - Extract text
    - Chunk text
    - Generate embeddings
    - Store in PostgreSQL (pgvector)
    """

    started_at = time.perf_counter()

    # ---------------------------------------
    # Validate file
    # ---------------------------------------
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a name")

    filename = os.path.basename(file.filename).lower()

    if not filename.endswith((".pdf", ".docx", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, and TXT allowed")

    file_ext = filename.split(".")[-1]

    # ---------------------------------------
    # Prevent overwrite (IMPORTANT)
    # ---------------------------------------
    file_path = os.path.join(UPLOAD_DIR, filename)
    name_without_ext = os.path.splitext(filename)[0]

    if os.path.exists(file_path):
        copy_index = 1

        while os.path.exists(file_path):
            suffix = f"_copy{copy_index}"
            filename = f"{name_without_ext}{suffix}.{file_ext}"
            file_path = os.path.join(UPLOAD_DIR, filename)
            copy_index += 1

    # ---------------------------------------
    # Save file
    # ---------------------------------------
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save error: {str(e)}")

    # ---------------------------------------
    # Extract text
    # ---------------------------------------
    try:
        text = load_document(file_path, file_ext, strategy)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="No text extracted from document")

    # ---------------------------------------
    # Split into chunks
    # ---------------------------------------
    extracted_at = time.perf_counter()

    settings_svc = SettingsService(db)
    chunk_size = settings_svc.chunk_size
    chunk_overlap = settings_svc.chunk_overlap

    chunks = split_text(
        text,
        chunk_size=chunk_size,
        overlap=chunk_overlap,
    )

    if not chunks:
        raise HTTPException(status_code=400, detail="Text could not be chunked")

    # ---------------------------------------
    # Embed + Store
    # ---------------------------------------
    chunked_at = time.perf_counter()

    try:
        embeddings = get_embeddings(chunks, db)
    except EmbeddingServiceError as e:
        logger.warning("Embedding failed for uploaded document %s: %s", filename, str(e))
        raise HTTPException(status_code=e.status_code, detail=str(e))
    except Exception as e:
        logger.exception("Embedding failed for uploaded document %s", filename)
        raise HTTPException(status_code=502, detail=f"Embedding service error: {str(e)}")

    if len(embeddings) != len(chunks):
        raise HTTPException(
            status_code=502,
            detail="Embedding service returned an unexpected number of vectors",
        )

    embedded_at = time.perf_counter()

    try:
        document = DocumentModel(
            title=name_without_ext,
            filename=filename,
            file_path=file_path,
            category=file_ext,
            status="active"
        )
        db.add(document)
        db.flush() # flush to get document.id
        
        db.bulk_insert_mappings(
            DocumentChunk,
            [
                {
                    "document_id": document.id,
                    "chunk_text": chunk,
                    "embedding": embedding,
                    "chunk_index": i,
                }
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
            ],
        )
        db.commit()

    except Exception as e:
        db.rollback()
        logger.exception("Database insert failed for uploaded document %s", filename)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    finished_at = time.perf_counter()
    logger.info(
        "Uploaded %s: %d chunks, extract=%.2fs chunk=%.2fs embed=%.2fs db=%.2fs total=%.2fs",
        filename,
        len(chunks),
        extracted_at - started_at,
        chunked_at - extracted_at,
        embedded_at - chunked_at,
        finished_at - embedded_at,
        finished_at - started_at,
    )

    return {
        "message": "Document processed successfully",
        "filename": filename,
        "documents_stored": 1,
        "chunks_stored": len(chunks)
    }


# ---------------------------------------
# Reindex All
# ---------------------------------------
@router.post("/documents/reindex-all")
def reindex_all_documents(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Queue a background re-ingestion job for every document marked needs_reindex.
    Returns immediately with the count of queued documents.
    """
    documents = (
        db.query(DocumentModel)
        .all()
    )

    if not documents:
        return {"queued": 0, "message": "No documents require reindexing."}

    # Immediately mark them as processing so the UI can reflect the change
    for doc in documents:
        doc.status = "processing"
    db.commit()

    doc_ids = [str(doc.id) for doc in documents]

    background_tasks.add_task(_reindex_documents_task, doc_ids)

    return {"queued": len(doc_ids), "message": f"{len(doc_ids)} document(s) queued for reindexing."}


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

                # Reconstruct text from existing chunks (ordered)
                existing_chunks = (
                    db.query(DocumentChunk)
                    .filter(DocumentChunk.document_id == document.id)
                    .order_by(DocumentChunk.chunk_index)
                    .all()
                )

                if not existing_chunks:
                    logger.warning("Reindex: document %s has no chunks, skipping", doc_id)
                    document.status = "failed"
                    db.commit()
                    continue

                # Try to reload from the original file if it still exists
                if document.file_path and document.file_path.endswith((".pdf", ".docx", ".txt")):
                    try:
                        file_ext = document.file_path.rsplit(".", 1)[-1].lower()
                        content = load_document(document.file_path, file_ext)
                    except Exception:
                        # Fallback to reconstructing from stored chunk text
                        content = "\n".join(c.chunk_text for c in existing_chunks)
                else:
                    content = "\n".join(c.chunk_text for c in existing_chunks)

                if not content or not content.strip():
                    document.status = "failed"
                    db.commit()
                    continue

                new_chunks = split_text(content, chunk_size=chunk_size, overlap=chunk_overlap)
                if not new_chunks:
                    document.status = "failed"
                    db.commit()
                    continue

                embeddings = get_embeddings(new_chunks, db)

                if len(embeddings) != len(new_chunks):
                    raise ValueError("Embedding count mismatch")

                # Replace chunks
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
                        }
                        for i, (chunk, embedding) in enumerate(zip(new_chunks, embeddings))
                    ],
                )

                document.status = "active"
                db.commit()
                logger.info("Reindex: document %s completed (%d chunks)", doc_id, len(new_chunks))

            except Exception as exc:
                logger.exception("Reindex: document %s failed: %s", doc_id, exc)
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


# -----------------------------------------------
# TIMETABLE FETCHER ROUTES
# -----------------------------------------------

class TimetableFetchRequest(BaseModel):
    year: str
    semester: str
    category: str
    option: str = "programme"  # programme | course | room | instructor
    data: List[str]
    label: Optional[str] = None  # Custom document label
    strategy: str = "timetable"  # auto | fast | timetable


@router.get("/timetable/years")
def timetable_get_years(admin_username: str = Depends(require_admin)):
    """Return static list of academic years."""
    return [
        {"value": "11", "label": "2024/2025"},
        {"value": "12", "label": "2025/2026"},
    ]


@router.get("/timetable/semesters")
def timetable_get_semesters(
    year: str,
    db: Session = Depends(get_db),
    admin_username: str = Depends(require_admin)
):
    """Proxy UDOM AJAX call to get semesters for a year."""
    svc = TimetableFetcherService(db)
    try:
        return svc.get_semesters(year)
    finally:
        svc.close()


@router.get("/timetable/categories")
def timetable_get_categories(
    year: str,
    semester: str,
    db: Session = Depends(get_db),
    admin_username: str = Depends(require_admin)
):
    """Proxy UDOM AJAX call to get timetable categories for a year + semester."""
    svc = TimetableFetcherService(db)
    try:
        return svc.get_categories(year, semester)
    finally:
        svc.close()


@router.get("/timetable/option-types")
def timetable_get_option_types(
    year: str,
    semester: str,
    category: str,
    db: Session = Depends(get_db),
    admin_username: str = Depends(require_admin)
):
    """Proxy UDOM AJAX call to get download option types (By Programme / By Course / etc.)."""
    svc = TimetableFetcherService(db)
    try:
        return svc.get_option_types(year, semester, category)
    finally:
        svc.close()


@router.get("/timetable/data-options")
def timetable_get_data_options(
    year: str,
    semester: str,
    category: str,
    option: str,
    db: Session = Depends(get_db),
    admin_username: str = Depends(require_admin)
):
    """Proxy UDOM AJAX call to get programme/course/room/instructor list."""
    svc = TimetableFetcherService(db)
    try:
        return svc.get_data_options(year, semester, category, option)
    except TimetableFetchError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
    finally:
        svc.close()


def _fetch_timetable_task(
    year: str,
    semester: str,
    category: str,
    option: str,
    data: List[str],
    label: str,
    admin_username: str,
    strategy: str = "timetable",
):
    """Background task: download PDF from UDOM and ingest into RAG pipeline."""
    from app.db.session import SessionLocal
    from app.db.models import User

    db = SessionLocal()
    svc = None
    try:
        admin_user = db.query(User).filter(User.username == admin_username).first()
        admin_id = admin_user.id if admin_user else None

        svc = TimetableFetcherService(db)
        pdf_bytes = svc.download_pdf(year, semester, category, option, data)
        result = svc.ingest_pdf(pdf_bytes, label, admin_id=admin_id, strategy=strategy)
        logger.info("Timetable fetch+ingest completed: %s", result)
    except TimetableFetchError as e:
        logger.error("Timetable fetch failed: %s", e)
    except Exception as e:
        logger.exception("Unexpected error during timetable fetch: %s", e)
    finally:
        if svc is not None:
            svc.close()
        db.close()


@router.post("/timetable/fetch")
def fetch_timetable(
    payload: TimetableFetchRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    admin_username: str = Depends(require_admin)
):
    """
    Trigger background download + ingestion of a UDOM timetable PDF.
    Returns immediately; the actual work runs asynchronously.
    """
    if not payload.data:
        raise HTTPException(status_code=400, detail="At least one data selection is required.")

    label = payload.label or f"Timetable {payload.year} Sem{payload.semester} {payload.option.title()}"

    background_tasks.add_task(
        _fetch_timetable_task,
        payload.year,
        payload.semester,
        payload.category,
        payload.option,
        payload.data,
        label,
        admin_username,
        payload.strategy,
    )

    return {
        "message": "Timetable fetch started in background.",
        "label": label,
        "status": "processing",
    }
