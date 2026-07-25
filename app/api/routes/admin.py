import os
import shutil
import logging
import time
from uuid import UUID, uuid4
from typing import Optional, List
from fastapi import APIRouter, BackgroundTasks, UploadFile, File, HTTPException, Depends, Request, Form
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
import base64
import csv
import io
import json

from app.utils.loaders import load_document
from app.utils.chunking import split_text
from app.core.config import settings
from app.services.embedding_service import EmbeddingServiceError, get_embeddings
from app.db.models import DocumentModel, DocumentChunk, RetrievalAlias
from app.db.session import get_db
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

router = APIRouter()
logger = logging.getLogger(__name__)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def serialize_document(document: DocumentModel, include_content: bool = False) -> dict:
    data = {
        "id": str(document.id),
        "title": document.title,
        "filename": document.filename,
        "file_path": document.file_path,
        "category": document.category,
        "uploaded_by": document.uploaded_by,
        "upload_date": document.upload_date,
        "status": document.status,
        "chunk_count": len(document.chunks),
    }
    if include_content:
        data["content"] = "\n\n".join(chunk.chunk_text for chunk in sorted(document.chunks, key=lambda c: c.chunk_index))
    return data


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

    # delete old chunks
    db.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).delete()

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


# ---------------------------------------
# Document CRUD
# ---------------------------------------
from sqlalchemy import Text, func, or_
from app.db.models import DocumentChunk

@router.get("/documents/", response_model=PaginatedDocumentResponse)
def list_documents(
    page: int = 1,
    limit: int = 50,
    search: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(DocumentModel, func.count(DocumentChunk.id).label("chunk_count")) \
        .outerjoin(DocumentChunk, DocumentModel.id == DocumentChunk.document_id)
        
    if search:
        search_filter = f"%{search}%"
        query = query.filter(
            (DocumentModel.filename.ilike(search_filter)) | 
            (DocumentModel.title.ilike(search_filter))
        )
        
    query = query.group_by(DocumentModel.id).order_by(DocumentModel.upload_date.desc())
    
    # Calculate totals
    total = query.count()
    total_pages = (total + limit - 1) // limit if limit > 0 else 1
    
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


@router.get("/documents/{document_id}", response_model=DocumentResponse)
def read_document(document_id: str, db: Session = Depends(get_db)):
    document = get_document_or_404(document_id, db)
    return serialize_document(document, include_content=True)


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

    return serialize_document(document, include_content=True)


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

    return serialize_document(document, include_content=True)


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


@router.post("/documents/delete-batch")
def delete_documents_batch(document_ids: List[str], db: Session = Depends(get_db)):
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
    programme: Optional[str] = Form(None, description="Programme this document belongs to, e.g. 'BSc Computer Science'"),
    year: Optional[int] = Form(None, description="Year of study this document applies to, e.g. 2"),
    db: Session = Depends(get_db)
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
        try:
            yield json.dumps({"progress": 15, "status": "Extracting text..."}) + "\\n"
            
            try:
                text = load_document(file_path, file_ext, strategy)
            except Exception as e:
                yield json.dumps({"error": f"Error reading file: {str(e)}"}) + "\\n"
                return

            if not text or not text.strip():
                yield json.dumps({"error": "No text extracted from document"}) + "\\n"
                return

            yield json.dumps({"progress": 25, "status": "Splitting text into chunks..."}) + "\\n"
            
            settings_svc = SettingsService(db)
            chunks = split_text(
                text,
                chunk_size=settings_svc.chunk_size,
                overlap=settings_svc.chunk_overlap,
            )

            if not chunks:
                yield json.dumps({"error": "Text could not be chunked"}) + "\\n"
                return

            yield json.dumps({"progress": 35, "status": f"Generating embeddings for {len(chunks)} chunks..."}) + "\\n"
            
            embeddings = []
            batch_size = 15 # default
            from app.services.embedding_service import EmbeddingService
            embedding_svc = EmbeddingService(db)
            
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
                }) + "\\n"
                
                try:
                    batch_embeddings = embedding_svc.embed_batch(batch_chunks)
                    embeddings.extend(batch_embeddings)
                except Exception as e:
                    logger.warning("Embedding failed for uploaded document %s: %s", filename, str(e))
                    yield json.dumps({"error": f"Embedding service error: {str(e)}"}) + "\\n"
                    return

            if len(embeddings) != len(chunks):
                yield json.dumps({"error": "Embedding service returned an unexpected number of vectors"}) + "\\n"
                return

            yield json.dumps({"progress": 90, "status": "Saving document to database..."}) + "\\n"

            document = DocumentModel(
                title=name_without_ext,
                filename=filename,
                file_path=file_path,
                category=file_ext,
                status="active"
            )
            db.add(document)
            db.flush() 
            
            # Build chunk metadata: always record ingestion time;
            # include programme and year when provided by the uploader.
            chunk_meta: dict = {}
            if programme:
                chunk_meta["programme"] = programme.strip()
            if year is not None:
                chunk_meta["year"] = str(year)

            db.bulk_insert_mappings(
                DocumentChunk,
                [
                    {
                        "document_id": document.id,
                        "chunk_text": chunk,
                        "embedding": embedding,
                        "chunk_index": i,
                        "metadata_": chunk_meta if chunk_meta else None,
                    }
                    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
                ]
            )
            db.commit()

            yield json.dumps({
                "progress": 100, 
                "status": "Done",
                "filename": filename,
                "chunks_stored": len(chunks)
            }) + "\\n"

        except Exception as e:
            logger.exception("Error during document ingestion generator")
            yield json.dumps({"error": f"Internal Server Error: {str(e)}"}) + "\\n"

    return StreamingResponse(ingest_generator(), media_type="application/x-ndjson")

# ---------------------------------------
# Reindex All
# ---------------------------------------
@router.post("/documents/reindex-all")
def reindex_all_documents(
    background_tasks: BackgroundTasks,
    force: bool = False,
    db: Session = Depends(get_db),
):
    """
    Queue a background re-ingestion job for documents.
    If force is True, reindexes all documents. Otherwise, only those marked needs_reindex.
    """
    query = db.query(DocumentModel)
    if not force:
        query = query.filter(DocumentModel.status == "needs_reindex")
    
    documents = query.all()

    if not documents:
        return {"queued": 0, "message": "No documents require reindexing."}

    # Immediately mark them as processing so the UI can reflect the change
    for doc in documents:
        doc.status = "processing"
    db.commit()

    doc_ids = [str(doc.id) for doc in documents]

    background_tasks.add_task(_reindex_documents_task, doc_ids)

    return {"queued": len(doc_ids), "message": f"{len(doc_ids)} document(s) queued for reindexing."}

@router.post("/documents/{document_id}/reindex")
def reindex_single_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Queue a background re-ingestion job for a specific document.
    """
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    doc = db.query(DocumentModel).filter(DocumentModel.id == doc_uuid).first()
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
                target_file_path = document.file_path
                if not target_file_path and document.filename:
                    import os
                    target_file_path = os.path.join("uploads", document.filename)

                content = None
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
                    logger.warning("Reindex: document %s content could not be loaded, failing safely to avoid corruption", doc_id)
                    document.status = "failed"
                    db.commit()
                    continue

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
    required = {"term", "aliases"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing CSV columns: {', '.join(sorted(missing))}")

    created = 0
    updated = 0
    skipped = 0
    errors: list[str] = []

    for line_number, row_data in enumerate(reader, start=2):
        try:
            term = _normalize_alias_term(row_data.get("term", ""))
            aliases = _parse_alias_csv_values(row_data.get("aliases"))
            category_raw = row_data.get("category")
            category = _normalize_alias_term(category_raw) if category_raw and category_raw.strip() else None
            weight = _parse_alias_weight(row_data.get("weight"))
            is_active = _parse_alias_bool(row_data.get("is_active"), default=True)
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
    
    status = {}
    for jtype in ["full", "announcements"]:
        if jtype in job_dict:
            j = job_dict[jtype]
            status[jtype] = {
                "status": j.status,
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

def _trigger_full_crawler_task():
    from app.core.scheduler import run_full_crawler
    import asyncio
    asyncio.run(run_full_crawler())

def _trigger_announcement_crawler_task():
    from app.core.scheduler import run_announcement_crawler
    import asyncio
    asyncio.run(run_announcement_crawler())

@router.post("/crawler/trigger-full")
def trigger_full_crawler(
    background_tasks: BackgroundTasks,
    admin_username: str = Depends(require_admin)
):
    background_tasks.add_task(_trigger_full_crawler_task)
    return {"message": "Full crawler started in the background."}

@router.post("/crawler/trigger-announcements")
def trigger_announcement_crawler(
    background_tasks: BackgroundTasks,
    admin_username: str = Depends(require_admin)
):
    background_tasks.add_task(_trigger_announcement_crawler_task)
    return {"message": "Announcement crawler started in the background."}
