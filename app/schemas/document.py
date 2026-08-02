from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field

class DocumentBase(BaseModel):
    title: Optional[str] = None
    filename: Optional[str] = None
    file_path: Optional[str] = None
    category: Optional[str] = None
    status: Optional[str] = "active"


class DocumentCreate(DocumentBase):
    content: str = Field(..., min_length=1)  # manual entry will chunk this


class DocumentUpdate(DocumentBase):
    content: Optional[str] = Field(default=None, min_length=1)


class DocumentResponse(DocumentBase):
    id: str
    source_url: Optional[str] = None
    uploaded_by: Optional[int] = None
    upload_date: datetime
    chunk_count: int = 0
    content: Optional[str] = None
    quality_status: str = "unchecked"
    indexing_status: str = "idle"
    quality_checked_at: Optional[datetime] = None
    ingestion_version: str = "1"
    duplicate_of_document_id: Optional[str] = None
    quality_summary: Dict[str, Any] = Field(default_factory=dict)
    storage_state: str = "not_applicable"
    storage_error: Optional[str] = None

    class Config:
        from_attributes = True


class DocumentChunkResponse(BaseModel):
    id: str
    document_id: str
    chunk_index: int
    chunk_text: str
    page_number: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True

class PaginatedDocumentResponse(BaseModel):
    items: List[DocumentResponse]
    total: int
    page: int
    limit: int
    total_pages: int


class DocumentQualityResponse(BaseModel):
    document_id: str
    lifecycle_status: str
    indexing_status: str
    quality_status: str
    quality_checked_at: Optional[datetime] = None
    ingestion_version: str
    duplicate_of_document_id: Optional[str] = None
    report: Dict[str, Any] = Field(default_factory=dict)


class DocumentQualityApproval(BaseModel):
    reason: str = Field(..., min_length=3, max_length=500)


class DocumentAuditRequest(BaseModel):
    document_ids: List[str] = Field(default_factory=list, max_length=1000)
