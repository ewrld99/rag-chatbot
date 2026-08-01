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
