"""Shared metadata builders for retrieval results."""

from __future__ import annotations

from typing import Any


def document_metadata(
    base_metadata: dict[str, Any] | None,
    *,
    source: str | None,
    chunk_index: int | None,
    page_number: int | None = None,
    document_title: str | None = None,
    source_url: str | None = None,
    document_status: str | None = None,
    content_hash: str | None = None,
    document_uploaded_at: Any = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = dict(base_metadata or {})
    metadata.update(
        {
            "source": source or "database",
            "chunk_index": chunk_index,
            "source_type": "document",
        }
    )
    if extra:
        metadata.update(extra)
    if page_number is not None:
        metadata["page_number"] = page_number
    if document_title:
        metadata["document_title"] = document_title
    if source_url:
        metadata["source_url"] = source_url
    metadata["document_status"] = document_status or "active"
    if content_hash:
        metadata["content_hash"] = content_hash
    if document_uploaded_at:
        metadata["document_uploaded_at"] = (
            document_uploaded_at.isoformat()
            if hasattr(document_uploaded_at, "isoformat")
            else str(document_uploaded_at)
        )
    return metadata


def faq_metadata(
    base_metadata: dict[str, Any] | None,
    *,
    faq_id: str,
    category: str | None,
    source: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = dict(base_metadata or {})
    metadata.update(
        {
            "source": source or (f"FAQ - {category}" if category else "FAQ"),
            "chunk_index": 0,
            "source_type": "faq",
            "faq_id": faq_id,
            "category": category,
        }
    )
    if extra:
        metadata.update(extra)
    return metadata
