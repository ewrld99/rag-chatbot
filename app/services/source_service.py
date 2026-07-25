from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, unquote, urlparse

from langchain_core.documents import Document

from app.core.config import settings


_UPLOAD_SUFFIXES = {".pdf", ".docx", ".doc", ".txt", ".md", ".xlsx", ".csv"}


def document_name(metadata: dict[str, Any]) -> str:
    """Return a safe display name without exposing a source URL."""
    raw_name = (
        metadata.get("source")
        or metadata.get("document_title")
        or metadata.get("source_url")
        or "UDOM document"
    )
    value = str(raw_name).strip()
    parsed = urlparse(value)

    if parsed.scheme and parsed.netloc:
        value = unquote(Path(parsed.path).name)
    else:
        value = Path(value.replace("\\", "/")).name

    value = re.sub(r"[\x00-\x1f\x7f]+", " ", value)
    return re.sub(r"\s+", " ", value).strip() or "UDOM document"


def _validated_http_url(value: object) -> str | None:
    if not value:
        return None

    candidate = str(value).strip()
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    if parsed.username or parsed.password:
        return None
    return candidate


def trusted_source_url(metadata: dict[str, Any], name: str) -> str | None:
    """Build a link from server-controlled metadata, never from model output."""
    external_url = _validated_http_url(metadata.get("source_url"))
    if external_url:
        return external_url

    source_url = _validated_http_url(metadata.get("source"))
    if source_url:
        return source_url

    if Path(name).suffix.lower() not in _UPLOAD_SUFFIXES:
        return None

    uploads_root = Path(settings.UPLOADS_DIR).resolve()
    local_file = (uploads_root / name).resolve()
    try:
        local_file.relative_to(uploads_root)
    except ValueError:
        return None

    if not local_file.is_file():
        return None

    return f"{settings.BASE_URL.rstrip('/')}/uploads/{quote(name)}"


def format_source_records(documents: Iterable[Document]) -> list[dict[str, Any]]:
    """Return one deterministic, trusted source record per document."""
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for doc in documents:
        metadata = dict(doc.metadata or {})
        name = document_name(metadata)
        url = trusted_source_url(metadata, name)
        document_id_value = metadata.get("document_id")
        document_id = str(document_id_value) if document_id_value is not None else None
        key = (document_id or "", name.casefold(), url or "")
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "document_id": document_id,
                "name": name,
                "url": url,
            }
        )

    return records
