from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import re
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.db.models import DocumentChunk, DocumentModel
from app.utils.chunking import PreparedChunk, normalize_chunk_text
from app.utils.loaders import ExtractedDocument


QUALITY_VERSION = "1"
_LIST_ITEM_RE = re.compile(
    r"(?im)^\s*(?:[-*]\s+|(?:\d+|[ivxlcdm]+|[a-z])[.)]\s+)"
)
_MOJIBAKE_RE = re.compile(r"(?:\ufffd|\u00c3.|\u00c2.|\u00e2\u20ac.)")
_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


@dataclass(frozen=True)
class QualityIssue:
    code: str
    severity: str
    message: str
    page_number: int | None = None
    chunk_index: int | None = None
    sample: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "code": self.code,
                "severity": self.severity,
                "message": self.message,
                "page_number": self.page_number,
                "chunk_index": self.chunk_index,
                "sample": self.sample,
            }.items()
            if value is not None
        }


@dataclass(frozen=True)
class DocumentQualityResult:
    status: str
    content_hash: str
    duplicate_of_document_id: UUID | None
    report: dict[str, Any]

    @property
    def blocks_activation(self) -> bool:
        return self.status == "review"


def normalized_document_content(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def normalized_document_hash(text: str) -> str:
    normalized = normalized_document_content(text)
    return sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""


class DocumentQualityService:
    """Evaluate extraction and chunk quality before a document is searchable."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def evaluate(
        self,
        extracted: ExtractedDocument,
        chunks: list[PreparedChunk],
        *,
        strategy: str = "auto",
        removed_duplicate_chunks: int = 0,
        current_document_id: UUID | None = None,
    ) -> DocumentQualityResult:
        issues: list[QualityIssue] = []
        content = extracted.content
        normalized = normalized_document_content(content)
        content_hash = normalized_document_hash(content)
        page_count = len(extracted.pages)
        non_empty_pages = sum(bool(page.content.strip()) for page in extracted.pages)
        ocr_pages = [page.page_number for page in extracted.pages if page.ocr_required]
        ocr_ratio = len(ocr_pages) / page_count if page_count else 0.0
        total_characters = len(content)
        printable_characters = sum(
            character.isprintable() or character in "\n\r\t"
            for character in content
        )
        printable_ratio = printable_characters / total_characters if total_characters else 0.0
        garbled_count = len(_MOJIBAKE_RE.findall(content))
        garbled_ratio = garbled_count / total_characters if total_characters else 0.0

        chunk_lengths = [len(chunk.text) for chunk in chunks]
        short_chunks = sum(length < 120 for length in chunk_lengths)
        short_ratio = short_chunks / len(chunks) if chunks else 0.0
        normalized_chunks = [normalize_chunk_text(chunk.text) for chunk in chunks]
        remaining_duplicates = len(normalized_chunks) - len(set(normalized_chunks))
        original_chunk_count = len(chunks) + max(0, removed_duplicate_chunks)
        pre_dedup_ratio = (
            removed_duplicate_chunks / original_chunk_count
            if original_chunk_count
            else 0.0
        )
        remaining_duplicate_ratio = (
            remaining_duplicates / len(chunks) if chunks else 0.0
        )
        chunks_with_pages = sum(chunk.page_number is not None for chunk in chunks)
        page_provenance_ratio = chunks_with_pages / len(chunks) if chunks else 0.0
        table_count = sum(len(page.tables) for page in extracted.pages)
        list_item_count = sum(len(_LIST_ITEM_RE.findall(chunk.text)) for chunk in chunks)
        source_tokens = set(_WORD_RE.findall(normalized))
        chunk_tokens = set(
            _WORD_RE.findall(" ".join(normalize_chunk_text(chunk.text) for chunk in chunks))
        )
        source_token_coverage_ratio = (
            len(source_tokens & chunk_tokens) / len(source_tokens)
            if source_tokens
            else 0.0
        )

        if not normalized or not chunks:
            issues.append(
                QualityIssue(
                    code="NO_USABLE_CONTENT",
                    severity="blocker",
                    message="No usable text chunks were produced from the document.",
                )
            )
        if extracted.file_type == "pdf" and ocr_ratio > 0.30:
            issues.append(
                QualityIssue(
                    code="OCR_REQUIRED",
                    severity="blocker",
                    message="More than 30% of PDF pages require OCR.",
                    page_number=next((page for page in ocr_pages if page is not None), None),
                )
            )
        elif extracted.file_type == "pdf" and ocr_pages:
            issues.append(
                QualityIssue(
                    code="OCR_CANDIDATE_PAGES",
                    severity="warning",
                    message="Some PDF pages contain images but little extractable text.",
                    page_number=next((page for page in ocr_pages if page is not None), None),
                )
            )
        if total_characters and printable_ratio < 0.90:
            issues.append(
                QualityIssue(
                    code="LOW_PRINTABLE_RATIO",
                    severity="blocker",
                    message="Less than 90% of extracted characters are printable.",
                )
            )
        if total_characters and garbled_ratio > 0.02:
            issues.append(
                QualityIssue(
                    code="GARBLED_TEXT",
                    severity="blocker",
                    message="More than 2% of extracted text appears garbled.",
                )
            )
        if remaining_duplicate_ratio > 0.20:
            issues.append(
                QualityIssue(
                    code="EXCESSIVE_CHUNK_DUPLICATION",
                    severity="blocker",
                    message="More than 20% of prepared chunks remain exact duplicates.",
                )
            )
        elif pre_dedup_ratio >= 0.05:
            issues.append(
                QualityIssue(
                    code="DUPLICATE_CHUNKS_REMOVED",
                    severity="warning",
                    message=f"Removed {removed_duplicate_chunks} exact duplicate chunks before embedding.",
                )
            )
        if strategy in {"almanac", "curriculum"} and not chunks:
            issues.append(
                QualityIssue(
                    code="STRUCTURED_PARSER_EMPTY",
                    severity="blocker",
                    message=f"The {strategy} parser produced no valid rows.",
                )
            )
        if extracted.file_type == "pdf" and chunks and page_provenance_ratio < 0.80:
            issues.append(
                QualityIssue(
                    code="LOW_PAGE_PROVENANCE",
                    severity="warning",
                    message="Fewer than 80% of PDF chunks retain a source page number.",
                )
            )
        if chunks and short_ratio > 0.20:
            issues.append(
                QualityIssue(
                    code="MANY_SHORT_CHUNKS",
                    severity="warning",
                    message="More than 20% of chunks contain fewer than 120 characters.",
                )
            )

        duplicate_document = self._exact_duplicate(content_hash, current_document_id)
        if duplicate_document is not None:
            issues.append(
                QualityIssue(
                    code="DUPLICATE_DOCUMENT",
                    severity="blocker",
                    message="The extracted content exactly duplicates an active document.",
                    sample=duplicate_document.title or duplicate_document.filename,
                )
            )

        simhash = self._simhash(normalized)
        near_duplicate = self._near_duplicate(simhash, current_document_id)
        if duplicate_document is None and near_duplicate is not None:
            issues.append(
                QualityIssue(
                    code="POSSIBLE_NEAR_DUPLICATE",
                    severity="warning",
                    message="The document is highly similar to another indexed document.",
                    sample=near_duplicate.title or near_duplicate.filename,
                )
            )

        blocker_count = sum(issue.severity == "blocker" for issue in issues)
        warning_count = sum(issue.severity == "warning" for issue in issues)
        status = "review" if blocker_count else "warning" if warning_count else "passed"
        metrics = {
            "page_count": page_count,
            "non_empty_pages": non_empty_pages,
            "ocr_required_pages": [page for page in ocr_pages if page is not None],
            "ocr_required_ratio": round(ocr_ratio, 4),
            "extracted_characters": total_characters,
            "printable_ratio": round(printable_ratio, 4),
            "garbled_ratio": round(garbled_ratio, 6),
            "table_count": table_count,
            "detected_list_items": list_item_count,
            "chunk_count": len(chunks),
            "chunk_length_min": min(chunk_lengths, default=0),
            "chunk_length_avg": round(sum(chunk_lengths) / len(chunk_lengths), 1) if chunk_lengths else 0.0,
            "chunk_length_max": max(chunk_lengths, default=0),
            "short_chunk_count": short_chunks,
            "pre_deduplication_chunk_count": original_chunk_count,
            "removed_duplicate_chunks": removed_duplicate_chunks,
            "remaining_duplicate_chunks": remaining_duplicates,
            "page_provenance_ratio": round(page_provenance_ratio, 4),
            "source_token_coverage_ratio": round(source_token_coverage_ratio, 4),
            "document_simhash": str(simhash),
        }
        report = {
            "version": QUALITY_VERSION,
            "status": status,
            "strategy": strategy,
            "file_type": extracted.file_type,
            "extraction_method": extracted.extraction_method,
            "blocker_count": blocker_count,
            "warning_count": warning_count,
            "metrics": metrics,
            "issues": [issue.to_dict() for issue in issues],
            "samples": {
                "first_chunk": chunks[0].text[:500] if chunks else "",
                "last_chunk": chunks[-1].text[:500] if chunks else "",
            },
        }
        return DocumentQualityResult(
            status=status,
            content_hash=content_hash,
            duplicate_of_document_id=duplicate_document.id if duplicate_document else None,
            report=report,
        )

    def audit_existing_chunks(self, document: DocumentModel) -> DocumentQualityResult:
        from app.utils.loaders import ExtractedDocument, ExtractedPage

        rows = sorted(document.chunks, key=lambda chunk: chunk.chunk_index)
        content = "\n\n".join(chunk.chunk_text for chunk in rows)
        extracted = ExtractedDocument(
            pages=(
                ExtractedPage(
                    page_number=None,
                    text=content,
                    extraction_method="legacy_chunks",
                ),
            ),
            file_type=document.category or "text",
            extraction_method="legacy_chunks",
        )
        chunks = [
            PreparedChunk(
                text=row.chunk_text,
                page_number=row.page_number,
                metadata=dict(row.metadata_ or {}),
            )
            for row in rows
        ]
        return self.evaluate(
            extracted,
            chunks,
            strategy="audit",
            current_document_id=document.id,
        )

    def _exact_duplicate(
        self,
        content_hash: str,
        current_document_id: UUID | None,
    ) -> DocumentModel | None:
        if not content_hash:
            return None
        query = self.db.query(DocumentModel).filter(
            DocumentModel.content_hash == content_hash,
            DocumentModel.status == "active",
            DocumentModel.quality_status != "review",
        )
        if current_document_id is not None:
            query = query.filter(DocumentModel.id != current_document_id)
        return query.order_by(DocumentModel.upload_date.asc(), DocumentModel.id.asc()).first()

    def _near_duplicate(
        self,
        simhash: int,
        current_document_id: UUID | None,
    ) -> DocumentModel | None:
        if not simhash:
            return None
        query = self.db.query(DocumentModel).filter(
            DocumentModel.status == "active",
            DocumentModel.quality_status != "review",
        )
        if current_document_id is not None:
            query = query.filter(DocumentModel.id != current_document_id)
        for document in query.yield_per(100):
            report = dict(document.quality_report or {})
            raw_other = dict(report.get("metrics") or {}).get("document_simhash")
            try:
                other = int(str(raw_other))
            except (TypeError, ValueError):
                continue
            if (simhash ^ other).bit_count() <= 6:
                return document
        return None

    @staticmethod
    def _simhash(text: str) -> int:
        tokens = _WORD_RE.findall(text)
        if not tokens:
            return 0
        features = tokens if len(tokens) < 5 else [" ".join(tokens[index:index + 5]) for index in range(len(tokens) - 4)]
        vector = [0] * 64
        for feature in features:
            value = int.from_bytes(sha256(feature.encode("utf-8")).digest()[:8], "big")
            for bit in range(64):
                vector[bit] += 1 if value & (1 << bit) else -1
        result = 0
        for bit, weight in enumerate(vector):
            if weight >= 0:
                result |= 1 << bit
        return result
