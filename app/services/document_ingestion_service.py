from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.time import application_now
from app.db.models import DocumentChunk, DocumentModel
from app.services.almanac_service import build_almanac_chunks
from app.services.curriculum_service import build_curriculum_chunks
from app.services.document_quality_service import (
    QUALITY_VERSION,
    DocumentQualityResult,
    DocumentQualityService,
)
from app.services.settings_service import SettingsService
from app.utils.chunking import (
    PreparedChunk,
    deduplicate_chunks,
    split_extracted_document,
)
from app.utils.loaders import (
    ExtractedDocument,
    extract_document,
    extracted_document_from_text,
)


ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class PreparedIngestion:
    extracted: ExtractedDocument
    chunks: tuple[PreparedChunk, ...]
    quality: DocumentQualityResult
    strategy: str


class DocumentIngestionService:
    """Shared extraction, chunking, quality, embedding, and persistence pipeline."""

    def __init__(self, db: Session) -> None:
        self.db = db
        self.settings = SettingsService(db)
        self.quality = DocumentQualityService(db)

    def prepare_file(
        self,
        file_path: str,
        file_type: str,
        *,
        strategy: str = "auto",
        current_document_id: UUID | None = None,
        base_metadata: dict[str, Any] | None = None,
    ) -> PreparedIngestion:
        extracted = extract_document(file_path, file_type, strategy)
        return self.prepare_extracted(
            extracted,
            strategy=strategy,
            current_document_id=current_document_id,
            base_metadata=base_metadata,
        )

    def prepare_text(
        self,
        content: str,
        *,
        file_type: str = "txt",
        strategy: str = "auto",
        current_document_id: UUID | None = None,
        base_metadata: dict[str, Any] | None = None,
        extraction_method: str = "text",
    ) -> PreparedIngestion:
        extracted = extracted_document_from_text(
            content,
            file_type=file_type,
            extraction_method=extraction_method,
        )
        return self.prepare_extracted(
            extracted,
            strategy=strategy,
            current_document_id=current_document_id,
            base_metadata=base_metadata,
        )

    def prepare_extracted(
        self,
        extracted: ExtractedDocument,
        *,
        strategy: str = "auto",
        current_document_id: UUID | None = None,
        base_metadata: dict[str, Any] | None = None,
    ) -> PreparedIngestion:
        normalized_strategy = strategy if strategy in {"almanac", "curriculum"} else "auto"
        if normalized_strategy == "almanac":
            rows = build_almanac_chunks(extracted.content)
            prepared = self._structured_chunks(rows, extracted, base_metadata)
        elif normalized_strategy == "curriculum":
            rows = build_curriculum_chunks(extracted.content)
            prepared = self._structured_chunks(rows, extracted, base_metadata)
        else:
            prepared = split_extracted_document(
                extracted,
                chunk_size=self.settings.chunk_size,
                overlap=self.settings.chunk_overlap,
            )
            if base_metadata:
                prepared = [
                    PreparedChunk(
                        text=chunk.text,
                        page_number=chunk.page_number,
                        metadata={**base_metadata, **chunk.metadata},
                    )
                    for chunk in prepared
                ]

        deduplicated = deduplicate_chunks(prepared)
        final_chunks = list(deduplicated.chunks)
        quality = self.quality.evaluate(
            extracted,
            final_chunks,
            strategy=normalized_strategy,
            removed_duplicate_chunks=deduplicated.removed_count,
            current_document_id=current_document_id,
        )
        return PreparedIngestion(
            extracted=extracted,
            chunks=tuple(final_chunks),
            quality=quality,
            strategy=normalized_strategy,
        )

    def prepare_existing_chunks(
        self,
        rows: list[DocumentChunk],
        *,
        file_type: str,
        strategy: str,
        current_document_id: UUID,
    ) -> PreparedIngestion:
        from app.utils.chunking import reconstruct_text_from_chunks

        content = reconstruct_text_from_chunks([row.chunk_text for row in rows])
        extracted = extracted_document_from_text(
            content,
            file_type=file_type,
            extraction_method="legacy_chunks",
        )
        prepared = [
            PreparedChunk(
                text=row.chunk_text,
                page_number=row.page_number,
                metadata=dict(row.metadata_ or {}),
            )
            for row in rows
        ]
        deduplicated = deduplicate_chunks(prepared)
        final_chunks = list(deduplicated.chunks)
        quality = self.quality.evaluate(
            extracted,
            final_chunks,
            strategy=strategy,
            removed_duplicate_chunks=deduplicated.removed_count,
            current_document_id=current_document_id,
        )
        return PreparedIngestion(
            extracted=extracted,
            chunks=tuple(final_chunks),
            quality=quality,
            strategy=strategy,
        )

    def embed(
        self,
        prepared: PreparedIngestion,
        progress: ProgressCallback | None = None,
    ) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for batch_index, total_batches, batch_embeddings in self.embedding_batches(prepared):
            embeddings.extend(batch_embeddings)
            if progress:
                progress(batch_index, total_batches)
        if len(embeddings) != len(prepared.chunks):
            raise RuntimeError("Embedding service returned an unexpected number of vectors")
        return embeddings

    def embedding_batches(
        self,
        prepared: PreparedIngestion,
    ):
        from app.services.embedding_service import EmbeddingService

        texts = [chunk.text for chunk in prepared.chunks]
        if not texts:
            return
        embedding_service = EmbeddingService(self.db)
        batch_size = 15 if embedding_service.provider == "local" else embedding_service.batch_size
        batch_size = max(1, int(batch_size))
        total_batches = (len(texts) + batch_size - 1) // batch_size
        for batch_index, start in enumerate(range(0, len(texts), batch_size), start=1):
            yield (
                batch_index,
                total_batches,
                embedding_service.embed_batch(texts[start:start + batch_size]),
            )

    def record_quality(
        self,
        document: DocumentModel,
        prepared: PreparedIngestion,
    ) -> None:
        result = self._preserved_override(document, prepared.quality)
        document.quality_status = result.status
        document.quality_report = result.report
        document.quality_checked_at = application_now()
        document.ingestion_version = QUALITY_VERSION
        document.content_hash = result.content_hash or document.content_hash
        document.duplicate_of_document_id = result.duplicate_of_document_id

    def preserve_override(
        self,
        document: DocumentModel,
        prepared: PreparedIngestion,
    ) -> PreparedIngestion:
        result = self._preserved_override(document, prepared.quality)
        if result is prepared.quality:
            return prepared
        return PreparedIngestion(
            extracted=prepared.extracted,
            chunks=prepared.chunks,
            quality=result,
            strategy=prepared.strategy,
        )

    def apply(
        self,
        document: DocumentModel,
        prepared: PreparedIngestion,
        embeddings: list[list[float]],
        *,
        persist_review_chunks: bool = False,
    ) -> bool:
        prepared = self.preserve_override(document, prepared)
        self.record_quality(document, prepared)
        document.indexing_status = "idle"
        if not prepared.chunks:
            document.status = "quality_review"
            return False
        if prepared.quality.blocks_activation and not persist_review_chunks:
            document.status = "quality_review"
            return False
        if len(embeddings) != len(prepared.chunks):
            raise RuntimeError("Embedding count does not match prepared chunks")

        self.db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document.id
        ).delete(synchronize_session=False)
        self.db.bulk_insert_mappings(
            DocumentChunk,
            [
                {
                    "document_id": document.id,
                    "chunk_index": index,
                    "chunk_text": chunk.text,
                    "embedding": embedding,
                    "page_number": chunk.page_number,
                    "metadata_": dict(chunk.metadata),
                }
                for index, (chunk, embedding) in enumerate(zip(prepared.chunks, embeddings))
            ],
        )
        if prepared.quality.blocks_activation:
            document.status = "quality_review"
            return False
        document.status = "active"
        return True

    def audit_document(self, document: DocumentModel) -> DocumentQualityResult:
        result = self._preserved_override(
            document,
            self.quality.audit_existing_chunks(document),
        )
        document.quality_status = result.status
        document.quality_report = result.report
        document.quality_checked_at = application_now()
        document.ingestion_version = QUALITY_VERSION
        document.content_hash = result.content_hash or document.content_hash
        document.duplicate_of_document_id = result.duplicate_of_document_id
        return result

    @staticmethod
    def _preserved_override(
        document: DocumentModel,
        result: DocumentQualityResult,
    ) -> DocumentQualityResult:
        if result.status == "overridden":
            return result
        previous_report = dict(document.quality_report or {})
        override = previous_report.get("override")
        if not (
            document.quality_status == "overridden"
            and isinstance(override, dict)
            and document.content_hash
            and document.content_hash == result.content_hash
        ):
            return result
        report = {
            **result.report,
            "status": "overridden",
            "calculated_status": result.status,
            "override": override,
        }
        return DocumentQualityResult(
            status="overridden",
            content_hash=result.content_hash,
            duplicate_of_document_id=result.duplicate_of_document_id,
            report=report,
        )

    def _structured_chunks(
        self,
        rows: list[tuple[str, dict[str, Any]]],
        extracted: ExtractedDocument,
        base_metadata: dict[str, Any] | None,
    ) -> list[PreparedChunk]:
        prepared: list[PreparedChunk] = []
        for text, metadata in rows:
            page_number = self._matching_page(text, metadata, extracted)
            prepared.append(
                PreparedChunk(
                    text=text,
                    page_number=page_number,
                    metadata={
                        **(base_metadata or {}),
                        **metadata,
                        "extraction_method": extracted.extraction_method,
                        "content_kind": "structured_record",
                    },
                )
            )
        return prepared

    @staticmethod
    def _matching_page(
        chunk_text: str,
        metadata: dict[str, Any],
        extracted: ExtractedDocument,
    ) -> int | None:
        def searchable(value: str) -> str:
            return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

        candidates = [
            str(metadata.get(key) or "").strip()
            for key in ("course_code", "course_title", "date")
        ]
        activity = re.search(r"(?im)^Activity:\s*(.+)$", chunk_text)
        if activity:
            candidates.insert(0, activity.group(1).strip()[:80])
        normalized_candidates = [
            searchable(candidate)
            for candidate in candidates
            if len(candidate) >= 4
        ]
        for page in extracted.pages:
            haystack = searchable(page.content)
            if any(candidate in haystack for candidate in normalized_candidates):
                return page.page_number

        # Extraction may merge words or alter punctuation. Fall back to a
        # conservative token-overlap match for structured records.
        activity_text = activity.group(1) if activity else chunk_text
        candidate_tokens = {
            token
            for token in searchable(activity_text).split()
            if len(token) >= 4
        }
        if not candidate_tokens:
            return None
        best_page: int | None = None
        best_ratio = 0.0
        for page in extracted.pages:
            page_tokens = set(searchable(page.content).split())
            ratio = len(candidate_tokens & page_tokens) / len(candidate_tokens)
            if ratio > best_ratio:
                best_page = page.page_number
                best_ratio = ratio
        if best_ratio >= 0.65 and len(candidate_tokens) >= 3:
            return best_page
        return None


def quality_summary(document: DocumentModel) -> dict[str, Any]:
    report = dict(document.quality_report or {})
    metrics = dict(report.get("metrics") or {})
    return {
        "status": document.quality_status or "unchecked",
        "warning_count": int(report.get("warning_count") or 0),
        "blocker_count": int(report.get("blocker_count") or 0),
        "metrics": {
            key: metrics.get(key)
            for key in (
                "page_count",
                "ocr_required_pages",
                "chunk_count",
                "removed_duplicate_chunks",
                "page_provenance_ratio",
            )
            if key in metrics
        },
    }
