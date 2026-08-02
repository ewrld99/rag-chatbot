from types import SimpleNamespace

from app.core.config import settings
from app.db.models import DocumentChunk, DocumentModel
from app.services.document_ingestion_service import (
    DocumentIngestionService,
    PreparedIngestion,
)
from app.services.document_quality_service import (
    DocumentQualityService,
    normalized_document_hash,
)
from app.utils.chunking import PreparedChunk, split_extracted_document
from app.utils.loaders import (
    ExtractedDocument,
    ExtractedPage,
    extract_pdf,
    extracted_document_from_text,
)


class _FakeTable:
    bbox = (100.0, 100.0, 300.0, 200.0)

    def extract(self):
        return [["Course", "Credits"], ["CS 101", "3"]]


class _FakeFilteredPage:
    def extract_text(self):
        return "Outside narrative text."


class _FakePage:
    images = []

    def find_tables(self):
        return SimpleNamespace(tables=[_FakeTable()])

    def filter(self, predicate):
        assert not predicate({"x0": 120, "x1": 140, "top": 120, "bottom": 140})
        assert predicate({"x0": 10, "x1": 20, "top": 10, "bottom": 20})
        return _FakeFilteredPage()

    def extract_text(self):
        return "Outside narrative text. Course Credits CS 101 3"


class _FakeScannedPage:
    images = [{"name": "scan"}]

    def find_tables(self):
        return SimpleNamespace(tables=[])

    def extract_text(self):
        return ""


class _FakePdf:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


def test_pdf_tables_are_removed_from_plain_text_before_markdown(monkeypatch):
    monkeypatch.setattr("app.utils.loaders.pdfplumber.open", lambda _path: _FakePdf([_FakePage()]))

    extracted = extract_pdf("sample.pdf")

    assert extracted.pages[0].page_number == 1
    assert extracted.content.count("CS 101") == 1
    assert "Outside narrative text." in extracted.content
    assert "| Course | Credits |" in extracted.content


def test_pdf_image_page_with_little_text_is_marked_for_ocr(monkeypatch):
    monkeypatch.setattr(
        "app.utils.loaders.pdfplumber.open",
        lambda _path: _FakePdf([_FakeScannedPage()]),
    )

    extracted = extract_pdf("scan.pdf")

    assert extracted.pages[0].ocr_required


def test_section_aware_chunks_retain_page_and_heading():
    extracted = ExtractedDocument(
        pages=(
            ExtractedPage(
                page_number=7,
                text=(
                    "3. DRESS CODE\n\n"
                    "- Display a student identity card.\n"
                    "- Wear neat formal or semi-formal clothing."
                ),
                headings=("3. DRESS CODE",),
                extraction_method="pdfplumber",
            ),
        ),
        file_type="pdf",
        extraction_method="pdfplumber",
    )

    chunks = split_extracted_document(extracted, chunk_size=500, overlap=50)

    assert len(chunks) == 1
    assert chunks[0].page_number == 7
    assert chunks[0].metadata["section_title"] == "3. DRESS CODE"
    assert chunks[0].metadata["content_kind"] == "list"
    assert "Display a student identity card" in chunks[0].text


def test_consecutive_list_sections_switch_heading_without_blank_line():
    extracted = ExtractedDocument(
        pages=(
            ExtractedPage(
                page_number=41,
                text=(
                    "(1) APPROPRIATE DRESSING\n"
                    "i. Display a student identity card.\n"
                    "ii. Wear neat formal or semi-formal clothing.\n"
                    "(2) INAPPROPRIATE DRESSING\n"
                    "i. Shorts must not be worn.\n"
                    "ii. Clothing with obscene graphics is prohibited."
                ),
                headings=(
                    "(1) APPROPRIATE DRESSING",
                    "(2) INAPPROPRIATE DRESSING",
                ),
                extraction_method="pdfplumber",
            ),
        ),
        file_type="pdf",
        extraction_method="pdfplumber",
    )

    chunks = split_extracted_document(extracted, chunk_size=500, overlap=50)

    assert len(chunks) == 2
    assert chunks[0].metadata["section_title"] == "(1) APPROPRIATE DRESSING"
    assert "Shorts must not be worn" not in chunks[0].text
    assert chunks[1].metadata["section_title"] == "(2) INAPPROPRIATE DRESSING"
    assert "Shorts must not be worn" in chunks[1].text


def test_numbered_policy_clauses_are_grouped_without_becoming_headings():
    extracted = ExtractedDocument(
        pages=(
            ExtractedPage(
                page_number=12,
                text=(
                    "12. POSTPONEMENT OF STUDIES\n\n"
                    "12.5 No applicant has permission until Senate communicates its decision.\n"
                    "12.6 A student must apply before the stated semester deadline.\n"
                    "12.7 Postponement is allowed within the maximum studentship duration."
                ),
                headings=("12. POSTPONEMENT OF STUDIES",),
                extraction_method="pdfplumber",
            ),
        ),
        file_type="pdf",
        extraction_method="pdfplumber",
    )

    chunks = split_extracted_document(extracted, chunk_size=500, overlap=50)

    combined = "\n".join(chunk.text for chunk in chunks)
    assert len(chunks) == 1
    assert "12.5 No applicant" in combined
    assert "12.6 A student" in combined
    assert "12.7 Postponement" in combined
    assert chunks[0].metadata["section_title"] == "12. POSTPONEMENT OF STUDIES"


def test_text_heading_detection_does_not_promote_numbered_policy_clauses():
    extracted = extracted_document_from_text(
        "12. POSTPONEMENT OF STUDIES\n"
        "12.5 No applicant has permission until Senate communicates its decision\n"
        "12.6 A student must apply before the stated semester deadline"
    )

    assert "12. POSTPONEMENT OF STUDIES" in extracted.pages[0].headings
    assert not any(heading.startswith("12.5 ") for heading in extracted.pages[0].headings)
    assert not any(heading.startswith("12.6 ") for heading in extracted.pages[0].headings)


def test_quality_gate_blocks_pdf_with_many_ocr_pages(db_session):
    extracted = ExtractedDocument(
        pages=tuple(
            ExtractedPage(
                page_number=index,
                text="Readable page text with enough characters for extraction." if index > 4 else "",
                extraction_method="pdfplumber",
                has_images=index <= 4,
                ocr_required=index <= 4,
            )
            for index in range(1, 11)
        ),
        file_type="pdf",
        extraction_method="pdfplumber",
    )
    chunks = [PreparedChunk(text="Readable extracted policy text.", page_number=5)]

    result = DocumentQualityService(db_session).evaluate(extracted, chunks)

    assert result.status == "review"
    assert any(issue["code"] == "OCR_REQUIRED" for issue in result.report["issues"])


def test_quality_gate_marks_exact_active_document_duplicate(db_session):
    content = "A sufficiently detailed official document about student registration rules."
    existing = DocumentModel(
        title="Existing",
        filename="existing.txt",
        status="active",
        content_hash=normalized_document_hash(content),
    )
    db_session.add(existing)
    db_session.flush()

    extracted = extracted_document_from_text(content)
    result = DocumentQualityService(db_session).evaluate(
        extracted,
        [PreparedChunk(text=content)],
    )

    assert result.status == "review"
    assert result.duplicate_of_document_id == existing.id
    assert any(issue["code"] == "DUPLICATE_DOCUMENT" for issue in result.report["issues"])


def test_reviewed_duplicate_cannot_replace_canonical_document(db_session):
    content = "Canonical official calendar content for the current academic year."
    canonical = DocumentModel(
        title="Canonical",
        filename="canonical.txt",
        status="active",
        quality_status="passed",
        content_hash=normalized_document_hash(content),
    )
    reviewed = DocumentModel(
        title="Reviewed duplicate",
        filename="duplicate.txt",
        status="active",
        quality_status="review",
        content_hash=normalized_document_hash(content),
    )
    db_session.add_all([canonical, reviewed])
    db_session.flush()

    result = DocumentQualityService(db_session).evaluate(
        extracted_document_from_text(content),
        [PreparedChunk(text=content)],
        current_document_id=canonical.id,
    )

    assert result.status != "review"
    assert result.duplicate_of_document_id is None


def test_blocked_reindex_retains_existing_chunks(db_session):
    document = DocumentModel(
        title="Rules",
        filename="rules.txt",
        status="active",
    )
    db_session.add(document)
    db_session.flush()
    existing = DocumentChunk(
        document_id=document.id,
        chunk_index=0,
        chunk_text="Existing searchable rules remain available in storage.",
        embedding=[0.1] * settings.EMBEDDING_DIMENSION,
        metadata_={},
    )
    db_session.add(existing)
    db_session.flush()

    ingestion = DocumentIngestionService(db_session)
    extracted = ExtractedDocument(
        pages=(
            ExtractedPage(
                page_number=1,
                text="",
                has_images=True,
                ocr_required=True,
                extraction_method="pdfplumber",
            ),
        ),
        file_type="pdf",
        extraction_method="pdfplumber",
    )
    new_chunk = PreparedChunk(text="Unreliable replacement text.", page_number=1)
    quality = DocumentQualityService(db_session).evaluate(extracted, [new_chunk])
    prepared = PreparedIngestion(
        extracted=extracted,
        chunks=(new_chunk,),
        quality=quality,
        strategy="auto",
    )

    activated = ingestion.apply(
        document,
        prepared,
        [[0.2] * settings.EMBEDDING_DIMENSION],
    )
    db_session.flush()

    rows = db_session.query(DocumentChunk).filter_by(document_id=document.id).all()
    assert not activated
    assert document.status == "quality_review"
    assert [row.chunk_text for row in rows] == [existing.chunk_text]
