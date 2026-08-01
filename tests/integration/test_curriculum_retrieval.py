from langchain_core.documents import Document
from uuid import uuid4

from app.core.config import settings
from app.db.models import DocumentChunk, DocumentModel
from app.services.grounding_service import GroundingService
from app.services.rag_pipeline import RAGPipeline


class _RetrievalService:
    def __init__(self, db):
        self.db = db


def _course_metadata(programme, year, semester, code, title, acronym="BSc SE"):
    return {
        "category": "curriculum",
        "record_type": "course",
        "academic_year": "2025/2026",
        "college": "COLLEGE OF INFORMATICS AND VIRTUAL EDUCATION (CIVE)",
        "programme": programme,
        "programme_acronym": acronym,
        "year_of_study": year,
        "semester": semester,
        "course_code": code,
        "course_title": title,
        "status": "Core",
        "credits": "7.5",
    }


def _course_text(metadata):
    return "\n".join(
        [
            "UDOM Undergraduate Curriculum Course",
            f"Programme: {metadata['programme']}",
            f"Programme Acronym: {metadata['programme_acronym']}",
            f"Year of Study: Year {metadata['year_of_study']}",
            f"Semester: Semester {metadata['semester']}",
            f"Course Code: {metadata['course_code']}",
            f"Course Title: {metadata['course_title']}",
            f"Status: {metadata['status']}",
            f"Credits: {metadata['credits']}",
        ]
    )


def test_curriculum_metadata_supplement_promotes_exact_course_rows(db_session):
    marker = uuid4().hex[:8]
    programme = f"Bachelor of Science in Software Engineering Retrieval Test {marker} (BSc SE)"
    document = DocumentModel(
        title="curriculum-test",
        filename="curriculum-test.pdf",
        category="pdf",
        status="active",
    )
    db_session.add(document)
    db_session.flush()

    rows = []
    for index, metadata in enumerate(
        [
            _course_metadata(programme, "1", "1", "LG 102", "Communication Skills"),
            _course_metadata(programme, "1", "1", "CP 111", "Principles of Programming Languages"),
            _course_metadata(programme, "1", "2", "CS 123", "Introduction to Software Engineering"),
        ],
        start=1,
    ):
        row = DocumentChunk(
            document_id=document.id,
            chunk_index=index,
            chunk_text=_course_text(metadata),
            embedding=[0.1] * settings.EMBEDDING_DIMENSION,
            metadata_=metadata,
        )
        db_session.add(row)
        rows.append(row)
    db_session.commit()

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.retrieval_service = _RetrievalService(db_session)
    wrong_semester_doc = Document(
        page_content=rows[2].chunk_text,
        metadata={**rows[2].metadata_, "chunk_id": str(rows[2].id)},
    )

    supplemented = pipeline._supplement_curriculum_documents(
        f"what are the semester one course for software engineering retrieval test {marker} first year?",
        [(wrong_semester_doc, 0.9)],
    )

    assert [doc.metadata["course_code"] for doc, _score in supplemented[:2]] == ["LG 102", "CP 111"]
    assert all(doc.metadata["semester"] == "1" for doc, _score in supplemented[:2])
    assert supplemented[2][0].metadata["course_code"] == "CS 123"


def test_curriculum_programme_match_prefers_exact_acronym_over_fuzzy_tokens(db_session):
    document = DocumentModel(
        title="curriculum-acronym-test",
        filename="curriculum-acronym-test.pdf",
        category="pdf",
        status="active",
    )
    db_session.add(document)
    db_session.flush()

    programmes = [
        _course_metadata(
            "Bachelor of Science in Computer Networks and Information Security Engineering (BSc CNISE)",
            "1",
            "1",
            "IS 111",
            "Information Security Foundations",
            acronym="BSc CNISE",
        ),
        _course_metadata(
            "Bachelor of Science in Instructional Design and Information Technology (BSc IDIT)",
            "1",
            "1",
            "ID 111",
            "Instructional Design Foundations",
            acronym="BSc IDIT",
        ),
    ]
    for index, metadata in enumerate(programmes, start=1):
        db_session.add(
            DocumentChunk(
                document_id=document.id,
                chunk_index=index,
                chunk_text=_course_text(metadata),
                embedding=[0.1] * settings.EMBEDDING_DIMENSION,
                metadata_=metadata,
            )
        )
    db_session.commit()

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.retrieval_service = _RetrievalService(db_session)

    assert pipeline._programme_from_query(
        "what are the courses for a IDIT student in semester one, first year?"
    ) == "Bachelor of Science in Instructional Design and Information Technology (BSc IDIT)"


def test_curriculum_programme_match_resolves_known_idt_typo(db_session):
    document = DocumentModel(
        title="curriculum-idt-alias-test",
        filename="curriculum-idt-alias-test.pdf",
        category="pdf",
        status="active",
    )
    db_session.add(document)
    db_session.flush()

    metadata = _course_metadata(
        "Bachelor of Science in Instructional Design and Information Technology (BSc IDIT)",
        "2",
        "1",
        "ID 211",
        "Instructional Systems Design",
        acronym="BSc IDIT",
    )
    db_session.add(
        DocumentChunk(
            document_id=document.id,
            chunk_index=1,
            chunk_text=_course_text(metadata),
            embedding=[0.1] * settings.EMBEDDING_DIMENSION,
            metadata_=metadata,
        )
    )
    db_session.commit()

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.retrieval_service = _RetrievalService(db_session)

    assert pipeline._programme_from_query(
        "what are the courses for a IDT student for semester one, second year"
    ) == "Bachelor of Science in Instructional Design and Information Technology (BSc IDIT)"


def test_direct_curriculum_course_answer_uses_metadata_rows():
    pipeline = RAGPipeline.__new__(RAGPipeline)
    programme = "Bachelor of Science in Software Engineering (BSc SE)"
    documents = [
        Document(
            page_content=_course_text(metadata),
            metadata={
                **metadata,
                "chunk_id": f"chunk-{index}",
                "chunk_index": index,
                "document_id": "curriculum-doc",
                "source": "undergraduate curriculum book 2025-2026.pdf",
                "curriculum_metadata_match": True,
            },
        )
        for index, metadata in enumerate(
            [
                _course_metadata(programme, "1", "1", "LG 102", "Communication Skills"),
                _course_metadata(programme, "1", "1", "CP 111", "Principles of Programming Languages"),
            ],
            start=1,
        )
    ]

    result = pipeline._direct_curriculum_course_answer(
        "what are the courses for a software engineering student in semester one, first year?",
        documents,
    )

    assert result is not None
    assert result["grounding"]["status"] == "grounded"
    assert result["grounding"]["evidence_ids"] == [
        GroundingService().evidence_id(document)
        for document in documents
    ]
    assert "| LG 102 | Communication Skills | Core | 7.5 |" in result["answer"]
    assert "| CP 111 | Principles of Programming Languages | Core | 7.5 |" in result["answer"]
    assert result["documents"] == documents


def test_direct_curriculum_course_answer_keeps_more_than_24_rows():
    pipeline = RAGPipeline.__new__(RAGPipeline)
    programme = "Bachelor of Science in Software Engineering (BSc SE)"
    documents = []
    for index in range(1, 31):
        metadata = _course_metadata(
            programme,
            "1",
            "1",
            f"SE {index:03d}",
            f"Software Engineering Course {index}",
        )
        documents.append(
            Document(
                page_content=_course_text(metadata),
                metadata={
                    **metadata,
                    "chunk_id": f"chunk-{index}",
                    "chunk_index": index,
                    "document_id": "curriculum-doc",
                    "source": "undergraduate curriculum book 2025-2026.pdf",
                    "curriculum_metadata_match": True,
                },
            )
        )

    result = pipeline._direct_curriculum_course_answer(
        "list all software engineering courses",
        documents,
    )

    assert result is not None
    assert len(result["documents"]) == 30
    assert len(result["grounding"]["evidence_ids"]) == 30
    assert "| SE 030 | Software Engineering Course 30 |" in result["answer"]


def test_debug_direct_curriculum_answer_reports_grounded_metadata():
    pipeline = RAGPipeline.__new__(RAGPipeline)
    programme = "Bachelor of Science in Software Engineering (BSc SE)"
    metadata = _course_metadata(
        programme,
        "1",
        "1",
        "SE 101",
        "Introduction to Software Engineering",
    )
    document = Document(
        page_content=_course_text(metadata),
        metadata={
            **metadata,
            "chunk_id": "chunk-debug",
            "chunk_index": 1,
            "document_id": "curriculum-doc",
            "source": "undergraduate curriculum book 2025-2026.pdf",
            "curriculum_metadata_match": True,
        },
    )

    class _Decision:
        intent = "UDOM_DOCUMENT_SEARCH"
        filters = None
        standalone_query = "software engineering courses"
        source = "test"
        reason = "test"

        @staticmethod
        def to_dict():
            return {"intent": "UDOM_DOCUMENT_SEARCH"}

    class _IntentRouter:
        @staticmethod
        def classify(*_args, **_kwargs):
            return _Decision()

    class _Generator:
        grounding = GroundingService()

        @staticmethod
        def set_model_preference(_preference):
            return None

        @staticmethod
        def grounding_metadata():
            return {
                "coverage": "none",
                "evidence_ids": [],
                "claim_count": 0,
                "supported_claim_count": 0,
                "repaired": False,
                "status": "refused",
            }

        @staticmethod
        def model_metadata():
            return {
                "requested_model": "auto",
                "selected_model": None,
                "fallback_used": False,
            }

    pipeline.generator = _Generator()
    pipeline.intent_router = _IntentRouter()
    pipeline._retrieve_document_context = lambda *_args, **_kwargs: {
        "documents": [document],
        "context": document.page_content,
        "confidence": {"sufficient": True},
        "debug_documents": [
            {
                "content": document.page_content[:200],
                "metadata": document.metadata,
                "score": 1.0,
            }
        ],
    }

    result = pipeline.run_debug(
        "what are the software engineering courses for first year semester one?"
    )

    assert result["debug"]["grounding"]["status"] == "grounded"
    assert result["debug"]["grounding"]["evidence_ids"] == [
        GroundingService().evidence_id(document)
    ]
