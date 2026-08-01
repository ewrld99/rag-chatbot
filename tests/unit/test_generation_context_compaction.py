from types import SimpleNamespace

from langchain_core.documents import Document

from app.core.config import settings
from app.db.models import DocumentChunk, DocumentModel
from app.services.grounding_service import GroundingService
from app.services.grounding_service import GroundingOutcome
from app.services.generation_service import GenerationService
from app.services.rag_pipeline import RAGPipeline


class _FakeRetrievalService:
    def format_context(self, documents, max_chars=None):
        chunks = []
        grounding = GroundingService()
        for document in documents:
            chunks.append(
                f'<document evidence_id="{grounding.evidence_id(document)}">'
                f"{document.page_content}"
                "</document>"
            )
        context = "\n".join(chunks)
        return context[:max_chars] if max_chars else context


def test_generation_context_compaction_preserves_evidence_ids(monkeypatch):
    import app.core.config

    monkeypatch.setattr(app.core.config.settings, "GENERATION_CONTEXT_MAX_DOCS", 2)
    monkeypatch.setattr(app.core.config.settings, "GENERATION_CONTEXT_MAX_CHARS", 2000)
    monkeypatch.setattr(app.core.config.settings, "GENERATION_CONTEXT_MAX_CHARS_PER_DOC", 220)
    monkeypatch.setattr(app.core.config.settings, "GENERATION_SIMPLE_CONTEXT_MAX_DOCS", 2)
    monkeypatch.setattr(app.core.config.settings, "GENERATION_SIMPLE_CONTEXT_MAX_CHARS", 2000)
    monkeypatch.setattr(app.core.config.settings, "GENERATION_SIMPLE_CONTEXT_MAX_CHARS_PER_DOC", 220)
    monkeypatch.setattr(app.core.config.settings, "GENERATION_PROCEDURE_CONTEXT_MAX_DOCS", 2)
    monkeypatch.setattr(app.core.config.settings, "GENERATION_PROCEDURE_CONTEXT_MAX_CHARS", 2000)
    monkeypatch.setattr(app.core.config.settings, "GENERATION_PROCEDURE_CONTEXT_MAX_CHARS_PER_DOC", 220)

    first_160_chars = "Procedure to postpone a year of study requires approval. " * 3
    long_tail = "Additional procedural detail. " * 80
    documents = [
        Document(
            page_content=first_160_chars + long_tail,
            metadata={
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "chunk_index": 1,
            },
        ),
        Document(
            page_content="Second relevant rule. " * 80,
            metadata={
                "chunk_id": "chunk-2",
                "document_id": "doc-1",
                "chunk_index": 2,
            },
        ),
        Document(
            page_content="Third lower-ranked rule. " * 80,
            metadata={
                "chunk_id": "chunk-3",
                "document_id": "doc-1",
                "chunk_index": 3,
            },
        ),
    ]

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.retrieval_service = _FakeRetrievalService()

    compact_documents, context, budget = pipeline._compact_generation_context(
        documents,
        "what is the procedure to postpone a year of study",
    )
    grounding = GroundingService()

    assert len(compact_documents) == 2
    assert len(context) <= 2000
    assert budget == "procedure"
    assert len(compact_documents[0].page_content) < len(documents[0].page_content)
    assert grounding.evidence_id(compact_documents[0]) == grounding.evidence_id(documents[0])
    assert grounding.evidence_id(compact_documents[1]) == grounding.evidence_id(documents[1])

    rewritten = Document(
        page_content="A generated compact excerpt can differ from original text.",
        metadata=dict(documents[0].metadata),
    )
    assert grounding.evidence_id(rewritten) == grounding.evidence_id(documents[0])


def test_procedure_context_promotes_contiguous_chunks_in_document_order(db_session, monkeypatch):
    monkeypatch.setattr(settings, "GENERATION_PROCEDURE_SECTION_WINDOW", 2)
    document = DocumentModel(
        title="Undergraduate Regulations",
        filename="regulations.pdf",
        status="active",
    )
    db_session.add(document)
    db_session.flush()

    rows = []
    for index in range(7):
        row = DocumentChunk(
            document_id=document.id,
            chunk_index=index,
            chunk_text=f"Regulation chunk {index}",
            embedding=[0.0] * settings.EMBEDDING_DIMENSION,
            metadata_={},
        )
        db_session.add(row)
        rows.append(row)
    db_session.flush()

    anchor = Document(
        page_content=rows[3].chunk_text,
        metadata={
            "chunk_id": str(rows[3].id),
            "document_id": str(document.id),
            "chunk_index": 3,
            "source_type": "document",
            "source": "regulations.pdf",
        },
    )
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.generator = GenerationService.__new__(GenerationService)
    pipeline.retrieval_service = SimpleNamespace(db=db_session)

    expanded = pipeline._procedure_context_documents(
        [anchor],
        "What is the procedure to postpone a year of study?",
    )

    assert [item.metadata["chunk_index"] for item in expanded] == [1, 2, 3, 4, 5]
    assert all(item.metadata["section_expansion"] for item in expanded)


def test_enumeration_context_promotes_detailed_policy_section(db_session, monkeypatch):
    monkeypatch.setattr(settings, "GENERATION_PROCEDURE_SECTION_WINDOW", 2)
    document = DocumentModel(
        title="Student By-Laws",
        filename="student-bylaws.pdf",
        status="active",
    )
    db_session.add(document)
    db_session.flush()

    contents = [
        "Earlier disciplinary provision.",
        "Rule 52 Debts\n- Students must repay official debts.",
        "Rule 53 Dress Code\n- Students must dress decently under the prescribed code.",
        "Dress code preamble and applicability.",
        (
            "3. Dress Code\n(1) Appropriate Dressing\n"
            "- Display a student identity card.\n"
            "- Wear neat, decent and well-covered formal or semi-formal clothing.\n"
            "- Keep hair well groomed.\n"
            "- Wear jeans without holes."
        ),
        "(2) Inappropriate Dressing\n- Transparent clothing is prohibited.",
        "Male dressing\n- Sleeveless shirts are prohibited.",
    ]
    rows = []
    for index, content in enumerate(contents):
        row = DocumentChunk(
            document_id=document.id,
            chunk_index=index,
            chunk_text=content,
            embedding=[0.0] * settings.EMBEDDING_DIMENSION,
            metadata_={},
        )
        db_session.add(row)
        rows.append(row)
    db_session.flush()

    generic = Document(
        page_content=rows[2].chunk_text,
        metadata={
            "chunk_id": str(rows[2].id),
            "document_id": str(document.id),
            "chunk_index": 2,
            "source_type": "document",
            "source": "student-bylaws.pdf",
        },
    )
    detailed = Document(
        page_content=rows[4].chunk_text,
        metadata={
            "chunk_id": str(rows[4].id),
            "document_id": str(document.id),
            "chunk_index": 4,
            "source_type": "document",
            "source": "student-bylaws.pdf",
        },
    )
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.generator = GenerationService.__new__(GenerationService)
    pipeline.retrieval_service = SimpleNamespace(db=db_session)

    expanded = pipeline._generation_context_documents(
        [generic, detailed],
        "What are the proper dress code rules for students?",
    )

    assert [item.metadata["chunk_index"] for item in expanded[:5]] == [4, 5, 6, 3, 2]
    assert expanded[0].metadata["section_expansion_kind"] == "enumeration"


def test_enumeration_answer_shape_rejects_generic_parent_rule():
    service = GenerationService.__new__(GenerationService)

    instructions = service._answer_shape_instructions(
        "What are the proper dress code rules for students?"
    )

    assert "Enumerate every distinct item" in instructions
    assert "generic parent rule" in instructions


def test_enumeration_answer_with_one_claim_requires_completeness_repair():
    service = GenerationService.__new__(GenerationService)
    evidence = GroundingService().build_evidence(
        [
            Document(
                page_content=(
                    "Appropriate Dressing\n"
                    "- Display a student identity card.\n"
                    "- Wear neat formal or semi-formal clothing.\n"
                    "- Wear jeans without holes."
                ),
                metadata={"chunk_id": "dress-code"},
            )
        ]
    )
    generic_outcome = GroundingOutcome(
        answer="Students must dress decently.",
        coverage="full",
        evidence_ids=("ev-generic",),
        claim_count=1,
        supported_claim_count=1,
        repaired=False,
        status="grounded",
    )

    assert service._should_repair_initial_outcome(
        "What are the proper dress code rules for students?",
        generic_outcome,
        evidence,
    )


def test_generation_context_compaction_keeps_larger_budget_for_complex_queries(monkeypatch):
    import app.core.config

    monkeypatch.setattr(app.core.config.settings, "GENERATION_CONTEXT_MAX_DOCS", 6)
    monkeypatch.setattr(app.core.config.settings, "GENERATION_CONTEXT_MAX_CHARS", 6500)
    monkeypatch.setattr(app.core.config.settings, "GENERATION_CONTEXT_MAX_CHARS_PER_DOC", 1100)
    monkeypatch.setattr(app.core.config.settings, "GENERATION_SIMPLE_CONTEXT_MAX_DOCS", 3)
    monkeypatch.setattr(app.core.config.settings, "GENERATION_SIMPLE_CONTEXT_MAX_CHARS", 3000)
    monkeypatch.setattr(app.core.config.settings, "GENERATION_SIMPLE_CONTEXT_MAX_CHARS_PER_DOC", 700)

    documents = [
        Document(
            page_content=f"Rule {index}. " * 160,
            metadata={
                "chunk_id": f"chunk-{index}",
                "document_id": "doc-1",
                "chunk_index": index,
            },
        )
        for index in range(1, 8)
    ]

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.retrieval_service = _FakeRetrievalService()

    compact_documents, context, budget = pipeline._compact_generation_context(
        documents,
        "how is gpa calculated from grade points and course weights",
    )

    assert budget == "complex"
    assert len(compact_documents) == 6
    assert len(context) <= 6500


def test_generation_max_tokens_are_adaptive(monkeypatch):
    import app.core.config

    monkeypatch.setattr(app.core.config.settings, "GENERATION_SIMPLE_MAX_TOKENS", 500)
    monkeypatch.setattr(app.core.config.settings, "GENERATION_MEDIUM_MAX_TOKENS", 750)
    monkeypatch.setattr(app.core.config.settings, "GENERATION_COMPLEX_MAX_TOKENS", 1000)
    monkeypatch.setattr(app.core.config.settings, "GENERATION_CONTEXT_MAX_CHARS", 6500)

    service = GenerationService.__new__(GenerationService)

    assert service._get_max_tokens("short context", "procedure to postpone study") == 750
    assert service._get_max_tokens("short context", "how is gpa calculated") == 1000
    assert service._get_max_tokens("short context", "list semester one courses") == 750
    assert service._get_max_tokens("x" * 6200, "tell me about examination rules") == 750


def test_curriculum_answer_shape_requests_course_table():
    service = GenerationService.__new__(GenerationService)

    instructions = service._answer_shape_instructions(
        "what are the semester one courses for software engineering first year?"
    )

    assert "Course Code, Course Title, Status, and Credits" in instructions
    assert "separate document element" in instructions


def test_semester_exam_date_question_does_not_get_curriculum_answer_shape():
    service = GenerationService.__new__(GenerationService)

    instructions = service._answer_shape_instructions(
        "the day the semester two examinations start"
    )

    assert "course-list" not in instructions
    assert "Course Code, Course Title" not in instructions


def test_procedure_questions_use_medium_context_budget(monkeypatch):
    import app.core.config

    monkeypatch.setattr(app.core.config.settings, "GENERATION_PROCEDURE_CONTEXT_MAX_DOCS", 5)
    monkeypatch.setattr(app.core.config.settings, "GENERATION_PROCEDURE_CONTEXT_MAX_CHARS", 5800)
    monkeypatch.setattr(app.core.config.settings, "GENERATION_PROCEDURE_CONTEXT_MAX_CHARS_PER_DOC", 1050)

    pipeline = RAGPipeline.__new__(RAGPipeline)
    budget = pipeline._generation_context_budget("what is the procedure to postpone a year of study")

    assert budget == {
        "tier": "procedure",
        "max_docs": 5,
        "max_chars": 5800,
        "max_chars_per_doc": 1050,
    }


def test_procedure_questions_do_not_use_direct_clause_shortcut():
    pipeline = RAGPipeline.__new__(RAGPipeline)

    assert not hasattr(pipeline, "_direct_procedure_step_answer")


def test_curriculum_course_list_questions_use_wider_context_budget():
    pipeline = RAGPipeline.__new__(RAGPipeline)

    budget = pipeline._generation_context_budget(
        "what are the semester one course for software engineering first year?"
    )

    assert budget["tier"] == "curriculum"
    assert budget["max_docs"] >= RAGPipeline._CURRICULUM_METADATA_MAX_ROWS
    assert budget["max_chars"] >= 12000


def test_semester_exam_date_question_uses_standard_context_budget():
    pipeline = RAGPipeline.__new__(RAGPipeline)

    budget = pipeline._generation_context_budget(
        "the day the semester two examinations start"
    )

    assert budget["tier"] == "standard"


def test_semester_exam_date_question_has_no_curriculum_constraints():
    pipeline = RAGPipeline.__new__(RAGPipeline)

    assert (
        pipeline._curriculum_constraints_from_query(
            "the day the semester two examinations start"
        )
        == {}
    )
