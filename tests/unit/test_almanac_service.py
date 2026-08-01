from app.services.almanac_service import build_almanac_chunks, parse_almanac_events
from app.services.rag_pipeline import RAGPipeline
from langchain_core.documents import Document


def test_parse_almanac_events_extracts_dated_rows_with_metadata():
    text = """
    DATE ACTIVITY
    ADMISSION/REGISTRATION FOR ACADEMIC YEAR 2026/2027
    Monday, November 16, 2026 Start of teaching for Semester I Academic Year 2026/27
    Monday, November 30, 2026 End of Semester I Registration of Continuing Undergraduate Students - Academic Year 2026/27
    Thursday, December 3, 2026 16th Graduation Ceremony Cluster 1
    """

    events = parse_almanac_events(text)

    assert len(events) == 3
    assert events[0].event_date.isoformat() == "2026-11-16"
    assert events[0].event_type == "teaching"
    assert events[0].academic_year == "2026/2027"
    assert "ADMISSION/REGISTRATION" not in events[0].activity
    assert events[1].event_type == "registration"
    assert events[2].event_type == "graduation"


def test_build_almanac_chunks_creates_searchable_event_chunks():
    chunks = build_almanac_chunks(
        "Tuesday, January 12, 2027 109th Senate Undergraduate Studies Committee M eeting"
    )

    assert len(chunks) == 1
    chunk_text, metadata = chunks[0]
    assert "Date: 2027-01-12" in chunk_text
    assert "109th Senate Undergraduate Studies Committee Meeting" in chunk_text
    assert metadata == {
        "category": "academic_calendar",
        "record_type": "almanac_event",
        "date": "2027-01-12",
        "academic_year": "2026/2027",
        "event_type": "meeting",
    }


def test_direct_almanac_answer_uses_retrieved_event_without_llm():
    pipeline = object.__new__(RAGPipeline)
    document = Document(
        page_content=(
            "UDOM Academic Almanac Event\n"
            "Date: 2026-10-12\n"
            "Activity: Start of Supplementary/Special University Examinations "
            "for all Undergraduate and Postgraduate Programmes.\n"
            "Event Type: examination\n"
            "Academic Year: 2025/2026"
        ),
        metadata={
            "category": "academic_calendar",
            "record_type": "almanac_event",
            "date": "2026-10-12",
            "event_type": "examination",
        },
    )

    result = pipeline._direct_almanac_event_answer(
        "when are the supplementary examinations start?",
        [document],
    )

    assert result is not None
    assert result["answer"] == (
        "Start of Supplementary/Special University Examinations for all "
        "Undergraduate and Postgraduate Programmes is on 2026-10-12."
    )
    assert result["grounding"]["status"] == "grounded"
    assert result["documents"] == [document]
