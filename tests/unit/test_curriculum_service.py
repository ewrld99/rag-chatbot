from app.services.curriculum_service import build_curriculum_chunks, parse_curriculum_courses


def test_parse_curriculum_courses_extracts_course_rows_with_context():
    text = """
    UNDERGRADUATE CURRICULUM GUIDEBOOK 2025/2026 ACADEMIC YEAR
    2.5 COLLEGE OF INFORMATICS AND VIRTUAL EDUCATION (CIVE)
    2.5.2 Bachelor of Science in Instructional Design and Information Technology (BSc IDIT)
    Programme Structure
    Semester 2, Year 1
    | Course Code | Course Name | Status | Credits |
    | --- | --- | --- | --- |
    | CP 1201 | Introduction to Database Systems | Core | 9.0 |
    | AI 1201 | Fundamentals and Principles of AI and Machine Learning | Core | 7.5 |
    """

    courses = parse_curriculum_courses(text)

    assert len(courses) == 2
    assert courses[0].academic_year == "2025/2026"
    assert courses[0].college == "COLLEGE OF INFORMATICS AND VIRTUAL EDUCATION (CIVE)"
    assert courses[0].programme == "Bachelor of Science in Instructional Design and Information Technology (BSc IDIT)"
    assert courses[0].programme_acronym == "BSc IDIT"
    assert courses[0].year_of_study == "1"
    assert courses[0].semester == "2"
    assert courses[0].course_code == "CP 1201"
    assert courses[0].course_title == "Introduction to Database Systems"
    assert courses[0].status == "Core"
    assert courses[0].credits == "9.0"
    assert courses[1].course_title == "Fundamentals and Principles of AI and Machine Learning"


def test_parse_curriculum_courses_merges_wrapped_programme_acronym():
    text = """
    UNDERGRADUATE CURRICULUM GUIDEBOOK 2025/2026 ACADEMIC YEAR
    2.5 COLLEGE OF INFORMATICS AND VIRTUAL EDUCATION (CIVE)
    2.5.3 Bachelor of Science in Computer Networks and Information Security Engineering (BSc
    CNISE)
    Programme Structure
    Year One
    Semester One
    IS 111 Information Security Foundations Core 7.5
    """

    courses = parse_curriculum_courses(text)

    assert len(courses) == 1
    assert courses[0].programme == (
        "Bachelor of Science in Computer Networks and Information Security Engineering (BSc CNISE)"
    )
    assert courses[0].programme_acronym == "BSc CNISE"


def test_build_curriculum_chunks_creates_searchable_course_chunks():
    chunks = build_curriculum_chunks(
        """
        2.1.1 Bachelor of Commerce in Accounting (BCom Accounting)
        Year One
        Semester One
        DS 102 Development Perspectives Core 7.5
        """
    )

    assert len(chunks) == 1
    chunk_text, metadata = chunks[0]
    assert "UDOM Undergraduate Curriculum Course" in chunk_text
    assert "Programme: Bachelor of Commerce in Accounting (BCom Accounting)" in chunk_text
    assert "Year of Study: Year 1" in chunk_text
    assert "Semester: Semester 1" in chunk_text
    assert "Course Code: DS 102" in chunk_text
    assert "Course Title: Development Perspectives" in chunk_text
    assert metadata["category"] == "curriculum"
    assert metadata["record_type"] == "course"
    assert metadata["course_code"] == "DS 102"
    assert metadata["credits"] == "7.5"


def test_parse_curriculum_courses_maps_bare_section_numbers_to_programme_list():
    text = """
    2.1 COLLEGE OF BUSINESS AND ECONOMICS (CoBE)
    1. Bachelor of Commerce in Accounting (BCom Accounting)
    2. Bachelor of Commerce in Finance (BCom Finance)
    2.1.2
    Programme Description
    Programme Structure
    Year One
    Semester One
    DS 102 Development Perspectives Core 7.5
    """

    courses = parse_curriculum_courses(text)

    assert len(courses) == 1
    assert courses[0].programme == "Bachelor of Commerce in Finance (BCom Finance)"
    assert courses[0].programme_acronym == "BCom Finance"
