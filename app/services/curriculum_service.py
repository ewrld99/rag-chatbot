from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


_ACADEMIC_YEAR_RE = re.compile(r"\b(20\d{2})\s*/\s*((?:20)?\d{2})\b")
_COLLEGE_RE = re.compile(
    r"^\s*(?P<section>\d+\.\d+)\s+(?P<college>(?:COLLEGE|SCHOOL|INSTITUTE)[^\n]+)$",
    re.IGNORECASE,
)
_PROGRAMME_RE = re.compile(
    r"^\s*(?P<section>\d+\.\d+(?:\.\d+)?)\s+(?P<programme>(?:Bachelor|Diploma|Certificate|Shahada)[^\n]+)$",
    re.IGNORECASE,
)
_SECTION_ONLY_RE = re.compile(r"^\s*(?P<section>\d+\.\d+\.\d+)\s*$")
_PROGRAMME_LIST_RE = re.compile(
    r"^\s*(?P<number>\d+)\.\s+(?P<programme>(?:Bachelor|Diploma|Certificate|Shahada)[^\n]+)$",
    re.IGNORECASE,
)
_ACRONYM_RE = re.compile(r"\((?P<acronym>[A-Z][A-Za-z.\s]{1,20})\)")
_DESCRIPTION_ACRONYM_RE = re.compile(r"\b(?P<acronym>B[A-Za-z]{1,5}(?:\s+[A-Z]{2,6})?)\s+programme\b")
_COURSE_CODE_RE = re.compile(r"^[A-Z]{2,4}\s*\d{3,4}[A-Z]?$")
_PLAIN_COURSE_RE = re.compile(
    r"^(?P<code>[A-Z]{2,4}\s*\d{3,4}[A-Z]?)\s+"
    r"(?P<title>.+?)\s+"
    r"(?P<status>Core|Elective|Optional|Option)\s+"
    r"(?P<credits>\d+(?:\.\d+)?)$",
    re.IGNORECASE,
)
_YEAR_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
}
_SEMESTER_WORDS = {
    "one": "1",
    "two": "2",
    "1": "1",
    "2": "2",
}


@dataclass(frozen=True)
class CurriculumCourse:
    academic_year: str | None
    college: str | None
    programme: str | None
    programme_acronym: str | None
    year_of_study: str | None
    semester: str | None
    course_code: str
    course_title: str
    status: str
    credits: str

    def chunk_text(self) -> str:
        lines = [
            "UDOM Undergraduate Curriculum Course",
        ]
        if self.academic_year:
            lines.append(f"Academic Year: {self.academic_year}")
        if self.college:
            lines.append(f"College/School/Institute: {self.college}")
        if self.programme:
            lines.append(f"Programme: {self.programme}")
        if self.programme_acronym:
            lines.append(f"Programme Acronym: {self.programme_acronym}")
        if self.year_of_study:
            lines.append(f"Year of Study: Year {self.year_of_study}")
        if self.semester:
            lines.append(f"Semester: Semester {self.semester}")
        lines.extend(
            [
                f"Course Code: {self.course_code}",
                f"Course Title: {self.course_title}",
                f"Status: {self.status}",
                f"Credits: {self.credits}",
            ]
        )
        return "\n".join(lines)

    def metadata(self) -> dict[str, Any]:
        return {
            "category": "curriculum",
            "record_type": "course",
            "academic_year": self.academic_year,
            "college": self.college,
            "programme": self.programme,
            "programme_acronym": self.programme_acronym,
            "year_of_study": self.year_of_study,
            "semester": self.semester,
            "course_code": self.course_code,
            "course_title": self.course_title,
            "status": self.status,
            "credits": self.credits,
        }


def parse_curriculum_courses(text: str) -> list[CurriculumCourse]:
    """Extract one searchable chunk per course row from a curriculum guidebook."""
    lines = _normalize_curriculum_text(text).splitlines()
    academic_year = _academic_year_from_text(text)
    college: str | None = None
    programme: str | None = None
    programme_acronym: str | None = None
    year_of_study: str | None = None
    semester: str | None = None
    current_college_section: str | None = None
    programme_by_section: dict[str, str] = {}
    courses: list[CurriculumCourse] = []
    seen: dict[tuple[str | None, str | None, str | None, str], int] = {}

    for line in lines:
        clean_line = _clean_value(line)
        college_match = _COLLEGE_RE.match(clean_line)
        if college_match:
            current_college_section = college_match.group("section")

        list_match = _PROGRAMME_LIST_RE.match(clean_line)
        if list_match and current_college_section:
            programme_by_section[f"{current_college_section}.{list_match.group('number')}"] = _clean_programme(
                list_match.group("programme")
            )

        section_match = _SECTION_ONLY_RE.match(clean_line)
        programme_match = _PROGRAMME_RE.match(clean_line)

        context = _context_from_line(line)
        section = None
        if section_match:
            section = section_match.group("section")
        elif programme_match:
            section = programme_match.group("section")

        if section and section in programme_by_section:
            mapped_programme = programme_by_section[section]
            context["programme"] = mapped_programme
            mapped_acronym = _acronym_from_text(mapped_programme)
            if mapped_acronym:
                context["programme_acronym"] = mapped_acronym

        college = context.get("college", college)
        programme = context.get("programme", programme)
        programme_acronym = context.get("programme_acronym", programme_acronym)
        year_of_study = context.get("year_of_study", year_of_study)
        semester = context.get("semester", semester)

        parsed = _course_from_line(line)
        if not parsed:
            continue

        course = CurriculumCourse(
            academic_year=academic_year,
            college=college,
            programme=programme,
            programme_acronym=programme_acronym,
            year_of_study=year_of_study,
            semester=semester,
            course_code=parsed["course_code"],
            course_title=parsed["course_title"],
            status=parsed["status"],
            credits=parsed["credits"],
        )
        key = (
            course.programme,
            course.year_of_study,
            course.semester,
            course.course_code,
        )
        existing_index = seen.get(key)
        if existing_index is None:
            seen[key] = len(courses)
            courses.append(course)
        elif len(course.course_title) > len(courses[existing_index].course_title):
            courses[existing_index] = course

    return courses


def build_curriculum_chunks(text: str) -> list[tuple[str, dict[str, Any]]]:
    return [(course.chunk_text(), course.metadata()) for course in parse_curriculum_courses(text)]


def _normalize_curriculum_text(text: str) -> str:
    normalized = str(text or "").replace("\xa0", " ")
    normalized = normalized.replace("\u2013", "-").replace("\u2014", "-")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    lines = [line.strip(" |") for line in normalized.splitlines() if line.strip(" |")]
    return "\n".join(_merge_wrapped_programme_lines(lines))


def _merge_wrapped_programme_lines(lines: list[str]) -> list[str]:
    merged: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if (
            index + 1 < len(lines)
            and _looks_like_incomplete_programme_line(line)
            and _looks_like_acronym_continuation(lines[index + 1])
        ):
            merged.append(f"{line} {lines[index + 1]}")
            index += 2
            continue
        merged.append(line)
        index += 1
    return merged


def _looks_like_incomplete_programme_line(line: str) -> bool:
    if not _PROGRAMME_RE.match(_clean_value(line)):
        return False
    return line.count("(") > line.count(")")


def _looks_like_acronym_continuation(line: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Za-z.\s]{1,20}\)?", _clean_value(line)))


def _context_from_line(line: str) -> dict[str, str | None]:
    context: dict[str, str | None] = {}
    clean_line = _clean_value(line)

    college_match = _COLLEGE_RE.match(clean_line)
    if college_match and "." not in college_match.group("college"):
        context["college"] = _clean_value(college_match.group("college"))

    programme_match = _PROGRAMME_RE.match(clean_line)
    if programme_match and "..." not in clean_line:
        programme = _clean_programme(programme_match.group("programme"))
        if programme:
            context["programme"] = programme
            acronym = _acronym_from_text(programme)
            if acronym:
                context["programme_acronym"] = acronym

    acronym_match = _DESCRIPTION_ACRONYM_RE.search(clean_line)
    if acronym_match:
        context["programme_acronym"] = _clean_value(acronym_match.group("acronym"))

    year_semester = _year_semester_from_line(clean_line)
    context.update(year_semester)
    return context


def _course_from_line(line: str) -> dict[str, str] | None:
    if "|" in line:
        parsed = _course_from_table_line(line)
        if parsed:
            return parsed

    match = _PLAIN_COURSE_RE.match(_clean_value(line))
    if not match:
        return None

    return _validated_course(
        match.group("code"),
        match.group("title"),
        match.group("status"),
        match.group("credits"),
    )


def _course_from_table_line(line: str) -> dict[str, str] | None:
    cells = [_clean_value(cell) for cell in line.strip().strip("|").split("|")]
    cells = [cell for cell in cells if cell]
    if len(cells) < 4:
        return None

    code_index = next((index for index, cell in enumerate(cells) if _COURSE_CODE_RE.match(cell)), None)
    if code_index is None:
        return None

    remaining = cells[code_index + 1 :]
    status_index = next(
        (
            index
            for index, cell in enumerate(remaining)
            if cell.lower() in {"core", "elective", "optional", "option"}
        ),
        None,
    )
    if status_index is None:
        return None

    credit_index = next(
        (index for index, cell in enumerate(remaining[status_index + 1 :], start=status_index + 1) if _is_credit(cell)),
        None,
    )
    if credit_index is None:
        return None

    title = " ".join(remaining[:status_index])
    return _validated_course(cells[code_index], title, remaining[status_index], remaining[credit_index])


def _validated_course(code: str, title: str, status: str, credits: str) -> dict[str, str] | None:
    course_code = _clean_course_code(code)
    course_title = _clean_title(title)
    status_value = _clean_value(status).title()
    credits_value = _clean_value(credits)

    if not _COURSE_CODE_RE.match(course_code):
        return None
    if len(course_title) < 3 or any(noise in course_title.lower() for noise in ("course title", "course name")):
        return None
    if not _is_credit(credits_value):
        return None

    return {
        "course_code": course_code,
        "course_title": course_title,
        "status": status_value,
        "credits": credits_value,
    }


def _year_semester_from_line(line: str) -> dict[str, str]:
    context: dict[str, str] = {}
    lowered = line.lower()

    combined = re.search(r"semester\s+(one|two|1|2)\s*,?\s*year\s+(one|two|three|four|five|1|2|3|4|5)", lowered)
    if combined:
        context["semester"] = _SEMESTER_WORDS[combined.group(1)]
        context["year_of_study"] = _year_value(combined.group(2))
        return context

    year_match = re.search(r"\byear\s+(one|two|three|four|five|1|2|3|4|5)\b", lowered)
    if year_match:
        context["year_of_study"] = _year_value(year_match.group(1))

    semester_match = re.search(r"\bsemester\s+(one|two|1|2)\b", lowered)
    if semester_match:
        context["semester"] = _SEMESTER_WORDS[semester_match.group(1)]

    return context


def _year_value(value: str) -> str:
    return _YEAR_WORDS.get(value, value)


def _academic_year_from_text(text: str) -> str | None:
    match = _ACADEMIC_YEAR_RE.search(text)
    if not match:
        return None
    end_year = match.group(2)
    if len(end_year) == 2:
        end_year = f"20{end_year}"
    return f"{match.group(1)}/{end_year}"


def _acronym_from_text(text: str) -> str | None:
    match = _ACRONYM_RE.search(text)
    return _clean_value(match.group("acronym")) if match else None


def _clean_programme(value: str) -> str:
    value = _clean_value(value)
    value = re.sub(r"\s+\d+$", "", value)
    value = value.rstrip(".")
    return value


def _clean_course_code(value: str) -> str:
    return re.sub(r"\s+", " ", _clean_value(value)).upper()


def _clean_title(value: str) -> str:
    value = _clean_value(value)
    value = re.sub(r"\b([A-Za-z])\s+([a-z]{2,})\b", r"\1\2", value)
    return value


def _clean_value(value: str | None) -> str:
    value = str(value or "").replace("\n", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .")


def _is_credit(value: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)?", _clean_value(value)))
