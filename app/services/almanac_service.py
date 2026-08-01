from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import re
from typing import Any


_EVENT_RE = re.compile(
    r"(?P<date>(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+"
    r"\d{1,2},\s+\d{4})\s+"
    r"(?P<activity>.*?)(?=(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+"
    r"\d{1,2},\s+\d{4}\s+|\Z)",
    re.DOTALL,
)
_ACADEMIC_YEAR_RE = re.compile(r"\b(?:20\d{2}/(?:20)?\d{2}|20\d{2}-20\d{4})\b")
_MONTH_HEADER_RE = re.compile(
    r"^\s*(?:January|February|March|April|May|June|July|August|September|October|November|December),\s+20\d{2}\s*$",
    re.IGNORECASE,
)
_DATE_START_RE = re.compile(
    r"^\s*(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AlmanacEvent:
    event_date: date
    activity: str
    academic_year: str | None
    event_type: str

    def chunk_text(self) -> str:
        lines = [
            "UDOM Academic Almanac Event",
            f"Date: {self.event_date.isoformat()}",
            f"Activity: {self.activity}",
            f"Event Type: {self.event_type}",
        ]
        if self.academic_year:
            lines.append(f"Academic Year: {self.academic_year}")
        return "\n".join(lines)

    def metadata(self) -> dict[str, Any]:
        return {
            "category": "academic_calendar",
            "record_type": "almanac_event",
            "date": self.event_date.isoformat(),
            "academic_year": self.academic_year,
            "event_type": self.event_type,
        }


def parse_almanac_events(text: str) -> list[AlmanacEvent]:
    """Extract one searchable event per dated almanac row."""
    clean_text = _normalize_almanac_text(text)
    events: list[AlmanacEvent] = []
    current_academic_year: str | None = None

    for match in _EVENT_RE.finditer(clean_text):
        raw_date = match.group("date")
        activity = _clean_activity(match.group("activity"))
        if not activity:
            continue

        explicit_year = _academic_year_from_text(activity)
        if explicit_year:
            current_academic_year = explicit_year

        event_date = datetime.strptime(raw_date, "%A, %B %d, %Y").date()
        events.append(
            AlmanacEvent(
                event_date=event_date,
                activity=activity,
                academic_year=explicit_year or current_academic_year or _academic_year_for_date(event_date),
                event_type=_classify_event_type(activity),
            )
        )

    return events


def build_almanac_chunks(text: str) -> list[tuple[str, dict[str, Any]]]:
    return [(event.chunk_text(), event.metadata()) for event in parse_almanac_events(text)]


def _normalize_almanac_text(text: str) -> str:
    normalized = str(text or "").replace("\xa0", " ")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"(?m)^Page \d+ of \d+\s*$", "", normalized)
    normalized = re.sub(r"(?m)^\|?\s*DATE\s*\|?\s*ACTIVITY\s*\|?\s*$", "", normalized, flags=re.IGNORECASE)
    normalized = "\n".join(
        clean_line
        for line in normalized.splitlines()
        if (clean_line := line.strip(" |"))
        and not _MONTH_HEADER_RE.match(clean_line)
        and not _is_section_heading(clean_line)
    )
    return re.sub(r"\n+", "\n", normalized)


def _clean_activity(activity: str) -> str:
    activity = re.sub(r"\|", " ", activity)
    activity = re.sub(r"-{2,}", " ", activity)
    activity = re.sub(r"\s+", " ", activity)
    activity = re.sub(r"\b([A-Za-z])\s+([a-z]{2,})\b", r"\1\2", activity)
    return activity.strip(" .") + "." if activity.strip(" .") else ""


def _is_section_heading(line: str) -> bool:
    if _DATE_START_RE.match(line):
        return False
    letters = [character for character in line if character.isalpha()]
    if len(letters) < 8:
        return False
    uppercase_ratio = sum(character.isupper() for character in letters) / len(letters)
    return uppercase_ratio >= 0.8


def _academic_year_from_text(text: str) -> str | None:
    match = _ACADEMIC_YEAR_RE.search(text)
    if not match:
        return None
    value = match.group(0)
    if "-" in value:
        return value.replace("-", "/")
    parts = value.split("/")
    if len(parts[1]) == 2:
        return f"{parts[0]}/20{parts[1]}"
    return value


def _academic_year_for_date(value: date) -> str:
    start_year = value.year if value.month >= 8 else value.year - 1
    return f"{start_year}/{start_year + 1}"


def _classify_event_type(activity: str) -> str:
    lowered = activity.lower()
    if "registration" in lowered:
        return "registration"
    if "orientation" in lowered:
        return "orientation"
    if "teaching" in lowered:
        return "teaching"
    if "examination" in lowered or "exam" in lowered:
        return "examination"
    if "graduation" in lowered or "convocation" in lowered:
        return "graduation"
    if "holiday" in lowered or "recess" in lowered:
        return "holiday"
    if "senate" in lowered or "council" in lowered or "committee" in lowered or "board" in lowered:
        return "meeting"
    return "academic_calendar"
