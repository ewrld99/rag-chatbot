from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import re

from app.utils.loaders import ExtractedDocument, ExtractedPage


_LIST_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s+|(?:\d+|[ivxlcdm]+|[a-z])[.)]\s+)",
    re.IGNORECASE,
)
_NUMBERED_HEADING_RE = re.compile(r"^\s*\d+(?:\.\d+)*\s+(.+)$")


@dataclass(frozen=True)
class PreparedChunk:
    text: str
    page_number: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ChunkDeduplication:
    chunks: tuple[PreparedChunk, ...]
    removed_count: int

def split_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    if not text or not text.strip():
        return []

    # Preserve single and double newlines for tables, but remove excessive whitespace
    clean_text = re.sub(r'[ \t]+', ' ', text)
    # Strip markdown headings so '## Title' becomes 'Title'
    clean_text = re.sub(r'(?m)^#+\s+', '', clean_text)
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text).strip()
    
    chunks = []
    start = 0
    text_len = len(clean_text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        # Try to find a natural break point if we're not at the very end
        if end < text_len:
            # 1. Try paragraph break
            break_idx = clean_text.rfind('\n\n', start, end)
            
            # 2. Try sentence break
            if break_idx == -1 or break_idx <= start + (chunk_size // 2):
                break_idx = max(clean_text.rfind('. ', start, end), 
                                clean_text.rfind('.\n', start, end))
                                
            # 3. Try newline (e.g. table rows)
            if break_idx == -1 or break_idx <= start + (chunk_size // 2):
                break_idx = clean_text.rfind('\n', start, end)
                
            # 4. Try space (word boundary)
            if break_idx == -1 or break_idx <= start + (chunk_size // 2):
                break_idx = clean_text.rfind(' ', start, end)
                
            if break_idx != -1 and break_idx > start:
                end = break_idx + 1  # Include the break character
        
        chunk = clean_text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        # Calculate next start using overlap, snapping to the next word boundary
        start_raw = max(end - overlap, start + 1)
        space_idx = clean_text.find(' ', start_raw, end)
        start = space_idx + 1 if space_idx != -1 else start_raw

    return chunks


def normalize_chunk_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    return normalized


def chunk_text_hash(text: str) -> str:
    return sha256(normalize_chunk_text(text).encode("utf-8")).hexdigest()


def reconstruct_text_from_chunks(
    texts: list[str],
    *,
    max_overlap: int = 500,
) -> str:
    """Rebuild source-like text without multiplying stored chunk overlap."""
    clean = [str(text or "").strip() for text in texts if str(text or "").strip()]
    if not clean:
        return ""
    result = clean[0]
    seen = {normalize_chunk_text(clean[0])}
    for value in clean[1:]:
        normalized = normalize_chunk_text(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        overlap_limit = min(max_overlap, len(result), len(value))
        overlap = 0
        for size in range(overlap_limit, 19, -1):
            if result.endswith(value[:size]):
                overlap = size
                break
        result += value[overlap:] if overlap else "\n\n" + value
    return result


def deduplicate_chunks(chunks: list[PreparedChunk]) -> ChunkDeduplication:
    unique: list[PreparedChunk] = []
    seen: set[str] = set()
    removed = 0
    for chunk in chunks:
        identity = chunk_text_hash(chunk.text)
        if not identity or identity in seen:
            removed += 1
            continue
        seen.add(identity)
        unique.append(chunk)
    return ChunkDeduplication(chunks=tuple(unique), removed_count=removed)


def split_extracted_document(
    document: ExtractedDocument,
    *,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[PreparedChunk]:
    chunks: list[PreparedChunk] = []
    for page in document.pages:
        chunks.extend(
            _split_page(
                page,
                chunk_size=max(200, int(chunk_size)),
                overlap=max(0, int(overlap)),
            )
        )
    return chunks


def _split_page(
    page: ExtractedPage,
    *,
    chunk_size: int,
    overlap: int,
) -> list[PreparedChunk]:
    known_headings = {heading.strip() for heading in page.headings}
    blocks = _page_blocks(page.content, known_headings=known_headings)
    if not blocks:
        return []

    section_title: str | None = None
    output: list[PreparedChunk] = []
    prose_buffer: list[str] = []

    def metadata(kind: str) -> dict[str, object]:
        values: dict[str, object] = {
            "extraction_method": page.extraction_method,
            "content_kind": kind,
        }
        if section_title:
            values["section_title"] = section_title
        if page.ocr_required:
            values["ocr_required"] = True
        return values

    def append_text(value: str, kind: str, *, prose_overlap: bool = False) -> None:
        text = value.strip()
        if not text:
            return
        prefix = f"{section_title}\n" if section_title and not text.startswith(section_title) else ""
        available = max(200, chunk_size - len(prefix))
        parts = (
            split_text(text, chunk_size=available, overlap=min(overlap, available // 4))
            if prose_overlap
            else _split_atomic_lines(text, available)
        )
        for part in parts:
            rendered = (prefix + part).strip()
            output.append(
                PreparedChunk(
                    text=rendered,
                    page_number=page.page_number,
                    metadata=metadata(kind),
                )
            )

    def flush_prose() -> None:
        if prose_buffer:
            append_text("\n\n".join(prose_buffer), "prose", prose_overlap=True)
            prose_buffer.clear()

    for block in blocks:
        if _is_heading(block, known_headings):
            flush_prose()
            section_title = block.strip().lstrip("#").strip()
            continue
        if _is_table_block(block):
            flush_prose()
            append_text(block, "table")
            continue
        if _is_list_block(block):
            flush_prose()
            append_text(block, "list")
            continue

        projected = len("\n\n".join([*prose_buffer, block]))
        if prose_buffer and projected > chunk_size:
            flush_prose()
        prose_buffer.append(block)

    flush_prose()
    if section_title and not output:
        append_text(section_title, "heading")
    return output


def _page_blocks(
    text: str,
    *,
    known_headings: set[str] | None = None,
) -> list[str]:
    clean = re.sub(r"[ \t]+", " ", str(text or ""))
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    if not clean:
        return []

    blocks: list[str] = []
    current: list[str] = []
    current_kind: str | None = None
    headings = known_headings or set()
    for raw_line in clean.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            if current:
                blocks.append("\n".join(current).strip())
                current = []
                current_kind = None
            continue
        if line.strip().lstrip("#").strip() in headings:
            if current:
                blocks.append("\n".join(current).strip())
                current = []
                current_kind = None
            blocks.append(line.strip())
            continue
        kind = "table" if line.lstrip().startswith("|") else "list" if _LIST_LINE_RE.match(line) else "text"
        if current_kind == "list" and kind == "text":
            # Wrapped list-item text belongs to the preceding list until a
            # blank line or another structured block provides a boundary.
            kind = "list"
        if current and kind != current_kind and {kind, current_kind} & {"table", "list"}:
            blocks.append("\n".join(current).strip())
            current = []
        current.append(line)
        current_kind = kind
    if current:
        blocks.append("\n".join(current).strip())
    return [block for block in blocks if block]


def _is_heading(block: str, known_headings: set[str]) -> bool:
    stripped = block.strip().lstrip("#").strip()
    if "\n" in stripped or len(stripped) > 120:
        return False
    if stripped in known_headings or block.lstrip().startswith("#") or _looks_numbered_heading(stripped):
        return True
    letters = [character for character in stripped if character.isalpha()]
    return bool(
        len(letters) >= 4
        and sum(character.isupper() for character in letters) / len(letters) >= 0.8
    )


def _looks_numbered_heading(text: str) -> bool:
    match = _NUMBERED_HEADING_RE.match(text)
    if not match:
        return False
    title = match.group(1).strip()
    words = re.findall(r"[A-Za-z]+", title)
    if not words or len(words) > 8 or len(title) > 80:
        return False
    significant = [word for word in words if word.lower() not in {"a", "an", "and", "for", "of", "the", "to"}]
    return bool(
        significant
        and sum(word[0].isupper() for word in significant) / len(significant) >= 0.75
    )


def _is_table_block(block: str) -> bool:
    lines = [line for line in block.splitlines() if line.strip()]
    return bool(lines and all(line.lstrip().startswith("|") for line in lines))


def _is_list_block(block: str) -> bool:
    lines = [line for line in block.splitlines() if line.strip()]
    return bool(lines and sum(bool(_LIST_LINE_RE.match(line)) for line in lines) >= 1)


def _split_atomic_lines(text: str, chunk_size: int) -> list[str]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    table_header = lines[:2] if _is_table_block(text) and len(lines) >= 2 else []
    parts: list[str] = []
    current = list(table_header) if table_header else []
    start_index = len(table_header)
    for line in lines[start_index:]:
        projected = "\n".join([*current, line])
        if current and len(projected) > chunk_size:
            parts.append("\n".join(current))
            current = list(table_header) if table_header else []
        if len(line) > chunk_size:
            if current and current != table_header:
                parts.append("\n".join(current))
                current = list(table_header) if table_header else []
            parts.extend(split_text(line, chunk_size=chunk_size, overlap=0))
            continue
        current.append(line)
    if current and (not table_header or current != table_header):
        parts.append("\n".join(current))
    if not parts and lines:
        parts.append("\n".join(lines))
    return parts

