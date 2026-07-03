import pytest
from app.utils.chunking import split_text

def test_split_text_handles_empty():
    assert split_text("") == []
    assert split_text("   \n  ") == []

def test_split_text_respects_newlines_and_spaces():
    text = "Hello\nWorld   test\tspaces"
    # The regex inside split_text replaces multiple spaces/tabs with single space, 
    # but preserves single newlines.
    chunks = split_text(text, chunk_size=100)
    assert len(chunks) == 1
    assert chunks[0] == "Hello\nWorld test spaces"

def test_split_text_breaks_at_word_boundaries():
    # Construct a string where splitting strictly at 20 characters
    # would break a word (e.g., "Software Engineering")
    text = "Introduction to Software Engineering"
    # at chunk_size=20, "Introduction to Soft" (20) -> break should backtrack to space
    chunks = split_text(text, chunk_size=20, overlap=5)
    
    assert len(chunks) >= 2
    # The first chunk should end at a word boundary
    assert chunks[0].endswith("to") or chunks[0].endswith("Introduction")

def test_split_text_preserves_table_formatting():
    # 3 newlines should become 2 newlines (paragraph break)
    text = "Header\n\n\nCourse | Credits\nCS101 | 3\n\nFooter"
    chunks = split_text(text, chunk_size=500)
    assert len(chunks) == 1
    assert "\n\n" in chunks[0]
    assert "Course | Credits\nCS101 | 3" in chunks[0]

def test_split_text_overlap():
    text = "A B C D E F G H I J K L M N O P"
    # Small chunks
    chunks = split_text(text, chunk_size=10, overlap=5)
    assert len(chunks) > 1
    # Check that there is overlap between first and second chunk
    # First chunk is ~10 chars, e.g., "A B C D E"
    # Second chunk should contain some characters from the first.
    words_chunk_1 = set(chunks[0].split())
    words_chunk_2 = set(chunks[1].split())
    # The intersection shouldn't be empty
    assert len(words_chunk_1.intersection(words_chunk_2)) > 0
