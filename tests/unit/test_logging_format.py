from app.core.logging import format_log_event


def test_format_log_event_renders_nested_fields_readably():
    output = format_log_event(
        "RAG latency",
        total_ms=123.4,
        stages={"retrieval_ms": 80.1, "generation_ms": 40.2},
        query="what are the courses?",
    )

    assert output.startswith("RAG latency\n")
    assert "  total_ms: 123.4" in output
    assert "  stages:" in output
    assert '    "retrieval_ms": 80.1' in output
    assert "  query: what are the courses?" in output
