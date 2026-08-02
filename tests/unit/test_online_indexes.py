import pytest

from app.db import online_indexes


def test_online_rollout_marks_complete_only_after_validation(monkeypatch):
    calls = []
    monkeypatch.setattr(
        online_indexes,
        "backfill_retrievability",
        lambda batch_size: calls.append(("backfill", batch_size)) or 7,
    )
    monkeypatch.setattr(
        online_indexes,
        "build_online_indexes",
        lambda: calls.append(("build", None)),
    )
    monkeypatch.setattr(
        online_indexes,
        "validate_online_indexes",
        lambda: calls.append(("validate", None)),
    )
    monkeypatch.setattr(
        online_indexes,
        "mark_complete",
        lambda: calls.append(("mark", None)),
    )

    assert online_indexes.run(batch_size=250) == 7
    assert calls == [
        ("backfill", 250),
        ("build", None),
        ("validate", None),
        ("mark", None),
    ]


def test_online_rollout_does_not_mark_failed_validation(monkeypatch):
    marked = []
    monkeypatch.setattr(online_indexes, "backfill_retrievability", lambda batch_size: 0)
    monkeypatch.setattr(online_indexes, "build_online_indexes", lambda: None)
    monkeypatch.setattr(
        online_indexes,
        "validate_online_indexes",
        lambda: (_ for _ in ()).throw(RuntimeError("invalid index")),
    )
    monkeypatch.setattr(online_indexes, "mark_complete", lambda: marked.append(True))

    with pytest.raises(RuntimeError, match="invalid index"):
        online_indexes.run()
    assert marked == []
