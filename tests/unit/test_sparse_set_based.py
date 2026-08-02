from app.services import sparse_retriever as sparse_module
from app.services.sparse_retriever import SparseRetriever


class _Rows:
    def fetchall(self):
        return []


class _DB:
    def __init__(self):
        self.execute_count = 0

    def execute(self, *_args, **_kwargs):
        self.execute_count += 1
        return _Rows()

    def rollback(self):
        pass


def test_sparse_variants_use_one_database_round_trip(monkeypatch):
    db = _DB()
    monkeypatch.setattr(
        sparse_module.AliasExpansionService,
        "get_expansions",
        lambda *_args, **_kwargs: {"gpa": ["grade point average"]},
    )

    SparseRetriever(db).retrieve("How is GPA calculated?", top_k=20)

    assert db.execute_count == 1
