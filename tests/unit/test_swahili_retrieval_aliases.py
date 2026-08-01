from app.services.alias_expansion_service import AliasExpansionService, AliasRecord
from app.services.query_normalization import build_sparse_query_variants
from app.services.sparse_retriever import SparseRetriever


def test_swahili_postponement_alias_expands_for_retrieval():
    expansions = AliasExpansionService(db=None).get_expansions(
        "utaratibu wa kuahirisha masomo"
    )

    assert "postponement" in expansions
    assert "deferment" in expansions["postponement"]
    assert "kuahirisha masomo" in expansions["postponement"]


def test_swahili_examination_terms_expand_sparse_query():
    variants = build_sparse_query_variants(
        "nitaruhusiwa kufanya mtihani kama coursework ni ndogo"
    )
    expanded = " ".join(variant.query for variant in variants)

    assert "examination" in expanded
    assert "coursework" in expanded
    assert "continuous assessment" in expanded


def test_idit_alias_expands_to_full_programme_name_for_fee_queries():
    expansions = AliasExpansionService(db=None).get_expansions(
        "fee for IDIT students"
    )
    variants = build_sparse_query_variants(
        "fee for IDIT students",
        alias_expansions=expansions,
    )
    expanded = " ".join(variant.query for variant in variants)

    assert "idit" in expansions
    assert "instructional design and information technology" in expanded
    assert "dm074" in expanded


def test_sparse_alias_boost_prioritizes_specific_programme_matches():
    retriever = SparseRetriever(db=None)
    terms = retriever._alias_boost_terms(
        {
            "fee": ["fees", "tuition"],
            "idit": [
                "instructional design and information technology",
                "dm074",
            ],
        }
    )

    boost = retriever._alias_text_boost(
        "Programme Code: DM074. Bachelor of Science in Instructional Design and Information Technology.",
        terms,
    )
    generic_boost = retriever._alias_text_boost(
        "A student pays annual tuition fees and direct costs.",
        terms,
    )

    assert boost > 0
    assert generic_boost == 0


def test_common_stopword_does_not_trigger_acronym_expansion(monkeypatch):
    service = AliasExpansionService(db=None)
    monkeypatch.setattr(
        service,
        "_load_records",
        lambda: (
            AliasRecord(
                term="to",
                aliases=("transport officer",),
                category="acronym",
                weight=1.0,
            ),
        ),
    )

    expansions = service.get_expansions(
        "what are the procedures to postpone year of study?"
    )

    assert "to" not in expansions


def test_sparse_expansion_is_bounded_for_large_alias_sets():
    variants = build_sparse_query_variants(
        "procedures to postpone year of study",
        alias_expansions={
            "postpone": [
                "postponement",
                "deferment",
                "defer",
                "intermission",
                "kuahirisha",
                "ahirisha",
                "kuahirisha masomo",
                "kusitisha masomo",
                "postponement of studies",
                "postponed",
                "postpone studies",
                "postpone study",
                "postpone year",
                "postpone year of study",
            ]
        },
    )
    expanded = next(variant.query for variant in variants if variant.label == "expanded")

    assert expanded.count(" OR ") + 1 <= 12
