import pytest
from app.services.generation_service import GenerationService, GroundedDraft, GroundedClaim, EvidenceChunk

@pytest.mark.asyncio
async def test_fast_path_deterministic_semantic_verify():
    generator = GenerationService()

    draft = GroundedDraft(
        coverage="full",
        answer="GPA is calculated using grade points divided by units.",
        claims=[
            GroundedClaim(
                claim="GPA is calculated using grade points divided by units",
                evidence_ids=["ev_1"],
            )
        ]
    )

    evidence = {
        "ev_1": EvidenceChunk(
            evidence_id="ev_1",
            document_id="doc_1",
            chunk_id="chunk_1",
            text="The GPA is calculated using grade points divided by total units enrolled in that semester.",
        )
    }

    report = await generator._semantic_verify_async(
        draft=draft,
        evidence=evidence,
        claim_indexes=[0],
        query="how is GPA calculated?",
    )

    assert 0 in report.verdicts
    assert report.verdicts[0] == "SUPPORTED"
