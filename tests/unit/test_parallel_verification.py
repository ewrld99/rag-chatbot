from __future__ import annotations

import time
from unittest.mock import MagicMock
from app.services.generation_service import GenerationService, VerificationReport, GroundedDraft, EvidenceChunk


def test_parallel_semantic_verify_batches_execute_concurrently():
    service = GenerationService.__new__(GenerationService)

    call_timestamps = []

    def mock_create_utility_completion(op_name, messages, **_kwargs):
        call_timestamps.append(time.time())
        time.sleep(0.1)  # Simulate small LLM latency
        mock_resp = MagicMock()
        mock_resp.choices = [
            MagicMock(content='{"verdicts": {"0": "SUPPORTED", "1": "SUPPORTED"}}')
        ]
        return mock_resp

    service._create_utility_completion = mock_create_utility_completion

    mock_grounding = MagicMock()
    mock_grounding.parse_verification.side_effect = lambda raw, batch: VerificationReport(
        verdicts={i: "SUPPORTED" for i in batch}, errors=[]
    )
    service.grounding = mock_grounding

    draft = GroundedDraft(
        answer="Sample answer",
        coverage="full",
        claims=[],
        evidence_ids=[],
    )
    evidence = {
        "ev-1": EvidenceChunk(
            id="ev-1",
            document_id="doc-1",
            text="Evidence text sample",
            source="source-1",
        )
    }
    claim_indexes = list(range(12))  # 12 claims split into multiple batches

    start_time = time.time()
    report = service._semantic_verify(draft, evidence, claim_indexes, query="test query")
    elapsed = time.time() - start_time

    assert len(report.verdicts) == 12
    for i in range(12):
        assert report.verdicts[i] == "SUPPORTED"

    # Concurrency verification: 6 batches running in parallel should take ~0.1-0.2s instead of 0.6s+
    assert elapsed < 0.55
