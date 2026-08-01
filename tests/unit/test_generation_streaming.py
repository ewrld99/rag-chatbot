import asyncio
import json

from app.services.generation_service import (
    GenerationService,
    _JSONClaimStreamExtractor,
)
from app.services.grounding_service import GroundingOutcome, GroundingService


def test_claim_stream_extractor_handles_fragmented_json_and_escapes():
    extractor = _JSONClaimStreamExtractor(ordered=True)
    fragments = [
        '{"coverage":"full","claims":[{"cl',
        'aim":"Submit the online form.\\nKeep the receipt.","evidence_ids":["ev-1"]},',
        '{"claim":"Wait for Senate approval.","evidence_ids":["ev-2"]}]}',
    ]

    rendered = "".join(
        part
        for fragment in fragments
        for part in extractor.feed(fragment)
    )

    assert rendered == (
        "1. Submit the online form.\nKeep the receipt.\n"
        "2. Wait for Senate approval.\n"
    )


class _StreamingGenerationService(GenerationService):
    def __init__(self):
        self.grounding = GroundingService()
        self._last_grounding_outcome = None
        self._last_model_execution = None
        self.model_preference = "auto"

    async def _generate_grounded_async(self, *_args, **kwargs):
        callback = kwargs["draft_delta_callback"]
        payload = json.dumps(
            {
                "coverage": "full",
                "claims": [
                    {
                        "claim": "Submit the request through SR2.",
                        "evidence_ids": ["ev-1"],
                    }
                ],
            }
        )
        for start in range(0, len(payload), 12):
            await callback(payload[start:start + 12])
        return GroundingOutcome(
            answer="1. Submit the request through SR2.",
            coverage="full",
            evidence_ids=("ev-1",),
            claim_count=1,
            supported_claim_count=1,
            repaired=False,
            status="grounded",
        )


def test_document_stream_emits_draft_tokens_then_verified_replacement():
    async def collect_events():
        service = _StreamingGenerationService()
        return [
            event
            async for event in service.stream_generate(
                "How do I postpone studies?",
                "<documents />",
            )
        ]

    events = asyncio.run(collect_events())

    assert any(event["type"] == "stream" for event in events)
    assert events[-1] == {
        "type": "replace",
        "answer": "1. Submit the request through SR2.",
        "provisional": False,
    }
