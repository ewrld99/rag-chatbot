import asyncio

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
        self.received_draft_callback = False

    async def _generate_grounded_async(self, *_args, **kwargs):
        self.received_draft_callback = kwargs.get("draft_delta_callback") is not None
        return GroundingOutcome(
            answer="1. Submit the request through SR2.",
            coverage="full",
            evidence_ids=("ev-1",),
            claim_count=1,
            supported_claim_count=1,
            repaired=False,
            status="grounded",
        )


def test_document_stream_emits_only_verified_replacement():
    service = _StreamingGenerationService()

    async def collect_events():
        return [
            event
            async for event in service.stream_generate(
                "How do I postpone studies?",
                "<documents />",
            )
        ]

    events = asyncio.run(collect_events())

    assert service.received_draft_callback is False
    assert events == [{
        "type": "replace",
        "answer": "1. Submit the request through SR2.",
        "provisional": False,
    }]
