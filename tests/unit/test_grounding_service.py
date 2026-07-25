from __future__ import annotations

import asyncio
import json

from langchain_core.documents import Document

from app.services.generation_service import GenerationService
from app.services.grounding_service import (
    GroundedClaim,
    GroundedDraft,
    GroundingService,
    VerificationReport,
)


def _document(
    text: str = "A candidate must attend at least 75 percent of a course.",
    chunk_id: str = "chunk-1",
    document_id: str = "document-1",
) -> Document:
    return Document(
        page_content=text,
        metadata={
            "chunk_id": chunk_id,
            "document_id": document_id,
            "chunk_index": 4,
            "source": "regulations.pdf",
        },
    )


def _draft(answer: str, evidence_id: str) -> GroundedDraft:
    return GroundedDraft(
        coverage="full",
        answer=answer,
        claims=[
            GroundedClaim(
                claim=answer,
                evidence_ids=[evidence_id],
            )
        ],
    )


def test_valid_claim_is_grounded_and_filters_sources():
    service = GroundingService()
    first = _document()
    second = _document(
        text="A separate rule applies to postgraduate candidates.",
        chunk_id="chunk-2",
        document_id="document-2",
    )
    evidence = service.build_evidence([first, second])
    first_id = service.evidence_id(first)
    draft = _draft(
        "A candidate must attend at least 75 percent of a course.",
        first_id,
    )

    validation = service.validate(draft, evidence)
    verification = VerificationReport(verdicts={0: "SUPPORTED"})
    outcome = service.finalize(
        draft,
        validation,
        verification,
        refusal="No evidence.",
        repaired=False,
    )

    assert validation.valid is True
    assert outcome.status == "grounded"
    assert outcome.evidence_ids == (first_id,)
    assert service.filter_documents([first, second], outcome.evidence_ids) == [first]


def test_claim_with_number_absent_from_evidence_is_rejected():
    service = GroundingService()
    document = _document()
    evidence = service.build_evidence([document])
    draft = _draft(
        "A candidate must attend at least 80 percent of a course.",
        service.evidence_id(document),
    )

    validation = service.validate(draft, evidence)

    assert validation.valid is False
    assert "80" in " ".join(validation.claim_errors[0])


def test_invalid_evidence_id_and_uncovered_answer_sentence_are_rejected():
    service = GroundingService()
    document = _document()
    evidence = service.build_evidence([document])
    draft = GroundedDraft(
        coverage="full",
        answer=(
            "A candidate must attend at least 75 percent of a course. "
            "All examinations are held in June."
        ),
        claims=[
            GroundedClaim(
                claim="A candidate must attend at least 75 percent of a course.",
                evidence_ids=["ev-not-retrieved"],
            )
        ],
    )

    validation = service.validate(draft, evidence)

    assert validation.valid is False
    assert "not retrieved" in " ".join(validation.claim_errors[0])
    assert any("without claim records" in error for error in validation.errors)


def test_partial_outcome_keeps_only_semantically_supported_claims():
    service = GroundingService()
    document = _document(
        "GPA uses grade points. The evidence does not state that every course has equal weight."
    )
    evidence = service.build_evidence([document])
    evidence_id = service.evidence_id(document)
    draft = GroundedDraft(
        coverage="full",
        answer="GPA uses grade points.\n\nEvery course has equal weight.",
        claims=[
            GroundedClaim(
                claim="GPA uses grade points.",
                evidence_ids=[evidence_id],
            ),
            GroundedClaim(
                claim="Every course has equal weight.",
                evidence_ids=[evidence_id],
            ),
        ],
    )

    outcome = service.finalize(
        draft,
        service.validate(draft, evidence),
        VerificationReport(
            verdicts={
                0: "SUPPORTED",
                1: "NOT_ENOUGH_INFORMATION",
            }
        ),
        refusal="No evidence.",
        repaired=True,
    )

    assert outcome.status == "partial"
    assert outcome.answer == "GPA uses grade points."
    assert outcome.supported_claim_count == 1


def test_stub_answer_is_rebuilt_from_claims_before_validation():
    service = GroundingService()
    document = _document(
        "Each course score is the grade point multiplied by its course weight. "
        "GPA is the total score divided by the total credit weight."
    )
    evidence = service.build_evidence([document])
    evidence_id = service.evidence_id(document)
    draft = GroundedDraft(
        coverage="full",
        answer="The GPA calculation has the following steps:",
        claims=[
            GroundedClaim(
                claim=(
                    "Each course score is the grade point multiplied by its "
                    "course weight."
                ),
                evidence_ids=[evidence_id],
            ),
            GroundedClaim(
                claim=(
                    "GPA is the total score divided by the total credit weight."
                ),
                evidence_ids=[evidence_id],
            ),
        ],
    )

    reconciled = service.reconcile_stub_answer(draft)
    validation = service.validate(reconciled, evidence)

    assert reconciled.answer == (
        "- Each course score is the grade point multiplied by its course weight.\n"
        "- GPA is the total score divided by the total credit weight."
    )
    assert validation.valid is True


def test_reconciliation_does_not_replace_a_detailed_answer():
    service = GroundingService()
    answer = (
        "Each course score is the grade point multiplied by its course weight.\n\n"
        "GPA is the total score divided by the total credit weight."
    )
    draft = GroundedDraft(
        coverage="full",
        answer=answer,
        claims=[
            GroundedClaim(claim=answer.splitlines()[0], evidence_ids=["ev-1"]),
            GroundedClaim(claim=answer.splitlines()[-1], evidence_ids=["ev-1"]),
        ],
    )

    assert service.reconcile_stub_answer(draft) is draft


def test_verifier_requires_every_expected_claim():
    service = GroundingService()
    report = service.parse_verification(
        '{"claims":[{"index":0,"verdict":"SUPPORTED"}]}',
        expected_indexes=[0, 1],
    )

    assert report.verdicts[0] == "SUPPORTED"
    assert report.verdicts[1] == "NOT_ENOUGH_INFORMATION"
    assert report.errors


class _StubGenerationService(GenerationService):
    def __init__(
        self,
        answer_payloads: list[dict],
        verification_reports: list[VerificationReport],
    ) -> None:
        self.grounding = GroundingService()
        self.answer_payloads = list(answer_payloads)
        self.verification_reports = list(verification_reports)
        self.model_preference = "auto"
        self._last_model_execution = None
        self._last_grounding_outcome = None

    def _document_completion(self, *_args, **_kwargs) -> str:
        return json.dumps(self.answer_payloads.pop(0))

    def _semantic_verify(self, *_args, **_kwargs) -> VerificationReport:
        return self.verification_reports.pop(0)


def test_generation_repairs_an_unsupported_claim_before_publishing():
    document = _document(
        "GPA uses grade points. Course weights are used in the calculation."
    )
    evidence_id = GroundingService().evidence_id(document)
    initial = {
        "coverage": "full",
        "answer": "GPA uses grade points. Every course has equal weight.",
        "claims": [
            {
                "claim": "GPA uses grade points.",
                "evidence_ids": [evidence_id],
            },
            {
                "claim": "Every course has equal weight.",
                "evidence_ids": [evidence_id],
            },
        ],
    }
    repaired = {
        "coverage": "partial",
        "answer": "GPA uses grade points.",
        "claims": [
            {
                "claim": "GPA uses grade points.",
                "evidence_ids": [evidence_id],
            }
        ],
    }
    generator = _StubGenerationService(
        [initial, repaired],
        [
            VerificationReport(
                verdicts={0: "SUPPORTED", 1: "NOT_ENOUGH_INFORMATION"}
            ),
            VerificationReport(verdicts={0: "SUPPORTED"}),
        ],
    )

    result = generator.generate_response(
        "How is GPA calculated?",
        "<documents />",
        documents=[document],
    )

    assert result["answer"] == "GPA uses grade points."
    assert result["grounding"]["repaired"] is True
    assert result["grounding"]["status"] == "partial"
    assert result["evidence_ids"] == [evidence_id]


def test_generation_refuses_after_two_malformed_drafts():
    generator = _StubGenerationService(
        [{"unexpected": "shape"}, {"still": "invalid"}],
        [],
    )

    result = generator.generate_response(
        "What is the policy?",
        "<documents />",
        documents=[_document()],
    )

    assert result["answer"] == GenerationService.DOCUMENT_REFUSAL
    assert result["grounding"]["status"] == "refused"
    assert result["evidence_ids"] == []


def test_async_finalization_reconciles_stub_answer():
    document = _document(
        "Each course score is the grade point multiplied by its course weight. "
        "GPA is the total score divided by the total credit weight."
    )
    grounding = GroundingService()
    evidence = grounding.build_evidence([document])
    evidence_id = grounding.evidence_id(document)
    raw = json.dumps(
        {
            "coverage": "full",
            "answer": "The GPA calculation has the following steps:",
            "claims": [
                {
                    "claim": (
                        "Each course score is the grade point multiplied by its "
                        "course weight."
                    ),
                    "evidence_ids": [evidence_id],
                },
                {
                    "claim": (
                        "GPA is the total score divided by the total credit weight."
                    ),
                    "evidence_ids": [evidence_id],
                },
            ],
        }
    )
    generator = object.__new__(GenerationService)
    generator.grounding = grounding

    async def verify(_draft, _evidence, indexes):
        return VerificationReport(
            verdicts={index: "SUPPORTED" for index in indexes}
        )

    generator._semantic_verify_async = verify
    outcome, reasons = asyncio.run(
        generator._finalize_grounded_answer_async(
            raw,
            evidence,
            repaired=False,
            allow_partial=False,
        )
    )

    assert reasons == []
    assert outcome.status == "grounded"
    assert outcome.answer.startswith("- Each course score")
    assert outcome.supported_claim_count == 2
