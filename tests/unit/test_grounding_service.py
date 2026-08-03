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


def test_invalid_evidence_id_is_repaired_when_claim_matches_retrieved_text():
    service = GroundingService()
    document = _document(
        "37.0 Appeal Fee. All appeals shall be accompanied by a "
        "non-refundable appeal fee TZS 15,000.00 or as may be prescribed "
        "by the University Council from time to time."
    )
    evidence = service.build_evidence([document])
    evidence_id = service.evidence_id(document)
    draft = GroundedDraft(
        coverage="full",
        answer=(
            "The appeal fee is TZS 15,000.00 or as may be prescribed by "
            "the University Council from time to time."
        ),
        claims=[
            GroundedClaim(
                claim=(
                    "The appeal fee is TZS 15,000.00 or as may be prescribed "
                    "by the University Council from time to time."
                ),
                evidence_ids=["ev-37c6debab54f6e32"],
            )
        ],
    )

    repaired = service.reconcile_evidence_ids(draft, evidence)
    validation = service.validate(repaired, evidence)

    assert repaired.claims[0].evidence_ids == [evidence_id]
    assert validation.valid is True


def test_invalid_evidence_id_is_not_repaired_when_number_is_absent():
    service = GroundingService()
    document = _document("Appeals must be submitted through the UDOM SR2 account.")
    evidence = service.build_evidence([document])
    draft = GroundedDraft(
        coverage="full",
        answer="The appeal fee is TZS 15,000.00.",
        claims=[
            GroundedClaim(
                claim="The appeal fee is TZS 15,000.00.",
                evidence_ids=["ev-37c6debab54f6e32"],
            )
        ],
    )

    repaired = service.reconcile_evidence_ids(draft, evidence)
    validation = service.validate(repaired, evidence)

    assert repaired.claims[0].evidence_ids == ["ev-37c6debab54f6e32"]
    assert validation.valid is False
    assert "not retrieved" in " ".join(validation.claim_errors[0])


def test_missing_evidence_id_is_repaired_when_claim_matches_retrieved_text():
    service = GroundingService()
    document = _document(
        "37.0 Appeal Fee. All appeals shall be accompanied by a "
        "non-refundable appeal fee TZS 15,000.00."
    )
    evidence = service.build_evidence([document])
    evidence_id = service.evidence_id(document)
    raw = json.dumps(
        {
            "mode": "evidence",
            "query": "what is the fee for appealing for remarking",
            "claims": [
                {
                    "claim": "The appeal fee for remarking is TZS 15,000.00.",
                }
            ],
        }
    )

    draft, error = service.parse_draft(raw)
    assert error is None
    assert draft is not None

    repaired = service.reconcile_evidence_ids(draft, evidence)
    validation = service.validate(repaired, evidence)

    assert repaired.claims[0].evidence_ids == [evidence_id]
    assert validation.valid is True


def test_parse_draft_normalizes_fallback_schema_variants():
    service = GroundingService()
    raw = json.dumps(
        {
            "answer": "The annual fee is TZS 1,500,000.00.",
            "claims": [
                {
                    "claim": "The annual fee is TZS 1,500,000.00.",
                    "evidence": ["ev-1234567890abcdef"],
                },
                {
                    "claim": "Supplementary examinations start on 2026-10-12.",
                    "evidence_id": "ev-68b82f7b90d78d78",
                }
            ],
        }
    )

    draft, error = service.parse_draft(raw)

    assert error is None
    assert draft is not None
    assert draft.coverage == "partial"
    assert draft.claims[0].evidence_ids == ["ev-1234567890abcdef"]
    assert draft.claims[1].evidence_ids == ["ev-68b82f7b90d78d78"]


def test_parse_draft_accepts_json_wrapped_in_markdown_fence():
    service = GroundingService()
    raw = """```json
{
  "coverage": "full",
  "claims": [
    {
      "claim": "Students must display an identity card.",
      "evidence_ids": ["ev-1234567890abcdef"]
    }
  ]
}
```"""

    draft, error = service.parse_draft(raw)

    assert error is None
    assert draft is not None
    assert draft.coverage == "full"
    assert draft.answer == "- Students must display an identity card."


def test_partial_outcome_keeps_only_semantically_supported_claims():
    service = GroundingService()
    document = _document(
        "GPA uses grade points. The evidence does not state that every course has equal weight."
    )
    evidence = service.build_evidence([document])
    evidence_id = service.evidence_id(document)
    draft = GroundedDraft(
        coverage="full",
        answer=(
            "**GPA calculation**\n\n"
            "- GPA uses grade points.\n"
            "- Every course has equal weight."
        ),
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
    assert outcome.answer == "- GPA uses grade points."
    assert outcome.supported_claim_count == 1


def test_partial_ordered_answer_is_renumbered_after_filtering():
    service = GroundingService()
    document = _document(
        "Submit the postponement request through SR2. "
        "The request is forwarded to the relevant approval authority."
    )
    evidence = service.build_evidence([document])
    evidence_id = service.evidence_id(document)
    draft = GroundedDraft(
        coverage="full",
        answer=(
            "6. Pay an unsupported fee.\n"
            "7. Submit the postponement request through SR2.\n"
            "8. The request is forwarded to the relevant approval authority."
        ),
        claims=[
            GroundedClaim(
                claim="Pay an unsupported fee.",
                evidence_ids=[evidence_id],
            ),
            GroundedClaim(
                claim="Submit the postponement request through SR2.",
                evidence_ids=[evidence_id],
            ),
            GroundedClaim(
                claim="The request is forwarded to the relevant approval authority.",
                evidence_ids=[evidence_id],
            ),
        ],
    )

    outcome = service.finalize(
        draft,
        service.validate(draft, evidence),
        VerificationReport(
            verdicts={
                0: "NOT_ENOUGH_INFORMATION",
                1: "SUPPORTED",
                2: "SUPPORTED",
            }
        ),
        refusal="No evidence.",
        repaired=True,
    )

    assert outcome.status == "partial"
    assert outcome.answer == (
        "1. Submit the postponement request through SR2.\n\n"
        "2. The request is forwarded to the relevant approval authority."
    )
    assert outcome.supported_claim_count == 2


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


def test_reconcile_claim_answer_alignment_uses_claims_as_canonical_answer():
    service = GroundingService()
    document = _document(
        "A student submits a postponement request through SR2. "
        "The request must be submitted by the twelfth week."
    )
    evidence_id = service.evidence_id(document)
    evidence = service.build_evidence([document])
    draft = GroundedDraft(
        coverage="full",
        answer=(
            "Submit the request online using SR2. "
            "It must reach the University no later than week twelve."
        ),
        claims=[
            GroundedClaim(
                claim="A student submits a postponement request through SR2.",
                evidence_ids=[evidence_id],
            ),
            GroundedClaim(
                claim="The request must be submitted by the twelfth week.",
                evidence_ids=[evidence_id],
            ),
        ],
    )

    reconciled = service.reconcile_claim_answer_alignment(draft)
    validation = service.validate(reconciled, evidence)

    assert reconciled.answer == (
        "- A student submits a postponement request through SR2.\n"
        "- The request must be submitted by the twelfth week."
    )
    assert validation.valid


def test_verifier_requires_every_expected_claim():
    service = GroundingService()
    report = service.parse_verification(
        '{"claims":[{"index":0,"verdict":"SUPPORTED"}]}',
        expected_indexes=[0, 1],
    )

    assert report.verdicts[0] == "SUPPORTED"
    assert report.verdicts[1] == "NOT_ENOUGH_INFORMATION"
    assert report.errors


def test_verification_payload_compacts_long_evidence():
    service = GroundingService()
    document = _document(" ".join(["evidence"] * 300))
    evidence = service.build_evidence([document])
    evidence_id = service.evidence_id(document)
    draft = _draft("The policy is supported by evidence.", evidence_id)

    payload = service.verification_payload(
        draft,
        evidence,
        [0],
        max_evidence_chars=120,
    )

    text = payload["claims"][0]["evidence"][0]["text"]
    assert len(text) <= 135
    assert text.endswith("...[truncated]")


def test_verification_payload_includes_resolved_question_for_relevance():
    service = GroundingService()
    document = _document("Issued by the Deputy Vice Chancellor.")
    evidence = service.build_evidence([document])
    evidence_id = service.evidence_id(document)
    draft = _draft("Issued by the Deputy Vice Chancellor.", evidence_id)

    payload = service.verification_payload(
        draft,
        evidence,
        [0],
        query="who is the Chancellor of UDOM; provide the person's full name",
    )

    assert payload["question"].startswith("who is the Chancellor")


class _StubGenerationService(GenerationService):
    def __init__(
        self,
        answer_payloads: list[dict],
        verification_reports: list[VerificationReport],
    ) -> None:
        self.grounding = GroundingService()
        self.answer_payloads = list(answer_payloads)
        self.verification_reports = list(verification_reports)
        self.refusal_queries: list[str] = []
        self.model_preference = "auto"
        self._last_model_execution = None
        self._last_grounding_outcome = None

    def _document_completion(self, *_args, **_kwargs) -> str:
        return json.dumps(self.answer_payloads.pop(0))

    def _repair_completion(self, **_kwargs) -> str:
        return json.dumps(self.answer_payloads.pop(0))

    def _semantic_verify(self, *_args, **_kwargs) -> VerificationReport:
        return self.verification_reports.pop(0)

    def generate_document_refusal(self, query, *_args, **_kwargs):
        self.refusal_queries.append(query)
        return "Generated document refusal."


class _RepairUnavailableGenerationService(_StubGenerationService):
    def _repair_completion(self, **kwargs) -> str:
        if kwargs.get("operation") == "document_answer_repair":
            from app.services.generation_resilience import GenerationUnavailableError

            raise GenerationUnavailableError(retry_after=30)
        return super()._repair_completion(**kwargs)


class _CaptureRepairGenerationService(_StubGenerationService):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.repair_kwargs = None

    def _repair_completion(self, **kwargs) -> str:
        self.repair_kwargs = kwargs
        return super()._repair_completion(**kwargs)


def test_generation_returns_verified_partial_without_repair_by_default():
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
    generator = _StubGenerationService(
        [initial],
        [
            VerificationReport(
                verdicts={0: "SUPPORTED", 1: "NOT_ENOUGH_INFORMATION"}
            )
        ],
    )

    result = generator.generate_response(
        "How is GPA calculated?",
        "<documents />",
        documents=[document],
    )

    assert result["answer"] == "GPA uses grade points."
    assert result["grounding"]["repaired"] is False
    assert result["grounding"]["status"] == "partial"
    assert result["evidence_ids"] == [evidence_id]


def test_generation_keeps_matching_claim_when_extra_claim_has_no_evidence_id():
    document = _document(
        "Student UDOM SR access ceases six months after graduation."
    )
    evidence_id = GroundingService().evidence_id(document)
    initial = {
        "coverage": "partial",
        "claims": [
            {
                "claim": "Student UDOM SR access ceases six months after graduation."
            },
            {
                "claim": (
                    "The exact duration depends on the circumstances of the "
                    "student's departure from the University."
                )
            },
        ],
    }
    generator = _StubGenerationService(
        [initial],
        [VerificationReport(verdicts={0: "SUPPORTED"})],
    )

    result = generator.generate_response(
        "How long until student UDOM SR access ceases after finishing study?",
        "<documents />",
        documents=[document],
    )

    assert result["answer"] == (
        "- Student UDOM SR access ceases six months after graduation."
    )
    assert result["grounding"]["status"] == "partial"
    assert result["grounding"]["claim_count"] == 2
    assert result["grounding"]["supported_claim_count"] == 1
    assert result["grounding"]["repaired"] is False
    assert result["evidence_ids"] == [evidence_id]


def test_procedure_partial_answer_runs_repair_before_returning(monkeypatch):
    monkeypatch.setattr(
        "app.core.config.settings.GENERATION_ENABLE_ANSWER_REPAIR",
        True,
    )
    document = _document(
        "A student may postpone a year of study with approval from the relevant authority. "
        "The student must submit a postponement request through SR2. "
        "The student must attach supporting reasons for the postponement."
    )
    evidence_id = GroundingService().evidence_id(document)
    initial = {
        "coverage": "full",
        "answer": (
            "A student may postpone a year of study with approval from the relevant authority.\n\n"
            "The student must collect a paper form from the office."
        ),
        "claims": [
            {
                "claim": "A student may postpone a year of study with approval from the relevant authority.",
                "evidence_ids": [evidence_id],
            },
            {
                "claim": "The student must collect a paper form from the office.",
                "evidence_ids": [evidence_id],
            },
        ],
    }
    repaired = {
        "coverage": "full",
        "answer": (
            "A student may postpone a year of study with approval from the relevant authority.\n\n"
            "The student must submit a postponement request through SR2.\n\n"
            "The student must attach supporting reasons for the postponement."
        ),
        "claims": [
            {
                "claim": "A student may postpone a year of study with approval from the relevant authority.",
                "evidence_ids": [evidence_id],
            },
            {
                "claim": "The student must submit a postponement request through SR2.",
                "evidence_ids": [evidence_id],
            },
            {
                "claim": "The student must attach supporting reasons for the postponement.",
                "evidence_ids": [evidence_id],
            },
        ],
    }
    generator = _CaptureRepairGenerationService(
        [initial, repaired],
        [
            VerificationReport(
                verdicts={0: "SUPPORTED", 1: "NOT_ENOUGH_INFORMATION"}
            ),
            VerificationReport(
                verdicts={0: "SUPPORTED", 1: "SUPPORTED", 2: "SUPPORTED"}
            ),
        ],
    )

    result = generator.generate_response(
        "What are the procedures to postpone a year of study?",
        "<documents />",
        documents=[document],
    )

    assert "submit a postponement request through SR2" in result["answer"]
    assert "attach supporting reasons" in result["answer"]
    assert result["grounding"]["repaired"] is True
    assert result["grounding"]["supported_claim_count"] == 3
    assert generator.repair_kwargs is not None


def test_repair_uses_compact_evidence_not_full_context(monkeypatch):
    monkeypatch.setattr(
        "app.core.config.settings.GENERATION_ENABLE_ANSWER_REPAIR",
        True,
    )
    document = _document("GPA uses grade points.")
    evidence_id = GroundingService().evidence_id(document)
    initial = {
        "coverage": "full",
        "answer": "Every course has equal weight.",
        "claims": [
            {
                "claim": "Every course has equal weight.",
                "evidence_ids": [evidence_id],
            }
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
    generator = _CaptureRepairGenerationService(
        [initial, repaired],
        [
            VerificationReport(verdicts={0: "NOT_ENOUGH_INFORMATION"}),
            VerificationReport(verdicts={0: "SUPPORTED"}),
        ],
    )

    result = generator.generate_response(
        "How is GPA calculated?",
        "<documents>very large context that should not be sent to repair</documents>",
        chat_history=[{"role": "user", "content": "large history should not be sent"}],
        documents=[document],
    )

    assert result["answer"] == "GPA uses grade points."
    assert generator.repair_kwargs["mode"] == "evidence"
    assert "context" not in generator.repair_kwargs
    assert "chat_history" not in generator.repair_kwargs
    assert generator.repair_kwargs["evidence"][evidence_id].text == "GPA uses grade points."


def test_generation_repairs_when_enabled_and_no_claims_are_supported(monkeypatch):
    monkeypatch.setattr(
        "app.core.config.settings.GENERATION_ENABLE_ANSWER_REPAIR",
        True,
    )
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
                verdicts={0: "NOT_ENOUGH_INFORMATION", 1: "NOT_ENOUGH_INFORMATION"}
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


def test_generation_recovers_repair_payload_mislabeled_as_none(monkeypatch):
    monkeypatch.setattr(
        "app.core.config.settings.GENERATION_ENABLE_ANSWER_REPAIR",
        True,
    )
    document = _document(
        "A student must apply for postponement through UDOM SR2 using form UDOM/PGS.F8."
    )
    evidence_id = GroundingService().evidence_id(document)
    initial = {
        "coverage": "full",
        "answer": "A student must apply for postponement through a paper letter.",
        "claims": [
            {
                "claim": "A student must apply for postponement through a paper letter.",
                "evidence_ids": [evidence_id],
            }
        ],
    }
    repaired = {
        "coverage": "none",
        "answer": "A student must apply for postponement through UDOM SR2 using form UDOM/PGS.F8.",
        "claims": [
            {
                "claim": "A student must apply for postponement through UDOM SR2 using form UDOM/PGS.F8.",
                "evidence_ids": [evidence_id],
            }
        ],
    }
    generator = _StubGenerationService(
        [initial, repaired],
        [
            VerificationReport(verdicts={0: "NOT_ENOUGH_INFORMATION"}),
            VerificationReport(verdicts={0: "SUPPORTED"}),
        ],
    )

    result = generator.generate_response(
        "How do I postpone a year of study?",
        "<documents />",
        documents=[document],
    )

    assert result["answer"] == repaired["answer"]
    assert result["grounding"]["status"] == "partial"
    assert result["grounding"]["repaired"] is True
    assert result["evidence_ids"] == [evidence_id]


def test_generation_skips_repair_when_disabled_and_no_claims_are_supported(monkeypatch):
    monkeypatch.setattr(
        "app.core.config.settings.GENERATION_ENABLE_ANSWER_REPAIR",
        False,
    )
    document = _document("GPA uses grade points.")
    evidence_id = GroundingService().evidence_id(document)
    initial = {
        "coverage": "full",
        "answer": "Every course has equal weight.",
        "claims": [
            {
                "claim": "Every course has equal weight.",
                "evidence_ids": [evidence_id],
            }
        ],
    }
    repair_payload = {
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
        [initial, repair_payload],
        [VerificationReport(verdicts={0: "NOT_ENOUGH_INFORMATION"})],
    )

    result = generator.generate_response(
        "How is GPA calculated?",
        "<documents />",
        documents=[document],
    )

    assert result["answer"] == "Generated document refusal."
    assert result["grounding"]["repaired"] is False
    assert result["grounding"]["status"] == "refused"
    assert generator.answer_payloads == [repair_payload]
    assert generator.refusal_queries == ["How is GPA calculated?"]


def test_generation_returns_verified_partial_when_repair_is_unavailable():
    document = _document(
        "A student may postpone a year of study with approval from the relevant authority."
    )
    evidence_id = GroundingService().evidence_id(document)
    initial = {
        "coverage": "full",
        "answer": (
            "A student may postpone a year of study with approval from the relevant authority.\n\n"
            "The form must be submitted within seven days."
        ),
        "claims": [
            {
                "claim": "A student may postpone a year of study with approval from the relevant authority.",
                "evidence_ids": [evidence_id],
            },
            {
                "claim": "The form must be submitted within seven days.",
                "evidence_ids": [evidence_id],
            },
        ],
    }
    generator = _RepairUnavailableGenerationService(
        [initial],
        [
            VerificationReport(
                verdicts={
                    0: "SUPPORTED",
                    1: "NOT_ENOUGH_INFORMATION",
                }
            )
        ],
    )

    result = generator.generate_response(
        "How do I postpone a year of study?",
        "<documents />",
        documents=[document],
    )

    assert result["answer"] == (
        "1. A student may postpone a year of study with approval from the relevant authority."
    )
    assert result["grounding"]["status"] == "partial"
    assert result["grounding"]["repaired"] is False
    assert result["evidence_ids"] == [evidence_id]


def test_semantic_verification_batches_claims(monkeypatch):
    monkeypatch.setattr("app.core.config.settings.GENERATION_VERIFICATION_BATCH_SIZE", 2)
    document = _document("A. B. C.")
    evidence_id = GroundingService().evidence_id(document)
    generator = _StubGenerationService([], [])
    calls = []

    def verify(_draft, _evidence, indexes, _query=""):
        calls.append(list(indexes))
        return VerificationReport(
            verdicts={index: "SUPPORTED" for index in indexes}
        )

    generator._semantic_verify = verify
    draft = GroundedDraft(
        coverage="full",
        answer="A.\n\nB.\n\nC.",
        claims=[
            GroundedClaim(claim="A.", evidence_ids=[evidence_id]),
            GroundedClaim(claim="B.", evidence_ids=[evidence_id]),
            GroundedClaim(claim="C.", evidence_ids=[evidence_id]),
        ],
    )

    report = GenerationService._semantic_verify(
        generator,
        draft,
        GroundingService().build_evidence([document]),
        [0, 1, 2],
    )

    assert calls == [[0, 1], [2]]
    assert report.verdicts == {
        0: "SUPPORTED",
        1: "SUPPORTED",
        2: "SUPPORTED",
    }


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

    assert result["answer"] == "Generated document refusal."
    assert result["grounding"]["status"] == "refused"
    assert result["evidence_ids"] == []
    assert generator.refusal_queries == ["What is the policy?"]


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
