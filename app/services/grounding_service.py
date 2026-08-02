from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import re
from typing import Any, Iterable, Literal

from langchain_core.documents import Document
from pydantic import BaseModel, Field, ValidationError


_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?:\s*%)?")
_MARKDOWN_RE = re.compile(r"[*_`>#|]")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "of",
        "or",
        "shall",
        "that",
        "the",
        "their",
        "this",
        "to",
        "with",
    }
)


class GroundedClaim(BaseModel):
    claim: str = Field(min_length=1, max_length=3000)
    evidence_ids: list[str] = Field(default_factory=list, max_length=8)

    model_config = {"extra": "ignore"}


class GroundedDraft(BaseModel):
    coverage: Literal["full", "partial", "none"]
    answer: str = Field(default="", max_length=16000)
    claims: list[GroundedClaim] = Field(default_factory=list, max_length=30)

    model_config = {"extra": "ignore"}


@dataclass(frozen=True)
class EvidenceChunk:
    evidence_id: str
    text: str
    document: Document


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    claim_errors: dict[int, list[str]] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return not self.errors and not self.claim_errors

    def valid_claim_indexes(self, claim_count: int) -> list[int]:
        return [
            index
            for index in range(claim_count)
            if index not in self.claim_errors
        ]

    def messages(self) -> list[str]:
        messages = list(self.errors)
        for index in sorted(self.claim_errors):
            messages.extend(
                f"claim {index}: {message}"
                for message in self.claim_errors[index]
            )
        return messages


@dataclass
class VerificationReport:
    verdicts: dict[int, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def supported_indexes(self) -> set[int]:
        return {
            index
            for index, verdict in self.verdicts.items()
            if verdict == "SUPPORTED"
        }


@dataclass(frozen=True)
class GroundingOutcome:
    answer: str
    coverage: Literal["full", "partial", "none"]
    evidence_ids: tuple[str, ...]
    claim_count: int
    supported_claim_count: int
    repaired: bool
    status: Literal["grounded", "partial", "refused"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "coverage": self.coverage,
            "evidence_ids": list(self.evidence_ids),
            "claim_count": self.claim_count,
            "supported_claim_count": self.supported_claim_count,
            "repaired": self.repaired,
            "status": self.status,
        }


class GroundingService:
    """Parses and validates claim-level evidence for document answers."""

    def evidence_id(self, document: Document) -> str:
        metadata = document.metadata or {}
        chunk_id = metadata.get("chunk_id")
        if chunk_id:
            raw_identity = f"chunk:{chunk_id}"
        else:
            document_id = metadata.get("document_id")
            chunk_index = metadata.get("chunk_index")
            if document_id is not None and chunk_index is not None:
                raw_identity = f"document:{document_id}:chunk:{chunk_index}"
            else:
                raw_identity = f"content:{document.page_content[:240]}"
        digest = sha256(raw_identity.encode("utf-8", errors="ignore")).hexdigest()[:16]
        return f"ev-{digest}"

    def build_evidence(self, documents: Iterable[Document]) -> dict[str, EvidenceChunk]:
        evidence: dict[str, EvidenceChunk] = {}
        for document in documents:
            evidence_id = self.evidence_id(document)
            evidence.setdefault(
                evidence_id,
                EvidenceChunk(
                    evidence_id=evidence_id,
                    text=document.page_content.strip(),
                    document=document,
                ),
            )
        return evidence

    def filter_documents(
        self,
        documents: Iterable[Document],
        evidence_ids: Iterable[str],
    ) -> list[Document]:
        allowed = set(evidence_ids)
        return [
            document
            for document in documents
            if self.evidence_id(document) in allowed
        ]

    def parse_draft(self, raw: str) -> tuple[GroundedDraft | None, str | None]:
        payload = self._extract_json_object(raw)
        if payload is None:
            return None, "The model did not return a JSON object."
        payload = self._normalize_draft_payload(payload)
        try:
            return GroundedDraft.model_validate(payload), None
        except ValidationError as exc:
            return None, f"The grounded answer schema was invalid: {exc.errors(include_url=False)}"

    def _normalize_draft_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload)
        claims = normalized.get("claims")

        if "coverage" not in normalized:
            normalized["coverage"] = "partial" if claims else "none"

        if isinstance(claims, list):
            normalized_claims = []
            for claim in claims:
                if not isinstance(claim, dict):
                    normalized_claims.append(claim)
                    continue

                normalized_claim = dict(claim)
                if "evidence_ids" not in normalized_claim and "evidence" in normalized_claim:
                    normalized_claim["evidence_ids"] = normalized_claim.get("evidence")
                if "evidence_ids" not in normalized_claim and "evidence_id" in normalized_claim:
                    normalized_claim["evidence_ids"] = [normalized_claim.get("evidence_id")]
                if "evidence_ids" not in normalized_claim:
                    normalized_claim["evidence_ids"] = []
                normalized_claims.append(normalized_claim)
            normalized["claims"] = normalized_claims

            if not str(normalized.get("answer") or "").strip():
                claim_texts = [
                    str(claim.get("claim") or "").strip()
                    for claim in normalized_claims
                    if isinstance(claim, dict)
                    and str(claim.get("claim") or "").strip()
                ]
                normalized["answer"] = "\n".join(
                    f"- {claim_text}" for claim_text in claim_texts
                )

        return normalized

    @staticmethod
    def render_claim_answer(
        draft: GroundedDraft,
        *,
        ordered: bool = False,
    ) -> GroundedDraft:
        """Render the user-facing answer from the canonical claim sequence."""
        if draft.coverage == "none" or not draft.claims:
            return draft
        lines = []
        for index, claim in enumerate(draft.claims, start=1):
            prefix = f"{index}." if ordered else "-"
            lines.append(f"{prefix} {claim.claim.strip()}")
        return draft.model_copy(update={"answer": "\n".join(lines)})

    def reconcile_stub_answer(self, draft: GroundedDraft) -> GroundedDraft:
        """Recover details placed in claims when a model returns only an answer stub."""
        if draft.coverage == "none" or len(draft.claims) < 2:
            return draft

        answer_units = self._answer_units(draft.answer)
        if len(answer_units) > 1:
            return draft

        has_claim_content = any(
            self._unit_covered(unit, claim.claim)
            for unit in answer_units
            for claim in draft.claims
        )
        if has_claim_content:
            return draft

        rebuilt_answer = "\n".join(
            f"- {claim.claim.strip()}"
            for claim in draft.claims
            if claim.claim.strip()
        )
        return draft.model_copy(update={"answer": rebuilt_answer})

    def reconcile_claim_answer_alignment(self, draft: GroundedDraft) -> GroundedDraft:
        """Use claim text as the canonical answer when model wording drifts."""
        if draft.coverage == "none" or not draft.claims:
            return draft

        answer_units = self._answer_units(draft.answer)
        normalized_answer = self._normalize_text(draft.answer)
        claims_missing_from_answer = any(
            self._normalize_text(claim.claim) not in normalized_answer
            for claim in draft.claims
            if self._normalize_text(claim.claim)
        )
        answer_has_unclaimed_units = any(
            not any(self._unit_covered(unit, claim.claim) for claim in draft.claims)
            for unit in answer_units
        )
        if not claims_missing_from_answer and not answer_has_unclaimed_units:
            return draft

        rebuilt_answer = "\n".join(
            f"- {claim.claim.strip()}"
            for claim in draft.claims
            if claim.claim.strip()
        )
        return draft.model_copy(update={"answer": rebuilt_answer})

    def reconcile_evidence_ids(
        self,
        draft: GroundedDraft,
        evidence: dict[str, EvidenceChunk],
    ) -> GroundedDraft:
        """Replace invalid model-cited IDs when a retrieved chunk clearly supports the claim."""
        if draft.coverage == "none" or not draft.claims or not evidence:
            return draft

        changed = False
        claims: list[GroundedClaim] = []
        for claim in draft.claims:
            unique_ids = list(dict.fromkeys(claim.evidence_ids))
            valid_ids = [evidence_id for evidence_id in unique_ids if evidence_id in evidence]
            invalid_ids = [evidence_id for evidence_id in unique_ids if evidence_id not in evidence]
            if unique_ids and not invalid_ids:
                claims.append(claim)
                continue

            repaired_ids = list(valid_ids)
            for evidence_id in self._matching_evidence_ids(claim.claim, evidence):
                if evidence_id not in repaired_ids:
                    repaired_ids.append(evidence_id)
                if len(repaired_ids) >= 8:
                    break

            if repaired_ids:
                claims.append(claim.model_copy(update={"evidence_ids": repaired_ids}))
                changed = True
            else:
                claims.append(claim)

        return draft.model_copy(update={"claims": claims}) if changed else draft

    def validate(
        self,
        draft: GroundedDraft,
        evidence: dict[str, EvidenceChunk],
    ) -> ValidationReport:
        report = ValidationReport()

        if draft.coverage == "none":
            if draft.claims:
                report.errors.append("Coverage 'none' cannot include factual claims.")
            return report

        if not draft.answer.strip():
            report.errors.append("A grounded answer cannot be empty.")
        if not draft.claims:
            report.errors.append("A grounded answer must contain at least one claim.")

        normalized_answer = self._normalize_text(draft.answer)
        for index, claim in enumerate(draft.claims):
            claim_messages: list[str] = []
            unique_ids = list(dict.fromkeys(claim.evidence_ids))
            if not unique_ids:
                claim_messages.append("does not cite evidence IDs")
            if len(unique_ids) != len(claim.evidence_ids):
                claim_messages.append("contains duplicate evidence IDs")

            invalid_ids = [
                evidence_id
                for evidence_id in unique_ids
                if evidence_id not in evidence
            ]
            if invalid_ids:
                claim_messages.append(
                    "references evidence that was not retrieved: "
                    + ", ".join(invalid_ids)
                )

            normalized_claim = self._normalize_text(claim.claim)
            if normalized_claim and normalized_claim not in normalized_answer:
                claim_messages.append("is not an exact factual unit from the answer")

            valid_evidence_text = " ".join(
                evidence[evidence_id].text
                for evidence_id in unique_ids
                if evidence_id in evidence
            )
            unsupported_numbers = sorted(
                self._numbers(claim.claim) - self._numbers(valid_evidence_text)
            )
            if unsupported_numbers:
                claim_messages.append(
                    "contains numbers absent from its evidence: "
                    + ", ".join(unsupported_numbers)
                )

            if claim_messages:
                report.claim_errors[index] = claim_messages

        uncovered_units = [
            unit
            for unit in self._answer_units(draft.answer)
            if not any(self._unit_covered(unit, claim.claim) for claim in draft.claims)
        ]
        if uncovered_units:
            report.errors.append(
                "Answer contains factual units without claim records: "
                + " | ".join(uncovered_units[:4])
            )

        return report

    def verification_payload(
        self,
        draft: GroundedDraft,
        evidence: dict[str, EvidenceChunk],
        claim_indexes: Iterable[int],
        max_evidence_chars: int | None = None,
        query: str | None = None,
    ) -> dict[str, Any]:
        claims: list[dict[str, Any]] = []
        for index in claim_indexes:
            claim = draft.claims[index]
            claims.append(
                {
                    "index": index,
                    "claim": claim.claim,
                    "evidence": [
                        {
                            "evidence_id": evidence_id,
                            "text": self._verification_excerpt(
                                evidence[evidence_id].text,
                                max_evidence_chars,
                            ),
                        }
                        for evidence_id in dict.fromkeys(claim.evidence_ids)
                        if evidence_id in evidence
                    ],
                }
            )
        payload: dict[str, Any] = {"claims": claims}
        if query and query.strip():
            payload["question"] = query.strip()
        return payload

    @staticmethod
    def _verification_excerpt(text: str, max_chars: int | None) -> str:
        clean = " ".join(str(text or "").split())
        if max_chars is None or max_chars <= 0 or len(clean) <= max_chars:
            return clean
        return clean[:max_chars].rsplit(" ", 1)[0] + " ...[truncated]"

    def parse_verification(
        self,
        raw: str,
        expected_indexes: Iterable[int],
    ) -> VerificationReport:
        expected = set(expected_indexes)
        payload = self._extract_json_object(raw)
        if payload is None or not isinstance(payload.get("claims"), list):
            return VerificationReport(
                errors=["The verifier did not return the required JSON object."]
            )

        verdicts: dict[int, str] = {}
        errors: list[str] = []
        for item in payload["claims"]:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            verdict = str(item.get("verdict") or "").strip().upper()
            if not isinstance(index, int) or index not in expected:
                continue
            if verdict not in {"SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFORMATION"}:
                continue
            verdicts.setdefault(index, verdict)

        missing = sorted(expected - set(verdicts))
        if missing:
            errors.append(
                "The verifier omitted claims: "
                + ", ".join(str(index) for index in missing)
            )
            for index in missing:
                verdicts[index] = "NOT_ENOUGH_INFORMATION"

        return VerificationReport(verdicts=verdicts, errors=errors)

    def finalize(
        self,
        draft: GroundedDraft,
        validation: ValidationReport,
        verification: VerificationReport,
        refusal: str,
        repaired: bool,
    ) -> GroundingOutcome:
        structurally_valid = set(validation.valid_claim_indexes(len(draft.claims)))
        supported = structurally_valid & verification.supported_indexes()
        supported_indexes = sorted(supported)
        all_supported = (
            validation.valid
            and len(supported_indexes) == len(draft.claims)
            and bool(draft.claims)
        )

        if all_supported:
            evidence_ids = self._claim_evidence_ids(draft, supported_indexes)
            return GroundingOutcome(
                answer=draft.answer.strip(),
                coverage=draft.coverage,
                evidence_ids=tuple(evidence_ids),
                claim_count=len(draft.claims),
                supported_claim_count=len(supported_indexes),
                repaired=repaired,
                status="grounded" if draft.coverage == "full" else "partial",
            )

        if supported_indexes:
            supported_answer = self._supported_answer_units(
                draft,
                supported_indexes,
            )
            return GroundingOutcome(
                answer=supported_answer,
                coverage="partial",
                evidence_ids=tuple(self._claim_evidence_ids(draft, supported_indexes)),
                claim_count=len(draft.claims),
                supported_claim_count=len(supported_indexes),
                repaired=repaired,
                status="partial",
            )

        return self.refusal_outcome(
            refusal=refusal,
            repaired=repaired,
            claim_count=len(draft.claims),
        )

    def refusal_outcome(
        self,
        refusal: str,
        repaired: bool,
        claim_count: int = 0,
    ) -> GroundingOutcome:
        return GroundingOutcome(
            answer=refusal,
            coverage="none",
            evidence_ids=(),
            claim_count=claim_count,
            supported_claim_count=0,
            repaired=repaired,
            status="refused",
        )

    def _supported_answer_units(
        self,
        draft: GroundedDraft,
        indexes: Iterable[int],
    ) -> str:
        supported_claims = [draft.claims[index].claim.strip() for index in indexes]
        supported_markdown: list[str] = []
        seen_normalized: set[str] = set()

        for unit in self._answer_markdown_units(draft.answer):
            if not any(self._unit_covered(unit["plain"], claim) for claim in supported_claims):
                continue
            normalized = self._normalize_text(unit["plain"])
            if not normalized or normalized in seen_normalized:
                continue
            seen_normalized.add(normalized)
            supported_markdown.append(unit["markdown"])

        if supported_markdown:
            return "\n\n".join(supported_markdown)

        return "\n\n".join(dict.fromkeys(supported_claims))

    def _claim_evidence_ids(
        self,
        draft: GroundedDraft,
        indexes: Iterable[int],
    ) -> list[str]:
        evidence_ids: list[str] = []
        for index in indexes:
            for evidence_id in draft.claims[index].evidence_ids:
                if evidence_id not in evidence_ids:
                    evidence_ids.append(evidence_id)
        return evidence_ids

    def _extract_json_object(self, raw: str) -> dict[str, Any] | None:
        if not raw:
            return None
        decoder = json.JSONDecoder()
        for position, character in enumerate(raw):
            if character != "{":
                continue
            try:
                value, _end = decoder.raw_decode(raw[position:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        return None

    def _answer_units(self, answer: str) -> list[str]:
        return [unit["plain"] for unit in self._answer_markdown_units(answer)]

    def _answer_markdown_units(self, answer: str) -> list[dict[str, str]]:
        units: list[dict[str, str]] = []
        for raw_line in answer.splitlines():
            line = raw_line.strip()
            if not line or re.fullmatch(r"[-:| ]+", line):
                continue

            is_heading = line.startswith("#")
            line = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", line)
            clean = _MARKDOWN_RE.sub("", line).strip()
            words = _WORD_RE.findall(clean)
            if is_heading or (len(words) <= 3 and not _NUMBER_RE.search(clean)):
                continue

            sentences = [
                sentence.strip()
                for sentence in _SENTENCE_BOUNDARY_RE.split(clean)
                if sentence.strip()
            ]
            marker = re.match(r"^\s*(?P<marker>(?:[-*+]|\d+[.)])\s+)", raw_line)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(_WORD_RE.findall(sentence)) >= 3 or _NUMBER_RE.search(sentence):
                    markdown = raw_line.strip()
                    if len(sentences) > 1:
                        markdown = f"{marker.group('marker') if marker else ''}{sentence}"
                    units.append(
                        {
                            "plain": sentence,
                            "markdown": markdown,
                        }
                    )
        return units

    def _unit_covered(self, unit: str, claim: str) -> bool:
        normalized_unit = self._normalize_text(unit)
        normalized_claim = self._normalize_text(claim)
        if not normalized_unit or not normalized_claim:
            return False
        if normalized_unit in normalized_claim or normalized_claim in normalized_unit:
            return True
        unit_tokens = set(_WORD_RE.findall(normalized_unit))
        claim_tokens = set(_WORD_RE.findall(normalized_claim))
        return bool(unit_tokens) and len(unit_tokens & claim_tokens) / len(unit_tokens) >= 0.8

    def _numbers(self, text: str) -> set[str]:
        numbers: set[str] = set()
        for match in _NUMBER_RE.findall(text or ""):
            raw = match.replace(",", "").replace("%", "").strip()
            try:
                number = float(raw)
            except ValueError:
                continue
            numbers.add(str(int(number)) if number.is_integer() else f"{number:g}")
        return numbers

    def _matching_evidence_ids(
        self,
        claim: str,
        evidence: dict[str, EvidenceChunk],
    ) -> list[str]:
        claim_numbers = self._numbers(claim)
        claim_tokens = self._content_tokens(claim)
        if not claim_tokens and not claim_numbers:
            return []

        matches: list[tuple[float, str]] = []
        for evidence_id, chunk in evidence.items():
            text = chunk.text or ""
            evidence_numbers = self._numbers(text)
            if claim_numbers and not claim_numbers <= evidence_numbers:
                continue

            evidence_tokens = self._content_tokens(text)
            token_overlap = claim_tokens & evidence_tokens
            if claim_tokens:
                overlap_ratio = len(token_overlap) / len(claim_tokens)
                min_overlap = 0.35 if claim_numbers else 0.6
                if overlap_ratio < min_overlap or len(token_overlap) < min(2, len(claim_tokens)):
                    continue
            else:
                overlap_ratio = 1.0

            number_bonus = 0.25 if claim_numbers else 0.0
            matches.append((overlap_ratio + number_bonus, evidence_id))

        matches.sort(reverse=True)
        return [evidence_id for _score, evidence_id in matches[:3]]

    def _content_tokens(self, text: str) -> set[str]:
        return {
            token
            for token in _WORD_RE.findall(self._normalize_text(text))
            if len(token) > 2 and token not in _STOPWORDS
        }

    def _normalize_text(self, text: str) -> str:
        return " ".join(_WORD_RE.findall(_MARKDOWN_RE.sub("", text).casefold()))
