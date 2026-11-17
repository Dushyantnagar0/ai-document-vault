"""Structured value decisions enforce filtering, not model semantic correctness."""

from typing import Any
from uuid import UUID

import pytest
from pydantic import ValidationError

from app.domain.documents import ProcessingError
from app.services.analysis_evidence import QuoteReference
from app.services.analysis_review import AnalysisReviewOutput, project_analysis_review
from app.services.chat_review import ChatReviewOutput
from app.services.evidence_review import ReviewOutput, ReviewPacket, prepare_review

pytestmark = pytest.mark.unit


def assessment(
    index: int,
    classification: str = "adds_conclusion",
    claims: list[str] | None = None,
    premises: list[str] | None = None,
) -> dict[str, Any]:
    useful = classification == "adds_conclusion"
    return {
        "block_index": index,
        "classification": classification,
        "conclusion_claim_ids": claims
        if claims is not None
        else ([f"i{index + 1}.1"] if useful else []),
        "premise_quote_ids": premises if premises is not None else (["q1"] if useful else []),
    }


def verdict(packet: ReviewPacket, assessments: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "decisions": [
            {
                "claim_id": claim.id,
                "supported": True,
                "quote_ids": ["q1"],
                "redundant_with": [],
            }
            for claim in packet.claims
        ],
        "required_qualifications": [],
        "scope_complete": True,
        "insight_assessments": assessments,
    }


def simple_packet(*insights: str) -> ReviewPacket:
    return prepare_review(
        {
            "summary": [{"text": "Two workers each complete three exports per hour."}],
            "insights": [{"text": text} for text in insights],
        },
        {"q1": QuoteReference(UUID(int=1), "Two workers each complete three exports per hour.")},
    )


def project(packet: ReviewPacket, raw: dict[str, Any]) -> Any:
    return project_analysis_review(packet, AnalysisReviewOutput.model_validate(raw))


def test_supported_trivial_counts_and_ranges_are_removed_without_redundancy() -> None:
    evidence = {
        f"q{number}": QuoteReference(UUID(int=1), f"Valve {number} is red.")
        for number in range(1, 9)
    }
    packet = prepare_review(
        {
            "summary": [{"text": "Valves 1 through 8 are red."}],
            "insights": [
                {"text": "At least 8 valves are red among valves 1 through 8."},
                {
                    "text": (
                        "Valve 1 through valve 8 form a complete red run "
                        "with no gaps in that range."
                    )
                },
            ],
        },
        evidence,
    )
    raw = verdict(packet, [assessment(index, "trivial_repackaging") for index in range(2)])
    for decision in raw["decisions"]:
        decision["quote_ids"] = list(evidence)
    result = project(packet, raw)
    assert result.payload["insights"] == []
    assert result.payload["summary"][0]["text"] == "Valves 1 through 8 are red."
    assert result.dropped_claim_ids == ("i1.1", "i2.1")
    assert result.scope_complete is True


def refund_packet(*, closure_caveat: bool = False) -> ReviewPacket:
    evidence = {
        "q1": QuoteReference(
            UUID(int=1), "Staff inspect the kit within two working days after return."
        ),
        "q2": QuoteReference(
            UUID(int=1),
            "For an on-time, complete return without chargeable damage, the full deposit is "
            "refunded within five working days after inspection.",
        ),
        "q3": QuoteReference(
            UUID(int=1), "Working days are office opening days, excluding announced closures."
        ),
    }
    candidate = (
        "Staff inspect the kit within two working days after return. "
        "For an on-time, complete return without chargeable damage, the full deposit refund "
        "is due within seven working days after return. "
        "Working days exclude announced closures."
    )
    if closure_caveat:
        candidate += " This assumes no extra announced closures interrupt the working-day count."
    return prepare_review(
        {
            "summary": [
                {"text": evidence["q1"].quote},
                {"text": evidence["q2"].quote},
                {"text": evidence["q3"].quote},
            ],
            "insights": [{"text": candidate}],
        },
        evidence,
    )


def refund_verdict(packet: ReviewPacket) -> dict[str, Any]:
    raw = verdict(packet, [assessment(0, claims=["i1.2"], premises=["q1", "q2", "q3"])])
    for decision in raw["decisions"]:
        decision["quote_ids"] = list(packet.catalog)
        if decision["claim_id"] == "i1.1":
            decision["redundant_with"] = ["s1.1"]
        if decision["claim_id"] == "i1.3":
            decision["redundant_with"] = ["s3.1"]
    return raw


def test_qualified_combined_deadline_survives_its_repeated_premises() -> None:
    packet = refund_packet()
    raw = refund_verdict(packet)
    result = project(packet, raw)
    assert len(result.payload["insights"]) == 1
    assert result.payload["insights"][0]["text"] == "".join(
        claim.text for claim in packet.claims if claim.kind == "insights"
    )
    assert "seven working days" in result.payload["insights"][0]["text"]
    assert len(result.payload["insights"][0]["support_quotes"]) == 3
    assert result.scope_complete is True and not result.dropped_claim_ids


def test_useful_classification_does_not_rescue_unsupported_closure_caveat() -> None:
    packet = refund_packet(closure_caveat=True)
    raw = refund_verdict(packet)
    raw["decisions"][-1].update(supported=False, quote_ids=[])
    result = project(packet, raw)
    assert result.payload["insights"] == []
    assert result.dropped_claim_ids == ("i1.1", "i1.2", "i1.3", "i1.4")


def test_accepted_classification_does_not_override_unsupported_conclusion() -> None:
    packet = simple_packet("Combined capacity is six exports per hour.")
    raw = verdict(packet, [assessment(0)])
    raw["decisions"][-1]["supported"] = False
    assert project(packet, raw).payload["insights"] == []


def test_accepted_classification_preserves_copied_identifier_guard() -> None:
    packet = simple_packet("Morgan Vale completes six exports per hour.")
    packet.catalog["q2"] = QuoteReference(UUID(int=2), "Morgan Vale is a worker.")
    raw = verdict(packet, [assessment(0)])
    result = project(packet, raw)
    assert result.payload["insights"] == []
    assert result.dropped_claim_ids == ("i1.1",)
    assert result.scope_complete is False


def test_rejected_insight_restores_necessary_qualification_as_literal_evidence() -> None:
    qualification = "Normal wear does not justify a deposit deduction."
    packet = prepare_review(
        {
            "summary": [{"text": "Staff inspect the returned kit."}],
            "insights": [{"text": qualification}],
        },
        {
            "q1": QuoteReference(UUID(int=1), "Staff inspect the returned kit."),
            "q2": QuoteReference(UUID(int=2), qualification),
        },
    )
    raw = verdict(packet, [assessment(0, "repeats_meaning")])
    raw["decisions"][-1]["quote_ids"] = ["q2"]
    raw["required_qualifications"] = [{"quote_ids": ["q2"], "conveyed_by_claim_ids": ["i1.1"]}]
    result = project(packet, raw)
    assert result.payload["insights"] == []
    assert result.payload["summary"][-1]["text"] == qualification
    assert result.appended_quote_ids == ("q2",) and result.scope_complete is True


def test_no_candidates_requires_and_accepts_empty_assessments() -> None:
    packet = simple_packet()
    raw = verdict(packet, [])
    assert project(packet, raw).payload["insights"] == []
    del raw["insight_assessments"]
    with pytest.raises(ValidationError):
        AnalysisReviewOutput.model_validate(raw)


@pytest.mark.parametrize("assessments", [[], [assessment(1)], [assessment(0), assessment(0)]])
def test_missing_unknown_and_duplicate_assessments_fail_closed(
    assessments: list[dict[str, Any]],
) -> None:
    packet = simple_packet("Combined capacity is six exports per hour.")
    with pytest.raises(ProcessingError) as caught:
        project(packet, verdict(packet, assessments))
    assert caught.value.diagnostics is not None
    assert caught.value.diagnostics.reason == "insight_assessment_identity_mismatch"


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("conclusion_claim_ids", ["unknown"]),
        ("conclusion_claim_ids", ["s1.1"]),
        ("conclusion_claim_ids", ["i1.1", "i1.1"]),
        ("premise_quote_ids", ["unknown"]),
        ("premise_quote_ids", ["q1", "q1"]),
    ],
)
def test_invalid_conclusion_or_premise_selections_fail(field: str, values: list[str]) -> None:
    packet = simple_packet("Combined capacity is six exports per hour.")
    raw = verdict(packet, [assessment(0)])
    raw["insight_assessments"][0][field] = values
    with pytest.raises(ProcessingError):
        project(packet, raw)


def test_conclusion_must_belong_to_assessed_block() -> None:
    packet = simple_packet(
        "Combined capacity is six exports per hour.", "The rate is six per hour."
    )
    raw = verdict(packet, [assessment(0, claims=["i2.1"]), assessment(1, "repeats_meaning")])
    with pytest.raises(ProcessingError):
        project(packet, raw)


def test_premises_must_be_bound_to_conclusion_support() -> None:
    packet = simple_packet("Combined capacity is six exports per hour.")
    packet.catalog["q2"] = QuoteReference(UUID(int=2), "Unrelated source fact.")
    with pytest.raises(ProcessingError) as caught:
        project(packet, verdict(packet, [assessment(0, premises=["q2"])]))
    assert caught.value.diagnostics is not None
    assert caught.value.diagnostics.reason == "unbound_insight_premises"


@pytest.mark.parametrize("useful", [False, True])
@pytest.mark.parametrize("field", ["quote_ids", "redundant_with"])
def test_raw_unknown_ids_fail_even_before_discard_or_overlap_normalization(
    useful: bool, field: str
) -> None:
    packet = simple_packet("Combined capacity is six exports per hour.")
    raw = verdict(packet, [assessment(0, "adds_conclusion" if useful else "trivial_repackaging")])
    raw["decisions"][-1][field] = ["unknown"]
    with pytest.raises(ProcessingError):
        project(packet, raw)


@pytest.mark.parametrize(
    "item",
    [
        assessment(0, claims=[]),
        assessment(0, premises=[]),
    ],
)
def test_usefulness_classification_and_selection_shapes_must_agree(item: dict[str, Any]) -> None:
    packet = simple_packet("Combined capacity is six exports per hour.")
    with pytest.raises(ValidationError):
        AnalysisReviewOutput.model_validate(verdict(packet, [item]))


@pytest.mark.parametrize("classification", ["trivial_repackaging", "repeats_meaning"])
def test_valid_references_cannot_rescue_a_rejected_candidate(classification: str) -> None:
    packet = simple_packet("There are two workers.")
    raw = verdict(packet, [assessment(0, classification, claims=["i1.1"], premises=["q1"])])
    result = project(packet, raw)
    assert result.payload["insights"] == []
    assert result.scope_complete is True


@pytest.mark.parametrize("field", ["conclusion_claim_ids", "premise_quote_ids"])
def test_rejected_candidate_still_validates_its_assessment_references(field: str) -> None:
    packet = simple_packet("There are two workers.")
    raw = verdict(packet, [assessment(0, "trivial_repackaging")])
    raw["insight_assessments"][0][field] = ["unknown"]
    with pytest.raises(ProcessingError):
        project(packet, raw)


def test_analysis_schema_is_isolated_from_base_and_chat_reviews() -> None:
    assert "insight_assessments" in AnalysisReviewOutput.model_fields
    assert "insight_assessments" not in ReviewOutput.model_fields
    assert "insight_assessments" not in ChatReviewOutput.model_fields
