"""Deterministic audit application; synthetic verdicts do not establish AI quality."""

from typing import Any
from uuid import UUID

import pytest

from app.domain.documents import ProcessingError
from app.services.analysis_evidence import QuoteReference
from app.services.evidence_review import (
    RequiredQualification,
    ReviewOutput,
    ReviewPacket,
    prepare_review,
    project_review,
)


def catalog(count: int = 2) -> dict[str, QuoteReference]:
    return {
        f"q{index}": QuoteReference(UUID(int=index), f"Original fact {index}.")
        for index in range(1, count + 1)
    }


def verdict(packet: ReviewPacket) -> dict[str, Any]:
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
    }


def test_internal_aspect_inventory_can_supplement_more_than_four_groups() -> None:
    packet = prepare_review({"answer": [{"text": "Original fact 1."}]}, catalog(8))
    result = project_review(
        packet,
        ReviewOutput.model_validate(verdict(packet)),
        required_qualifications_override=tuple(
            RequiredQualification(quote_ids=[f"q{i}"], conveyed_by_claim_ids=[])
            for i in range(1, 9)
        ),
    )
    assert "Original fact 8." in str(result.payload)
    assert result.scope_complete is True


def test_aspect_inventory_override_cannot_hide_unknown_raw_qualification() -> None:
    packet = prepare_review({"answer": [{"text": "Original fact 1."}]}, catalog())
    raw = verdict(packet)
    raw["required_qualifications"] = [{"quote_ids": ["unknown"], "conveyed_by_claim_ids": []}]
    with pytest.raises(ProcessingError):
        project_review(
            packet, ReviewOutput.model_validate(raw), required_qualifications_override=()
        )


def test_internal_aspect_inventory_is_bounded() -> None:
    packet = prepare_review({"answer": [{"text": "Original fact 1."}]}, catalog(9))
    with pytest.raises(ProcessingError):
        project_review(
            packet,
            ReviewOutput.model_validate(verdict(packet)),
            required_qualifications_override=tuple(
                RequiredQualification(quote_ids=[f"q{i}"], conveyed_by_claim_ids=[])
                for i in range(1, 10)
            ),
        )


def test_claim_identity_and_exact_unicode_prose_survive_unchanged_projection() -> None:
    text = "Zoë owns the file. \nA second fact remains unchanged."
    draft = {"summary": [{"text": text}], "insights": [], "model": "ignored metadata"}
    packet = prepare_review(draft, catalog())
    assert [claim.id for claim in packet.claims] == ["s1.1", "s1.2"]
    assert "".join(claim.text for claim in packet.claims) == text
    assert prepare_review(draft, catalog()).content == packet.content
    result = project_review(packet, ReviewOutput.model_validate(verdict(packet)))
    assert result.payload["summary"][0]["text"] == text
    assert result.scope_complete is True
    assert result.dropped_claim_ids == ()


@pytest.mark.parametrize("failure", ["unknown_claim", "unknown_quote", "duplicate_decision"])
def test_excluded_claims_do_not_bypass_identity_validation(failure: str) -> None:
    packet = prepare_review(
        {"answer": [{"text": "Requested fact."}, {"text": "Unrelated fact."}]}, catalog()
    )
    raw = verdict(packet)
    excluded = "a2.1"
    if failure == "unknown_claim":
        excluded = "unknown-canary"
    elif failure == "unknown_quote":
        raw["decisions"][1]["quote_ids"] = ["unknown-canary"]
    else:
        raw["decisions"].append(dict(raw["decisions"][1]))
    with pytest.raises(ProcessingError) as caught:
        project_review(
            packet, ReviewOutput.model_validate(raw), excluded_claim_ids=frozenset({excluded})
        )
    assert caught.value.code == "EVIDENCE_REVIEW_INVALID"
    assert "unknown-canary" not in str(caught.value)


@pytest.mark.parametrize(
    "kind,reason",
    [
        ("missing", "claim_identity_mismatch"),
        ("duplicate", "claim_identity_mismatch"),
        ("unknown", "claim_identity_mismatch"),
        ("unknown_quote", "invalid_support_selection"),
        ("duplicate_quote", "invalid_support_selection"),
    ],
)
def test_untrusted_identity_or_support_selection_is_rejected_safely(kind: str, reason: str) -> None:
    packet = prepare_review({"summary": [{"text": "First fact. Second fact."}]}, catalog())
    raw = verdict(packet)
    if kind == "missing":
        raw["decisions"].pop()
    elif kind == "duplicate":
        raw["decisions"].append(dict(raw["decisions"][0]))
    elif kind == "unknown":
        raw["decisions"][0]["claim_id"] = "unknown-canary"
    elif kind == "unknown_quote":
        raw["decisions"][0]["quote_ids"] = ["unknown-canary"]
    elif kind == "duplicate_quote":
        raw["decisions"][0]["quote_ids"] = ["q1", "q1"]
    else:
        raw["decisions"][0]["quote_ids"] = []
    with pytest.raises(ProcessingError) as caught:
        project_review(packet, ReviewOutput.model_validate(raw))
    assert caught.value.code == "EVIDENCE_REVIEW_INVALID"
    assert caught.value.diagnostics is not None
    assert caught.value.diagnostics.as_dict()["reason"] == reason
    assert "unknown-canary" not in str(caught.value)


def test_repaired_late_fact_uses_actual_original_source_and_literal_quote() -> None:
    late = "Final escalation contact: Mira Stone."
    evidence = catalog()
    evidence["q2"] = QuoteReference(UUID(int=42), late)
    text = "The final escalation contact is Mira Stone."
    packet = prepare_review(
        {
            "summary": [
                {
                    "text": text,
                    "source_ids": [str(UUID(int=1))],
                    "support_quotes": [
                        {"source_id": str(UUID(int=1)), "quote": "Wrong early fact"}
                    ],
                }
            ],
            "insights": [],
        },
        evidence,
    )
    raw = verdict(packet)
    raw["decisions"][0]["quote_ids"] = ["q2"]
    result = project_review(packet, ReviewOutput.model_validate(raw))
    block = result.payload["summary"][0]
    assert block["text"] == text
    assert block["source_ids"] == [str(UUID(int=42))]
    assert block["support_quotes"] == [{"source_id": str(UUID(int=42)), "quote": late}]
    assert "Wrong early fact" not in str(result.payload)


def test_missing_qualification_is_appended_as_literal_original_text() -> None:
    qualifier = "Releasing a hold does not create a universal new thirty-day clock."
    evidence = catalog()
    evidence["q2"] = QuoteReference(UUID(int=2), qualifier)
    packet = prepare_review(
        {"answer": [{"text": "Ordinary closed requests are kept thirty days."}]}, evidence
    )
    raw = verdict(packet)
    raw["required_qualifications"] = [{"quote_ids": ["q2"], "conveyed_by_claim_ids": []}]
    result = project_review(packet, ReviewOutput.model_validate(raw))
    assert result.payload["answer"][-1] == {
        "text": qualifier,
        "source_ids": [str(UUID(int=2))],
        "support_quotes": [{"source_id": str(UUID(int=2)), "quote": qualifier}],
    }
    assert result.appended_quote_ids == ("q2",)
    assert result.scope_complete is True


def test_useful_inference_can_share_evidence_while_redundant_insight_is_removed() -> None:
    evidence = {
        "q1": QuoteReference(UUID(int=1), "Two workers each complete three exports per hour.")
    }
    useful = "Under those assumptions the combined rate is six exports per hour."
    packet = prepare_review(
        {
            "summary": [{"text": evidence["q1"].quote}],
            "insights": [
                {"text": "There are two workers completing three exports each per hour."},
                {"text": useful},
            ],
        },
        evidence,
    )
    raw = verdict(packet)
    raw["decisions"][1]["redundant_with"] = ["s1.1"]
    result = project_review(packet, ReviewOutput.model_validate(raw))
    assert [block["text"] for block in result.payload["insights"]] == [useful]
    assert result.payload["insights"][0]["support_quotes"][0]["quote"] == evidence["q1"].quote
    assert result.dropped_claim_ids == ("i1.1",)
    assert result.scope_complete is True


def test_union_overflow_splits_supported_claims_when_block_capacity_remains() -> None:
    packet = prepare_review(
        {"summary": [{"text": "First supported fact. Second fact."}]}, catalog(10)
    )
    raw = verdict(packet)
    raw["decisions"][0]["quote_ids"] = [f"q{index}" for index in range(1, 6)]
    raw["decisions"][1]["quote_ids"] = [f"q{index}" for index in range(6, 11)]
    result = project_review(packet, ReviewOutput.model_validate(raw))
    assert result.payload["summary"][0]["text"] == "First supported fact. "
    assert len(result.payload["summary"][0]["support_quotes"]) == 5
    assert result.payload["summary"][1]["text"] == "Second fact."
    assert len(result.payload["summary"][1]["support_quotes"]) == 5
    assert result.dropped_claim_ids == ()
    assert result.scope_complete is True


def test_summary_block_overflow_does_not_silently_claim_complete_scope() -> None:
    packet = prepare_review(
        {"summary": [{"text": f"Fact {index}."} for index in range(17)]}, catalog()
    )
    result = project_review(packet, ReviewOutput.model_validate(verdict(packet)))
    assert len(result.payload["summary"]) == 16
    assert result.dropped_claim_ids == ("s17.1",)
    assert result.scope_complete is False


def test_all_unsupported_summary_is_not_publishable() -> None:
    packet = prepare_review({"summary": [{"text": "Unsupported fact."}], "insights": []}, catalog())
    raw = verdict(packet)
    raw["decisions"][0].update(supported=False, quote_ids=[])
    with pytest.raises(ProcessingError) as caught:
        project_review(packet, ReviewOutput.model_validate(raw))
    assert caught.value.diagnostics is not None
    assert caught.value.diagnostics.as_dict()["reason"] == "empty_summary"


def test_claim_limit_rejects_before_any_review_request_can_be_prepared() -> None:
    draft = {"answer": [{"text": "A fact. " * 65}]}
    with pytest.raises(ProcessingError) as caught:
        prepare_review(draft, catalog())
    assert caught.value.diagnostics is not None
    assert caught.value.diagnostics.as_dict()["reason"] == "claim_limit"


def test_rejected_mixed_insight_cannot_suppress_later_useful_entry() -> None:
    packet = prepare_review(
        {
            "summary": [{"text": "A source fact."}],
            "insights": [
                {"text": "A useful consequence. A source fact."},
                {"text": "A useful consequence."},
            ],
        },
        catalog(),
    )
    raw = verdict(packet)
    by_id = {item["claim_id"]: item for item in raw["decisions"]}
    by_id["i1.2"]["redundant_with"] = ["s1.1"]
    by_id["i2.1"]["redundant_with"] = ["i1.1"]
    result = project_review(packet, ReviewOutput.model_validate(raw))
    assert [block["text"] for block in result.payload["insights"]] == ["A useful consequence."]
    assert result.dropped_claim_ids == ("i1.1", "i1.2")
    assert result.scope_complete is True


def test_copied_condition_identifier_false_approval_is_dropped_without_autoattachment() -> None:
    evidence = {
        "q1": QuoteReference(UUID(int=1), "Condition 39: approval threshold is 139 units."),
        "q2": QuoteReference(UUID(int=2), "Condition 40: approval threshold is 140 units."),
    }
    packet = prepare_review(
        {
            "summary": [
                {
                    "text": (
                        "Approval thresholds are listed. Condition 40 has a threshold of 140 units."
                    )
                }
            ]
        },
        evidence,
    )
    raw = verdict(packet)  # Both claims falsely approved against only q1.
    result = project_review(packet, ReviewOutput.model_validate(raw))
    assert result.payload["summary"][0]["text"] == "Approval thresholds are listed. "
    assert result.payload["summary"][0]["support_quotes"] == [
        {"source_id": str(UUID(int=1)), "quote": evidence["q1"].quote}
    ]
    assert result.dropped_claim_ids == ("s1.2",)
    assert result.scope_complete is False


def test_copied_reference_with_correct_quote_allows_whitespace_and_case_variation() -> None:
    evidence = {"q1": QuoteReference(UUID(int=1), "Condition40: threshold is 140 units.")}
    text = "CONDITION  40 has a threshold of 140 units."
    packet = prepare_review({"summary": [{"text": text}]}, evidence)
    result = project_review(packet, ReviewOutput.model_validate(verdict(packet)))
    assert result.payload["summary"][0]["text"] == text
    assert result.scope_complete is True


def test_copied_proper_name_requires_selected_source_containing_that_name() -> None:
    evidence = {
        "q1": QuoteReference(UUID(int=1), "Owner: Alice Wong."),
        "q2": QuoteReference(UUID(int=2), "Final contact: Mira Stone."),
    }
    packet = prepare_review(
        {"summary": [{"text": "Alice Wong is named owner. Mira Stone is the final contact."}]},
        evidence,
    )
    result = project_review(packet, ReviewOutput.model_validate(verdict(packet)))
    assert result.payload["summary"][0]["text"] == "Alice Wong is named owner. "
    assert result.dropped_claim_ids == ("s1.2",)
    assert result.scope_complete is False


@pytest.mark.parametrize(
    "source,text",
    [
        (
            "Two workers each complete three exports per hour under these planning assumptions.",
            "Under those assumptions, combined capacity is six exports per hour.",
        ),
        (
            "Twelve components are available and each kit needs three, with no other shortages.",
            "Under those assumptions, there are enough components for four kits.",
        ),
    ],
)
def test_copied_identifier_guard_does_not_require_derived_scalar_values_literally(
    source: str, text: str
) -> None:
    packet = prepare_review(
        {"answer": [{"text": text}]}, {"q1": QuoteReference(UUID(int=1), source)}
    )
    result = project_review(packet, ReviewOutput.model_validate(verdict(packet)))
    assert result.payload["answer"][0]["text"] == text
    assert result.dropped_claim_ids == ()
    assert result.scope_complete is True


@pytest.mark.parametrize("text", ["The date is 20 January 2028.", "The date is 2028-01-20."])
def test_date_rendering_variation_is_not_treated_as_a_copied_named_identifier(text: str) -> None:
    evidence = {
        "q1": QuoteReference(UUID(int=1), "The stated date is 2028-01-20."),
        "q2": QuoteReference(UUID(int=2), "The stated date is 20 January 2028."),
    }
    packet = prepare_review({"answer": [{"text": text}]}, evidence)
    result = project_review(packet, ReviewOutput.model_validate(verdict(packet)))
    assert result.payload["answer"][0]["text"] == text
    assert result.scope_complete is True


def test_negative_factual_assertion_cannot_bypass_the_reference_coverage_guard() -> None:
    evidence = {
        "q1": QuoteReference(UUID(int=1), "Condition 39 has a threshold of 139 units."),
        "q2": QuoteReference(UUID(int=2), "Condition 40 has a threshold of 140 units."),
    }
    packet = prepare_review(
        {
            "summary": [
                {"text": "Thresholds are listed. Condition 40 does not have a threshold of 139."}
            ]
        },
        evidence,
    )
    result = project_review(packet, ReviewOutput.model_validate(verdict(packet)))
    assert result.dropped_claim_ids == ("s1.2",)
    assert result.scope_complete is False


def test_unclaimed_coverage_does_not_repeat_an_already_literal_qualification() -> None:
    quote = "Releasing a hold does not create a universal new thirty-day clock."
    packet = prepare_review(
        {"answer": [{"text": quote}]}, {"q1": QuoteReference(UUID(int=1), quote)}
    )
    raw = verdict(packet)
    raw["required_qualifications"] = [{"quote_ids": ["q1"], "conveyed_by_claim_ids": []}]
    result = project_review(packet, ReviewOutput.model_validate(raw))
    assert [block["text"] for block in result.payload["answer"]] == [quote]
    assert result.appended_quote_ids == ()


def test_qualification_in_cited_quote_is_not_assumed_to_be_conveyed_by_prose() -> None:
    quote = "A legal hold blocks deletion. Releasing it does not start a new retention clock."
    packet = prepare_review(
        {"answer": [{"text": "A legal hold blocks deletion."}]},
        {"q1": QuoteReference(UUID(int=1), quote)},
    )
    raw = verdict(packet)
    raw["required_qualifications"] = [{"quote_ids": ["q1"], "conveyed_by_claim_ids": []}]
    result = project_review(packet, ReviewOutput.model_validate(raw))
    assert result.payload["answer"][-1]["text"] == quote
    assert result.appended_quote_ids == ("q1",)


def test_retained_claim_coverage_skips_duplicate_qualification_at_block_cap() -> None:
    quote = "Releasing a hold does not start a universal new thirty-day clock."
    evidence = {"q1": QuoteReference(UUID(int=1), quote)} | {
        f"q{index}": QuoteReference(UUID(int=index), f"Recorded fact {index}.")
        for index in range(2, 9)
    }
    packet = prepare_review(
        {"answer": [{"text": reference.quote} for reference in evidence.values()]}, evidence
    )
    raw = verdict(packet)
    for index, decision in enumerate(raw["decisions"], 1):
        decision["quote_ids"] = [f"q{index}"]
    raw["required_qualifications"] = [{"quote_ids": ["q1"], "conveyed_by_claim_ids": ["a1.1"]}]
    result = project_review(packet, ReviewOutput.model_validate(raw))
    assert len(result.payload["answer"]) == 8
    assert result.payload["answer"][0]["text"] == quote
    assert result.appended_quote_ids == ()
    assert result.scope_complete is True


def test_identifier_guard_dropped_coverage_claim_cannot_suppress_literal_qualification() -> None:
    quote = "Condition 40 applies only after review."
    evidence = {
        "q1": QuoteReference(UUID(int=1), "Condition 39 applies after review."),
        "q2": QuoteReference(UUID(int=2), quote),
    }
    packet = prepare_review(
        {
            "answer": [
                {"text": "Review conditions are listed."},
                {"text": "Condition 40 needs review."},
            ]
        },
        evidence,
    )
    raw = verdict(packet)
    raw["required_qualifications"] = [{"quote_ids": ["q2"], "conveyed_by_claim_ids": ["a2.1"]}]
    result = project_review(packet, ReviewOutput.model_validate(raw))
    assert result.dropped_claim_ids == ("a2.1",)
    assert result.payload["answer"][-1]["text"] == quote
    assert result.appended_quote_ids == ("q2",)
    assert result.scope_complete is False


def test_all_declared_coverage_claims_must_survive_supported_projection() -> None:
    quote = "Releasing a hold does not reset the original retention clock."
    packet = prepare_review(
        {
            "answer": [
                {"text": "Ordinary closed items are retained thirty days."},
                {"text": "Releasing a hold does not reset the original retention clock."},
            ]
        },
        {
            "q1": QuoteReference(UUID(int=1), "Ordinary closed items are retained thirty days."),
            "q2": QuoteReference(UUID(int=2), quote),
        },
    )
    raw = verdict(packet)
    raw["decisions"][1].update(supported=False, quote_ids=[])
    raw["required_qualifications"] = [
        {"quote_ids": ["q2"], "conveyed_by_claim_ids": ["a1.1", "a2.1"]}
    ]
    result = project_review(packet, ReviewOutput.model_validate(raw))
    assert result.payload["answer"][-1]["text"] == quote
    assert result.appended_quote_ids == ("q2",)
    assert result.dropped_claim_ids == ("a2.1",)
    assert result.scope_complete is False


@pytest.mark.parametrize("coverage", [["foreign-claim-canary"], ["a1.1", "foreign-claim-canary"]])
def test_unknown_qualification_coverage_appends_known_literal_passage(
    coverage: list[str],
) -> None:
    packet = prepare_review({"answer": [{"text": "A stated fact."}]}, catalog())
    raw = verdict(packet)
    raw["required_qualifications"] = [{"quote_ids": ["q1"], "conveyed_by_claim_ids": coverage}]
    result = project_review(packet, ReviewOutput.model_validate(raw))
    assert [block["text"] for block in result.payload["answer"]] == [
        "A stated fact.",
        "Original fact 1.",
    ]
    assert result.appended_quote_ids == ("q1",)
    assert "foreign-claim-canary" not in str(result.payload)


def test_duplicate_qualification_quote_ids_append_one_literal_reference() -> None:
    packet = prepare_review({"answer": [{"text": "A stated fact."}]}, catalog())
    raw = verdict(packet)
    raw["required_qualifications"] = [{"quote_ids": ["q2", "q2"], "conveyed_by_claim_ids": []}]
    result = project_review(packet, ReviewOutput.model_validate(raw))
    assert result.appended_quote_ids == ("q2",)
    assert result.payload["answer"][-1]["text"] == "Original fact 2."
    assert result.payload["answer"][-1]["support_quotes"] == [
        {"source_id": str(UUID(int=2)), "quote": "Original fact 2."}
    ]


def test_duplicate_qualification_coverage_ids_preserve_valid_coverage() -> None:
    packet = prepare_review({"answer": [{"text": "A stated fact."}]}, catalog())
    raw = verdict(packet)
    raw["required_qualifications"] = [
        {"quote_ids": ["q1"], "conveyed_by_claim_ids": ["a1.1", "a1.1"]}
    ]
    result = project_review(packet, ReviewOutput.model_validate(raw))
    assert result.payload["answer"][0]["text"] == "A stated fact."
    assert len(result.payload["answer"]) == 1
    assert result.appended_quote_ids == ()


def test_repeated_qualification_groups_append_literal_passage_once() -> None:
    packet = prepare_review({"answer": [{"text": "A stated fact."}]}, catalog())
    raw = verdict(packet)
    raw["required_qualifications"] = [
        {"quote_ids": ["q2"], "conveyed_by_claim_ids": []},
        {"quote_ids": ["q2"], "conveyed_by_claim_ids": []},
    ]
    result = project_review(packet, ReviewOutput.model_validate(raw))
    assert result.appended_quote_ids == ("q2",)
    assert [block["text"] for block in result.payload["answer"]] == [
        "A stated fact.",
        "Original fact 2.",
    ]


@pytest.mark.parametrize("reverse", [False, True])
def test_repeated_qualification_groups_do_not_union_different_coverage(
    reverse: bool,
) -> None:
    packet = prepare_review(
        {"answer": [{"text": "A stated fact."}, {"text": "Another stated fact."}]},
        catalog(),
    )
    raw = verdict(packet)
    raw["decisions"][1]["quote_ids"] = ["q2"]
    groups = [
        {"quote_ids": ["q1", "q2"], "conveyed_by_claim_ids": ["a1.1"]},
        {"quote_ids": ["q1", "q2"], "conveyed_by_claim_ids": ["a2.1"]},
    ]
    raw["required_qualifications"] = list(reversed(groups)) if reverse else groups
    result = project_review(packet, ReviewOutput.model_validate(raw))
    # Each declared coverage omits a passage. Unioning the coverage lists would
    # incorrectly suppress both literal supplements despite neither being complete.
    assert set(result.appended_quote_ids) == {"q1", "q2"}
    assert result.appended_quote_ids == (("q1", "q2") if reverse else ("q2", "q1"))
    assert {block["text"] for block in result.payload["answer"][2:]} == {
        "Original fact 1.",
        "Original fact 2.",
    }


def test_unknown_qualification_quote_id_is_still_rejected() -> None:
    packet = prepare_review({"answer": [{"text": "A stated fact."}]}, catalog())
    raw = verdict(packet)
    raw["required_qualifications"] = [
        {"quote_ids": ["foreign-quote-canary"], "conveyed_by_claim_ids": []}
    ]
    with pytest.raises(ProcessingError) as caught:
        project_review(packet, ReviewOutput.model_validate(raw))
    assert caught.value.diagnostics is not None
    assert caught.value.diagnostics.as_dict()["reason"] == "invalid_qualification_selection"
    assert "foreign-quote-canary" not in str(caught.value)


def test_retained_insight_can_cover_qualification_without_duplicate_append() -> None:
    quote = (
        "Closed items use their original closure date for retention. A hold blocks deletion. "
        "Releasing it does not reset that clock."
    )
    insight = "After a hold is released, retention still follows the original closure date."
    packet = prepare_review(
        {
            "summary": [
                {
                    "text": (
                        "Closed items use their closure date for retention. A hold blocks deletion."
                    )
                }
            ],
            "insights": [{"text": insight}],
        },
        {"q1": QuoteReference(UUID(int=1), quote)},
    )
    raw = verdict(packet)
    raw["required_qualifications"] = [{"quote_ids": ["q1"], "conveyed_by_claim_ids": ["i1.1"]}]
    result = project_review(packet, ReviewOutput.model_validate(raw))
    assert len(result.payload["summary"]) == 1
    assert result.payload["insights"][0]["text"] == insight
    assert result.appended_quote_ids == ()
    assert result.dropped_claim_ids == ()
    assert result.scope_complete is True


def test_forced_qualification_supplement_overrides_false_model_prose_coverage() -> None:
    quote = "A legal hold blocks deletion. Releasing it does not start a new retention clock."
    packet = prepare_review(
        {"answer": [{"text": "A legal hold blocks deletion."}]},
        {"q1": QuoteReference(UUID(int=1), quote)},
    )
    raw = verdict(packet)
    # The model incorrectly claims the first sentence conveys the complete rule.
    raw["required_qualifications"] = [{"quote_ids": ["q1"], "conveyed_by_claim_ids": ["a1.1"]}]
    output = ReviewOutput.model_validate(raw)
    ordinary = project_review(packet, output, force_qualification_quotes=False)
    assert ordinary == project_review(packet, output)
    assert len(ordinary.payload["answer"]) == 1
    assert ordinary.appended_quote_ids == ()
    forced = project_review(packet, output, force_qualification_quotes=True)
    assert forced.payload["answer"][0]["text"] == "A legal hold blocks deletion."
    assert forced.payload["answer"][-1]["text"] == quote
    assert forced.payload["answer"][-1]["support_quotes"] == [
        {"source_id": str(UUID(int=1)), "quote": quote}
    ]
    assert forced.appended_quote_ids == ("q1",)
    assert forced.scope_complete is True


def test_forced_supplement_overflow_marks_incomplete_scope() -> None:
    quote = "Releasing a hold does not start a new retention clock."
    evidence = {"q1": QuoteReference(UUID(int=1), quote)} | {
        f"q{index}": QuoteReference(UUID(int=index), f"Recorded fact {index}.")
        for index in range(2, 9)
    }
    packet = prepare_review(
        {"answer": [{"text": reference.quote} for reference in evidence.values()]}, evidence
    )
    raw = verdict(packet)
    for index, decision in enumerate(raw["decisions"], 1):
        decision["quote_ids"] = [f"q{index}"]
    packet.catalog["q9"] = QuoteReference(UUID(int=9), "A different essential qualification.")
    raw["required_qualifications"] = [{"quote_ids": ["q9"], "conveyed_by_claim_ids": ["a1.1"]}]
    output = ReviewOutput.model_validate(raw)
    # The default policy detects q9 absent from the covering claim binding.
    assert project_review(packet, output).scope_complete is False
    forced = project_review(packet, output, force_qualification_quotes=True)
    assert len(forced.payload["answer"]) == 8
    assert forced.appended_quote_ids == ()
    assert forced.scope_complete is False


def test_forced_supplements_do_not_repeat_literal_text_across_groups() -> None:
    packet = prepare_review({"answer": [{"text": "The ordinary rule applies."}]}, catalog())
    raw = verdict(packet)
    raw["required_qualifications"] = [
        {"quote_ids": ["q1"], "conveyed_by_claim_ids": ["a1.1"]},
        {"quote_ids": ["q1", "q2"], "conveyed_by_claim_ids": ["a1.1"]},
    ]
    result = project_review(
        packet, ReviewOutput.model_validate(raw), force_qualification_quotes=True
    )
    assert [block["text"] for block in result.payload["answer"]] == [
        "The ordinary rule applies.",
        "Original fact 1.",
        "Original fact 2.",
    ]
    assert result.appended_quote_ids == ("q1", "q2")


def test_qualification_coverage_requires_matching_retained_claim_source_bindings() -> None:
    packet = prepare_review(
        {"answer": [{"text": "A hold prevents deletion until release."}]},
        {
            "q1": QuoteReference(UUID(int=1), "A hold prevents deletion until release. "),
            "q2": QuoteReference(UUID(int=1), "Release does not restart the ordinary clock."),
        },
    )
    raw = verdict(packet)
    raw["required_qualifications"] = [
        {"quote_ids": ["q1", "q2"], "conveyed_by_claim_ids": ["a1.1"]}
    ]
    result = project_review(packet, ReviewOutput.model_validate(raw))
    assert [block["text"] for block in result.payload["answer"]] == [
        "A hold prevents deletion until release.",
        "Release does not restart the ordinary clock.",
    ]
    assert result.appended_quote_ids == ("q2",)


def test_complete_qualification_with_matching_claim_binding_needs_no_supplement() -> None:
    packet = prepare_review(
        {
            "answer": [
                {
                    "text": (
                        "A hold prevents deletion until release. Release keeps the original clock."
                    )
                }
            ]
        },
        {
            "q1": QuoteReference(UUID(int=1), "A hold prevents deletion until release. "),
            "q2": QuoteReference(UUID(int=1), "Release does not restart the ordinary clock."),
        },
    )
    raw = verdict(packet)
    raw["decisions"][1]["quote_ids"] = ["q2"]
    raw["required_qualifications"] = [
        {"quote_ids": ["q1", "q2"], "conveyed_by_claim_ids": ["a1.1", "a1.2"]}
    ]
    result = project_review(packet, ReviewOutput.model_validate(raw))
    assert len(result.payload["answer"]) == 1
    assert result.appended_quote_ids == ()


def test_same_passage_false_coverage_remains_a_known_semantic_limitation() -> None:
    # A deliberately false verdict binds an incomplete statement to a passage
    # containing BOTH facts. Structural consistency cannot detect that omission.
    packet = prepare_review(
        {"answer": [{"text": "A hold prevents deletion until release."}]},
        {
            "q1": QuoteReference(
                UUID(int=1),
                "A hold prevents deletion until release; "
                "release does not restart the ordinary clock.",
            )
        },
    )
    raw = verdict(packet)
    raw["required_qualifications"] = [{"quote_ids": ["q1"], "conveyed_by_claim_ids": ["a1.1"]}]
    result = project_review(packet, ReviewOutput.model_validate(raw))
    assert result.appended_quote_ids == ()
    assert "restart" not in result.payload["answer"][0]["text"]


def test_union_split_overflow_still_drops_claim_when_chat_block_cap_is_full() -> None:
    packet = prepare_review(
        {
            "answer": [{"text": f"Recorded fact {n}."} for n in range(7)]
            + [{"text": "First supported fact. Second fact."}]
        },
        catalog(10),
    )
    raw = verdict(packet)
    raw["decisions"][7]["quote_ids"] = [f"q{n}" for n in range(1, 6)]
    raw["decisions"][8]["quote_ids"] = [f"q{n}" for n in range(6, 11)]
    result = project_review(packet, ReviewOutput.model_validate(raw))
    assert len(result.payload["answer"]) == 8
    assert result.dropped_claim_ids == ("a8.2",)
    assert result.scope_complete is False


@pytest.mark.parametrize("phrase", ["data pages", "Data pages"])
def test_source_heading_does_not_turn_lowercase_storage_paraphrase_into_a_name(phrase: str) -> None:
    packet = prepare_review(
        {"answer": [{"text": f"Rows are stored in {phrase}."}]},
        {
            "q1": QuoteReference(UUID(int=1), "Each row is stored in a data page."),
            "q2": QuoteReference(UUID(int=2), "Data Pages\n\nLayout details follow."),
        },
    )
    result = project_review(packet, ReviewOutput.model_validate(verdict(packet)))
    assert result.dropped_claim_ids == ()
    assert result.payload["answer"][0]["text"] == f"Rows are stored in {phrase}."


@pytest.mark.parametrize("name", ["Mira Stone", "MIRA STONE"])
def test_capitalized_copied_name_still_requires_its_selected_original_source(name: str) -> None:
    packet = prepare_review(
        {"answer": [{"text": f"{name} is the final contact."}]},
        {
            "q1": QuoteReference(UUID(int=1), "The source lists contact details."),
            "q2": QuoteReference(UUID(int=2), "Final contact: Mira Stone."),
        },
    )
    result = project_review(packet, ReviewOutput.model_validate(verdict(packet)))
    assert result.dropped_claim_ids == ("a1.1",)
    assert result.scope_complete is False


@pytest.mark.parametrize("source_label", ["condition", "Condition", "CONDITION"])
@pytest.mark.parametrize("claim_label", ["condition", "Condition", "CONDITION"])
@pytest.mark.parametrize("correct_support", [False, True])
def test_numeric_record_reference_checks_source_and_claim_case_independently(
    source_label: str, claim_label: str, correct_support: bool
) -> None:
    text = f"{claim_label} 40 has a threshold of 140 units."
    packet = prepare_review(
        {"answer": [{"text": text}]},
        {
            "q1": QuoteReference(UUID(int=1), f"{source_label} 39 has threshold 139."),
            "q2": QuoteReference(UUID(int=2), f"{source_label} 40 has threshold 140."),
        },
    )
    raw = verdict(packet)
    raw["decisions"][0]["quote_ids"] = ["q2"] if correct_support else ["q1"]
    result = project_review(packet, ReviewOutput.model_validate(raw))
    assert result.dropped_claim_ids == (() if correct_support else ("a1.1",))
    assert result.scope_complete is correct_support
    if correct_support:
        assert result.payload["answer"][0]["text"] == text
        assert result.payload["answer"][0]["support_quotes"][0]["quote"] == (
            f"{source_label} 40 has threshold 140."
        )
    else:
        assert result.payload["answer"] == []


def test_numeric_reference_guard_preserves_unicode_claim_text_and_source_whitespace() -> None:
    text = "Le dossier conserve condition\u00a040 pour Zoë et cafe\u0301."
    quote = "condition\u00a040 applies to the café dossier."
    packet = prepare_review(
        {"answer": [{"text": text}]}, {"q1": QuoteReference(UUID(int=1), quote)}
    )
    result = project_review(packet, ReviewOutput.model_validate(verdict(packet)))
    assert result.dropped_claim_ids == ()
    assert result.scope_complete is True
    assert result.payload["answer"][0]["text"] == text
    assert result.payload["answer"][0]["support_quotes"][0]["quote"] == quote


def test_title_case_verb_heading_does_not_reject_lowercase_ordinary_prose() -> None:
    packet = prepare_review(
        {"answer": [{"text": "The table rows are stored in pages."}]},
        {
            "q1": QuoteReference(UUID(int=1), "Each row of a table is stored in a data page."),
            "q2": QuoteReference(UUID(int=2), "How Rows Are Stored\n\nStorage overview."),
        },
    )
    result = project_review(packet, ReviewOutput.model_validate(verdict(packet)))
    assert result.dropped_claim_ids == ()
    assert result.scope_complete is True


@pytest.mark.parametrize("name", ["Mira Stone", "MIRA STONE"])
def test_capitalized_copied_name_survives_with_correct_case_insensitive_support(name: str) -> None:
    packet = prepare_review(
        {"answer": [{"text": f"{name} is the final contact."}]},
        {"q1": QuoteReference(UUID(int=1), "Final contact: Mira Stone.")},
    )
    result = project_review(packet, ReviewOutput.model_validate(verdict(packet)))
    assert result.dropped_claim_ids == ()
    assert result.scope_complete is True


def test_supported_without_references_is_removed_instead_of_published_or_fatal() -> None:
    packet = prepare_review({"answer": [{"text": "No. Original fact 1."}]}, catalog())
    raw = verdict(packet)
    raw["decisions"][0]["quote_ids"] = []
    result = project_review(packet, ReviewOutput.model_validate(raw))
    assert result.payload["answer"][0]["text"] == "Original fact 1."
    assert result.dropped_claim_ids == ("a1.1",)
    assert not result.scope_complete
