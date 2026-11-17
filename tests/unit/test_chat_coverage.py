"""Structured requested-scope enforcement; model semantic judgments remain fallible."""

from typing import Any
from uuid import UUID

import pytest
from pydantic import ValidationError

from app.domain.conversation import Evidence
from app.domain.documents import ProcessingError
from app.services.analysis_evidence import QuoteReference
from app.services.chat_review import (
    ChatReviewOutput,
    RequiredAspect,
    parse_legacy_chat_review,
    project_chat_review,
    project_legacy_chat_review,
)
from app.services.evidence_review import ReviewPacket, prepare_review
from app.services.grounding import AnswerOutput, validate_answer


def comparison_packet() -> ReviewPacket:
    return prepare_review(
        {
            "answer": [
                {"text": "Written approval replaces the recorded deadline."},
                {"text": "Staff preserve time for inspection before the next collection."},
            ]
        },
        {
            "q1": QuoteReference(
                UUID(int=1),
                "Written approval before the recorded deadline replaces it "
                "with a revised deadline.",
            ),
            "q2": QuoteReference(
                UUID(int=1),
                "After an approved extension, the twenty-four-hour grace period starts "
                "at the revised deadline.",
            ),
            "q3": QuoteReference(
                UUID(int=1),
                "Staff preserve time for inspection before the next collection.",
            ),
            "q4": QuoteReference(UUID(int=1), "Membership includes a monthly newsletter."),
        },
        question="How does approval change the deadline and grace period, and why inspect first?",
    )


def comparison_verdict() -> dict[str, Any]:
    return {
        "required_aspects": [
            {
                "aspect": "Approval's effect on the recorded deadline",
                "quote_ids": ["q1"],
                "conveyed_by_claim_ids": ["a1.1"],
            },
            {
                "aspect": "Revised deadline as the grace-period reference",
                "quote_ids": ["q2"],
                "conveyed_by_claim_ids": [],
            },
            {
                "aspect": "Inspection before the next collection",
                "quote_ids": ["q3"],
                "conveyed_by_claim_ids": ["a2.1"],
            },
        ],
        "decisions": [
            {
                "claim_id": "a1.1",
                "supported": True,
                # Correct source selection alone does not convey q2's consequence.
                "quote_ids": ["q1", "q2"],
                "redundant_with": [],
            },
            {
                "claim_id": "a2.1",
                "supported": True,
                "quote_ids": ["q3"],
                "redundant_with": [],
            },
        ],
        "scope_complete": True,
        "answerability": "answered",
        "relevant_claim_ids": ["a1.1", "a2.1"],
        "unanswered_parts": [],
    }


def test_source_inventory_precedes_draft_verdict_and_is_required() -> None:
    schema = ChatReviewOutput.model_json_schema()
    assert next(iter(schema["properties"])) == "required_aspects"
    raw = comparison_verdict()
    raw.pop("required_aspects")
    with pytest.raises(ValidationError):
        ChatReviewOutput.model_validate(raw)


@pytest.mark.parametrize("already_cited", [False, True])
def test_missing_comparison_consequence_is_restored_even_if_its_source_was_cited(
    already_cited: bool,
) -> None:
    raw = comparison_verdict()
    if not already_cited:
        raw["decisions"][0]["quote_ids"] = ["q1"]
    packet = comparison_packet()
    projected = project_chat_review(packet, ChatReviewOutput.model_validate(raw))
    assert projected.appended_quote_ids == ("q2",)
    assert projected.payload["answer"][-1]["text"] == packet.catalog["q2"].quote
    assert projected.scope_complete
    assert all("newsletter" not in block["text"] for block in projected.payload["answer"])


def test_complete_paraphrase_does_not_append_duplicate_original_passage() -> None:
    original = comparison_packet()
    packet = prepare_review(
        {
            "answer": [
                {
                    "text": "Written approval moves the deadline; "
                    "its new date starts the 24-hour grace period."
                },
                {"text": original.claims[1].text},
            ]
        },
        original.catalog,
        question="How does approval affect timing and inspection?",
    )
    raw = comparison_verdict()
    raw["required_aspects"][1]["conveyed_by_claim_ids"] = ["a1.1"]
    projected = project_chat_review(packet, ChatReviewOutput.model_validate(raw))
    assert projected.appended_quote_ids == ()
    assert len(projected.payload["answer"]) == 2
    assert projected.scope_complete


@pytest.mark.parametrize("mapping", [["unknown"], ["a1.1", "unknown"], ["a2.1"]])
def test_unknown_or_unbound_coverage_cannot_suppress_a_required_literal(mapping: list[str]) -> None:
    raw = comparison_verdict()
    raw["required_aspects"][1]["conveyed_by_claim_ids"] = mapping
    projected = project_chat_review(comparison_packet(), ChatReviewOutput.model_validate(raw))
    assert projected.appended_quote_ids == ("q2",)


def test_dropped_covering_prose_is_supplemented_without_reversing_scope_downgrade() -> None:
    raw = comparison_verdict()
    raw["required_aspects"][1]["conveyed_by_claim_ids"] = ["a1.1"]
    raw["decisions"][0]["supported"] = False
    projected = project_chat_review(comparison_packet(), ChatReviewOutput.model_validate(raw))
    assert "a1.1" in projected.dropped_claim_ids
    assert projected.appended_quote_ids == ("q1", "q2")
    assert not projected.scope_complete


def test_aspect_mapped_only_to_irrelevant_claims_neither_appends_nor_certifies_complete() -> None:
    raw = comparison_verdict()
    raw["relevant_claim_ids"] = ["a2.1"]
    raw["required_aspects"] = [raw["required_aspects"][0], raw["required_aspects"][2]]
    projected = project_chat_review(comparison_packet(), ChatReviewOutput.model_validate(raw))
    assert projected.appended_quote_ids == ()
    assert len(projected.payload["answer"]) == 1
    assert not projected.scope_complete


@pytest.mark.parametrize("answerability", ["answered", "insufficient"])
def test_unknown_aspect_source_fails_even_when_abstaining(answerability: str) -> None:
    raw = comparison_verdict()
    raw["required_aspects"][1]["quote_ids"] = ["unknown"]
    raw["answerability"] = answerability
    with pytest.raises(ProcessingError) as caught:
        project_chat_review(comparison_packet(), ChatReviewOutput.model_validate(raw))
    assert caught.value.code == "EVIDENCE_REVIEW_INVALID"


def test_answered_cannot_have_empty_inventory_or_unsupported_requested_aspect() -> None:
    for aspects in (
        [],
        [{"aspect": "Measured delay", "quote_ids": [], "conveyed_by_claim_ids": []}],
    ):
        raw = comparison_verdict()
        raw["required_aspects"] = aspects
        with pytest.raises(ValidationError):
            ChatReviewOutput.model_validate(raw)


@pytest.mark.parametrize("labels", [[], ["Some other missing component"]])
def test_unsupported_aspect_and_missing_component_labels_must_agree(labels: list[str]) -> None:
    raw = comparison_verdict()
    raw["answerability"] = "partial"
    raw["required_aspects"].append(
        {"aspect": "Measured delay", "quote_ids": [], "conveyed_by_claim_ids": []}
    )
    raw["unanswered_parts"] = labels
    with pytest.raises(ValidationError):
        ChatReviewOutput.model_validate(raw)


def test_unsupported_aspect_remains_partial_despite_positive_scope_boolean() -> None:
    raw = comparison_verdict()
    raw["answerability"] = "partial"
    raw["required_aspects"].append(
        {"aspect": "Measured delay", "quote_ids": [], "conveyed_by_claim_ids": []}
    )
    raw["unanswered_parts"] = [" Measured delay "]
    output = ChatReviewOutput.model_validate(raw)
    projected = project_chat_review(comparison_packet(), output)
    assert output.unanswered_parts == ["Measured delay"]
    assert not projected.scope_complete
    assert all("Measured delay" not in block["text"] for block in projected.payload["answer"])


@pytest.mark.parametrize(
    "change",
    [
        {"aspect": " "},
        {"aspect": "x" * 121},
        {"aspect": "topic\nname"},
        {"aspect": "claim [1]"},
        {"quote_ids": ["q" * 31]},
        {"quote_ids": ["q1", "q2", "q3", "q4"]},
        {"conveyed_by_claim_ids": ["a1.1"] * 9},
        {"quote_ids": [], "conveyed_by_claim_ids": ["a1.1"]},
    ],
)
def test_aspect_fields_remain_bounded_plain_metadata(change: dict[str, Any]) -> None:
    raw = comparison_verdict()["required_aspects"][0]
    with pytest.raises(ValidationError):
        RequiredAspect.model_validate({**raw, **change})


def test_eight_aspects_fit_projection_but_ninth_is_rejected() -> None:
    catalog = {
        f"q{index}": QuoteReference(UUID(int=1), f"Requested fact {index}.")
        for index in range(1, 9)
    }
    packet = prepare_review({"answer": [{"text": "Requested fact 1."}]}, catalog)
    raw = comparison_verdict()
    raw["required_aspects"] = [
        {
            "aspect": f"Requested fact {index}",
            "quote_ids": [f"q{index}"],
            "conveyed_by_claim_ids": ["a1.1"] if index == 1 else [],
        }
        for index in range(1, 9)
    ]
    raw["decisions"] = [{**raw["decisions"][0], "quote_ids": ["q1"]}]
    raw["relevant_claim_ids"] = ["a1.1"]
    projected = project_chat_review(packet, ChatReviewOutput.model_validate(raw))
    assert len(projected.payload["answer"]) == 8
    assert len(projected.appended_quote_ids) == 7
    assert projected.scope_complete
    raw["required_aspects"].append(raw["required_aspects"][0])
    with pytest.raises(ValidationError):
        ChatReviewOutput.model_validate(raw)


def test_citation_window_omission_downgrades_an_otherwise_complete_inventory() -> None:
    catalog = {
        f"q{index}": QuoteReference(UUID(int=1), f"Requested fact {index}.")
        for index in range(1, 10)
    }
    packet = prepare_review(
        {
            "answer": [
                {"text": "The initial eight checks apply."},
                {"text": "The final check applies."},
            ]
        },
        catalog,
    )
    raw = comparison_verdict()
    raw["required_aspects"] = [
        {"aspect": "Final check", "quote_ids": ["q9"], "conveyed_by_claim_ids": ["a2.1"]}
    ]
    raw["decisions"][0]["quote_ids"] = [f"q{index}" for index in range(1, 9)]
    raw["decisions"][1]["quote_ids"] = ["q9"]
    projected = project_chat_review(packet, ChatReviewOutput.model_validate(raw))
    assert projected.scope_complete
    evidence = Evidence(
        UUID(int=1),
        UUID(int=2),
        UUID(int=3),
        "checks.txt",
        (" " * 1300).join(ref.quote for ref in catalog.values()),
        (),
    )
    answer = validate_answer(
        AnswerOutput.model_validate(
            {"evidence_status": "answered", "blocks": projected.payload["answer"]}
        ),
        [evidence],
    )
    assert answer.evidence_status == "partial"
    assert len(answer.citations) == 8
    assert len(answer.blocks) == 1


def test_legacy_review_requires_explicit_parser_and_preserves_historical_projection() -> None:
    raw = comparison_verdict()
    raw.pop("required_aspects")
    raw["required_qualifications"] = []
    with pytest.raises(ValidationError):
        ChatReviewOutput.model_validate(raw)
    legacy = parse_legacy_chat_review(raw, prompt_version="chat-evidence-review-v6")
    historical = project_legacy_chat_review(comparison_packet(), legacy)
    assert historical.scope_complete
    assert historical.appended_quote_ids == ()
    assert len(historical.payload["answer"]) == 2
    with pytest.raises(ValueError):
        parse_legacy_chat_review(raw, prompt_version="chat-evidence-review-v7")
