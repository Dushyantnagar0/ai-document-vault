"""Projection given explicit audit decisions, not live semantic evaluation."""

import json
from dataclasses import replace
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.domain.documents import ProcessingError
from app.prompts.analysis import generation_prompt_tokens
from app.services.analysis_evidence import QuoteReference
from app.services.chat_review import (
    CHAT_REVIEW_SYSTEM,
    ChatReviewOutput,
    chat_review_input_limit,
    compact_chat_packet,
    project_chat_review,
    review_input_reservation,
)
from app.services.evidence_review import REVIEW_REPAIR, ReviewPacket, prepare_review
from app.services.grounding import INSUFFICIENT_ANSWER, AnswerOutput, validate_answer


def packet(question: str = "Explain Kafka.") -> ReviewPacket:
    return prepare_review(
        {"answer": [{"text": "The primary handles writes."}, {"text": "Replicas can lag."}]},
        {
            "q1": QuoteReference(UUID(int=1), "The primary handles writes."),
            "q2": QuoteReference(UUID(int=2), "Replicas can lag."),
            "q3": QuoteReference(UUID(int=3), "Read routing depends on the freshness policy."),
        },
        question=question,
    )


def verdict(**changes: Any) -> dict[str, Any]:
    return {
        "answerability": "answered",
        "unanswered_parts": ["requested measured lag duration"]
        if changes.get("answerability") == "partial"
        else [],
        "relevant_claim_ids": ["a1.1", "a2.1"],
        "scope_complete": True,
        "required_qualifications": [],
        "decisions": [
            {
                "claim_id": "a1.1",
                "supported": True,
                "quote_ids": ["q1"],
                "redundant_with": [],
            },
            {
                "claim_id": "a2.1",
                "supported": True,
                "quote_ids": ["q2"],
                "redundant_with": [],
            },
        ],
        **changes,
    }


@pytest.mark.parametrize("missing", ["answerability", "relevant_claim_ids", "unanswered_parts"])
def test_chat_relevance_judgments_are_required_not_defaulted(missing: str) -> None:
    raw = verdict()
    raw.pop(missing)
    with pytest.raises(ValidationError):
        ChatReviewOutput.model_validate(raw)


@pytest.mark.parametrize("question", ["Explain Kafka.", "What is tomorrow's weather?"])
def test_valid_citations_do_not_rescue_unanswerable_unrelated_background(question: str) -> None:
    raw = verdict(
        answerability="insufficient",
        relevant_claim_ids=[],
        required_qualifications=[{"quote_ids": ["q3"], "conveyed_by_claim_ids": []}],
    )
    result = project_chat_review(packet(question), ChatReviewOutput.model_validate(raw))
    assert result.payload == {"answer": []}
    assert result.appended_quote_ids == ()
    assert result.dropped_claim_ids == ("a1.1", "a2.1")
    rendered = validate_answer(
        AnswerOutput.model_validate(
            {"evidence_status": "insufficient", "blocks": result.payload["answer"]}
        ),
        [],
    )
    assert rendered.answer == INSUFFICIENT_ANSWER
    assert rendered.citations == []


def test_empty_relevance_is_insufficient_even_if_auditor_claims_answered() -> None:
    result = project_chat_review(
        packet(), ChatReviewOutput.model_validate(verdict(relevant_claim_ids=[]))
    )
    assert result.payload == {"answer": []}


def test_genuine_partial_keeps_only_relevant_audited_prose() -> None:
    result = project_chat_review(
        packet("Explain replica lag and name its measured duration."),
        ChatReviewOutput.model_validate(
            verdict(answerability="partial", relevant_claim_ids=["a2.1"])
        ),
    )
    assert [b["text"] for b in result.payload["answer"]] == ["Replicas can lag."]
    assert result.dropped_claim_ids == ("a1.1",)
    assert not result.scope_complete


@pytest.mark.parametrize(
    "question", ["How do standby copies behave?", "Will those always show the latest writes?"]
)
def test_audited_paraphrases_and_followups_need_no_keyword_match(question: str) -> None:
    result = project_chat_review(packet(question), ChatReviewOutput.model_validate(verdict()))
    assert len(result.payload["answer"]) == 2
    assert result.scope_complete


def test_duplicate_relevance_members_add_no_prose_or_evidence() -> None:
    output = ChatReviewOutput.model_validate(verdict(relevant_claim_ids=["a1.1", "a1.1"]))
    result = project_chat_review(packet(), output)
    assert [b["text"] for b in result.payload["answer"]] == ["The primary handles writes."]


def test_unknown_relevance_claim_is_rejected_safely() -> None:
    with pytest.raises(ProcessingError) as failure:
        project_chat_review(
            packet(), ChatReviewOutput.model_validate(verdict(relevant_claim_ids=["secret-canary"]))
        )
    assert failure.value.code == "EVIDENCE_REVIEW_INVALID"
    assert "secret-canary" not in str(failure.value)


@pytest.mark.parametrize("selection", ["support", "qualification", "claim"])
def test_abstention_does_not_bypass_shared_source_identity_checks(selection: str) -> None:
    raw = verdict(answerability="insufficient", relevant_claim_ids=[])
    if selection == "support":
        raw["decisions"][0]["quote_ids"] = ["unknown"]
    elif selection == "qualification":
        raw["required_qualifications"] = [{"quote_ids": ["unknown"], "conveyed_by_claim_ids": []}]
    else:
        raw["decisions"][0]["claim_id"] = "unknown"
    with pytest.raises(ProcessingError) as failure:
        project_chat_review(packet(), ChatReviewOutput.model_validate(raw))
    assert failure.value.code == "EVIDENCE_REVIEW_INVALID"


def test_qualification_covering_only_irrelevant_claims_is_not_reattached() -> None:
    raw = verdict(
        answerability="partial",
        relevant_claim_ids=["a2.1"],
        required_qualifications=[{"quote_ids": ["q1"], "conveyed_by_claim_ids": ["a1.1"]}],
    )
    result = project_chat_review(packet(), ChatReviewOutput.model_validate(raw))
    assert [b["text"] for b in result.payload["answer"]] == ["Replicas can lag."]
    assert result.appended_quote_ids == ()


def test_unconveyed_qualification_still_restores_original_caveat() -> None:
    raw = verdict(
        answerability="partial",
        relevant_claim_ids=["a2.1"],
        required_qualifications=[{"quote_ids": ["q3"], "conveyed_by_claim_ids": []}],
    )
    result = project_chat_review(packet(), ChatReviewOutput.model_validate(raw))
    assert [b["text"] for b in result.payload["answer"]] == [
        "Replicas can lag.",
        "Read routing depends on the freshness policy.",
    ]
    assert result.appended_quote_ids == ("q3",)


def test_no_supported_relevant_prose_cannot_be_rescued_by_qualification_excerpts() -> None:
    raw = verdict(
        relevant_claim_ids=["a2.1"],
        required_qualifications=[{"quote_ids": ["q3"], "conveyed_by_claim_ids": []}],
    )
    raw["decisions"][1]["supported"] = False
    result = project_chat_review(packet(), ChatReviewOutput.model_validate(raw))
    assert result.payload == {"answer": []}
    assert not result.appended_quote_ids


def dense_packet() -> ReviewPacket:
    """Many short extracted passages reproduce UUID-dominated wire overhead."""
    return prepare_review(
        {"answer": [{"text": "Rows are persisted in engine-managed pages."}]},
        {
            f"q{index + 1}": QuoteReference(
                uuid5(NAMESPACE_URL, f"database-source-{index // 40}"),
                f"Passage {index + 1}: rows remain on durable pages; cafés retain Unicode.\n",
            )
            for index in range(230)
        },
        question="Where do records persist?",
    )


def test_compact_packet_saves_dense_catalog_tokens_without_changing_evidence() -> None:
    original = dense_packet()
    content = json.loads(original.content)
    content["retrieval_query"] = "How does durable record storage work?"
    original = replace(original, content=json.dumps(content, ensure_ascii=False))
    compact = compact_chat_packet(original)
    wire = json.loads(compact.content)
    source_index = wire.pop("source_index")
    assert len(source_index) == 6
    for item in wire["evidence"]:
        item["source_id"] = source_index[item["source_id"]]
    assert wire == content
    assert compact.catalog is original.catalog
    assert compact.claims is original.claims
    assert compact.kinds == original.kinds
    assert compact_chat_packet(compact) == compact
    original_tokens = generation_prompt_tokens(
        ChatReviewOutput, CHAT_REVIEW_SYSTEM, original.content
    )
    compact_tokens = generation_prompt_tokens(ChatReviewOutput, CHAT_REVIEW_SYSTEM, compact.content)
    assert compact_tokens < original_tokens - 2000


def test_compact_source_aliases_preserve_shared_source_groups_and_order() -> None:
    original = dense_packet()
    compact = json.loads(compact_chat_packet(original).content)
    evidence = compact["evidence"]
    assert [item["quote_id"] for item in evidence] == list(original.catalog)
    assert [item["text"] for item in evidence] == [r.quote for r in original.catalog.values()]
    assert [evidence[i]["source_id"] for i in (0, 39, 40, 79, 80, 229)] == [
        "s1",
        "s1",
        "s2",
        "s2",
        "s3",
        "s6",
    ]


@pytest.mark.parametrize("change", ["text", "source", "duplicate", "missing", "index"])
def test_compaction_cannot_silently_accept_broken_packet_bindings(change: str) -> None:
    original = packet()
    wire = json.loads(original.content)
    if change == "text":
        wire["evidence"][0]["text"] = "Unrelated replacement."
    elif change == "source":
        wire["evidence"][0]["source_id"] = str(UUID(int=999))
    elif change == "duplicate":
        wire["evidence"].append(dict(wire["evidence"][0]))
    elif change == "missing":
        wire["evidence"].pop()
    else:
        wire["source_index"] = []
    with pytest.raises(ProcessingError):
        compact_chat_packet(replace(original, content=json.dumps(wire)))


@pytest.mark.parametrize(
    "updates,expected",
    [
        ({}, 14224),
        ({"chat_prompt_max_tokens": 6000, "chat_max_output_tokens": 300}, 7324),
        ({"CHAT_MODEL_MAX_INPUT_TOKENS": 1000}, 1000),
        ({"CHAT_CONTEXT_TOKENS": 6000, "evidence_review_max_output_tokens": 4096}, 1904),
        ({"CHAT_CONTEXT_TOKENS": 1000, "evidence_review_max_output_tokens": 4096}, 0),
    ],
)
def test_review_input_limit_obeys_all_caps(updates: dict[str, Any], expected: int) -> None:
    settings = Settings(_env_file=None, openai_api_key="unit-placeholder").model_copy(
        update=updates
    )
    assert chat_review_input_limit(settings) == expected


def test_reservation_includes_full_catalog_sixty_four_claim_frames_and_repair() -> None:
    settings = Settings(_env_file=None, openai_api_key="unit-placeholder")
    original = dense_packet()
    # A sentence-heavy draft exercises maximum claim metadata, while the exact
    # draft remains within the configured visible-text budget.
    draft = {"answer": [{"text": "A durable record remains saved. " * 64}]}
    actual = compact_chat_packet(prepare_review(draft, original.catalog, question="Where is it?"))
    wire = json.loads(actual.content)
    wire["retrieval_query"] = "Where do the records persist after a restart?"
    actual_tokens = generation_prompt_tokens(
        ChatReviewOutput, CHAT_REVIEW_SYSTEM + REVIEW_REPAIR, json.dumps(wire, ensure_ascii=False)
    )
    reserved = review_input_reservation(
        original.catalog, "Where is it?", wire["retrieval_query"], settings
    )
    assert len(actual.claims) == 64
    assert actual_tokens <= reserved
    assert review_input_reservation({}, "Where is it?", None, settings) < reserved
    larger = settings.model_copy(update={"chat_max_output_tokens": 1800})
    assert (
        review_input_reservation(original.catalog, "Where is it?", wire["retrieval_query"], larger)
        > reserved
    )


def test_reservation_includes_question_query_and_encoding_room() -> None:
    settings = Settings(_env_file=None, openai_api_key="unit-placeholder")
    catalog = packet().catalog
    minimal = review_input_reservation(catalog, "Where?", None, settings)
    context = review_input_reservation(
        catalog, "Where is the replicated record? " * 40, "Replica details " * 80, settings
    )
    assert context > minimal
    small_output = settings.model_copy(update={"chat_max_output_tokens": 1})
    assert review_input_reservation(catalog, "Where?", None, small_output) < minimal


@pytest.mark.parametrize(
    "answerability,parts",
    [("partial", []), ("answered", ["measured percentage speedup"])],
)
def test_answerability_and_missing_requested_parts_must_agree(
    answerability: str, parts: list[str]
) -> None:
    with pytest.raises(ValidationError):
        ChatReviewOutput.model_validate(
            verdict(answerability=answerability, unanswered_parts=parts)
        )


@pytest.mark.parametrize(
    "parts",
    [["   "], ["x" * 241], ["topic\nname"], ["unsupported value [1]"], ["a", "b", "c", "d", "e"]],
)
def test_missing_parts_are_bounded_plain_labels(parts: list[str]) -> None:
    with pytest.raises(ValidationError):
        ChatReviewOutput.model_validate(verdict(answerability="partial", unanswered_parts=parts))


def test_missing_request_metadata_survives_without_publishing_an_uncited_source_claim() -> None:
    output = ChatReviewOutput.model_validate(
        verdict(
            answerability="partial",
            relevant_claim_ids=["a2.1"],
            unanswered_parts=[
                "  exact measured percentage speedup  ",
                "exact measured percentage speedup",
            ],
        )
    )
    assert output.unanswered_parts == ["exact measured percentage speedup"]
    projection = project_chat_review(packet(), output)
    assert projection.payload["answer"][0]["text"] == "Replicas can lag."
    assert all("speedup" not in block["text"] for block in projection.payload["answer"])
    assert "unanswered_parts" in ChatReviewOutput.model_json_schema()["required"]
