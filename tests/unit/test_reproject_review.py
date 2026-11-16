"""Offline replay checks bind unchanged saved verdicts to actual rendering."""

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

import pytest

from app.db.models import ChatTurn
from app.domain.conversation import Evidence
from app.services.analysis_evidence import QuoteReference
from app.services.chat_context import answer_quote_catalog
from app.services.chat_review import ChatReviewOutput, compact_chat_packet
from app.services.evidence_review import prepare_review
from scripts.evaluate_review import fingerprint
from scripts.reproject_review import replay_case, restore_packet
from scripts.reproject_service_review import _stored_audit_record


def saved_record() -> dict[str, Any]:
    draft = {"answer": [{"text": "A hold blocks deletion until release."}]}
    packet = prepare_review(
        draft,
        {
            "q1": QuoteReference(UUID(int=1), "A hold blocks deletion until release. "),
            "q2": QuoteReference(UUID(int=1), "Release does not restart the ordinary clock."),
        },
        question="How does a hold affect retention?",
    )
    request = {"content": packet.content, "schema": {}, "system": "Frozen audit"}
    return {
        "draft": draft,
        "request": request,
        "request_sha256": fingerprint(request),
        "review": {
            "required_qualifications": [{"quote_ids": ["q2"], "conveyed_by_claim_ids": ["a1.1"]}],
            "decisions": [
                {
                    "claim_id": "a1.1",
                    "supported": True,
                    "quote_ids": ["q1"],
                    "redundant_with": [],
                }
            ],
            "scope_complete": True,
        },
    }


def test_replay_restores_literal_qualification_despite_false_coverage() -> None:
    evidence = Evidence(
        UUID(int=1),
        UUID(int=2),
        UUID(int=3),
        "retention.docx",
        "A hold blocks deletion until release. Release does not restart the ordinary clock.",
        (),
    )
    result = replay_case(saved_record(), [evidence])
    assert result["paid_calls"] == 0
    assert result["renderer_checked"] is True
    assert "Release does not restart the ordinary clock." in result["answer"]
    assert result["appended_qualifications_retained"] is True
    assert result["all_displayed_support_in_citations"] is True


@pytest.mark.parametrize("tamper", ["request", "draft", "duplicate_quote"])
def test_replay_rejects_unbound_or_ambiguous_saved_inputs(tamper: str) -> None:
    record = deepcopy(saved_record())
    if tamper == "request":
        record["request"]["system"] = "Changed audit"
    elif tamper == "draft":
        record["draft"]["answer"][0]["text"] = "A different claim."
    else:
        content = json.loads(record["request"]["content"])
        content["evidence"].append(content["evidence"][0])
        record["request"]["content"] = json.dumps(content)
        record["request_sha256"] = fingerprint(record["request"])
    with pytest.raises(ValueError):
        restore_packet(record)


def test_projection_only_result_does_not_claim_renderer_evidence() -> None:
    result = replay_case(saved_record())
    assert result["renderer_checked"] is False
    assert "citations" not in result


def test_new_chat_review_replay_abstains_without_reattaching_qualifications() -> None:
    record = saved_record()
    record["review"].update(
        answerability="insufficient", relevant_claim_ids=[], unanswered_parts=[]
    )
    record["request"]["schema"] = ChatReviewOutput.model_json_schema()
    record["request_sha256"] = fingerprint(record["request"])
    result = replay_case(record)
    assert result["relevance_enforced"] is True
    assert result["payload"] == {"answer": []}
    assert result["appended_quote_ids"] == ()
    assert not result["scope_complete"]
    with pytest.raises(ValueError, match="CANNOT_BYPASS_CHAT_RELEVANCE"):
        replay_case(record, force_qualification_quotes=True)


def test_review_replay_restores_optional_planner_context_exactly() -> None:
    record = saved_record()
    content = json.loads(record["request"]["content"])
    content["retrieval_query"] = "What is the ordinary clock after release?"
    record["request"]["content"] = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
    record["request_sha256"] = fingerprint(record["request"])
    packet = restore_packet(record)
    assert json.loads(packet.content) == content


def compact_saved_record() -> dict[str, Any]:
    record = saved_record()
    packet = restore_packet(record)
    packet.catalog["q3"] = QuoteReference(
        UUID(int=2), "A second source preserves Zoë's exact text.\n"
    )
    original = prepare_review(
        record["draft"], packet.catalog, question="How does a hold affect retention?"
    )
    content = json.loads(original.content)
    content["retrieval_query"] = "How does releasing that hold affect the ordinary clock?"
    record["request"]["content"] = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
    record["request_sha256"] = fingerprint(record["request"])
    packet = compact_chat_packet(restore_packet(record))
    record["request"]["content"] = packet.content
    record["request_sha256"] = fingerprint(record["request"])
    return record


def test_compact_replay_roundtrip_restores_exact_original_source_provenance() -> None:
    record = compact_saved_record()
    content = json.loads(record["request"]["content"])
    assert content["source_index"] == {"s1": str(UUID(int=1)), "s2": str(UUID(int=2))}
    packet = restore_packet(record)
    assert json.loads(packet.content) == content
    assert list(packet.catalog) == ["q1", "q2", "q3"]
    assert packet.catalog["q1"].source_id == UUID(int=1)
    assert packet.catalog["q3"] == QuoteReference(
        UUID(int=2), "A second source preserves Zoë's exact text.\n"
    )
    evidence = Evidence(
        UUID(int=1),
        UUID(int=2),
        UUID(int=3),
        "retention.docx",
        "A hold blocks deletion until release. Release does not restart the ordinary clock.",
        (),
    )
    record["review"].update(answerability="answered", relevant_claim_ids=["a1.1"])
    result = replay_case(record, [evidence])
    assert result["relevance_enforced"] is True
    assert result["renderer_checked"] is True
    assert result["citations"]
    assert all(c["source_id"] == str(UUID(int=1)) for c in result["citations"])


@pytest.mark.parametrize(
    "tamper",
    [
        "unknown_alias",
        "raw_uuid_in_compact",
        "invalid_uuid",
        "duplicate_source",
        "extra_binding",
        "index_not_mapping",
        "non_string_source",
    ],
)
def test_compact_replay_rejects_unknown_or_ambiguous_source_bindings(tamper: str) -> None:
    record = compact_saved_record()
    content = json.loads(record["request"]["content"])
    if tamper == "unknown_alias":
        content["evidence"][0]["source_id"] = "s99"
    elif tamper == "raw_uuid_in_compact":
        content["evidence"][0]["source_id"] = str(UUID(int=1))
    elif tamper == "invalid_uuid":
        content["source_index"]["s1"] = "not-a-uuid"
    elif tamper == "duplicate_source":
        content["source_index"]["s2"] = str(UUID(int=1))
    elif tamper == "extra_binding":
        content["source_index"]["s3"] = str(UUID(int=3))
    elif tamper == "index_not_mapping":
        content["source_index"] = []
    else:
        content["source_index"]["s1"] = 1
    record["request"]["content"] = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
    record["request_sha256"] = fingerprint(record["request"])
    with pytest.raises(ValueError, match="SAVED_SOURCE"):
        restore_packet(record)


@pytest.mark.parametrize("status", ["answered", "partial", "insufficient"])
def test_legacy_chat_replay_preserves_frozen_status_without_inventing_missing_parts(
    status: str,
) -> None:
    record = compact_saved_record()
    record["review"].update(
        answerability=status,
        relevant_claim_ids=[] if status == "insufficient" else ["a1.1"],
        scope_complete=status == "answered",
    )
    assert "unanswered_parts" not in record["review"]
    frozen_request = deepcopy(record["request"])
    result = replay_case(record)
    assert result["relevance_enforced"] is True
    assert result["missing_parts_available"] is False
    assert record["request"] == frozen_request
    assert "unanswered_parts" not in record["review"]
    assert bool(result["payload"]["answer"]) == (status != "insufficient")
    if status == "partial":
        assert result["scope_complete"] is False


def test_current_chat_replay_requires_declared_missing_parts_field() -> None:
    record = compact_saved_record()
    record["review"].update(answerability="answered", relevant_claim_ids=["a1.1"])
    record["request"]["schema"] = ChatReviewOutput.model_json_schema()
    record["request_sha256"] = fingerprint(record["request"])
    with pytest.raises(ValueError):
        replay_case(record)


@pytest.mark.parametrize(
    "status,parts",
    [("answered", []), ("partial", ["measured restore duration"]), ("insufficient", [])],
)
def test_current_chat_replay_reports_actual_missing_parts_available(
    status: str, parts: list[str]
) -> None:
    record = compact_saved_record()
    record["review"].update(
        answerability=status,
        relevant_claim_ids=[] if status == "insufficient" else ["a1.1"],
        unanswered_parts=parts,
        scope_complete=status == "answered",
    )
    record["request"]["schema"] = ChatReviewOutput.model_json_schema()
    record["request_sha256"] = fingerprint(record["request"])
    result = replay_case(record)
    assert result["missing_parts_available"] is True


def test_raw_missing_parts_select_current_validation_without_schema_metadata() -> None:
    record = saved_record()
    record["review"].update(
        answerability="partial",
        relevant_claim_ids=["a1.1"],
        unanswered_parts=["a recorded deletion date"],
    )
    result = replay_case(record)
    assert result["missing_parts_available"] is True
    record["review"]["unanswered_parts"] = []
    with pytest.raises(ValueError):
        replay_case(record)


def test_current_partial_replay_renders_missing_request_metadata_without_a_citation() -> None:
    record = saved_record()
    record["request"]["schema"] = {"properties": {"unanswered_parts": {}}}
    record["request_sha256"] = fingerprint(record["request"])
    record["review"].update(
        answerability="partial",
        relevant_claim_ids=["a1.1"],
        unanswered_parts=["a confirmed deletion date"],
        scope_complete=False,
    )
    evidence = Evidence(
        UUID(int=1),
        UUID(int=2),
        UUID(int=3),
        "retention.docx",
        "A hold blocks deletion until release. Release does not restart the ordinary clock.",
        (),
    )
    result = replay_case(record, [evidence])
    assert result["evidence_status"] == "partial"
    assert (
        "Not established by the supplied evidence:\n- a confirmed deletion date" in result["answer"]
    )
    assert all("a confirmed deletion date" not in c["excerpt"] for c in result["citations"])


@pytest.mark.parametrize("complete_scope,expected", [(True, "answered"), (False, "partial")])
def test_relevance_replay_uses_reviewed_answerability_instead_of_draft_partial(
    complete_scope: bool,
    expected: str,
) -> None:
    record = saved_record()
    record["draft_evidence_status"] = "partial"
    record["complete_scope"] = complete_scope
    record["review"].update(
        answerability="answered",
        relevant_claim_ids=["a1.1"],
        unanswered_parts=[],
    )
    evidence = Evidence(
        UUID(int=1),
        UUID(int=2),
        UUID(int=3),
        "retention.docx",
        "A hold blocks deletion until release. Release does not restart the ordinary clock.",
        (),
    )
    result = replay_case(record, [evidence])
    assert result["evidence_status"] == expected


def test_advertised_current_schema_cannot_fall_back_to_source_only_review() -> None:
    record = saved_record()
    record["request"]["schema"] = ChatReviewOutput.model_json_schema()
    record["request_sha256"] = fingerprint(record["request"])
    with pytest.raises(ValueError):
        replay_case(record)


@pytest.mark.parametrize("compact,with_gaps", [(False, False), (True, False), (True, True)])
def test_service_audit_rebuilds_versioned_query_wire_and_recorded_draft_status(
    compact: bool,
    with_gaps: bool,
) -> None:
    record = saved_record()
    question = "How does a hold affect retention?"
    query = "What does release of that hold change?"
    evidence = (
        Evidence(
            UUID(int=1),
            UUID(int=2),
            UUID(int=3),
            "retention.docx",
            "A hold blocks deletion until release. Release does not restart the ordinary clock.",
            (),
        ),
    )
    packet = prepare_review(record["draft"], answer_quote_catalog(evidence), question=question)
    content = json.loads(packet.content)
    content["retrieval_query"] = query
    packet = replace(packet, content=json.dumps(content, ensure_ascii=False, separators=(",", ":")))
    if compact:
        packet = compact_chat_packet(packet)
    review = {**record["review"], "answerability": "answered", "relevant_claim_ids": ["a1.1"]}
    if with_gaps:
        review["unanswered_parts"] = []
    metadata = {
        "claims": [{"block_index": c.block_index, "text": c.text} for c in packet.claims],
        "prompt_version": "chat-evidence-review-v3" if with_gaps else "chat-evidence-review-v1",
        "request_sha256": hashlib.sha256(packet.content.encode()).hexdigest(),
        "review": review,
        "draft_evidence_status": "partial",
    }
    if compact:
        metadata["wire_version"] = "source-index-v1"
    turn = cast(
        ChatTurn,
        SimpleNamespace(
            retrieval_query=query,
            context_snapshot={
                "evidence_review": metadata,
                "payload": {"complete_scope": True},
            },
        ),
    )
    rebuilt = _stored_audit_record(turn, {"question": question}, evidence)
    assert rebuilt["request"]["content"] == packet.content
    assert rebuilt["draft_evidence_status"] == "partial"
    assert ("unanswered_parts" in rebuilt["request"]["schema"].get("properties", {})) == with_gaps
    result = replay_case(rebuilt, evidence)
    assert result["evidence_status"] == "answered"
    assert result["missing_parts_available"] == with_gaps
    del metadata["draft_evidence_status"]
    assert (
        _stored_audit_record(turn, {"question": question}, evidence)["draft_evidence_status"]
        == "answered"
    )
