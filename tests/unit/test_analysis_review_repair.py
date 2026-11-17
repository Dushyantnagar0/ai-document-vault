"""Bounded analysis repair with synthetic providers, not model-quality evidence.

Persistent failure returns no publishable analysis. Transaction behavior remains
covered by the ingestion integration suite.
"""

import json
from typing import Any
from uuid import uuid4

import pytest
from test_analysis_quality import DraftGenerator, config

from app.core.config import Settings
from app.domain.documents import AIFailureDetails, Chunk, Generated, ProcessingError
from app.services.analysis import analyse_document
from app.services.analysis_review import (
    ANALYSIS_REVIEW_SYSTEM,
    AnalysisReviewOutput,
    analysis_review_repair_instruction,
)

pytestmark = pytest.mark.unit
INSIGHT = (
    "Combined capacity is six validated exports per hour; exports awaiting validation do not count."
)
RAW_ERROR_CANARY = "private-provider-body-canary-do-not-forward"


def source() -> list[Chunk]:
    return [
        Chunk(
            uuid4(),
            0,
            "Two workers process three exports each per hour. "
            "Only validated exports count toward that hourly capacity. "
            "Operators review alerts after each shift.",
            35,
            (),
        )
    ]


class QualifiedInsights:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> Generated[Any]:
        self.calls.append(kwargs)
        return Generated(
            kwargs["schema"].model_validate(
                {"insights": [{"text": INSIGHT, "quote_ids": ["q1", "q2"]}]}
            ),
            Settings.CHAT_MODEL,
            200,
            200,
        )


class RepairReviewer:
    def __init__(self, first: str = "invalid_identity", second: str = "valid") -> None:
        self.calls: list[dict[str, Any]] = []
        self.first, self.second = first, second

    def generate(self, **kwargs: Any) -> Generated[Any]:
        self.calls.append(kwargs)
        scenario = self.first if len(self.calls) == 1 else self.second
        if scenario in {"reported_identity", "unknown_diagnostic"}:
            raise ProcessingError(
                "EVIDENCE_REVIEW_INVALID",
                RAW_ERROR_CANARY,
                diagnostics=AIFailureDetails(
                    reason="insight_assessment_identity_mismatch"
                    if scenario == "reported_identity"
                    else RAW_ERROR_CANARY,
                    input_tokens=71,
                    output_tokens=13,
                ),
            )
        request = json.loads(kwargs["content"])
        raw = {
            "decisions": [
                {
                    "claim_id": c["claim_id"],
                    "supported": True,
                    "quote_ids": ["q1", "q2"] if c["kind"] == "insights" else ["q1"],
                    "redundant_with": [],
                }
                for c in request["claims"]
            ],
            "required_qualifications": [{"quote_ids": ["q2"], "conveyed_by_claim_ids": ["i1.1"]}],
            "scope_complete": True,
            "insight_assessments": [
                {
                    "block_index": index,
                    "classification": "adds_conclusion",
                    "conclusion_claim_ids": ["i1.1"],
                    "premise_quote_ids": ["q1", "q2", "q3"]
                    if scenario == "unbound_premises"
                    else ["q1", "q2"],
                }
                for index in ([0, 1, 2] if scenario == "invalid_identity" else [0])
            ],
        }
        # A deliberate raw invalid response exercises local schema revalidation.
        # The real adapter can reject it earlier through the same bound schema.
        result = raw if scenario == "invalid_identity" else kwargs["schema"].model_validate(raw)
        return Generated(result, Settings.ANALYSIS_REVIEW_MODEL, 300, 300)


@pytest.mark.parametrize(
    ("first", "reason", "input_tokens", "output_tokens"),
    [
        ("invalid_identity", "schema_validation", 900, 900),
        ("reported_identity", "insight_assessment_identity_mismatch", 671, 613),
    ],
)
def test_identity_failure_repairs_frozen_candidates_and_preserves_qualification(
    first: str, reason: str, input_tokens: int, output_tokens: int
) -> None:
    draft, insights, reviewer = DraftGenerator(), QualifiedInsights(), RepairReviewer(first)
    reservations: list[bool] = []

    def reserve() -> float:
        reservations.append(True)
        return 7

    chunks = source()
    result = analyse_document(
        chunks, config(), draft, reserve, insight_generator=insights, reviewer=reviewer
    )
    assert [len(p.calls) for p in (draft, insights, reviewer)] == [1, 1, 2]
    assert len(reservations) == 4
    initial, repaired = reviewer.calls
    assert initial["content"] == repaired["content"]
    assert json.loads(initial["content"])["insight_candidates"] == [
        {"block_index": 0, "claim_ids": ["i1.1"]}
    ]
    assert issubclass(initial["schema"], AnalysisReviewOutput)
    assert initial["schema"] is repaired["schema"]
    assert initial["system"] == ANALYSIS_REVIEW_SYSTEM
    assert repaired["system"] == ANALYSIS_REVIEW_SYSTEM + analysis_review_repair_instruction(reason)
    assert all(c["timeout_seconds"] == 7 for c in reviewer.calls)
    assert all(c["max_output_tokens"] == 4096 for c in reviewer.calls)
    assert (
        result.payload["summary"][0]["text"] == "Two workers process three exports each per hour."
    )
    assert result.payload["insights"][0]["text"] == INSIGHT
    quotes = result.payload["insights"][0]["support_quotes"]
    assert len(quotes) == 2 and all(q["quote"] in chunks[0].text for q in quotes)
    assert result.payload["analysis_scope_complete"] is True
    assert result.payload["evidence_review"]["dropped_claim_ids"] == []
    assert result.payload["evidence_review"]["appended_quote_ids"] == []
    usage = result.payload["usage"]
    history = usage["call_history"]
    assert [c["stage"] for c in history] == [
        "final",
        "insights",
        "evidence_review",
        "evidence_review",
    ]
    assert history[-2]["outcome"] == "failed"
    assert history[-2]["diagnostics"]["reason"] == reason
    assert history[-2]["repair"] is False
    assert history[-1]["outcome"] == "completed" and history[-1]["repair"] is True
    assert usage["known_input_tokens"] == input_tokens
    assert usage["known_output_tokens"] == output_tokens
    assert RAW_ERROR_CANARY not in repaired["system"] + repaired["content"]
    assert RAW_ERROR_CANARY not in json.dumps(result.payload)


def test_identity_then_unbound_premises_fails_after_one_review_repair() -> None:
    draft, insights = DraftGenerator(), QualifiedInsights()
    reviewer = RepairReviewer(second="unbound_premises")
    reservations: list[bool] = []

    def reserve() -> float:
        reservations.append(True)
        return 7

    with pytest.raises(ProcessingError) as caught:
        analyse_document(
            source(), config(), draft, reserve, insight_generator=insights, reviewer=reviewer
        )
    assert caught.value.code == "EVIDENCE_REVIEW_INVALID"
    assert caught.value.diagnostics is not None
    assert caught.value.diagnostics.reason == "unbound_insight_premises"
    assert caught.value.diagnostics.input_tokens == caught.value.diagnostics.output_tokens == 300
    assert [len(p.calls) for p in (draft, insights, reviewer)] == [1, 1, 2]
    assert len(reservations) == 4
    assert reviewer.calls[0]["content"] == reviewer.calls[1]["content"]


def test_unknown_review_error_never_enters_repair_prompt_or_saved_metadata() -> None:
    reviewer = RepairReviewer(first="unknown_diagnostic")
    result = analyse_document(
        source(),
        config(),
        DraftGenerator(),
        lambda: 7,
        insight_generator=QualifiedInsights(),
        reviewer=reviewer,
    )
    assert len(reviewer.calls) == 2
    repaired = reviewer.calls[-1]
    assert repaired["system"] == (
        ANALYSIS_REVIEW_SYSTEM + analysis_review_repair_instruction("structured_parse_failure")
    )
    assert RAW_ERROR_CANARY not in repaired["system"] + repaired["content"]
    assert RAW_ERROR_CANARY not in json.dumps(result.payload)
    assert result.payload["usage"]["call_history"][-2]["diagnostics"]["reason"] == (
        "structured_parse_failure"
    )
    assert result.payload["insights"][0]["text"] == INSIGHT


def test_candidate_repair_cannot_exceed_reserved_document_budget() -> None:
    reviewer = RepairReviewer()
    reservations: list[bool] = []

    def reserve() -> float:
        reservations.append(True)
        return 7

    with pytest.raises(ProcessingError) as caught:
        analyse_document(
            source(),
            config(max_document_ai_calls=3),
            DraftGenerator(),
            reserve,
            insight_generator=QualifiedInsights(),
            reviewer=reviewer,
        )
    assert caught.value.code == "ai_call_budget"
    assert len(reservations) == 3 and len(reviewer.calls) == 1
