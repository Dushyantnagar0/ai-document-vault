"""Final analysis orchestration uses frozen drafts and bounded original-source review.

Providers here are deterministic substitutes, not semantic-quality evidence.
"""

import json
from typing import Any
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.domain.documents import AIFailureDetails, Chunk, Generated, ProcessingError
from app.services.analysis import analyse_document
from app.services.analysis_evidence import QUOTE_CATALOG_VERSION
from app.services.evidence_review import PROJECTION_VERSION, REVIEW_PROMPT_VERSION
from app.services.insight_generation import INSIGHT_PROMPT_VERSION

pytestmark = pytest.mark.unit


class DraftGenerator:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> Generated[Any]:
        self.calls.append(kwargs)
        first = json.loads(kwargs["content"])["evidence"][0]["quote_id"]
        return Generated(
            result=kwargs["schema"].model_validate(
                {
                    "summary": [
                        {
                            "text": "Two workers process three exports each per hour.",
                            "quote_ids": [first],
                        }
                    ],
                    # The focused stage must replace this joint draft output.
                    "insights": [{"text": "Discard this joint draft.", "quote_ids": [first]}],
                }
            ),
            model=Settings.CHAT_MODEL,
            input_tokens=100,
            output_tokens=100,
        )


class InsightGenerator:
    def __init__(self, *, duplicate: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.duplicate = duplicate

    def generate(self, **kwargs: Any) -> Generated[Any]:
        self.calls.append(kwargs)
        request = json.loads(kwargs["content"])
        first = request["evidence"][0]["quote_id"]
        return Generated(
            result=kwargs["schema"].model_validate(
                {
                    "insights": [
                        {
                            "text": (
                                "Under these assumptions, combined capacity "
                                "is six exports per hour."
                            ),
                            "quote_ids": [first, first] if self.duplicate else [first],
                        }
                    ]
                }
            ),
            model=Settings.CHAT_MODEL,
            input_tokens=200,
            output_tokens=200,
        )


class Reviewer:
    def __init__(self, *, partial: bool = False, invalid: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.partial, self.invalid = partial, invalid

    def generate(self, **kwargs: Any) -> Generated[Any]:
        self.calls.append(kwargs)
        request = json.loads(kwargs["content"])
        first = request["evidence"][0]["quote_id"]
        decisions = [
            {
                "claim_id": "unknown" if self.invalid else claim["claim_id"],
                "supported": True,
                "quote_ids": [first],
                "redundant_with": [],
            }
            for claim in request["claims"]
        ]
        return Generated(
            result=kwargs["schema"].model_validate(
                {
                    "decisions": decisions,
                    "required_qualifications": [],
                    "scope_complete": not self.partial,
                }
            ),
            model=Settings.CHAT_MODEL,
            input_tokens=300,
            output_tokens=300,
        )


def config(**values: Any) -> Settings:
    return Settings(_env_file=None, openai_api_key="unit-placeholder", **values)


def source(count: int = 1) -> list[Chunk]:
    return [
        Chunk(uuid4(), i, "Two workers process three exports each per hour.", 10, ())
        for i in range(count)
    ]


def test_final_summary_freezes_before_distinct_insights_and_review() -> None:
    draft, insights, reviewer = DraftGenerator(), InsightGenerator(), Reviewer()
    reservations: list[bool] = []

    def reserve() -> float:
        reservations.append(True)
        return 7

    chunks = source(2)
    result = analyse_document(
        chunks, config(), draft, reserve, insight_generator=insights, reviewer=reviewer
    )
    assert len(reservations) == 3
    assert [len(p.calls) for p in (draft, insights, reviewer)] == [1, 1, 1]
    assert all(p.calls[0]["timeout_seconds"] == 7 for p in (draft, insights, reviewer))
    assert json.loads(insights.calls[0]["content"])["frozen_summary"] == [
        result.payload["summary"][0]["text"]
    ]
    assert "six exports" in result.payload["insights"][0]["text"]
    assert "Discard this" not in json.dumps(result.payload)
    assert result.covered_chunk_ids == tuple(c.id for c in chunks)
    assert result.payload["analysis_coverage"] == 1.0
    assert result.payload["analysis_scope_complete"] is True
    assert result.payload["analysis_warnings"] == []
    history = result.payload["usage"]["call_history"]
    assert [call["stage"] for call in history] == ["final", "insights", "evidence_review"]
    assert result.payload["usage"]["known_input_tokens"] == 600
    assert result.payload["usage"]["known_output_tokens"] == 600
    review = result.payload["evidence_review"]
    assert review["prompt_version"] == REVIEW_PROMPT_VERSION
    assert review["projection_version"] == PROJECTION_VERSION
    assert review["insight_prompt_version"] == INSIGHT_PROMPT_VERSION
    assert review["quote_catalog_version"] == QUOTE_CATALOG_VERSION
    assert len(review["catalog_sha256"]) == 64
    assert all(v["quote"] in chunks[0].text for v in review["quote_bindings"].values())


def test_partial_review_does_not_misrepresent_extraction_participation() -> None:
    result = analyse_document(
        source(),
        config(),
        DraftGenerator(),
        lambda: 30,
        insight_generator=InsightGenerator(),
        reviewer=Reviewer(partial=True),
    )
    assert result.payload["analysis_scope_complete"] is False
    assert result.payload["analysis_warnings"][0]["code"] == "ANALYSIS_SCOPE_PARTIAL"
    assert result.payload["analysis_coverage"] == 1.0


def test_persistently_invalid_reviewer_decisions_fail_after_one_repair() -> None:
    reviewer = Reviewer(invalid=True)
    with pytest.raises(ProcessingError) as failure:
        analyse_document(
            source(),
            config(),
            DraftGenerator(),
            lambda: 30,
            insight_generator=InsightGenerator(),
            reviewer=reviewer,
        )
    assert failure.value.code == "EVIDENCE_REVIEW_INVALID"
    assert len(reviewer.calls) == 2


def test_invalid_candidates_prevent_review_and_are_not_repaired() -> None:
    insights, reviewer = InsightGenerator(duplicate=True), Reviewer()
    with pytest.raises(ProcessingError) as failure:
        analyse_document(
            source(),
            config(),
            DraftGenerator(),
            lambda: 30,
            insight_generator=insights,
            reviewer=reviewer,
        )
    assert failure.value.code == "INSIGHT_CANDIDATES_INVALID"
    assert len(insights.calls) == 1 and not reviewer.calls


def test_persisted_reservation_failure_prevents_quality_request() -> None:
    calls = 0
    insights = InsightGenerator()

    def reserve() -> float:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ProcessingError("AI_CALL_BUDGET_EXHAUSTED", "Budget exhausted.")
        return 30

    with pytest.raises(ProcessingError) as failure:
        analyse_document(
            source(),
            config(),
            DraftGenerator(),
            reserve,
            insight_generator=insights,
            reviewer=Reviewer(),
        )
    assert failure.value.code == "AI_CALL_BUDGET_EXHAUSTED"
    assert not insights.calls


def test_quality_adapters_must_be_injected_together() -> None:
    draft = DraftGenerator()
    with pytest.raises(ProcessingError) as failure:
        analyse_document(source(), config(), draft, lambda: 30, reviewer=Reviewer())
    assert failure.value.code == "analysis_configuration_invalid"
    assert not draft.calls


def test_complete_review_prompt_limit_prevents_reservation_and_request(monkeypatch: Any) -> None:
    from app.services import analysis
    from app.services.evidence_review import ReviewOutput

    real_counter = analysis.generation_prompt_tokens
    monkeypatch.setattr(
        analysis,
        "generation_prompt_tokens",
        lambda schema, system, content: (
            Settings.CHAT_MODEL_MAX_INPUT_TOKENS + 1
            if schema is ReviewOutput
            else real_counter(schema, system, content)
        ),
    )
    reservations: list[bool] = []
    reviewer = Reviewer()

    def reserve() -> float:
        reservations.append(True)
        return 30

    with pytest.raises(ProcessingError) as failure:
        analyse_document(
            source(),
            config(),
            DraftGenerator(),
            reserve,
            insight_generator=InsightGenerator(),
            reviewer=reviewer,
        )
    assert failure.value.code == "analysis_prompt_limit"
    assert len(reservations) == 2 and not reviewer.calls


def test_expired_deadline_prevents_quality_request_after_reservation() -> None:
    remaining = iter([30.0, 0.0])
    insights = InsightGenerator()
    with pytest.raises(ProcessingError) as failure:
        analyse_document(
            source(),
            config(),
            DraftGenerator(),
            lambda: next(remaining),
            insight_generator=insights,
            reviewer=Reviewer(),
        )
    assert failure.value.code == "processing_deadline"
    assert not insights.calls


def test_final_only_quality_review_uses_bounded_surviving_original_passages() -> None:
    chunks = [
        Chunk(
            uuid4(),
            i,
            "Two workers process three exports each per hour. "
            + " ".join(f"detail{j}" for j in range(250)),
            520,
            (),
        )
        for i in range(12)
    ]
    draft, insights, reviewer = DraftGenerator(), InsightGenerator(), Reviewer()
    result = analyse_document(
        chunks,
        config(analysis_group_max_tokens=1800),
        draft,
        lambda: 30,
        insight_generator=insights,
        reviewer=reviewer,
    )
    assert len(draft.calls) > 1
    assert len(insights.calls) == len(reviewer.calls) == 1
    assert result.covered_chunk_ids == tuple(chunk.id for chunk in chunks)
    final_evidence = json.loads(draft.calls[-1]["content"])["evidence"]
    audited = json.loads(reviewer.calls[0]["content"])["evidence"]
    assert [item["text"] for item in audited] == [item["text"] for item in final_evidence]
    assert all(any(item["text"] in chunk.text for chunk in chunks) for item in audited)
    assert all("Discard this joint draft" not in item["text"] for item in audited)


class QualificationReviewer(Reviewer):
    def __init__(self, *, persistently_invalid: bool = False) -> None:
        super().__init__()
        self.persistently_invalid = persistently_invalid

    def generate(self, **kwargs: Any) -> Generated[Any]:
        response = super().generate(**kwargs)
        if len(self.calls) == 1 or self.persistently_invalid:
            raw = response.result.model_dump(mode="json")
            raw["required_qualifications"] = [
                {"quote_ids": ["unknown-qualification-source"], "conveyed_by_claim_ids": []}
            ]
            return Generated(
                kwargs["schema"].model_validate(raw),
                response.model,
                response.input_tokens,
                response.output_tokens,
            )
        return response


def test_invalid_qualification_review_repairs_same_packet_and_preserves_failed_usage() -> None:
    from app.services.evidence_review import REVIEW_REPAIR, REVIEW_SYSTEM

    reviewer = QualificationReviewer()
    reservations: list[bool] = []

    def reserve() -> float:
        reservations.append(True)
        return 9

    result = analyse_document(
        source(),
        config(),
        DraftGenerator(),
        reserve,
        insight_generator=InsightGenerator(),
        reviewer=reviewer,
    )
    assert len(reviewer.calls) == 2
    assert len(reservations) == 4
    assert reviewer.calls[0]["content"] == reviewer.calls[1]["content"]
    assert reviewer.calls[0]["system"] == REVIEW_SYSTEM
    assert reviewer.calls[1]["system"] == REVIEW_SYSTEM + REVIEW_REPAIR
    assert "unknown-qualification-source" not in reviewer.calls[1]["content"]
    assert all(call["timeout_seconds"] == 9 for call in reviewer.calls)
    history = result.payload["usage"]["call_history"]
    assert [call["stage"] for call in history] == [
        "final",
        "insights",
        "evidence_review",
        "evidence_review",
    ]
    assert history[-2]["outcome"] == "failed"
    assert history[-2]["error_code"] == "EVIDENCE_REVIEW_INVALID"
    assert history[-2]["diagnostics"]["reason"] == "invalid_qualification_selection"
    assert history[-2]["input_tokens"] == history[-2]["output_tokens"] == 300
    assert history[-2]["repair"] is False
    assert history[-1]["outcome"] == "completed" and history[-1]["repair"] is True
    assert result.payload["usage"]["known_input_tokens"] == 900
    assert result.payload["usage"]["known_output_tokens"] == 900


def test_persistently_unknown_qualification_fails_after_exactly_one_review_repair() -> None:
    reviewer = QualificationReviewer(persistently_invalid=True)
    with pytest.raises(ProcessingError) as caught:
        analyse_document(
            source(),
            config(),
            DraftGenerator(),
            lambda: 30,
            insight_generator=InsightGenerator(),
            reviewer=reviewer,
        )
    assert caught.value.code == "EVIDENCE_REVIEW_INVALID"
    assert caught.value.diagnostics is not None
    assert caught.value.diagnostics.reason == "invalid_qualification_selection"
    assert len(reviewer.calls) == 2


@pytest.mark.parametrize(
    "code", ["ai_invalid_json", "ai_schema_invalid", "ai_output_truncated", "ai_invalid_output"]
)
def test_structured_review_failure_gets_one_bounded_repair(code: str) -> None:
    class InitiallyMalformed(Reviewer):
        def generate(self, **kwargs: Any) -> Generated[Any]:
            if not self.calls:
                self.calls.append(kwargs)
                raise ProcessingError(
                    code,
                    "Invalid structured response.",
                    diagnostics=AIFailureDetails(reason="schema_validation", input_tokens=71),
                )
            return super().generate(**kwargs)

    reviewer = InitiallyMalformed()
    result = analyse_document(
        source(),
        config(),
        DraftGenerator(),
        lambda: 30,
        insight_generator=InsightGenerator(),
        reviewer=reviewer,
    )
    assert len(reviewer.calls) == 2
    first = result.payload["usage"]["call_history"][-2]
    assert first["error_code"] == code and first["outcome"] == "failed"
    assert first["diagnostics"]["input_tokens"] == 71
    assert first["input_tokens"] == 71
    assert first["repair"] is False


@pytest.mark.parametrize(
    "code", ["AI_UNAVAILABLE", "ai_timeout", "ai_rate_limited", "ai_call_budget"]
)
def test_transport_quota_and_budget_review_failures_are_not_retried(code: str) -> None:
    class UnavailableReviewer(Reviewer):
        def generate(self, **kwargs: Any) -> Generated[Any]:
            self.calls.append(kwargs)
            raise ProcessingError(code, "Review unavailable.")

    reviewer = UnavailableReviewer()
    with pytest.raises(ProcessingError) as caught:
        analyse_document(
            source(),
            config(),
            DraftGenerator(),
            lambda: 30,
            insight_generator=InsightGenerator(),
            reviewer=reviewer,
        )
    assert caught.value.code == code
    assert len(reviewer.calls) == 1


def test_review_repair_respects_persisted_reservation_failure() -> None:
    reviewer = QualificationReviewer()
    reservations = 0

    def reserve() -> float:
        nonlocal reservations
        reservations += 1
        if reservations == 4:
            raise ProcessingError("ai_call_budget", "No review-repair allowance remains.")
        return 30

    with pytest.raises(ProcessingError) as caught:
        analyse_document(
            source(),
            config(),
            DraftGenerator(),
            reserve,
            insight_generator=InsightGenerator(),
            reviewer=reviewer,
        )
    assert caught.value.code == "ai_call_budget"
    assert len(reviewer.calls) == 1
    assert reservations == 4


def test_review_repair_cannot_exceed_document_call_limit() -> None:
    reviewer = QualificationReviewer()
    reservations: list[bool] = []

    def reserve() -> float:
        reservations.append(True)
        return 30

    with pytest.raises(ProcessingError) as caught:
        analyse_document(
            source(),
            config(max_document_ai_calls=3),
            DraftGenerator(),
            reserve,
            insight_generator=InsightGenerator(),
            reviewer=reviewer,
        )
    assert caught.value.code == "ai_call_budget"
    assert len(reservations) == 3 and len(reviewer.calls) == 1


def test_expired_deadline_prevents_second_review_request() -> None:
    reviewer = QualificationReviewer()
    remaining = iter([30.0, 30.0, 30.0, 0.0])
    with pytest.raises(ProcessingError) as caught:
        analyse_document(
            source(),
            config(),
            DraftGenerator(),
            lambda: next(remaining),
            insight_generator=InsightGenerator(),
            reviewer=reviewer,
        )
    assert caught.value.code == "processing_deadline"
    assert len(reviewer.calls) == 1


def test_local_review_schema_validation_failure_is_repairable() -> None:
    class InvalidSchemaReviewer(Reviewer):
        def generate(self, **kwargs: Any) -> Generated[Any]:
            if not self.calls:
                self.calls.append(kwargs)
                return Generated(
                    {"unexpected": "invalid-schema-canary"}, Settings.CHAT_MODEL, 101, 51
                )
            return super().generate(**kwargs)

    reviewer = InvalidSchemaReviewer()
    result = analyse_document(
        source(),
        config(),
        DraftGenerator(),
        lambda: 30,
        insight_generator=InsightGenerator(),
        reviewer=reviewer,
    )
    assert len(reviewer.calls) == 2
    first = result.payload["usage"]["call_history"][-2]
    assert first["error_code"] == "ai_schema_invalid"
    assert first["input_tokens"] == 101 and first["output_tokens"] == 51
    assert "invalid-schema-canary" not in json.dumps(result.payload)
    assert "invalid-schema-canary" not in reviewer.calls[1]["content"]
