"""Production answer review with deterministic verdicts; no provider/network evidence."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import pytest

from app.core.config import Settings
from app.domain.conversation import Evidence, TurnReservation
from app.domain.documents import Generated, ProcessingError
from app.services.analysis import FactualBlock
from app.services.chat import ChatProcessor
from app.services.chat_review import ChatReviewOutput
from app.services.grounding import AnswerOutput

pytestmark = pytest.mark.unit


def reservation() -> TurnReservation:
    now = datetime.now(UTC)
    return TurnReservation(
        UUID(int=1),
        UUID(int=2),
        UUID(int=3),
        UUID(int=4),
        UUID(int=5),
        UUID(int=6),
        UUID(int=7),
        "When can it be deleted?",
        now + timedelta(seconds=90),
        now + timedelta(seconds=120),
        "test",
        1,
    )


def fixture() -> tuple[ChatProcessor, AnswerOutput, tuple[Evidence, ...]]:
    evidence = Evidence(
        UUID(int=8),
        UUID(int=6),
        UUID(int=7),
        "rules.pdf",
        "A legal hold blocks deletion. Release does not restart the retention clock.",
        (),
    )
    processor = object.__new__(ChatProcessor)
    processor.settings = Settings(_env_file=None, openai_api_key="test-placeholder")
    processor.reviewer = None  # type: ignore[assignment]
    draft = AnswerOutput(
        evidence_status="answered",
        blocks=[
            FactualBlock.model_validate(
                {
                    "text": "A legal hold blocks deletion.",
                    "source_ids": [evidence.id],
                    "support_quotes": [
                        {"source_id": evidence.id, "quote": "A legal hold blocks deletion."}
                    ],
                }
            )
        ],
    )
    return processor, draft, (evidence,)


@pytest.mark.parametrize("overview", [False, True])
async def test_review_policy_restores_missing_rule_and_keeps_original_claims(
    monkeypatch: pytest.MonkeyPatch,
    overview: bool,
) -> None:
    processor, draft, evidence = fixture()
    calls: list[dict[str, Any]] = []

    async def generate(*args: Any, **kwargs: Any) -> Generated[ChatReviewOutput]:
        calls.append(kwargs)
        return Generated(
            ChatReviewOutput.model_validate(
                {
                    "required_aspects": [
                        {
                            "aspect": "Retention clock after release",
                            "quote_ids": ["q2"],
                            "conveyed_by_claim_ids": ["a1.1"],
                        }
                    ],
                    "decisions": [
                        {
                            "claim_id": "a1.1",
                            "supported": True,
                            "quote_ids": ["q1"],
                            "redundant_with": [],
                        }
                    ],
                    "scope_complete": True,
                    "answerability": "answered",
                    "unanswered_parts": [],
                    "relevant_claim_ids": ["a1.1"],
                }
            ),
            "test-model",
            100,
            50,
        )

    monkeypatch.setattr(processor, "_generate", generate)
    answer, metadata = await processor._review_answer(
        reservation(),
        draft,
        evidence,
        overview=overview,
        complete_scope=True,
    )
    assert len(calls) == 1
    assert answer.answer.startswith("A legal hold blocks deletion.")
    assert "Release does not restart" in answer.answer
    assert metadata["force_qualification_quotes"] is False
    assert metadata["qualification_policy"] == "required_aspect_source_bindings"
    assert metadata["claims"][0]["id"] == "a1.1"
    assert metadata["quote_bindings"]["q2"]["quote"].startswith("Release")
    assert all(len(c["excerpt"]) <= 1200 for c in answer.citations)


async def test_review_input_budget_fails_before_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor, draft, evidence = fixture()
    processor.settings = processor.settings.model_copy(
        update={
            "chat_prompt_max_tokens": 1,
            "chat_max_output_tokens": 1,
            "INSTRUCTION_RESERVE_TOKENS": 1,
        }
    )
    called = False

    async def generate(*args: Any, **kwargs: Any) -> None:
        nonlocal called
        called = True
        raise AssertionError("Provider must not be invoked")

    monkeypatch.setattr(processor, "_generate", generate)
    record: dict[str, Any] = {}
    with pytest.raises(ProcessingError, match="input budget") as caught:
        await processor._review_answer(
            reservation(),
            draft,
            evidence,
            overview=False,
            complete_scope=True,
            review_record=record,
        )
    assert not called
    assert record["projection_status"] == "input_rejected"
    assert record["input_tokens"] > record["input_limit"]
    assert record["claims"][0]["text"] == draft.blocks[0].text
    assert caught.value.diagnostics is not None
    assert caught.value.diagnostics.as_dict()["reason"] == "input_token_limit"


async def test_review_failure_does_not_return_unaudited_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor, draft, evidence = fixture()

    async def generate(*args: Any, **kwargs: Any) -> None:
        raise ProcessingError("AI_UNAVAILABLE", "Provider unavailable")

    monkeypatch.setattr(processor, "_generate", generate)
    with pytest.raises(ProcessingError, match="Provider unavailable"):
        await processor._review_answer(
            reservation(), draft, evidence, overview=False, complete_scope=True
        )


async def test_invalid_review_retains_bounded_audit_without_returning_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor, draft, evidence = fixture()

    async def generate(*args: Any, **kwargs: Any) -> Generated[ChatReviewOutput]:
        return Generated(
            ChatReviewOutput.model_validate(
                {
                    "required_aspects": [
                        {
                            "aspect": "Retention clock after release",
                            "quote_ids": ["unknown-source"],
                            "conveyed_by_claim_ids": [],
                        }
                    ],
                    "decisions": [
                        {
                            "claim_id": "a1.1",
                            "supported": True,
                            "quote_ids": ["q1"],
                            "redundant_with": [],
                        }
                    ],
                    "scope_complete": True,
                    "answerability": "answered",
                    "unanswered_parts": [],
                    "relevant_claim_ids": ["a1.1"],
                }
            ),
            "test-model",
            100,
            50,
        )

    monkeypatch.setattr(processor, "_generate", generate)
    record: dict[str, Any] = {}
    with pytest.raises(ProcessingError) as caught:
        await processor._review_answer(
            reservation(),
            draft,
            evidence,
            overview=False,
            complete_scope=True,
            review_record=record,
        )
    assert caught.value.code == "EVIDENCE_REVIEW_INVALID"
    assert record["projection_status"] == "failed"
    assert record["projection_error"]["reason"] == "invalid_qualification_selection"
    assert record["review"]["required_aspects"][0]["quote_ids"] == ["unknown-source"]
    assert record["claims"][0]["text"] == draft.blocks[0].text
    assert "q1" in record["quote_bindings"]
    assert "unknown-source" not in record["quote_bindings"]


@pytest.mark.parametrize("allow_repair,expected_calls", [(True, 2), (False, 1)])
async def test_review_repair_is_bounded_and_uses_unchanged_evidence(
    monkeypatch: pytest.MonkeyPatch,
    allow_repair: bool,
    expected_calls: int,
) -> None:
    processor, draft, evidence = fixture()
    calls: list[tuple[Any, ...]] = []

    async def generate(*args: Any, **kwargs: Any) -> Generated[ChatReviewOutput]:
        calls.append(args)
        return Generated(
            ChatReviewOutput.model_validate(
                {
                    "required_aspects": [
                        {
                            "aspect": "Retention clock after release",
                            "quote_ids": ["unknown"] if len(calls) == 1 else ["q2"],
                            "conveyed_by_claim_ids": [],
                        }
                    ],
                    "decisions": [
                        {
                            "claim_id": "a1.1",
                            "supported": True,
                            "quote_ids": ["q1"],
                            "redundant_with": [],
                        }
                    ],
                    "scope_complete": True,
                    "answerability": "answered",
                    "unanswered_parts": [],
                    "relevant_claim_ids": ["a1.1"],
                }
            ),
            "test-model",
            100,
            50,
        )

    monkeypatch.setattr(processor, "_generate", generate)
    if allow_repair:
        answer, record = await processor._review_answer(
            reservation(),
            draft,
            evidence,
            overview=False,
            complete_scope=True,
            allow_repair=allow_repair,
        )
        assert "A legal hold blocks deletion." in answer.answer
        assert record["review_attempt"] == 2 and record["repair"]["attempted"]
        assert calls[0][3] == calls[1][3]
        assert calls[0][2] in calls[1][2]
        assert "unknown" not in calls[1][2]
    else:
        with pytest.raises(ProcessingError):
            await processor._review_answer(
                reservation(),
                draft,
                evidence,
                overview=False,
                complete_scope=True,
                allow_repair=False,
            )
    assert len(calls) == expected_calls


@pytest.mark.parametrize("complete_scope,expected", [(True, "answered"), (False, "partial")])
async def test_review_can_correct_provisional_partial_without_overriding_source_scope(
    monkeypatch: pytest.MonkeyPatch, complete_scope: bool, expected: str
) -> None:
    processor, draft, evidence = fixture()
    draft = draft.model_copy(update={"evidence_status": "partial"})

    async def generate(*args: Any, **kwargs: Any) -> Generated[ChatReviewOutput]:
        return Generated(
            ChatReviewOutput(
                required_aspects=[
                    {
                        "aspect": "Deletion while a legal hold applies",
                        "quote_ids": ["q1"],
                        "conveyed_by_claim_ids": ["a1.1"],
                    }
                ],
                decisions=[
                    {
                        "claim_id": "a1.1",
                        "supported": True,
                        "quote_ids": ["q1"],
                        "redundant_with": [],
                    }
                ],
                scope_complete=True,
                answerability="answered",
                relevant_claim_ids=["a1.1"],
                unanswered_parts=[],
            ),
            "test-model",
            100,
            50,
        )

    monkeypatch.setattr(processor, "_generate", generate)
    answer, record = await processor._review_answer(
        reservation(), draft, evidence, overview=False, complete_scope=complete_scope
    )
    assert answer.evidence_status == expected
    assert answer.blocks[0].text == draft.blocks[0].text
    assert answer.citations
    assert record["draft_evidence_status"] == "partial"
    assert record["answerability"] == "answered"


@pytest.mark.parametrize("complete_scope,expected", [(True, "answered"), (False, "partial")])
async def test_irrelevant_primary_write_pruning_preserves_complete_fee_answer(
    monkeypatch: pytest.MonkeyPatch, complete_scope: bool, expected: str
) -> None:
    processor, _, source = fixture()
    evidence = (
        replace(source[0], text="The fee is 480 credits."),
        replace(source[0], id=UUID(int=9), text="The primary handles writes."),
    )
    draft = AnswerOutput(
        evidence_status="answered",
        blocks=[
            FactualBlock.model_validate(
                {
                    "text": item.text,
                    "source_ids": [item.id],
                    "support_quotes": [{"source_id": item.id, "quote": item.text}],
                }
            )
            for item in evidence
        ],
    )

    async def generate(*args: Any, **kwargs: Any) -> Generated[ChatReviewOutput]:
        return Generated(
            ChatReviewOutput.model_validate(
                {
                    "required_aspects": [
                        {
                            "aspect": "The fee amount",
                            "quote_ids": ["q1"],
                            "conveyed_by_claim_ids": ["a1.1"],
                        }
                    ],
                    "decisions": [
                        {
                            "claim_id": f"a{index}.1",
                            "supported": True,
                            "quote_ids": [f"q{index}"],
                            "redundant_with": [],
                        }
                        for index in (1, 2)
                    ],
                    "scope_complete": True,
                    "answerability": "answered",
                    "unanswered_parts": [],
                    "relevant_claim_ids": ["a1.1"],
                }
            ),
            "test-model",
            100,
            50,
        )

    monkeypatch.setattr(processor, "_generate", generate)
    answer, record = await processor._review_answer(
        replace(reservation(), question="What is the fee?"),
        draft,
        evidence,
        overview=False,
        complete_scope=complete_scope,
    )
    assert answer.evidence_status == expected
    assert answer.blocks[0].text == "The fee is 480 credits."
    assert len(answer.blocks) == len(answer.citations) == 1
    assert answer.citations[0]["source_id"] == str(evidence[0].id)
    assert "primary" not in answer.answer
    assert record["scope_complete"]
    assert record["unanswered_parts"] == []
    assert record["dropped_claim_ids"] == ["a2.1"]


async def test_partial_missing_component_is_rendered_separately_from_factual_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor, draft, evidence = fixture()
    missing = "exact measured percentage speedup"

    async def generate(*args: Any, **kwargs: Any) -> Generated[ChatReviewOutput]:
        return Generated(
            ChatReviewOutput(
                required_aspects=[
                    {
                        "aspect": "Deletion while a legal hold applies",
                        "quote_ids": ["q1"],
                        "conveyed_by_claim_ids": ["a1.1"],
                    },
                    {"aspect": missing, "quote_ids": [], "conveyed_by_claim_ids": []},
                ],
                decisions=[
                    {
                        "claim_id": "a1.1",
                        "supported": True,
                        "quote_ids": ["q1"],
                        "redundant_with": [],
                    }
                ],
                scope_complete=False,
                answerability="partial",
                relevant_claim_ids=["a1.1"],
                unanswered_parts=[missing],
            ),
            "test-model",
            100,
            50,
        )

    monkeypatch.setattr(processor, "_generate", generate)
    answer, record = await processor._review_answer(
        reservation(), draft, evidence, overview=False, complete_scope=True
    )
    assert answer.evidence_status == "partial"
    assert "Not established by the supplied evidence:\n- " + missing in answer.answer
    assert len(answer.blocks) == 1 and answer.blocks[0].text == draft.blocks[0].text
    assert len(answer.citations) == 1
    assert answer.citations[0]["source_id"] == str(evidence[0].id)
    assert missing not in answer.citations[0]["excerpt"]
    assert missing not in answer.blocks[0].text
    assert record["unanswered_parts"] == [missing]
