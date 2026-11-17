"""Frozen regression exercises production audit orchestration without live calls."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import MagicMock, Mock
from uuid import UUID

import pytest

from app.adapters.async_openai import AsyncOpenAIGenerationProvider
from app.core.config import Settings
from app.domain.conversation import Evidence
from app.domain.documents import Generated, ProcessingError
from app.services.chat_review import CHAT_REVIEW_PROMPT_VERSION
from app.services.evidence_review import REVIEW_PROMPT_VERSION
from scripts import evaluate_chat_review
from scripts.evaluate_chat_review import generate_case, recipe, selected_cases
from scripts.evaluate_review import BUDGET_FILE
from scripts.live_budget import _request_fingerprint


class Provider:
    def __init__(self, result: dict[str, Any], *, fail: bool = False) -> None:
        self.result, self.fail = result, fail
        self.calls: list[dict[str, Any]] = []

    async def generate(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.fail:
            raise ProcessingError("ai_timeout", "Provider timed out.")
        return Generated(kwargs["schema"].model_validate(self.result), "unit-model", 10, 5)


def settings(**values: Any) -> Settings:
    return Settings(_env_file=None, openai_api_key="test-placeholder", **values)


def evidence() -> tuple[Evidence, ...]:
    return (
        Evidence(
            UUID(int=1),
            UUID(int=2),
            UUID(int=3),
            "policy.docx",
            "A hold blocks deletion until release. Release does not restart the ordinary clock.",
            (),
        ),
    )


def payload(*, intent: str = "factual", complete: bool = True) -> dict[str, Any]:
    return {
        "question": "How does a legal hold affect retention?",
        "intent": intent,
        "complete_scope": complete,
        "history": [{"role": "user", "content": "A frozen earlier question"}],
        "evidence_scope": "retrieved_excerpts",
        "evidence": [{"old_quote_id": "must-be-recataloged"}],
    }


def review() -> dict[str, Any]:
    return {
        "required_aspects": [
            {
                "aspect": "Effect after release",
                "quote_ids": ["q2"],
                "conveyed_by_claim_ids": ["a1.1"],
            }
        ],
        "decisions": [
            {"claim_id": "a1.1", "supported": True, "quote_ids": ["q1"], "redundant_with": []}
        ],
        "scope_complete": True,
        "answerability": "answered",
        "unanswered_parts": [],
        "relevant_claim_ids": ["a1.1"],
    }


async def test_frozen_supported_case_uses_current_audit_and_qualification_policy() -> None:
    draft = Provider(
        {
            "evidence_status": "answered",
            "blocks": [{"text": "A hold blocks deletion until release.", "quote_ids": ["q1"]}],
        }
    )
    auditor = Provider(review())
    answer, metadata = await generate_case(settings(), payload(), evidence(), draft, auditor)
    assert len(draft.calls) == len(auditor.calls) == 1
    assert "Release does not restart" in answer.answer
    assert metadata["force_qualification_quotes"] is False
    assert metadata["qualification_policy"] == "required_aspect_source_bindings"
    request = json.loads(draft.calls[0]["content"])
    assert request["history"] == payload()["history"]
    assert request["evidence"][0]["passages"][0]["quote_id"] == "q1"
    assert "old_quote_id" not in request["evidence"][0]


async def test_empty_insufficient_draft_spends_no_audit_call() -> None:
    draft = Provider({"evidence_status": "insufficient", "blocks": []})
    auditor = Provider({})
    answer, metadata = await generate_case(settings(), payload(), evidence(), draft, auditor)
    assert answer.evidence_status == "insufficient"
    assert len(draft.calls) == 1
    assert auditor.calls == []
    assert metadata["audit_skipped"] == "empty_draft"


async def test_incomplete_exhaustive_case_remains_literal_without_audit() -> None:
    draft = Provider({"quote_ids": ["q1"]})
    auditor = Provider({})
    answer, metadata = await generate_case(
        settings(), payload(intent="exhaustive", complete=False), evidence(), draft, auditor
    )
    assert answer.evidence_status == "partial"
    assert "A hold blocks deletion until release." in answer.answer
    assert auditor.calls == []
    assert metadata["audit_skipped"] == "literal_excerpts"


async def test_audit_failure_never_publishes_an_unaudited_frozen_answer() -> None:
    draft = Provider(
        {
            "evidence_status": "answered",
            "blocks": [{"text": "A hold blocks deletion until release.", "quote_ids": ["q1"]}],
        }
    )
    auditor = Provider({}, fail=True)
    with pytest.raises(ProcessingError, match="Provider timed out"):
        await generate_case(settings(), payload(), evidence(), draft, auditor)
    assert len(draft.calls) == len(auditor.calls) == 1


def test_suite_selection_keeps_core_and_quality_cases_separate() -> None:
    assert [case.id for case in selected_cases("quality", "Q4,Q3")] == ["Q3", "Q4"]
    assert len(selected_cases("core", None)) == 20
    with pytest.raises(ValueError, match="UNKNOWN_QUALITY_CASE_ID"):
        selected_cases("quality", "A1")


def test_frozen_recipe_records_chat_review_version_instead_of_analysis_version() -> None:
    assert recipe(settings())["review_prompt"] == CHAT_REVIEW_PROMPT_VERSION
    assert recipe(settings())["review_prompt"] != REVIEW_PROMPT_VERSION


@pytest.mark.parametrize("effort", ["none", "low"])
async def test_frozen_run_uses_and_records_configured_answer_effort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, effort: Literal["none", "low"]
) -> None:
    configured = settings(
        local_storage_dir=tmp_path,
        openai_answer_reasoning_effort=effort,
        openai_review_reasoning_effort="none",
    )
    case = selected_cases("quality", "Q3")[0]
    frozen_payload = {**payload(), "question": case.question}
    source = tmp_path / "frozen.json"
    source.write_text(
        json.dumps(
            {
                "cases": {
                    case.id: {
                        "status": "COMPLETED",
                        "question": case.question,
                        "document_id": str(evidence()[0].document_id),
                        "response": {"turn_id": str(UUID(int=4))},
                    }
                }
            }
        )
    )
    session = MagicMock()
    session.__enter__.return_value.get.return_value = SimpleNamespace(
        context_snapshot={"payload": frozen_payload}
    )
    engine = Mock()
    monkeypatch.setattr(evaluate_chat_review, "get_settings", lambda: configured)
    monkeypatch.setattr(evaluate_chat_review, "create_engine", lambda *args, **kwargs: engine)
    monkeypatch.setattr(evaluate_chat_review, "Session", lambda *args: session)
    monkeypatch.setattr(evaluate_chat_review, "saved_evidence", lambda *args: evidence())
    calls: list[tuple[str, dict[str, Any]]] = []

    async def substitute(provider: Any, **kwargs: Any) -> Generated[Any]:
        calls.append((provider.reasoning_effort, kwargs))
        result = (
            {
                "evidence_status": "answered",
                "blocks": [{"text": "A hold blocks deletion until release.", "quote_ids": ["q1"]}],
            }
            if kwargs["schema"].__name__ == "AnswerSelectionOutput"
            else review()
        )
        return Generated(kwargs["schema"].model_validate(result), "unit-model", 10, 5)

    monkeypatch.setattr(AsyncOpenAIGenerationProvider, "generate", substitute)
    result = await evaluate_chat_review.run(source, "configured-effort", [case])
    assert [value for value, _ in calls] == [effort, "none"]
    assert [kwargs["schema"].__name__ for _, kwargs in calls] == [
        "AnswerSelectionOutput",
        "ChatReviewOutput",
    ]
    assert result["used"] == 2 and result["completed"] == 1
    artifact = Path(result["artifact"])
    saved = json.loads(await asyncio.to_thread(artifact.read_text))
    assert saved["recipe"]["configuration"]["openai_answer_reasoning_effort"] == effort
    assert saved["recipe"]["configuration"]["openai_review_reasoning_effort"] == "none"
    ledger = json.loads((tmp_path / "demo" / BUDGET_FILE).read_text())
    assert [entry["label"] for entry in ledger["entries"]] == [
        "frozen.Q3.draft",
        "frozen.Q3.review",
    ]
    assert [entry["fingerprint"] for entry in ledger["entries"]] == [
        _request_fingerprint(kwargs, profile) for profile, kwargs in calls
    ]
    frozen_bytes = await asyncio.to_thread(artifact.read_bytes)
    configured = settings(
        local_storage_dir=tmp_path,
        openai_answer_reasoning_effort="none" if effort == "low" else "low",
        openai_review_reasoning_effort="none",
    )
    with pytest.raises(ValueError, match="FROZEN_RUN_RECIPE_CHANGED"):
        await evaluate_chat_review.run(source, "configured-effort", [case])
    assert len(calls) == 2
    assert await asyncio.to_thread(artifact.read_bytes) == frozen_bytes
