"""Frozen regression exercises production audit orchestration without live calls."""

import json
from typing import Any
from uuid import UUID

import pytest

from app.core.config import Settings
from app.domain.conversation import Evidence
from app.domain.documents import Generated, ProcessingError
from app.services.chat_review import CHAT_REVIEW_PROMPT_VERSION
from app.services.evidence_review import REVIEW_PROMPT_VERSION
from scripts.evaluate_chat_review import generate_case, recipe, selected_cases


class Provider:
    def __init__(self, result: dict[str, Any], *, fail: bool = False) -> None:
        self.result, self.fail = result, fail
        self.calls: list[dict[str, Any]] = []

    async def generate(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.fail:
            raise ProcessingError("ai_timeout", "Provider timed out.")
        return Generated(kwargs["schema"].model_validate(self.result), "unit-model", 10, 5)


def settings() -> Settings:
    return Settings(_env_file=None, openai_api_key="test-placeholder")


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
        "required_qualifications": [{"quote_ids": ["q2"], "conveyed_by_claim_ids": ["a1.1"]}],
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
    assert metadata["qualification_policy"] == "retained_claim_source_bindings"
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
