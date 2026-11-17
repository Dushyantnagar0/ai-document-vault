"""Offline live-harness guards; these tests never establish provider quality."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.domain.conversation import Replay, TurnReservation
from app.services.chat import ChatProcessor
from app.services.chat_review import CHAT_REVIEW_PROMPT_VERSION
from app.services.conversations import TurnStore
from scripts import evaluate_service_review
from scripts.evaluate_service_review import execute_case, initialize_run, main
from scripts.live_budget import AsyncBudgetedGenerationProvider, RequestBudget


def reservation(question: str) -> TurnReservation:
    now = datetime.now(UTC)
    return TurnReservation(
        *(uuid4() for _ in range(7)),
        question,
        now + timedelta(seconds=90),
        now + timedelta(seconds=120),
        str(uuid4()),
        1,
    )


@pytest.mark.parametrize("status", ["reserved", "failed", "completed", "interrupted"])
def test_existing_run_is_preserved_without_new_work(tmp_path: Path, status: str) -> None:
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"status": status, "session_id": "saved-identity"}))
    saved = path.read_bytes()
    with pytest.raises(ValueError, match="ALREADY_EXISTS"):
        initialize_run(path, uuid4(), "databases")
    assert path.read_bytes() == saved


def test_new_run_preserves_original_reported_questions_and_distinct_request_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.json"
    state = initialize_run(path, uuid4(), "databases")
    assert state["evidence_type"] == "actual_service_pipeline_not_http_e2e"
    assert [case["question"] for case in state["cases"]][:2] == [
        "What is database",
        "Where does Data in databases is stored ?",
    ]
    assert len({case["client_request_id"] for case in state["cases"]}) == 3
    assert all(case["status"] == "unstarted" for case in state["cases"])
    assert json.loads(path.read_text()) == state


@pytest.mark.asyncio
@pytest.mark.parametrize("response_status", [200, 502])
async def test_actual_reservation_followed_by_replay_never_runs_processor_twice(
    tmp_path: Path, response_status: int
) -> None:
    record = {"question": "What is database", "client_request_id": str(uuid4())}
    saved_response = Replay(response_status, {"answer": "A saved response"})
    reserved = reservation(record["question"])
    store = SimpleNamespace(reserve=AsyncMock(side_effect=[reserved, saved_response]))
    processor = SimpleNamespace(run=AsyncMock(return_value=saved_response))
    budget = RequestBudget(tmp_path / "budget.json")
    response = await execute_case(
        cast(TurnStore, store), cast(ChatProcessor, processor), uuid4(), record, budget
    )
    assert response == saved_response
    processor.run.assert_awaited_once_with(reserved)
    assert store.reserve.await_count == 2
    assert record["replay_equal"] and record["replay_new_paid_calls"] == 0
    assert record["status"] == ("completed" if response_status == 200 else "failed")


@pytest.mark.asyncio
async def test_mismatched_replay_fails_closed(tmp_path: Path) -> None:
    question = "A question"
    store = SimpleNamespace(
        reserve=AsyncMock(side_effect=[reservation(question), Replay(200, {"changed": True})])
    )
    processor = SimpleNamespace(run=AsyncMock(return_value=Replay(200, {"answer": "saved"})))
    with pytest.raises(ValueError, match="REPLAY_MISMATCH"):
        await execute_case(
            cast(TurnStore, store),
            cast(ChatProcessor, processor),
            uuid4(),
            {"question": question, "client_request_id": str(uuid4())},
            RequestBudget(tmp_path / "budget.json"),
        )
    processor.run.assert_awaited_once()


def test_cli_requires_live_before_loading_key(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden() -> Any:
        pytest.fail("Missing-live CLI must not load key/settings")

    monkeypatch.setattr("scripts.evaluate_service_review.get_settings", forbidden)
    monkeypatch.setattr(
        "sys.argv", ["evaluate_service_review", "--run-id", "test", "--document-id", str(uuid4())]
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


@pytest.mark.parametrize("effort", ["none", "low"])
async def test_service_run_separates_planner_answer_and_review_profiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, effort: Literal["none", "low"]
) -> None:
    configured = Settings(
        _env_file=None,
        openai_api_key="test-placeholder",
        local_storage_dir=tmp_path,
        openai_answer_reasoning_effort=effort,
        openai_review_reasoning_effort="none",
    )
    engine = Mock(dispose=AsyncMock())
    artifact = tmp_path / "demo" / "service-review-effort-bound.json"

    async def preflight(*args: Any) -> None:
        saved = json.loads(artifact.read_text())
        assert saved["recipe"]["configuration"]["openai_answer_reasoning_effort"] == effort
        assert saved["recipe"]["review_prompt"] == CHAT_REVIEW_PROMPT_VERSION
        assert "saved_analysis_prompt" not in saved["recipe"]

    observed: list[ChatProcessor] = []

    async def execute(
        store: TurnStore,
        processor: ChatProcessor,
        session_id: Any,
        record: dict[str, Any],
        budget: RequestBudget,
    ) -> Replay:
        observed.append(processor)
        record.update(status="completed", turn_id=str(uuid4()))
        return Replay(200, {"evidence_status": "insufficient", "citations": []})

    monkeypatch.setattr(evaluate_service_review, "api_engine", lambda *args: engine)
    monkeypatch.setattr(evaluate_service_review, "preflight_and_create", preflight)
    monkeypatch.setattr(evaluate_service_review, "execute_case", execute)
    monkeypatch.setattr(evaluate_service_review, "capture_persisted", AsyncMock())
    result = await evaluate_service_review.run(configured, "effort-bound", uuid4(), "databases")
    assert len(observed) == 3
    processor = observed[0]
    planner = cast(AsyncBudgetedGenerationProvider, processor.generator)
    answerer = cast(AsyncBudgetedGenerationProvider, processor.answerer)
    reviewer = cast(AsyncBudgetedGenerationProvider, processor.reviewer)
    assert len({id(planner), id(answerer), id(reviewer)}) == 3
    assert (planner.label, planner.profile, planner.delegate.reasoning_effort) == (
        "service.planner",
        "none",
        "none",
    )
    assert (answerer.label, answerer.profile, answerer.delegate.reasoning_effort) == (
        "service.answer",
        effort,
        effort,
    )
    assert (reviewer.label, reviewer.profile, reviewer.delegate.reasoning_effort) == (
        "service.review",
        "none",
        "none",
    )
    assert result["used"] == 0
    assert json.loads(artifact.read_text())["status"] == "completed"
    engine.dispose.assert_awaited_once()
