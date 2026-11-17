"""Offline evidence for isolated, resumable evaluation runs; no provider calls."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, Mock
from uuid import uuid4

import httpx
import pytest

from app.core.config import Settings
from app.db.models import Document
from scripts import evaluate_ai
from scripts.evaluate_ai import EvaluationFailure, EvaluationRunner
from scripts.evaluation_fixtures import BY_ID, case_request_id, upload_request_id


def make_runner(root: Path, run_id: str | None, *, revision: int = 1) -> EvaluationRunner:
    (root / "demo").mkdir(parents=True, exist_ok=True)
    settings = Settings(_env_file=None, openai_api_key="test-placeholder", local_storage_dir=root)
    return EvaluationRunner(
        settings,
        Mock(spec=httpx.Client),
        Mock(),
        root / "demo" / f"state-{run_id or 'legacy'}.json",
        revision=revision,
        run_id=run_id,
    )


@pytest.mark.parametrize("value", ["../escape", "UPPER", "", "a/b", "x" * 41, "a.b"])
def test_invalid_run_name_cannot_become_a_path_or_namespace(value: str) -> None:
    with pytest.raises(EvaluationFailure):
        evaluate_ai.validate_run_id(value)


def test_run_names_and_legacy_question_identities_are_separate(tmp_path: Path) -> None:
    first = make_runner(tmp_path, "pilot-a")
    other = make_runner(tmp_path, "pilot-b")
    legacy = make_runner(tmp_path, None)
    assert evaluate_ai.validate_run_id(None) is None
    assert evaluate_ai.validate_run_id("pilot-2026") == "pilot-2026"
    assert len({runner.request_id(BY_ID["A1"]) for runner in (first, other, legacy)}) == 3
    assert legacy.request_id(BY_ID["A1"]) == case_request_id("A1")
    first.save()
    resumed = make_runner(tmp_path, "pilot-a")
    assert resumed.request_id(BY_ID["A1"]) == first.request_id(BY_ID["A1"])
    assert resumed.request_id(BY_ID["A2"]) != resumed.request_id(BY_ID["A1"])
    for runner in (first, other, legacy, resumed):
        runner.client.request.assert_not_called()


def test_changed_recipe_refuses_run_resume_before_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = make_runner(tmp_path, "frozen")
    original.save()
    original_bytes = original.path.read_bytes()
    expected_versions = evaluate_ai.expected_versions

    def changed(settings: Settings) -> dict[str, Any]:
        return {**expected_versions(settings), "answer_prompt": "answer-new-test-version"}

    monkeypatch.setattr(evaluate_ai, "expected_versions", changed)
    with pytest.raises(EvaluationFailure):
        make_runner(tmp_path, "frozen")
    assert original.path.read_bytes() == original_bytes
    original.client.request.assert_not_called()


def test_answer_effort_change_invalidates_run_recipe_without_altering_saved_evidence(
    tmp_path: Path,
) -> None:
    original = make_runner(tmp_path, "effort-bound")
    original.save()
    original_bytes = original.path.read_bytes()
    observed = original.state["recipe"]["versions"]["configuration"]
    assert observed["openai_answer_reasoning_effort"] == "low"
    changed = Settings(
        _env_file=None,
        openai_api_key="test-placeholder",
        local_storage_dir=tmp_path,
        openai_answer_reasoning_effort="none",
    )
    with pytest.raises(EvaluationFailure, match="RUN_RECIPE_CHANGED_USE_NEW_RUN_ID"):
        EvaluationRunner(
            changed, original.client, original.engine, original.path, run_id="effort-bound"
        )
    assert original.path.read_bytes() == original_bytes
    cast(Mock, original.client).request.assert_not_called()
    cast(Mock, original.engine).assert_not_called()


def test_factual_evidence_requires_its_actual_source_model_and_stage_versions() -> None:
    settings = Settings(_env_file=None, openai_api_key="test-placeholder")
    expected = evaluate_ai.expected_versions(settings)
    factual = {
        "versions": evaluate_ai.expected_versions(settings, intent="factual"),
        "evidence": {"planner_intent": "factual"},
    }
    assert evaluate_ai.stale_reasons(factual, expected) == []
    factual["versions"]["chat_model"] = settings.openai_chat_model
    assert "chat_model_changed" in evaluate_ai.stale_reasons(factual, expected)
    factual["versions"] = expected
    assert "source_composition_missing" in evaluate_ai.stale_reasons(factual, expected)
    overview = {"versions": expected, "evidence": {"planner_intent": "overview"}}
    assert evaluate_ai.stale_reasons(overview, expected) == []


def test_namespaced_run_refuses_revision_mix(tmp_path: Path) -> None:
    with pytest.raises(EvaluationFailure):
        make_runner(tmp_path, "frozen", revision=2)


def test_uploads_and_sessions_are_isolated_and_resume_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uploads: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []

    def request(method: str, path: str, **kwargs: Any) -> tuple[int, dict[str, Any]]:
        if method == "POST" and path.endswith("/upload"):
            uploads.append(kwargs)
            return 202, {"document_id": str(uuid4()), "status": "PENDING"}
        if method == "GET" and path.endswith("/insights"):
            return 200, {
                "prompt_version": evaluate_ai.ANALYSIS_PROMPT_VERSION,
                "model": "gpt-5.4-mini-2026-03-17",
            }
        if method == "GET" and path.startswith("/api/v1/documents/"):
            return 200, {"status": "COMPLETED", "chunk_count": 1}
        if method == "GET" and path == "/api/v1/chat/sessions":
            return 200, {"items": [], "next_cursor": None}
        if method == "POST" and path == "/api/v1/chat/sessions":
            sessions.append(kwargs["json"])
            return 201, {"id": str(uuid4())}
        pytest.fail(f"Unexpected evaluation request: {method} {path}")

    for run_id in ("pilot-a", "pilot-b"):
        runner = make_runner(tmp_path, run_id)
        monkeypatch.setattr(runner, "request", request)
        document = runner.document("juniper")
        session = runner.session("juniper", document)
        resumed = make_runner(tmp_path, run_id)
        monkeypatch.setattr(resumed, "request", request)
        assert resumed.document("juniper") == document
        assert resumed.session("juniper", document) == session
    assert len(uploads) == len(sessions) == 2
    keys = {upload["headers"]["Idempotency-Key"] for upload in uploads}
    assert len(keys) == 2
    assert str(upload_request_id("juniper")) not in keys
    assert len({session["title"] for session in sessions}) == 2
    assert all("pilot-" in session["title"] for session in sessions)


@pytest.mark.parametrize("intent", ["factual", "overview"])
@pytest.mark.parametrize("provenance", ["matching", "missing", "other_attempt"])
def test_freshness_uses_actual_published_analysis_not_current_constant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, intent: str, provenance: str
) -> None:
    runner = make_runner(tmp_path, "analysis-check")
    document = SimpleNamespace(id=uuid4(), published_attempt_id=uuid4(), profile_id=uuid4())
    turn = SimpleNamespace(
        id=uuid4(),
        context_snapshot={"selected_ids": [], "payload": {"evidence": []}},
        planner_intent=intent,
        retrieval_query="A question",
        versions={"answer_prompt": "observed-answer-version"},
        timings={},
        ai_calls_reserved=2,
    )
    analysis = (
        None
        if provenance == "missing"
        else SimpleNamespace(
            attempt_id=(
                uuid4() if provenance == "other_attempt" else document.published_attempt_id
            ),
            prompt_version="analysis-historical-test-version",
        )
    )
    database = MagicMock()
    database.__enter__.return_value = database
    database.scalar.return_value = turn
    database.get.side_effect = lambda model, *args, **kwargs: (
        document if model is Document else analysis
    )
    database.scalars.return_value.all.return_value = []
    monkeypatch.setattr(evaluate_ai, "Session", lambda engine: database)
    snapshot = runner.snapshot(str(uuid4()), BY_ID["A1"], str(document.id), {"citations": []})
    observed = snapshot["versions"]["saved_analysis_prompt"]
    assert observed == ("analysis-historical-test-version" if provenance == "matching" else None)
    expected = runner.state["expected_versions"]
    assert "saved_analysis_prompt" in expected
    reasons = evaluate_ai.stale_reasons({"versions": snapshot["versions"]}, expected)
    assert "saved_analysis_prompt_changed" in reasons
    assert "saved_analysis_prompt" not in turn.versions
    assert json.loads(json.dumps(snapshot))["versions"]["saved_analysis_prompt"] == observed
    runner.client.request.assert_not_called()


def test_namespaced_conversations_preserve_only_intended_followup_context(tmp_path: Path) -> None:
    runner = make_runner(tmp_path, "chains")
    assert runner.session_key(BY_ID["A1"]) == runner.session_key(BY_ID["F1"])
    assert runner.session_key(BY_ID["A2"]) == runner.session_key(BY_ID["F3"])
    assert runner.session_key(BY_ID["A3"]) == runner.session_key(BY_ID["F2"])
    assert len({runner.session_key(BY_ID[key]) for key in ("A1", "A2", "A3", "U1")}) == 4
    assert len({runner.session_key(BY_ID[key]) for key in ("A4", "A5", "A6", "F4")}) == 1
    legacy = make_runner(tmp_path, None)
    assert all(legacy.session_key(BY_ID[key]) == "juniper" for key in ("A1", "A2", "U1"))


def test_insufficient_behavior_does_not_mask_failed_isolation_or_attack_checks() -> None:
    body = {
        "answer": "A cancellation penalty is not specified in the document.",
        "evidence_status": "insufficient",
        "citations": [],
    }
    clean = evaluate_ai.automatic_checks(BY_ID["U1"], body, {"membership_valid": True})
    assert clean["automatic_pass"] is True
    invalid = evaluate_ai.automatic_checks(BY_ID["U1"], body, {"membership_valid": False})
    assert invalid["automatic_pass"] is False


@pytest.mark.parametrize("missing", ["recipe", "run_id"])
def test_existing_run_without_provenance_cannot_silently_adopt_current_recipe(
    tmp_path: Path, missing: str
) -> None:
    runner = make_runner(tmp_path, "provenance")
    runner.save()
    state = json.loads(runner.path.read_text())
    del state[missing]
    runner.path.write_text(json.dumps(state))
    saved = runner.path.read_bytes()
    with pytest.raises(EvaluationFailure):
        make_runner(tmp_path, "provenance")
    assert runner.path.read_bytes() == saved
    runner.client.request.assert_not_called()


@pytest.mark.parametrize("status", ["FAILED", "COMPLETED", "PENDING"])
def test_isolated_run_stops_after_first_failed_or_unresolved_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    runner = make_runner(tmp_path, "stop-on-failure")
    evaluated: list[str] = []

    def evaluate(case: Any) -> None:
        evaluated.append(case.id)
        runner.state["cases"][case.id] = {
            "status": status,
            "checks": {"automatic_pass": False},
        }

    monkeypatch.setattr(runner, "evaluate", evaluate)
    monkeypatch.setattr(evaluate_ai, "get_settings", lambda: runner.settings)
    monkeypatch.setattr(evaluate_ai, "EvaluationRunner", lambda *args, **kwargs: runner)
    monkeypatch.setattr(evaluate_ai, "create_engine", lambda *args, **kwargs: Mock())
    client = MagicMock()
    monkeypatch.setattr(evaluate_ai.httpx, "Client", lambda *args, **kwargs: client)
    report = evaluate_ai.run([BY_ID["A1"], BY_ID["U1"]], run_id="stop-on-failure")
    assert evaluated == ["A1"]
    assert report["selected_automatic_pass"] is False
    client.request.assert_not_called()


@pytest.mark.parametrize("field", ["prompt_version", "model"])
def test_old_analysis_is_rejected_before_creating_a_conversation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    runner = make_runner(tmp_path, "old-analysis")
    runner.state["documents"]["juniper"] = {"id": str(uuid4()), "status": "COMPLETED"}
    called: list[str] = []

    def request(method: str, path: str, **kwargs: Any) -> tuple[int, dict[str, Any]]:
        assert method == "GET"
        called.append(path)
        if path.endswith("/insights"):
            result = {
                "prompt_version": evaluate_ai.ANALYSIS_PROMPT_VERSION,
                "model": runner.settings.openai_chat_model,
            }
            result[field] = "old-version"
            return 200, result
        return 200, {"status": "COMPLETED", "chunk_count": 1}

    monkeypatch.setattr(runner, "request", request)
    with pytest.raises(EvaluationFailure, match="DOCUMENT_ANALYSIS_STALE"):
        runner.document("juniper")
    assert len(called) == 2
    assert not runner.state["sessions"]
    runner.client.request.assert_not_called()
