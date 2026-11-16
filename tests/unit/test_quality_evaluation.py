"""Synthetic offline quality records exercise preparation/replay/review, never live AI."""

import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock
from uuid import uuid4

import httpx
import pytest

from app.adapters.extraction import DOCX_MEDIA_TYPE, extractor_for
from app.core.config import Settings
from app.prompts.analysis import ANALYSIS_PROMPT_VERSION
from scripts import evaluate_quality as quality
from scripts.evaluate_ai import EvaluationFailure, expected_versions
from scripts.evaluation_fixtures import CASES, SUITE_VERSION
from scripts.quality_fixtures import (
    DOCUMENT_KEY,
    EXPECTED_INTENTS,
    GUIDE_SECTIONS,
    GUIDE_TEXT,
    QUALITY_CASES,
    QUALITY_VERSION,
    guide_docx,
    quality_request_id,
)

pytestmark = pytest.mark.unit


def configured(root: Path) -> Settings:
    return Settings(_env_file=None, openai_api_key="offline-placeholder", local_storage_dir=root)


def runner(root: Path, revision: int = 1) -> quality.QualityRunner:
    return quality.QualityRunner(
        configured(root),
        Mock(spec=httpx.Client),
        Mock(),
        root / f"{QUALITY_VERSION}.json",
        revision=revision,
    )


def test_authored_guide_has_representative_narrative_and_qualified_facts(tmp_path: Path) -> None:
    extracted = extractor_for(DOCX_MEDIA_TYPE, configured(tmp_path)).extract(guide_docx())
    assert 800 <= len(extracted.text.split()) <= 1100
    assert extracted.text.index("Intake and acceptance") < extracted.text.index(
        "Manual review rules"
    )
    assert extracted.text.index("Retention and exceptions") < extracted.text.index(
        "Urgent acknowledgement"
    )
    for _heading, paragraph in GUIDE_SECTIONS:
        assert paragraph in extracted.text
    for fact in (
        "durable log",
        "sixteen routine requests",
        "twelve routine requests",
        "below 0.80",
        "greater than 1,000",
        "exactly 1,000",
        "ninety days after",
        "until the hold is released",
        "09:00 to 17:00 UTC on weekdays",
        "not a resolution guarantee",
        "Duplicate external email",
    ):
        assert fact in extracted.text


def test_quality_identities_and_case_scope_do_not_change_core_suite() -> None:
    assert len(CASES) == 20
    assert [case.id for case in QUALITY_CASES] == ["Q1", "Q2", "Q3", "Q4"]
    assert QUALITY_CASES[1].requires == ("Q1",)
    assert EXPECTED_INTENTS["Q3"] == "factual" and EXPECTED_INTENTS["Q4"] == "overview"
    assert not {case.id for case in QUALITY_CASES} & {case.id for case in CASES}
    assert quality_request_id("Q1") == quality_request_id("Q1", 1)
    assert quality_request_id("Q1", 1) != quality_request_id("Q1", 2)
    assert quality_request_id("upload") != quality_request_id("Q1")


def test_fixture_identity_survives_uncertain_upload_without_regeneration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = runner(tmp_path)
    first.prepare_fixture()
    planned = json.loads(first.path.read_text())["documents"][DOCUMENT_KEY]
    attempted: list[tuple[bytes, str]] = []

    def uncertain(method: str, path: str, **values: Any) -> tuple[int, dict[str, Any]]:
        assert method == "POST"
        assert json.loads(first.path.read_text())["documents"][DOCUMENT_KEY] == planned
        attempted.append((values["files"]["file"][1], values["headers"]["Idempotency-Key"]))
        raise httpx.ReadTimeout("Synthetic lost acknowledgement")

    monkeypatch.setattr(first, "request", uncertain)
    with pytest.raises(httpx.ReadTimeout):
        first.document(DOCUMENT_KEY)
    resumed = runner(tmp_path)
    resumed.prepare_fixture()
    identifier = str(uuid4())

    def accepted(method: str, path: str, **values: Any) -> tuple[int, dict[str, Any]]:
        if method == "POST":
            attempted.append((values["files"]["file"][1], values["headers"]["Idempotency-Key"]))
            return 202, {"document_id": identifier}
        return 200, {"status": "COMPLETED", "chunk_count": 2}

    monkeypatch.setattr(resumed, "request", accepted)
    assert resumed.document(DOCUMENT_KEY) == identifier
    assert attempted[0] == attempted[1]
    assert hashlib.sha256(attempted[0][0]).hexdigest() == planned["sha256"]


def test_changed_fixture_is_rejected_without_upload(tmp_path: Path) -> None:
    evaluation = runner(tmp_path)
    evaluation.prepare_fixture()
    fixture = tmp_path / evaluation.state["documents"][DOCUMENT_KEY]["filename"]
    fixture.write_bytes(b"tampered")
    with pytest.raises(EvaluationFailure, match="FIXTURE_BYTES_CHANGED"):
        evaluation.prepare_fixture()
    evaluation.client.request.assert_not_called()


def test_actual_base_evaluation_persists_quality_id_then_replays_without_new_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation = runner(tmp_path)
    document_id, session_id, source_id = str(uuid4()), str(uuid4()), str(uuid4())
    response = {
        "answer": "Accepted means durably recorded and queued.",
        "evidence_status": "answered",
        "citations": [{"source_id": source_id, "document_id": document_id}],
    }
    saved: dict[str, Any] = {}
    sent: list[str] = []
    monkeypatch.setattr(evaluation, "document", lambda key: document_id)
    monkeypatch.setattr(evaluation, "session", lambda key, identifier: session_id)
    monkeypatch.setattr(evaluation, "turn", lambda *args: saved or None)
    monkeypatch.setattr(
        evaluation,
        "snapshot",
        lambda *args: {
            "versions": expected_versions(configured(tmp_path)),
            "planner_intent": "factual",
            "membership_valid": True,
            "selected_membership_valid": True,
            "supplied_text": GUIDE_TEXT,
        },
    )

    def request(method: str, path: str, **values: Any) -> tuple[int, dict[str, Any]]:
        assert method == "POST" and path.endswith("/messages")
        record = json.loads(evaluation.path.read_text())["cases"]["Q1"]
        assert record["status"] == "PENDING"
        assert record["client_request_id"] == str(quality_request_id("Q1"))
        sent.append(values["json"]["client_request_id"])
        saved.update(status="COMPLETED", body=response, http_status=200)
        return 200, response

    monkeypatch.setattr(evaluation, "request", request)
    evaluation.evaluate(QUALITY_CASES[0])
    evaluation.evaluate(QUALITY_CASES[0])
    assert sent == [str(quality_request_id("Q1"))]
    assert evaluation.state["cases"]["Q1"]["response"] == response
    assert evaluation.state["suite"] == "quality"
    assert evaluation.state["version"] == SUITE_VERSION


def completed_state(tmp_path: Path) -> dict[str, Any]:
    source_id = str(uuid4())
    summary = (
        "Accepted intake is durably queued for asynchronous workers. Lease recovery handles "
        "crashes. Planning capacity is 16 per hour. Confidence below 0.80 and values above "
        "1,000 require manual review. Retention is 30 days after closure; active requests "
        "are excluded, disputes have 90 days after close and a legal hold prevents deletion. "
        "Urgent acknowledgement targets five minutes on staffed weekdays, "
        "not a resolution guarantee."
    )
    state: dict[str, Any] = {
        "version": SUITE_VERSION,
        "suite": "quality",
        "quality_version": QUALITY_VERSION,
        "expected_versions": expected_versions(configured(tmp_path)),
        "documents": {
            DOCUMENT_KEY: {
                "id": str(uuid4()),
                "sha256": "a" * 64,
                "definition_sha256": quality._digest(GUIDE_TEXT),
            }
        },
        "analysis": {
            "prompt_version": ANALYSIS_PROMPT_VERSION,
            "summary": [{"text": summary, "source_ids": [source_id]}],
            "insights": [
                {
                    "text": "High confidence cannot bypass mandatory value review.",
                    "source_ids": [source_id],
                }
            ],
        },
        "cases": {},
        "sessions": {},
    }
    answers = {
        "Q1": "An accepted request is recorded in a durable log and queued.",
        "Q2": "Acknowledgement precedes worker processing; complete means the work has finished.",
        "Q3": "Ordinary: 30 days after closure; active excluded; dispute: 90 after close; "
        "legal hold until released.",
        "Q4": "Intake accepts work, workers renew leases for recovery. Capacity is 16 per hour. "
        "Retention has exceptions, manual review applies, acknowledgement is not a "
        "resolution guarantee.",
    }
    for case in QUALITY_CASES:
        state["cases"][case.id] = {
            "status": "COMPLETED",
            "revision": 1,
            "client_request_id": str(quality_request_id(case.id)),
            "versions": expected_versions(configured(tmp_path)),
            "response": {
                "answer": answers[case.id],
                "evidence_status": "answered",
                "citations": [
                    {"source_id": source_id, "document_id": state["documents"][DOCUMENT_KEY]["id"]}
                ],
            },
            "evidence": {
                "planner_intent": EXPECTED_INTENTS[case.id],
                "membership_valid": True,
                "selected_membership_valid": True,
                "supplied_text": GUIDE_TEXT,
            },
        }
    return state


def review_data(state: dict[str, Any], reviewer_type: str = "assistant") -> dict[str, Any]:
    return {
        "reviewer_type": reviewer_type,
        "reviewer": "Synthetic offline reviewer",
        "binding": quality.review_binding(state),
        "checks": {
            key: {
                "result": "pass",
                "notes": "Synthetic check; no actual AI output was reviewed.",
                "source_refs": ["guide:Intake and acceptance"],
            }
            for key in quality.REQUIRED_REVIEWS
        },
    }


def test_stale_versions_block_implicit_calls_and_revision_archives_previous_result(
    tmp_path: Path,
) -> None:
    evaluation = runner(tmp_path)
    evaluation.state = completed_state(tmp_path)
    evaluation.state["case_history"] = {}
    original = evaluation.state["cases"]["Q1"]
    original["versions"]["answer_prompt"] = "answer-old"
    with pytest.raises(EvaluationFailure, match="STALE_EVIDENCE_EXPLICIT_REVISION_REQUIRED"):
        evaluation.preflight()
    evaluation.client.request.assert_not_called()
    evaluation.revision = 2
    evaluation.preflight()
    evaluation.activate_revision(QUALITY_CASES[0])
    assert evaluation.state["case_history"]["Q1"][0]["response"] == original["response"]
    assert evaluation.request_id(QUALITY_CASES[0]) == quality_request_id("Q1", 2)


def test_indicators_are_separate_from_source_review_and_human_approval(tmp_path: Path) -> None:
    state = completed_state(tmp_path)
    report = quality.quality_report(state)
    assert report["automatic_checks_passed"] is True
    assert report["source_review_passed"] is False and report["human_approval"] is False
    quality.record_review(state, review_data(state))
    report = quality.quality_report(state)
    assert report["source_review_passed"] is True and report["human_approval"] is False
    quality.record_review(state, review_data(state, "human"))
    assert quality.quality_report(state)["human_approval"] is True


def test_failed_review_records_failure_and_never_counts_as_approval(tmp_path: Path) -> None:
    state = completed_state(tmp_path)
    review = review_data(state, "human")
    review["checks"]["insight_source_support"]["result"] = "fail"
    quality.record_review(state, review)
    assert len(state["source_reviews"]) == 1
    assert quality.quality_report(state)["source_review_passed"] is False
    assert quality.quality_report(state)["human_approval"] is False


@pytest.mark.parametrize(
    "problem",
    ["missing_check", "wrong_sha", "stale_answer", "stale_analysis", "bad_ref", "changed_output"],
)
def test_review_rejects_unbound_incomplete_or_stale_evidence(tmp_path: Path, problem: str) -> None:
    state = completed_state(tmp_path)
    review = review_data(state)
    if problem == "missing_check":
        del review["checks"]["summary_themes"]
    elif problem == "wrong_sha":
        review["binding"]["document_sha256"] = "b" * 64
    elif problem == "stale_answer":
        state["cases"]["Q1"]["versions"]["answer_prompt"] = "old-answer"
    elif problem == "stale_analysis":
        state["analysis"]["prompt_version"] = "old-analysis"
    elif problem == "bad_ref":
        review["checks"]["summary_themes"]["source_refs"] = ["invented-source"]
    else:
        state["cases"]["Q1"]["response"]["answer"] = "Changed after review."
    with pytest.raises(EvaluationFailure):
        quality.record_review(state, review)
    assert not state.get("source_reviews")


def test_previous_review_becomes_uncredited_when_output_or_prompt_changes(tmp_path: Path) -> None:
    state = completed_state(tmp_path)
    quality.record_review(state, review_data(state))
    state["analysis"]["insights"][0]["text"] = "A replacement insight."
    assert quality.quality_report(state)["source_review_passed"] is False
    assert len(state["source_reviews"]) == 1


def test_later_failed_review_prevents_old_approval_from_masking_failure(tmp_path: Path) -> None:
    state = completed_state(tmp_path)
    quality.record_review(state, review_data(state, "human"))
    later = review_data(state, "human")
    later["checks"]["citation_support"]["result"] = "fail"
    quality.record_review(state, later)
    report = quality.quality_report(state)
    assert report["source_review_passed"] is False and report["human_approval"] is False
    assert len(state["source_reviews"]) == 2


def test_core_artifact_cannot_be_overwritten_as_quality(tmp_path: Path) -> None:
    path = tmp_path / f"{QUALITY_VERSION}.json"
    original = json.dumps(
        {"version": SUITE_VERSION, "suite": "core", "cases": {}, "documents": {}, "sessions": {}}
    )
    path.write_text(original)
    with pytest.raises(EvaluationFailure, match="NOT_A_QUALITY_ARTIFACT"):
        runner(tmp_path)
    assert path.read_text() == original


def test_copy_and_duplicate_insight_flags_do_not_claim_semantic_support() -> None:
    text = GUIDE_SECTIONS[0][1]
    indicators = quality.analysis_indicators(
        {
            "summary": [{"text": text}],
            "insights": [{"text": text}, {"text": text}],
        }
    )
    assert indicators["insights_textually_distinct"] is False
    assert indicators["verbatim_block_flags"] == [0, 1, 2]
    assert indicators["semantic_support_established"] is False


def test_focused_retention_routing_is_a_quality_check(tmp_path: Path) -> None:
    state = completed_state(tmp_path)
    case = QUALITY_CASES[2]
    state["cases"]["Q3"]["evidence"]["planner_intent"] = "overview"
    checks = quality.case_checks(case, state["cases"]["Q3"])
    assert checks["expected_facts"] is True
    assert checks["automatic_pass"] is False


def test_default_and_review_commands_never_construct_live_settings_or_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    forbidden = Mock(side_effect=AssertionError("No live dependency in read-only mode"))
    monkeypatch.setattr(quality, "get_settings", forbidden)
    monkeypatch.setattr(quality, "run_live", forbidden)
    path = tmp_path / "quality.json"
    assert quality.main(["--state-file", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["completed_chat_cases"] == 0
    state = completed_state(tmp_path)
    path.write_text(json.dumps(state))
    review = tmp_path / "review.json"
    review.write_text(json.dumps(review_data(state)))
    assert quality.main(["--state-file", str(path), "--record-review", str(review)]) == 0
    assert json.loads(capsys.readouterr().out)["human_approval"] is False
    forbidden.assert_not_called()


def test_review_template_cannot_overwrite_the_evidence_artifact(tmp_path: Path) -> None:
    path = tmp_path / "quality.json"
    path.write_text(json.dumps(completed_state(tmp_path)))
    original = path.read_bytes()
    with pytest.raises(SystemExit):
        quality.main(["--state-file", str(path), "--review-template", str(path)])
    assert path.read_bytes() == original


def completed_runner(tmp_path: Path) -> quality.QualityRunner:
    evaluation = runner(tmp_path)
    evaluation.prepare_fixture()
    fixture_record = dict(evaluation.state["documents"][DOCUMENT_KEY])
    evaluation.state = completed_state(tmp_path)
    identifier = evaluation.state["documents"][DOCUMENT_KEY]["id"]
    evaluation.state["documents"][DOCUMENT_KEY] = {
        **fixture_record,
        "id": identifier,
        "status": "COMPLETED",
    }
    evaluation.state["case_history"] = {}
    evaluation.state["sessions"][f"{DOCUMENT_KEY}:revision:1"] = str(uuid4())
    failed_review = review_data(evaluation.state)
    failed_review["checks"]["distinct_insights"]["result"] = "fail"
    quality.record_review(evaluation.state, failed_review)
    evaluation.save()
    return evaluation


def test_changed_analysis_same_revision_refuses_without_mutation_or_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation = completed_runner(tmp_path)
    original = evaluation.path.read_bytes()
    monkeypatch.setattr(quality, "ANALYSIS_PROMPT_VERSION", "analysis-next")
    with pytest.raises(EvaluationFailure, match="SAVED_ANALYSIS_STALE"):
        evaluation.preflight()
    with pytest.raises(EvaluationFailure, match="SAVED_ANALYSIS_STALE"):
        evaluation.prepare_analysis_iteration()
    assert evaluation.path.read_bytes() == original
    evaluation.client.request.assert_not_called()


def test_higher_analysis_iteration_archives_all_evidence_before_planned_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation = completed_runner(tmp_path)
    original = json.loads(evaluation.path.read_text())
    old_document = original["documents"][DOCUMENT_KEY]
    fixture = tmp_path / old_document["filename"]
    content = fixture.read_bytes()
    monkeypatch.setattr(quality, "ANALYSIS_PROMPT_VERSION", "analysis-next")
    evaluation.revision = 2
    evaluation.preflight()
    evaluation.prepare_analysis_iteration()
    saved = json.loads(evaluation.path.read_text())
    planned = saved["documents"][DOCUMENT_KEY]
    assert planned["status"] == "PLANNED" and "id" not in planned
    assert planned["upload_request_id"] == str(quality_request_id("upload", 2))
    assert planned["revision"] == 2 and planned["analysis_prompt"] == "analysis-next"
    assert planned["sha256"] == old_document["sha256"] and fixture.read_bytes() == content
    assert saved["document_history"][DOCUMENT_KEY][0]["id"] == old_document["id"]
    assert saved["analysis_history"][DOCUMENT_KEY][0]["analysis"] == original["analysis"]
    assert "analysis" not in saved and saved["cases"] == {}
    for case in QUALITY_CASES:
        assert saved["case_history"][case.id][0] == original["cases"][case.id]
    assert saved["source_reviews"] == original["source_reviews"]
    assert saved["sessions"] == original["sessions"]
    assert saved["report"]["completed_chat_cases"] == 0
    assert saved["report"]["source_review_passed"] is False
    evaluation.client.request.assert_not_called()


def test_new_chat_revision_with_unchanged_analysis_keeps_document_and_analysis(
    tmp_path: Path,
) -> None:
    evaluation = completed_runner(tmp_path)
    before = evaluation.path.read_bytes()
    evaluation.revision = 2
    evaluation.preflight()
    evaluation.prepare_analysis_iteration()
    assert evaluation.path.read_bytes() == before
    assert "document_history" not in evaluation.state
    assert evaluation.state["analysis"]["prompt_version"] == ANALYSIS_PROMPT_VERSION
    evaluation.client.request.assert_not_called()


def test_analysis_iteration_uncertain_upload_resumes_same_identity_and_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation = completed_runner(tmp_path)
    monkeypatch.setattr(quality, "ANALYSIS_PROMPT_VERSION", "analysis-next")
    evaluation.revision = 2
    evaluation.prepare_analysis_iteration()
    planned = json.loads(evaluation.path.read_text())["documents"][DOCUMENT_KEY]
    attempts: list[tuple[bytes, str]] = []

    def uncertain(method: str, path: str, **values: Any) -> tuple[int, dict[str, Any]]:
        assert method == "POST"
        durable = json.loads(evaluation.path.read_text())
        assert durable["documents"][DOCUMENT_KEY] == planned
        assert durable["analysis_history"][DOCUMENT_KEY][0]["analysis"]
        attempts.append((values["files"]["file"][1], values["headers"]["Idempotency-Key"]))
        raise httpx.ReadTimeout("Synthetic lost revision-two acknowledgement")

    monkeypatch.setattr(evaluation, "request", uncertain)
    with pytest.raises(httpx.ReadTimeout):
        evaluation.document(DOCUMENT_KEY)
    resumed = runner(tmp_path, revision=2)
    resumed.preflight()
    resumed.prepare_fixture()
    resumed.prepare_analysis_iteration()
    new_id = str(uuid4())

    def accepted(method: str, path: str, **values: Any) -> tuple[int, dict[str, Any]]:
        if method == "POST":
            attempts.append((values["files"]["file"][1], values["headers"]["Idempotency-Key"]))
            return 202, {"document_id": new_id}
        return 200, {"status": "COMPLETED", "chunk_count": 2}

    monkeypatch.setattr(resumed, "request", accepted)
    assert resumed.document(DOCUMENT_KEY) == new_id
    assert attempts[0] == attempts[1]
    assert attempts[0][1] == str(quality_request_id("upload", 2))
    assert len(resumed.state["document_history"][DOCUMENT_KEY]) == 1
    assert len(resumed.state["analysis_history"][DOCUMENT_KEY]) == 1
    assert len(resumed.state["source_reviews"]) == 1


@pytest.mark.parametrize("obstacle", ["pending", "higher_history", "tampered_fixture"])
def test_analysis_iteration_rejects_unresolved_or_unsafe_prior_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, obstacle: str
) -> None:
    evaluation = completed_runner(tmp_path)
    evaluation.revision = 2
    if obstacle == "pending":
        evaluation.state["cases"]["Q1"]["status"] = "PENDING"
    elif obstacle == "higher_history":
        evaluation.state["case_history"]["Q1"] = [{"revision": 3}]
    else:
        document = evaluation.state["documents"][DOCUMENT_KEY]
        (tmp_path / document["filename"]).write_bytes(b"unexpected changed source")
    original = json.loads(json.dumps(evaluation.state))
    monkeypatch.setattr(quality, "ANALYSIS_PROMPT_VERSION", "analysis-next")
    with pytest.raises(EvaluationFailure):
        evaluation.prepare_analysis_iteration()
    assert evaluation.state == original
    evaluation.client.request.assert_not_called()


def test_lower_revision_cannot_send_a_planned_newer_analysis_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation = completed_runner(tmp_path)
    monkeypatch.setattr(quality, "ANALYSIS_PROMPT_VERSION", "analysis-next")
    evaluation.revision = 2
    evaluation.prepare_analysis_iteration()
    lower = runner(tmp_path, revision=1)
    with pytest.raises(EvaluationFailure, match="QUALITY_REVISION_BEHIND_DOCUMENT"):
        lower.preflight()
    lower.client.request.assert_not_called()


def test_report_only_higher_revision_does_not_prepare_analysis_iteration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    evaluation = completed_runner(tmp_path)
    original = evaluation.path.read_bytes()
    forbidden = Mock(side_effect=AssertionError("A report must not start a live iteration"))
    monkeypatch.setattr(quality, "run_live", forbidden)
    monkeypatch.setattr(quality, "ANALYSIS_PROMPT_VERSION", "analysis-next")
    assert quality.main(["--state-file", str(evaluation.path), "--revision", "2"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert "analysis_prompt_changed" in report["stale_reasons"]
    assert report["automatic_checks_passed"] is False
    assert evaluation.path.read_bytes() == original
    forbidden.assert_not_called()


@pytest.mark.parametrize("latest_in", ["document", "active_case", "case_history", "pending_case"])
def test_cli_omitted_live_revision_resumes_latest_without_restoring_revision_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    latest_in: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = completed_state(tmp_path)
    if latest_in == "document":
        state["documents"][DOCUMENT_KEY]["revision"] = 3
    elif latest_in == "case_history":
        state["case_history"] = {"Q1": [{"revision": 3}]}
    else:
        state["cases"]["Q1"]["revision"] = 3
        if latest_in == "pending_case":
            state["cases"]["Q1"]["status"] = "PENDING"
    path = tmp_path / "quality.json"
    path.write_text(json.dumps(state))
    original = path.read_bytes()
    live = Mock(return_value={"requested_revision": 3})
    monkeypatch.setattr(quality, "run_live", live)
    assert quality.main(["--live", "--state-file", str(path)]) == 0
    live.assert_called_once_with(path, 3)
    assert json.loads(capsys.readouterr().out)["requested_revision"] == 3
    assert path.read_bytes() == original


def test_cli_fresh_omitted_live_revision_starts_at_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "new-quality.json"
    live = Mock(return_value={"requested_revision": 1})
    monkeypatch.setattr(quality, "run_live", live)
    assert quality.main(["--live", "--state-file", str(path)]) == 0
    live.assert_called_once_with(path, 1)
    capsys.readouterr()
    assert not path.exists()


def test_cli_explicit_lower_revision_refuses_before_any_live_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    state = completed_state(tmp_path)
    state["documents"][DOCUMENT_KEY]["revision"] = 3
    for record in state["cases"].values():
        record["revision"] = 3
    path = tmp_path / "quality.json"
    path.write_text(json.dumps(state))
    original = path.read_bytes()
    forbidden = Mock(side_effect=AssertionError("No configuration, HTTP or paid work on downgrade"))
    monkeypatch.setattr(quality, "get_settings", forbidden)
    monkeypatch.setattr(quality, "create_engine", forbidden)
    monkeypatch.setattr(quality.httpx, "Client", forbidden)
    monkeypatch.setattr(quality, "run_live", forbidden)
    assert quality.main(["--live", "--state-file", str(path), "--revision", "1"]) == 1
    assert json.loads(capsys.readouterr().out)["error"] == "QUALITY_REVISION_BEHIND_CURRENT"
    forbidden.assert_not_called()
    assert path.read_bytes() == original


def test_runner_refuses_lower_chat_revision_even_when_document_is_still_revision_one(
    tmp_path: Path,
) -> None:
    evaluation = completed_runner(tmp_path)
    evaluation.state["cases"]["Q1"]["revision"] = 3
    with pytest.raises(EvaluationFailure, match="QUALITY_REVISION_BEHIND_CURRENT"):
        evaluation.preflight()
    evaluation.client.request.assert_not_called()
