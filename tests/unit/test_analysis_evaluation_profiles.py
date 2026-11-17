"""Offline tool checks retain role separation and durable failed-request identity."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from app.core.config import Settings
from app.domain.documents import ProcessingError
from app.services.analysis_evidence import QuoteReference
from scripts import evaluate_insight_candidates, evaluate_review
from scripts.analysis_evaluation_profile import expected_review_profile


def isolated_settings(root: Path) -> Settings:
    return Settings(_env_file=None, openai_api_key="test-placeholder", local_storage_dir=root)


@pytest.mark.parametrize("chat", [False, True])
def test_frozen_review_uses_actual_role_profile_and_failure_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chat: bool
) -> None:
    settings = isolated_settings(tmp_path)
    monkeypatch.setattr(evaluate_review, "get_settings", lambda: settings)
    monkeypatch.setattr(evaluate_review, "create_engine", lambda *a, **k: MagicMock())
    monkeypatch.setattr(evaluate_review, "Session", lambda *a: MagicMock())
    kind = "answer" if chat else "summary"
    draft = {kind: [{"text": "A released hold does not reset the retention clock."}]}
    if not chat:
        draft["insights"] = []
    monkeypatch.setattr(
        evaluate_review,
        "load_case",
        lambda *a: (
            draft,
            {"q1": QuoteReference(UUID(int=1), draft[kind][0]["text"])},
            "What changes on release?" if chat else None,
        ),
    )
    calls: list[dict[str, Any]] = []

    class FailedProvider:
        def __init__(self, configured: Settings, **profile: Any) -> None:
            self.profile = profile

        def generate(self, **request: Any) -> Any:
            calls.append({"profile": self.profile, "request": request})
            raise ProcessingError("synthetic_provider_failure", "Synthetic unavailable response")

    monkeypatch.setattr(evaluate_review, "OpenAIGenerationProvider", FailedProvider)
    effort = "none" if chat else "low"
    evaluate_review.run(["narrative"], effort, "profile-test")
    path = tmp_path / "demo" / "review-experiment-profile-test.json"
    before = path.read_bytes()
    evaluate_review.run(["narrative"], effort, "profile-test")
    assert len(calls) == 1 and path.read_bytes() == before
    expected_model = settings.openai_chat_model if chat else settings.openai_analysis_review_model
    assert calls[0]["profile"] == {"model": expected_model, "reasoning_effort": effort}
    saved = json.loads(before)["cases"][f"narrative:{effort}"]
    assert saved["request"]["model"] == expected_model
    assert saved["request"]["reasoning_effort"] == effort
    if not chat:
        assert saved["request"]["review_profile"] == expected_review_profile(settings)
        assert json.loads(saved["request"]["content"])["insight_candidates"] == []
    saved["request"]["model"] = "historical-model"
    saved["request_sha256"] = evaluate_review.fingerprint(saved["request"])
    altered = {"run_id": "profile-test", "cases": {f"narrative:{effort}": saved}}
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="EXPERIMENT_RECIPE_CHANGED"):
        evaluate_review.run(["narrative"], effort, "profile-test")
    assert len(calls) == 1


def test_focused_insight_audit_uses_manifested_bound_schema_and_pinned_reviewer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = isolated_settings(tmp_path)
    module = evaluate_insight_candidates
    monkeypatch.setattr(module, "get_settings", lambda: settings)
    monkeypatch.setattr(module, "create_engine", lambda *a, **k: MagicMock())
    monkeypatch.setattr(module, "Session", lambda *a: MagicMock())
    generation = {"insights": []}
    directory = tmp_path / "demo"
    directory.mkdir()
    path = directory / "insight-experiment-audit-test.json"
    path.write_text(json.dumps({"cases": {"held-out": {"generation": generation}}}))
    calls: list[dict[str, Any]] = []

    class FailedProvider:
        def __init__(self, configured: Settings, **profile: Any) -> None:
            self.profile = profile

        def generate(self, **request: Any) -> Any:
            calls.append({"profile": self.profile, "request": request})
            raise ProcessingError("synthetic_provider_failure", "Synthetic unavailable response")

    monkeypatch.setattr(module, "OpenAIGenerationProvider", FailedProvider)
    module.run(["held-out"], "audit-test", audit_only=True, audit_revision=2)
    assert len(calls) == 1
    assert calls[0]["profile"] == {
        "model": settings.openai_analysis_review_model,
        "reasoning_effort": "low",
    }
    content = json.loads(calls[0]["request"]["content"])
    assert content["insight_candidates"] == []
    schema = calls[0]["request"]["schema"].model_json_schema()
    assert schema["properties"]["insight_assessments"]["maxItems"] == 0
    saved = json.loads(path.read_text())
    assert saved["cases"]["held-out:audit-2"]["review_profile"] == expected_review_profile()
    assert saved["cases"]["held-out"]["generation"] == generation
    module.run(["held-out"], "audit-test", audit_only=True, audit_revision=2)
    assert len(calls) == 1
