"""Core synthesis/explanation evaluation; default and review commands never call AI."""

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import create_engine

from app.core.config import get_settings
from app.prompts.analysis import ANALYSIS_PROMPT_VERSION
from app.prompts.chat import ANSWER_PROMPT_VERSION, CHAT_PIPELINE_VERSION, PLANNER_PROMPT_VERSION
from scripts.evaluate_ai import (
    EvaluationFailure,
    EvaluationRunner,
    _atomic_write,
    _now,
    automatic_checks,
    evaluation_lock,
    load_state,
    matches,
    stale_reasons,
)
from scripts.evaluation_fixtures import SUITE_VERSION, EvaluationCase
from scripts.live_ingestion import BASE_URL
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

REQUIRED_REVIEWS = (
    "summary_themes",
    "summary_qualifications",
    "distinct_insights",
    "insight_source_support",
    "concept_explanation",
    "contextual_why",
    "focused_retention_scope",
    "overview_synthesis",
    "citation_support",
)
THEMES = {
    "intake": (r"accept|intake|durable|queue",),
    "workers": (r"worker|asynchronous",),
    "recovery": (r"recover|lease|crash|interrupted",),
    "capacity": (r"capacity|throughput|sixteen|\b16\b",),
    "retention": (r"retention|retain|thirty|\b30\b",),
}
QUALIFICATIONS = {
    "manual_review": (r"manual", r"0[.]80?|80\s*%|eighty percent", r"1,?000|thousand"),
    "retention_exceptions": (r"active", r"ninety|\b90\b", r"hold", r"clos"),
    "urgent_target": (
        r"acknowledg",
        r"five|\b5\b",
        r"weekday|staffed|09:00|09[–-]17",
        r"(?:not|no|without|rather than|isn't).{0,65}(?:resolution|guarantee|complet)",
    ),
}


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _current_versions(state: dict[str, Any]) -> dict[str, Any]:
    return {
        **state.get("expected_versions", {}),
        "planner_prompt": PLANNER_PROMPT_VERSION,
        "answer_prompt": ANSWER_PROMPT_VERSION,
        "pipeline": CHAT_PIPELINE_VERSION,
    }


def analysis_indicators(analysis: dict[str, Any]) -> dict[str, Any]:
    """Observable heuristics only; these cannot establish semantic source support."""
    summary = [str(block.get("text", "")) for block in analysis.get("summary", [])]
    insights = [str(block.get("text", "")) for block in analysis.get("insights", [])]
    combined = " ".join(summary)
    normalized_summary = {" ".join(text.lower().split()) for text in summary}
    normalized_insights = [" ".join(text.lower().split()) for text in insights]
    normalized_guide = " ".join(GUIDE_TEXT.lower().split())
    copied = [
        index
        for index, text in enumerate([*summary, *insights])
        if len(text.split()) >= 12 and " ".join(text.lower().split()) in normalized_guide
    ]
    return {
        "summary_themes": {key: matches(patterns, combined) for key, patterns in THEMES.items()},
        "summary_qualifications": {
            key: matches(patterns, combined) for key, patterns in QUALIFICATIONS.items()
        },
        "insights_present": bool(normalized_insights),
        "insights_textually_distinct": bool(normalized_insights)
        and len(set(normalized_insights)) == len(normalized_insights)
        and all(text and text not in normalized_summary for text in normalized_insights),
        "verbatim_block_flags": copied,
        "semantic_support_established": False,
    }


def case_checks(case: EvaluationCase, record: dict[str, Any]) -> dict[str, Any]:
    checks = automatic_checks(case, record.get("response", {}), record.get("evidence", {}))
    checks["planner_intent_expected"] = (
        record.get("evidence", {}).get("planner_intent") == EXPECTED_INTENTS[case.id]
    )
    checks["automatic_pass"] &= checks["planner_intent_expected"]
    return checks


def _stale(state: dict[str, Any]) -> list[str]:
    reasons = []
    document = state.get("documents", {}).get(DOCUMENT_KEY)
    if document and document.get("definition_sha256") != _digest(GUIDE_TEXT):
        reasons.append("fixture_definition_changed")
    analysis = state.get("analysis", {})
    if analysis and analysis.get("prompt_version") != ANALYSIS_PROMPT_VERSION:
        reasons.append("analysis_prompt_changed")
    for case in QUALITY_CASES:
        record = state.get("cases", {}).get(case.id, {})
        if record.get("status") == "COMPLETED":
            reasons.extend(
                f"{case.id}:{reason}" for reason in stale_reasons(record, _current_versions(state))
            )
    return reasons


def review_binding(state: dict[str, Any]) -> dict[str, Any]:
    if _stale(state):
        raise EvaluationFailure("SOURCE_REVIEW_STALE_VERSIONS")
    records = state.get("cases", {})
    if not state.get("analysis") or any(
        records.get(case.id, {}).get("status") != "COMPLETED" for case in QUALITY_CASES
    ):
        raise EvaluationFailure("SOURCE_REVIEW_EVIDENCE_INCOMPLETE")
    document = state["documents"][DOCUMENT_KEY]
    return {
        "quality_version": QUALITY_VERSION,
        "document_sha256": document["sha256"],
        "analysis_prompt": state["analysis"]["prompt_version"],
        "answer_prompt": ANSWER_PROMPT_VERSION,
        "evidence_digest": _digest(
            {
                "document_id": document.get("id"),
                "analysis": state["analysis"],
                "cases": {
                    case.id: {
                        key: records[case.id].get(key)
                        for key in ("client_request_id", "response", "versions", "evidence")
                    }
                    for case in QUALITY_CASES
                },
            }
        ),
    }


class ReviewCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    result: Literal["pass", "fail"]
    notes: str = Field(min_length=1, max_length=4000)
    source_refs: list[str] = Field(min_length=1, max_length=64)


class SourceReview(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    reviewer_type: Literal["assistant", "human"]
    reviewer: str = Field(min_length=1, max_length=100)
    binding: dict[str, Any]
    checks: dict[str, ReviewCheck]


def source_references(state: dict[str, Any]) -> set[str]:
    refs = {f"guide:{title}" for title, _ in GUIDE_SECTIONS}
    for field in ("summary", "insights"):
        for block in state.get("analysis", {}).get(field, []):
            refs.update(str(value) for value in block.get("source_ids", []))
    for record in state.get("cases", {}).values():
        refs.update(
            str(item["source_id"]) for item in record.get("response", {}).get("citations", [])
        )
    return refs


def record_review(state: dict[str, Any], data: dict[str, Any]) -> None:
    try:
        review = SourceReview.model_validate(data)
    except ValidationError:
        raise EvaluationFailure("INVALID_SOURCE_REVIEW") from None
    if review.binding != review_binding(state):
        raise EvaluationFailure("SOURCE_REVIEW_BINDING_MISMATCH")
    if set(review.checks) != set(REQUIRED_REVIEWS):
        raise EvaluationFailure("SOURCE_REVIEW_CHECKS_INCOMPLETE")
    valid_refs = source_references(state)
    if not review.reviewer.strip() or any(
        not check.notes.strip() or any(ref not in valid_refs for ref in check.source_refs)
        for check in review.checks.values()
    ):
        raise EvaluationFailure("INVALID_SOURCE_REVIEW_REFERENCES")
    state.setdefault("source_reviews", []).append({**review.model_dump(), "recorded_at": _now()})


def quality_report(state: dict[str, Any]) -> dict[str, Any]:
    records = state.get("cases", {})
    completed = sum(records.get(case.id, {}).get("status") == "COMPLETED" for case in QUALITY_CASES)
    stale = _stale(state)
    indicators = analysis_indicators(state.get("analysis", {}))
    analysis_ok = (
        bool(state.get("analysis"))
        and all(indicators["summary_themes"].values())
        and all(indicators["summary_qualifications"].values())
        and indicators["insights_textually_distinct"]
        and not indicators["verbatim_block_flags"]
    )
    passed = sum(
        records.get(case.id, {}).get("status") == "COMPLETED"
        and case_checks(case, records[case.id])["automatic_pass"]
        for case in QUALITY_CASES
    )
    bound_reviews = []
    try:
        binding = review_binding(state)
        bound_reviews = [
            review
            for review in state.get("source_reviews", [])
            if review.get("binding") == binding
            and set(review.get("checks", {})) == set(REQUIRED_REVIEWS)
        ]
    except EvaluationFailure:
        pass

    def passed_review(review: dict[str, Any]) -> bool:
        return all(check.get("result") == "pass" for check in review["checks"].values())

    source_passed = bool(bound_reviews) and passed_review(bound_reviews[-1])
    human_reviews = [review for review in bound_reviews if review["reviewer_type"] == "human"]
    human_approval = source_passed and bool(human_reviews) and passed_review(human_reviews[-1])
    gaps = []
    for case in QUALITY_CASES:
        record = records.get(case.id, {})
        if record.get("status") != "COMPLETED":
            gaps.append(f"{case.id}:not_completed")
        elif not case_checks(case, record)["automatic_pass"]:
            gaps.append(f"{case.id}:automatic_checks_failed")
    for category in ("summary_themes", "summary_qualifications"):
        gaps.extend(
            f"analysis:{category}:{name}"
            for name, value in indicators[category].items()
            if not value
        )
    if not indicators["insights_textually_distinct"]:
        gaps.append("analysis:distinct_insights_unconfirmed")
    if indicators["verbatim_block_flags"]:
        gaps.append("analysis:verbatim_prose_flagged")
    if not source_passed:
        gaps.append("source_review:not_passed")
    return {
        "suite": "quality",
        "quality_version": QUALITY_VERSION,
        "observed_insight_review_prompt": state.get("analysis", {})
        .get("insight_review", {})
        .get("prompt_version"),
        "configured_insight_review_prompt": None,
        "editorial_review_in_normal_processing": False,
        "planned_chat_cases": len(QUALITY_CASES),
        "completed_chat_cases": completed,
        "automatic_chat_checks_passed": passed,
        "analysis_indicators": indicators,
        "automatic_checks_passed": completed == len(QUALITY_CASES)
        and passed == completed
        and analysis_ok
        and not stale,
        "source_review_passed": source_passed,
        "human_approval": human_approval,
        "stale_reasons": stale,
        "gaps": gaps,
        "required_source_review_checks": list(REQUIRED_REVIEWS),
        "limitations": "Heuristics do not establish source support; reviews retain reviewer type.",
    }


def _highest_revision(state: dict[str, Any]) -> int:
    records = [
        state.get("documents", {}).get(DOCUMENT_KEY, {}),
        *state.get("cases", {}).values(),
        *state.get("document_history", {}).get(DOCUMENT_KEY, []),
        *(record for history in state.get("case_history", {}).values() for record in history),
    ]
    return max(int(record.get("revision", 1)) for record in records)


def _selected_revision(state: dict[str, Any], requested: int | None) -> int:
    highest = _highest_revision(state)
    if requested is None:
        return highest
    if requested < highest:
        raise EvaluationFailure("QUALITY_REVISION_BEHIND_CURRENT")
    return requested


class QualityRunner(EvaluationRunner):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.path.exists() and self.state.get("quality_version") != QUALITY_VERSION:
            raise EvaluationFailure("NOT_A_QUALITY_ARTIFACT")
        self.state.update(suite="quality", quality_version=QUALITY_VERSION)

    def request_id(self, case: EvaluationCase) -> UUID:
        return quality_request_id(case.id, self.revision)

    def session(self, key: str, document_id: str) -> str:
        return super().session(f"{key}:revision:{self.revision}", document_id)

    def save(self) -> None:
        for record in self.state["cases"].values():
            reasons = stale_reasons(record, _current_versions(self.state))
            record.update(stale_evidence=bool(reasons), stale_reasons=reasons)
        self.state.update(updated_at=_now(), report=quality_report(self.state))
        _atomic_write(self.path, json.dumps(self.state, ensure_ascii=False, indent=2).encode())

    def prepare_fixture(self) -> None:
        fixture = self.path.parent / f"{QUALITY_VERSION}-request-guide.docx"
        known = self.state["documents"].get(DOCUMENT_KEY)
        if known:
            if (
                known.get("definition_sha256") != _digest(GUIDE_TEXT)
                or not fixture.exists()
                or hashlib.sha256(fixture.read_bytes()).hexdigest() != known["sha256"]
            ):
                raise EvaluationFailure("FIXTURE_BYTES_CHANGED")
            return
        if fixture.exists():
            raise EvaluationFailure("UNRECORDED_QUALITY_FIXTURE")
        content = guide_docx()
        _atomic_write(fixture, content)
        self.state["documents"][DOCUMENT_KEY] = {
            "filename": fixture.name,
            "sha256": hashlib.sha256(content).hexdigest(),
            "definition_sha256": _digest(GUIDE_TEXT),
            "upload_request_id": str(quality_request_id("upload", self.revision)),
            "revision": self.revision,
            "analysis_prompt": ANALYSIS_PROMPT_VERSION,
            "status": "PLANNED",
        }
        self.save()

    def _highest_recorded_revision(self) -> int:
        return _highest_revision(self.state)

    def _check_analysis_iteration(self) -> None:
        if self.revision <= self._highest_recorded_revision():
            raise EvaluationFailure("SAVED_ANALYSIS_STALE_NO_AUTOMATIC_REPROCESSING")
        known = self.state["documents"].get(DOCUMENT_KEY, {})
        if known.get("status") != "COMPLETED" or not known.get("id"):
            raise EvaluationFailure("PRIOR_QUALITY_DOCUMENT_NOT_COMPLETED")
        if any(record.get("status") == "PENDING" for record in self.state["cases"].values()):
            raise EvaluationFailure("PRIOR_QUALITY_WORK_PENDING")

    def prepare_analysis_iteration(self) -> None:
        """An explicitly higher live revision may replace changed analysis, preserving history.

        Called only by the live path, after source bytes/readiness are checked.
        A planned/pending replacement has no current analysis and is resumed unchanged.
        """
        analysis = self.state.get("analysis")
        if not analysis or analysis.get("prompt_version") == ANALYSIS_PROMPT_VERSION:
            return
        self._check_analysis_iteration()
        self.prepare_fixture()  # Verify the exact saved bytes before changing any artifact state.
        known = self.state["documents"][DOCUMENT_KEY]
        archived_at = _now()
        self.state.setdefault("document_history", {}).setdefault(DOCUMENT_KEY, []).append(
            {
                **json.loads(json.dumps(known)),
                "archived_at": archived_at,
                "replacement_revision": self.revision,
            }
        )
        self.state.setdefault("analysis_history", {}).setdefault(DOCUMENT_KEY, []).append(
            {
                "document_id": known["id"],
                "revision": known.get("revision", 1),
                "analysis": json.loads(json.dumps(analysis)),
                "archived_at": archived_at,
                "replacement_revision": self.revision,
            }
        )
        for case_id, record in self.state["cases"].items():
            history = self.state.setdefault("case_history", {}).setdefault(case_id, [])
            if not any(
                previous.get("revision", 1) == record.get("revision", 1)
                and previous.get("client_request_id") == record.get("client_request_id")
                for previous in history
            ):
                history.append(json.loads(json.dumps(record)))
        self.state["documents"][DOCUMENT_KEY] = {
            key: known[key] for key in ("filename", "sha256", "definition_sha256")
        } | {
            "upload_request_id": str(quality_request_id("upload", self.revision)),
            "revision": self.revision,
            "status": "PLANNED",
            "analysis_prompt": ANALYSIS_PROMPT_VERSION,
            "reprocess_reason": "analysis_prompt_changed",
        }
        # Archive complete before clearing current evidence; publish one atomic artifact
        # containing both history and the replacement identity before its first POST.
        self.state.pop("analysis")
        self.state["cases"] = {}
        self.save()

    def preflight(self) -> None:
        known = self.state["documents"].get(DOCUMENT_KEY, {})
        if known and self.revision < int(known.get("revision", 1)):
            raise EvaluationFailure("QUALITY_REVISION_BEHIND_DOCUMENT")
        _selected_revision(self.state, self.revision)
        if (
            not self.state.get("analysis")
            and known.get("analysis_prompt", ANALYSIS_PROMPT_VERSION) != ANALYSIS_PROMPT_VERSION
        ):
            raise EvaluationFailure("PENDING_ANALYSIS_PROMPT_CHANGED")
        if (
            self.state.get("analysis", {}).get("prompt_version", ANALYSIS_PROMPT_VERSION)
            != ANALYSIS_PROMPT_VERSION
        ):
            self._check_analysis_iteration()
        for record in self.state["cases"].values():
            if (
                record.get("status") in {"COMPLETED", "FAILED"}
                and record.get("revision", 1) == self.revision
                and stale_reasons(record, _current_versions(self.state))
            ):
                raise EvaluationFailure("STALE_EVIDENCE_EXPLICIT_REVISION_REQUIRED")


def run_live(path: Path, revision: int | None = None) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(path.parent):
        # Resolve again under the lock before settings, clients or any HTTP work.
        revision = _selected_revision(_read_quality(path), revision)
        settings = get_settings()
        engine = create_engine(
            settings.database_url,
            hide_parameters=True,
            connect_args={"connect_timeout": 3, "options": "-c statement_timeout=5000"},
        )
        try:
            with httpx.Client(base_url=BASE_URL, timeout=10, follow_redirects=False) as client:
                runner = QualityRunner(settings, client, engine, path, revision=revision)
                runner.preflight()
                code, readiness = runner.request("GET", "/health/ready")
                if code != 200 or readiness.get("status") != "ready":
                    raise EvaluationFailure("SERVICES_NOT_READY")
                runner.prepare_fixture()
                runner.prepare_analysis_iteration()
                identifier = runner.document(DOCUMENT_KEY)
                code, insights = runner.request("GET", f"/api/v1/documents/{identifier}/insights")
                if code != 200:
                    raise EvaluationFailure("ANALYSIS_UNAVAILABLE")
                runner.state["analysis"] = insights["analysis"]
                runner.state["analysis"]["prompt_version"] = insights["prompt_version"]
                runner.save()
                runner.preflight()
                for case in QUALITY_CASES:
                    runner.evaluate(case)
                    record = runner.state["cases"].get(case.id, {})
                    if record.get("status") != "COMPLETED":
                        break
                runner.save()
                return quality_report(runner.state)
        finally:
            engine.dispose()


def _read_quality(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "version": SUITE_VERSION,
            "suite": "quality",
            "quality_version": QUALITY_VERSION,
            "cases": {},
            "documents": {},
            "sessions": {},
        }
    state = load_state(path)
    if state.get("quality_version") != QUALITY_VERSION or state.get("suite") != "quality":
        raise EvaluationFailure("NOT_A_QUALITY_ARTIFACT")
    return state


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Permit one fixture upload and four chat turns; "
        "never enabled by review/report commands.",
    )
    parser.add_argument("--state-file", type=Path)
    parser.add_argument(
        "--revision",
        type=int,
        default=None,
        help="Omitted: resume the highest recorded revision, or 1 for a fresh run. "
        "Explicit older revisions are refused. With --live, a higher revision may also "
        "reprocess the same fixture once when its analysis prompt changed, "
        "preserving old evidence.",
    )
    parser.add_argument("--record-review", type=Path)
    parser.add_argument("--review-template", type=Path)
    args = parser.parse_args(argv)
    if (args.revision is not None and args.revision < 1) or (
        args.live and (args.record_review or args.review_template)
    ):
        parser.error("Use a positive revision; review commands must run separately from live work.")
    path = (
        args.state_file
        or Path(os.environ.get("LOCAL_STORAGE_DIR", "/app/data/uploads"))
        / "demo"
        / f"{QUALITY_VERSION}.json"
    )
    if args.review_template and args.review_template.resolve() == path.resolve():
        parser.error("The review template must have a different path from the evidence artifact.")
    try:
        if args.live:
            revision = _selected_revision(_read_quality(path), args.revision)
            report = run_live(path, revision)
        else:
            state = _read_quality(path)
            if args.review_template:
                template = {
                    "reviewer_type": "assistant",
                    "reviewer": "TODO",
                    "binding": review_binding(state),
                    "checks": {
                        key: {
                            "result": "fail",
                            "notes": "Pending source review.",
                            "source_refs": [],
                        }
                        for key in REQUIRED_REVIEWS
                    },
                }
                _atomic_write(args.review_template, json.dumps(template, indent=2).encode())
            if args.record_review:
                with evaluation_lock(path.parent):
                    state = _read_quality(path)
                    record_review(state, json.loads(args.record_review.read_text()))
                    state["report"] = quality_report(state)
                    _atomic_write(path, json.dumps(state, ensure_ascii=False, indent=2).encode())
            report = quality_report(state)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (EvaluationFailure, OSError, ValueError, httpx.HTTPError) as error:
        code = (
            str(error) if isinstance(error, EvaluationFailure) else "QUALITY_EVALUATION_UNAVAILABLE"
        )
        print(json.dumps({"suite": "quality", "error": code}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
