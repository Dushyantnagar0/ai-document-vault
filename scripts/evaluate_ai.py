"""Explicit opt-in, resumable HTTP evaluation. Never run automatically in tests.

python -m scripts.evaluate_ai --live --suite core --cases A1,F1
The state file, fixture bytes and stable request IDs prevent silent paid reruns.
Use --revision 2 only to explicitly authorize new selected questions, retaining
revision-one results. Stale replays are reported, never automatically regenerated.
Regex checks are preliminary; semantic source support requires human review.
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
import time
import unicodedata
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import NAMESPACE_URL, UUID, uuid5

import httpx
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, defer

from app.core.config import ConfigurationError, Settings, get_settings
from app.db.models import ChatTurn, Document, DocumentChunk, DocumentInsights, PerformanceSample
from app.prompts.analysis import ANALYSIS_PROMPT_VERSION
from app.prompts.chat import ANSWER_PROMPT_VERSION, CHAT_PIPELINE_VERSION, PLANNER_PROMPT_VERSION
from app.services.source_answer import SOURCE_COMPOSITION_VERSION, SOURCE_SELECTION_VERSION
from scripts.analysis_evaluation_profile import expected_review_profile, observed_review_profile
from scripts.evaluation_fixtures import (
    BY_ID,
    CASES,
    SUITE_VERSION,
    EvaluationCase,
    additional_fixture,
    case_request_id,
    upload_request_id,
)
from scripts.live_ingestion import BASE_URL, MIME_TYPES, _demo_file

POLL_SECONDS = 180
DOCUMENT_KEYS = tuple(sorted({case.document for case in CASES}))
SAFE_CODE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")


class EvaluationFailure(Exception):
    pass


def validate_run_id(value: str | None) -> str | None:
    if value is not None and not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,39}", value):
        raise EvaluationFailure("INVALID_RUN_ID")
    return value


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_write(path: Path, content: bytes) -> None:
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".evaluation-", delete=False
        ) as stream:
            temporary = stream.name
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


@contextmanager
def evaluation_lock(directory: Path) -> Iterator[None]:
    directory.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        directory / f".{SUITE_VERSION}.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise EvaluationFailure("INVALID_LOCK_FILE")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise EvaluationFailure("EVALUATION_ALREADY_RUNNING") from None
        yield
    finally:
        os.close(descriptor)


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "suite": "core",
            "version": SUITE_VERSION,
            "documents": {},
            "sessions": {},
            "cases": {},
        }
    if path.stat().st_size > 4 * 1024 * 1024:
        raise EvaluationFailure("INVALID_STATE_FILE")
    try:
        state = json.loads(path.read_text())
        if state["version"] != SUITE_VERSION or any(
            not isinstance(state[key], dict) for key in ("documents", "sessions", "cases")
        ):
            raise ValueError
        return cast(dict[str, Any], state)
    except (ValueError, KeyError, TypeError):
        raise EvaluationFailure("INVALID_STATE_FILE") from None


def select_cases(value: str | None) -> tuple[EvaluationCase, ...]:
    if value is None:
        return CASES
    selected = {identifier.strip().upper() for identifier in value.split(",")}
    if not selected or not selected.issubset(BY_ID):
        raise EvaluationFailure("UNKNOWN_CASE_ID")
    return tuple(case for case in CASES if case.id in selected)


def matches(patterns: Sequence[str], text: str) -> bool:
    return all(re.search(pattern, text, re.IGNORECASE) is not None for pattern in patterns)


def _supplied_source_text(source: dict[str, Any]) -> str:
    """Reconstruct actual ordered evidence; historical snapshots stored full text."""
    if "passages" in source:
        passages = source["passages"]
        if not isinstance(passages, list) or any(
            not isinstance(passage, dict) or not isinstance(passage.get("text"), str)
            for passage in passages
        ):
            raise EvaluationFailure("INVALID_SAVED_EVIDENCE")
        return "".join(passage["text"] for passage in passages)
    text = source.get("text")
    if not isinstance(text, str):
        raise EvaluationFailure("INVALID_SAVED_EVIDENCE")
    return text


def expected_versions(settings: Settings, *, intent: str | None = None) -> dict[str, Any]:
    """Match persisted turn recipes, including configured answer/review effort.

    Changing effort makes earlier evidence stale without rewriting its settings
    or implicitly sending another request.
    """
    versions: dict[str, Any] = {
        "saved_analysis_prompt": ANALYSIS_PROMPT_VERSION,
        "saved_analysis_review": expected_review_profile(settings),
        "chat_model": settings.openai_chat_model,
        "embedding_model": settings.openai_embedding_model,
        "planner_prompt": PLANNER_PROMPT_VERSION,
        "answer_prompt": ANSWER_PROMPT_VERSION,
        "pipeline": CHAT_PIPELINE_VERSION,
        "retrieval": "exact-cosine-v1",
        "configuration": settings.chat_configuration,
    }
    return factual_versions(versions) if intent == "factual" else versions


def factual_versions(versions: dict[str, Any]) -> dict[str, Any]:
    """Expected role-specific versions; historical records are never rewritten."""
    model = versions.get("configuration", {}).get("openai_source_answer_model")
    return {
        **versions,
        "planner_model": versions["configuration"]["openai_chat_model"],
        "chat_model": model,
        "source_answer_model": model,
        "answer_prompt": SOURCE_COMPOSITION_VERSION,
        "source_selection": SOURCE_SELECTION_VERSION,
        "source_composition": SOURCE_COMPOSITION_VERSION,
    }


def stale_reasons(record: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    observed = record.get("versions") or record.get("evidence", {}).get("versions", {})
    if not expected:
        return ["current_configuration_unavailable"]
    if not isinstance(observed, dict):
        observed = {}
    if record.get("evidence", {}).get("planner_intent") == "factual" and expected.get(
        "configuration", {}
    ).get("openai_source_answer_model"):
        expected = factual_versions(expected)
    return [
        f"{name}_missing" if name not in observed else f"{name}_changed"
        for name, value in expected.items()
        if name not in observed or observed[name] != value
    ]


def current_completed(record: dict[str, Any], expected: dict[str, Any]) -> bool:
    return record.get("status") == "COMPLETED" and not stale_reasons(record, expected)


def automatic_checks(
    case: EvaluationCase, body: dict[str, Any], snapshot: dict[str, Any]
) -> dict[str, Any]:
    answer = body.get("answer", "")
    citations = body.get("citations", [])
    support = bool(citations) and body.get("evidence_status") in {"answered", "partial"}
    membership = bool(snapshot.get("membership_valid"))
    expected = matches(case.expected, answer)
    forbidden_absent = not any(
        re.search(pattern, answer, re.IGNORECASE) for pattern in case.forbidden
    )
    status_ok = body.get("evidence_status") == case.status
    if case.status == "insufficient":
        behavior = (
            status_ok
            and membership
            and forbidden_absent
            and not citations
            and bool(
                re.search(
                    r"cannot find|not (?:provided|specified|stated)|no information",
                    answer,
                    re.IGNORECASE,
                )
            )
        )
    else:
        behavior = status_ok and support and membership and expected and forbidden_absent
    return {
        "expected_facts": expected,
        "status_expected": status_ok,
        "citation_membership": membership,
        "citations_present_if_supported": support
        if case.status != "insufficient"
        else not citations,
        "attack_markers_absent": forbidden_absent,
        "automatic_pass": behavior,
        "retrieval_pass": (
            bool(snapshot.get("selected_membership_valid"))
            and matches(case.expected, snapshot.get("supplied_text", ""))
        )
        if case.id[0] in {"A", "F"}
        else None,
        "human_semantic_review": "required",
    }


def summarize(state: dict[str, Any]) -> dict[str, Any]:
    records = state["cases"]
    expected = state.get("expected_versions", {})
    completed = [
        records[case.id] for case in CASES if records.get(case.id, {}).get("status") == "COMPLETED"
    ]
    af = [records.get(case.id, {}) for case in CASES if case.id[0] in {"A", "F"}]
    answer_pass = sum(
        current_completed(record, expected) and bool(record.get("checks", {}).get("automatic_pass"))
        for record in af
    )
    retrieval_pass = sum(
        current_completed(record, expected) and bool(record.get("checks", {}).get("retrieval_pass"))
        for record in af
    )
    groups = {}
    for prefix, required in (("U", 4), ("C", 2), ("I", 2)):
        passed = sum(
            current_completed(records.get(case.id, {}), expected)
            and bool(records.get(case.id, {}).get("checks", {}).get("automatic_pass"))
            for case in CASES
            if case.id.startswith(prefix)
        )
        groups[prefix] = {"passed": passed, "required": required, "gate_met": passed == required}
    current = [record for record in completed if current_completed(record, expected)]
    all_done = len(current) == 20
    membership = all(
        record.get("checks", {}).get("citation_membership") is True for record in completed
    )
    return {
        "suite": "core",
        "planned": 20,
        "completed": len(completed),
        "current_configuration_completed": len(current),
        "stale_cases": [
            case.id
            for case in CASES
            if records.get(case.id, {}).get("status") == "COMPLETED"
            and stale_reasons(records[case.id], expected)
        ],
        "statuses": dict(Counter(record.get("status", "PENDING") for record in records.values())),
        "unrun": [case.id for case in CASES if case.id not in records],
        "answerable_followup": {
            "automatic_pass": answer_pass,
            "retrieval_pass": retrieval_pass,
            "required": 11,
            "denominator": 12,
        },
        "groups": groups,
        "membership_100_percent": membership if all_done else None,
        "automatic_gate_met": all_done
        and membership
        and answer_pass >= 11
        and retrieval_pass >= 11
        and all(group["gate_met"] for group in groups.values()),
        "human_semantic_review": (
            "required; automatic regex/membership checks do not establish entailment"
        ),
        "release_gate": "pending human review and remaining delivery checks",
    }


class EvaluationRunner:
    def __init__(
        self,
        settings: Settings,
        client: httpx.Client,
        engine: Engine,
        path: Path,
        *,
        revision: int = 1,
        run_id: str | None = None,
    ) -> None:
        self.run_id = validate_run_id(run_id)
        if self.run_id is not None and revision != 1:
            raise EvaluationFailure("ISOLATED_RUN_REQUIRES_REVISION_ONE")
        if type(revision) is not int or revision < 1:
            raise EvaluationFailure("INVALID_EVALUATION_REVISION")
        self.settings, self.client, self.engine, self.path = settings, client, engine, path
        self.revision = revision
        existing_state = path.exists()
        self.state = load_state(path)
        recipe = {
            "versions": expected_versions(settings),
            "processing": {
                name: getattr(settings, name)
                for name in (
                    "chunk_size_tokens",
                    "chunk_overlap_tokens",
                    "embedding_batch_size",
                    "embedding_batch_max_tokens",
                    "analysis_group_max_tokens",
                    "analysis_max_output_tokens",
                    "summary_max_output_tokens",
                    "max_document_ai_calls",
                    "max_extracted_tokens",
                    "max_document_chunks",
                    "processing_max_attempts",
                    "processing_deadline_seconds",
                )
            },
        }
        if self.run_id is not None:
            if existing_state and self.state.get("run_id") != self.run_id:
                raise EvaluationFailure("RUN_ID_MISMATCH")
            if existing_state and self.state.get("recipe") != recipe:
                raise EvaluationFailure("RUN_RECIPE_CHANGED_USE_NEW_RUN_ID")
            self.state.update(run_id=self.run_id, recipe=recipe)
        elif self.state.get("run_id") is not None:
            raise EvaluationFailure("RUN_ID_REQUIRED")
        self.state["expected_versions"] = expected_versions(settings)
        self.state.setdefault("case_history", {})

    def request_id(self, case: EvaluationCase) -> UUID:
        record = self.state["cases"].get(case.id, {})
        if record.get("revision", 1) == self.revision and record.get("client_request_id"):
            return UUID(record["client_request_id"])
        if self.run_id is not None:
            return uuid5(
                NAMESPACE_URL, f"document-vault:{SUITE_VERSION}:run:{self.run_id}:case:{case.id}"
            )
        return case_request_id(case.id, self.revision)

    def session_key(self, case: EvaluationCase) -> str:
        if self.run_id is None:
            return case.document
        # Connected dependency components share history; unrelated questions do not.
        connected = {case.id}
        while True:
            expanded = connected | {
                item.id
                for item in CASES
                if item.id in connected or connected.intersection(item.requires)
            }
            expanded |= {
                dependency for item in CASES if item.id in expanded for dependency in item.requires
            }
            if expanded == connected:
                break
            connected = expanded
        root = next(item.id for item in CASES if item.id in connected)
        return f"{case.document}:{root}"

    def activate_revision(self, case: EvaluationCase) -> dict[str, Any]:
        previous = self.state["cases"].get(case.id, {})
        if not previous or previous.get("revision", 1) == self.revision:
            return cast(dict[str, Any], previous)
        if previous.get("status") == "PENDING":
            raise EvaluationFailure("CASE_PENDING_CANNOT_CHANGE_REVISION")
        history = self.state["case_history"].setdefault(case.id, [])
        restored = next(
            (record for record in history if record.get("revision", 1) == self.revision), None
        )
        old_revision = previous.get("revision", 1)
        if not any(record.get("revision", 1) == old_revision for record in history):
            history.append({**json.loads(json.dumps(previous)), "revision": old_revision})
        current = (
            json.loads(json.dumps(restored))
            if restored is not None
            else {
                "question": case.question,
                "revision": self.revision,
                "client_request_id": str(case_request_id(case.id, self.revision)),
            }
        )
        self.state["cases"][case.id] = current
        return current

    def save(self) -> None:
        for record in self.state["cases"].values():
            reasons = stale_reasons(record, self.state["expected_versions"])
            record["stale_evidence"] = bool(reasons)
            record["stale_reasons"] = reasons
        self.state["updated_at"] = _now()
        self.state["report"] = summarize(self.state)
        _atomic_write(self.path, json.dumps(self.state, ensure_ascii=False, indent=2).encode())

    def request(self, method: str, path: str, **kwargs: Any) -> tuple[int, dict[str, Any]]:
        response = self.client.request(method, path, **kwargs)
        try:
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError
        except ValueError:
            raise EvaluationFailure("INVALID_API_RESPONSE") from None
        return response.status_code, payload

    def prepare_document_retry(self, key: str) -> None:
        """Explicit replacement of a confirmed failed document, never an uncertain upload."""
        known = self.state["documents"].get(key)
        if not known:
            raise EvaluationFailure("RETRY_DOCUMENT_NOT_RECORDED")
        if known.get("status") == "PLANNED" and not known.get("id"):
            return  # Identity was saved before an upload with an uncertain acknowledgement.
        identifier = str(UUID(known["id"]))
        code, actual = self.request("GET", f"/api/v1/documents/{identifier}")
        if code != 200:
            raise EvaluationFailure("DOCUMENT_STATUS_UNAVAILABLE")
        if actual.get("status") in {"PENDING", "PROCESSING"} and known.get("revision", 1) > 1:
            return  # Resume the already accepted replacement without incrementing.
        if actual.get("status") != "FAILED":
            raise EvaluationFailure("RETRY_REQUIRES_FAILED_DOCUMENT")
        case_records = list(self.state["cases"].items())
        for case_id, history in self.state.get("case_history", {}).items():
            case_records.extend((case_id, record) for record in history)
        if key in self.state["sessions"] or any(
            case_id in BY_ID
            and BY_ID[case_id].document == key
            and (
                record.get("session_id")
                or record.get("turn_id")
                or record.get("response") is not None
                or record.get("status") in {"PENDING", "COMPLETED", "FAILED"}
            )
            for case_id, record in case_records
        ):
            raise EvaluationFailure("RETRY_DOCUMENT_HAS_ACCEPTED_CASE_OR_SESSION")
        code, sessions = self.request(
            "GET", "/api/v1/chat/sessions", params={"document_id": identifier, "limit": "1"}
        )
        if code != 200:
            raise EvaluationFailure("SESSION_SEARCH_FAILED")
        if sessions.get("items") or sessions.get("next_cursor"):
            raise EvaluationFailure("RETRY_DOCUMENT_HAS_ACCEPTED_CASE_OR_SESSION")
        filename = known["filename"]
        if Path(filename).name != filename:
            raise EvaluationFailure("INVALID_FIXTURE_FILENAME")
        original = self.path.parent / filename
        content = original.read_bytes()
        checksum = hashlib.sha256(content).hexdigest()
        if checksum != known["sha256"]:
            raise EvaluationFailure("FIXTURE_BYTES_CHANGED")
        revision = known.get("revision", 1) + 1
        replacement = (
            self.path.parent / f"{SUITE_VERSION}-{key}-revision-{revision}{original.suffix}"
        )
        if replacement.exists():
            if replacement.read_bytes() != content:
                raise EvaluationFailure("FIXTURE_BYTES_CHANGED")
        else:
            _atomic_write(replacement, content)
        archived = json.loads(json.dumps(known))
        archived.update(observed_at=_now(), observed_document=actual)
        self.state.setdefault("document_history", {}).setdefault(key, []).append(archived)
        self.state["documents"][key] = {
            "filename": replacement.name,
            "sha256": checksum,
            "upload_request_id": str(
                uuid5(
                    NAMESPACE_URL,
                    f"document-vault:{SUITE_VERSION}:{key}:upload-revision:{revision}",
                )
            ),
            "revision": revision,
            "status": "PLANNED",
        }
        self.save()  # Must precede the first outbound upload, including a failed acknowledgement.

    def document(self, key: str) -> str:
        known = self.state["documents"].get(key)
        if not known:
            if key in {"juniper", "maple"}:
                format_name, version = ("pdf", 1) if key == "juniper" else ("docx", 2)
                fixture = _demo_file(self.settings.local_storage_dir, format_name, version)
            else:
                fixture = self.path.parent / f"{SUITE_VERSION}-{key}.docx"
                if not fixture.exists():
                    _atomic_write(fixture, additional_fixture(key))
            known = {
                "filename": fixture.name,
                "sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
                "upload_request_id": str(
                    uuid5(
                        NAMESPACE_URL,
                        f"document-vault:{SUITE_VERSION}:run:{self.run_id}:upload:{key}",
                    )
                    if self.run_id is not None
                    else upload_request_id(key)
                ),
                "revision": 1,
                "status": "PLANNED",
            }
            self.state["documents"][key] = known
            self.save()
        if not known.get("id"):
            filename = known["filename"]
            if Path(filename).name != filename:
                raise EvaluationFailure("INVALID_FIXTURE_FILENAME")
            fixture = self.path.parent / filename
            content = fixture.read_bytes()
            if hashlib.sha256(content).hexdigest() != known["sha256"]:
                raise EvaluationFailure("FIXTURE_BYTES_CHANGED")
            format_name = "pdf" if fixture.suffix == ".pdf" else "docx"
            code, body = self.request(
                "POST",
                "/api/v1/documents/upload",
                files={"file": (fixture.name, content, MIME_TYPES[format_name])},
                headers={"Idempotency-Key": known["upload_request_id"]},
                timeout=20,
            )
            if code != 202:
                raise EvaluationFailure("DOCUMENT_ACCEPTANCE_FAILED")
            known.update(id=str(UUID(body["document_id"])), status="PENDING")
            self.save()
        identifier = str(UUID(known["id"]))
        deadline = time.monotonic() + POLL_SECONDS
        while True:
            code, body = self.request("GET", f"/api/v1/documents/{identifier}")
            if code != 200:
                raise EvaluationFailure("DOCUMENT_STATUS_UNAVAILABLE")
            self.state["documents"][key].update(
                status=body["status"], chunk_count=body.get("chunk_count")
            )
            self.save()
            if body["status"] == "COMPLETED":
                if self.run_id is not None:
                    code, analysis = self.request("GET", f"/api/v1/documents/{identifier}/insights")
                    if code != 200 or (
                        analysis.get("prompt_version") != ANALYSIS_PROMPT_VERSION
                        or analysis.get("model") != self.settings.openai_chat_model
                        or observed_review_profile(analysis.get("analysis", {}))
                        != expected_review_profile(self.settings)
                    ):
                        raise EvaluationFailure("DOCUMENT_ANALYSIS_STALE_USE_NEW_RUN_ID")
                    known["analysis_prompt"] = analysis["prompt_version"]
                    known["analysis_model"] = analysis["model"]
                    known["analysis_review"] = observed_review_profile(analysis["analysis"])
                    self.save()
                return identifier
            if body["status"] == "FAILED":
                raise EvaluationFailure("DOCUMENT_PROCESSING_FAILED")
            if time.monotonic() >= deadline:
                raise EvaluationFailure("DOCUMENT_PENDING_TIMEOUT")
            time.sleep(2)

    def session(self, key: str, document_id: str) -> str:
        if key in self.state["sessions"]:
            return str(UUID(self.state["sessions"][key]))
        title = (
            f"{SUITE_VERSION}:run:{self.run_id}:{key}"
            if self.run_id is not None
            else f"{SUITE_VERSION}:{key}"
        )
        found: list[str] = []
        cursor = None
        for _ in range(10):
            params = {"document_id": document_id, "limit": "100"}
            if cursor:
                params["cursor"] = cursor
            code, page = self.request("GET", "/api/v1/chat/sessions", params=params)
            if code != 200:
                raise EvaluationFailure("SESSION_SEARCH_FAILED")
            found.extend(str(UUID(item["id"])) for item in page["items"] if item["title"] == title)
            cursor = page.get("next_cursor")
            if not cursor:
                break
        else:
            raise EvaluationFailure("SESSION_SEARCH_LIMIT")
        if len(found) > 1:
            raise EvaluationFailure("AMBIGUOUS_EVALUATION_SESSION")
        if found:
            identifier = found[0]
        else:
            code, body = self.request(
                "POST", "/api/v1/chat/sessions", json={"document_id": document_id, "title": title}
            )
            if code != 201:
                raise EvaluationFailure("SESSION_CREATION_FAILED")
            identifier = str(UUID(body["id"]))
        self.state["sessions"][key] = identifier
        self.save()
        return identifier

    def turn(self, session_id: str, case: EvaluationCase) -> dict[str, Any] | None:
        with Session(self.engine) as database:
            turn = database.scalar(
                select(ChatTurn).where(
                    ChatTurn.session_id == UUID(session_id),
                    ChatTurn.client_request_id == self.request_id(case),
                )
            )
            if turn is None:
                return None
            fingerprint = hashlib.sha256(
                unicodedata.normalize("NFC", case.question).strip().encode()
            ).hexdigest()
            if turn.question_fingerprint != fingerprint:
                raise EvaluationFailure("PERSISTED_CASE_QUESTION_MISMATCH")
            return {
                "status": turn.status,
                "body": turn.response_body,
                "http_status": turn.response_status,
                "turn_id": str(turn.id),
                "versions": turn.versions,
            }

    def snapshot(
        self, session_id: str, case: EvaluationCase, document_id: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        with Session(self.engine) as database:
            turn = database.scalar(
                select(ChatTurn).where(
                    ChatTurn.session_id == UUID(session_id),
                    ChatTurn.client_request_id == self.request_id(case),
                )
            )
            document = database.get(
                Document, UUID(document_id), options=[defer(Document.extracted_text)]
            )
            if turn is None or document is None:
                raise EvaluationFailure("EVALUATION_EVIDENCE_UNAVAILABLE")
            selected = [UUID(value) for value in turn.context_snapshot.get("selected_ids", [])]
            cited = [UUID(item["source_id"]) for item in body.get("citations", [])]
            ids = set(selected + cited)
            rows = (
                database.execute(
                    select(
                        DocumentChunk.id,
                        DocumentChunk.document_id,
                        DocumentChunk.attempt_id,
                        DocumentChunk.profile_id,
                        DocumentChunk.text,
                    ).where(DocumentChunk.id.in_(ids))
                ).all()
                if ids
                else []
            )
            valid = {
                row.id
                for row in rows
                if row.document_id == document.id
                and row.attempt_id == document.published_attempt_id
                and row.profile_id == document.profile_id
            }
            supplied = turn.context_snapshot.get("payload", {}).get("evidence", [])
            supplied_text = "\n".join(
                _supplied_source_text(item)
                for item in supplied
                if UUID(item["source_id"]) in set(selected) & valid
            )
            versions = dict(turn.versions or {})
            versions["saved_analysis_prompt"] = None
            versions["saved_analysis_review"] = {}
            # Observe the published analysis even for factual queries. A current
            # chat prompt does not make an older ingestion result current.
            analysis = database.get(DocumentInsights, document.id)
            if analysis is not None and analysis.attempt_id == document.published_attempt_id:
                versions["saved_analysis_prompt"] = analysis.prompt_version
                versions["saved_analysis_review"] = observed_review_profile(analysis.payload)
            usage = database.scalars(
                select(PerformanceSample)
                .where(PerformanceSample.turn_id == turn.id)
                .order_by(PerformanceSample.recorded_at)
            ).all()
            return {
                "membership_valid": all(identifier in valid for identifier in selected)
                and all(identifier in valid and identifier in selected for identifier in cited)
                and all(
                    item.get("document_id") == document_id for item in body.get("citations", [])
                ),
                "selected_membership_valid": bool(selected)
                and all(identifier in valid for identifier in selected),
                "selected_ids": [str(identifier) for identifier in selected],
                "published_selected_text": "\n".join(
                    row.text for row in rows if row.id in selected and row.id in valid
                ),
                "supplied_text": supplied_text,
                "planner_intent": turn.planner_intent,
                "retrieval_query": turn.retrieval_query,
                "versions": versions,
                "timings": turn.timings,
                "ai_calls_reserved": turn.ai_calls_reserved,
                "usage_available": [
                    {
                        "operation": sample.operation,
                        "outcome": sample.outcome,
                        "input_tokens": sample.input_tokens,
                        "output_tokens": sample.output_tokens,
                    }
                    for sample in usage
                ],
            }

    def evaluate(self, case: EvaluationCase) -> None:
        previous = self.activate_revision(case)
        if previous and previous.get("question") != case.question:
            raise EvaluationFailure("CASE_DEFINITION_CHANGED")
        missing = [
            identifier
            for identifier in case.requires
            if not current_completed(
                self.state["cases"].get(identifier, {}), self.state["expected_versions"]
            )
        ]
        if missing:
            self.state["cases"][case.id] = {
                **previous,
                "revision": self.revision,
                "client_request_id": str(self.request_id(case)),
                "status": "SKIPPED",
                "question": case.question,
                "reason": "PREREQUISITE_NOT_COMPLETED",
                "requires": missing,
            }
            self.save()
            return
        document_id = self.document(case.document)
        session_id = self.session(self.session_key(case), document_id)
        record = {
            **previous,
            "revision": self.revision,
            "question": case.question,
            "document_id": document_id,
            "session_id": session_id,
            "client_request_id": str(self.request_id(case)),
            "status": "PENDING",
            "human_semantic_review": "required",
        }
        self.state["cases"][case.id] = record
        self.save()
        saved = self.turn(session_id, case)
        if saved is None and (
            previous.get("status") in {"COMPLETED", "FAILED"}
            or previous.get("response") is not None
        ):
            raise EvaluationFailure("SAVED_TURN_MISSING_NO_AUTOMATIC_RETRY")
        if saved is None:
            try:
                code, body = self.request(
                    "POST",
                    f"/api/v1/chat/sessions/{session_id}/messages",
                    json={
                        "question": case.question,
                        "client_request_id": str(self.request_id(case)),
                    },
                    timeout=105,
                )
                record["last_http_status"] = code
                record["last_http_response"] = body
                self.save()
                if code == 200:
                    saved = {"status": "COMPLETED", "body": body, "http_status": code}
                elif code not in {409, 429, 503, 504}:
                    record.update(status="FAILED", response=body)
                    self.save()
                    return
            except httpx.HTTPError:
                record["transport_status"] = "UNCERTAIN_PERSISTED_UUID"
        deadline = time.monotonic() + 120
        while saved is None or saved["status"] == "PENDING":
            saved = self.turn(session_id, case)
            if saved is not None and saved["status"] != "PENDING":
                break
            if time.monotonic() >= deadline:
                record.update(status="PENDING", error_code="TURN_PENDING_TIMEOUT")
                self.save()
                return
            time.sleep(2)
        record.update(
            status=saved["status"],
            response=saved.get("body"),
            http_status=saved.get("http_status"),
            observed_at=_now(),
            versions=saved.get("versions", {}),
        )
        if saved["status"] == "COMPLETED":
            snapshot = self.snapshot(session_id, case, document_id, saved["body"])
            record.update(evidence=snapshot, checks=automatic_checks(case, saved["body"], snapshot))
            record["versions"] = snapshot.get("versions", saved.get("versions", {}))
        self.save()


def run(
    cases: Sequence[EvaluationCase],
    *,
    revision: int = 1,
    retry_failed_documents: Sequence[str] = (),
    run_id: str | None = None,
) -> dict[str, Any]:
    run_id = validate_run_id(run_id)
    if run_id is not None and revision != 1:
        raise EvaluationFailure("ISOLATED_RUN_REQUIRES_REVISION_ONE")
    if any(key not in {case.document for case in cases} for key in retry_failed_documents):
        raise EvaluationFailure("RETRY_DOCUMENT_NOT_SELECTED")
    settings = get_settings()
    directory = settings.local_storage_dir / "demo"
    with evaluation_lock(directory):
        suffix = f"-{run_id}" if run_id is not None else ""
        path = directory / f"{SUITE_VERSION}{suffix}.json"
        engine = create_engine(
            settings.database_url,
            hide_parameters=True,
            connect_args={"connect_timeout": 3, "options": "-c statement_timeout=5000"},
        )
        try:
            with httpx.Client(base_url=BASE_URL, timeout=10, follow_redirects=False) as client:
                runner = EvaluationRunner(
                    settings, client, engine, path, revision=revision, run_id=run_id
                )
                runner.save()  # Freeze the isolated run's recipe before any HTTP work.
                for key in dict.fromkeys(retry_failed_documents):
                    runner.prepare_document_retry(key)
                runner.state.pop("last_run_error", None)
                for case in cases:
                    try:
                        runner.evaluate(case)
                        if runner.state["cases"].get(case.id, {}).get("status") == "PENDING":
                            break  # Do not pile new questions onto an unresolved accepted turn.
                        result = runner.state["cases"].get(case.id, {})
                        if run_id is not None and (
                            result.get("status") != "COMPLETED"
                            or not result.get("checks", {}).get("automatic_pass")
                            or not current_completed(result, runner.state["expected_versions"])
                        ):
                            break  # Inspect failed/stale pilot evidence before spending more.
                    except EvaluationFailure as error:
                        record = runner.state["cases"].setdefault(
                            case.id, {"question": case.question}
                        )
                        runner.state["last_run_error"] = {
                            "case_id": case.id,
                            "revision": revision,
                            "code": str(error),
                        }
                        if str(error) != "CASE_PENDING_CANNOT_CHANGE_REVISION":
                            record.update(status="BLOCKED", error_code=str(error))
                        runner.save()
                        break
                selected_pass = all(
                    runner.state["cases"].get(case.id, {}).get("revision", 1) == revision
                    and current_completed(
                        runner.state["cases"].get(case.id, {}), runner.state["expected_versions"]
                    )
                    and runner.state["cases"]
                    .get(case.id, {})
                    .get("checks", {})
                    .get("automatic_pass")
                    is True
                    for case in cases
                )
                return {
                    **summarize(runner.state),
                    "state_file": str(path),
                    "selected_case_ids": [case.id for case in cases],
                    "selected_automatic_pass": selected_pass,
                    "run_id": run_id,
                    "requested_revision": revision,
                    "last_run_error": runner.state.get("last_run_error"),
                }
        finally:
            engine.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Explicit opt-in, resumable core live AI evaluation."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        required=True,
        help="Permit paid document analysis and chat calls.",
    )
    parser.add_argument("--suite", choices=("core",), default="core")
    parser.add_argument("--cases", help="Comma-separated IDs; omitted cases are not sent to AI.")
    parser.add_argument(
        "--run-id",
        help="Fresh isolated run identity (lowercase letters/digits/hyphens, max 40). "
        "Reuse unchanged to resume; a new ID permits paid ingestion and questions.",
    )

    def positive_revision(value: str) -> int:
        try:
            revision = int(value)
            if revision < 1:
                raise ValueError
            return revision
        except ValueError:
            raise argparse.ArgumentTypeError("Revision must be a positive integer") from None

    parser.add_argument(
        "--revision",
        type=positive_revision,
        default=1,
        help="Explicitly choose a case revision; a new revision permits new paid questions.",
    )
    parser.add_argument(
        "--retry-failed-document",
        choices=DOCUMENT_KEYS,
        action="append",
        default=[],
        help="Explicitly replace a confirmed FAILED fixture document; selected cases must use it.",
    )
    args = parser.parse_args(argv)
    try:
        report = run(
            select_cases(args.cases),
            revision=args.revision,
            retry_failed_documents=args.retry_failed_document,
            run_id=args.run_id,
        )
    except (
        EvaluationFailure,
        ConfigurationError,
        SQLAlchemyError,
        httpx.HTTPError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ):
        # Never serialize SDK/connection exceptions, settings or credential URLs.
        print(
            json.dumps(
                {
                    "suite": "core",
                    "error_code": "EVALUATION_UNAVAILABLE",
                    "human_semantic_review": "required",
                }
            )
        )
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["selected_automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
