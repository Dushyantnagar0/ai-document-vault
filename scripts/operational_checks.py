"""Explicit isolated hard-death checks; this script NEVER runs Docker commands.

Run each phase in the existing test service (same shared test uploads volume):
  docker compose -p vault-test -f compose.yaml -f compose.test.yaml --profile test \
    run --rm --no-deps test python -m scripts.operational_checks PHASE --isolated

Phases, in order:
  prepare-worker
  [Parent hard-kills and restarts ONLY the vault-test worker immediately.]
  recover-worker --confirm-worker-killed
  prepare-api
  [Parent hard-kills and restarts ONLY the vault-test API immediately.]
  recover-api --confirm-api-killed

The parent records the actual kill/restart commands as separate evidence. Barrier
readiness reports a short kill window; a missed window fails rather than passing.
Lease/retry timestamps are accelerated only on this run's guarded synthetic IDs;
results explicitly disclose that this is not a measured natural-expiry latency.
Use --state /app/data/uploads/operational-checks/another-run.json for a new run.
Keep other test traffic paused during provider-counter comparisons. Never supply
a real key. No phase automatically creates another run or regenerates a question.
"""

import argparse
import base64
import binascii
import fcntl
import hashlib
import json
import os
import re
import stat
import time
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import Any, Self, cast
from uuid import UUID, uuid4

import httpx
import pymupdf
from sqlalchemy import Engine, create_engine, func, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, defer

from app.core.config import ConfigurationError, Settings, get_settings
from app.db.models import (
    ChatMessage,
    ChatSession,
    ChatSessionDocument,
    ChatTurn,
    Document,
    DocumentChunk,
    DocumentInsights,
    ProcessingAttempt,
)

TOOL_VERSION = "vault-isolated-operations-v1"
API_URL = "http://api:8000"
STUB_URL = "http://provider-stub:4010"
TEST_KEY = "test-placeholder-never-a-real-key"
UUID_FIELDS = (
    "run_id",
    "worker_nonce",
    "api_nonce",
    "upload_request_id",
    "chat_request_id",
    "new_question_request_id",
)
OPTIONAL_UUID_FIELDS = (
    "document_id",
    "attempt_id",
    "claim_token",
    "session_id",
    "turn_id",
    "turn_token",
)


class OperationalFailure(Exception):
    def __init__(self, code: str, http_status: int | None = None) -> None:
        super().__init__(code)
        self.code, self.http_status = code, http_status


def assert_isolated(
    settings: Settings,
    isolated: bool,
    environment: Mapping[str, str] | None = None,
) -> None:
    environment = os.environ if environment is None else environment
    if (
        not isolated
        or environment.get("VAULT_INTEGRATION_TESTS") != "1"
        or environment.get("OPENAI_BASE_URL") != f"{STUB_URL}/v1"
        or settings.postgres_db != "vault_test"
        or settings.postgres_user != "vault_test"
        or settings.postgres_host != "postgres"
        or settings.openai_api_key.get_secret_value() != TEST_KEY
        or settings.redis_url.get_secret_value() != "redis://redis:6379/1"
    ):
        raise OperationalFailure("REFUSING_NONISOLATED_CONFIGURATION")


def assert_confirmation(phase: str, worker_killed: bool, api_killed: bool) -> None:
    if phase == "recover-worker" and not worker_killed:
        raise OperationalFailure("WORKER_KILL_CONFIRMATION_REQUIRED")
    if phase == "recover-api" and not api_killed:
        raise OperationalFailure("API_KILL_CONFIRMATION_REQUIRED")


def _uuid(value: Any) -> str:
    try:
        if not isinstance(value, str) or len(value) != 36 or str(UUID(value)) != value:
            raise ValueError
    except ValueError:
        raise OperationalFailure("INVALID_OPERATIONAL_STATE") from None
    return value


def _pdf(nonce: str) -> bytes:
    with pymupdf.open() as document:  # type: ignore[no-untyped-call]
        page = document.new_page()
        page.insert_text(
            (72, 72),
            "Operational recovery fixture\nOwner: Test Operator\n"
            f"[operational:{nonce}]\nRenewal date: 12 December 2027\n",
        )
        return cast(bytes, document.tobytes(garbage=4, deflate=True))


def new_state() -> dict[str, Any]:
    state: dict[str, Any] = {"tool": TOOL_VERSION, **{name: str(uuid4()) for name in UUID_FIELDS}}
    payload = _pdf(state["worker_nonce"])
    state.update(
        filename=f"operational-{state['run_id']}.pdf",
        fixture_pdf=base64.b64encode(payload).decode(),
        sha256=hashlib.sha256(payload).hexdigest(),
        clock_accelerations=[],
    )
    return state


def validate_state(state: Any) -> dict[str, Any]:
    try:
        if not isinstance(state, dict) or state.get("tool") != TOOL_VERSION:
            raise OperationalFailure("STATE_OWNER_MISMATCH")
        for name in UUID_FIELDS:
            _uuid(state[name])
        for name in OPTIONAL_UUID_FIELDS:
            if name in state:
                _uuid(state[name])
        if state["filename"] != f"operational-{state['run_id']}.pdf":
            raise OperationalFailure("INVALID_OPERATIONAL_STATE")
        if not isinstance(state.get("clock_accelerations"), list) or not all(
            isinstance(value, str) for value in state["clock_accelerations"]
        ):
            raise OperationalFailure("INVALID_OPERATIONAL_STATE")
        payload = base64.b64decode(state["fixture_pdf"], validate=True)
        if (
            not payload.startswith(b"%PDF-")
            or hashlib.sha256(payload).hexdigest() != state["sha256"]
        ):
            raise OperationalFailure("INVALID_OPERATIONAL_STATE")
    except (KeyError, ValueError, TypeError, binascii.Error):
        raise OperationalFailure("INVALID_OPERATIONAL_STATE") from None
    return state


class StateFile:
    """Exclusive, atomically saved state in the isolated upload volume's own subdir."""

    def __init__(self, settings: Settings, path: Path | None, *, create: bool = False) -> None:
        self.directory = settings.local_storage_dir / "operational-checks"
        self.path = path if path is not None else self.directory / "recovery-v1.json"
        if (
            self.path.parent != self.directory
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\.json", self.path.name) is None
        ):
            raise OperationalFailure("STATE_LOCATION_INVALID")
        self.create, self.directory_fd, self.lock_fd = create, -1, -1
        self.data: dict[str, Any] = {}

    def __enter__(self) -> Self:
        self.directory.mkdir(mode=0o700, exist_ok=True)
        self.directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            self.lock_fd = os.open(
                self.path.name + ".lock",
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                0o600,
                dir_fd=self.directory_fd,
            )
            if not stat.S_ISREG(os.fstat(self.lock_fd).st_mode):
                raise OperationalFailure("INVALID_STATE_LOCK")
            try:
                fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise OperationalFailure("OPERATIONAL_STATE_BUSY") from None
            try:
                descriptor = os.open(
                    self.path.name,
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                    dir_fd=self.directory_fd,
                )
            except FileNotFoundError:
                if not self.create:
                    raise OperationalFailure("OPERATIONAL_STATE_MISSING") from None
                self.data = new_state()
                self.save()
            else:
                with os.fdopen(descriptor, "rb") as source:
                    if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                        raise OperationalFailure("INVALID_OPERATIONAL_STATE")
                    payload = source.read(65537)
                if len(payload) > 65536:
                    raise OperationalFailure("INVALID_OPERATIONAL_STATE")
                self.data = validate_state(json.loads(payload))
        except BaseException:
            self.close()
            raise
        return self

    def save(self) -> None:
        temporary = f".operational-{uuid4()}.tmp"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=self.directory_fd,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
                json.dump(self.data, destination, indent=2, allow_nan=False)
                destination.write("\n")
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(
                temporary,
                self.path.name,
                src_dir_fd=self.directory_fd,
                dst_dir_fd=self.directory_fd,
            )
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary, dir_fd=self.directory_fd)

    def close(self) -> None:
        for name in ("lock_fd", "directory_fd"):
            descriptor = getattr(self, name)
            setattr(self, name, -1)
            if descriptor >= 0:
                os.close(descriptor)

    def __exit__(
        self,
        kind: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def check_document(document: Document | None, state: dict[str, Any]) -> None:
    if (
        document is None
        or str(document.id) != state.get("document_id")
        or str(document.upload_request_id) != state["upload_request_id"]
        or document.sha256 != state["sha256"]
        or document.filename != state["filename"]
        or document.media_type != "application/pdf"
    ):
        raise OperationalFailure("DOCUMENT_OWNERSHIP_MISMATCH")


def question(state: dict[str, Any]) -> str:
    return f"Who is the owner? [operational:{state['api_nonce']}]"


def check_turn(turn: ChatTurn | None, state: dict[str, Any]) -> None:
    fingerprint = hashlib.sha256(
        unicodedata.normalize("NFC", question(state)).strip().encode()
    ).hexdigest()
    if (
        turn is None
        or str(turn.id) != state.get("turn_id")
        or str(turn.session_id) != state.get("session_id")
        or str(turn.client_request_id) != state["chat_request_id"]
        or str(turn.claim_token) != state.get("turn_token")
        or turn.question_fingerprint != fingerprint
    ):
        raise OperationalFailure("TURN_OWNERSHIP_MISMATCH")


def accelerate_worker(document: Document, state: dict[str, Any], now: datetime) -> None:
    check_document(document, state)
    if (
        document.status != "PROCESSING"
        or str(document.claim_token) != state.get("claim_token")
        or str(document.current_attempt_id) != state.get("attempt_id")
        or document.deadline_at <= now
    ):
        raise OperationalFailure("WORKER_INTERRUPTION_WINDOW_MISSED")
    document.lease_until = now - timedelta(seconds=1)


def _json(response: httpx.Response, expected: int = 200) -> dict[str, Any]:
    if response.status_code != expected:
        raise OperationalFailure("UNEXPECTED_HTTP_STATUS", response.status_code)
    result = response.json()
    if not isinstance(result, dict):
        raise OperationalFailure("INVALID_HTTP_RESPONSE")
    return result


def _poll(
    label: str, operation: Callable[[], dict[str, Any] | None], seconds: float = 90
) -> dict[str, Any]:
    deadline, report_at = time.monotonic() + seconds, time.monotonic() + 10
    while time.monotonic() < deadline:
        result = operation()
        if result is not None:
            return result
        if time.monotonic() >= report_at:
            print(json.dumps({"phase": label, "status": "waiting"}), flush=True)
            report_at = time.monotonic() + 10
        time.sleep(0.2)
    raise OperationalFailure(f"{label.upper().replace('-', '_')}_TIMEOUT")


class Checks:
    def __init__(
        self,
        settings: Settings,
        engine: Engine,
        state: StateFile,
        api: httpx.Client,
        provider: httpx.Client,
    ) -> None:
        self.settings, self.engine, self.file = settings, engine, state
        self.state, self.api, self.provider = state.data, api, provider

    def verify_dependencies(self) -> None:
        health = _json(self.provider.get("/health"))
        if health.get("fixture") is not True or health.get("barrier_version") != 1:
            raise OperationalFailure("FIXTURE_BARRIER_REQUIRED")
        _json(self.provider.get("/stats"))  # Authenticates the exact fixture-only key.
        with Session(self.engine) as database:
            if (
                database.scalar(text("SELECT current_database()")) != "vault_test"
                or database.scalar(text("SELECT current_schema()")) != "public"
            ):
                raise OperationalFailure("REFUSING_NONISOLATED_DATABASE")

    def _document(self, database: Session, *, lock: bool = False) -> Document:
        document = database.get(
            Document,
            UUID(self.state["document_id"]),
            options=[defer(Document.extracted_text)],
            with_for_update=lock,
        )
        check_document(document, self.state)
        assert document is not None
        return document

    def _barrier(self, nonce: str) -> dict[str, Any]:
        value = _json(self.provider.get(f"/barriers/{nonce}")).get("barrier")
        if not isinstance(value, dict):
            raise OperationalFailure("BARRIER_MISSING")
        return value

    def _arm(self, nonce: str, phase: str) -> None:
        _json(
            self.provider.post(
                "/barriers/arm", json={"nonce": nonce, "phase": phase, "ttl_seconds": 60}
            )
        )

    def _release(self, nonce: str) -> None:
        _json(self.provider.post(f"/barriers/{nonce}/release", json={}))

    def _ready_window(self, barrier: dict[str, Any]) -> int | None:
        if not barrier.get("entered") or barrier.get("active") != 1:
            return None
        remaining = min(
            float(barrier["remaining_seconds"]),
            self.settings.openai_generation_timeout_seconds - float(barrier["elapsed_seconds"]),
        )
        if remaining < 5:
            raise OperationalFailure("INTERRUPTION_WINDOW_MISSED")
        return int(remaining - 2)

    def _assert_unpublished(self, database: Session, identifier: UUID) -> None:
        if (
            database.scalar(
                select(func.count())
                .select_from(DocumentChunk)
                .where(DocumentChunk.document_id == identifier)
            )
            or database.get(DocumentInsights, identifier) is not None
        ):
            raise OperationalFailure("PARTIAL_PUBLICATION_DETECTED")

    def prepare_worker(self) -> dict[str, Any]:
        if self.state.get("worker_verified"):
            raise OperationalFailure("WORKER_PHASE_ALREADY_VERIFIED")
        self._arm(self.state["worker_nonce"], "analysis")
        if "document_id" not in self.state:
            with Session(self.engine) as database:
                existing = database.scalar(
                    select(Document.id).where(
                        Document.upload_request_id == UUID(self.state["upload_request_id"])
                    )
                )
            if existing is not None:
                self.state["document_id"] = str(existing)
            else:
                result = _json(
                    self.api.post(
                        "/api/v1/documents/upload",
                        files={
                            "file": (
                                self.state["filename"],
                                base64.b64decode(self.state["fixture_pdf"]),
                                "application/pdf",
                            )
                        },
                        headers={"Idempotency-Key": self.state["upload_request_id"]},
                    ),
                    202,
                )
                self.state["document_id"] = _uuid(result["document_id"])
            self.file.save()

        def ready() -> dict[str, Any] | None:
            with Session(self.engine) as database:
                document = self._document(database)
                if document.status in {"FAILED", "COMPLETED"}:
                    raise OperationalFailure("WORKER_INTERRUPTION_WINDOW_MISSED")
                barrier = self._barrier(self.state["worker_nonce"])
                window = self._ready_window(barrier)
                if (
                    document.status != "PROCESSING"
                    or document.stage != "ANALYSING"
                    or window is None
                ):
                    return None
                self._assert_unpublished(database, document.id)
                self.state.update(
                    attempt_id=_uuid(str(document.current_attempt_id)),
                    claim_token=_uuid(str(document.claim_token)),
                )
                self.file.save()
                return {
                    "phase": "ready-to-kill-worker",
                    "document_id": str(document.id),
                    "kill_within_seconds": window,
                }

        return _poll("prepare-worker", ready, 25)

    def recover_worker(self) -> dict[str, Any]:
        if self.state.get("worker_verified"):
            return {"phase": "worker-recovery-verified", "replayed_check": True}
        if not self.state.get("worker_expired"):
            with Session(self.engine) as database, database.begin():
                document = self._document(database, lock=True)
                original = database.get(ProcessingAttempt, UUID(self.state["attempt_id"]))
                if (
                    original is None
                    or original.document_id != document.id
                    or str(original.claim_token) != self.state["claim_token"]
                ):
                    raise OperationalFailure("ATTEMPT_OWNERSHIP_MISMATCH")
                if original.outcome != "INTERRUPTED":
                    self._assert_unpublished(database, document.id)
                    accelerate_worker(document, self.state, datetime.now(UTC))
                    self.state["clock_accelerations"].append("owned_worker_lease")
            self.state["worker_expired"] = True
        self.file.save()
        self._release(self.state["worker_nonce"])

        def verified() -> dict[str, Any] | None:
            with Session(self.engine) as database, database.begin():
                document = self._document(database, lock=True)
                if document.status == "FAILED":
                    raise OperationalFailure("WORKER_RECOVERY_FAILED")
                if (
                    document.status == "PENDING"
                    and document.error
                    and document.error.get("code") == "WORKER_INTERRUPTED"
                    and not self.state.get("retry_wait_accelerated")
                ):
                    document.next_dispatch_at = datetime.now(UTC) - timedelta(seconds=1)
                    self.state["retry_wait_accelerated"] = True
                    self.state["clock_accelerations"].append("owned_retry_wait")
                    self.file.save()
                if document.status != "COMPLETED":
                    return None
                attempts = database.scalars(
                    select(ProcessingAttempt)
                    .where(ProcessingAttempt.document_id == document.id)
                    .order_by(ProcessingAttempt.attempt_number)
                ).all()
                chunks = database.execute(
                    select(DocumentChunk.id, DocumentChunk.attempt_id).where(
                        DocumentChunk.document_id == document.id
                    )
                ).all()
                insights = database.get(DocumentInsights, document.id)
                if (
                    len(attempts) != 2
                    or str(attempts[0].id) != self.state["attempt_id"]
                    or attempts[0].outcome != "INTERRUPTED"
                    or attempts[1].outcome != "SUCCEEDED"
                    or document.published_attempt_id != attempts[1].id
                    or not chunks
                    or any(attempt_id != attempts[1].id for _, attempt_id in chunks)
                    or insights is None
                    or insights.attempt_id != attempts[1].id
                ):
                    raise OperationalFailure("WORKER_RECOVERY_INVARIANT_FAILED")
                self.state["worker_verified"] = True
                self.file.save()
                return {
                    "phase": "worker-recovery-verified",
                    "document_id": str(document.id),
                    "attempt_outcomes": [attempt.outcome for attempt in attempts],
                    "published_chunks": len(chunks),
                    "parent_confirmed_kill": True,
                    "clock_accelerated": bool(self.state["clock_accelerations"]),
                }

        result = _poll("recover-worker", verified)
        if (
            _json(self.api.get(f"/api/v1/documents/{self.state['document_id']}"))["status"]
            != "COMPLETED"
        ):
            raise OperationalFailure("RECOVERED_DOCUMENT_NOT_VISIBLE")
        return result

    def _session_owned(self, database: Session) -> ChatSession:
        self._document(database)
        session = database.get(ChatSession, UUID(self.state["session_id"]))
        members = database.scalars(
            select(ChatSessionDocument.document_id).where(
                ChatSessionDocument.session_id == UUID(self.state["session_id"])
            )
        ).all()
        if (
            session is None
            or session.title != f"Operational {self.state['run_id']}"
            or list(members) != [UUID(self.state["document_id"])]
        ):
            raise OperationalFailure("SESSION_OWNERSHIP_MISMATCH")
        return session

    def prepare_api(self) -> dict[str, Any]:
        if not self.state.get("worker_verified") or self.state.get("api_verified"):
            raise OperationalFailure("API_PHASE_ORDER_INVALID")
        self._arm(self.state["api_nonce"], "answer")
        if "session_id" not in self.state:
            title = f"Operational {self.state['run_id']}"
            with Session(self.engine) as database:
                existing_sessions = database.scalars(
                    select(ChatSession.id)
                    .join(ChatSessionDocument, ChatSessionDocument.session_id == ChatSession.id)
                    .where(
                        ChatSession.title == title,
                        ChatSessionDocument.document_id == UUID(self.state["document_id"]),
                    )
                ).all()
            if len(existing_sessions) > 1:
                raise OperationalFailure("SESSION_OWNERSHIP_AMBIGUOUS")
            if existing_sessions:
                self.state["session_id"] = str(existing_sessions[0])
            else:
                session = _json(
                    self.api.post(
                        "/api/v1/chat/sessions",
                        json={"document_id": self.state["document_id"], "title": title},
                    ),
                    201,
                )
                self.state["session_id"] = _uuid(session["id"])
            self.file.save()
        with Session(self.engine) as database:
            self._session_owned(database)
        try:
            response = self.api.post(
                f"/api/v1/chat/sessions/{self.state['session_id']}/messages",
                json={
                    "question": question(self.state),
                    "client_request_id": self.state["chat_request_id"],
                },
                timeout=1,
            )
            if response.status_code != 409:
                raise OperationalFailure("API_INTERRUPTION_WINDOW_MISSED", response.status_code)
        except httpx.ReadTimeout:
            pass  # Accepted work is shielded from this deliberate client disconnect.

        def ready() -> dict[str, Any] | None:
            with Session(self.engine) as database:
                turn = database.scalar(
                    select(ChatTurn).where(
                        ChatTurn.session_id == UUID(self.state["session_id"]),
                        ChatTurn.client_request_id == UUID(self.state["chat_request_id"]),
                    )
                )
                if turn is None:
                    return None
                if turn.status != "PENDING":
                    raise OperationalFailure("API_INTERRUPTION_WINDOW_MISSED")
                window = self._ready_window(self._barrier(self.state["api_nonce"]))
                if window is None:
                    return None
                assistant = database.scalar(
                    select(ChatMessage).where(
                        ChatMessage.turn_id == turn.id, ChatMessage.role == "assistant"
                    )
                )
                if (
                    assistant is None
                    or assistant.status != "PENDING"
                    or assistant.content is not None
                ):
                    raise OperationalFailure("PARTIAL_ASSISTANT_PUBLICATION_DETECTED")
                self.state.update(
                    turn_id=str(turn.id),
                    turn_token=str(turn.claim_token),
                    turn_calls=turn.ai_calls_reserved,
                )
                check_turn(turn, self.state)
                self.file.save()
                return {
                    "phase": "ready-to-kill-api",
                    "turn_id": str(turn.id),
                    "kill_within_seconds": window,
                }

        return _poll("prepare-api", ready, 20)

    def _counts(self) -> dict[str, int]:
        data = _json(self.provider.get("/stats"))
        fingerprint = hashlib.sha256(question(self.state).encode()).hexdigest()
        return {
            "embeddings": int(data["counts"].get("embeddings", 0)),
            "planner_for_question": int(data["planner_questions"].get(fingerprint, 0)),
            "answer_for_question": int(data["answer_questions"].get(fingerprint, 0)),
        }

    def recover_api(self) -> dict[str, Any]:
        if self.state.get("api_verified"):
            return {"phase": "api-recovery-verified", "replayed_check": True}
        baseline = self._counts()
        with Session(self.engine) as database, database.begin():
            # Match the application's session -> turn lock order.
            session = database.get(
                ChatSession, UUID(self.state["session_id"]), with_for_update=True
            )
            owned = self._session_owned(database)
            turn = database.get(ChatTurn, UUID(self.state["turn_id"]), with_for_update=True)
            check_turn(turn, self.state)
            assert turn is not None
            if session is None or session.id != owned.id:
                raise OperationalFailure("API_INTERRUPTION_WINDOW_MISSED")
            if turn.status == "PENDING" and session.active_turn_id == turn.id:
                now = datetime.now(UTC)
                turn.deadline_at, turn.expires_at = (
                    now - timedelta(seconds=2),
                    now - timedelta(seconds=1),
                )
                self.state["clock_accelerations"].append("owned_turn_expiry")
            elif (
                turn.status != "FAILED"
                or turn.response_status != 504
                or not turn.error
                or turn.error.get("code") != "CHAT_TIMEOUT"
            ):
                raise OperationalFailure("API_INTERRUPTION_WINDOW_MISSED")
        self.file.save()
        self._release(self.state["api_nonce"])
        endpoint = f"/api/v1/chat/sessions/{self.state['session_id']}/messages"
        payload = {
            "question": question(self.state),
            "client_request_id": self.state["chat_request_id"],
        }

        def replayed() -> dict[str, Any] | None:
            response = self.api.post(endpoint, json=payload)
            if (
                response.status_code == 409
                and response.json().get("error", {}).get("code") == "CHAT_IN_PROGRESS"
            ):
                return None
            if (
                response.status_code != 504
                or response.json().get("error", {}).get("code") != "CHAT_TIMEOUT"
            ):
                raise OperationalFailure(
                    "API_RECOVERY_DID_NOT_REPLAY_TIMEOUT", response.status_code
                )
            return cast(dict[str, Any], response.json())

        saved = _poll("recover-api", replayed)
        repeated = self.api.post(endpoint, json=payload)
        if repeated.status_code != 504 or repeated.json() != saved or self._counts() != baseline:
            raise OperationalFailure("REPLAY_REGENERATED_OR_CHANGED_RESULT")
        history = _json(self.api.get(endpoint))
        if not any(
            item["turn_id"] == self.state["turn_id"]
            and item["role"] == "assistant"
            and item["status"] == "FAILED"
            and item["turn_status"] == "FAILED"
            for item in history["items"]
        ):
            raise OperationalFailure("FAILED_TURN_NOT_VISIBLE_IN_HISTORY")
        with Session(self.engine) as database:
            turn = database.get(ChatTurn, UUID(self.state["turn_id"]))
            check_turn(turn, self.state)
            assert turn is not None
            if turn.response_body != saved or turn.ai_calls_reserved != self.state["turn_calls"]:
                raise OperationalFailure("REPLAY_ADMISSION_CHANGED")
        new_payload = {
            "question": "Who is the owner?",
            "client_request_id": self.state["new_question_request_id"],
        }
        result = _json(self.api.post(endpoint, json=new_payload, timeout=15))
        if (
            result.get("status") != "COMPLETED"
            or not result.get("citations")
            or "Test Operator" not in result.get("answer", "")
        ):
            raise OperationalFailure("NEW_QUESTION_AFTER_RECOVERY_FAILED")
        if any(
            source["document_id"] != self.state["document_id"] for source in result["citations"]
        ):
            raise OperationalFailure("NEW_QUESTION_SOURCE_SCOPE_FAILED")
        self.state["api_verified"] = True
        self.file.save()
        return {
            "phase": "api-recovery-verified",
            "turn_id": self.state["turn_id"],
            "replay_status": 504,
            "provider_counts_unchanged": True,
            "new_question_status": 200,
            "parent_confirmed_kill": True,
            "clock_accelerated": "owned_turn_expiry" in self.state["clock_accelerations"],
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "phase", choices=("prepare-worker", "recover-worker", "prepare-api", "recover-api")
    )
    parser.add_argument("--isolated", action="store_true")
    parser.add_argument("--confirm-worker-killed", action="store_true")
    parser.add_argument("--confirm-api-killed", action="store_true")
    parser.add_argument("--state", type=Path)
    arguments = parser.parse_args(argv)
    engine: Engine | None = None
    try:
        if not arguments.isolated:
            raise OperationalFailure("EXPLICIT_ISOLATED_FLAG_REQUIRED")
        assert_confirmation(
            arguments.phase, arguments.confirm_worker_killed, arguments.confirm_api_killed
        )
        settings = get_settings()
        assert_isolated(settings, arguments.isolated)
        with StateFile(
            settings, arguments.state, create=arguments.phase == "prepare-worker"
        ) as state:
            engine = create_engine(
                settings.database_url,
                hide_parameters=True,
                pool_size=2,
                max_overflow=0,
                connect_args={
                    "connect_timeout": 3,
                    "options": "-c statement_timeout=3000 -c lock_timeout=2000",
                },
            )
            with (
                httpx.Client(base_url=API_URL, timeout=5, trust_env=False) as api,
                httpx.Client(
                    base_url=STUB_URL,
                    timeout=5,
                    trust_env=False,
                    headers={"Authorization": f"Bearer {TEST_KEY}"},
                ) as provider,
            ):
                checks = Checks(settings, engine, state, api, provider)
                checks.verify_dependencies()
                action = {
                    "prepare-worker": checks.prepare_worker,
                    "recover-worker": checks.recover_worker,
                    "prepare-api": checks.prepare_api,
                    "recover-api": checks.recover_api,
                }[arguments.phase]
                result = action()
                print(json.dumps(result), flush=True)
        return 0
    except OperationalFailure as error:
        print(
            json.dumps(
                {"phase": arguments.phase, "failed": error.code, "http_status": error.http_status}
            ),
            flush=True,
        )
        return 1
    except (
        ConfigurationError,
        SQLAlchemyError,
        httpx.HTTPError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ):
        print(
            json.dumps(
                {"phase": arguments.phase, "failed": "OPERATIONAL_DEPENDENCY_OR_STATE_FAILURE"}
            ),
            flush=True,
        )
        return 1
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
