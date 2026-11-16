"""Isolated-operation guards/state ownership and deterministic fixture barriers."""

import base64
import hashlib
import importlib.util
import json
import stat
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from uuid import UUID, uuid4

import httpx
import pymupdf
import pytest

from app.core.config import Settings
from app.db.models import ChatTurn, Document
from scripts import operational_checks as operations

pytestmark = pytest.mark.unit


def settings(tmp_path: Path, **overrides: Any) -> Settings:
    return Settings(
        _env_file=None,
        **{
            "openai_api_key": operations.TEST_KEY,
            "postgres_db": "vault_test",
            "postgres_user": "vault_test",
            "redis_url": "redis://redis:6379/1",
            "local_storage_dir": tmp_path,
            **overrides,
        },
    )


def environment() -> dict[str, str]:
    return {"VAULT_INTEGRATION_TESTS": "1", "OPENAI_BASE_URL": operations.STUB_URL + "/v1"}


@pytest.mark.parametrize(
    "change", ["flag", "database", "user", "host", "key", "redis", "stub", "optin"]
)
def test_isolation_guard_rejects_any_nonisolated_component(tmp_path: Path, change: str) -> None:
    overrides: dict[str, Any] = {}
    changed = {
        "database": ("postgres_db", "personal"),
        "user": ("postgres_user", "personal"),
        "host": ("postgres_host", "personal"),
        "key": ("openai_api_key", "different-placeholder"),
        "redis": ("redis_url", "redis://redis:6379/0"),
    }
    if change in changed:
        key, value = changed[change]
        overrides[key] = value
    env = environment()
    if change == "stub":
        env["OPENAI_BASE_URL"] = "https://example.invalid/v1"
    if change == "optin":
        env.pop("VAULT_INTEGRATION_TESTS")
    with pytest.raises(operations.OperationalFailure, match="REFUSING_NONISOLATED"):
        operations.assert_isolated(settings(tmp_path, **overrides), change != "flag", env)


def test_exact_isolated_configuration_is_accepted(tmp_path: Path) -> None:
    operations.assert_isolated(settings(tmp_path), True, environment())


@pytest.mark.parametrize(
    "phase,worker,api",
    [
        ("recover-worker", False, True),
        ("recover-api", True, False),
    ],
)
def test_recovery_requires_operator_kill_confirmation(phase: str, worker: bool, api: bool) -> None:
    with pytest.raises(operations.OperationalFailure, match="KILL_CONFIRMATION_REQUIRED"):
        operations.assert_confirmation(phase, worker, api)


def test_cli_refuses_before_loading_settings_or_touching_services(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden() -> Settings:
        pytest.fail("Refused command must not load settings or construct dependencies")

    monkeypatch.setattr(operations, "get_settings", forbidden)
    assert operations.main(["prepare-worker"]) == 1
    assert operations.main(["recover-worker", "--isolated"]) == 1
    output = capsys.readouterr().out
    assert "EXPLICIT_ISOLATED_FLAG_REQUIRED" in output
    assert "WORKER_KILL_CONFIRMATION_REQUIRED" in output


def test_state_is_saved_before_work_and_pdf_contains_exact_barrier_marker(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    with operations.StateFile(configured, None, create=True) as state:
        original = dict(state.data)
        assert state.path.exists()
        assert stat.S_IMODE(state.path.stat().st_mode) == 0o600
        payload = base64.b64decode(state.data["fixture_pdf"])
        assert hashlib.sha256(payload).hexdigest() == state.data["sha256"]
        with pymupdf.open(stream=payload, filetype="pdf") as document:  # type: ignore[no-untyped-call]
            content = "".join(page.get_text() for page in document)
        assert f"[operational:{state.data['worker_nonce']}]" in content
        assert "Owner: Test Operator" in content
    with operations.StateFile(configured, None) as state:
        assert state.data == original


def test_state_lock_is_exclusive_and_unrelated_file_is_never_overwritten(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    with operations.StateFile(configured, None, create=True):
        with pytest.raises(operations.OperationalFailure, match="STATE_BUSY"):
            with operations.StateFile(configured, None):
                pass
    path = tmp_path / "operational-checks" / "unrelated.json"
    path.write_text('{"unrelated": true}')
    with pytest.raises(operations.OperationalFailure, match="STATE_OWNER_MISMATCH"):
        with operations.StateFile(configured, path, create=True):
            pass
    assert json.loads(path.read_text()) == {"unrelated": True}


def test_state_symlink_and_outside_location_are_rejected(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    with pytest.raises(operations.OperationalFailure, match="STATE_LOCATION_INVALID"):
        operations.StateFile(configured, tmp_path / "outside.json", create=True)
    root = tmp_path / "operational-checks"
    root.mkdir()
    target = tmp_path / "target"
    target.write_text("preserve")
    path = root / "symlink.json"
    path.symlink_to(target)
    with pytest.raises(OSError):
        with operations.StateFile(configured, path, create=True):
            pass
    assert target.read_text() == "preserve"


def owned_document(state: dict[str, Any]) -> Document:
    return cast(
        Document,
        SimpleNamespace(
            id=UUID(state["document_id"]),
            upload_request_id=UUID(state["upload_request_id"]),
            sha256=state["sha256"],
            filename=state["filename"],
            media_type="application/pdf",
            status="PROCESSING",
            claim_token=UUID(state["claim_token"]),
            current_attempt_id=UUID(state["attempt_id"]),
            deadline_at=datetime.now(UTC) + timedelta(minutes=10),
            lease_until=datetime.now(UTC) + timedelta(minutes=3),
            ai_calls_reserved=2,
        ),
    )


def worker_state() -> dict[str, Any]:
    state = operations.new_state()
    state.update(document_id=str(uuid4()), attempt_id=str(uuid4()), claim_token=str(uuid4()))
    return state


@pytest.mark.parametrize("field", ["id", "upload_request_id", "sha256", "filename", "media_type"])
def test_unrelated_document_cannot_be_clock_accelerated(field: str) -> None:
    state = worker_state()
    document = owned_document(state)
    before = document.lease_until
    setattr(document, field, uuid4() if field in {"id", "upload_request_id"} else "different")
    with pytest.raises(operations.OperationalFailure, match="DOCUMENT_OWNERSHIP_MISMATCH"):
        operations.accelerate_worker(document, state, datetime.now(UTC))
    assert document.lease_until == before


def test_clock_acceleration_only_changes_matching_claim_lease() -> None:
    state = worker_state()
    document = owned_document(state)
    now = datetime.now(UTC)
    deadline = document.deadline_at
    operations.accelerate_worker(document, state, now)
    assert document.lease_until == now - timedelta(seconds=1)
    assert document.status == "PROCESSING" and document.ai_calls_reserved == 2
    assert document.deadline_at == deadline
    document.claim_token = uuid4()
    with pytest.raises(operations.OperationalFailure, match="WINDOW_MISSED"):
        operations.accelerate_worker(document, state, now)


def test_turn_ownership_includes_question_request_session_and_claim() -> None:
    state = worker_state()
    state.update(session_id=str(uuid4()), turn_id=str(uuid4()), turn_token=str(uuid4()))
    turn = cast(
        ChatTurn,
        SimpleNamespace(
            id=UUID(state["turn_id"]),
            session_id=UUID(state["session_id"]),
            client_request_id=UUID(state["chat_request_id"]),
            claim_token=UUID(state["turn_token"]),
            question_fingerprint=hashlib.sha256(operations.question(state).encode()).hexdigest(),
        ),
    )
    operations.check_turn(turn, state)
    turn.client_request_id = uuid4()
    with pytest.raises(operations.OperationalFailure, match="TURN_OWNERSHIP_MISMATCH"):
        operations.check_turn(turn, state)


@pytest.fixture
def provider_fixture(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "fixtures" / "provider_stub.py"
    spec = importlib.util.spec_from_file_location("operational_provider_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_barrier_is_one_shot_and_does_not_match_unrelated_phase_or_nonce(
    provider_fixture: ModuleType,
) -> None:
    registry = provider_fixture.BarrierRegistry()
    nonce = str(uuid4())
    registry.arm(nonce, "analysis", 2)
    registry.wait("answer", {"question": f"[operational:{nonce}]"})
    registry.wait("analysis", {"text": f"[operational:{uuid4()}]"})
    assert registry.snapshot(nonce)["consumed"] is False
    pending = threading.Thread(
        target=registry.wait,
        args=("analysis", {"items": [{"text": f"[operational:{nonce}]"}]}),
        daemon=True,
    )
    pending.start()
    try:
        deadline = time.monotonic() + 1
        while not registry.snapshot(nonce)["entered"] and time.monotonic() < deadline:
            time.sleep(0.005)
        assert registry.snapshot(nonce)["active"] == 1
        # A retry with the same marker must not block behind the first request.
        registry.wait("analysis", {"text": f"[operational:{nonce}]"})
        assert registry.arm(nonce, "analysis", 2)["consumed"] is True
    finally:
        registry.release(nonce)
        pending.join(timeout=2)
    assert not pending.is_alive()
    assert registry.snapshot(nonce)["active"] == 0


def test_barrier_expiry_and_lifetime_validation(provider_fixture: ModuleType) -> None:
    clock = SimpleNamespace(now=0.0)
    registry = provider_fixture.BarrierRegistry(clock=lambda: clock.now)
    nonce = str(uuid4())
    registry.arm(nonce, "answer", 1)
    clock.now = 2
    registry.wait("answer", {"question": f"[operational:{nonce}]"})
    assert registry.snapshot(nonce)["expired"] is True
    assert registry.snapshot(nonce)["consumed"] is False
    for ttl in (0, 61, True, float("nan")):
        with pytest.raises(ValueError):
            registry.arm(str(uuid4()), "answer", ttl)


@pytest.mark.enable_socket
def test_barrier_http_control_requires_exact_fixture_auth(provider_fixture: ModuleType) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), provider_fixture.Handler)
    serving = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    serving.start()
    nonce = str(uuid4())
    try:
        with httpx.Client(
            base_url=f"http://127.0.0.1:{server.server_port}", trust_env=False
        ) as client:
            assert client.get("/health").json()["barrier_version"] == 1
            assert (
                client.post("/barriers/arm", json={"nonce": nonce, "phase": "answer"}).status_code
                == 403
            )
            headers = {"Authorization": provider_fixture.TEST_AUTHORIZATION}
            response = client.post(
                "/barriers/arm",
                headers=headers,
                json={"nonce": nonce, "phase": "answer", "ttl_seconds": 2},
            )
            assert response.status_code == 200
            assert client.get(f"/barriers/{nonce}").status_code == 403
            assert (
                client.get(f"/barriers/{nonce}", headers=headers).json()["barrier"]["consumed"]
                is False
            )
            released = client.post(f"/barriers/{nonce}/release", headers=headers, json={})
            assert released.json()["barrier"]["released"] is True
            embedded = client.post(
                "/v1/embeddings",
                headers=headers,
                json={"model": "embedding-fixture", "input": ["Owner: Test Operator"]},
            )
            assert embedded.status_code == 200
            assert len(embedded.json()["data"][0]["embedding"]) == 1536
            analyzed = client.post(
                "/v1/chat/completions",
                headers=headers,
                json={
                    "model": "generation-fixture",
                    "messages": [
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "items": [
                                        {"source_id": str(uuid4()), "text": "Owner: Test Operator"}
                                    ]
                                }
                            ),
                        }
                    ],
                    "response_format": {"json_schema": {"name": "AnalysisOutput"}},
                },
            )
            assert analyzed.status_code == 200
            result = json.loads(analyzed.json()["choices"][0]["message"]["content"])
            assert result["summary"][0]["support_quotes"][0]["quote"] == "Owner: Test Operator"
    finally:
        server.shutdown()
        server.server_close()
        serving.join(timeout=2)
