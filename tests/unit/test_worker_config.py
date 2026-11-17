"""Offline worker boundaries; real Redis consumption is an integration gate."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock, call
from uuid import uuid4

import pytest

from app.core.config import Settings, get_settings
from app.core.readiness import Readiness
from app.tasks import celery_app, health
from app.tasks.celery_app import (
    INGEST_TASK_NAME,
    PROBE_TASK_NAME,
    QUEUE_NAME,
    create_celery_app,
    write_probe_receipt,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for field in Settings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)
        monkeypatch.delenv(field, raising=False)
    get_settings.cache_clear()
    celery_app._worker_engine_for_process.cache_clear()


def test_queue_and_delivery_configuration(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key="test-placeholder",
        local_storage_dir=tmp_path,
    )
    application = create_celery_app(settings)
    configuration = application.conf
    assert configuration.broker_url == settings.redis_url.get_secret_value()
    assert configuration.result_backend is None
    assert configuration.task_ignore_result is True
    assert configuration.task_publish_retry is False
    assert configuration.task_create_missing_queues is False
    assert configuration.task_acks_late is True
    assert configuration.task_reject_on_worker_lost is True
    assert configuration.worker_prefetch_multiplier == 1
    assert configuration.worker_concurrency == settings.celery_worker_concurrency
    assert configuration.task_default_queue == QUEUE_NAME
    assert [queue.name for queue in configuration.task_queues] == [QUEUE_NAME]
    assert configuration.task_soft_time_limit == 540
    assert configuration.task_time_limit == 600
    assert configuration.broker_transport_options["visibility_timeout"] == 3600
    assert configuration.broker_connection_retry is True
    assert configuration.broker_connection_retry_on_startup is True
    assert configuration.broker_connection_max_retries is None
    assert configuration.accept_content == ["json"]
    assert PROBE_TASK_NAME in application.tasks
    assert INGEST_TASK_NAME in application.tasks
    assert application.tasks[INGEST_TASK_NAME].max_retries == 0
    assert application.tasks[PROBE_TASK_NAME].max_retries == 0
    nonce = str(uuid4())
    assert application.tasks[PROBE_TASK_NAME].run(nonce) is None
    assert (tmp_path / ".probes" / nonce).read_text() == nonce


@pytest.mark.parametrize(
    "nonce",
    [
        "",
        "../escape",
        "/tmp/escape",
        "x" * 1000,
        "a" * 36,
        "550E8400-E29B-41D4-A716-446655440000",
        "550e8400e29b41d4a716446655440000",
        None,
    ],
)
def test_invalid_probe_nonce_writes_nothing(tmp_path: Path, nonce: Any) -> None:
    with pytest.raises(ValueError, match="canonical UUID"):
        write_probe_receipt(tmp_path, nonce)
    assert not (tmp_path / ".probes").exists()


def test_probe_redelivery_and_symlinks(tmp_path: Path) -> None:
    nonce = str(uuid4())
    write_probe_receipt(tmp_path, nonce)
    write_probe_receipt(tmp_path, nonce)
    assert (tmp_path / ".probes" / nonce).read_text() == nonce
    target = tmp_path / "outside"
    target.write_text("unchanged")
    other_nonce = str(uuid4())
    (tmp_path / ".probes" / other_nonce).symlink_to(target)
    with pytest.raises(ValueError, match="regular file"):
        write_probe_receipt(tmp_path, other_nonce)
    assert target.read_text() == "unchanged"


def test_probe_directory_symlink_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / ".probes").symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        write_probe_receipt(tmp_path, str(uuid4()))
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("missing", [None, "ping", "registered", "active_queues"])
def test_health_requires_own_node_all_components(missing: str | None) -> None:
    node = "worker@test-host"
    inspector = Mock()
    inspector.ping.return_value = {node: {"ok": "pong"}}
    inspector.registered.return_value = {node: [PROBE_TASK_NAME, INGEST_TASK_NAME]}
    inspector.active_queues.return_value = {node: [{"name": QUEUE_NAME}]}
    if missing is not None:
        getattr(inspector, missing).return_value = {"worker@other": []}
    application = SimpleNamespace(control=SimpleNamespace(inspect=Mock(return_value=inspector)))
    assert health.worker_ready(application, node=node) is (missing is None)
    application.control.inspect.assert_called_once_with(destination=[node], timeout=2)


def test_health_rejects_wrong_task_and_queue() -> None:
    node = "worker@test-host"
    inspector = Mock()
    inspector.ping.return_value = {node: {"ok": "pong"}}
    inspector.registered.return_value = {node: ["unrelated.task"]}
    inspector.active_queues.return_value = {node: [{"name": "other"}]}
    application = SimpleNamespace(control=SimpleNamespace(inspect=Mock(return_value=inspector)))
    assert health.worker_ready(application, node=node) is False


def test_health_sanitizes_broker_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(_: Any) -> bool:
        raise RuntimeError("redis://:private-marker@host/0")

    monkeypatch.setattr(health, "worker_ready", fail)
    assert health.main() == 1
    assert "private-marker" not in capsys.readouterr().out


def test_ingestion_entrypoint_passes_uuid_and_task_id_and_reuses_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(
        _env_file=None, openai_api_key="test-placeholder", local_storage_dir=tmp_path
    )
    engine = Mock()
    make_engine = Mock(return_value=engine)
    process = Mock()
    ingestion_module = ModuleType("app.services.ingestion")
    monkeypatch.setattr(ingestion_module, "process_document", process, raising=False)
    monkeypatch.setitem(sys.modules, "app.services.ingestion", ingestion_module)
    monkeypatch.setattr(celery_app, "worker_engine", make_engine)
    application = create_celery_app(settings)
    task = application.tasks[INGEST_TASK_NAME]
    make_engine.assert_not_called()
    identifier = uuid4()
    task_id = str(uuid4())
    task.push_request(id=task_id)
    try:
        task.run(str(identifier))
        task.run(str(identifier))
    finally:
        task.pop_request()
    task.run(str(identifier))
    make_engine.assert_called_once_with(settings)
    process.assert_has_calls(
        [call(engine, identifier, settings, task_id)] * 2
        + [call(engine, identifier, settings, None)]
    )


@pytest.mark.parametrize(
    "identifier",
    ["", "../escape", "x" * 1000, "550E8400-E29B-41D4-A716-446655440000", None],
)
def test_invalid_ingestion_id_does_not_create_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, identifier: Any
) -> None:
    make_engine = Mock()
    monkeypatch.setattr(celery_app, "worker_engine", make_engine)
    settings = Settings(
        _env_file=None, openai_api_key="test-placeholder", local_storage_dir=tmp_path
    )
    task = create_celery_app(settings).tasks[INGEST_TASK_NAME]
    with pytest.raises(ValueError, match="canonical UUID"):
        task.run(identifier)
    make_engine.assert_not_called()


def test_worker_engine_cache_is_separated_by_process_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(_env_file=None, openai_api_key="test-placeholder")
    first, second = Mock(), Mock()
    make_engine = Mock(side_effect=[first, second])
    monkeypatch.setattr(celery_app, "worker_engine", make_engine)
    assert celery_app._worker_engine_for_process(100, settings) is first
    assert celery_app._worker_engine_for_process(100, settings) is first
    assert celery_app._worker_engine_for_process(101, settings) is second
    assert make_engine.call_count == 2


@pytest.mark.parametrize("missing_task", [None, PROBE_TASK_NAME, INGEST_TASK_NAME])
def test_health_requires_both_tasks(
    monkeypatch: pytest.MonkeyPatch, missing_task: str | None
) -> None:
    node = "worker@test-host"
    inspector = Mock()
    inspector.ping.return_value = {node: {"ok": "pong"}}
    inspector.registered.return_value = {
        node: [name for name in (PROBE_TASK_NAME, INGEST_TASK_NAME) if name != missing_task]
    }
    inspector.active_queues.return_value = {node: [{"name": QUEUE_NAME}]}
    application = SimpleNamespace(control=SimpleNamespace(inspect=Mock(return_value=inspector)))
    assert health.worker_ready(application, node=node) is (missing_task is None)
    monkeypatch.setattr(celery_app, "app", application)
    assert Readiness._worker_ready() is (missing_task is None)
