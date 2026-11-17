"""Worker transport, a shared-volume probe and document ingestion entrypoint."""

import os
import stat
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from celery import Celery
from kombu.entity import Queue
from sqlalchemy import Engine

from app.core.config import Settings, get_settings
from app.db.session import worker_engine

QUEUE_NAME = "ingestion"
PROBE_TASK_NAME = "app.tasks.probe"
INGEST_TASK_NAME = "app.tasks.ingest"


@lru_cache(maxsize=1)
def _worker_engine_for_process(process_id: int, settings: Settings) -> Engine:
    """Create on first task only; the PID key prevents reuse across prefork."""
    return worker_engine(settings)


def _document_uuid(value: str) -> UUID:
    if not isinstance(value, str) or len(value) != 36:
        raise ValueError("Document ID must be a canonical UUID")
    try:
        identifier = UUID(value)
    except ValueError:
        raise ValueError("Document ID must be a canonical UUID") from None
    if str(identifier) != value:
        raise ValueError("Document ID must be a canonical UUID")
    return identifier


def write_probe_receipt(storage_dir: Path, nonce: str) -> None:
    """Write one canonical UUID receipt; duplicate deliveries are harmless.

    Directory-relative, no-follow/exclusive opens prevent a symlink or uploaded
    filename from redirecting the probe outside its dedicated shared directory.
    """
    if not isinstance(nonce, str) or len(nonce) != 36:
        raise ValueError("Probe nonce must be a canonical UUID")
    try:
        canonical = str(UUID(nonce))
    except ValueError:
        raise ValueError("Probe nonce must be a canonical UUID") from None
    if canonical != nonce:
        raise ValueError("Probe nonce must be a canonical UUID")

    probe_dir = storage_dir / ".probes"
    probe_dir.mkdir(mode=0o700, exist_ok=True)
    directory_fd = os.open(probe_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    temporary_name = f".{nonce}.{uuid4()}.tmp"
    try:
        try:
            info = os.stat(nonce, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("Probe receipt must be a regular file") from None
        receipt_fd = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        with os.fdopen(receipt_fd, "w", encoding="ascii") as receipt:
            receipt.write(nonce)
        os.replace(temporary_name, nonce, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
    finally:
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.close(directory_fd)


def worker_configuration(settings: Settings) -> dict[str, Any]:
    """No result store or automatic application retries; PostgreSQL owns work."""
    return {
        "broker_url": settings.redis_url.get_secret_value(),
        "result_backend": None,
        "task_ignore_result": True,
        "task_store_errors_even_if_ignored": False,
        "task_serializer": "json",
        "accept_content": ["json"],
        "task_default_queue": QUEUE_NAME,
        # Kombu's Queue constructor has no annotations; this fixed name is local.
        "task_queues": (Queue(QUEUE_NAME),),  # type: ignore[no-untyped-call]
        "task_create_missing_queues": False,
        "task_acks_late": True,
        "task_reject_on_worker_lost": True,
        "task_publish_retry": False,
        "worker_prefetch_multiplier": 1,
        "worker_concurrency": settings.celery_worker_concurrency,
        "task_soft_time_limit": 540,
        "task_time_limit": 600,
        "broker_transport_options": {
            "visibility_timeout": 3600,
            "socket_connect_timeout": 2,
            "socket_timeout": 2,
            "retry_on_timeout": False,
        },
        "broker_connection_timeout": 2,
        # A transport outage must not permanently remove the consumer. These
        # connection retries do not reserve business attempts or provider calls;
        # per-document deadlines/call budgets still govern accepted work.
        "broker_connection_retry": True,
        "broker_connection_retry_on_startup": True,
        "broker_connection_max_retries": None,
        "timezone": "UTC",
        "enable_utc": True,
    }


def create_celery_app(settings: Settings | None = None) -> Celery:
    """Configure lazily so importing definitions does not load credentials."""
    application = Celery("document_vault", set_as_current=False)
    application.add_defaults(
        lambda: worker_configuration(settings if settings is not None else get_settings())
    )

    def probe(nonce: str) -> None:
        configured = settings if settings is not None else get_settings()
        write_probe_receipt(configured.local_storage_dir, nonce)

    def ingest(task: Any, document_id: str) -> None:
        identifier = _document_uuid(document_id)
        configured = settings if settings is not None else get_settings()
        engine = _worker_engine_for_process(os.getpid(), configured)
        # Import orchestration only when consuming valid work, not during
        # worker registration/readiness or before the prefork child starts.
        from app.services.ingestion import process_document

        task_id = getattr(task.request, "id", None)
        process_document(
            engine, identifier, configured, task_id if isinstance(task_id, str) else None
        )

    application.task(name=PROBE_TASK_NAME, ignore_result=True, max_retries=0, shared=False)(probe)
    application.task(
        name=INGEST_TASK_NAME, bind=True, ignore_result=True, max_retries=0, shared=False
    )(ingest)
    return application


app = create_celery_app()
