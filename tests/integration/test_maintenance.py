"""Actual private-schema retention and tempdir-only orphan deletion safety."""

import asyncio
import os
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.orm import Session
from test_processing_claims import ClaimDatabase
from test_processing_claims import claim_database as claim_database
from test_processing_claims import new_document as new_document
from test_retrieval import Published
from test_retrieval import publish as publish
from test_retrieval import retrieval_engine as retrieval_engine

from app.adapters.storage import LocalFileStorage, LockedFile, StaleFile
from app.db.models import (
    ChatMessage,
    ChatSession,
    ChatSessionDocument,
    ChatTurn,
    Document,
    PerformanceSample,
    ProcessingAttempt,
)
from app.domain.conversation import TurnReservation
from app.services import maintenance
from app.services.conversations import TurnStore
from app.services.maintenance import run_maintenance

pytestmark = [
    pytest.mark.integration,
    pytest.mark.enable_socket,
    pytest.mark.skipif(
        os.environ.get("VAULT_INTEGRATION_TESTS") != "1", reason="requires isolated Compose stack"
    ),
]


@pytest.fixture(autouse=True)
def isolated_tables(claim_database: ClaimDatabase) -> None:
    assert claim_database.settings.postgres_db == "vault_test"
    with Session(claim_database.engine) as database, database.begin():
        schema = database.scalar(text("SELECT current_schema()"))
        assert isinstance(schema, str) and schema.startswith("claims_test_")
        database.execute(text("TRUNCATE TABLE documents, chat_sessions CASCADE"))


@pytest.fixture
def now() -> datetime:
    return datetime.now(UTC)


def put(storage: LocalFileStorage, *, original: bool = False) -> tuple[str, Path]:
    key, path = storage.create_staging()
    with storage.open_staging(key) as stream:
        stream.write(b"original")
        stream.flush()
        if original:
            storage.finalize(key)
            path = storage.root / "originals" / key
    return key, path


def aged(path: Path, now: datetime, delta: timedelta) -> None:
    timestamp = (now - delta).timestamp()
    os.utime(path, (timestamp, timestamp), follow_symlinks=False)


def reference(database: ClaimDatabase, new_document: Callable[..., UUID], key: str) -> UUID:
    identifier = new_document()
    with Session(database.engine) as session, session.begin():
        document = session.get(Document, identifier)
        assert document is not None
        document.storage_key = key
    return identifier


async def test_referenced_original_retained_and_old_orphans_deleted_recent_untouched(
    retrieval_engine: AsyncEngine,
    claim_database: ClaimDatabase,
    new_document: Callable[..., UUID],
    tmp_path: Path,
    now: datetime,
) -> None:
    storage = LocalFileStorage(tmp_path)
    key, referenced = put(storage, original=True)
    reference(claim_database, new_document, key)
    aged(referenced, now, timedelta(days=3))
    _, orphan = put(storage, original=True)
    aged(orphan, now, timedelta(days=2))
    _, staged = put(storage)
    aged(staged, now, timedelta(hours=2))
    _, recent = put(storage, original=True)
    aged(recent, now, timedelta(hours=12))
    result = await run_maintenance(retrieval_engine, storage, now)
    assert result.database_available is True
    assert result.originals_deleted == result.staging_deleted == 1
    assert result.referenced_skipped == 1
    assert referenced.read_bytes() == recent.read_bytes() == b"original"
    assert not orphan.exists() and not staged.exists()


async def test_active_writer_and_recent_staging_are_not_deleted(
    retrieval_engine: AsyncEngine,
    tmp_path: Path,
    now: datetime,
) -> None:
    storage = LocalFileStorage(tmp_path)
    key, path = storage.create_staging()
    stream = storage.open_staging(key)
    _, recent = put(storage)
    aged(recent, now, timedelta(minutes=30))
    try:
        stream.write(b"still uploading")
        stream.flush()
        aged(path, now, timedelta(hours=2))
        result = await run_maintenance(retrieval_engine, storage, now)
        assert result.staging_deleted == 0
        assert result.busy_or_changed_skipped == 1
        assert path.exists() and recent.exists()
    finally:
        stream.close()
    assert (await run_maintenance(retrieval_engine, storage, now)).staging_deleted == 1
    assert recent.exists()


async def test_telemetry_cutoff_is_strict_and_durable_document_attempt_chat_survive(
    retrieval_engine: AsyncEngine,
    claim_database: ClaimDatabase,
    tmp_path: Path,
    now: datetime,
    publish: Callable[[Sequence[Sequence[float]]], Published],
) -> None:
    published = publish([[1.0] + [0.0] * 1535])
    session_id = uuid4()
    with Session(claim_database.engine) as database, database.begin():
        database.add(ChatSession(id=session_id, title="Retained conversation", next_turn_number=1))
        database.flush()
        database.add(ChatSessionDocument(session_id=session_id, document_id=published.document_id))
        document = database.get(Document, published.document_id)
        assert document is not None
        attempt_id = document.published_attempt_id
    store = TurnStore(retrieval_engine, claim_database.settings)
    turn = await store.reserve(session_id, uuid4(), "Unsupported question?", str(uuid4()))
    assert isinstance(turn, TurnReservation)
    await store.complete(
        turn,
        {
            "answer": "No supporting evidence found.",
            "evidence_status": "insufficient",
            "citations": [],
        },
        timings={},
    )
    cutoff = now - timedelta(days=7)
    old_id, exact_id, recent_id = uuid4(), uuid4(), uuid4()
    with Session(claim_database.engine) as database, database.begin():
        for identifier, timestamp in (
            (old_id, cutoff - timedelta(microseconds=1)),
            (exact_id, cutoff),
            (recent_id, now),
        ):
            database.add(
                PerformanceSample(
                    id=identifier,
                    recorded_at=timestamp,
                    operation="retention.fixture",
                    duration_ms=1,
                    outcome="succeeded",
                    document_id=published.document_id,
                    attempt_id=attempt_id,
                    turn_id=turn.turn_id,
                )
            )
    result = await run_maintenance(retrieval_engine, LocalFileStorage(tmp_path), now)
    assert result.performance_samples_deleted == 1
    with Session(claim_database.engine) as database:
        assert set(database.scalars(select(PerformanceSample.id)).all()) == {exact_id, recent_id}
        assert database.get(Document, published.document_id) is not None
        assert database.get(ProcessingAttempt, attempt_id) is not None
        assert database.get(ChatSession, session_id) is not None
        assert database.get(ChatTurn, turn.turn_id) is not None
        assert (
            database.scalar(
                select(func.count())
                .select_from(ChatMessage)
                .where(ChatMessage.turn_id == turn.turn_id)
            )
            == 2
        )


async def test_preloaded_references_do_not_consume_orphan_deletion_budget(
    retrieval_engine: AsyncEngine,
    claim_database: ClaimDatabase,
    new_document: Callable[..., UUID],
    tmp_path: Path,
    now: datetime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(maintenance, "MAX_FILE_DELETIONS", 2)
    storage = LocalFileStorage(tmp_path)
    referenced = []
    for _ in range(3):
        key, path = put(storage, original=True)
        reference(claim_database, new_document, key)
        aged(path, now, timedelta(days=3))
        referenced.append(path)
    _, orphan = put(storage, original=True)
    aged(orphan, now, timedelta(days=2))
    result = await run_maintenance(retrieval_engine, storage, now)
    assert result.referenced_skipped == 3
    assert result.originals_deleted == 1
    assert not orphan.exists() and all(path.exists() for path in referenced)


async def test_file_and_telemetry_deletion_batches_are_bounded(
    retrieval_engine: AsyncEngine,
    claim_database: ClaimDatabase,
    tmp_path: Path,
    now: datetime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(maintenance, "MAX_FILE_DELETIONS", 2)
    monkeypatch.setattr(maintenance, "MAX_PERFORMANCE_DELETIONS", 2)
    storage = LocalFileStorage(tmp_path)
    paths = [put(storage, original=True)[1] for _ in range(5)]
    for path in paths:
        aged(path, now, timedelta(days=2))
    with Session(claim_database.engine) as database, database.begin():
        database.add_all(
            [
                PerformanceSample(
                    recorded_at=now - timedelta(days=8),
                    operation="batch.fixture",
                    duration_ms=1,
                    outcome="succeeded",
                )
                for _ in range(5)
            ]
        )
    result = await run_maintenance(retrieval_engine, storage, now)
    assert result.performance_samples_deleted == result.originals_deleted == 2
    assert result.file_limit_reached is True
    assert sum(path.exists() for path in paths) == 3
    with Session(claim_database.engine) as database:
        assert database.scalar(select(func.count()).select_from(PerformanceSample)) == 3


async def test_fresh_reference_committed_after_preload_preserves_original(
    retrieval_engine: AsyncEngine,
    claim_database: ClaimDatabase,
    new_document: Callable[..., UUID],
    tmp_path: Path,
    now: datetime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = LocalFileStorage(tmp_path)
    key, path = put(storage, original=True)
    aged(path, now, timedelta(days=2))
    original_lock = storage.lock_candidate

    def lock_and_reference(candidate: StaleFile) -> LockedFile | None:
        locked = original_lock(candidate)
        assert locked is not None
        reference(claim_database, new_document, key)
        return locked

    monkeypatch.setattr(storage, "lock_candidate", lock_and_reference)
    result = await run_maintenance(retrieval_engine, storage, now)
    assert result.originals_deleted == 0 and result.referenced_skipped == 1
    assert path.exists()


async def test_database_unavailable_means_no_file_deletion(tmp_path: Path, now: datetime) -> None:
    class UnavailableDatabase:
        def begin(self) -> Any:
            raise OperationalError("simulated outage", {}, RuntimeError("unavailable"))

    storage = LocalFileStorage(tmp_path)
    _, path = put(storage, original=True)
    aged(path, now, timedelta(days=2))
    result = await run_maintenance(cast(AsyncEngine, UnavailableDatabase()), storage, now)
    assert result.database_available is False
    assert result.originals_deleted == result.staging_deleted == 0
    assert path.exists()


async def test_database_recheck_failure_preserves_file_and_releases_lock(
    retrieval_engine: AsyncEngine,
    tmp_path: Path,
    now: datetime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = LocalFileStorage(tmp_path)
    _, path = put(storage, original=True)
    aged(path, now, timedelta(days=2))

    async def unavailable(engine: AsyncEngine, key: str) -> bool:
        raise OperationalError("simulated outage", {}, RuntimeError("unavailable"))

    monkeypatch.setattr(maintenance, "_is_referenced", unavailable)
    result = await run_maintenance(retrieval_engine, storage, now)
    assert result.database_available is False and path.exists()
    (candidate,) = storage.old_candidates("originals", (now - timedelta(days=1)).timestamp())
    locked = storage.lock_candidate(candidate)
    assert locked is not None
    locked.close()


async def test_cancellation_during_reference_recheck_releases_lock_without_delete(
    retrieval_engine: AsyncEngine,
    tmp_path: Path,
    now: datetime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = LocalFileStorage(tmp_path)
    _, path = put(storage, original=True)
    aged(path, now, timedelta(days=2))
    checking = asyncio.Event()

    async def blocked(engine: AsyncEngine, key: str) -> bool:
        checking.set()
        await asyncio.Event().wait()
        return False

    monkeypatch.setattr(maintenance, "_is_referenced", blocked)
    pending = asyncio.create_task(run_maintenance(retrieval_engine, storage, now))
    try:
        await asyncio.wait_for(checking.wait(), timeout=2)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    finally:
        if not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
    assert path.exists()
    (candidate,) = storage.old_candidates("originals", (now - timedelta(days=1)).timestamp())
    locked = storage.lock_candidate(candidate)
    assert locked is not None
    locked.close()
