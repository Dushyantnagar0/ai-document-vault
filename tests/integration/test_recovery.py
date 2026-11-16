"""Real isolated PostgreSQL recovery with deterministic transport substitutes."""

import asyncio
import os
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import Session
from test_processing_claims import (
    ClaimDatabase,
    require_claim,
)
from test_processing_claims import (
    claim_database as claim_database,
)
from test_processing_claims import (
    new_document as new_document,
)

from app.db.models import Document, ProcessingAttempt
from app.domain.documents import ProcessingError
from app.services.ingestion import Claim, claim_document
from app.services.recovery import dispatch_document, recover_once

pytestmark = [
    pytest.mark.integration,
    pytest.mark.enable_socket,
    pytest.mark.skipif(
        os.environ.get("VAULT_INTEGRATION_TESTS") != "1", reason="requires isolated Compose stack"
    ),
]


@pytest.fixture(autouse=True)
def retire_test_work(claim_database: ClaimDatabase) -> Iterator[None]:
    """Prevent intentionally pending work from leaking into the next scan case."""
    yield
    with Session(claim_database.engine) as session, session.begin():
        now = datetime.now(UTC)
        session.execute(
            update(ProcessingAttempt)
            .where(ProcessingAttempt.outcome == "RUNNING")
            .values(outcome="FAILED", ended_at=now)
        )
        session.execute(
            update(Document)
            .where(Document.status.in_(["PENDING", "PROCESSING"]))
            .values(
                status="FAILED",
                stage="FAILED",
                terminal_at=now,
                claim_token=None,
                lease_until=None,
                dispatch_reserved_until=None,
            )
        )


@pytest.fixture
async def recovery_engine(claim_database: ClaimDatabase) -> AsyncIterator[AsyncEngine]:
    assert claim_database.settings.postgres_db == "vault_test"
    with Session(claim_database.engine) as session:
        schema = session.scalar(text("SELECT current_schema()"))
    assert isinstance(schema, str) and schema.startswith("claims_test_")
    # The fixture owns a generated schema, never public application tables.
    assert schema.removeprefix("claims_test_").isalnum()
    engine = create_async_engine(
        claim_database.settings.async_database_url,
        pool_size=3,
        max_overflow=0,
        pool_timeout=2,
        hide_parameters=True,
        connect_args={
            "connect_timeout": 3,
            "options": (
                f"-c search_path={schema},public -c statement_timeout=5000 -c lock_timeout=3000"
            ),
        },
    )
    try:
        yield engine
    finally:
        await engine.dispose()


class RecordingDispatcher:
    def __init__(self) -> None:
        self.calls: list[UUID] = []

    def submit(self, document_id: UUID) -> None:
        self.calls.append(document_id)


class UnavailableDispatcher:
    def submit(self, document_id: UUID) -> None:
        raise ProcessingError("DISPATCH_UNAVAILABLE", "Simulated transport outage", retryable=True)


class BlockingDispatcher(RecordingDispatcher):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def submit(self, document_id: UUID) -> None:
        super().submit(document_id)
        self.started.set()
        if not self.release.wait(timeout=3):
            raise ProcessingError("DISPATCH_UNAVAILABLE", "Test publication barrier expired")


async def test_missed_publication_is_recovered_and_bookkept(
    claim_database: ClaimDatabase,
    recovery_engine: AsyncEngine,
    new_document: Callable[..., UUID],
) -> None:
    identifier = new_document()
    dispatcher = RecordingDispatcher()
    await recover_once(recovery_engine, claim_database.settings, dispatcher)
    assert dispatcher.calls == [identifier]
    with Session(claim_database.engine) as session:
        document = session.get(Document, identifier)
        assert document is not None
        assert document.status == "PENDING"
        assert document.last_dispatched_at is not None
        assert document.stage == "QUEUED" and document.error is None
        assert document.dispatch_reserved_until is None
    await recover_once(recovery_engine, claim_database.settings, dispatcher)
    assert dispatcher.calls == [identifier]


async def test_publish_failure_keeps_durable_recoverable_intent(
    claim_database: ClaimDatabase,
    recovery_engine: AsyncEngine,
    new_document: Callable[..., UUID],
) -> None:
    identifier = new_document()
    await dispatch_document(recovery_engine, UnavailableDispatcher(), identifier)
    with Session(claim_database.engine) as session:
        document = session.get(Document, identifier)
        assert document is not None
        assert document.status == "PENDING"
        assert document.stage == "DISPATCH_PENDING"
        assert document.error is not None
        assert document.error["code"] == "DISPATCH_UNAVAILABLE"
        assert document.last_dispatched_at is None
        assert document.dispatch_reserved_until is None
    dispatcher = RecordingDispatcher()
    await recover_once(recovery_engine, claim_database.settings, dispatcher)
    assert dispatcher.calls == [identifier]
    with Session(claim_database.engine) as session:
        document = session.get(Document, identifier)
        assert document is not None
        assert document.status == "PENDING"
        assert document.last_dispatched_at is not None
        assert document.stage == "QUEUED" and document.error is None


async def test_immediate_and_recovery_race_has_single_publisher(
    claim_database: ClaimDatabase,
    recovery_engine: AsyncEngine,
    new_document: Callable[..., UUID],
) -> None:
    identifier = new_document()
    dispatcher = BlockingDispatcher()
    immediate = asyncio.create_task(dispatch_document(recovery_engine, dispatcher, identifier))
    try:
        assert await asyncio.wait_for(asyncio.to_thread(dispatcher.started.wait, 2), timeout=3)
        await recover_once(recovery_engine, claim_database.settings, dispatcher)
        assert dispatcher.calls == [identifier]
    finally:
        dispatcher.release.set()
        await asyncio.wait_for(immediate, timeout=5)
    with Session(claim_database.engine) as session:
        document = session.get(Document, identifier)
        assert document is not None
        assert document.dispatch_reserved_until is None
        assert document.last_dispatched_at is not None


async def test_fast_claim_survives_publication_bookkeeping(
    claim_database: ClaimDatabase,
    recovery_engine: AsyncEngine,
    new_document: Callable[..., UUID],
) -> None:
    identifier = new_document()

    class ClaimingDispatcher:
        claim: Claim | None = None

        def submit(self, document_id: UUID) -> None:
            self.claim = claim_document(
                claim_database.engine, document_id, claim_database.settings, "test-fast-claim"
            )

    dispatcher = ClaimingDispatcher()
    await dispatch_document(recovery_engine, dispatcher, identifier)
    assert dispatcher.claim is not None
    with Session(claim_database.engine) as session:
        document = session.get(Document, identifier)
        assert document is not None
        assert document.status == "PROCESSING"
        assert document.stage == "EXTRACTING"
        assert document.claim_token == dispatcher.claim.token
        assert document.current_attempt_id == dispatcher.claim.attempt_id
        assert document.dispatch_reserved_until is None
        assert document.last_dispatched_at is not None


async def test_expired_lease_interrupts_attempt_and_schedules_bounded_retry(
    claim_database: ClaimDatabase,
    recovery_engine: AsyncEngine,
    new_document: Callable[..., UUID],
) -> None:
    claim = require_claim(claim_database, new_document())
    with Session(claim_database.engine) as session, session.begin():
        document = session.get(Document, claim.document_id)
        assert document is not None
        document.lease_until = datetime.now(UTC) - timedelta(seconds=1)
    before = datetime.now(UTC)
    dispatcher = RecordingDispatcher()
    await recover_once(recovery_engine, claim_database.settings, dispatcher)
    after = datetime.now(UTC)
    assert dispatcher.calls == []
    with Session(claim_database.engine) as session:
        document = session.get(Document, claim.document_id)
        attempt = session.get(ProcessingAttempt, claim.attempt_id)
        assert document is not None and attempt is not None
        assert document.status == "PENDING" and document.stage == "RETRY_WAIT"
        assert document.claim_token is None and document.lease_until is None
        assert (
            before + timedelta(seconds=15)
            <= document.next_dispatch_at
            <= after + timedelta(seconds=15)
        )
        assert attempt.outcome == "INTERRUPTED"
        assert attempt.ended_at is not None


async def test_expired_work_is_filtered_before_scan_limit(
    claim_database: ClaimDatabase,
    recovery_engine: AsyncEngine,
    new_document: Callable[..., UUID],
) -> None:
    healthy = [require_claim(claim_database, new_document()) for _ in range(100)]
    expired = require_claim(claim_database, new_document())
    with Session(claim_database.engine) as session, session.begin():
        document = session.get(Document, expired.document_id)
        assert document is not None
        document.lease_until = datetime.now(UTC) - timedelta(seconds=1)
    dispatcher = RecordingDispatcher()
    await recover_once(recovery_engine, claim_database.settings, dispatcher)
    assert dispatcher.calls == []
    with Session(claim_database.engine) as session:
        document = session.get(Document, expired.document_id)
        attempt = session.get(ProcessingAttempt, expired.attempt_id)
        assert document is not None and attempt is not None
        assert document.status == "PENDING"
        assert attempt.outcome == "INTERRUPTED"
        statuses = session.scalars(
            select(Document.status).where(Document.id.in_([claim.document_id for claim in healthy]))
        ).all()
        assert len(statuses) == 100
        assert set(statuses) == {"PROCESSING"}
