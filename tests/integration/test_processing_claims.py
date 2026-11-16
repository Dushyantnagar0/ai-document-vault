"""Real PostgreSQL claim/fencing checks, isolated from the running reconciler."""

import os
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory
from celery.exceptions import SoftTimeLimitExceeded
from sqlalchemy import Engine, create_engine, event, func, select, text
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateSchema, DropSchema

from app.core.config import Settings
from app.db.models import Document, IndexProfile, ProcessingAttempt
from app.db.profiles import profile_values
from app.domain.documents import ProcessingError
from app.services import ingestion
from app.services.ingestion import Claim, LostClaim, claim_document, heartbeat, reserve_call

pytestmark = [
    pytest.mark.integration,
    pytest.mark.enable_socket,
    pytest.mark.skipif(
        os.environ.get("VAULT_INTEGRATION_TESTS") != "1", reason="requires isolated Compose stack"
    ),
]


def test_new_claim_clears_current_dispatch_error_but_preserves_attempt_history(
    claim_database: "ClaimDatabase", new_document: Callable[..., UUID]
) -> None:
    identifier = new_document()
    with Session(claim_database.engine) as session, session.begin():
        document = session.get(Document, identifier)
        assert document is not None
        document.error = {"code": "DISPATCH_UNAVAILABLE", "message": "Transport was unavailable"}
    require_claim(claim_database, identifier)
    with Session(claim_database.engine) as session:
        document = session.get(Document, identifier)
        assert document is not None and document.error is None


def test_soft_worker_limit_retries_without_calling_provider_or_publishing(
    claim_database: "ClaimDatabase",
    new_document: Callable[..., UUID],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identifier = new_document()

    def interrupted(*args: Any) -> Any:
        raise SoftTimeLimitExceeded()

    monkeypatch.setattr("app.adapters.extraction.extractor_for", interrupted)
    ingestion.process_document(claim_database.engine, identifier, claim_database.settings)
    with Session(claim_database.engine) as session:
        document = session.get(Document, identifier)
        assert document is not None
        assert document.status == "PENDING" and document.stage == "RETRY_WAIT"
        assert document.ai_calls_reserved == 0 and document.published_attempt_id is None
        assert document.error is not None and document.error["code"] == "PROCESSING_INTERRUPTED"
        attempt = session.get(ProcessingAttempt, document.current_attempt_id)
        assert attempt is not None and attempt.outcome == "RETRY"


@dataclass(frozen=True)
class ClaimDatabase:
    engine: Engine
    settings: Settings
    profile_id: UUID


@pytest.fixture(scope="module")
def claim_database() -> Iterator[ClaimDatabase]:
    settings = Settings(_env_file=None)
    assert settings.postgres_db == "vault_test", "Refusing a non-test database"
    schema = f"claims_test_{uuid4().hex}"
    administration = create_engine(
        settings.database_url, hide_parameters=True, connect_args={"connect_timeout": 3}
    )
    engine = create_engine(
        settings.database_url,
        pool_size=5,
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
    created = False
    try:
        with administration.begin() as connection:
            namespace = connection.scalar(
                text(
                    "SELECT n.nspname FROM pg_extension e "
                    "JOIN pg_namespace n ON n.oid=e.extnamespace WHERE e.extname='vector'"
                )
            )
            assert namespace == "public", "Apply the foundation migration before these tests"
            connection.execute(CreateSchema(schema))
            created = True
        # Apply the actual migration functions in this schema. Unlike public
        # tables, these deliberately expired fake-file rows cannot be dispatched
        # by the running API's recovery loop or consumed by its worker.
        scripts = ScriptDirectory.from_config(Config("alembic.ini"))
        with engine.begin() as connection:
            with Operations.context(MigrationContext.configure(connection)):
                for revision in reversed(list(scripts.iterate_revisions("head", "base"))):
                    revision.module.upgrade()
        with Session(engine) as session, session.begin():
            profile = IndexProfile(**profile_values(settings))
            session.add(profile)
            session.flush()
            profile_id = profile.id
        yield ClaimDatabase(engine, settings, profile_id)
    finally:
        engine.dispose()
        try:
            if created:
                with administration.begin() as connection:
                    connection.execute(DropSchema(schema, cascade=True))
        finally:
            administration.dispose()


@pytest.fixture
def new_document(claim_database: ClaimDatabase) -> Callable[..., UUID]:
    def create(*, created_at: datetime | None = None) -> UUID:
        now = datetime.now(UTC)
        identifier = uuid4()
        with Session(claim_database.engine) as session, session.begin():
            session.add(
                Document(
                    id=identifier,
                    filename="claims-fixture.pdf",
                    media_type="application/pdf",
                    byte_size=8,
                    sha256="a" * 64,
                    storage_key=str(uuid4()),
                    profile_id=claim_database.profile_id,
                    request_fingerprint="b" * 64,
                    status="PENDING",
                    stage="QUEUED",
                    created_at=created_at or now,
                    updated_at=now,
                    next_dispatch_at=now - timedelta(seconds=1),
                    deadline_at=now + timedelta(seconds=1800),
                    warnings=[],
                    attempt_count=0,
                    ai_calls_reserved=0,
                )
            )
        return identifier

    return create


def require_claim(database: ClaimDatabase, identifier: UUID) -> Claim:
    result = claim_document(database.engine, identifier, database.settings, str(uuid4()))
    assert result is not None
    return result


def interrupt_for_retry(database: ClaimDatabase, claim: Claim) -> None:
    now = datetime.now(UTC)
    with Session(database.engine) as session, session.begin():
        document = session.get(Document, claim.document_id, with_for_update=True)
        attempt = session.get(ProcessingAttempt, claim.attempt_id)
        assert document is not None and attempt is not None
        document.status, document.stage = "PENDING", "RETRY_WAIT"
        document.claim_token, document.lease_until = None, None
        document.next_dispatch_at = now - timedelta(seconds=1)
        attempt.outcome, attempt.ended_at = "INTERRUPTED", now


def test_concurrent_claim_has_one_winner_and_one_attempt(
    claim_database: ClaimDatabase, new_document: Callable[..., UUID]
) -> None:
    identifier = new_document()
    barrier = threading.Barrier(2, timeout=2)

    def compete() -> Claim | None:
        barrier.wait()
        return claim_document(
            claim_database.engine, identifier, claim_database.settings, str(uuid4())
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(compete) for _ in range(2)]
        claims = [future.result(timeout=5) for future in futures]
    winners = [claim for claim in claims if claim is not None]
    assert len(winners) == 1
    with Session(claim_database.engine) as session:
        assert (
            session.scalar(
                select(func.count())
                .select_from(ProcessingAttempt)
                .where(ProcessingAttempt.document_id == identifier)
            )
            == 1
        )
        document = session.get(Document, identifier)
        assert document is not None
        assert document.attempt_count == 1
        assert document.current_attempt_id == winners[0].attempt_id


@pytest.mark.parametrize("superseded", [False, True])
def test_invalid_or_superseded_token_cannot_reserve_or_heartbeat(
    claim_database: ClaimDatabase, new_document: Callable[..., UUID], superseded: bool
) -> None:
    claim = require_claim(claim_database, new_document())
    if superseded:
        interrupt_for_retry(claim_database, claim)
        current = require_claim(claim_database, claim.document_id)
        invalid = claim
    else:
        current = claim
        invalid = replace(claim, token=uuid4())
    with pytest.raises(LostClaim):
        reserve_call(claim_database.engine, invalid, claim_database.settings)
    assert heartbeat(claim_database.engine, invalid) is False
    with Session(claim_database.engine) as session:
        document = session.get(Document, claim.document_id)
        assert document is not None
        assert document.claim_token == current.token
        assert document.ai_calls_reserved == 0


def test_expired_lease_cannot_renew_or_reserve(
    claim_database: ClaimDatabase, new_document: Callable[..., UUID]
) -> None:
    claim = require_claim(claim_database, new_document())
    expired = datetime.now(UTC) - timedelta(seconds=1)
    with Session(claim_database.engine) as session, session.begin():
        document = session.get(Document, claim.document_id)
        assert document is not None
        document.lease_until = expired
    assert heartbeat(claim_database.engine, claim) is False
    with pytest.raises(LostClaim):
        reserve_call(claim_database.engine, claim, claim_database.settings)
    with Session(claim_database.engine) as session:
        document = session.get(Document, claim.document_id)
        assert document is not None
        assert document.lease_until == expired
        assert document.ai_calls_reserved == 0


def test_call_budget_is_cumulative_across_attempts(
    claim_database: ClaimDatabase, new_document: Callable[..., UUID]
) -> None:
    settings = claim_database.settings.model_copy(update={"max_document_ai_calls": 2})
    first = require_claim(claim_database, new_document())
    assert reserve_call(claim_database.engine, first, settings) > 0
    interrupt_for_retry(claim_database, first)
    second = require_claim(claim_database, first.document_id)
    assert reserve_call(claim_database.engine, second, settings) > 0
    with pytest.raises(ProcessingError) as failure:
        reserve_call(claim_database.engine, second, settings)
    assert failure.value.code == "AI_CALL_BUDGET_EXHAUSTED"
    with Session(claim_database.engine) as session:
        document = session.get(Document, first.document_id)
        assert document is not None
        assert document.ai_calls_reserved == 2
        assert document.attempt_count == 2


@pytest.mark.parametrize("age,accepted", [(899, True), (900, False)])
def test_direct_first_claim_obeys_wait_budget(
    claim_database: ClaimDatabase,
    new_document: Callable[..., UUID],
    monkeypatch: pytest.MonkeyPatch,
    age: int,
    accepted: bool,
) -> None:
    now = datetime.now(UTC)
    identifier = new_document(created_at=now - timedelta(seconds=age))
    monkeypatch.setattr(ingestion, "utcnow", lambda: now)
    claim = claim_document(claim_database.engine, identifier, claim_database.settings, None)
    assert (claim is not None) is accepted
    with Session(claim_database.engine) as session:
        document = session.get(Document, identifier)
        assert document is not None
        assert document.status == ("PROCESSING" if accepted else "FAILED")
        assert document.attempt_count == (1 if accepted else 0)


def test_reservation_rechecks_clock_after_waiting_for_row_lock(
    claim_database: ClaimDatabase,
    new_document: Callable[..., UUID],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = require_claim(claim_database, new_document())
    before = datetime.now(UTC)
    clock = SimpleNamespace(now=before)
    monkeypatch.setattr(ingestion, "utcnow", lambda: clock.now)
    attempted_select = threading.Event()
    main_thread = threading.get_ident()

    def observe_lock_query(
        connection: Any,
        cursor: Any,
        statement: str,
        parameters: Any,
        context: Any,
        executemany: bool,
    ) -> None:
        if threading.get_ident() != main_thread and "FOR UPDATE" in statement.upper():
            attempted_select.set()

    event.listen(claim_database.engine, "before_cursor_execute", observe_lock_query)
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        with Session(claim_database.engine) as session, session.begin():
            document = session.get(Document, claim.document_id, with_for_update=True)
            assert document is not None
            document.lease_until = before + timedelta(seconds=10)
            session.flush()
            future = executor.submit(
                reserve_call, claim_database.engine, claim, claim_database.settings
            )
            assert attempted_select.wait(timeout=2), "Worker never issued its locking SELECT"
            assert not future.done(), "Reservation unexpectedly bypassed the held row lock"
            clock.now = before + timedelta(seconds=20)
        with pytest.raises(LostClaim):
            future.result(timeout=5)
    finally:
        event.remove(claim_database.engine, "before_cursor_execute", observe_lock_query)
        executor.shutdown(wait=True, cancel_futures=True)
