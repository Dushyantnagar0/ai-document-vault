"""Real commit-ack ambiguity and immediate dispatch, isolated from worker recovery."""

import os
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any
from uuid import UUID, uuid4

import httpx
import pymupdf
import pytest
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory
from sqlalchemy import Engine, create_engine, func, select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSessionTransaction, create_async_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool
from sqlalchemy.schema import CreateSchema, DropSchema

from app.api import documents
from app.core.config import Settings
from app.db.models import Document, IndexProfile
from app.main import create_app

pytestmark = [
    pytest.mark.integration,
    pytest.mark.enable_socket,
    pytest.mark.skipif(
        os.environ.get("VAULT_INTEGRATION_TESTS") != "1", reason="requires isolated Compose stack"
    ),
]


@dataclass(frozen=True)
class AcceptanceDatabase:
    engine: Engine
    settings: Settings
    schema: str


@pytest.fixture(scope="module")
def acceptance_database() -> Iterator[AcceptanceDatabase]:
    settings = Settings(_env_file=None, openai_api_key="test-placeholder-never-a-real-key")
    assert settings.postgres_db == "vault_test", "Refusing a non-test database"
    schema = f"acceptance_test_{uuid4().hex}"
    administration = create_engine(
        settings.database_url, hide_parameters=True, connect_args={"connect_timeout": 3}
    )
    engine = create_engine(
        settings.database_url,
        pool_size=2,
        max_overflow=0,
        hide_parameters=True,
        connect_args={
            "connect_timeout": 3,
            "options": f"-c search_path={schema},public -c statement_timeout=5000",
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
        scripts = ScriptDirectory.from_config(Config("alembic.ini"))
        with engine.begin() as connection:
            with Operations.context(MigrationContext.configure(connection)):
                for revision in reversed(list(scripts.iterate_revisions("head", "base"))):
                    revision.module.upgrade()
        yield AcceptanceDatabase(engine, settings, schema)
    finally:
        engine.dispose()
        try:
            if created:
                with administration.begin() as connection:
                    connection.execute(DropSchema(schema, cascade=True))
        finally:
            administration.dispose()


@dataclass
class ObservingDispatcher:
    engine: Engine
    calls: list[UUID] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)

    def submit(self, document_id: UUID) -> None:
        self.calls.append(document_id)
        # This is a separate connection from upload acceptance and its dispatch
        # reservation. An uncommitted intent cannot be observed here.
        with Session(self.engine) as session:
            document = session.get(Document, document_id)
            self.observations.append(
                {}
                if document is None
                else {
                    "id": document.id,
                    "status": document.status,
                    "stage": document.stage,
                    "due": document.next_dispatch_at <= datetime.now(UTC),
                    "deadline_future": document.deadline_at > datetime.now(UTC),
                    "reserved": document.dispatch_reserved_until is not None,
                    "profile_exists": session.get(IndexProfile, document.profile_id) is not None,
                }
            )


@dataclass(frozen=True)
class AcceptanceClient:
    client: httpx.AsyncClient
    dispatcher: ObservingDispatcher


@pytest.fixture
async def acceptance_client(
    acceptance_database: AcceptanceDatabase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[AcceptanceClient]:
    settings = acceptance_database.settings.model_copy(update={"local_storage_dir": tmp_path})
    monkeypatch.setattr(documents, "get_settings", lambda: settings)
    engine = create_async_engine(
        settings.async_database_url,
        poolclass=NullPool,
        hide_parameters=True,
        connect_args={
            "connect_timeout": 3,
            "options": (
                f"-c search_path={acceptance_database.schema},public -c statement_timeout=5000"
            ),
        },
    )
    dispatcher = ObservingDispatcher(acceptance_database.engine)
    application = create_app()
    application.state.engine = engine
    application.state.dispatcher = dispatcher
    # No lifespan means no API reconciler and no worker can see this private
    # schema. The submitted task is only observed, never sent to Redis/OpenAI.
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://acceptance-test"
        ) as client:
            yield AcceptanceClient(client, dispatcher)
    finally:
        await engine.dispose()


def pdf_bytes() -> bytes:
    with pymupdf.open() as document:
        document.new_page().insert_text((72, 72), "Acceptance recovery fixture: limit is 72.")
        return document.tobytes()


async def test_lost_commit_acknowledgement_preserves_accepted_original_and_replays(
    acceptance_database: AcceptanceDatabase,
    acceptance_client: AcceptanceClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_exit = AsyncSessionTransaction.__aexit__
    injected = False

    async def lose_ack_after_real_commit(
        transaction: AsyncSessionTransaction,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        nonlocal injected
        await original_exit(transaction, exc_type, exc_value, traceback)
        if exc_type is None and not injected:
            injected = True
            raise OperationalError("COMMIT", {}, Exception("synthetic-commit-acknowledgement-lost"))

    monkeypatch.setattr(AsyncSessionTransaction, "__aexit__", lose_ack_after_real_commit)
    request_id = uuid4()
    body = pdf_bytes()
    first = await acceptance_client.client.post(
        "/api/v1/documents/upload",
        headers={"Idempotency-Key": str(request_id)},
        files={"file": ("commit-ack.pdf", body, "application/pdf")},
    )
    assert injected, "The intended real commit boundary was not exercised"
    assert first.status_code == 503, first.text
    assert first.json()["error"]["code"] == "DATABASE_UNAVAILABLE"
    assert "synthetic-commit-acknowledgement-lost" not in first.text
    assert acceptance_client.dispatcher.calls == []

    with Session(acceptance_database.engine) as session:
        saved = session.scalar(select(Document).where(Document.upload_request_id == request_id))
        assert saved is not None, "The injected lost acknowledgement must occur after DB commit"
        identifier = saved.id
        assert saved.status == "PENDING"
        assert saved.next_dispatch_at <= datetime.now(UTC)

    original = await acceptance_client.client.get(f"/api/v1/documents/{identifier}/file")
    assert original.status_code == 200
    assert original.content == body

    monkeypatch.setattr(AsyncSessionTransaction, "__aexit__", original_exit)
    replay = await acceptance_client.client.post(
        "/api/v1/documents/upload",
        headers={"Idempotency-Key": str(request_id)},
        files={"file": ("commit-ack.pdf", body, "application/pdf")},
    )
    assert replay.status_code == 202, replay.text
    assert replay.json()["document_id"] == str(identifier)
    after_replay = await acceptance_client.client.get(f"/api/v1/documents/{identifier}/file")
    assert after_replay.status_code == 200
    assert after_replay.content == body
    assert len(list((tmp_path / "originals").iterdir())) == 1
    assert list((tmp_path / "staging").iterdir()) == []
    with Session(acceptance_database.engine) as session:
        assert (
            session.scalar(
                select(func.count())
                .select_from(Document)
                .where(Document.upload_request_id == request_id)
            )
            == 1
        )


async def test_successful_upload_immediately_dispatches_only_committed_intent(
    acceptance_database: AcceptanceDatabase, acceptance_client: AcceptanceClient
) -> None:
    response = await acceptance_client.client.post(
        "/api/v1/documents/upload",
        headers={"Idempotency-Key": str(uuid4())},
        files={"file": ("immediate-dispatch.pdf", pdf_bytes(), "application/pdf")},
    )
    assert response.status_code == 202, response.text
    identifier = UUID(response.json()["document_id"])
    assert acceptance_client.dispatcher.calls == [identifier]
    assert acceptance_client.dispatcher.observations == [
        {
            "id": identifier,
            "status": "PENDING",
            "stage": "QUEUED",
            "due": True,
            "deadline_future": True,
            "reserved": True,
            "profile_exists": True,
        }
    ]
    with Session(acceptance_database.engine) as session:
        document = session.get(Document, identifier)
        assert document is not None
        assert document.last_dispatched_at is not None
        assert document.dispatch_reserved_until is None
