"""Real HTTP middleware and private PostgreSQL route telemetry, without providers."""

import json
import os
from collections import Counter
from collections.abc import AsyncIterator, Callable
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from sqlalchemy import delete, select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask
from starlette.responses import JSONResponse
from test_processing_claims import ClaimDatabase
from test_processing_claims import claim_database as claim_database
from test_processing_claims import new_document as new_document
from test_retrieval import retrieval_engine as retrieval_engine

from app.db.models import PerformanceSample
from app.main import create_app
from app.services import telemetry

pytestmark = [
    pytest.mark.integration,
    pytest.mark.enable_socket,
    pytest.mark.skipif(
        os.environ.get("VAULT_INTEGRATION_TESTS") != "1", reason="requires isolated Compose stack"
    ),
]


@pytest.fixture(autouse=True)
def empty_private_samples(claim_database: ClaimDatabase) -> None:
    assert claim_database.settings.postgres_db == "vault_test", "Refusing non-test telemetry writes"
    with Session(claim_database.engine) as session, session.begin():
        schema = session.scalar(text("SELECT current_schema()"))
        assert isinstance(schema, str) and schema.startswith("claims_test_")
        session.execute(delete(PerformanceSample))


@pytest.fixture
async def client(retrieval_engine: AsyncEngine) -> AsyncIterator[httpx.AsyncClient]:
    application = create_app()
    application.state.engine = retrieval_engine
    # No lifespan: only the private schema is visible and no worker, maintenance
    # or recovery loop runs. HTTP responses still execute the real middleware.
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://telemetry-test"
    ) as instance:
        yield instance


def samples(database: ClaimDatabase) -> list[dict[str, Any]]:
    with Session(database.engine) as session:
        rows = session.scalars(
            select(PerformanceSample).order_by(PerformanceSample.recorded_at)
        ).all()
        return [
            {
                column.name: getattr(row, column.name)
                for column in PerformanceSample.__table__.columns
            }
            for row in rows
        ]


async def test_http_templates_status_and_duration_persist_without_sensitive_inputs(
    client: httpx.AsyncClient,
    claim_database: ClaimDatabase,
    new_document: Callable[..., UUID],
) -> None:
    identifier = new_document()
    query_marker = "synthetic-private-query-marker"
    key_marker = "synthetic-private-key-marker"
    question_marker = "synthetic-private-question-marker"
    request_id = str(uuid4())
    response = await client.get(
        f"/api/v1/documents/{identifier}",
        params={"cursor": query_marker},
        headers={"X-Request-ID": request_id, "Authorization": f"Bearer {key_marker}"},
    )
    assert response.status_code == 200
    assert response.json()["id"] == str(identifier)
    invalid = await client.get(f"/api/v1/documents/invalid-{query_marker}")
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "VALIDATION_ERROR"
    invalid_question = await client.post(
        f"/api/v1/chat/sessions/{uuid4()}/messages",
        json={"question": question_marker, "client_request_id": "invalid-request-uuid"},
        headers={"X-API-Key": key_marker},
    )
    assert invalid_question.status_code == 422
    assert question_marker not in invalid_question.text

    persisted = samples(claim_database)
    assert len(persisted) == 3  # ASGI completion includes each response's background sample.
    assert Counter(row["operation"] for row in persisted) == {
        "http.GET /api/v1/documents/{document_id}": 2,
        "http.POST /api/v1/chat/sessions/{session_id}/messages": 1,
    }
    assert Counter(row["outcome"] for row in persisted) == {"SUCCEEDED": 1, "FAILED": 2}
    assert Counter(row["error_class"] for row in persisted) == {None: 1, "HTTP_422": 2}
    assert all(
        isinstance(row["duration_ms"], (int, float)) and row["duration_ms"] >= 0
        for row in persisted
    )
    serialized = json.dumps(persisted, default=str)
    for forbidden in (str(identifier), request_id, query_marker, key_marker, question_marker):
        assert forbidden not in serialized
    assert all(row["document_id"] is None and row["turn_id"] is None for row in persisted)

    # Unmatched paths carry no route object and must never become raw-path labels.
    for _ in range(2):
        unknown = await client.get(f"/unknown-private-{uuid4()}?question={question_marker}")
        assert unknown.status_code == 404
        assert unknown.json()["error"]["code"] == "NOT_FOUND"
    assert samples(claim_database) == persisted


async def test_telemetry_write_failure_does_not_change_successful_http_response(
    client: httpx.AsyncClient,
    claim_database: ClaimDatabase,
    new_document: Callable[..., UUID],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identifier = new_document()
    attempted: list[AsyncEngine] = []

    def unavailable(engine: AsyncEngine) -> None:
        attempted.append(engine)
        raise OperationalError("INSERT", {}, RuntimeError("synthetic-telemetry-db-failure"))

    # Only the telemetry writer fails; the request still reads real PostgreSQL.
    monkeypatch.setattr(telemetry, "async_sessionmaker", unavailable)
    response = await client.get(f"/api/v1/documents/{identifier}")
    assert response.status_code == 200
    assert response.json()["id"] == str(identifier)
    assert response.headers["X-Request-ID"]
    assert "synthetic-telemetry-db-failure" not in response.text
    assert len(attempted) == 1
    assert samples(claim_database) == []


async def test_existing_response_background_and_real_telemetry_both_complete(
    retrieval_engine: AsyncEngine, claim_database: ClaimDatabase
) -> None:
    application = create_app()
    application.state.engine = retrieval_engine
    completed: list[str] = []

    @application.get("/telemetry-background-fixture")
    async def background_fixture() -> JSONResponse:
        return JSONResponse({"ok": True}, background=BackgroundTask(completed.append, "original"))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://telemetry-test"
    ) as instance:
        response = await instance.get("/telemetry-background-fixture")
    assert response.status_code == 200 and response.json() == {"ok": True}
    assert completed == ["original"]
    persisted = samples(claim_database)
    assert len(persisted) == 1
    assert persisted[0]["operation"] == "http.GET /telemetry-background-fixture"
    assert persisted[0]["outcome"] == "SUCCEEDED"
