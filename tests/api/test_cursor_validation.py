"""Malformed opaque cursors are rejected before any application database access."""

import base64
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import conversations, documents, metrics
from app.api.documents import APIError, decode_cursor
from app.services.metrics import _decode_cursor

TIMESTAMP = "2026-11-16T00:00:00+00:00"
IDENTIFIER = "00000000-0000-0000-0000-000000000123"
ROUTES = (
    ("/api/v1/documents", [TIMESTAMP, IDENTIFIER], "INVALID_CURSOR"),
    ("/api/v1/chat/sessions", [TIMESTAMP, IDENTIFIER], "INVALID_CURSOR"),
    (
        "/api/v1/metrics/processing",
        [TIMESTAMP, IDENTIFIER, "2026-11-15T00:00:00Z", "2026-11-17T00:00:00Z"],
        "METRICS_CURSOR_INVALID",
    ),
)


def encoded(payload: Any) -> str:
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()


class NoDatabase:
    def connect(self) -> None:
        pytest.fail("Invalid cursors must not reach a database connection")


@pytest.fixture
async def client(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[httpx.AsyncClient]:
    application = FastAPI()
    for router in (documents.router, conversations.router, metrics.router):
        application.include_router(router)
    application.state.engine = NoDatabase()

    def forbidden_session(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Invalid cursors must not construct a database session")

    monkeypatch.setattr(documents, "async_sessionmaker", lambda *args, **kwargs: forbidden_session)
    monkeypatch.setattr(
        conversations, "async_sessionmaker", lambda *args, **kwargs: forbidden_session
    )

    @application.exception_handler(APIError)
    async def error_handler(request: Request, error: APIError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status,
            content={"error": {"code": error.code, "message": error.message}},
        )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://test"
    ) as instance:
        yield instance


INVALID_MEMBERS = [
    pytest.param(
        path,
        encoded([*payload[:index], value, *payload[index + 1 :]]),
        code,
        id=f"{path}-{index}-{type(value).__name__}",
    )
    for path, payload, code in ROUTES
    for index in range(len(payload))
    for value in (123, False, [], {}, None)
]

INVALID_SHAPES = [
    pytest.param(path, encoded(value), code, id=f"{path}-shape-{index}")
    for path, payload, code in ROUTES
    for index, value in enumerate(
        (None, True, 123, {}, "text", [], payload[:-1], [*payload, "extra"])
    )
]

INVALID_REPRESENTATIONS = [
    pytest.param(path, cursor, code, id=f"{path}-representation-{index}")
    for path, payload, code in ROUTES
    for index, cursor in enumerate(
        (
            "not-base64!",
            encoded(payload) + "!",
            "wA==",
            "e25vdC1qc29ufQ==",
            "A",
        )
    )
]

INVALID_STRINGS = [
    pytest.param(
        path,
        encoded([*payload[:index], value, *payload[index + 1 :]]),
        code,
        id=f"{path}-string-{index}-{value}",
    )
    for path, payload, code in ROUTES
    for index in range(len(payload))
    for value in (
        ("not-a-uuid", "")
        if index == 1
        else ("not-a-date", "2026-11-16T00:00:00", "0001-01-01T00:00:00+23:59")
    )
]


@pytest.mark.parametrize(
    "path,cursor,code",
    [
        *INVALID_MEMBERS,
        *INVALID_SHAPES,
        *INVALID_REPRESENTATIONS,
        *INVALID_STRINGS,
    ],
)
async def test_malformed_cursor_is_a_stable_422_before_database(
    client: httpx.AsyncClient,
    path: str,
    cursor: str,
    code: str,
) -> None:
    response = await client.get(path, params={"cursor": cursor})
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == code


@pytest.mark.parametrize(
    "timestamp",
    [
        TIMESTAMP,
        "2026-11-16T05:30:00+05:30",
        "2026-11-15T19:00:00-05:00",
        "2026-11-16T00:00:00Z",
    ],
)
def test_valid_cursor_roundtrip_retains_identity_and_normalizes_utc(timestamp: str) -> None:
    expected = datetime(2026, 11, 16, tzinfo=UTC)
    assert decode_cursor(encoded([timestamp, IDENTIFIER])) == (expected, UUID(IDENTIFIER))
    metrics_cursor = _decode_cursor(
        encoded(
            [
                timestamp,
                IDENTIFIER,
                "2026-11-15T00:00:00Z",
                "2026-11-17T00:00:00Z",
            ]
        )
    )
    assert metrics_cursor.started_at == expected
    assert metrics_cursor.identifier == UUID(IDENTIFIER)
    assert metrics_cursor.start == datetime(2026, 11, 15, tzinfo=UTC)
    assert metrics_cursor.end == datetime(2026, 11, 17, tzinfo=UTC)
