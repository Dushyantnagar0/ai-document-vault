"""Metrics routes/validation need no provider key, database or application lifespan."""

from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import metrics
from app.api.documents import APIError


class NoDatabase:
    def connect(self) -> None:
        pytest.fail("Invalid metrics requests must fail before database access")


@pytest.fixture
def application() -> FastAPI:
    app = FastAPI()
    app.include_router(metrics.router)
    app.state.engine = NoDatabase()

    @app.exception_handler(APIError)
    async def error_handler(request: Request, error: APIError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status,
            content={"error": {"code": error.code, "message": error.message}},
        )

    return app


async def test_document_metrics_response_and_swagger_schema(
    application: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Stub:
        def __init__(self, engine: object) -> None:
            pass

        async def document_statistics(self) -> dict[str, Any]:
            return {
                "measured_at": datetime(2026, 11, 15, tzinfo=UTC),
                "scope": "current_installation",
                "document_count": 1,
                "status_counts": {"COMPLETED": 1},
                "original_bytes": 123,
                "extracted_pages": 2,
                "extracted_blocks": 3,
                "published_chunks": 4,
            }

    monkeypatch.setattr(metrics, "AsyncMetricsService", Stub)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/metrics/documents")
        schema = (await client.get("/openapi.json")).json()
    assert response.status_code == 200
    assert response.json()["original_bytes"] == 123
    assert response.json()["measured_at"].endswith("Z")
    for path in ("/api/v1/metrics/documents", "/api/v1/metrics/processing"):
        assert schema["paths"][path]["get"]["responses"]["200"]["content"]["application/json"][
            "schema"
        ]["$ref"]
    assert "DurationSummary" in schema["components"]["schemas"]


@pytest.mark.parametrize(
    "query",
    [
        {"limit": 0},
        {"limit": 101},
        {"start": "not-a-date"},
        {"start": "2026-11-01T00:00:00", "end": "2026-11-02T00:00:00Z"},
        {"start": "2026-11-02T00:00:00Z", "end": "2026-11-01T00:00:00Z"},
        {"start": "2026-11-01T00:00:00Z", "end": "2026-11-09T00:00:00Z"},
        {"cursor": "%%%"},
        {"cursor": "x" * 1025},
    ],
)
async def test_invalid_queries_are_422_without_database_access(
    application: FastAPI,
    query: dict[str, Any],
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/metrics/processing", params=query)
    assert response.status_code == 422
