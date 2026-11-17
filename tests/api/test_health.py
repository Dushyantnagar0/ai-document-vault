"""HTTP contracts use a substitute; real-service health is integration evidence."""

from typing import Any
from uuid import UUID

import httpx
import pytest

from app.main import create_app


class ReadinessStub:
    dependencies = {"broker": "unavailable", "worker": "unavailable"}

    def __init__(self, database: str = "ready") -> None:
        self.database = database

    async def core(self) -> dict[str, str]:
        return {"database": self.database, "schema": "ready", "storage": "ready"}


@pytest.mark.parametrize(
    "database,code,status", [("ready", 200, "degraded"), ("unavailable", 503, "unavailable")]
)
async def test_core_health_and_dependency_degradation(
    database: str, code: int, status: str
) -> None:
    app = create_app()
    app.state.readiness = ReadinessStub(database)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health/ready")
    assert response.status_code == code
    assert response.json()["status"] == status
    assert response.json()["components"]["worker"] == "unavailable"


async def test_liveness_does_not_require_configuration_or_dependencies() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app()), base_url="http://test"
    ) as client:
        response = await client.get("/health/live")
        missing = await client.get("/api/v1/not-a-real-route")
    assert response.status_code == 200
    assert missing.status_code == 404
    payload: dict[str, Any] = missing.json()["error"]
    assert payload["code"] == "NOT_FOUND"
    assert str(UUID(payload["request_id"])) == missing.headers["X-Request-ID"]
