"""Liveness is independent; readiness preserves saved reads during broker outages."""

from typing import Literal, cast

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.core.readiness import Readiness

router = APIRouter(prefix="/health", tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["live", "ready", "degraded", "unavailable"]
    components: dict[str, str]


@router.get("/live", response_model=HealthResponse)
async def live() -> HealthResponse:
    return HealthResponse(status="live", components={"process": "ready"})


@router.get("/ready", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
async def ready(request: Request) -> JSONResponse:
    readiness = cast(Readiness, request.app.state.readiness)
    core = await readiness.core()
    core_ok = all(value == "ready" for value in core.values())
    dependencies = readiness.dependencies.copy()
    status: Literal["ready", "degraded", "unavailable"] = "unavailable"
    if core_ok:
        status = "ready" if all(v == "ready" for v in dependencies.values()) else "degraded"
    payload = HealthResponse(status=status, components={**core, **dependencies})
    return JSONResponse(status_code=200 if core_ok else 503, content=payload.model_dump())
