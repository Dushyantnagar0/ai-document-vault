"""Required document statistics and processing-monitoring HTTP endpoints."""

from datetime import datetime
from typing import Annotated, Any, Literal, cast
from uuid import UUID

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncEngine

from app.api.documents import APIError
from app.services.metrics import AsyncMetricsService, MetricsQueryError

router = APIRouter(prefix="/api/v1/metrics", tags=["metrics"])


class DurationSummary(BaseModel):
    sample_count: int
    small_sample: bool = Field(description="Fewer than 20 samples; no performance guarantee.")
    p50_ms: float | None
    p95_ms: float | None


class DocumentStatistics(BaseModel):
    measured_at: datetime
    scope: Literal["current_installation"]
    document_count: int
    status_counts: dict[str, int]
    original_bytes: int = Field(
        description=(
            "Original-file bytes referenced by metadata; "
            "excludes orphan files and database/index overhead."
        )
    )
    extracted_pages: int
    extracted_blocks: int
    published_chunks: int


class MetricsWindow(BaseModel):
    start: datetime
    end: datetime
    boundary: Literal["[start,end)"]


class AttemptRecord(BaseModel):
    id: UUID
    document_id: UUID
    attempt_number: int
    requested_at: datetime
    started_at: datetime
    heartbeat_at: datetime
    ended_at: datetime | None
    outcome: str
    stage: str
    stage_durations: dict[str, Any]
    error: dict[str, Any] | None
    broker_task_id: str | None
    queue_wait_ms: float
    processing_ms: float | None


class AttemptPage(BaseModel):
    items: list[AttemptRecord]
    next_cursor: str | None


class AttemptStatistics(BaseModel):
    total: int = Field(description="Attempts whose started_at falls inside the window.")
    outcomes: dict[str, int]
    terminal_count: int
    success_rate: float | None = Field(
        description="SUCCEEDED / all terminal attempts started in the window; excludes RUNNING."
    )
    current_in_progress: int = Field(
        description="All persisted RUNNING attempts, independent of the requested time window."
    )
    queue_wait: DurationSummary
    processing_time: DurationSummary
    stage_durations: dict[str, DurationSummary]
    history: AttemptPage


class DocumentOutcomes(BaseModel):
    completed: int
    failed: int
    failed_before_first_attempt: int
    terminal_count: int = Field(
        description=(
            "Documents completed/failed by terminal_at inside the window; separate from attempts."
        )
    )


class PreAttemptErrors(BaseModel):
    scope: Literal["current_document_state"]
    by_code: dict[str, int]
    dispatch_error_count: int


class KnownUsage(BaseModel):
    input_tokens: int | None
    input_token_samples: int
    output_tokens: int | None
    output_token_samples: int


class OperationStatistics(BaseModel):
    latency: DurationSummary
    outcomes: dict[str, int]
    error_classes: dict[str, int]
    usage: KnownUsage


class PerformanceRetention(BaseModel):
    retention_days: Literal[7]
    cutoff: datetime
    oldest_available: datetime | None
    requested_outside_retention: bool
    telemetry_complete: Literal[False]
    collection: Literal["best_effort"]


class ProcessingStatistics(BaseModel):
    measured_at: datetime
    window: MetricsWindow
    attempts: AttemptStatistics
    documents: DocumentOutcomes
    pre_attempt_errors: PreAttemptErrors
    operations: dict[str, OperationStatistics]
    performance_retention: PerformanceRetention


@router.get("/documents", response_model=DocumentStatistics)
async def document_statistics(request: Request) -> DocumentStatistics:
    service = AsyncMetricsService(cast(AsyncEngine, request.app.state.engine))
    return DocumentStatistics.model_validate(await service.document_statistics())


@router.get("/processing", response_model=ProcessingStatistics)
async def processing_statistics(
    request: Request,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: Annotated[str | None, Query(max_length=1024)] = None,
) -> ProcessingStatistics:
    """UTC-aware [start,end), default last 24h; maximum span seven days.

    Cursor continuation retains its original window if bounds are omitted.
    Raw telemetry is best effort; retained attempts outlive its seven-day policy.
    """
    service = AsyncMetricsService(cast(AsyncEngine, request.app.state.engine))
    try:
        result = await service.processing_statistics(
            start=start, end=end, limit=limit, cursor=cursor
        )
    except MetricsQueryError as error:
        raise APIError(422, error.code, error.message) from None
    return ProcessingStatistics.model_validate(result)
