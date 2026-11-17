"""Read-only PostgreSQL aggregates over durable work and best-effort telemetry."""

import base64
import binascii
import json
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import (
    ColumnElement,
    Float,
    SQLColumnExpression,
    Text,
    case,
    cast,
    func,
    literal,
    select,
    text,
    true,
    tuple_,
)
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from app.db.models import Document, DocumentChunk, PerformanceSample, ProcessingAttempt

STATUSES = ("PENDING", "PROCESSING", "COMPLETED", "FAILED")
OUTCOMES = ("RUNNING", "SUCCEEDED", "FAILED", "RETRY", "INTERRUPTED")
TERMINAL_OUTCOMES = OUTCOMES[1:]


class MetricsQueryError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message


def utcnow() -> datetime:
    return datetime.now(UTC)


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise MetricsQueryError(
            "METRICS_WINDOW_INVALID", "Metrics times must include a UTC offset."
        )
    return value.astimezone(UTC)


@dataclass(frozen=True)
class Cursor:
    started_at: datetime
    identifier: UUID
    start: datetime
    end: datetime


def _decode_cursor(value: str) -> Cursor:
    try:
        if not isinstance(value, str) or len(value) > 1024:
            raise ValueError
        payload = json.loads(base64.b64decode(value, altchars=b"-_", validate=True).decode())
        if (
            not isinstance(payload, list)
            or len(payload) != 4
            or any(not isinstance(member, str) for member in payload)
        ):
            raise ValueError
        started, identifier, start, end = payload
        return Cursor(
            _utc(datetime.fromisoformat(started)),
            UUID(identifier),
            _utc(datetime.fromisoformat(start)),
            _utc(datetime.fromisoformat(end)),
        )
    except (ValueError, TypeError, UnicodeError, binascii.Error, OverflowError):
        raise MetricsQueryError("METRICS_CURSOR_INVALID", "Metrics cursor is invalid.") from None


def _encode_cursor(started: datetime, identifier: UUID, start: datetime, end: datetime) -> str:
    payload = [started.isoformat(), str(identifier), start.isoformat(), end.isoformat()]
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()


def _window(
    start: datetime | None,
    end: datetime | None,
    cursor: Cursor | None,
    measured: datetime,
) -> tuple[datetime, datetime]:
    if cursor is not None:
        if (start is not None and _utc(start) != cursor.start) or (
            end is not None and _utc(end) != cursor.end
        ):
            raise MetricsQueryError(
                "METRICS_CURSOR_INVALID", "Cursor belongs to a different metrics window."
            )
        start, end = cursor.start, cursor.end
    else:
        end = _utc(end) if end is not None else measured
        start = _utc(start) if start is not None else end - timedelta(hours=24)
    if start >= end or end - start > timedelta(days=7):
        raise MetricsQueryError(
            "METRICS_WINDOW_INVALID", "Use start < end and a window of at most seven days."
        )
    if cursor is not None and not start <= cursor.started_at < end:
        raise MetricsQueryError(
            "METRICS_CURSOR_INVALID", "Cursor position falls outside its metrics window."
        )
    return start, end


def _aggregates(
    value: SQLColumnExpression[Any], prefix: str = ""
) -> tuple[ColumnElement[Any], ...]:
    return (
        func.count(value).label(f"{prefix}sample_count"),
        func.percentile_cont(0.5).within_group(value).label(f"{prefix}p50_ms"),
        func.percentile_cont(0.95).within_group(value).label(f"{prefix}p95_ms"),
    )


def _duration(row: Mapping[Any, Any], prefix: str = "") -> dict[str, Any]:
    count = int(row[f"{prefix}sample_count"])
    return {
        "sample_count": count,
        "small_sample": count < 20,
        "p50_ms": float(row[f"{prefix}p50_ms"]) if row[f"{prefix}p50_ms"] is not None else None,
        "p95_ms": float(row[f"{prefix}p95_ms"]) if row[f"{prefix}p95_ms"] is not None else None,
    }


class AsyncMetricsService:
    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    @asynccontextmanager
    async def _snapshot(self) -> AsyncIterator[AsyncConnection]:
        # Counts, percentiles and pages within one response share an MVCC view.
        async with self.engine.connect() as connection:
            await connection.execution_options(isolation_level="REPEATABLE READ")
            async with connection.begin():
                await connection.execute(text("SET TRANSACTION READ ONLY"))
                yield connection

    async def document_statistics(self) -> dict[str, Any]:
        published = (
            select(func.count(DocumentChunk.id))
            .select_from(DocumentChunk)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(
                Document.status == "COMPLETED",
                Document.published_attempt_id == DocumentChunk.attempt_id,
                Document.profile_id == DocumentChunk.profile_id,
            )
            .correlate(None)
            .scalar_subquery()
        )
        statement = select(
            func.count(Document.id).label("document_count"),
            func.coalesce(func.sum(Document.byte_size), 0).label("original_bytes"),
            func.coalesce(func.sum(Document.page_count), 0).label("extracted_pages"),
            func.coalesce(func.sum(Document.block_count), 0).label("extracted_blocks"),
            published.label("published_chunks"),
            *(
                func.count(Document.id).filter(Document.status == status).label(status)
                for status in STATUSES
            ),
        )
        async with self._snapshot() as connection:
            measured = utcnow()
            row = (await connection.execute(statement)).mappings().one()
            return {
                "measured_at": measured,
                "scope": "current_installation",
                **{
                    key: int(row[key])
                    for key in (
                        "document_count",
                        "original_bytes",
                        "extracted_pages",
                        "extracted_blocks",
                        "published_chunks",
                    )
                },
                "status_counts": {status: int(row[status]) for status in STATUSES},
            }

    async def processing_statistics(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise MetricsQueryError(
                "METRICS_LIMIT_INVALID", "Metrics page size must be from 1 to 100."
            )
        position = _decode_cursor(cursor) if cursor is not None else None
        measured = utcnow()
        start, end = _window(start, end, position, measured)
        attempt_window = (ProcessingAttempt.started_at >= start, ProcessingAttempt.started_at < end)
        performance_window = (
            PerformanceSample.recorded_at >= start,
            PerformanceSample.recorded_at < end,
        )
        queue_ms = (
            func.extract("epoch", ProcessingAttempt.started_at - ProcessingAttempt.requested_at)
            * 1000
        )
        processing_ms = case(
            (
                ProcessingAttempt.outcome.in_(TERMINAL_OUTCOMES),
                func.extract("epoch", ProcessingAttempt.ended_at - ProcessingAttempt.started_at)
                * 1000,
            ),
            else_=None,
        )
        async with self._snapshot() as connection:
            outcome_rows = await connection.execute(
                select(ProcessingAttempt.outcome, func.count(ProcessingAttempt.id))
                .where(*attempt_window)
                .group_by(ProcessingAttempt.outcome)
            )
            outcomes = dict.fromkeys(OUTCOMES, 0)
            outcomes.update({name: int(count) for name, count in outcome_rows})
            terminal_count = sum(outcomes[name] for name in TERMINAL_OUTCOMES)
            current = await connection.scalar(
                select(func.count(ProcessingAttempt.id)).where(
                    ProcessingAttempt.outcome == "RUNNING"
                )
            )
            durations = (
                (
                    await connection.execute(
                        select(
                            *_aggregates(queue_ms, "queue_"),
                            *_aggregates(processing_ms, "processing_"),
                        ).where(*attempt_window)
                    )
                )
                .mappings()
                .one()
            )
            stages = (
                func.jsonb_each(ProcessingAttempt.stage_durations)
                .table_valued("key", "value")
                .lateral()
            )
            numeric_duration = case(
                (
                    func.jsonb_typeof(stages.c.value) == "number",
                    cast(cast(stages.c.value, Text), Float),
                ),
                else_=None,
            )
            stage_rows = (
                (
                    await connection.execute(
                        select(stages.c.key, *_aggregates(numeric_duration))
                        .select_from(ProcessingAttempt)
                        .join(stages, true())
                        .where(*attempt_window, numeric_duration.is_not(None))
                        .group_by(stages.c.key)
                    )
                )
                .mappings()
                .all()
            )
            document_outcomes = (
                (
                    await connection.execute(
                        select(
                            func.count(Document.id)
                            .filter(Document.status == "COMPLETED")
                            .label("completed"),
                            func.count(Document.id)
                            .filter(Document.status == "FAILED")
                            .label("failed"),
                            func.count(Document.id)
                            .filter(Document.status == "FAILED", Document.attempt_count == 0)
                            .label("failed_before_first_attempt"),
                        ).where(Document.terminal_at >= start, Document.terminal_at < end)
                    )
                )
                .mappings()
                .one()
            )
            pre_attempt_code = Document.error["code"].astext
            pre_attempt_rows = await connection.execute(
                select(pre_attempt_code, func.count(Document.id))
                .where(Document.attempt_count == 0, pre_attempt_code.is_not(None))
                .group_by(pre_attempt_code)
            )
            pre_attempt_errors = {code: int(count) for code, count in pre_attempt_rows}

            history = (
                select(
                    ProcessingAttempt.id,
                    ProcessingAttempt.document_id,
                    ProcessingAttempt.attempt_number,
                    ProcessingAttempt.requested_at,
                    ProcessingAttempt.started_at,
                    ProcessingAttempt.heartbeat_at,
                    ProcessingAttempt.ended_at,
                    ProcessingAttempt.outcome,
                    ProcessingAttempt.stage,
                    ProcessingAttempt.stage_durations,
                    ProcessingAttempt.error,
                    ProcessingAttempt.broker_task_id,
                )
                .where(*attempt_window)
                .order_by(ProcessingAttempt.started_at.desc(), ProcessingAttempt.id.desc())
                .limit(limit + 1)
            )
            if position is not None:
                history = history.where(
                    tuple_(ProcessingAttempt.started_at, ProcessingAttempt.id)
                    < tuple_(literal(position.started_at), literal(position.identifier))
                )
            history_rows = (await connection.execute(history)).mappings().all()
            page = [dict(row) for row in history_rows[:limit]]
            for item in page:
                item["queue_wait_ms"] = (
                    item["started_at"] - item["requested_at"]
                ).total_seconds() * 1000
                item["processing_ms"] = (
                    (item["ended_at"] - item["started_at"]).total_seconds() * 1000
                    if item["ended_at"] is not None and item["outcome"] in TERMINAL_OUTCOMES
                    else None
                )
            next_cursor = (
                _encode_cursor(page[-1]["started_at"], page[-1]["id"], start, end)
                if len(history_rows) > limit
                else None
            )

            operations: dict[str, dict[str, Any]] = {}
            operation_rows = (
                (
                    await connection.execute(
                        select(
                            PerformanceSample.operation,
                            *_aggregates(PerformanceSample.duration_ms),
                            func.sum(PerformanceSample.input_tokens).label("input_tokens"),
                            func.count(PerformanceSample.input_tokens).label("input_token_samples"),
                            func.sum(PerformanceSample.output_tokens).label("output_tokens"),
                            func.count(PerformanceSample.output_tokens).label(
                                "output_token_samples"
                            ),
                        )
                        .where(*performance_window)
                        .group_by(PerformanceSample.operation)
                    )
                )
                .mappings()
                .all()
            )
            for row in operation_rows:
                operations[row["operation"]] = {
                    "latency": _duration(row),
                    "outcomes": {},
                    "error_classes": {},
                    "usage": {
                        key: (int(row[key]) if row[key] is not None else None)
                        for key in (
                            "input_tokens",
                            "input_token_samples",
                            "output_tokens",
                            "output_token_samples",
                        )
                    },
                }
            performance_outcomes = await connection.execute(
                select(
                    PerformanceSample.operation,
                    PerformanceSample.outcome,
                    func.count(PerformanceSample.id),
                )
                .where(*performance_window)
                .group_by(PerformanceSample.operation, PerformanceSample.outcome)
            )
            for operation, outcome, count in performance_outcomes:
                operations[operation]["outcomes"][outcome] = int(count)
            performance_errors = await connection.execute(
                select(
                    PerformanceSample.operation,
                    PerformanceSample.error_class,
                    func.count(PerformanceSample.id),
                )
                .where(*performance_window, PerformanceSample.error_class.is_not(None))
                .group_by(PerformanceSample.operation, PerformanceSample.error_class)
            )
            for operation, error_class, count in performance_errors:
                operations[operation]["error_classes"][error_class] = int(count)
            oldest = await connection.scalar(select(func.min(PerformanceSample.recorded_at)))
            return {
                "measured_at": measured,
                "window": {"start": start, "end": end, "boundary": "[start,end)"},
                "attempts": {
                    "total": sum(outcomes.values()),
                    "outcomes": outcomes,
                    "terminal_count": terminal_count,
                    "success_rate": outcomes["SUCCEEDED"] / terminal_count
                    if terminal_count
                    else None,
                    "current_in_progress": int(current or 0),
                    "queue_wait": _duration(durations, "queue_"),
                    "processing_time": _duration(durations, "processing_"),
                    "stage_durations": {row["key"]: _duration(row) for row in stage_rows},
                    "history": {"items": page, "next_cursor": next_cursor},
                },
                "documents": {
                    **{key: int(value) for key, value in document_outcomes.items()},
                    "terminal_count": int(
                        document_outcomes["completed"] + document_outcomes["failed"]
                    ),
                },
                "pre_attempt_errors": {
                    "scope": "current_document_state",
                    "by_code": pre_attempt_errors,
                    "dispatch_error_count": sum(
                        count
                        for code, count in pre_attempt_errors.items()
                        if code.upper().startswith("DISPATCH")
                    ),
                },
                "operations": operations,
                "performance_retention": {
                    "retention_days": 7,
                    "cutoff": measured - timedelta(days=7),
                    "oldest_available": oldest,
                    "requested_outside_retention": start < measured - timedelta(days=7),
                    "telemetry_complete": False,
                    "collection": "best_effort",
                },
            }
