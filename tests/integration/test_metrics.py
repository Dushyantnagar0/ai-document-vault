"""Exact aggregate/window contracts against private-schema PostgreSQL fixtures."""

import os
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.orm import Session
from test_processing_claims import ClaimDatabase
from test_processing_claims import claim_database as claim_database
from test_processing_claims import new_document as new_document
from test_retrieval import Published
from test_retrieval import publish as publish
from test_retrieval import retrieval_engine as retrieval_engine

from app.api.metrics import router
from app.db.models import Document, PerformanceSample, ProcessingAttempt
from app.services import metrics
from app.services.metrics import AsyncMetricsService, MetricsQueryError

pytestmark = [
    pytest.mark.integration,
    pytest.mark.enable_socket,
    pytest.mark.skipif(
        os.environ.get("VAULT_INTEGRATION_TESTS") != "1", reason="requires isolated Compose stack"
    ),
]


@pytest.fixture(autouse=True)
def empty_metrics_tables(claim_database: ClaimDatabase) -> None:
    assert claim_database.settings.postgres_db == "vault_test"
    with Session(claim_database.engine) as database, database.begin():
        schema = database.scalar(text("SELECT current_schema()"))
        assert isinstance(schema, str) and schema.startswith("claims_test_")
        database.execute(text("TRUNCATE TABLE documents CASCADE"))


@pytest.fixture
def measured(monkeypatch: pytest.MonkeyPatch) -> datetime:
    now = datetime(2026, 11, 15, 12, tzinfo=UTC)
    monkeypatch.setattr(metrics, "utcnow", lambda: now)
    return now


@pytest.fixture
async def service(retrieval_engine: AsyncEngine, measured: datetime) -> AsyncMetricsService:
    return AsyncMetricsService(retrieval_engine)


def add_document(
    database: ClaimDatabase,
    *,
    now: datetime,
    status: str = "PENDING",
    byte_size: int = 10,
    terminal_at: datetime | None = None,
    error: dict[str, Any] | None = None,
) -> UUID:
    identifier = uuid4()
    with Session(database.engine) as session, session.begin():
        session.add(
            Document(
                id=identifier,
                filename="metrics-fixture.pdf",
                media_type="application/pdf",
                byte_size=byte_size,
                sha256="a" * 64,
                storage_key=str(uuid4()),
                profile_id=database.profile_id,
                request_fingerprint="b" * 64,
                status=status,
                stage=status,
                created_at=now,
                updated_at=now,
                terminal_at=terminal_at,
                error=error,
                next_dispatch_at=now,
                deadline_at=now + timedelta(seconds=1800),
                warnings=[],
                attempt_count=0,
                ai_calls_reserved=0,
            )
        )
    return identifier


def add_attempt(
    database: ClaimDatabase,
    *,
    started: datetime,
    outcome: str,
    queue_ms: float = 100,
    processing_ms: float = 1000,
    byte_size: int = 10,
    durations: dict[str, Any] | None = None,
) -> tuple[UUID, UUID]:
    status = {"FAILED": "FAILED", "RETRY": "PENDING", "RUNNING": "PROCESSING"}[outcome]
    ended = None if outcome == "RUNNING" else started + timedelta(milliseconds=processing_ms)
    identifier = add_document(
        database,
        now=started - timedelta(milliseconds=queue_ms),
        status=status,
        byte_size=byte_size,
        terminal_at=ended if status == "FAILED" else None,
    )
    attempt_id, token = uuid4(), uuid4()
    with Session(database.engine) as session, session.begin():
        session.add(
            ProcessingAttempt(
                id=attempt_id,
                document_id=identifier,
                attempt_number=1,
                claim_token=token,
                requested_at=started - timedelta(milliseconds=queue_ms),
                started_at=started,
                heartbeat_at=started,
                ended_at=ended,
                outcome=outcome,
                stage="EXTRACTING",
                stage_started_at=started,
                stage_durations=durations or {},
                error={"code": "FIXTURE_FAILURE", "message": "Known failure"}
                if outcome == "FAILED"
                else None,
            )
        )
        session.flush()
        document = session.get(Document, identifier)
        assert document is not None
        document.current_attempt_id, document.attempt_count = attempt_id, 1
        if outcome == "RUNNING":
            document.claim_token, document.lease_until = token, started + timedelta(seconds=180)
    return identifier, attempt_id


@pytest.fixture
def known_metrics(
    claim_database: ClaimDatabase,
    measured: datetime,
    publish: Callable[[Sequence[Sequence[float]]], Published],
) -> dict[str, Any]:
    start = measured - timedelta(days=1)
    published = publish([[1.0] + [0.0] * 1535])
    with Session(claim_database.engine) as database, database.begin():
        document = database.get(Document, published.document_id)
        assert document is not None
        attempt = database.get(ProcessingAttempt, document.published_attempt_id)
        assert attempt is not None
        document.byte_size, document.page_count, document.block_count = 100, 2, 3
        document.chunk_count = 999  # The API must count actual published rows.
        document.terminal_at = start + timedelta(seconds=1)
        attempt.requested_at = start - timedelta(milliseconds=100)
        attempt.started_at = attempt.heartbeat_at = attempt.stage_started_at = start
        attempt.ended_at = start + timedelta(seconds=1)
        attempt.stage_durations = {"extracting_ms": 100, "embedding_ms": 200, "total_ms": 1000}
    add_attempt(
        claim_database,
        started=start + timedelta(hours=1),
        outcome="FAILED",
        queue_ms=300,
        processing_ms=3000,
        byte_size=200,
        durations={"extracting_ms": 300},
    )
    add_attempt(
        claim_database,
        started=start + timedelta(hours=2),
        outcome="RETRY",
        queue_ms=500,
        processing_ms=5000,
        byte_size=300,
        durations={"chunking_ms": 500},
    )
    add_attempt(
        claim_database,
        started=start + timedelta(hours=3),
        outcome="RUNNING",
        queue_ms=700,
        byte_size=400,
    )
    add_attempt(
        claim_database, started=start - timedelta(hours=1), outcome="RUNNING", byte_size=500
    )
    add_document(
        claim_database,
        now=start + timedelta(hours=4),
        byte_size=7,
        error={"code": "DISPATCH_UNAVAILABLE", "message": "Awaiting transport recovery"},
    )
    add_document(
        claim_database,
        now=start + timedelta(hours=5),
        status="FAILED",
        byte_size=11,
        terminal_at=start + timedelta(hours=5, minutes=15),
        error={"code": "PROCESSING_BUDGET_EXHAUSTED", "message": "No first claim"},
    )
    with Session(claim_database.engine) as database, database.begin():
        database.add_all(
            [
                PerformanceSample(
                    recorded_at=start,
                    operation="api.documents.list",
                    duration_ms=100,
                    outcome="succeeded",
                ),
                PerformanceSample(
                    recorded_at=start + timedelta(seconds=1),
                    operation="api.documents.list",
                    duration_ms=300,
                    outcome="failed",
                    error_class="DatabaseUnavailable",
                ),
                PerformanceSample(
                    recorded_at=start + timedelta(hours=1),
                    operation="chat.answer",
                    duration_ms=200,
                    outcome="succeeded",
                    input_tokens=10,
                    output_tokens=0,
                ),
                PerformanceSample(
                    recorded_at=start + timedelta(hours=2),
                    operation="chat.answer",
                    duration_ms=600,
                    outcome="failed",
                    error_class="ProviderTimeout",
                    output_tokens=5,
                ),
                PerformanceSample(
                    recorded_at=measured,
                    operation="excluded.end",
                    duration_ms=1,
                    outcome="succeeded",
                ),
            ]
        )
    return {"start": start, "end": measured}


async def test_document_snapshot_counts_referenced_bytes_and_published_rows(
    service: AsyncMetricsService,
    known_metrics: dict[str, Any],
    measured: datetime,
) -> None:
    result = await service.document_statistics()
    assert result == {
        "measured_at": measured,
        "scope": "current_installation",
        "document_count": 7,
        "status_counts": {"PENDING": 2, "PROCESSING": 2, "COMPLETED": 1, "FAILED": 2},
        "original_bytes": 1518,
        "extracted_pages": 2,
        "extracted_blocks": 3,
        "published_chunks": 1,
    }


async def test_attempt_document_denominators_durations_and_known_usage(
    service: AsyncMetricsService,
    known_metrics: dict[str, Any],
) -> None:
    result = await service.processing_statistics()
    attempts = result["attempts"]
    assert attempts["total"] == 4
    assert attempts["outcomes"] == {
        "RUNNING": 1,
        "SUCCEEDED": 1,
        "FAILED": 1,
        "RETRY": 1,
        "INTERRUPTED": 0,
    }
    assert attempts["terminal_count"] == 3
    assert attempts["success_rate"] == pytest.approx(1 / 3)
    assert attempts["current_in_progress"] == 2
    assert attempts["queue_wait"] == {
        "sample_count": 4,
        "small_sample": True,
        "p50_ms": 400,
        "p95_ms": 670,
    }
    assert attempts["processing_time"] == {
        "sample_count": 3,
        "small_sample": True,
        "p50_ms": 3000,
        "p95_ms": 4800,
    }
    extracting = attempts["stage_durations"]["extracting_ms"]
    assert extracting == {"sample_count": 2, "small_sample": True, "p50_ms": 200, "p95_ms": 290}
    assert result["documents"] == {
        "completed": 1,
        "failed": 2,
        "failed_before_first_attempt": 1,
        "terminal_count": 3,
    }
    assert result["pre_attempt_errors"] == {
        "scope": "current_document_state",
        "by_code": {"DISPATCH_UNAVAILABLE": 1, "PROCESSING_BUDGET_EXHAUSTED": 1},
        "dispatch_error_count": 1,
    }
    operation = result["operations"]["api.documents.list"]
    assert operation["latency"] == extracting
    assert operation["outcomes"] == {"succeeded": 1, "failed": 1}
    assert operation["error_classes"] == {"DatabaseUnavailable": 1}
    assert operation["usage"] == {
        "input_tokens": None,
        "input_token_samples": 0,
        "output_tokens": None,
        "output_token_samples": 0,
    }
    assert result["operations"]["chat.answer"]["usage"] == {
        "input_tokens": 10,
        "input_token_samples": 1,
        "output_tokens": 5,
        "output_token_samples": 2,
    }
    assert "excluded.end" not in result["operations"]
    assert result["performance_retention"]["telemetry_complete"] is False
    assert result["performance_retention"]["oldest_available"] == known_metrics["start"]
    assert all(
        "claim_token" not in item and "text" not in item for item in attempts["history"]["items"]
    )


async def test_empty_metrics_use_null_percentiles_and_no_invented_success_rate(
    service: AsyncMetricsService,
) -> None:
    documents = await service.document_statistics()
    assert (
        documents["document_count"]
        == documents["original_bytes"]
        == documents["published_chunks"]
        == 0
    )
    result = await service.processing_statistics()
    assert result["attempts"]["success_rate"] is None
    assert result["attempts"]["queue_wait"] == {
        "sample_count": 0,
        "small_sample": True,
        "p50_ms": None,
        "p95_ms": None,
    }
    assert result["attempts"]["history"] == {"items": [], "next_cursor": None}
    assert result["operations"] == {}
    assert result["performance_retention"]["oldest_available"] is None


async def test_processing_http_response_validates_real_aggregate_payload(
    retrieval_engine: AsyncEngine,
    known_metrics: dict[str, Any],
) -> None:
    application = FastAPI()
    application.include_router(router)
    application.state.engine = retrieval_engine
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/metrics/processing", params={"limit": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["attempts"]["total"] == 4
    assert len(body["attempts"]["history"]["items"]) == 2
    assert body["attempts"]["history"]["next_cursor"] is not None
    assert body["window"]["end"].endswith("Z")
    assert body["performance_retention"]["telemetry_complete"] is False


async def test_twenty_samples_clear_small_sample_flag(
    service: AsyncMetricsService,
    claim_database: ClaimDatabase,
    measured: datetime,
) -> None:
    with Session(claim_database.engine) as database, database.begin():
        database.add_all(
            [
                PerformanceSample(
                    recorded_at=measured - timedelta(seconds=1),
                    operation="api.sampled",
                    duration_ms=index,
                    outcome="succeeded",
                )
                for index in range(20)
            ]
        )
    result = await service.processing_statistics()
    latency = result["operations"]["api.sampled"]["latency"]
    assert latency["sample_count"] == 20
    assert latency["small_sample"] is False
    assert latency["p50_ms"] == pytest.approx(9.5)
    assert latency["p95_ms"] == pytest.approx(18.05)


async def test_utc_boundaries_and_timezone_conversion_are_half_open(
    service: AsyncMetricsService,
    claim_database: ClaimDatabase,
    measured: datetime,
) -> None:
    start = measured - timedelta(hours=1)
    _, included = add_attempt(claim_database, started=start, outcome="FAILED")
    add_attempt(claim_database, started=measured, outcome="FAILED")
    add_attempt(claim_database, started=start - timedelta(microseconds=1), outcome="FAILED")
    offset = timezone(timedelta(hours=5, minutes=30))
    result = await service.processing_statistics(
        start=start.astimezone(offset), end=measured.astimezone(offset)
    )
    assert result["window"] == {"start": start, "end": measured, "boundary": "[start,end)"}
    assert [row["id"] for row in result["attempts"]["history"]["items"]] == [included]


async def test_attempt_cursor_orders_ties_and_retains_original_window(
    service: AsyncMetricsService,
    claim_database: ClaimDatabase,
    measured: datetime,
) -> None:
    started = measured - timedelta(hours=1)
    identifiers = [
        add_attempt(claim_database, started=started, outcome="FAILED")[1] for _ in range(5)
    ]
    first = await service.processing_statistics(limit=2)
    second = await service.processing_statistics(
        limit=2, cursor=first["attempts"]["history"]["next_cursor"]
    )
    third = await service.processing_statistics(
        limit=2, cursor=second["attempts"]["history"]["next_cursor"]
    )
    assert first["window"] == second["window"] == third["window"]
    assert [
        row["id"] for page in (first, second, third) for row in page["attempts"]["history"]["items"]
    ] == sorted(identifiers, reverse=True)
    assert third["attempts"]["history"]["next_cursor"] is None
    with pytest.raises(MetricsQueryError, match="different metrics window"):
        await service.processing_statistics(
            start=started, end=measured, cursor=first["attempts"]["history"]["next_cursor"]
        )


async def test_old_attempt_history_survives_unavailable_performance_window(
    service: AsyncMetricsService,
    claim_database: ClaimDatabase,
    measured: datetime,
) -> None:
    start, end = measured - timedelta(days=30), measured - timedelta(days=29)
    add_attempt(claim_database, started=start + timedelta(hours=1), outcome="FAILED")
    recent = measured - timedelta(hours=1)
    with Session(claim_database.engine) as database, database.begin():
        database.add(
            PerformanceSample(
                recorded_at=recent, operation="recent.only", duration_ms=1, outcome="succeeded"
            )
        )
    result = await service.processing_statistics(start=start, end=end)
    assert result["attempts"]["total"] == 1
    assert result["operations"] == {}
    assert result["performance_retention"] == {
        "retention_days": 7,
        "cutoff": measured - timedelta(days=7),
        "oldest_available": recent,
        "requested_outside_retention": True,
        "telemetry_complete": False,
        "collection": "best_effort",
    }


@pytest.mark.parametrize("kind", ["naive", "equal", "reverse", "wide"])
async def test_invalid_time_windows_are_rejected(
    service: AsyncMetricsService,
    measured: datetime,
    kind: str,
) -> None:
    start, end = measured - timedelta(days=1), measured
    if kind == "naive":
        start = start.replace(tzinfo=None)
    elif kind == "equal":
        start = end
    elif kind == "reverse":
        start = end + timedelta(seconds=1)
    else:
        start = end - timedelta(days=7, microseconds=1)
    with pytest.raises(MetricsQueryError) as error:
        await service.processing_statistics(start=start, end=end)
    assert error.value.code == "METRICS_WINDOW_INVALID"
