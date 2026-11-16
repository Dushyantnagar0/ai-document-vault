"""Read-only, content-free document/attempt diagnostics; never starts AI work."""

import argparse
import json
import math
import re
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any, Never
from uuid import UUID

from sqlalchemy import Connection, Engine, create_engine, func, select
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import ConfigurationError, get_settings
from app.db.models import Document, PerformanceSample, ProcessingAttempt
from app.prompts.analysis import REPAIR_REASONS

STAGES = {
    "QUEUED",
    "DISPATCH_PENDING",
    "RETRY_WAIT",
    "EXTRACTING",
    "CHUNKING",
    "EMBEDDING",
    "ANALYSING",
    "COMPLETED",
    "FAILED",
}
STATUSES = {"PENDING", "PROCESSING", "COMPLETED", "FAILED"}
OUTCOMES = {
    "RUNNING",
    "SUCCEEDED",
    "FAILED",
    "RETRY",
    "INTERRUPTED",
    "succeeded",
    "failed",
    "interrupted",
}
OPERATIONS = {
    "embedding.document",
    "generation.document",
    "processing.total",
    "embedding.query",
    "chat.planner",
    "chat.answer",
    "chat.evidence_review",
}
ERROR_CODES = {
    "ai_invalid_output",
    "ai_output_truncated",
    "ai_invalid_json",
    "ai_schema_invalid",
    "ai_refused",
    "ai_timeout",
    "ai_connection",
    "ai_quota",
    "ai_rate_limited",
    "ai_authentication",
    "ai_configuration",
    "ai_unavailable",
    "ai_provider_error",
    "ai_model_mismatch",
    "processing_deadline",
    "embedding_input_limit",
    "embedding_invalid_output",
    "generation_input_limit",
    "analysis_citation_invalid",
    "analysis_insight_review_invalid",
    "analysis_output_budget",
    "analysis_input_limit",
    "analysis_reduction_limit",
    "analysis_input_invalid",
    "analysis_prompt_limit",
    "ai_call_budget",
    "analysis_reduction_stalled",
    "analysis_coverage_invalid",
    "INDEX_PROFILE_MISMATCH",
    "DISPATCH_UNAVAILABLE",
    "PROCESSING_DEADLINE",
    "WORKER_INTERRUPTED",
    "PROCESSING_BUDGET_EXHAUSTED",
    "PROCESSING_FAILED",
    "PROCESSING_INTERRUPTED",
    "DATABASE_UNAVAILABLE",
    "NO_EXTRACTABLE_TEXT",
    "EXTRACTED_TEXT_LIMIT",
    "EXTRACTED_TOKEN_LIMIT",
    "CHUNK_MAPPING_FAILED",
    "CHUNK_LIMIT",
    "CHUNK_TOKEN_LIMIT",
    "UPLOAD_TOO_LARGE",
    "INVALID_DOCUMENT",
    "EXTRACTION_FAILED",
    "INVALID_PDF",
    "ENCRYPTED_PDF",
    "PDF_PAGE_LIMIT",
    "DOCX_ENTRY_LIMIT",
    "INVALID_DOCX",
    "DOCX_EXPANSION_LIMIT",
    "ENCRYPTED_DOCX",
    "UNSUPPORTED_FORMAT",
    "STORAGE_UNAVAILABLE",
    "CHAT_DEADLINE",
    "CHAT_TIMEOUT",
    "CHAT_FAILED",
    "CHAT_INPUT_LIMIT",
    "AI_CALL_BUDGET_EXHAUSTED",
    "CHAT_INVALID_OUTPUT",
    "EVIDENCE_REVIEW_INPUT_LIMIT",
    "EVIDENCE_REVIEW_INVALID",
    "CHUNKING_FAILED",
    "CHUNK_COUNT_LIMIT",
    "INCOMPLETE_ANALYSIS",
    "INVALID_STORAGE_FILE",
    "INVALID_STORAGE_KEY",
    "INVALID_VECTORS",
    "RETRIEVAL_UNAVAILABLE",
    "STORAGE_CONFLICT",
    "STORAGE_ERROR",
    "STORAGE_NOT_FOUND",
}
DETAIL_LABELS = set(REPAIR_REASONS) | {
    "length",
    "stop",
    "content_filter",
    "refusal",
    "schema_validation",
    "invalid_json",
    "missing_structured_output",
    "parsing_failed",
    "provider_refusal",
    "output_token_limit",
    "input_token_limit",
    "claim_identity_mismatch",
    "invalid_support_selection",
    "invalid_overlap_selection",
    "invalid_qualification_selection",
    "unknown_quote_id",
    "duplicate_quote_id",
    "unknown_insight_id",
    "duplicate_insight_id",
    "support_window_exceeded",
    "inconsistent_evidence_status",
    "structured_parse_failure",
    "missing_raw_response",
    "missing_parsed_response",
    "tool_calls",
    "function_call",
    "unknown",
}
DETAIL_COUNTS = {
    "input_tokens",
    "output_tokens",
    "max_output_tokens",
    "validation_error_count",
    "repair_attempt",
    "status_code",
}
ATTEMPT_LIMIT = 100


class DiagnosticError(Exception):
    """Only a static code may cross the CLI failure boundary."""


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        raise DiagnosticError("INVALID_ARGUMENTS")


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if 0 <= value <= 1_000_000_000_000_000 and math.isfinite(value) else None


def _label(value: Any, allowed: set[str]) -> str:
    return value if isinstance(value, str) and value in allowed else "UNKNOWN"


def _error(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, Any] = {"code": _label(value.get("code"), ERROR_CODES)}
    details = value.get("details")
    if isinstance(details, dict):
        safe = {}
        for key in ("validation_reason", "reason", "failure_kind", "finish_reason"):
            if isinstance(details.get(key), str) and details[key] in DETAIL_LABELS:
                safe[key] = details[key]
        for key in DETAIL_COUNTS:
            if (number := _number(details.get(key))) is not None:
                safe[key] = number
        request_id = details.get("provider_request_id")
        if isinstance(request_id, str) and re.fullmatch(r"req_[A-Za-z0-9_-]{1,120}", request_id):
            safe["provider_request_id"] = request_id
        if safe:
            result["details"] = safe
    # Never echo arbitrary message/details, exception strings or provider bodies.
    return result


def _id(value: Any) -> str | None:
    try:
        return str(UUID(str(value))) if value is not None else None
    except (ValueError, TypeError, AttributeError):
        return None


def _time(value: Any) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else None


def _durations(value: Any) -> dict[str, int | float]:
    if not isinstance(value, dict):
        return {}
    allowed = {f"{stage.lower()}_ms" for stage in STAGES} | {"total_ms"}
    return {key: number for key in allowed if (number := _number(value.get(key))) is not None}


def _attempt(row: Mapping[Any, Any]) -> dict[str, Any]:
    return {
        "attempt_id": _id(row["id"]),
        "attempt_number": row["attempt_number"],
        "broker_task_id": _id(row["broker_task_id"]),
        "broker_task_id_present": row["broker_task_id"] is not None,
        "outcome": _label(row["outcome"], OUTCOMES),
        "stage": _label(row["stage"], STAGES),
        "times": {
            name: _time(row[name])
            for name in (
                "requested_at",
                "started_at",
                "heartbeat_at",
                "stage_started_at",
                "ended_at",
            )
        },
        "stage_durations_ms": _durations(row["stage_durations"]),
        "error": _error(row["error"]),
    }


def _performance(connection: Connection, identifier: UUID) -> list[dict[str, Any]]:
    sample = PerformanceSample
    # Aggregate in PostgreSQL; never load prompts, messages, vectors or raw samples.
    rows = (
        connection.execute(
            select(
                sample.operation,
                sample.outcome,
                sample.error_class,
                func.count().label("sample_count"),
                func.sum(sample.duration_ms).label("total_duration_ms"),
                func.sum(sample.input_tokens).label("input_tokens"),
                func.count(sample.input_tokens).label("input_token_samples"),
                func.sum(sample.output_tokens).label("output_tokens"),
                func.count(sample.output_tokens).label("output_token_samples"),
            )
            .where(sample.document_id == identifier, sample.operation.in_(sorted(OPERATIONS)))
            .group_by(sample.operation, sample.outcome, sample.error_class)
            .order_by(sample.operation, sample.outcome, sample.error_class)
        )
        .mappings()
        .all()
    )
    return [
        {
            "operation": _label(row["operation"], OPERATIONS),
            "outcome": _label(row["outcome"], OUTCOMES),
            "error_code": _label(row["error_class"], ERROR_CODES) if row["error_class"] else None,
            "sample_count": row["sample_count"],
            "total_duration_ms": _number(row["total_duration_ms"]),
            "known_usage": {
                key: _number(row[key])
                for key in (
                    "input_tokens",
                    "input_token_samples",
                    "output_tokens",
                    "output_token_samples",
                )
            },
        }
        for row in rows
    ]


def diagnose(
    connection: Connection,
    *,
    document_id: UUID | None,
    filename: str | None,
    limit: int,
    max_ai_calls: int,
) -> dict[str, Any]:
    """Read one snapshot. Explicit column selection excludes large/private content."""
    condition = (
        Document.id == document_id if document_id is not None else Document.filename == filename
    )
    total = connection.scalar(select(func.count()).select_from(Document).where(condition)) or 0
    if not total:
        raise DiagnosticError("DOCUMENT_NOT_FOUND")
    columns = [
        Document.id,
        Document.status,
        Document.stage,
        Document.error,
        Document.byte_size,
        Document.extracted_tokens,
        Document.page_count,
        Document.block_count,
        Document.chunk_count,
        Document.attempt_count,
        Document.ai_calls_reserved,
        Document.current_attempt_id,
        Document.published_attempt_id,
        Document.created_at,
        Document.updated_at,
        Document.terminal_at,
        Document.deadline_at,
        Document.lease_until,
        Document.heartbeat_at,
        Document.next_dispatch_at,
        Document.last_dispatched_at,
    ]
    rows = (
        connection.execute(
            select(*columns)
            .where(condition)
            .order_by(Document.created_at.desc(), Document.id.desc())
            .limit(limit)
        )
        .mappings()
        .all()
    )
    documents = []
    for row in rows:
        identifier = row["id"]
        attempt = ProcessingAttempt
        attempts = (
            connection.execute(
                select(
                    attempt.id,
                    attempt.attempt_number,
                    attempt.broker_task_id,
                    attempt.outcome,
                    attempt.stage,
                    attempt.requested_at,
                    attempt.started_at,
                    attempt.heartbeat_at,
                    attempt.stage_started_at,
                    attempt.ended_at,
                    attempt.stage_durations,
                    attempt.error,
                )
                .where(attempt.document_id == identifier)
                .order_by(attempt.attempt_number.desc())
                .limit(ATTEMPT_LIMIT)
            )
            .mappings()
            .all()
        )
        documents.append(
            {
                "document_id": _id(identifier),
                "status": _label(row["status"], STATUSES),
                "stage": _label(row["stage"], STAGES),
                "error": _error(row["error"]),
                "counts": {
                    key: row[key]
                    for key in (
                        "byte_size",
                        "extracted_tokens",
                        "page_count",
                        "block_count",
                        "chunk_count",
                        "attempt_count",
                    )
                },
                "call_budget": {
                    "reserved": row["ai_calls_reserved"],
                    "configured_maximum": max_ai_calls,
                    "remaining": max(0, max_ai_calls - row["ai_calls_reserved"]),
                },
                "current_attempt_id": _id(row["current_attempt_id"]),
                "published_attempt_id": _id(row["published_attempt_id"]),
                "times": {
                    key: _time(row[key])
                    for key in (
                        "created_at",
                        "updated_at",
                        "terminal_at",
                        "deadline_at",
                        "lease_until",
                        "heartbeat_at",
                        "next_dispatch_at",
                        "last_dispatched_at",
                    )
                },
                "attempts": [_attempt(item) for item in attempts],
                "attempts_truncated": row["attempt_count"] > len(attempts),
                "performance": _performance(connection, identifier),
            }
        )
    return {
        "measured_at": datetime.now(UTC).isoformat(),
        "read_only": True,
        "selection": {
            "kind": "document_id" if document_id else "exact_filename",
            "total_matches": total,
            "returned": len(documents),
            "truncated": total > len(documents),
            "order": "newest_first",
        },
        "documents": documents,
        "telemetry": {
            "collection": "best_effort",
            "target_retention_days": 7,
            "missing_usage_is_unknown": True,
            "billing_complete": False,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    engine: Engine | None = None
    try:
        parser = Parser(description=__doc__)
        target = parser.add_mutually_exclusive_group(required=True)
        target.add_argument("--document-id")
        target.add_argument("--filename")
        parser.add_argument(
            "--limit", type=int, default=5, help="Newest filename matches, 1–5 (default 5)."
        )
        args = parser.parse_args(argv)
        if not 1 <= args.limit <= 5:
            raise DiagnosticError("INVALID_LIMIT")
        try:
            identifier = UUID(args.document_id) if args.document_id is not None else None
        except ValueError:
            raise DiagnosticError("INVALID_DOCUMENT_ID") from None
        if args.filename is not None and (
            not args.filename.strip() or len(args.filename) > 255 or "\x00" in args.filename
        ):
            raise DiagnosticError("INVALID_FILENAME")
        settings = get_settings()
        engine = create_engine(
            settings.database_url,
            hide_parameters=True,
            pool_pre_ping=False,
            connect_args={
                "connect_timeout": 3,
                "options": "-c statement_timeout=5000 -c lock_timeout=2000",
            },
        )
        with engine.connect() as raw:
            connection = raw.execution_options(
                isolation_level="REPEATABLE READ", postgresql_readonly=True
            )
            with connection.begin():
                report = diagnose(
                    connection,
                    document_id=identifier,
                    filename=args.filename,
                    limit=args.limit,
                    max_ai_calls=settings.max_document_ai_calls,
                )
        print(json.dumps(report, allow_nan=False))
        return 0
    except DiagnosticError as error:
        code = str(error)
    except ConfigurationError:
        code = "CONFIGURATION_INVALID"
    except SQLAlchemyError:
        code = "DATABASE_UNAVAILABLE"
    except Exception:
        code = "DIAGNOSTIC_FAILED"
    finally:
        if engine is not None:
            with suppress(Exception):
                engine.dispose()
    print(json.dumps({"error": {"code": code}, "read_only": True}))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
