"""Content-free diagnostic queries and CLI failure boundaries; no services/providers."""

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import OperationalError

from app.core.config import ConfigurationError
from app.domain.documents import AIFailureDetails
from scripts import diagnose as diagnostic

pytestmark = pytest.mark.unit
DOCUMENT = UUID(int=1)
ATTEMPT = UUID(int=2)


@pytest.mark.parametrize(
    "reason",
    [
        "unknown_quote_id",
        "duplicate_quote_id",
        "support_window_exceeded",
        "inconsistent_evidence_status",
    ],
)
def test_source_reference_reasons_remain_visible_in_safe_diagnostics(reason: str) -> None:
    assert diagnostic._error(
        {"code": "analysis_citation_invalid", "details": {"reason": reason}}
    ) == {"code": "analysis_citation_invalid", "details": {"reason": reason}}


BROKER = UUID(int=3)
PRIVATE = "private-content-and-provider-key-never-print"


def document(identifier: UUID = DOCUMENT) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(
        (
            "current_attempt_id",
            "published_attempt_id",
            "created_at",
            "updated_at",
            "terminal_at",
            "deadline_at",
            "lease_until",
            "heartbeat_at",
            "next_dispatch_at",
            "last_dispatched_at",
        ),
    )
    row.update(
        id=identifier,
        status="FAILED",
        stage="FAILED",
        byte_size=99,
        extracted_tokens=80,
        page_count=None,
        block_count=2,
        chunk_count=None,
        attempt_count=1,
        ai_calls_reserved=3,
        error={
            "code": "ai_invalid_output",
            "message": PRIVATE,
            "details": {"failure_kind": "missing_structured_output", "raw": PRIVATE},
        },
        current_attempt_id=ATTEMPT,
    )
    return row


def attempt() -> dict[str, Any]:
    return {
        "id": ATTEMPT,
        "attempt_number": 1,
        "broker_task_id": str(BROKER),
        "outcome": "FAILED",
        "stage": "ANALYSING",
        "requested_at": datetime(2026, 11, 15, tzinfo=UTC),
        "started_at": None,
        "heartbeat_at": None,
        "stage_started_at": None,
        "ended_at": None,
        "stage_durations": {
            "embedding_ms": 90,
            "analysing_ms": 180,
            "raw": PRIVATE,
            "total_ms": float("nan"),
            "extracting_ms": -2,
        },
        "error": {"code": "ai_invalid_output", "message": PRIVATE},
    }


def performance(operation: str, samples: int, tokens: int | None) -> dict[str, Any]:
    return {
        "operation": operation,
        "outcome": "failed" if tokens is None else "succeeded",
        "error_class": "ai_invalid_output" if tokens is None else None,
        "sample_count": samples,
        "total_duration_ms": 130.5,
        "input_tokens": tokens,
        "input_token_samples": int(tokens is not None),
        "output_tokens": None,
        "output_token_samples": 0,
    }


def connection(documents: list[dict[str, Any]], total: int | None = None) -> MagicMock:
    result = MagicMock()
    result.scalar.return_value = len(documents) if total is None else total
    rows = [documents]
    for _ in documents:
        rows.extend(
            [
                [attempt()],
                [
                    performance("embedding.document", 1, 80),
                    performance("generation.document", 2, None),
                ],
            ]
        )
    responses = []
    for batch in rows:
        response = MagicMock()
        response.mappings.return_value.all.return_value = batch
        responses.append(response)
    result.execute.side_effect = responses
    return result


def test_failure_keeps_durable_analysis_stage_counts_ids_and_unknown_usage() -> None:
    conn = connection([document()])
    report = diagnostic.diagnose(
        conn, document_id=DOCUMENT, filename=None, limit=5, max_ai_calls=64
    )
    saved = report["documents"][0]
    assert saved["status"] == "FAILED" and saved["stage"] == "FAILED"
    assert saved["attempts"][0]["stage"] == "ANALYSING"
    assert saved["attempts"][0]["broker_task_id"] == str(BROKER)
    assert saved["attempts"][0]["stage_durations_ms"] == {"embedding_ms": 90, "analysing_ms": 180}
    assert saved["call_budget"] == {"reserved": 3, "configured_maximum": 64, "remaining": 61}
    assert saved["performance"][1]["sample_count"] == 2
    assert saved["performance"][1]["known_usage"]["input_tokens"] is None
    assert saved["performance"][1]["known_usage"]["output_token_samples"] == 0
    encoded = json.dumps(report, allow_nan=False)
    assert PRIVATE not in encoded and "extracted_text" not in encoded
    for call in [*conn.scalar.call_args_list, *conn.execute.call_args_list]:
        sql = str(call.args[0].compile(dialect=postgresql.dialect()))
        assert sql.lstrip().startswith("SELECT")
        assert all(
            name not in sql
            for name in (
                "extracted_text",
                "embedding,",
                "document_chunks",
                "chat_messages",
                "claim_token",
                "storage_key",
            )
        )


def test_filename_lookup_returns_all_five_latest_matches_and_reports_truncation() -> None:
    docs = [document(UUID(int=index)) for index in range(1, 6)]
    conn = connection(docs, total=8)
    name = "literal ' OR 1=1 --.pdf"
    report = diagnostic.diagnose(conn, document_id=None, filename=name, limit=5, max_ai_calls=64)
    assert len(report["documents"]) == 5
    assert report["selection"] == {
        "kind": "exact_filename",
        "total_matches": 8,
        "returned": 5,
        "truncated": True,
        "order": "newest_first",
    }
    statement = conn.execute.call_args_list[0].args[0].compile(dialect=postgresql.dialect())
    assert name not in str(statement) and name in statement.params.values()
    assert "ORDER BY documents.created_at DESC, documents.id DESC" in str(statement)
    assert 5 in statement.params.values()
    assert name not in json.dumps(report)


def test_no_match_stops_before_attempt_or_performance_queries() -> None:
    conn = connection([], total=0)
    with pytest.raises(diagnostic.DiagnosticError, match="DOCUMENT_NOT_FOUND"):
        diagnostic.diagnose(conn, document_id=DOCUMENT, filename=None, limit=5, max_ai_calls=64)
    conn.execute.assert_not_called()


def test_error_details_are_allowlisted_not_a_provider_or_message_dump() -> None:
    saved = diagnostic._error(
        {
            "code": "ai_invalid_output",
            "message": PRIVATE,
            "details": {
                "validation_reason": "missing_source_quote",
                "failure_kind": PRIVATE,
                "finish_reason": "length",
                "output_tokens": 123,
                "max_output_tokens": float("inf"),
                "validation_error_count": True,
                "raw": PRIVATE,
                "nested": {"key": PRIVATE},
            },
        }
    )
    assert saved == {
        "code": "ai_invalid_output",
        "details": {
            "validation_reason": "missing_source_quote",
            "finish_reason": "length",
            "output_tokens": 123,
        },
    }
    assert diagnostic._error({"code": PRIVATE}) == {"code": "UNKNOWN"}


@pytest.mark.parametrize(
    "code,reason",
    [
        ("ai_output_truncated", "output_token_limit"),
        ("ai_invalid_json", "invalid_json"),
        ("ai_schema_invalid", "schema_validation"),
    ],
)
def test_new_provider_diagnostics_keep_safe_reason_usage_budget_and_request_id(
    code: str,
    reason: str,
) -> None:
    details = AIFailureDetails(
        reason=reason,
        finish_reason="length",
        input_tokens=400,
        output_tokens=768,
        max_output_tokens=768,
        provider_request_id="req_fixture_123",
    ).as_dict()
    safe = diagnostic._error({"code": code, "message": PRIVATE, "details": details})
    assert safe == {"code": code, "details": details}
    details["provider_request_id"] = PRIVATE
    assert (
        "provider_request_id"
        not in diagnostic._error({"code": code, "details": details})["details"]
    )
    assert diagnostic._number(10**1000) is None


@pytest.mark.parametrize(
    "argv,code",
    [
        ([], "INVALID_ARGUMENTS"),
        (["--document-id", PRIVATE], "INVALID_DOCUMENT_ID"),
        (["--document-id", ""], "INVALID_DOCUMENT_ID"),
        (["--document-id", str(DOCUMENT), "--filename", PRIVATE], "INVALID_ARGUMENTS"),
        (["--filename", ""], "INVALID_FILENAME"),
        (["--filename", "a.pdf", "--limit", "6"], "INVALID_LIMIT"),
        (["--filename", "a.pdf", "--limit", PRIVATE], "INVALID_ARGUMENTS"),
    ],
)
def test_cli_rejects_invalid_input_before_settings_or_database(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    code: str,
) -> None:
    def forbidden() -> None:
        pytest.fail("Invalid input must not load settings or connect")

    monkeypatch.setattr(diagnostic, "get_settings", forbidden)
    assert diagnostic.main(argv) == 1
    output = capsys.readouterr()
    assert json.loads(output.out) == {"error": {"code": code}, "read_only": True}
    assert not output.err and PRIVATE not in output.out


def test_cli_uses_read_only_snapshot_and_disposes_without_raw_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    conn = connection([document()])
    conn.execution_options.return_value = conn
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    factory = MagicMock(return_value=engine)
    monkeypatch.setattr(
        diagnostic,
        "get_settings",
        lambda: SimpleNamespace(database_url=PRIVATE, max_document_ai_calls=64),
    )
    monkeypatch.setattr(diagnostic, "create_engine", factory)
    assert diagnostic.main(["--document-id", str(DOCUMENT)]) == 0
    conn.execution_options.assert_called_once_with(
        isolation_level="REPEATABLE READ", postgresql_readonly=True
    )
    engine.dispose.assert_called_once()
    assert factory.call_args.kwargs["hide_parameters"] is True
    assert json.loads(capsys.readouterr().out)["read_only"] is True
    engine.connect.side_effect = OperationalError(PRIVATE, {}, RuntimeError(PRIVATE))
    assert diagnostic.main(["--document-id", str(DOCUMENT)]) == 1
    output = capsys.readouterr().out
    assert PRIVATE not in output and json.loads(output)["error"]["code"] == "DATABASE_UNAVAILABLE"


def test_invalid_configuration_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def invalid() -> None:
        raise ConfigurationError(PRIVATE)

    monkeypatch.setattr(diagnostic, "get_settings", invalid)
    assert diagnostic.main(["--filename", "a.pdf"]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "CONFIGURATION_INVALID"
