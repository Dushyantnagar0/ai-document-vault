"""Metadata-first chat diagnostics must not load or print private payloads."""

import json
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import OperationalError

from scripts import diagnose_chat as diagnostic

pytestmark = pytest.mark.unit
PRIVATE = "secret-document-and-provider-content"


def connection(total: int = 1) -> MagicMock:
    result = MagicMock()
    result.scalar.return_value = total
    row = {
        "id": UUID(int=1),
        "session_id": UUID(int=2),
        "turn_number": 1,
        "status": "FAILED",
        "response_status": 502,
        "ai_calls_reserved": 4,
        "planner_intent": "factual",
        "evidence_status": None,
        "versions": {
            "answer_prompt": "answer-v2",
            "pipeline": "grounded-chat-v2",
            "raw": PRIVATE,
            "planner_prompt": PRIVATE,
        },
        "error": {
            "code": "CHAT_INVALID_OUTPUT",
            "message": PRIVATE,
            "details": {"reason": "unknown_quote_id", "raw": PRIVATE},
        },
        "created_at": None,
        "ended_at": None,
    }
    result.execute.return_value.mappings.return_value.all.return_value = [row] if total else []
    return result


def test_report_preserves_failure_and_versions_without_private_payloads() -> None:
    database = connection(21)
    report = diagnostic.diagnose_chat(database, document_id=UUID(int=3), limit=1)
    assert report["read_only"] and report["truncated"]
    assert report["total_matches"] == 21 and report["returned"] == 1
    turn = report["turns"][0]
    assert turn["ai_calls_reserved"] == 4 and turn["response_status"] == 502
    assert turn["error"] == {
        "code": "CHAT_INVALID_OUTPUT",
        "details": {"reason": "unknown_quote_id"},
    }
    assert turn["versions"] == {"answer_prompt": "answer-v2", "pipeline": "grounded-chat-v2"}
    assert PRIVATE not in json.dumps(report)
    for call in database.execute.call_args_list:
        sql = str(call.args[0].compile(dialect=postgresql.dialect()))
        for column in (
            "retrieval_query",
            "context_snapshot",
            "response_body",
            "claim_token",
            "question_fingerprint",
        ):
            assert column not in sql
        assert sql.startswith("SELECT ") and " LIMIT " in sql


def test_no_turns_is_an_explicit_empty_read_only_report() -> None:
    report = diagnostic.diagnose_chat(connection(0), document_id=UUID(int=3))
    assert report["turns"] == [] and report["total_matches"] == 0 and not report["truncated"]


@pytest.mark.parametrize(
    "arguments,code",
    [
        (["--document-id", "bad"], "INVALID_DOCUMENT_ID"),
        (["--document-id", str(UUID(int=3)), "--limit", "0"], "INVALID_LIMIT"),
        (["--document-id", str(UUID(int=3)), "--limit", "51"], "INVALID_LIMIT"),
        ([], "INVALID_ARGUMENTS"),
    ],
)
def test_invalid_arguments_do_not_connect(
    arguments: list[str],
    code: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = MagicMock()
    monkeypatch.setattr(diagnostic, "create_engine", engine)
    assert diagnostic.main(arguments) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == code
    engine.assert_not_called()


def test_database_exception_never_prints_connection_or_exception(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(diagnostic, "get_settings", lambda: MagicMock(database_url=PRIVATE))
    engine = MagicMock()
    engine.connect.side_effect = OperationalError(PRIVATE, {}, Exception(PRIVATE))
    monkeypatch.setattr(diagnostic, "create_engine", lambda *args, **kwargs: engine)
    assert diagnostic.main(["--document-id", str(UUID(int=3))]) == 1
    output = capsys.readouterr().out
    assert PRIVATE not in output and "DATABASE_UNAVAILABLE" in output
    engine.dispose.assert_called_once()


def test_review_input_failure_keeps_safe_code_and_counts_without_private_content() -> None:
    database = connection()
    rows = database.execute.return_value.mappings.return_value.all.return_value
    rows[0]["error"] = {
        "code": "EVIDENCE_REVIEW_INPUT_LIMIT",
        "message": PRIVATE,
        "details": {
            "reason": "input_token_limit",
            "input_tokens": 14483,
            "max_output_tokens": 4096,
            "raw": PRIVATE,
        },
    }
    report = diagnostic.diagnose_chat(database, document_id=UUID(int=3))
    assert report["turns"][0]["error"] == {
        "code": "EVIDENCE_REVIEW_INPUT_LIMIT",
        "details": {
            "reason": "input_token_limit",
            "input_tokens": 14483,
            "max_output_tokens": 4096,
        },
    }
    assert PRIVATE not in json.dumps(report)
