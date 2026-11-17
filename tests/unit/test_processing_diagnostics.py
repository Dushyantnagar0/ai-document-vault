"""Useful failure correlation without retaining private provider/document content."""

import json
import logging
from typing import Any
from uuid import uuid4

import pytest

from app.domain.documents import AIFailureDetails, ProcessingError
from app.services import ingestion
from app.services.ingestion import Claim, MeasuredGeneration, processing_event


def claim() -> Claim:
    return Claim(uuid4(), uuid4(), uuid4(), str(uuid4()), "application/pdf", uuid4())


def test_diagnostics_allowlist_rejects_arbitrary_text_and_invalid_usage() -> None:
    private = "private-provider-body-and-key"
    result = AIFailureDetails(
        reason=private,
        finish_reason=private,
        provider_request_id=private,
        input_tokens=True,
        output_tokens=-1,
        max_output_tokens=1_000_000_001,
    ).as_dict()
    assert private not in json.dumps(result)
    assert result == {
        "reason": "structured_parse_failure",
        "finish_reason": None,
        "provider_request_id": None,
        "input_tokens": None,
        "output_tokens": None,
        "max_output_tokens": None,
    }


def test_processing_failure_event_correlates_ids_but_never_error_body(
    caplog: pytest.LogCaptureFixture,
) -> None:
    private = "private-document-or-provider-content"
    current = claim()
    error = ProcessingError(
        "ai_output_truncated",
        private,
        diagnostics=AIFailureDetails(
            reason="output_token_limit",
            finish_reason="length",
            input_tokens=100,
            output_tokens=1500,
            max_output_tokens=1500,
            provider_request_id="req_test123",
        ),
    )
    with caplog.at_level(logging.INFO, logger="app.services.ingestion"):
        processing_event(
            "document_generation_failed",
            current,
            stage="ANALYSING",
            error=error,
            generation_call=2,
        )
    assert private not in caplog.text
    event = json.loads(caplog.records[-1].message)
    assert event["document_id"] == str(current.document_id)
    assert event["attempt_id"] == str(current.attempt_id)
    assert event["generation_call"] == 2
    assert event["details"]["finish_reason"] == "length"
    assert event["details"]["provider_request_id"] == "req_test123"


def test_failed_generation_keeps_available_usage_in_performance_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[dict[str, Any]] = []
    failure = ProcessingError(
        "ai_schema_invalid",
        "Structured response did not match the schema",
        diagnostics=AIFailureDetails(
            reason="schema_validation",
            finish_reason="stop",
            input_tokens=300,
            output_tokens=90,
            max_output_tokens=1500,
        ),
    )

    class FailedProvider:
        def generate(self, **kwargs: Any) -> Any:
            raise failure

    monkeypatch.setattr(ingestion, "record_sample", lambda *args, **kwargs: recorded.append(kwargs))
    measured = MeasuredGeneration(object(), claim(), FailedProvider())  # type: ignore[arg-type]
    for _ in range(2):
        with pytest.raises(ProcessingError) as caught:
            measured.generate(
                schema=object,
                system="private",
                content="private",
                max_output_tokens=1500,
                timeout_seconds=30,
            )
        assert caught.value is failure
    assert measured.calls == 2 and len(recorded) == 2
    assert all(row["input_tokens"] == 300 and row["output_tokens"] == 90 for row in recorded)
    assert all(row["outcome"] == "failed" for row in recorded)
