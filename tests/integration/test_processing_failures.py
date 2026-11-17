"""Injected failures around real private-schema transactions, with offline providers."""

import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pymupdf
import pytest
from celery.exceptions import SoftTimeLimitExceeded
from sqlalchemy import event, func, select
from sqlalchemy.exc import (
    DataError,
    DBAPIError,
    IntegrityError,
    InterfaceError,
    OperationalError,
    ProgrammingError,
)
from sqlalchemy.orm import Session
from test_processing_claims import ClaimDatabase, require_claim
from test_processing_claims import claim_database as claim_database
from test_processing_claims import new_document as new_document

from app.adapters.openai import OpenAIEmbeddingProvider as ActualEmbeddingProvider
from app.adapters.storage import LocalFileStorage
from app.adapters.vectors import PgVectorPublisher
from app.core.config import Settings
from app.db.models import (
    Document,
    DocumentChunk,
    DocumentInsights,
    PerformanceSample,
    ProcessingAttempt,
)
from app.domain.documents import AIFailureDetails, EmbeddingBatch, Generated, ProcessingError
from app.services import ingestion

pytestmark = [
    pytest.mark.integration,
    pytest.mark.enable_socket,
    pytest.mark.skipif(
        os.environ.get("VAULT_INTEGRATION_TESTS") != "1", reason="requires isolated Compose stack"
    ),
]


class DriverFailure(Exception):
    def __init__(self, state: str | None) -> None:
        super().__init__("Injected database interruption")
        self.sqlstate = state


@pytest.mark.parametrize("component", [0.0, 1e-46])
def test_unusable_provider_vector_fails_before_analysis_and_atomic_publication(
    claim_database: ClaimDatabase,
    pipeline: tuple[UUID, Settings],
    monkeypatch: pytest.MonkeyPatch,
    component: float,
) -> None:
    """Real worker service/PG state, actual adapter validation, synthetic transport."""
    from app.adapters import openai

    identifier, configured = pipeline
    calls = []

    class InvalidEmbeddings(ActualEmbeddingProvider):
        def __init__(self, settings: Settings) -> None:
            self.settings = settings

            def create(**kwargs: Any) -> dict[str, Any]:
                calls.append("embedding")
                return {
                    "model": settings.openai_embedding_model,
                    "data": [
                        {"index": index, "embedding": [component] * 1536}
                        for index in range(len(kwargs["input"]))
                    ],
                    "usage": {"prompt_tokens": 7},
                }

            self._embeddings = SimpleNamespace(client=SimpleNamespace(create=create))

    def forbidden_generation(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Analysis must not run after unusable document embeddings")

    monkeypatch.setattr(openai, "OpenAIEmbeddingProvider", InvalidEmbeddings)
    monkeypatch.setattr(openai, "OpenAIGenerationProvider", forbidden_generation)
    ingestion.process_document(claim_database.engine, identifier, configured)
    assert calls == ["embedding"]
    with Session(claim_database.engine) as database:
        document = database.get(Document, identifier)
        assert document is not None and document.status == "FAILED"
        assert document.error is not None and document.error["code"] == "embedding_invalid_output"
        assert document.error["details"]["input_tokens"] == 7
        sample = database.scalar(
            select(PerformanceSample).where(
                PerformanceSample.document_id == identifier,
                PerformanceSample.operation == "embedding.document",
            )
        )
        assert sample is not None and sample.outcome == "failed"
        assert sample.input_tokens == 7 and sample.output_tokens is None
        assert document.ai_calls_reserved == 1 and document.published_attempt_id is None
        assert database.get(DocumentInsights, identifier) is None
        assert (
            database.scalar(
                select(func.count())
                .select_from(DocumentChunk)
                .where(DocumentChunk.document_id == identifier)
            )
            == 0
        )
        attempt = database.get(ProcessingAttempt, document.current_attempt_id)
        assert (
            attempt is not None and attempt.outcome == "FAILED" and attempt.error == document.error
        )


@pytest.mark.parametrize(
    "schema_name,reserved", [("InsightCandidates", 3), ("AnalysisReviewOutput", 5)]
)
def test_failed_final_quality_stage_cannot_publish_chunks_or_analysis(
    claim_database: ClaimDatabase,
    pipeline: tuple[UUID, Settings],
    monkeypatch: pytest.MonkeyPatch,
    schema_name: str,
    reserved: int,
) -> None:
    from app.adapters import openai

    identifier, configured = pipeline
    original = openai.OpenAIGenerationProvider.generate

    def interrupted(provider: Any, **kwargs: Any) -> Generated[Any]:
        if kwargs["schema"].__name__ == schema_name:
            raise ProcessingError("ai_invalid_output", "Injected final quality-stage failure")
        return original(provider, **kwargs)

    monkeypatch.setattr(openai.OpenAIGenerationProvider, "generate", interrupted)
    ingestion.process_document(claim_database.engine, identifier, configured)
    with Session(claim_database.engine) as database:
        document = database.get(Document, identifier)
        assert document is not None and document.status == "FAILED"
        assert document.error is not None and document.error["code"] == "ai_invalid_output"
        assert document.ai_calls_reserved == reserved and document.published_attempt_id is None
        assert database.get(DocumentInsights, identifier) is None
        assert (
            database.scalar(
                select(func.count())
                .select_from(DocumentChunk)
                .where(DocumentChunk.document_id == identifier)
            )
            == 0
        )
        attempt = database.get(ProcessingAttempt, document.current_attempt_id)
        assert attempt is not None and attempt.outcome == "FAILED"
        assert attempt.error == document.error


def test_safe_provider_failure_metadata_is_durable_and_logs_business_failure(
    claim_database: ClaimDatabase,
    new_document: Callable[..., UUID],
    caplog: pytest.LogCaptureFixture,
) -> None:
    identifier = new_document()
    current = require_claim(claim_database, identifier)
    diagnostics = AIFailureDetails(
        reason="output_token_limit",
        finish_reason="length",
        input_tokens=300,
        output_tokens=1500,
        max_output_tokens=1500,
        provider_request_id="req_test123",
    )
    ingestion.finish_failure(
        claim_database.engine,
        current,
        claim_database.settings,
        ProcessingError(
            "ai_output_truncated",
            "The AI response reached its output limit",
            diagnostics=diagnostics,
        ),
    )
    with Session(claim_database.engine) as database:
        document = database.get(Document, identifier)
        attempt = database.get(ProcessingAttempt, current.attempt_id)
        assert document is not None and attempt is not None
        assert document.status == "FAILED" and attempt.stage == "EXTRACTING"
        assert document.error is not None and document.error["details"] == diagnostics.as_dict()
        assert attempt.error == document.error
    assert "document_processing_failed" in caplog.text
    assert str(identifier) in caplog.text and "output_token_limit" in caplog.text


@pytest.fixture
def pipeline(
    claim_database: ClaimDatabase,
    new_document: Callable[..., UUID],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[UUID, Settings]:
    configured = claim_database.settings.model_copy(update={"local_storage_dir": tmp_path})
    with pymupdf.open() as fixture:  # type: ignore[no-untyped-call]
        fixture.new_page().insert_text((72, 72), "Owner: Test Operator.\nRenewal: December 2027.")
        content = fixture.tobytes()
    storage = LocalFileStorage(tmp_path)
    key, _ = storage.create_staging()
    with storage.open_staging(key) as stream:
        stream.write(content)
        stream.flush()
        storage.finalize(key)
    identifier = new_document()
    with Session(claim_database.engine) as database, database.begin():
        document = database.get(Document, identifier)
        assert document is not None
        document.storage_key, document.byte_size, document.sha256 = (
            key,
            len(content),
            hashlib.sha256(content).hexdigest(),
        )

    class Embeddings:
        def __init__(self, settings: Settings) -> None:
            self.settings = settings

        def embed_many(self, texts: list[str], *, timeout_seconds: float) -> EmbeddingBatch:
            return EmbeddingBatch(
                [[1.0] + [0.0] * 1535 for _ in texts], self.settings.openai_embedding_model, 7
            )

    class Generation:
        def __init__(self, settings: Settings, **options: Any) -> None:
            self.settings = settings

        def generate(self, **kwargs: Any) -> Generated[Any]:
            content = json.loads(kwargs["content"])
            if kwargs["schema"].__name__ == "SelectionOutput":
                item = content["evidence"][0]
                payload = {
                    "summary": [{"text": item["text"], "quote_ids": [item["quote_id"]]}],
                    "insights": [],
                }
            elif kwargs["schema"].__name__ == "InsightCandidates":
                payload = {"insights": []}
            elif kwargs["schema"].__name__ == "AnalysisReviewOutput":
                payload = {
                    "required_qualifications": [],
                    "insight_assessments": [],
                    "decisions": [
                        {
                            "claim_id": claim["claim_id"],
                            "supported": True,
                            "quote_ids": [
                                item["quote_id"]
                                for item in content["evidence"]
                                if claim["text"].strip() in item["text"]
                            ][:8],
                            "redundant_with": [],
                        }
                        for claim in content["claims"]
                    ],
                    "scope_complete": True,
                }
            else:
                item = content["items"][0]
                quote = item["text"].splitlines()[0]
                payload = {
                    "summary": [
                        {
                            "text": quote,
                            "source_ids": [item["source_id"]],
                            "support_quotes": [{"source_id": item["source_id"], "quote": quote}],
                        }
                    ],
                    "insights": [],
                }
            return Generated(
                kwargs["schema"].model_validate(payload), self.settings.openai_chat_model, 20, 20
            )

    monkeypatch.setattr("app.adapters.openai.OpenAIEmbeddingProvider", Embeddings)
    monkeypatch.setattr("app.adapters.openai.OpenAIGenerationProvider", Generation)
    return identifier, configured


@pytest.mark.parametrize(
    "error_type,state,invalidated,retryable",
    [
        (OperationalError, "08006", False, True),
        (OperationalError, "40001", False, True),
        (OperationalError, "40P01", False, True),
        (OperationalError, "55P03", False, True),
        (OperationalError, "57P01", False, True),
        (OperationalError, "57014", False, True),
        (OperationalError, None, False, True),
        (InterfaceError, None, False, True),
        (DBAPIError, None, True, True),
        (DBAPIError, "08003", False, True),
        (OperationalError, "28P01", False, False),
        (OperationalError, "3D000", False, False),
        (IntegrityError, "23505", False, False),
        (IntegrityError, "23505", True, False),
        (ProgrammingError, "42601", False, False),
        (DataError, "22003", False, False),
        (DBAPIError, None, False, False),
    ],
)
def test_publication_database_failure_rolls_back_and_only_known_transients_retry(
    claim_database: ClaimDatabase,
    pipeline: tuple[UUID, Settings],
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[DBAPIError],
    state: str | None,
    invalidated: bool,
    retryable: bool,
) -> None:
    identifier, configured = pipeline
    original = PgVectorPublisher.publish
    failure = error_type(
        "Injected publication operation",
        {},
        DriverFailure(state),
        connection_invalidated=invalidated,
    )

    def interrupted(publisher: PgVectorPublisher, *args: Any, **kwargs: Any) -> None:
        original(publisher, *args, **kwargs)
        # Real inserted chunks must roll back with the failed transaction.
        publisher.session.flush()
        raise failure

    monkeypatch.setattr(PgVectorPublisher, "publish", interrupted)
    ingestion.process_document(claim_database.engine, identifier, configured)
    with Session(claim_database.engine) as database:
        document = database.get(Document, identifier)
        assert document is not None and document.error is not None
        assert document.status == ("PENDING" if retryable else "FAILED")
        assert document.error["code"] == (
            "DATABASE_UNAVAILABLE" if retryable else "PROCESSING_FAILED"
        )
        assert document.ai_calls_reserved == 4
        assert document.published_attempt_id is None
        assert (
            database.scalar(
                select(func.count())
                .select_from(DocumentChunk)
                .where(DocumentChunk.document_id == identifier)
            )
            == 0
        )
        assert database.get(DocumentInsights, identifier) is None
        attempt = database.get(ProcessingAttempt, document.current_attempt_id)
        assert attempt is not None
        assert attempt.outcome == ("RETRY" if retryable else "FAILED")
        assert attempt.error == document.error


@pytest.mark.parametrize(
    "interrupt_target",
    ["app.adapters.extraction.pymupdf.open", "app.services.chunking._chunk_document"],
)
def test_parser_or_chunker_soft_interruption_becomes_bounded_retry_without_provider_calls(
    claim_database: ClaimDatabase,
    pipeline: tuple[UUID, Settings],
    monkeypatch: pytest.MonkeyPatch,
    interrupt_target: str,
) -> None:
    identifier, configured = pipeline

    def interrupted(*args: Any, **kwargs: Any) -> None:
        raise SoftTimeLimitExceeded()

    monkeypatch.setattr(interrupt_target, interrupted)
    ingestion.process_document(claim_database.engine, identifier, configured)
    with Session(claim_database.engine) as database:
        document = database.get(Document, identifier)
        assert document is not None and document.error is not None
        assert document.status == "PENDING" and document.error["code"] == "PROCESSING_INTERRUPTED"
        assert document.ai_calls_reserved == 0
        assert document.published_attempt_id is None
        attempt = database.get(ProcessingAttempt, document.current_attempt_id)
        assert attempt is not None and attempt.outcome == "RETRY"


def test_soft_limit_during_real_telemetry_insert_is_not_swallowed(
    claim_database: ClaimDatabase,
    pipeline: tuple[UUID, Settings],
) -> None:
    identifier, configured = pipeline
    signals = []

    def interrupt_once(
        connection: Any,
        cursor: Any,
        statement: str,
        parameters: Any,
        context: Any,
        executemany: bool,
    ) -> None:
        if "INSERT INTO SYSTEM_PERFORMANCE_SAMPLES" in statement.upper() and not signals:
            signals.append(True)
            raise SoftTimeLimitExceeded()

    event.listen(claim_database.engine, "before_cursor_execute", interrupt_once)
    try:
        ingestion.process_document(claim_database.engine, identifier, configured)
    finally:
        event.remove(claim_database.engine, "before_cursor_execute", interrupt_once)
    assert len(signals) == 1
    with Session(claim_database.engine) as database:
        document = database.get(Document, identifier)
        assert document is not None and document.error is not None
        assert document.status == "PENDING" and document.error["code"] == "PROCESSING_INTERRUPTED"
        assert document.ai_calls_reserved == 1
        assert document.published_attempt_id is None
        assert (
            database.scalar(
                select(func.count())
                .select_from(DocumentChunk)
                .where(DocumentChunk.document_id == identifier)
            )
            == 0
        )


def test_failure_persistence_outage_leaves_last_durable_claim_for_lease_recovery(
    claim_database: ClaimDatabase,
    new_document: Callable[..., UUID],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = require_claim(claim_database, new_document())
    with Session(claim_database.engine) as database:
        before = database.get(Document, claim.document_id)
        assert before is not None
        lease = before.lease_until

    def unavailable(*args: Any, **kwargs: Any) -> None:
        raise OperationalError("Injected connection loss", {}, DriverFailure("08006"))

    with monkeypatch.context() as temporary:
        temporary.setattr(ingestion, "Session", unavailable)
        ingestion.finish_failure(
            claim_database.engine,
            claim,
            claim_database.settings,
            ProcessingError("PROCESSING_FAILED", "Original failure"),
        )
    with Session(claim_database.engine) as database:
        document = database.get(Document, claim.document_id)
        attempt = database.get(ProcessingAttempt, claim.attempt_id)
        assert document is not None and attempt is not None
        assert document.status == "PROCESSING" and document.claim_token == claim.token
        assert document.lease_until == lease and document.terminal_at is None
        assert attempt.outcome == "RUNNING" and attempt.ended_at is None


def test_record_sample_directly_propagates_soft_limit(
    claim_database: ClaimDatabase,
    new_document: Callable[..., UUID],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = require_claim(claim_database, new_document())

    def interrupted(*args: Any, **kwargs: Any) -> None:
        raise SoftTimeLimitExceeded()

    monkeypatch.setattr(ingestion, "Session", interrupted)
    with pytest.raises(SoftTimeLimitExceeded):
        ingestion.record_sample(claim_database.engine, "test.interruption", time.monotonic(), claim)
