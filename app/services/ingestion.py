"""Fenced synchronous worker orchestration; no transaction spans a provider call."""

import json
import logging
import random
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar
from uuid import UUID, uuid4

from celery.exceptions import SoftTimeLimitExceeded
from pydantic import BaseModel
from sqlalchemy import Engine
from sqlalchemy.exc import (
    DataError,
    DBAPIError,
    DisconnectionError,
    IntegrityError,
    InterfaceError,
    OperationalError,
    ProgrammingError,
    SQLAlchemyError,
)
from sqlalchemy.exc import TimeoutError as PoolTimeoutError
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models import (
    Document,
    DocumentInsights,
    IndexProfile,
    PerformanceSample,
    ProcessingAttempt,
)
from app.db.profiles import profile_values
from app.domain.documents import Chunk, Generated, GenerationProvider, ProcessingError


def utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class Claim:
    document_id: UUID
    attempt_id: UUID
    token: UUID
    storage_key: str
    media_type: str
    profile_id: UUID


def processing_event(
    event: str,
    claim: Claim,
    *,
    stage: str | None = None,
    outcome: str | None = None,
    error: ProcessingError | None = None,
    generation_call: int | None = None,
) -> None:
    """Correlate business outcomes without filenames, text, keys or raw exceptions."""
    payload: dict[str, Any] = {
        "event": event,
        "document_id": str(claim.document_id),
        "attempt_id": str(claim.attempt_id),
        "stage": stage,
        "outcome": outcome,
    }
    if generation_call is not None:
        payload["generation_call"] = generation_call
    if error is not None:
        payload["error_code"] = error.code
        if error.diagnostics is not None:
            payload["details"] = error.diagnostics.as_dict()
    logging.getLogger(__name__).log(
        logging.WARNING if error else logging.INFO, "%s", json.dumps(payload, sort_keys=True)
    )


class LostClaim(ProcessingError):
    def __init__(self) -> None:
        super().__init__(
            "CLAIM_EXPIRED", "Processing ownership expired; recovery will determine the outcome"
        )


def owned(document: Document, claim: Claim, now: datetime) -> bool:
    return (
        document.status == "PROCESSING"
        and document.claim_token == claim.token
        and document.current_attempt_id == claim.attempt_id
        and document.lease_until is not None
        and document.lease_until > now
        and document.deadline_at > now
    )


def claim_document(
    engine: Engine, document_id: UUID, settings: Settings, task_id: str | None
) -> Claim | None:
    now = utcnow()
    with Session(engine) as session, session.begin():
        doc = session.get(Document, document_id, with_for_update=True)
        now = utcnow()
        if doc is None or doc.status != "PENDING" or doc.next_dispatch_at > now:
            return None
        if (
            doc.deadline_at <= now
            or doc.attempt_count >= settings.processing_max_attempts
            or doc.ai_calls_reserved >= settings.max_document_ai_calls
            or (doc.attempt_count == 0 and doc.created_at <= now - timedelta(seconds=900))
        ):
            doc.status, doc.stage, doc.terminal_at = "FAILED", "FAILED", now
            doc.updated_at = now
            doc.error = {
                "code": "PROCESSING_BUDGET_EXHAUSTED",
                "message": "Processing time, attempt or AI-call limit exhausted; upload again",
            }
            return None
        token, attempt_id = uuid4(), uuid4()
        attempt = ProcessingAttempt(
            id=attempt_id,
            document_id=doc.id,
            attempt_number=doc.attempt_count + 1,
            claim_token=token,
            requested_at=doc.next_dispatch_at,
            started_at=now,
            heartbeat_at=now,
            outcome="RUNNING",
            stage="EXTRACTING",
            stage_started_at=now,
            stage_durations={},
            broker_task_id=task_id,
        )
        session.add(attempt)
        session.flush()
        doc.status, doc.stage = "PROCESSING", "EXTRACTING"
        doc.error = None
        doc.attempt_count += 1
        doc.current_attempt_id, doc.claim_token = attempt_id, token
        doc.heartbeat_at, doc.updated_at = now, now
        doc.lease_until = now + timedelta(seconds=180)
        return Claim(doc.id, attempt_id, token, doc.storage_key, doc.media_type, doc.profile_id)


def reserve_call(engine: Engine, claim: Claim, settings: Settings) -> float:
    now = utcnow()
    with Session(engine) as session, session.begin():
        doc = session.get(Document, claim.document_id, with_for_update=True)
        now = utcnow()
        if doc is None or not owned(doc, claim, now):
            raise LostClaim()
        if doc.ai_calls_reserved >= settings.max_document_ai_calls:
            raise ProcessingError(
                "AI_CALL_BUDGET_EXHAUSTED",
                "Document AI-call budget exhausted; no further provider calls will be made",
            )
        doc.ai_calls_reserved += 1
        return (doc.deadline_at - now).total_seconds()


def heartbeat(engine: Engine, claim: Claim) -> bool:
    now = utcnow()
    with Session(engine) as session, session.begin():
        doc = session.get(Document, claim.document_id, with_for_update=True)
        now = utcnow()
        if doc is None or not owned(doc, claim, now):
            return False
        doc.heartbeat_at, doc.lease_until = now, now + timedelta(seconds=180)
        attempt = session.get(ProcessingAttempt, claim.attempt_id)
        assert attempt is not None
        attempt.heartbeat_at = now
        return True


def set_stage(engine: Engine, claim: Claim, stage: str) -> None:
    with Session(engine) as session, session.begin():
        doc = session.get(Document, claim.document_id, with_for_update=True)
        now = utcnow()
        if doc is None or not owned(doc, claim, utcnow()):
            raise LostClaim()
        doc.stage, doc.updated_at = stage, utcnow()
        attempt = session.get(ProcessingAttempt, claim.attempt_id)
        assert attempt is not None
        now = utcnow()
        attempt.stage_durations = {
            **attempt.stage_durations,
            f"{attempt.stage.lower()}_ms": (now - attempt.stage_started_at).total_seconds() * 1000,
        }
        attempt.stage, attempt.stage_started_at = stage, now
    processing_event("document_processing_stage", claim, stage=stage)


def transient_database_error(error: SQLAlchemyError) -> bool:
    """Only known connection/transaction interruptions permit another attempt."""
    if isinstance(error, (IntegrityError, ProgrammingError, DataError)):
        return False
    if isinstance(error, (DisconnectionError, PoolTimeoutError)):
        return True
    if not isinstance(error, DBAPIError):
        return False
    if error.connection_invalidated:
        return True
    state = getattr(error.orig, "sqlstate", None) or getattr(error.orig, "pgcode", None)
    if isinstance(state, str):
        return state.startswith(("08", "40")) or state in {"55P03", "57P01", "57014"}
    # Network loss/closed connections may have no server response/SQLSTATE.
    return state is None and isinstance(error, (OperationalError, InterfaceError))


def finish_failure(
    engine: Engine, claim: Claim, settings: Settings, error: ProcessingError
) -> None:
    try:
        _persist_failure(engine, claim, settings, error)
    except SQLAlchemyError as persistence_error:
        if not transient_database_error(persistence_error):
            raise
        logging.getLogger(__name__).warning(
            "Failure persistence unavailable [%s] for document %s; awaiting lease recovery",
            type(persistence_error).__name__,
            claim.document_id,
        )
        # The failed transaction rolls back. Do not report a fabricated terminal
        # state: its last durable claim/attempt remains for ordinary lease repair.


def _persist_failure(
    engine: Engine, claim: Claim, settings: Settings, error: ProcessingError
) -> None:
    now = utcnow()
    with Session(engine) as session, session.begin():
        doc = session.get(Document, claim.document_id, with_for_update=True)
        now = utcnow()
        if doc is None or not owned(doc, claim, now):
            return
        attempt = session.get(ProcessingAttempt, claim.attempt_id)
        assert attempt is not None
        safe: dict[str, Any] = {"code": error.code, "message": error.message}
        if error.diagnostics is not None:
            safe["details"] = error.diagnostics.as_dict()
        failed_stage = doc.stage
        delay = (15 if doc.attempt_count == 1 else 60) + random.uniform(0, 3)
        if error.retry_after_seconds is not None:
            delay = max(delay, min(error.retry_after_seconds, 60))
        retry = (
            error.retryable
            and doc.attempt_count < settings.processing_max_attempts
            and now + timedelta(seconds=delay) < doc.deadline_at
            and doc.ai_calls_reserved < settings.max_document_ai_calls
        )
        attempt.stage_durations = {
            **attempt.stage_durations,
            f"{attempt.stage.lower()}_ms": (now - attempt.stage_started_at).total_seconds() * 1000,
        }
        attempt.outcome, attempt.ended_at, attempt.error = "RETRY" if retry else "FAILED", now, safe
        doc.status, doc.stage = ("PENDING", "RETRY_WAIT") if retry else ("FAILED", "FAILED")
        doc.error, doc.updated_at = safe, now
        doc.claim_token, doc.lease_until = None, None
        doc.dispatch_reserved_until = None
        if retry:
            doc.next_dispatch_at = now + timedelta(seconds=delay)
            doc.last_dispatched_at = None
        else:
            doc.terminal_at = now
    processing_event(
        "document_processing_retry" if retry else "document_processing_failed",
        claim,
        stage=failed_stage,
        outcome="RETRY" if retry else "FAILED",
        error=error,
    )


def record_sample(
    engine: Engine,
    operation: str,
    started: float,
    claim: Claim,
    *,
    outcome: str = "succeeded",
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    error_class: str | None = None,
) -> None:
    try:
        with Session(engine) as session, session.begin():
            session.add(
                PerformanceSample(
                    operation=operation,
                    duration_ms=(time.monotonic() - started) * 1000,
                    outcome=outcome,
                    document_id=claim.document_id,
                    attempt_id=claim.attempt_id,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    error_class=error_class,
                )
            )
    except SoftTimeLimitExceeded:
        raise
    except Exception:
        pass  # Mandatory attempt state is separate; telemetry cannot undo success.


T = TypeVar("T", bound=BaseModel)


class MeasuredGeneration:
    def __init__(self, engine: Engine, claim: Claim, provider: GenerationProvider) -> None:
        self.engine, self.claim, self.provider = engine, claim, provider
        self.calls = 0

    def generate(
        self,
        *,
        schema: type[T],
        system: str,
        content: str,
        max_output_tokens: int,
        timeout_seconds: float,
    ) -> Generated[T]:
        started = time.monotonic()
        self.calls += 1
        try:
            result = self.provider.generate(
                schema=schema,
                system=system,
                content=content,
                max_output_tokens=max_output_tokens,
                timeout_seconds=timeout_seconds,
            )
        except ProcessingError as error:
            details = error.diagnostics.as_dict() if error.diagnostics is not None else {}
            input_usage, output_usage = details.get("input_tokens"), details.get("output_tokens")
            processing_event(
                "document_generation_failed",
                self.claim,
                stage="ANALYSING",
                error=error,
                generation_call=self.calls,
            )
            record_sample(
                self.engine,
                "generation.document",
                started,
                self.claim,
                outcome="failed",
                error_class=error.code,
                input_tokens=input_usage if isinstance(input_usage, int) else None,
                output_tokens=output_usage if isinstance(output_usage, int) else None,
            )
            raise
        record_sample(
            self.engine,
            "generation.document",
            started,
            self.claim,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
        return result


def process_document(
    engine: Engine, document_id: UUID, settings: Settings, task_id: str | None = None
) -> None:
    from app.adapters.extraction import extractor_for
    from app.adapters.openai import OpenAIEmbeddingProvider, OpenAIGenerationProvider
    from app.adapters.storage import LocalFileStorage
    from app.adapters.vectors import PgVectorPublisher
    from app.services.analysis import analyse_document
    from app.services.chunking import chunk_document

    claim = claim_document(engine, document_id, settings, task_id)
    if claim is None:
        return
    processing_event("document_processing_started", claim, stage="EXTRACTING", outcome="RUNNING")
    stopped = threading.Event()

    def renew() -> None:
        while not stopped.wait(30):
            try:
                if not heartbeat(engine, claim):
                    return
            except Exception:
                return  # Expiry fences the worker; never resurrect uncertain ownership.

    thread = threading.Thread(target=renew, daemon=True)
    thread.start()
    started = time.monotonic()
    processing_outcome = "failed"
    try:
        with Session(engine) as session:
            profile = session.get(IndexProfile, claim.profile_id)
            if profile is None or profile.fingerprint != profile_values(settings)["fingerprint"]:
                raise ProcessingError(
                    "INDEX_PROFILE_MISMATCH",
                    "Configured index recipe differs from the accepted document profile",
                )
        extraction = extractor_for(claim.media_type, settings).extract(
            LocalFileStorage(settings.local_storage_dir).read(claim.storage_key)
        )
        set_stage(engine, claim, "CHUNKING")
        chunks = chunk_document(extraction, settings)
        vectors: list[list[float]] = []
        embeddings = OpenAIEmbeddingProvider(settings)
        set_stage(engine, claim, "EMBEDDING")
        offset = 0
        while offset < len(chunks):
            batch: list[Chunk] = []
            tokens = 0
            while offset < len(chunks) and len(batch) < settings.embedding_batch_size:
                chunk = chunks[offset]
                if batch and tokens + chunk.token_count > settings.embedding_batch_max_tokens:
                    break
                batch.append(chunk)
                tokens += chunk.token_count
                offset += 1
            remaining = reserve_call(engine, claim, settings)
            call_started = time.monotonic()
            try:
                embedded = embeddings.embed_many(
                    [chunk.text for chunk in batch],
                    timeout_seconds=min(remaining, settings.openai_embedding_timeout_seconds),
                )
            except ProcessingError as error:
                details = error.diagnostics.as_dict() if error.diagnostics is not None else {}
                known_input = details.get("input_tokens")
                record_sample(
                    engine,
                    "embedding.document",
                    call_started,
                    claim,
                    outcome="failed",
                    error_class=error.code,
                    input_tokens=known_input if isinstance(known_input, int) else None,
                )
                raise
            vectors.extend(embedded.vectors)
            record_sample(
                engine,
                "embedding.document",
                call_started,
                claim,
                input_tokens=embedded.input_tokens,
            )
        set_stage(engine, claim, "ANALYSING")
        analysis = analyse_document(
            chunks,
            settings,
            MeasuredGeneration(engine, claim, OpenAIGenerationProvider(settings)),
            lambda: reserve_call(engine, claim, settings),
            insight_generator=MeasuredGeneration(
                engine,
                claim,
                OpenAIGenerationProvider(
                    settings, reasoning_effort=settings.openai_review_reasoning_effort
                ),
            ),
            reviewer=MeasuredGeneration(
                engine,
                claim,
                OpenAIGenerationProvider(
                    settings, reasoning_effort=settings.openai_review_reasoning_effort
                ),
            ),
        )
        if set(analysis.covered_chunk_ids) != {chunk.id for chunk in chunks} or len(vectors) != len(
            chunks
        ):
            raise ProcessingError(
                "INCOMPLETE_ANALYSIS", "Not all extracted chunks were analysed and embedded"
            )
        now = utcnow()
        with Session(engine) as session, session.begin():
            doc = session.get(Document, document_id, with_for_update=True)
            now = utcnow()
            if doc is None or not owned(doc, claim, now) or doc.profile_id != claim.profile_id:
                raise LostClaim()
            assert doc.lease_until is not None
            publication_deadline = min(doc.lease_until, doc.deadline_at)
            PgVectorPublisher(session).publish(
                doc.id, claim.attempt_id, claim.profile_id, chunks, vectors
            )
            warnings = [asdict(warning) for warning in extraction.warnings]
            session.add(
                DocumentInsights(
                    document_id=doc.id,
                    attempt_id=claim.attempt_id,
                    payload=analysis.payload,
                    covered_chunks=len(chunks),
                    total_chunks=len(chunks),
                    extraction_coverage=extraction.coverage,
                    warnings=warnings,
                    model=analysis.model,
                    prompt_version=analysis.prompt_version,
                )
            )
            import tiktoken

            doc.extracted_text, doc.extracted_tokens = (
                extraction.text,
                len(
                    tiktoken.get_encoding("cl100k_base").encode(
                        extraction.text, disallowed_special=()
                    )
                ),
            )
            doc.page_count, doc.block_count, doc.chunk_count = (
                extraction.page_count,
                len(extraction.blocks),
                len(chunks),
            )
            doc.extractor_version = f"{extraction.extractor_name}:{extraction.extractor_version}"
            doc.extraction_coverage, doc.warnings = extraction.coverage, warnings
            doc.status, doc.stage, doc.error = "COMPLETED", "COMPLETED", None
            doc.published_attempt_id = claim.attempt_id
            doc.terminal_at, doc.updated_at = now, now
            doc.claim_token, doc.lease_until, doc.dispatch_reserved_until = None, None, None
            attempt = session.get(ProcessingAttempt, claim.attempt_id)
            assert attempt is not None
            attempt.outcome, attempt.stage, attempt.ended_at = "SUCCEEDED", "COMPLETED", now
            attempt.stage_durations = {
                **attempt.stage_durations,
                "analysing_ms": (now - attempt.stage_started_at).total_seconds() * 1000,
                "total_ms": (time.monotonic() - started) * 1000,
            }
            session.flush()
            if utcnow() >= publication_deadline:
                raise LostClaim()
        processing_outcome = "succeeded"
        processing_event(
            "document_processing_completed", claim, stage="COMPLETED", outcome="SUCCEEDED"
        )
    except LostClaim:
        processing_outcome = "interrupted"
    except SoftTimeLimitExceeded:
        processing_outcome = "interrupted"
        finish_failure(
            engine,
            claim,
            settings,
            ProcessingError(
                "PROCESSING_INTERRUPTED",
                "Worker time limit interrupted processing; retrying within remaining budgets",
                retryable=True,
            ),
        )
    except ProcessingError as error:
        finish_failure(engine, claim, settings, error)
    except SQLAlchemyError as error:
        retryable = transient_database_error(error)
        logging.getLogger(__name__).error(
            "Database processing interruption [%s] for document %s",
            type(error).__name__,
            document_id,
        )
        finish_failure(
            engine,
            claim,
            settings,
            ProcessingError(
                "DATABASE_UNAVAILABLE" if retryable else "PROCESSING_FAILED",
                "Database operation interrupted; retries remain subject to processing budgets"
                if retryable
                else "Document results could not be saved; inspect diagnostics and upload again",
                retryable=retryable,
            ),
        )
    except Exception as error:
        logging.getLogger(__name__).error(
            "Unexpected processing failure [%s] for document %s", type(error).__name__, document_id
        )
        finish_failure(
            engine,
            claim,
            settings,
            ProcessingError(
                "PROCESSING_FAILED",
                "Document processing failed; inspect service diagnostics and upload again",
            ),
        )
    finally:
        stopped.set()
        thread.join(timeout=2)
        record_sample(engine, "processing.total", started, claim, outcome=processing_outcome)
