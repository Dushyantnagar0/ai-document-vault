"""Safe durable upload acceptance and persisted document/source APIs."""

import asyncio
import base64
import binascii
import hashlib
import json
import unicodedata
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, BinaryIO, cast
from urllib.parse import quote
from uuid import UUID, uuid4

from fastapi import APIRouter, File, Header, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, tuple_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from sqlalchemy.orm import defer

from app.adapters.storage import LocalFileStorage, validate_upload
from app.core.config import get_settings
from app.db.models import Document, DocumentInsights
from app.db.profiles import ensure_profile
from app.domain.documents import ProcessingError
from app.services.ingestion import utcnow
from app.services.recovery import dispatch_document

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@dataclass
class APIError(Exception):
    status: int
    code: str
    message: str


class DocumentView(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    filename: str
    media_type: str
    byte_size: int
    sha256: str
    status: str
    stage: str
    profile_id: UUID
    created_at: datetime
    updated_at: datetime
    terminal_at: datetime | None
    error: dict[str, Any] | None
    page_count: int | None
    block_count: int | None
    chunk_count: int | None
    extracted_tokens: int | None
    attempt_count: int
    ai_calls_reserved: int
    extraction_coverage: str | None
    warnings: list[dict[str, Any]]


class UploadAccepted(BaseModel):
    document_id: UUID
    status: str
    stage: str
    status_url: str


class DocumentList(BaseModel):
    items: list[DocumentView]
    next_cursor: str | None


class InsightRecord(BaseModel):
    """Describe saved public fields without discarding persisted extensions."""

    model_config = ConfigDict(extra="allow")


class InsightQuote(InsightRecord):
    source_id: UUID
    quote: str


class InsightBlock(InsightRecord):
    text: str
    source_ids: list[UUID]
    support_quotes: list[InsightQuote]


class AnalysisLineage(InsightRecord):
    stage: str
    covered_chunk_ids: list[UUID]
    cited_chunk_ids: list[UUID]
    call_number: int


class AnalysisCall(InsightRecord):
    stage: str
    repair: bool
    outcome: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str | None = None
    error_code: str | None = None
    validation_reason: str | None = None
    diagnostics: dict[str, str | int | None] | None = None


class AnalysisUsage(InsightRecord):
    calls: int
    known_input_tokens: int
    known_output_tokens: int
    all_calls_reported_usage: bool
    call_history: list[AnalysisCall]


class InsightReviewMetadata(InsightRecord):
    """Read preserved experimental payloads without implying an active stage."""

    prompt_version: str
    candidate_count: int
    retained_count: int


class SavedAnalysis(InsightRecord):
    summary: list[InsightBlock]
    insights: list[InsightBlock]
    analysis_coverage: float | None = Field(
        default=None,
        description="Extracted-chunk participation, not semantic or visual completeness.",
    )
    covered_extracted_chunks: int | None = None
    total_extracted_chunks: int | None = None
    covered_chunk_ids: list[UUID] | None = None
    coverage_basis: str | None = None
    analysis_scope_complete: bool | None = Field(
        default=None,
        description="Reviewer scope assessment; separate from extracted-chunk participation.",
    )
    prompt_version: str | None = None
    model: str | None = None
    lineage: list[AnalysisLineage] | None = None
    usage: AnalysisUsage | None = None
    insight_review: InsightReviewMetadata | None = Field(
        default=None, description="Historical editorial experiment metadata, when present."
    )


class AnalysisCoverage(BaseModel):
    covered_extracted_chunks: int
    total_extracted_chunks: int


class InsightWarning(InsightRecord):
    code: str
    message: str
    locator: dict[str, str | int] | None = None


class DocumentInsightsView(BaseModel):
    document_id: UUID
    analysis: SavedAnalysis
    analysis_coverage: AnalysisCoverage
    extraction_coverage: str = Field(
        description="Observed native-text extraction coverage; not a visual completeness guarantee."
    )
    warnings: list[InsightWarning]
    model: str
    prompt_version: str


def display_filename(filename: str | None) -> str:
    value = (
        unicodedata.normalize("NFC", filename or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    )
    if not value or len(value) > 255 or any(unicodedata.category(char) == "Cc" for char in value):
        raise APIError(
            422,
            "INVALID_FILENAME",
            "Filename must contain 1–255 characters without control characters",
        )
    return value


def fingerprint(checksum: str, media_type: str, filename: str) -> str:
    return hashlib.sha256(
        json.dumps([checksum, media_type, filename], ensure_ascii=False).encode()
    ).hexdigest()


def accepted(document: Document) -> UploadAccepted:
    return UploadAccepted(
        document_id=document.id,
        status=document.status,
        stage=document.stage,
        status_url=f"/api/v1/documents/{document.id}",
    )


def translate_processing(error: ProcessingError) -> APIError:
    status = 422
    if error.code in {"UPLOAD_TOO_LARGE", "DOCX_ENTRY_LIMIT", "DOCX_EXPANSION_LIMIT"}:
        status = 413
    elif error.code == "UNSUPPORTED_FORMAT":
        status = 415
    elif error.code.startswith("STORAGE_"):
        status = 503
    return APIError(status, error.code, error.message)


@router.post("/upload", status_code=202, response_model=UploadAccepted)
async def upload(
    request: Request,
    file: Annotated[UploadFile, File(description="PDF or DOCX; processed asynchronously")],
    request_key: Annotated[UUID | None, Header(alias="Idempotency-Key")] = None,
) -> UploadAccepted:
    settings = get_settings()
    filename = display_filename(file.filename)
    storage = await asyncio.to_thread(LocalFileStorage, settings.local_storage_dir)
    staging_key, stage_path = await asyncio.to_thread(storage.create_staging)
    final_key: str | None = None
    committed = False
    acceptance_attempted = False
    stream: BinaryIO | None = None
    try:
        checksum, size = hashlib.sha256(), 0
        stream = await asyncio.to_thread(storage.open_staging, staging_key)
        while content := await file.read(64 * 1024):
            size += len(content)
            if size > settings.max_upload_bytes:
                raise APIError(413, "UPLOAD_TOO_LARGE", "File exceeds the configured upload limit")
            checksum.update(content)
            await asyncio.to_thread(stream.write, content)
        await asyncio.to_thread(stream.flush)
        media_type = await asyncio.to_thread(
            validate_upload, stage_path, filename, file.content_type, settings
        )
        request_fingerprint = fingerprint(checksum.hexdigest(), media_type, filename)
        final_key = await asyncio.to_thread(storage.finalize, staging_key)
        engine = cast(AsyncEngine, request.app.state.engine)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        now = utcnow()
        try:
            acceptance_attempted = True
            async with factory() as session, session.begin():
                profile_id = await ensure_profile(session, settings)
                document = Document(
                    id=uuid4(),
                    filename=filename,
                    media_type=media_type,
                    byte_size=size,
                    sha256=checksum.hexdigest(),
                    storage_key=final_key,
                    profile_id=profile_id,
                    upload_request_id=request_key,
                    request_fingerprint=request_fingerprint,
                    status="PENDING",
                    stage="QUEUED",
                    created_at=now,
                    updated_at=now,
                    next_dispatch_at=now,
                    deadline_at=now + timedelta(seconds=settings.processing_deadline_seconds),
                    warnings=[],
                    attempt_count=0,
                    ai_calls_reserved=0,
                )
                session.add(document)
                await session.flush()
            committed = True
        except IntegrityError:
            acceptance_attempted = False  # Definite rollback, unlike a lost commit acknowledgement.
            if request_key is None:
                raise
            async with factory() as session:
                winner = await session.scalar(
                    select(Document).where(Document.upload_request_id == request_key)
                )
                if winner is None:
                    raise
                if winner.request_fingerprint != request_fingerprint:
                    raise APIError(
                        409,
                        "IDEMPOTENCY_CONFLICT",
                        "This upload request ID was already used for different content or metadata",
                    ) from None
                return accepted(winner)
        try:
            await dispatch_document(engine, request.app.state.dispatcher, document.id)
        except Exception:
            pass  # Durable intent is committed; recovery can publish after DB/broker recovery.
        return accepted(document)
    except ProcessingError as error:
        raise translate_processing(error) from None
    finally:
        if stream is not None:
            await asyncio.to_thread(stream.close)
        await file.close()
        with suppress(ProcessingError):
            if final_key is not None and not committed and not acceptance_attempted:
                await asyncio.to_thread(storage.delete, final_key)
            else:
                await asyncio.to_thread(storage.discard_staging, staging_key)


def decode_cursor(cursor: str) -> tuple[datetime, UUID]:
    try:
        if not isinstance(cursor, str) or len(cursor) > 250:
            raise ValueError
        payload = json.loads(base64.b64decode(cursor, altchars=b"-_", validate=True).decode())
        if (
            not isinstance(payload, list)
            or len(payload) != 2
            or any(not isinstance(member, str) for member in payload)
        ):
            raise ValueError
        timestamp, identifier = payload
        parsed = datetime.fromisoformat(timestamp)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError
        return parsed.astimezone(UTC), UUID(identifier)
    except (ValueError, TypeError, UnicodeError, binascii.Error, OverflowError):
        raise APIError(422, "INVALID_CURSOR", "Pagination cursor is invalid") from None


@router.get("", response_model=DocumentList)
async def list_documents(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: str | None = None,
    status: Annotated[str | None, Query(pattern="^(PENDING|PROCESSING|COMPLETED|FAILED)$")] = None,
) -> DocumentList:
    factory = async_sessionmaker(request.app.state.engine, expire_on_commit=False)
    query = (
        select(Document)
        .options(defer(Document.extracted_text))
        .order_by(Document.created_at.desc(), Document.id.desc())
        .limit(limit + 1)
    )
    if cursor:
        timestamp, identifier = decode_cursor(cursor)
        query = query.where(tuple_(Document.created_at, Document.id) < (timestamp, identifier))
    if status:
        query = query.where(Document.status == status)
    async with factory() as session:
        rows = list((await session.scalars(query)).all())
    page, next_cursor = rows[:limit], None
    if len(rows) > limit:
        last = page[-1]
        next_cursor = base64.urlsafe_b64encode(
            json.dumps([last.created_at.isoformat(), str(last.id)]).encode()
        ).decode()
    return DocumentList(
        items=[DocumentView.model_validate(doc) for doc in page], next_cursor=next_cursor
    )


async def get_document(request: Request, document_id: UUID) -> Document:
    factory = async_sessionmaker(request.app.state.engine, expire_on_commit=False)
    async with factory() as session:
        document = await session.get(
            Document, document_id, options=[defer(Document.extracted_text)]
        )
        if document is None:
            raise APIError(404, "DOCUMENT_NOT_FOUND", "Document not found")
        return document


@router.get("/{document_id}", response_model=DocumentView)
async def retrieve(request: Request, document_id: UUID) -> DocumentView:
    return DocumentView.model_validate(await get_document(request, document_id))


@router.get("/{document_id}/file")
async def original(request: Request, document_id: UUID) -> StreamingResponse:
    doc = await get_document(request, document_id)
    storage = await asyncio.to_thread(LocalFileStorage, get_settings().local_storage_dir)

    source = await asyncio.to_thread(storage.open, doc.storage_key)

    def content() -> Iterator[bytes]:
        try:
            while chunk := source.read(64 * 1024):
                yield chunk
        finally:
            source.close()

    return StreamingResponse(
        content(),
        media_type=doc.media_type,
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{quote(doc.filename, safe='')}",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get(
    "/{document_id}/insights",
    response_model=DocumentInsightsView,
    response_model_exclude_unset=True,
)
async def insights(request: Request, document_id: UUID) -> dict[str, Any]:
    doc = await get_document(request, document_id)
    if doc.status != "COMPLETED":
        raise APIError(
            409,
            "DOCUMENT_NOT_READY",
            f"Document is {doc.status.lower()}; "
            "inspect its status and error before requesting insights",
        )
    factory = async_sessionmaker(request.app.state.engine, expire_on_commit=False)
    async with factory() as session:
        result = await session.get(DocumentInsights, document_id)
        if result is None:
            raise APIError(
                503,
                "INSIGHTS_UNAVAILABLE",
                "Completed document insights are temporarily unavailable",
            )
        return {
            "document_id": str(document_id),
            "analysis": result.payload,
            "analysis_coverage": {
                "covered_extracted_chunks": result.covered_chunks,
                "total_extracted_chunks": result.total_chunks,
            },
            "extraction_coverage": result.extraction_coverage,
            "warnings": [*result.warnings, *result.payload.get("analysis_warnings", [])],
            "model": result.model,
            "prompt_version": result.prompt_version,
        }
