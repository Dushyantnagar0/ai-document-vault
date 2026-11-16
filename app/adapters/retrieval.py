"""Exact cosine retrieval constrained to allowed published document evidence."""

import math
from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from sqlalchemy import ColumnElement, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from sqlalchemy.orm import defer

from app.db.models import Document, DocumentChunk
from app.domain.conversation import Evidence
from app.domain.documents import ProcessingError, SourceSpan
from app.domain.vectors import float32_vector


class VectorRetriever(Protocol):
    async def search(
        self,
        allowed_doc_ids: Sequence[UUID],
        profile_id: UUID,
        query: Sequence[float],
        limit: int,
    ) -> tuple[Evidence, ...]: ...

    async def get_chunks(
        self,
        allowed_doc_ids: Sequence[UUID],
        profile_id: UUID,
        source_ids: Sequence[UUID],
    ) -> tuple[Evidence, ...]: ...


def _query_vector(query: Sequence[float]) -> list[float]:
    """Validate the actual float32 vector sent to the fixed pgvector schema."""
    try:
        return float32_vector(query, 1536)
    except ValueError:
        raise ValueError("Query must be a finite nonzero 1536-dimensional float32 vector") from None


def _scope(allowed_doc_ids: Sequence[UUID], profile_id: UUID) -> tuple[ColumnElement[bool], ...]:
    return (
        Document.id.in_(tuple(allowed_doc_ids)),
        Document.status == "COMPLETED",
        Document.profile_id == profile_id,
        DocumentChunk.profile_id == profile_id,
        Document.published_attempt_id == DocumentChunk.attempt_id,
    )


def _evidence(chunk: DocumentChunk, filename: str, score: float | None = None) -> Evidence:
    return Evidence(
        id=chunk.id,
        document_id=chunk.document_id,
        profile_id=chunk.profile_id,
        filename=filename,
        text=chunk.text,
        source_spans=tuple(SourceSpan(**span) for span in chunk.source_spans),
        score=score,
    )


class PgVectorRetriever:
    def __init__(self, engine: AsyncEngine) -> None:
        self._sessions = async_sessionmaker(engine, expire_on_commit=False)

    async def search(
        self,
        allowed_doc_ids: Sequence[UUID],
        profile_id: UUID,
        query: Sequence[float],
        limit: int,
    ) -> tuple[Evidence, ...]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Retrieval limit must be an integer from 1 to 100")
        vector = _query_vector(query)
        if not allowed_doc_ids:
            return ()
        distance = DocumentChunk.embedding.cosine_distance(vector)
        statement = (
            select(DocumentChunk, Document.filename, distance.label("distance"))
            .options(defer(DocumentChunk.embedding))
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(*_scope(allowed_doc_ids, profile_id))
            .order_by(distance, DocumentChunk.id)
            .limit(limit)
        )
        try:
            async with self._sessions() as session:
                rows = (await session.execute(statement)).all()
                # A zero stored vector has undefined cosine distance. Never
                # expose nonfinite scores as valid ranked evidence.
                return tuple(
                    _evidence(chunk, filename, 1.0 - float(distance_value))
                    for chunk, filename, distance_value in rows
                    if math.isfinite(float(distance_value))
                )
        except SQLAlchemyError:
            raise ProcessingError(
                "RETRIEVAL_UNAVAILABLE",
                "Document evidence is temporarily unavailable.",
                retryable=True,
            ) from None

    async def get_chunks(
        self,
        allowed_doc_ids: Sequence[UUID],
        profile_id: UUID,
        source_ids: Sequence[UUID],
    ) -> tuple[Evidence, ...]:
        """Resolve source IDs in first-requested order, ignoring unknown/hidden IDs."""
        if not allowed_doc_ids or not source_ids:
            return ()
        requested = tuple(dict.fromkeys(source_ids))
        statement = (
            select(DocumentChunk, Document.filename)
            .options(defer(DocumentChunk.embedding))
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(*_scope(allowed_doc_ids, profile_id), DocumentChunk.id.in_(requested))
        )
        try:
            async with self._sessions() as session:
                rows = (await session.execute(statement)).all()
                evidence = {chunk.id: _evidence(chunk, filename) for chunk, filename in rows}
                return tuple(
                    evidence[identifier] for identifier in requested if identifier in evidence
                )
        except SQLAlchemyError:
            raise ProcessingError(
                "RETRIEVAL_UNAVAILABLE",
                "Document evidence is temporarily unavailable.",
                retryable=True,
            ) from None
