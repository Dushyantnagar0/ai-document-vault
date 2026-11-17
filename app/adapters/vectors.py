"""Selected vector-publication adapter, participating in the caller's transaction."""

from collections.abc import Sequence
from dataclasses import asdict
from typing import Protocol
from uuid import UUID

from sqlalchemy.orm import Session

from app.db.models import DocumentChunk
from app.domain.documents import Chunk, ProcessingError
from app.domain.vectors import float32_vector


class VectorPublisher(Protocol):
    def publish(
        self,
        document_id: UUID,
        attempt_id: UUID,
        profile_id: UUID,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None: ...


class PgVectorPublisher:
    """Never commits independently; chunks/insights/status become visible together.

    A future external index needs the documented cross-store publication migration,
    rather than pretending it can join a PostgreSQL transaction.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def publish(
        self,
        document_id: UUID,
        attempt_id: UUID,
        profile_id: UUID,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        try:
            if len(chunks) != len(vectors):
                raise ValueError
            validated = [float32_vector(vector, 1536) for vector in vectors]
        except (TypeError, ValueError):
            raise ProcessingError(
                "INVALID_VECTORS",
                "Vectors must cover every chunk with finite nonzero float32 "
                "1536-dimensional values",
            ) from None
        self.session.add_all(
            [
                DocumentChunk(
                    id=chunk.id,
                    document_id=document_id,
                    attempt_id=attempt_id,
                    profile_id=profile_id,
                    ordinal=chunk.ordinal,
                    text=chunk.text,
                    token_count=chunk.token_count,
                    source_spans=[asdict(span) for span in chunk.source_spans],
                    embedding=vector,
                )
                for chunk, vector in zip(chunks, validated, strict=True)
            ]
        )
