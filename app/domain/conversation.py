"""Provider/database independent evidence supplied to grounded conversation."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, TypeVar
from uuid import UUID

from pydantic import BaseModel

from app.domain.documents import EmbeddingBatch, Generated, SourceSpan

T = TypeVar("T", bound=BaseModel)


class ConversationError(Exception):
    def __init__(
        self, status: int, code: str, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message
        self.details = details or {}


@dataclass(frozen=True)
class TurnReservation:
    turn_id: UUID
    session_id: UUID
    token: UUID
    user_message_id: UUID
    assistant_message_id: UUID
    document_id: UUID
    profile_id: UUID
    question: str
    deadline_at: datetime
    expires_at: datetime
    request_id: str
    turn_number: int


@dataclass(frozen=True)
class Replay:
    status: int
    body: dict[str, Any]


@dataclass(frozen=True)
class Evidence:
    id: UUID
    document_id: UUID
    profile_id: UUID
    filename: str
    text: str
    source_spans: tuple[SourceSpan, ...]
    text_offset: int = 0
    score: float | None = None


class AsyncEmbeddingProvider(Protocol):
    async def embed_many(
        self, texts: Sequence[str], *, timeout_seconds: float
    ) -> EmbeddingBatch: ...


class AsyncGenerationProvider(Protocol):
    async def generate(
        self,
        *,
        schema: type[T],
        system: str,
        content: str,
        max_output_tokens: int,
        timeout_seconds: float,
    ) -> Generated[T]: ...
