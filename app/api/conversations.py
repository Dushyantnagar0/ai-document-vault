"""Document-scoped sessions, complete HTTP answers and persistent ordered history."""

import asyncio
from datetime import datetime
from typing import Annotated, Any, Literal, cast
from uuid import UUID, uuid4

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.orm import defer

from app.api.documents import decode_cursor
from app.db.models import ChatMessage, ChatSession, ChatSessionDocument, ChatTurn, Document
from app.domain.conversation import ConversationError, Replay
from app.services.chat import ChatProcessor
from app.services.conversations import TurnStore
from app.services.ingestion import utcnow

router = APIRouter(prefix="/api/v1/chat", tags=["conversation"])


class CreateSession(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: UUID
    title: str | None = Field(default=None, min_length=1, max_length=120)


class AskQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(
        min_length=1, description="Subject to the installation's character and token limits"
    )
    client_request_id: UUID = Field(
        description="Reuse for transport retries of this exact question"
    )


class SessionView(BaseModel):
    id: UUID
    title: str
    document_ids: list[UUID]
    created_at: datetime
    updated_at: datetime


class SessionsPage(BaseModel):
    items: list[SessionView]
    next_cursor: str | None


class CitationView(BaseModel):
    source_id: UUID
    document_id: UUID
    filename: str
    locators: list[dict[str, str | int]]
    excerpt: str = Field(max_length=1200)


class MessageView(BaseModel):
    id: UUID
    turn_id: UUID
    sequence: int
    role: Literal["user", "assistant"]
    content: str | None
    status: Literal["PENDING", "COMPLETED", "FAILED"]
    turn_status: Literal["PENDING", "COMPLETED", "FAILED"]
    evidence_status: Literal["answered", "partial", "insufficient"] | None = None
    evidence_scope: Literal["retrieved_excerpts", "document_summary"] | None = None
    citations: list[CitationView] = Field(max_length=8)
    created_at: datetime


class MessagesPage(BaseModel):
    items: list[MessageView]
    next_cursor: str | None


class ChatAnswer(BaseModel):
    turn_id: UUID
    message_id: UUID
    status: Literal["COMPLETED"]
    answer: str
    evidence_status: Literal["answered", "partial", "insufficient"]
    citations: list[CitationView] = Field(max_length=8)
    evidence_scope: Literal["retrieved_excerpts", "document_summary"]
    extraction_coverage: str
    warnings: list[dict[str, Any]]
    model: str
    prompt_version: str
    planner_prompt_version: str


@router.post("/sessions", status_code=201, response_model=SessionView)
async def create_session(request: Request, data: CreateSession) -> SessionView:
    factory = async_sessionmaker(request.app.state.engine, expire_on_commit=False)
    async with factory() as session, session.begin():
        document = await session.get(
            Document, data.document_id, options=[defer(Document.extracted_text)]
        )
        if document is None:
            raise ConversationError(404, "DOCUMENT_NOT_FOUND", "Document not found")
        if document.status != "COMPLETED":
            raise ConversationError(
                409,
                "DOCUMENT_NOT_READY",
                "Wait for document processing to complete before starting a conversation",
            )
        title = (data.title if data.title is not None else document.filename[:120]).strip()
        if not title:
            raise ConversationError(422, "INVALID_TITLE", "Session title must contain text")
        now = utcnow()
        chat = ChatSession(
            id=uuid4(), title=title, created_at=now, updated_at=now, next_turn_number=1
        )
        session.add(chat)
        await session.flush()
        session.add(ChatSessionDocument(session_id=chat.id, document_id=document.id))
    return SessionView(
        id=chat.id,
        title=chat.title,
        document_ids=[data.document_id],
        created_at=chat.created_at,
        updated_at=chat.updated_at,
    )


@router.get("/sessions", response_model=SessionsPage)
async def list_sessions(
    request: Request,
    document_id: UUID | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: Annotated[str | None, Query(max_length=250)] = None,
) -> SessionsPage:
    import base64
    import json

    query = (
        select(ChatSession, ChatSessionDocument.document_id)
        .join(ChatSessionDocument, ChatSessionDocument.session_id == ChatSession.id)
        .order_by(ChatSession.updated_at.desc(), ChatSession.id.desc())
        .limit(limit + 1)
    )
    if document_id:
        query = query.where(ChatSessionDocument.document_id == document_id)
    if cursor:
        timestamp, identifier = decode_cursor(cursor)
        query = query.where(
            tuple_(ChatSession.updated_at, ChatSession.id) < (timestamp, identifier)
        )
    async with async_sessionmaker(request.app.state.engine)() as session:
        rows = (await session.execute(query)).all()
    page = rows[:limit]
    next_cursor = None
    if len(rows) > limit:
        last = page[-1][0]
        next_cursor = base64.urlsafe_b64encode(
            json.dumps([last.updated_at.isoformat(), str(last.id)]).encode()
        ).decode()
    return SessionsPage(
        items=[
            SessionView(
                id=chat.id,
                title=chat.title,
                document_ids=[member],
                created_at=chat.created_at,
                updated_at=chat.updated_at,
            )
            for chat, member in page
        ],
        next_cursor=next_cursor,
    )


@router.post(
    "/sessions/{session_id}/messages",
    response_model=ChatAnswer,
    responses={
        409: {"description": "Not ready, busy, in progress or mismatched replay"},
        502: {"description": "Saved provider/grounding failure"},
        504: {"description": "Saved question deadline failure"},
    },
)
async def ask(request: Request, session_id: UUID, data: AskQuestion) -> JSONResponse:
    store = cast(TurnStore, request.app.state.turn_store)
    reservation = await store.reserve(
        session_id, data.client_request_id, data.question, request.state.request_id
    )
    if isinstance(reservation, Replay):
        return JSONResponse(status_code=reservation.status, content=reservation.body)
    processor = cast(ChatProcessor, request.app.state.chat_processor)
    task = asyncio.create_task(processor.run(reservation))
    tasks: dict[UUID, asyncio.Task[Replay]] = request.app.state.turn_tasks
    tasks[reservation.turn_id] = task

    def finished(completed: asyncio.Task[Replay]) -> None:
        tasks.pop(reservation.turn_id, None)
        if not completed.cancelled():
            # Consume dependency failure; durable lease recovery remains authoritative.
            completed.exception()

    task.add_done_callback(finished)
    result = await asyncio.shield(task)
    return JSONResponse(status_code=result.status, content=result.body)


@router.get("/sessions/{session_id}/messages", response_model=MessagesPage)
async def history(
    request: Request,
    session_id: UUID,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: Annotated[int, Query(ge=0, le=2147483647)] = 0,
) -> dict[str, Any]:
    async with async_sessionmaker(request.app.state.engine)() as session:
        if await session.get(ChatSession, session_id) is None:
            raise ConversationError(404, "SESSION_NOT_FOUND", "Conversation not found")
        rows = (
            await session.execute(
                select(
                    ChatMessage, ChatTurn.status, ChatTurn.evidence_status, ChatTurn.evidence_scope
                )
                .join(ChatTurn, ChatMessage.turn_id == ChatTurn.id)
                .where(ChatMessage.session_id == session_id, ChatMessage.sequence > cursor)
                .order_by(ChatMessage.sequence)
                .limit(limit + 1)
            )
        ).all()
        items = [
            {
                "id": str(message.id),
                "turn_id": str(message.turn_id),
                "sequence": message.sequence,
                "role": message.role,
                "content": message.content,
                "status": message.status,
                "turn_status": turn_status,
                "evidence_status": evidence_status if message.role == "assistant" else None,
                "evidence_scope": evidence_scope if message.role == "assistant" else None,
                "citations": message.citations,
                "created_at": message.created_at.isoformat(),
            }
            for message, turn_status, evidence_status, evidence_scope in rows[:limit]
        ]
    return {
        "items": items,
        "next_cursor": str(items[-1]["sequence"]) if len(rows) > limit else None,
    }
