"""Transactional turn reservations, durable replay and fenced finalization."""

import hashlib
import json
import unicodedata
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import tiktoken
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.models import (
    ChatMessage,
    ChatSession,
    ChatSessionDocument,
    ChatTurn,
    Document,
    IndexProfile,
)
from app.db.profiles import profile_values
from app.domain.conversation import ConversationError, Replay, TurnReservation
from app.domain.documents import AIFailureDetails
from app.prompts.chat import ANSWER_PROMPT_VERSION, CHAT_PIPELINE_VERSION, PLANNER_PROMPT_VERSION
from app.services.analysis_evidence import QUOTE_CATALOG_VERSION
from app.services.chat_review import CHAT_REVIEW_PROMPT_VERSION
from app.services.evidence_review import PROJECTION_VERSION


def utcnow() -> datetime:
    return datetime.now(UTC)


def _tokens(value: Any) -> int:
    return len(
        tiktoken.get_encoding("o200k_base").encode(
            json.dumps(value, ensure_ascii=False), disallowed_special=()
        )
    )


def _json_copy(value: dict[str, Any]) -> dict[str, Any]:
    try:
        return dict(json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False)))
    except (TypeError, ValueError, OverflowError):
        raise ConversationError(
            422, "INVALID_CHAT_CONTEXT", "Conversation data must be finite JSON."
        ) from None


def _fingerprint(question: str) -> str:
    if not isinstance(question, str):
        raise ConversationError(422, "INVALID_QUESTION", "Question must contain text.")
    try:
        normalized = unicodedata.normalize("NFC", question).strip()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    except UnicodeError:
        raise ConversationError(
            422, "INVALID_QUESTION", "Question contains invalid Unicode."
        ) from None


def _saved(turn: ChatTurn) -> Replay:
    if turn.response_status is None or turn.response_body is None:
        raise ConversationError(
            503, "CHAT_REPLAY_UNAVAILABLE", "The saved turn response is unavailable."
        )
    return Replay(turn.response_status, _json_copy(turn.response_body))


def _error_body(request_id: str, code: str, message: str, turn_id: UUID) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
            "details": {"turn_id": str(turn_id)},
        }
    }


class TurnStore:
    def __init__(self, engine: AsyncEngine, settings: Settings) -> None:
        self._sessions = async_sessionmaker(engine, expire_on_commit=False)
        self.settings = settings

    async def reserve(
        self,
        session_id: UUID,
        client_request_id: UUID,
        question: str,
        request_id: str,
    ) -> TurnReservation | Replay:
        fingerprint = _fingerprint(question)
        async with self._sessions() as database, database.begin():
            conversation = await database.get(ChatSession, session_id, with_for_update=True)
            if conversation is None:
                raise ConversationError(404, "CHAT_SESSION_NOT_FOUND", "Conversation not found.")
            existing = await database.scalar(
                select(ChatTurn)
                .where(
                    ChatTurn.session_id == session_id,
                    ChatTurn.client_request_id == client_request_id,
                )
                .with_for_update()
            )
            if existing is not None:
                if existing.question_fingerprint != fingerprint:
                    raise ConversationError(
                        409,
                        "IDEMPOTENCY_CONFLICT",
                        "Request ID was used for a different question.",
                        {"turn_id": str(existing.id)},
                    )
                if existing.status in {"COMPLETED", "FAILED"}:
                    return _saved(existing)
                raise ConversationError(
                    409,
                    "CHAT_IN_PROGRESS",
                    "This question is still in progress.",
                    {"turn_id": str(existing.id)},
                )

            # Replay precedes admission limits, readiness and the busy check.
            if (
                not question.strip()
                or len(question) > self.settings.question_max_chars
                or len(tiktoken.get_encoding("o200k_base").encode(question, disallowed_special=()))
                > self.settings.question_max_tokens
            ):
                raise ConversationError(
                    422,
                    "INVALID_QUESTION",
                    "Question is empty or exceeds the configured character/token limit.",
                )
            if conversation.active_turn_id is not None:
                raise ConversationError(
                    409,
                    "CHAT_BUSY",
                    "Another question is active in this conversation.",
                    {"turn_id": str(conversation.active_turn_id)},
                )
            document_ids = (
                await database.scalars(
                    select(ChatSessionDocument.document_id).where(
                        ChatSessionDocument.session_id == session_id,
                    )
                )
            ).all()
            if len(document_ids) != 1:
                raise ConversationError(
                    409, "CHAT_SCOPE_INVALID", "Conversation requires exactly one document."
                )
            document = await database.get(Document, document_ids[0])
            if document is None:
                raise ConversationError(
                    404, "DOCUMENT_NOT_FOUND", "Conversation document not found."
                )
            if document.status != "COMPLETED":
                raise ConversationError(
                    409,
                    "DOCUMENT_NOT_READY",
                    "Document processing must complete before asking questions.",
                )
            profile = await database.get(IndexProfile, document.profile_id)
            if (
                profile is None
                or profile.dimensions != 1536
                or profile.embedding_provider != "openai"
                or profile.embedding_model != self.settings.openai_embedding_model
                or profile.fingerprint != profile_values(self.settings)["fingerprint"]
            ):
                raise ConversationError(
                    409,
                    "INDEX_PROFILE_MISMATCH",
                    "Document index profile differs from the configured embedding recipe.",
                )

            now = utcnow()
            turn_id, token, user_id, assistant_id = (uuid4() for _ in range(4))
            turn_number = conversation.next_turn_number
            deadline = now + timedelta(seconds=self.settings.chat_timeout_seconds)
            expiry = now + timedelta(seconds=120)
            turn = ChatTurn(
                id=turn_id,
                session_id=session_id,
                turn_number=turn_number,
                client_request_id=client_request_id,
                question_fingerprint=fingerprint,
                status="PENDING",
                claim_token=token,
                created_at=now,
                deadline_at=deadline,
                expires_at=expiry,
                ai_calls_reserved=0,
                context_snapshot={"request_id": request_id},
                versions={
                    "pipeline": CHAT_PIPELINE_VERSION,
                    "chat_model": self.settings.openai_chat_model,
                    "embedding_model": self.settings.openai_embedding_model,
                    "planner_prompt": PLANNER_PROMPT_VERSION,
                    "answer_prompt": ANSWER_PROMPT_VERSION,
                    "quote_catalog": QUOTE_CATALOG_VERSION,
                    "evidence_review": CHAT_REVIEW_PROMPT_VERSION,
                    "evidence_projection": PROJECTION_VERSION,
                    "retrieval": "exact-cosine-v1",
                    "configuration": self.settings.chat_configuration,
                },
                timings={},
            )
            database.add(turn)
            await database.flush()  # The nullable session pointer references an existing turn.
            database.add_all(
                [
                    ChatMessage(
                        id=user_id,
                        session_id=session_id,
                        turn_id=turn_id,
                        sequence=2 * turn_number - 1,
                        role="user",
                        content=question,
                        status="COMPLETED",
                        citations=[],
                        created_at=now,
                    ),
                    ChatMessage(
                        id=assistant_id,
                        session_id=session_id,
                        turn_id=turn_id,
                        sequence=2 * turn_number,
                        role="assistant",
                        content=None,
                        status="PENDING",
                        citations=[],
                        created_at=now,
                    ),
                ]
            )
            conversation.next_turn_number += 1
            conversation.active_turn_id, conversation.updated_at = turn_id, now
            return TurnReservation(
                turn_id,
                session_id,
                token,
                user_id,
                assistant_id,
                document.id,
                profile.id,
                question,
                deadline,
                expiry,
                request_id,
                turn_number,
            )

    @staticmethod
    async def _locked_rows(
        database: AsyncSession,
        session_id: UUID,
        turn_id: UUID,
    ) -> tuple[ChatSession, ChatTurn, datetime]:
        # All writers, including recovery, acquire session before turn.
        conversation = await database.get(ChatSession, session_id, with_for_update=True)
        if conversation is None:
            raise ConversationError(404, "CHAT_SESSION_NOT_FOUND", "Conversation not found.")
        turn = await database.scalar(
            select(ChatTurn)
            .where(
                ChatTurn.id == turn_id,
                ChatTurn.session_id == session_id,
            )
            .with_for_update()
        )
        if turn is None:
            raise ConversationError(404, "CHAT_TURN_NOT_FOUND", "Question turn not found.")
        return conversation, turn, utcnow()

    @staticmethod
    def _guard(
        conversation: ChatSession,
        turn: ChatTurn,
        reservation: TurnReservation,
        now: datetime,
        *,
        allow_after_deadline: bool = False,
    ) -> None:
        if (
            turn.claim_token != reservation.token
            or turn.status != "PENDING"
            or conversation.active_turn_id != turn.id
            or turn.expires_at <= now
        ):
            raise ConversationError(
                409,
                "CHAT_TURN_STALE",
                "Question ownership expired or was replaced.",
                {"turn_id": str(turn.id)},
            )
        if not allow_after_deadline and turn.deadline_at <= now:
            raise ConversationError(
                504, "CHAT_TIMEOUT", "Question exceeded its deadline.", {"turn_id": str(turn.id)}
            )

    async def before_call(self, reservation: TurnReservation) -> float:
        async with self._sessions() as database, database.begin():
            conversation, turn, now = await self._locked_rows(
                database, reservation.session_id, reservation.turn_id
            )
            self._guard(conversation, turn, reservation, now)
            # Normal factual path: planner + embedding + draft + audit. One
            # extra reservation permits the existing bounded draft repair.
            if turn.ai_calls_reserved >= 5:
                raise ConversationError(
                    409, "CHAT_CALL_BUDGET_EXHAUSTED", "Question exhausted its AI-call budget."
                )
            turn.ai_calls_reserved += 1
            return (turn.deadline_at - now).total_seconds()

    async def history(self, reservation: TurnReservation) -> list[dict[str, str]]:
        async with self._sessions() as database, database.begin():
            conversation, turn, now = await self._locked_rows(
                database, reservation.session_id, reservation.turn_id
            )
            self._guard(conversation, turn, reservation, now)
            ids = (
                await database.scalars(
                    select(ChatTurn.id)
                    .where(
                        ChatTurn.session_id == reservation.session_id,
                        ChatTurn.status == "COMPLETED",
                        ChatTurn.turn_number < reservation.turn_number,
                    )
                    .order_by(ChatTurn.turn_number.desc())
                    .limit(self.settings.chat_history_max_turns)
                )
            ).all()
            messages = (
                await database.scalars(
                    select(ChatMessage)
                    .where(
                        ChatMessage.session_id == reservation.session_id,
                        ChatMessage.turn_id.in_(ids),
                        ChatMessage.status == "COMPLETED",
                    )
                    .order_by(ChatMessage.sequence)
                )
            ).all()
            history = [
                {"role": message.role, "content": message.content or ""} for message in messages
            ]
        budget = self.settings.chat_history_max_tokens
        while len(history) > 1 and _tokens(history) > budget:
            history.pop(0)
        if history and _tokens(history) > budget:
            original = history[0]["content"]
            history[0]["content"] = ""
            if _tokens(history) > budget:
                raise ConversationError(
                    503,
                    "CHAT_HISTORY_BUDGET_TOO_SMALL",
                    "History token budget cannot fit a message.",
                )
            low, high = 0, len(original)
            while low < high:
                middle = (low + high + 1) // 2
                history[0]["content"] = original[:middle]
                if _tokens(history) <= budget:
                    low = middle
                else:
                    high = middle - 1
            history[0]["content"] = original[:low]
        return history

    async def save_context(
        self,
        reservation: TurnReservation,
        *,
        query: str,
        intent: str,
        evidence_scope: str,
        context: dict[str, Any],
        timings: dict[str, Any],
    ) -> None:
        context = _json_copy(context)
        timings = _json_copy(timings)
        # Provider input and persisted audit are different bounded records. The
        # audit is never fed back as conversation history or retrieval evidence.
        prompt_context = {key: value for key, value in context.items() if key != "evidence_review"}
        review = context.get("evidence_review")
        review_budget = (
            self.settings.retrieval_max_tokens
            + self.settings.chat_max_output_tokens
            + self.settings.evidence_review_max_output_tokens
            + self.settings.INSTRUCTION_RESERVE_TOKENS
        )
        if (
            not isinstance(query, str)
            or not query.strip()
            or len(query) > 4000
            or intent not in {"factual", "overview", "exhaustive"}
            or evidence_scope not in {"retrieved_excerpts", "document_summary"}
            or _tokens(prompt_context) > self.settings.chat_prompt_max_tokens
            or (
                review is not None
                and (not isinstance(review, dict) or _tokens(review) > review_budget)
            )
        ):
            raise ConversationError(
                422, "INVALID_CHAT_CONTEXT", "Conversation context exceeds its bounded contract."
            )
        async with self._sessions() as database, database.begin():
            conversation, turn, now = await self._locked_rows(
                database, reservation.session_id, reservation.turn_id
            )
            self._guard(conversation, turn, reservation, now)
            turn.retrieval_query, turn.planner_intent, turn.evidence_scope = (
                query,
                intent,
                evidence_scope,
            )
            turn.context_snapshot = {**context, "request_id": reservation.request_id}
            turn.timings = {**turn.timings, **timings}

    @staticmethod
    async def _assistant(database: AsyncSession, turn: ChatTurn) -> ChatMessage:
        message = await database.scalar(
            select(ChatMessage).where(
                ChatMessage.session_id == turn.session_id,
                ChatMessage.turn_id == turn.id,
                ChatMessage.role == "assistant",
            )
        )
        if message is None:
            raise ConversationError(
                503, "CHAT_STATE_INVALID", "Saved assistant message is unavailable."
            )
        return message

    async def complete(
        self,
        reservation: TurnReservation,
        body: dict[str, Any],
        *,
        timings: dict[str, Any],
    ) -> Replay:
        body, timings = _json_copy(body), _json_copy(timings)
        if (
            not isinstance(body.get("answer"), str)
            or body.get("evidence_status") not in {"answered", "partial", "insufficient"}
            or not isinstance(body.get("citations"), list)
        ):
            raise ConversationError(
                422, "INVALID_CHAT_RESULT", "Conversation result has invalid fields."
            )
        async with self._sessions() as database, database.begin():
            conversation, turn, now = await self._locked_rows(
                database, reservation.session_id, reservation.turn_id
            )
            if turn.claim_token == reservation.token and turn.status in {"COMPLETED", "FAILED"}:
                return _saved(turn)
            self._guard(conversation, turn, reservation, now)
            assistant = await self._assistant(database, turn)
            assistant.content, assistant.citations = body["answer"], body["citations"]
            assistant.status = "COMPLETED"
            turn.status, turn.ended_at = "COMPLETED", now
            turn.response_status, turn.response_body = 200, body
            turn.evidence_status = body["evidence_status"]
            turn.timings = {**turn.timings, **timings}
            conversation.active_turn_id, conversation.updated_at = None, now
            return Replay(200, body)

    async def fail(
        self,
        reservation: TurnReservation,
        *,
        code: str,
        message: str,
        status: int,
        diagnostics: AIFailureDetails | None = None,
    ) -> Replay:
        async with self._sessions() as database, database.begin():
            conversation, turn, now = await self._locked_rows(
                database, reservation.session_id, reservation.turn_id
            )
            if turn.claim_token == reservation.token and turn.status in {"COMPLETED", "FAILED"}:
                return _saved(turn)
            self._guard(conversation, turn, reservation, now, allow_after_deadline=True)
            return await self._failure(
                database,
                conversation,
                turn,
                now,
                reservation.request_id,
                code,
                message,
                status,
                diagnostics,
            )

    async def _failure(
        self,
        database: AsyncSession,
        conversation: ChatSession,
        turn: ChatTurn,
        now: datetime,
        request_id: str,
        code: str,
        message: str,
        status: int,
        diagnostics: AIFailureDetails | None = None,
    ) -> Replay:
        body = _error_body(request_id, code, message, turn.id)
        if diagnostics is not None:
            body["error"]["details"].update(
                {key: value for key, value in diagnostics.as_dict().items() if value is not None}
            )
        assistant = await self._assistant(database, turn)
        assistant.content, assistant.citations, assistant.status = message, [], "FAILED"
        turn.status, turn.ended_at, turn.error = "FAILED", now, body["error"]
        turn.response_status, turn.response_body = status, body
        if conversation.active_turn_id == turn.id:
            conversation.active_turn_id, conversation.updated_at = None, now
        return Replay(status, body)

    async def expire_pending(self) -> int:
        count = 0
        async with self._sessions() as database, database.begin():
            # Lock sessions before the limit so one busy owner cannot starve
            # other expired turns; every writer keeps session-before-turn order.
            candidates = (
                await database.execute(
                    select(ChatTurn.session_id, ChatTurn.id)
                    .join(ChatSession, ChatSession.id == ChatTurn.session_id)
                    .where(ChatTurn.status == "PENDING", ChatTurn.expires_at <= utcnow())
                    .order_by(ChatTurn.expires_at, ChatTurn.id)
                    .with_for_update(skip_locked=True, of=ChatSession)
                    .limit(100)
                )
            ).all()
            for session_id, turn_id in candidates:
                conversation, turn, now = await self._locked_rows(database, session_id, turn_id)
                if turn.status != "PENDING" or turn.expires_at > now:
                    continue
                request_id = turn.context_snapshot.get("request_id")
                if not isinstance(request_id, str):
                    request_id = str(uuid4())
                await self._failure(
                    database,
                    conversation,
                    turn,
                    now,
                    request_id,
                    "CHAT_TIMEOUT",
                    "Question expired before completion.",
                    504,
                )
                count += 1
        return count
