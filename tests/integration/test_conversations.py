"""Turn persistence/concurrency against actual private-schema PostgreSQL tables."""

import asyncio
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pytest
import tiktoken
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.orm import Session
from test_processing_claims import ClaimDatabase
from test_processing_claims import claim_database as claim_database
from test_processing_claims import new_document as new_document
from test_retrieval import Published
from test_retrieval import publish as publish
from test_retrieval import retrieval_engine as retrieval_engine

from app.db.models import (
    ChatMessage,
    ChatSession,
    ChatSessionDocument,
    ChatTurn,
    Document,
    DocumentChunk,
    IndexProfile,
)
from app.db.profiles import profile_values
from app.domain.conversation import ConversationError, Replay, TurnReservation
from app.domain.documents import AIFailureDetails
from app.main import create_app
from app.services import conversations
from app.services.conversations import TurnStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.enable_socket,
    pytest.mark.skipif(
        os.environ.get("VAULT_INTEGRATION_TESTS") != "1", reason="requires isolated Compose stack"
    ),
]


@pytest.fixture
def new_session(
    claim_database: ClaimDatabase,
    publish: Callable[[Sequence[Sequence[float]]], Published],
) -> Callable[[], UUID]:
    def create() -> UUID:
        document = publish([[1.0] + [0.0] * 1535])
        identifier = uuid4()
        with Session(claim_database.engine) as database, database.begin():
            database.add(
                ChatSession(id=identifier, title="Persistence fixture", next_turn_number=1)
            )
            database.flush()
            database.add(
                ChatSessionDocument(session_id=identifier, document_id=document.document_id)
            )
        return identifier

    return create


@pytest.fixture
async def store(retrieval_engine: AsyncEngine, claim_database: ClaimDatabase) -> TurnStore:
    return TurnStore(retrieval_engine, claim_database.settings)


def reservation(value: TurnReservation | Replay) -> TurnReservation:
    assert isinstance(value, TurnReservation)
    return value


def answer(text: str = "No supporting evidence was found for this question.") -> dict[str, object]:
    return {"answer": text, "evidence_status": "insufficient", "citations": []}


@pytest.mark.parametrize("evidence_status", ["answered", "partial", "insufficient"])
async def test_history_retains_assistant_evidence_status_and_scope_across_clients(
    store: TurnStore,
    retrieval_engine: AsyncEngine,
    new_session: Callable[[], UUID],
    evidence_status: str,
) -> None:
    identifier = new_session()
    accepted = reservation(
        await store.reserve(identifier, uuid4(), "What is stated?", str(uuid4()))
    )
    await store.save_context(
        accepted,
        query="stated facts",
        intent="exhaustive",
        evidence_scope="retrieved_excerpts",
        context={},
        timings={},
    )
    # Seed store metadata directly: this test checks persisted history presentation,
    # while generation/source validation has independent unit and HTTP flow tests.
    await store.complete(
        accepted,
        {
            "answer": "History presentation fixture.",
            "evidence_status": evidence_status,
            "citations": [],
        },
        timings={},
    )
    for _ in range(2):
        application = create_app()
        application.state.engine = retrieval_engine
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://history-test"
        ) as client:
            response = await client.get(f"/api/v1/chat/sessions/{identifier}/messages")
        assert response.status_code == 200
        messages = response.json()["items"]
        assert messages[0]["evidence_status"] is None and messages[0]["evidence_scope"] is None
        assert messages[1]["evidence_status"] == evidence_status
        assert messages[1]["evidence_scope"] == "retrieved_excerpts"


@pytest.mark.parametrize("same_key", [True, False])
async def test_concurrent_reservation_has_one_turn_and_two_ordered_messages(
    store: TurnStore,
    new_session: Callable[[], UUID],
    claim_database: ClaimDatabase,
    same_key: bool,
) -> None:
    identifier, key = new_session(), uuid4()
    results = await asyncio.gather(
        store.reserve(identifier, key, "What is stated?", str(uuid4())),
        store.reserve(identifier, key if same_key else uuid4(), "What is stated?", str(uuid4())),
        return_exceptions=True,
    )
    accepted = [result for result in results if isinstance(result, TurnReservation)]
    rejected = [result for result in results if isinstance(result, ConversationError)]
    assert len(accepted) == len(rejected) == 1
    assert rejected[0].status == 409
    assert rejected[0].code == ("CHAT_IN_PROGRESS" if same_key else "CHAT_BUSY")
    assert rejected[0].details["turn_id"] == str(accepted[0].turn_id)
    with Session(claim_database.engine) as database:
        session = database.get(ChatSession, identifier)
        assert session is not None
        assert session.active_turn_id == accepted[0].turn_id
        assert session.next_turn_number == 2
        turns = database.scalars(select(ChatTurn).where(ChatTurn.session_id == identifier)).all()
        messages = database.scalars(
            select(ChatMessage)
            .where(ChatMessage.session_id == identifier)
            .order_by(ChatMessage.sequence)
        ).all()
        assert len(turns) == 1
        assert [(message.sequence, message.role, message.status) for message in messages] == [
            (1, "user", "COMPLETED"),
            (2, "assistant", "PENDING"),
        ]


async def test_replay_precedes_busy_and_new_question_limits_and_preserves_original(
    store: TurnStore,
    retrieval_engine: AsyncEngine,
    new_session: Callable[[], UUID],
    claim_database: ClaimDatabase,
) -> None:
    identifier, key, request_id = new_session(), uuid4(), str(uuid4())
    original = "  Café budget?  "
    first = reservation(await store.reserve(identifier, key, original, request_id))
    body = answer()
    saved = await store.complete(first, body, timings={"answer_ms": 1})
    active = reservation(await store.reserve(identifier, uuid4(), "Next question?", str(uuid4())))
    limited = TurnStore(
        retrieval_engine,
        claim_database.settings.model_copy(
            update={"question_max_chars": 1, "question_max_tokens": 1}
        ),
    )
    replay = await limited.reserve(identifier, key, "Cafe\u0301 budget?", str(uuid4()))
    assert replay == saved == Replay(200, body)
    with pytest.raises(ConversationError) as error:
        await store.reserve(identifier, key, "Different question", str(uuid4()))
    assert error.value.code == "IDEMPOTENCY_CONFLICT"
    with Session(claim_database.engine) as database:
        message = database.get(ChatMessage, first.user_message_id)
        session = database.get(ChatSession, identifier)
        assert message is not None and message.content == original
        assert session is not None and session.active_turn_id == active.turn_id


async def test_context_completion_and_saved_versions_are_atomic(
    store: TurnStore,
    new_session: Callable[[], UUID],
    claim_database: ClaimDatabase,
) -> None:
    first = reservation(
        await store.reserve(new_session(), uuid4(), "What is stated?", "original-request")
    )
    assert 0 < await store.before_call(first) <= 90
    with Session(claim_database.engine) as database:
        chunk = database.scalar(
            select(DocumentChunk).where(DocumentChunk.document_id == first.document_id)
        )
        document = database.get(Document, first.document_id)
        assert chunk is not None and document is not None
        citation = {
            "source_id": str(chunk.id),
            "document_id": str(document.id),
            "filename": document.filename,
            "locators": [{"page": 1}],
            "excerpt": chunk.text,
        }
        body = {"answer": chunk.text, "evidence_status": "answered", "citations": [citation]}
    await store.save_context(
        first,
        query="stated facts",
        intent="factual",
        evidence_scope="retrieved_excerpts",
        context={"history": [], "sources": [citation], "request_id": "replace-me"},
        timings={"retrieval_ms": 2},
    )
    assert await store.complete(first, body, timings={"answer_ms": 3}) == Replay(200, body)
    with Session(claim_database.engine) as database:
        turn = database.get(ChatTurn, first.turn_id)
        message = database.get(ChatMessage, first.assistant_message_id)
        session = database.get(ChatSession, first.session_id)
        assert turn is not None and message is not None and session is not None
        assert turn.status == message.status == "COMPLETED"
        assert turn.ai_calls_reserved == 1
        assert turn.retrieval_query == "stated facts"
        assert turn.context_snapshot["request_id"] == "original-request"
        assert turn.context_snapshot["sources"] == [citation]
        assert turn.response_body == body and turn.response_status == 200
        assert turn.timings == {"retrieval_ms": 2, "answer_ms": 3}
        assert turn.versions["chat_model"] == claim_database.settings.openai_chat_model
        assert "openai_api_key" not in turn.versions
        assert message.content == body["answer"] and message.citations == [citation]
        assert session.active_turn_id is None


async def test_call_budget_is_persisted_and_cannot_be_exceeded(
    store: TurnStore,
    new_session: Callable[[], UUID],
    claim_database: ClaimDatabase,
) -> None:
    first = reservation(await store.reserve(new_session(), uuid4(), "Question", str(uuid4())))
    for _ in range(5):
        assert await store.before_call(first) > 0
    with pytest.raises(ConversationError) as error:
        await store.before_call(first)
    assert error.value.code == "CHAT_CALL_BUDGET_EXHAUSTED"
    with Session(claim_database.engine) as database:
        turn = database.get(ChatTurn, first.turn_id)
        assert turn is not None and turn.ai_calls_reserved == 5


async def test_timeout_failure_can_persist_after_deadline_and_replays(
    store: TurnStore,
    new_session: Callable[[], UUID],
    monkeypatch: pytest.MonkeyPatch,
    claim_database: ClaimDatabase,
) -> None:
    key = uuid4()
    first = reservation(await store.reserve(new_session(), key, "Question", "stable-request"))
    monkeypatch.setattr(conversations, "utcnow", lambda: first.deadline_at + timedelta(seconds=1))
    with pytest.raises(ConversationError) as error:
        await store.before_call(first)
    assert error.value.status == 504
    with pytest.raises(ConversationError) as error:
        await store.complete(first, answer(), timings={})
    assert error.value.status == 504
    failed = await store.fail(first, code="CHAT_TIMEOUT", message="Deadline exceeded.", status=504)
    assert failed.status == 504
    assert failed.body["error"]["request_id"] == "stable-request"
    assert await store.reserve(first.session_id, key, "Question", "new-request") == failed
    with Session(claim_database.engine) as database:
        turn = database.get(ChatTurn, first.turn_id)
        assistant = database.get(ChatMessage, first.assistant_message_id)
        assert turn is not None and assistant is not None
        assert turn.status == assistant.status == "FAILED"
        assert assistant.content == "Deadline exceeded."


@pytest.mark.parametrize("unsafe_metadata", [False, True])
async def test_provider_failure_diagnostics_are_sanitized_persisted_and_replayed(
    store: TurnStore,
    new_session: Callable[[], UUID],
    claim_database: ClaimDatabase,
    unsafe_metadata: bool,
) -> None:
    key = uuid4()
    first = reservation(await store.reserve(new_session(), key, "Question", "stable-request"))
    private = "PRIVATE_RAW_PROVIDER_BODY"
    diagnostics = AIFailureDetails(
        reason=private if unsafe_metadata else "unknown_quote_id",
        input_tokens=42,
        output_tokens=7,
        max_output_tokens=1200,
        finish_reason=private if unsafe_metadata else "stop",
        provider_request_id=private if unsafe_metadata else "req_safe-123",
    )
    failed = await store.fail(
        first,
        code="CHAT_INVALID_OUTPUT",
        message="Chat selected invalid evidence references.",
        status=502,
        diagnostics=diagnostics,
    )
    details = failed.body["error"]["details"]
    assert details == {
        "turn_id": str(first.turn_id),
        "reason": "structured_parse_failure" if unsafe_metadata else "unknown_quote_id",
        "input_tokens": 42,
        "output_tokens": 7,
        "max_output_tokens": 1200,
        **(
            {}
            if unsafe_metadata
            else {"finish_reason": "stop", "provider_request_id": "req_safe-123"}
        ),
    }
    assert private not in json.dumps(failed.body)
    assert await store.reserve(first.session_id, key, "Question", "later-request") == failed
    # A late owner cannot replace the persisted diagnosis with a different result.
    assert await store.fail(first, code="OTHER", message="Late failure.", status=503) == failed
    with Session(claim_database.engine) as database:
        turn = database.get(ChatTurn, first.turn_id)
        assert turn is not None
        assert turn.error == failed.body["error"]
        assert turn.response_body == failed.body
        assert turn.response_status == 502 and turn.status == "FAILED"


async def test_stale_token_cannot_write_or_release_active_turn(
    store: TurnStore,
    new_session: Callable[[], UUID],
    claim_database: ClaimDatabase,
) -> None:
    first = reservation(await store.reserve(new_session(), uuid4(), "Question", str(uuid4())))
    stale = replace(first, token=uuid4())
    operations = [
        store.before_call(stale),
        store.complete(stale, answer(), timings={}),
        store.fail(stale, code="ERROR", message="Not the owner.", status=502),
        store.save_context(
            stale,
            query="Question",
            intent="factual",
            evidence_scope="retrieved_excerpts",
            context={},
            timings={},
        ),
    ]
    for operation in operations:
        with pytest.raises(ConversationError) as error:
            await operation
        assert error.value.code == "CHAT_TURN_STALE"
    with Session(claim_database.engine) as database:
        session = database.get(ChatSession, first.session_id)
        turn = database.get(ChatTurn, first.turn_id)
        assert session is not None and turn is not None
        assert session.active_turn_id == first.turn_id and turn.status == "PENDING"
        assert turn.ai_calls_reserved == 0


async def test_expiry_uses_lease_and_old_owner_cannot_release_new_turn(
    store: TurnStore,
    new_session: Callable[[], UUID],
    monkeypatch: pytest.MonkeyPatch,
    claim_database: ClaimDatabase,
) -> None:
    first = reservation(await store.reserve(new_session(), uuid4(), "Old question", "old-request"))
    clock = SimpleNamespace(now=first.deadline_at + timedelta(seconds=1))
    monkeypatch.setattr(conversations, "utcnow", lambda: clock.now)
    assert await store.expire_pending() == 0
    clock.now = first.expires_at + timedelta(seconds=1)
    # Other tests' abandoned pending turns may also be eligible in this module's
    # private schema; count at least this one, then require an idempotent rescan.
    assert await store.expire_pending() >= 1
    assert await store.expire_pending() == 0
    second = reservation(
        await store.reserve(first.session_id, uuid4(), "New question", str(uuid4()))
    )
    late = await store.complete(first, answer("Late output must not replace failure."), timings={})
    assert late.status == 504
    with Session(claim_database.engine) as database:
        session = database.get(ChatSession, first.session_id)
        old = database.get(ChatTurn, first.turn_id)
        assert session is not None and old is not None
        assert session.active_turn_id == second.turn_id
        assert old.status == "FAILED" and old.response_status == 504


async def test_history_uses_last_four_completed_turns_and_excludes_failures(
    store: TurnStore,
    new_session: Callable[[], UUID],
) -> None:
    identifier = new_session()
    for index in range(6):
        turn = reservation(
            await store.reserve(identifier, uuid4(), f"Question {index}", str(uuid4()))
        )
        if index == 2:
            await store.fail(turn, code="PROVIDER_FAILED", message="Visible failure.", status=502)
        else:
            await store.complete(turn, answer(f"No evidence for question {index}."), timings={})
    current = reservation(await store.reserve(identifier, uuid4(), "Follow-up", str(uuid4())))
    history = await store.history(current)
    assert [message["role"] for message in history] == ["user", "assistant"] * 4
    assert [message["content"] for message in history[::2]] == [
        "Question 1",
        "Question 3",
        "Question 4",
        "Question 5",
    ]
    assert all("Visible failure" not in message["content"] for message in history)


async def test_history_token_budget_clips_recent_assistant_without_losing_budget(
    store: TurnStore,
    retrieval_engine: AsyncEngine,
    new_session: Callable[[], UUID],
    claim_database: ClaimDatabase,
) -> None:
    identifier = new_session()
    previous = reservation(await store.reserve(identifier, uuid4(), "Question", str(uuid4())))
    content = "No supporting evidence. " * 100
    await store.complete(previous, answer(content), timings={})
    current = reservation(await store.reserve(identifier, uuid4(), "Follow-up", str(uuid4())))
    limited = TurnStore(
        retrieval_engine,
        claim_database.settings.model_copy(update={"chat_history_max_tokens": 32}),
    )
    history = await limited.history(current)
    assert len(history) == 1 and history[0]["role"] == "assistant"
    assert 0 < len(history[0]["content"]) < len(content)
    assert content.startswith(history[0]["content"])
    assert (
        len(
            tiktoken.get_encoding("o200k_base").encode(
                json.dumps(history, ensure_ascii=False), disallowed_special=()
            )
        )
        <= 32
    )


@pytest.mark.parametrize("question", ["", "   ", "x" * 4001])
async def test_invalid_new_question_does_not_reserve_a_turn(
    store: TurnStore,
    new_session: Callable[[], UUID],
    claim_database: ClaimDatabase,
    question: str,
) -> None:
    identifier = new_session()
    with pytest.raises(ConversationError) as error:
        await store.reserve(identifier, uuid4(), question, str(uuid4()))
    assert error.value.status == 422
    with Session(claim_database.engine) as database:
        session = database.get(ChatSession, identifier)
        assert session is not None and session.active_turn_id is None
        assert session.next_turn_number == 1


async def test_missing_session_and_unready_document_rejected(
    store: TurnStore,
    new_session: Callable[[], UUID],
    claim_database: ClaimDatabase,
) -> None:
    with pytest.raises(ConversationError) as error:
        await store.reserve(uuid4(), uuid4(), "Question", str(uuid4()))
    assert error.value.status == 404
    identifier = new_session()
    with Session(claim_database.engine) as database, database.begin():
        member = database.scalar(
            select(ChatSessionDocument).where(ChatSessionDocument.session_id == identifier)
        )
        assert member is not None
        document = database.get(Document, member.document_id)
        assert document is not None
        document.status = "PENDING"
    with pytest.raises(ConversationError) as error:
        await store.reserve(identifier, uuid4(), "Question", str(uuid4()))
    assert error.value.code == "DOCUMENT_NOT_READY"


async def test_before_call_checks_fresh_clock_after_waiting_for_session_lock(
    store: TurnStore,
    retrieval_engine: AsyncEngine,
    new_session: Callable[[], UUID],
    claim_database: ClaimDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = reservation(await store.reserve(new_session(), uuid4(), "Question", str(uuid4())))
    clock = SimpleNamespace(now=first.deadline_at - timedelta(seconds=1))
    monkeypatch.setattr(conversations, "utcnow", lambda: clock.now)
    attempted = asyncio.Event()

    def observe(
        connection: object,
        cursor: object,
        statement: str,
        parameters: object,
        context: object,
        executemany: bool,
    ) -> None:
        if "FOR UPDATE" in statement.upper():
            attempted.set()

    event.listen(retrieval_engine.sync_engine, "before_cursor_execute", observe)
    pending: asyncio.Task[float] | None = None
    try:
        with Session(claim_database.engine) as database, database.begin():
            assert database.get(ChatSession, first.session_id, with_for_update=True) is not None
            pending = asyncio.create_task(store.before_call(first))
            await asyncio.wait_for(attempted.wait(), timeout=2)
            assert not pending.done()
            clock.now = first.deadline_at + timedelta(seconds=1)
        with pytest.raises(ConversationError) as error:
            await asyncio.wait_for(pending, timeout=3)
        assert error.value.status == 504
    finally:
        event.remove(retrieval_engine.sync_engine, "before_cursor_execute", observe)
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)


async def test_profile_mismatch_rejected_before_reserving(
    store: TurnStore,
    new_session: Callable[[], UUID],
    claim_database: ClaimDatabase,
) -> None:
    identifier = new_session()
    values = profile_values(claim_database.settings)
    values.update(fingerprint=uuid4().hex * 2, embedding_model="unsupported-test-profile")
    with Session(claim_database.engine) as database, database.begin():
        profile = IndexProfile(**values)
        database.add(profile)
        database.flush()
        membership = database.scalar(
            select(ChatSessionDocument).where(ChatSessionDocument.session_id == identifier)
        )
        assert membership is not None
        document = database.get(Document, membership.document_id)
        assert document is not None
        document.profile_id = profile.id
    with pytest.raises(ConversationError) as error:
        await store.reserve(identifier, uuid4(), "Question", str(uuid4()))
    assert error.value.code == "INDEX_PROFILE_MISMATCH"
    with Session(claim_database.engine) as database:
        session = database.get(ChatSession, identifier)
        assert session is not None and session.active_turn_id is None


async def test_prompt_and_review_have_independent_persisted_budgets_and_audit_is_not_history(
    store: TurnStore,
    new_session: Callable[[], UUID],
    claim_database: ClaimDatabase,
) -> None:
    current = reservation(
        await store.reserve(new_session(), uuid4(), "Question", "review-context-request")
    )
    prompt_context = {"sources": [{"text": "source " * 11_000}]}
    review = {"claim_records": "audit_only_marker " * 1_000}
    context = {**prompt_context, "evidence_review": review}
    encoding = tiktoken.get_encoding("o200k_base")

    def tokens(value: object) -> int:
        return len(encoding.encode(json.dumps(value, ensure_ascii=False)))

    settings = claim_database.settings
    review_budget = (
        settings.retrieval_max_tokens
        + settings.chat_max_output_tokens
        + settings.evidence_review_max_output_tokens
        + settings.INSTRUCTION_RESERVE_TOKENS
    )
    assert tokens(prompt_context) < settings.chat_prompt_max_tokens
    assert tokens(review) < review_budget
    assert tokens(context) > settings.chat_prompt_max_tokens
    await store.save_context(
        current,
        query="Question",
        intent="factual",
        evidence_scope="retrieved_excerpts",
        context=context,
        timings={},
    )
    with Session(claim_database.engine) as database:
        turn = database.get(ChatTurn, current.turn_id)
        assert turn is not None
        assert turn.context_snapshot["sources"] == prompt_context["sources"]
        assert turn.context_snapshot["evidence_review"] == review
    await store.complete(current, answer("A saved answer."), timings={})
    following = reservation(
        await store.reserve(current.session_id, uuid4(), "Follow up?", str(uuid4()))
    )
    history = await store.history(following)
    assert history == [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "A saved answer."},
    ]
    assert "audit_only_marker" not in json.dumps(history)


@pytest.mark.parametrize("invalid_part", ["prompt", "audit", "audit_type"])
async def test_oversized_context_does_not_overwrite_persisted_inputs(
    store: TurnStore,
    new_session: Callable[[], UUID],
    claim_database: ClaimDatabase,
    invalid_part: str,
) -> None:
    current = reservation(
        await store.reserve(new_session(), uuid4(), "Question", "context-request")
    )
    invalid_context = (
        {"sources": [{"text": "word " * 13_000}]}
        if invalid_part == "prompt"
        else {"evidence_review": {"records": "word " * 13_000}}
        if invalid_part == "audit"
        else {"evidence_review": ["wrong shape"]}
    )
    with pytest.raises(ConversationError) as error:
        await store.save_context(
            current,
            query="Question",
            intent="factual",
            evidence_scope="retrieved_excerpts",
            context=invalid_context,
            timings={},
        )
    assert error.value.code == "INVALID_CHAT_CONTEXT"
    with Session(claim_database.engine) as database:
        turn = database.get(ChatTurn, current.turn_id)
        assert turn is not None
        assert turn.context_snapshot == {"request_id": "context-request"}
        assert turn.planner_intent is None
