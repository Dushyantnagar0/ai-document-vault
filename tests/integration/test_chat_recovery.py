"""Real private-schema PostgreSQL recovery/fencing; no API worker or provider calls."""

import asyncio
import os
from collections.abc import Callable
from dataclasses import replace
from datetime import timedelta
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_conversations import answer, reservation
from test_conversations import new_session as new_session
from test_conversations import store as store
from test_processing_claims import ClaimDatabase
from test_processing_claims import claim_database as claim_database
from test_processing_claims import new_document as new_document
from test_retrieval import publish as publish
from test_retrieval import retrieval_engine as retrieval_engine

from app.db.models import ChatMessage, ChatSession, ChatTurn
from app.domain.conversation import ConversationError
from app.services.conversations import TurnStore, utcnow

pytestmark = [
    pytest.mark.integration,
    pytest.mark.enable_socket,
    pytest.mark.skipif(
        os.environ.get("VAULT_INTEGRATION_TESTS") != "1",
        reason="Requires isolated Compose PostgreSQL",
    ),
]


def expire_at(database: ClaimDatabase, turn_id: UUID, seconds_ago: int) -> None:
    with Session(database.engine) as session, session.begin():
        turn = session.get(ChatTurn, turn_id)
        assert turn is not None
        turn.expires_at = utcnow() - timedelta(seconds=seconds_ago)
        turn.deadline_at = turn.expires_at - timedelta(seconds=1)


async def test_recovery_skips_locked_oldest_session_and_expires_other_turn(
    store: TurnStore,
    new_session: Callable[[], UUID],
    claim_database: ClaimDatabase,
) -> None:
    oldest = reservation(
        await store.reserve(new_session(), uuid4(), "Oldest question", str(uuid4()))
    )
    other = reservation(await store.reserve(new_session(), uuid4(), "Other question", str(uuid4())))
    expire_at(claim_database, oldest.turn_id, 2)
    expire_at(claim_database, other.turn_id, 1)
    with Session(claim_database.engine) as blocking, blocking.begin():
        assert blocking.get(ChatSession, oldest.session_id, with_for_update=True) is not None
        # The database statement timeout is 5s. Recovery must finish while this
        # oldest session stays locked rather than waiting for that timeout/release.
        assert await asyncio.wait_for(store.expire_pending(), timeout=2) == 1
        with Session(claim_database.engine) as inspection:
            locked = inspection.get(ChatTurn, oldest.turn_id)
            recovered = inspection.get(ChatTurn, other.turn_id)
            locked_session = inspection.get(ChatSession, oldest.session_id)
            recovered_session = inspection.get(ChatSession, other.session_id)
            assert locked is not None and locked.status == "PENDING"
            assert locked_session is not None and locked_session.active_turn_id == oldest.turn_id
            assert recovered is not None and recovered.status == "FAILED"
            assert recovered.response_status == 504 and recovered.ai_calls_reserved == 0
            assert recovered.response_body["error"]["code"] == "CHAT_TIMEOUT"
            assert recovered_session is not None and recovered_session.active_turn_id is None
    assert await asyncio.wait_for(store.expire_pending(), timeout=2) == 1
    assert await store.expire_pending() == 0
    with Session(claim_database.engine) as inspection:
        recovered = inspection.get(ChatTurn, oldest.turn_id)
        assistant = inspection.get(ChatMessage, oldest.assistant_message_id)
        assert recovered is not None and assistant is not None
        assert recovered.status == assistant.status == "FAILED"
        assert (
            recovered.response_status == 504
            and assistant.content == "Question expired before completion."
        )
        assert recovered.ai_calls_reserved == 0


async def test_recovered_terminal_turn_is_immutable_and_cannot_reserve_more_calls(
    store: TurnStore,
    new_session: Callable[[], UUID],
    claim_database: ClaimDatabase,
) -> None:
    key, question = uuid4(), "Expired original question"
    original = reservation(
        await store.reserve(new_session(), key, question, "stable-recovery-request")
    )
    assert await store.before_call(original) > 0
    expire_at(claim_database, original.turn_id, 1)
    assert await store.expire_pending() == 1
    with Session(claim_database.engine) as inspection:
        terminal = inspection.get(ChatTurn, original.turn_id)
        assert terminal is not None
        saved_body, saved_end, saved_timings = (
            terminal.response_body,
            terminal.ended_at,
            terminal.timings,
        )
    active = reservation(
        await store.reserve(original.session_id, uuid4(), "New intentional question", str(uuid4()))
    )
    late = await store.complete(
        original, answer("Late output must never replace timeout."), timings={"answer_ms": 999}
    )
    assert late.status == 504 and late.body == saved_body
    replay = await store.reserve(original.session_id, key, question, "different-transport-request")
    assert replay == late
    with pytest.raises(ConversationError) as captured:
        await store.before_call(original)
    assert captured.value.code == "CHAT_TURN_STALE"
    with pytest.raises(ConversationError) as captured:
        await store.complete(replace(original, token=uuid4()), answer("Wrong owner."), timings={})
    assert captured.value.code == "CHAT_TURN_STALE"
    with Session(claim_database.engine) as inspection:
        terminal = inspection.get(ChatTurn, original.turn_id)
        current = inspection.get(ChatTurn, active.turn_id)
        conversation = inspection.get(ChatSession, original.session_id)
        assert terminal is not None and current is not None and conversation is not None
        assert terminal.response_body == saved_body and terminal.ended_at == saved_end
        assert terminal.timings == saved_timings and terminal.ai_calls_reserved == 1
        assert terminal.status == "FAILED" and current.status == "PENDING"
        assert current.ai_calls_reserved == 0 and conversation.active_turn_id == active.turn_id
        turns = inspection.scalars(
            select(ChatTurn).where(ChatTurn.session_id == original.session_id)
        ).all()
        messages = inspection.scalars(
            select(ChatMessage)
            .where(ChatMessage.session_id == original.session_id)
            .order_by(ChatMessage.sequence)
        ).all()
        assert len(turns) == 2
        assert [(message.sequence, message.role, message.status) for message in messages] == [
            (1, "user", "COMPLETED"),
            (2, "assistant", "FAILED"),
            (3, "user", "COMPLETED"),
            (4, "assistant", "PENDING"),
        ]
        assert messages[1].content == "Question expired before completion."
        assert messages[1].citations == []
