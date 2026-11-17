"""Opt-in service-level live checks with real PG/vectors/history and a hard budget.

This executes actual ChatProcessor/TurnStore, not HTTP or a browser. It creates
new inspectable sessions only; originals, vectors and prior history are untouched.
Every planner, embedding, draft and reviewer request uses the shared100ledger.
Interrupted/failed/completed run IDs are never resumed or silently retried.
"""

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from sqlalchemy.orm import defer

from app.adapters.async_openai import AsyncOpenAIEmbeddingProvider, AsyncOpenAIGenerationProvider
from app.adapters.retrieval import PgVectorRetriever
from app.core.config import Settings, get_settings
from app.db.models import (
    ChatMessage,
    ChatSession,
    ChatSessionDocument,
    ChatTurn,
    Document,
    IndexProfile,
)
from app.db.profiles import profile_values
from app.db.session import api_engine
from app.domain.conversation import Replay, TurnReservation
from app.services.chat import ChatProcessor
from app.services.chat_review import CHAT_REVIEW_PROMPT_VERSION
from app.services.conversations import TurnStore
from scripts.evaluate_ai import _atomic_write, evaluation_lock, expected_versions, validate_run_id
from scripts.evaluate_review import BUDGET_FILE
from scripts.live_budget import (
    AsyncBudgetedEmbeddingProvider,
    AsyncBudgetedGenerationProvider,
    RequestBudget,
)

QUESTIONS = {
    "databases": (
        "What is database",
        "Where does Data in databases is stored ?",
        "What is the current weather outside my home right now?",
    ),
    "narrative": (
        "Summarize the retention rules and exceptions: include the clocks for ordinary and "
        "disputed requests, active requests, and legal holds.",
        "And does releasing a legal hold start a new universal thirty-day retention clock?",
        "What is the current weather outside my home right now?",
    ),
}


def initialize_run(path: Path, document_id: UUID, suite: str) -> dict[str, Any]:
    """Reserve the run before business writes; never overwrite uncertain evidence."""
    if path.exists() or path.is_symlink():
        raise ValueError("SERVICE_RUN_ALREADY_EXISTS")
    if suite not in QUESTIONS:
        raise ValueError("INVALID_SERVICE_SUITE")
    state: dict[str, Any] = {
        "evidence_type": "actual_service_pipeline_not_http_e2e",
        "status": "reserved",
        "document_id": str(document_id),
        "session_id": str(uuid4()),
        "suite": suite,
        "created_at": datetime.now(UTC).isoformat(),
        "cases": [
            {"question": question, "client_request_id": str(uuid4()), "status": "unstarted"}
            for question in QUESTIONS[suite]
        ],
    }
    _atomic_write(path, json.dumps(state, ensure_ascii=False, indent=2).encode())
    return state


async def preflight_and_create(
    engine: AsyncEngine, settings: Settings, state: dict[str, Any]
) -> None:
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as database, database.begin():
        document = await database.get(
            Document, UUID(state["document_id"]), options=[defer(Document.extracted_text)]
        )
        if document is None or document.status != "COMPLETED":
            raise ValueError("DOCUMENT_NOT_READY")
        profile = await database.get(IndexProfile, document.profile_id)
        if profile is None or profile.fingerprint != profile_values(settings)["fingerprint"]:
            raise ValueError("INDEX_PROFILE_MISMATCH")
        if (document.chunk_count or 0) < 1 or document.published_attempt_id is None:
            raise ValueError("DOCUMENT_HAS_NO_PUBLISHED_EVIDENCE")
        now = datetime.now(UTC)
        session_id = UUID(state["session_id"])
        database.add(
            ChatSession(
                id=session_id,
                title=f"Quality check: {state['suite']} {now:%Y-%m-%d %H:%M}",
                created_at=now,
                updated_at=now,
                next_turn_number=1,
            )
        )
        await database.flush()
        database.add(ChatSessionDocument(session_id=session_id, document_id=document.id))
        state["filename"] = document.filename
        state["profile_id"] = str(profile.id)


async def execute_case(
    store: TurnStore,
    processor: ChatProcessor,
    session_id: UUID,
    record: dict[str, Any],
    budget: RequestBudget,
) -> Replay:
    """Actual reservation/run/replay; a replay must spend zero provider slots."""
    client_id = UUID(record["client_request_id"])
    reserved = await store.reserve(session_id, client_id, record["question"], str(uuid4()))
    if not isinstance(reserved, TurnReservation):
        raise ValueError("UNEXPECTED_EXISTING_TURN")
    record["turn_id"] = str(reserved.turn_id)
    response = await processor.run(reserved)
    before_replay = budget.snapshot()
    replay = await store.reserve(session_id, client_id, record["question"], str(uuid4()))
    if not isinstance(replay, Replay) or replay != response or budget.snapshot() != before_replay:
        raise ValueError("SERVICE_REPLAY_MISMATCH")
    record.update(
        status="completed" if response.status == 200 else "failed",
        response_status=response.status,
        response=response.body,
        replay_equal=True,
        replay_new_paid_calls=0,
    )
    return response


async def capture_persisted(engine: AsyncEngine, state: dict[str, Any]) -> None:
    async with async_sessionmaker(engine)() as database:
        turns = (
            await database.scalars(
                select(ChatTurn)
                .where(ChatTurn.session_id == UUID(state["session_id"]))
                .order_by(ChatTurn.turn_number)
            )
        ).all()
        state["persisted_turns"] = [
            {
                "id": str(turn.id),
                "number": turn.turn_number,
                "status": turn.status,
                "calls_reserved": turn.ai_calls_reserved,
                "versions": turn.versions,
                "timings": turn.timings,
                "context": turn.context_snapshot,
            }
            for turn in turns
        ]
        messages = (
            await database.scalars(
                select(ChatMessage)
                .where(ChatMessage.session_id == UUID(state["session_id"]))
                .order_by(ChatMessage.sequence)
            )
        ).all()
        state["persisted_history"] = [
            {
                "id": str(message.id),
                "sequence": message.sequence,
                "role": message.role,
                "content": message.content,
                "citations": message.citations,
            }
            for message in messages
        ]


async def run(settings: Settings, run_id: str, document_id: UUID, suite: str) -> dict[str, Any]:
    validate_run_id(run_id)
    directory = settings.local_storage_dir / "demo"
    with evaluation_lock(directory):
        path = directory / f"service-review-{run_id}.json"
        budget = RequestBudget(directory / BUDGET_FILE, limit=100)
        state = initialize_run(path, document_id, suite)
        state["budget_before"] = budget.snapshot()
        # This run reuses an existing document; its saved analysis can have an
        # older prompt. Record the current chat recipe without relabeling it.
        state["recipe"] = {
            **{
                name: value
                for name, value in expected_versions(settings).items()
                if name != "saved_analysis_prompt"
            },
            "review_prompt": CHAT_REVIEW_PROMPT_VERSION,
        }

        def save() -> None:
            _atomic_write(path, json.dumps(state, ensure_ascii=False, indent=2).encode())

        save()
        engine = api_engine(settings)
        try:
            await preflight_and_create(engine, settings, state)
            save()
            store = TurnStore(engine, settings)
            processor = ChatProcessor(
                engine,
                settings,
                store,
                PgVectorRetriever(engine),
                AsyncBudgetedGenerationProvider(
                    AsyncOpenAIGenerationProvider(settings), budget, "service.planner"
                ),
                AsyncBudgetedEmbeddingProvider(
                    AsyncOpenAIEmbeddingProvider(settings),
                    budget,
                    "service.embedding",
                    profile=f"{settings.openai_embedding_model}.{settings.openai_embedding_dimensions}",
                ),
                reviewer=AsyncBudgetedGenerationProvider(
                    AsyncOpenAIGenerationProvider(
                        settings, reasoning_effort=settings.openai_review_reasoning_effort
                    ),
                    budget,
                    "service.review",
                    profile=settings.openai_review_reasoning_effort,
                ),
                answerer=AsyncBudgetedGenerationProvider(
                    AsyncOpenAIGenerationProvider(
                        settings, reasoning_effort=settings.openai_answer_reasoning_effort
                    ),
                    budget,
                    "service.answer",
                    profile=settings.openai_answer_reasoning_effort,
                ),
                source_answerer=AsyncBudgetedGenerationProvider(
                    AsyncOpenAIGenerationProvider(
                        settings, model=settings.openai_source_answer_model, reasoning_effort="low"
                    ),
                    budget,
                    "service.source_answer",
                    profile=f"{settings.openai_source_answer_model}:low",
                ),
            )
            for case in state["cases"]:
                # Reserve before any call; crashes remain explicit and cannot be resumed.
                case["status"] = "reserved"
                save()
                response = await execute_case(
                    store, processor, UUID(state["session_id"]), case, budget
                )
                await capture_persisted(engine, state)
                state["budget_after"] = budget.snapshot()
                save()
                print(
                    json.dumps(
                        {"turn_id": case["turn_id"], "status": response.status, **budget.snapshot()}
                    ),
                    flush=True,
                )
                if response.status != 200:
                    state["status"] = "failed"
                    break
            else:
                state["status"] = "completed"
            save()
        except Exception as error:
            # Never retain raw provider/DB exceptions, prompts or credentials in error text.
            state.update(status="interrupted", error_class=type(error).__name__)
            state["budget_after"] = budget.snapshot()
            save()
            raise ValueError("SERVICE_EVALUATION_INTERRUPTED_SEE_ARTIFACT") from None
        finally:
            await engine.dispose()
        return {"artifact": str(path), "session_id": state["session_id"], **budget.snapshot()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Explicitly authorize paid requests")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--document-id", required=True, type=UUID)
    parser.add_argument("--suite", choices=tuple(QUESTIONS), default="databases")
    args = parser.parse_args()
    if not args.live:
        parser.error("--live is required; this command makes paid requests")
    print(json.dumps(asyncio.run(run(get_settings(), args.run_id, args.document_id, args.suite))))


if __name__ == "__main__":
    main()
