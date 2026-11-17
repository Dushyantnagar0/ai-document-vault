"""Opt-in paid generation regression on frozen retrieval/history, not API E2E.

Every paid source stage or legacy draft/audit uses the shared durable100-request budget. Completed,
failed and interrupted cases are never retried within a run; select a new run ID
explicitly for new work. No embeddings, planner calls or business-state writes.
"""

import argparse
import asyncio
import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.orm import Session

from app.adapters.async_openai import AsyncOpenAIGenerationProvider
from app.adapters.retrieval import VectorRetriever
from app.core.config import Settings, get_settings
from app.db.models import ChatTurn
from app.domain.conversation import (
    AsyncEmbeddingProvider,
    AsyncGenerationProvider,
    Evidence,
    TurnReservation,
)
from app.domain.documents import ProcessingError
from app.prompts.analysis import generation_prompt_tokens
from app.prompts.chat import (
    ANSWER_PROMPT_VERSION,
    ANSWER_SYSTEM,
    CHAT_PIPELINE_VERSION,
    EXHAUSTIVE_SYSTEM,
)
from app.services.analysis_evidence import QUOTE_CATALOG_VERSION
from app.services.chat import ChatProcessor
from app.services.chat_context import (
    AnswerSelectionOutput,
    ExhaustiveSelectionOutput,
    _evidence_json,
    answer_quote_catalog,
    exhaustive_selection_to_answer,
    resolve_answer_selection,
)
from app.services.chat_review import CHAT_REVIEW_PROMPT_VERSION
from app.services.conversations import TurnStore
from app.services.evidence_review import PROJECTION_VERSION
from app.services.grounding import INSUFFICIENT_ANSWER, GroundedAnswer, validate_answer
from app.services.source_answer import (
    SOURCE_COMPOSITION_VERSION,
    SOURCE_SELECTION_VERSION,
    prepare_selection_packet,
)
from scripts.evaluate_ai import (
    _atomic_write,
    automatic_checks,
    evaluation_lock,
    select_cases,
    validate_run_id,
)
from scripts.evaluate_review import BUDGET_FILE, fingerprint
from scripts.evaluation_fixtures import EvaluationCase
from scripts.live_budget import AsyncBudgetedGenerationProvider, RequestBudget
from scripts.quality_fixtures import QUALITY_CASES
from scripts.reproject_review import saved_evidence


class FrozenStore:
    """Local call guard only; frozen replay never reserves real chat turns."""

    def __init__(self) -> None:
        self.calls = 0

    async def before_call(self, reservation: TurnReservation) -> float:
        self.calls += 1
        if self.calls > 3:
            raise ValueError("FROZEN_CASE_CALL_LIMIT")
        return max(0.001, (reservation.deadline_at - datetime.now(UTC)).total_seconds())


class FrozenProcessor(ChatProcessor):
    async def _sample(self, *args: Any, **kwargs: Any) -> None:
        """No production telemetry or business state is changed by frozen replay."""


class RecordingProvider:
    def __init__(
        self,
        delegate: AsyncGenerationProvider,
        record: dict[str, Any],
        save: Callable[[], None],
        phase: str,
    ) -> None:
        self.delegate, self.record, self.save, self.phase = delegate, record, save, phase

    async def generate(self, **kwargs: Any) -> Any:
        observation = {
            "phase": self.phase,
            "status": "reserved",
            "schema": kwargs["schema"].model_json_schema(),
            "system": kwargs["system"],
            "content": kwargs["content"],
            "max_output_tokens": kwargs["max_output_tokens"],
        }
        self.record.setdefault("requests", []).append(observation)
        self.save()
        result = await self.delegate.generate(**kwargs)
        observation.update(
            status="completed",
            result=result.result.model_dump(mode="json"),
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
        self.save()
        return result


def make_reservation(
    question: str, evidence: Sequence[Evidence], timeout: float
) -> TurnReservation:
    identifier = uuid4()
    now = datetime.now(UTC)
    return TurnReservation(
        identifier,
        identifier,
        identifier,
        identifier,
        identifier,
        evidence[0].document_id if evidence else UUID(int=0),
        evidence[0].profile_id if evidence else UUID(int=0),
        question,
        now + timedelta(seconds=timeout),
        now + timedelta(seconds=timeout),
        str(identifier),
        1,
    )


async def generate_case(
    settings: Settings,
    payload: dict[str, Any],
    evidence: tuple[Evidence, ...],
    generator: AsyncGenerationProvider,
    reviewer: AsyncGenerationProvider,
) -> tuple[GroundedAnswer, dict[str, Any]]:
    """Run the current factual source flow or the unchanged legacy route."""
    if not evidence:
        return GroundedAnswer(INSUFFICIENT_ANSWER, "insufficient", []), {
            "audit_skipped": "no_evidence"
        }
    reservation = make_reservation(payload["question"], evidence, settings.chat_timeout_seconds)
    processor = FrozenProcessor(
        cast(AsyncEngine, None),
        settings,
        cast(TurnStore, FrozenStore()),
        cast(VectorRetriever, None),
        generator,
        cast(AsyncEmbeddingProvider, None),
        reviewer=reviewer,
        answerer=generator,
        source_answerer=generator,
    )
    complete_scope = bool(payload["complete_scope"])
    if payload["intent"] == "factual":
        async with asyncio.timeout(settings.chat_timeout_seconds):
            return await processor._source_answer(
                reservation,
                evidence,
                resolved_query=payload.get("retrieval_query"),
                complete_scope=complete_scope,
            )
    extractive = payload["intent"] == "exhaustive" and not complete_scope
    request = {**payload, "evidence": _evidence_json(evidence)}
    content = json.dumps(request, ensure_ascii=False, separators=(",", ":"))
    schema = ExhaustiveSelectionOutput if extractive else AnswerSelectionOutput
    system = EXHAUSTIVE_SYSTEM if extractive else ANSWER_SYSTEM
    if generation_prompt_tokens(schema, system, content) > settings.chat_prompt_max_tokens:
        raise ValueError("FROZEN_CONTEXT_EXCEEDS_CURRENT_PROMPT_LIMIT")
    async with asyncio.timeout(settings.chat_timeout_seconds):
        generated = await processor._generate(
            reservation,
            schema,
            system,
            content,
            settings.chat_max_output_tokens,
            "chat.answer",
            settings.openai_generation_timeout_seconds,
            provider=processor.answerer,
        )
        selection = (
            exhaustive_selection_to_answer(
                ExhaustiveSelectionOutput.model_validate(generated.result), evidence
            )
            if extractive
            else AnswerSelectionOutput.model_validate(generated.result)
        )
        draft = resolve_answer_selection(selection, evidence)
        if extractive or not draft.blocks:
            answer = validate_answer(draft, evidence, complete_scope=complete_scope)
            return answer, {"audit_skipped": "literal_excerpts" if extractive else "empty_draft"}
        return await processor._review_answer(
            reservation,
            draft,
            evidence,
            overview=payload["intent"] == "overview",
            complete_scope=complete_scope,
        )


def recipe(settings: Settings) -> dict[str, Any]:
    return {
        "evaluation_type": "frozen_evidence_generation_regression",
        "pipeline": CHAT_PIPELINE_VERSION,
        "answer_prompt": ANSWER_PROMPT_VERSION,
        "review_prompt": CHAT_REVIEW_PROMPT_VERSION,
        "projection": PROJECTION_VERSION,
        "quote_catalog": QUOTE_CATALOG_VERSION,
        "model": settings.openai_chat_model,
        "source_answer_model": settings.openai_source_answer_model,
        "source_answer_reasoning_effort": "low",
        "source_selection": SOURCE_SELECTION_VERSION,
        "source_composition": SOURCE_COMPOSITION_VERSION,
        "configuration": settings.chat_configuration,
    }


async def run(source: Path, run_id: str, cases: Sequence[EvaluationCase]) -> dict[str, Any]:
    validate_run_id(run_id)
    settings = get_settings()
    directory = settings.local_storage_dir / "demo"
    output = directory / f"chat-review-frozen-{run_id}.json"
    frozen = json.loads(await asyncio.to_thread(source.read_text))
    current_recipe = recipe(settings)
    binding = {"source_sha256": fingerprint(frozen), "recipe": current_recipe}
    budget = RequestBudget(directory / BUDGET_FILE, limit=100)
    with evaluation_lock(directory):
        state: dict[str, Any] = (
            json.loads(output.read_text()) if output.exists() else {**binding, "cases": {}}
        )
        if any(state.get(key) != value for key, value in binding.items()):
            raise ValueError("FROZEN_RUN_RECIPE_CHANGED")

        def save() -> None:
            _atomic_write(output, json.dumps(state, ensure_ascii=False, indent=2).encode())

        engine = create_engine(settings.database_url, hide_parameters=True)
        try:
            for case in cases:
                if case.id in state["cases"]:
                    continue
                original = frozen["cases"][case.id]
                if original["status"] != "COMPLETED" or original["question"] != case.question:
                    raise ValueError("FROZEN_CASE_NOT_COMPLETE_OR_CHANGED")
                turn_id = original["response"]["turn_id"]
                with Session(engine) as database:
                    database.execute(text("SET TRANSACTION READ ONLY"))
                    turn = database.get(ChatTurn, UUID(turn_id))
                    if turn is None:
                        raise ValueError("FROZEN_TURN_UNAVAILABLE")
                    payload = dict(turn.context_snapshot["payload"])
                    if payload["intent"] == "factual":
                        if not turn.retrieval_query:
                            raise ValueError("FROZEN_QUERY_UNAVAILABLE")
                        payload["retrieval_query"] = turn.retrieval_query
                    if payload["question"] != case.question:
                        raise ValueError("FROZEN_QUESTION_CHANGED")
                    evidence = saved_evidence(database, turn_id)
                    if any(str(e.document_id) != original["document_id"] for e in evidence):
                        raise ValueError("FROZEN_DOCUMENT_MEMBERSHIP_INVALID")
                record: dict[str, Any] = {
                    "status": "reserved",
                    "source_turn_id": turn_id,
                    "frozen_payload": payload,
                    "question": case.question,
                    "expectations": {
                        "patterns": case.expected,
                        "status": case.status,
                        "forbidden": case.forbidden,
                    },
                    "semantic_review": "required",
                    "full_api_e2e": False,
                }
                state["cases"][case.id] = record
                save()
                factual = payload["intent"] == "factual"
                effort = "low" if factual else settings.openai_answer_reasoning_effort
                model = (
                    settings.openai_source_answer_model if factual else settings.openai_chat_model
                )
                generator = RecordingProvider(
                    AsyncBudgetedGenerationProvider(
                        AsyncOpenAIGenerationProvider(
                            settings, reasoning_effort=effort, model=model
                        ),
                        budget,
                        f"frozen.{case.id}.source_answer" if factual else f"frozen.{case.id}.draft",
                        profile=f"{model}:{effort}" if factual else effort,
                    ),
                    record,
                    save,
                    "source_answer" if factual else "draft",
                )
                reviewer = RecordingProvider(
                    AsyncBudgetedGenerationProvider(
                        AsyncOpenAIGenerationProvider(
                            settings, reasoning_effort=settings.openai_review_reasoning_effort
                        ),
                        budget,
                        f"frozen.{case.id}.review",
                        profile=settings.openai_review_reasoning_effort,
                    ),
                    record,
                    save,
                    "review",
                )
                try:
                    answer, audit = await generate_case(
                        settings, payload, evidence, generator, reviewer
                    )
                    body = {
                        "answer": answer.answer,
                        "evidence_status": answer.evidence_status,
                        "citations": answer.citations,
                    }
                    snapshot = {
                        "membership_valid": all(
                            c["document_id"] == original["document_id"] for c in answer.citations
                        ),
                        "selected_membership_valid": all(
                            str(e.document_id) == original["document_id"] for e in evidence
                        ),
                        "supplied_text": "\n".join(e.text for e in evidence),
                    }
                    record.update(
                        status="completed",
                        response=body,
                        audit=audit,
                        checks=automatic_checks(case, body, snapshot),
                        retained_blocks=[b.model_dump(mode="json") for b in answer.blocks],
                    )
                except ProcessingError as error:
                    record.update(
                        status="failed",
                        error_code=error.code,
                        diagnostics=error.diagnostics.as_dict() if error.diagnostics else None,
                    )
                except TimeoutError:
                    record.update(status="failed", error_code="FROZEN_CASE_DEADLINE")
                save()
                print(
                    json.dumps({"case": case.id, "status": record["status"], **budget.snapshot()}),
                    flush=True,
                )
        finally:
            engine.dispose()
        return {
            "artifact": str(output),
            "evaluation_type": "frozen_evidence_generation_regression",
            "full_api_e2e": False,
            **budget.snapshot(),
            "completed": sum(r["status"] == "completed" for r in state["cases"].values()),
            "automatic_pass": sum(
                bool(r.get("checks", {}).get("automatic_pass")) for r in state["cases"].values()
            ),
        }


def check_only(source: Path, cases: Sequence[EvaluationCase]) -> dict[str, Any]:
    """Verify all frozen bindings/prompt budgets before any provider is constructed."""
    settings = get_settings()
    state = json.loads(source.read_text())
    engine = create_engine(settings.database_url, hide_parameters=True)
    results = {}
    try:
        with Session(engine) as database:
            database.execute(text("SET TRANSACTION READ ONLY"))
            for case in cases:
                original = state["cases"][case.id]
                if original["status"] != "COMPLETED" or original["question"] != case.question:
                    raise ValueError("FROZEN_CASE_NOT_COMPLETE_OR_CHANGED")
                turn_id = original["response"]["turn_id"]
                turn = database.get(ChatTurn, UUID(turn_id))
                if turn is None:
                    raise ValueError("FROZEN_TURN_UNAVAILABLE")
                payload = dict(turn.context_snapshot["payload"])
                if payload["intent"] == "factual":
                    if not turn.retrieval_query:
                        raise ValueError("FROZEN_QUERY_UNAVAILABLE")
                    payload["retrieval_query"] = turn.retrieval_query
                evidence = saved_evidence(database, turn_id)
                if payload["question"] != case.question or any(
                    str(e.document_id) != original["document_id"] for e in evidence
                ):
                    raise ValueError("FROZEN_CASE_BINDING_CHANGED")
                extractive = payload["intent"] == "exhaustive" and not payload["complete_scope"]
                if payload["intent"] == "factual":
                    packet = prepare_selection_packet(
                        payload["question"],
                        payload.get("retrieval_query"),
                        answer_quote_catalog(evidence),
                        settings,
                    )
                    results[case.id] = {
                        "intent": "factual",
                        "sources": len(evidence),
                        "input_tokens": packet.input_tokens,
                        "maximum_paid_calls": 3 if evidence else 0,
                    }
                    continue
                content = json.dumps(
                    {**payload, "evidence": _evidence_json(evidence)},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                input_tokens = generation_prompt_tokens(
                    ExhaustiveSelectionOutput if extractive else AnswerSelectionOutput,
                    EXHAUSTIVE_SYSTEM if extractive else ANSWER_SYSTEM,
                    content,
                )
                if input_tokens > settings.chat_prompt_max_tokens:
                    raise ValueError("FROZEN_CONTEXT_EXCEEDS_CURRENT_PROMPT_LIMIT")
                results[case.id] = {
                    "intent": payload["intent"],
                    "sources": len(evidence),
                    "input_tokens": input_tokens,
                    "maximum_paid_calls": 0 if not evidence else 1 if extractive else 2,
                }
    finally:
        engine.dispose()
    return {
        "paid_calls": 0,
        "full_api_e2e": False,
        "evaluation_type": "frozen_evidence_generation_regression",
        "maximum_new_paid_calls": sum(r["maximum_paid_calls"] for r in results.values()),
        "cases": results,
    }


def selected_cases(suite: str, value: str | None) -> tuple[EvaluationCase, ...]:
    if suite == "core":
        return select_cases(value)
    if suite != "quality":
        raise ValueError("UNKNOWN_FROZEN_SUITE")
    if value is None:
        return QUALITY_CASES
    selected = {identifier.strip().upper() for identifier in value.split(",")}
    if not selected or not selected.issubset({case.id for case in QUALITY_CASES}):
        raise ValueError("UNKNOWN_QUALITY_CASE_ID")
    return tuple(case for case in QUALITY_CASES if case.id in selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--check-only", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--cases")
    parser.add_argument("--suite", choices=("core", "quality"), default="core")
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    source = args.source or get_settings().local_storage_dir / (
        "demo/eval-quality-v1.json"
        if args.suite == "quality"
        else "demo/eval-core-v1-prompts-v11-20261116.json"
    )
    cases = selected_cases(args.suite, args.cases)
    if args.check_only:
        print(json.dumps(check_only(source, cases), indent=2))
    else:
        if not args.run_id:
            parser.error("--live requires --run-id")
        print(json.dumps(asyncio.run(run(source, args.run_id, cases)), indent=2))


if __name__ == "__main__":
    main()
