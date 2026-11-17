"""Bounded asynchronous RAG; facts come from original evidence, never history."""

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from sqlalchemy.orm import defer

from app.adapters.retrieval import VectorRetriever
from app.core.config import Settings
from app.db.models import Document, DocumentInsights, PerformanceSample
from app.domain.conversation import (
    AsyncEmbeddingProvider,
    AsyncGenerationProvider,
    ConversationError,
    Evidence,
    Replay,
    TurnReservation,
)
from app.domain.documents import AIFailureDetails, Generated, ProcessingError
from app.prompts.analysis import generation_prompt_tokens
from app.prompts.chat import (
    ANSWER_PROMPT_VERSION,
    ANSWER_REPAIR,
    ANSWER_SYSTEM,
    EXHAUSTIVE_SYSTEM,
    PLANNER_PROMPT_VERSION,
    PLANNER_SYSTEM,
)
from app.services.chat_context import (
    AnswerSelectionOutput,
    ExhaustiveSelectionOutput,
    answer_quote_catalog,
    exhaustive_selection_to_answer,
    prepare_context,
    resolve_answer_selection,
)
from app.services.chat_review import (
    CHAT_REVIEW_PROMPT_VERSION,
    CHAT_REVIEW_SYSTEM,
    ChatReviewOutput,
    chat_review_input_limit,
    compact_chat_packet,
    project_chat_review,
    review_input_reservation,
)
from app.services.conversations import TurnStore
from app.services.evidence_review import (
    PROJECTION_VERSION,
    REVIEW_REPAIR,
    REVIEW_REPAIRABLE_CODES,
    prepare_review,
)
from app.services.grounding import (
    INSUFFICIENT_ANSWER,
    AnswerOutput,
    GroundedAnswer,
    PlannerOutput,
    append_evidence_gaps,
    validate_answer,
)
from app.services.ingestion import utcnow
from app.services.source_answer import (
    SOURCE_COMPOSITION_VERSION,
    SOURCE_SELECTION_VERSION,
    CompositionOutput,
    SourceAnswerPlan,
    SourceSelectionOutput,
    prepare_composition_packet,
    prepare_selection_packet,
    project_composition,
    validate_selection,
)

T = TypeVar("T", bound=BaseModel)


class ChatProcessor:
    def __init__(
        self,
        engine: AsyncEngine,
        settings: Settings,
        store: TurnStore,
        retriever: VectorRetriever,
        generator: AsyncGenerationProvider,
        embedder: AsyncEmbeddingProvider,
        *,
        reviewer: AsyncGenerationProvider,
        source_answerer: AsyncGenerationProvider,
        answerer: AsyncGenerationProvider | None = None,
    ) -> None:
        self.engine, self.settings, self.store = engine, settings, store
        self.retriever, self.generator, self.embedder = retriever, generator, embedder
        self.reviewer = reviewer
        self.source_answerer = source_answerer
        self.answerer = generator if answerer is None else answerer
        self.semaphore = asyncio.Semaphore(settings.api_max_concurrent_ai_requests)

    async def _sample(
        self,
        reservation: TurnReservation,
        operation: str,
        started: float,
        *,
        result: Generated[Any] | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        error: str | None = None,
    ) -> None:
        try:
            async with async_sessionmaker(self.engine)() as session, session.begin():
                session.add(
                    PerformanceSample(
                        operation=operation,
                        duration_ms=(time.monotonic() - started) * 1000,
                        outcome="failed" if error else "succeeded",
                        error_class=error,
                        document_id=reservation.document_id,
                        turn_id=reservation.turn_id,
                        input_tokens=result.input_tokens if result else input_tokens,
                        output_tokens=result.output_tokens if result else output_tokens,
                    )
                )
        except Exception:
            pass

    async def _generate(
        self,
        reservation: TurnReservation,
        schema: type[T],
        system: str,
        content: str,
        output: int,
        operation: str,
        request_timeout_seconds: float,
        *,
        provider: AsyncGenerationProvider | None = None,
    ) -> Generated[T]:
        remaining = await self.store.before_call(reservation)
        started = time.monotonic()
        try:
            result = await (provider or self.generator).generate(
                schema=schema,
                system=system,
                content=content,
                max_output_tokens=output,
                timeout_seconds=min(remaining, request_timeout_seconds),
            )
        except ProcessingError as error:
            details = error.diagnostics.as_dict() if error.diagnostics is not None else {}
            input_usage, output_usage = details.get("input_tokens"), details.get("output_tokens")
            await self._sample(
                reservation,
                operation,
                started,
                error=error.code,
                input_tokens=input_usage if isinstance(input_usage, int) else None,
                output_tokens=output_usage if isinstance(output_usage, int) else None,
            )
            raise
        await self._sample(reservation, operation, started, result=result)
        return result

    async def _source_answer(
        self,
        reservation: TurnReservation,
        evidence: tuple[Evidence, ...],
        *,
        resolved_query: str | None = None,
        complete_scope: bool = True,
        source_record: dict[str, Any] | None = None,
        persist: Callable[[], Awaitable[None]] | None = None,
    ) -> tuple[GroundedAnswer, dict[str, Any]]:
        """Select then compose with one shared structural repair and durable stage progress."""
        settings = self.settings
        record = source_record if source_record is not None else {}
        record.update(
            selection_version=SOURCE_SELECTION_VERSION,
            composition_version=SOURCE_COMPOSITION_VERSION,
            model=settings.openai_source_answer_model,
            reasoning_effort="low",
        )
        catalog = answer_quote_catalog(evidence)
        plan: SourceAnswerPlan | None = None
        repaired = False
        repairable = {
            "ai_invalid_output",
            "ai_output_truncated",
            "ai_invalid_json",
            "ai_schema_invalid",
            "SOURCE_ANSWER_INVALID",
            "SOURCE_ANSWER_OUTPUT_LIMIT",
            "CHAT_INVALID_OUTPUT",
        }
        for stage in ("selection", "composition"):
            stage_record: dict[str, Any] = {"status": "preflight"}
            record[stage] = stage_record
            if (
                stage == "composition"
                and plan is not None
                and plan.evidence_status == "insufficient"
            ):
                answer = GroundedAnswer(INSUFFICIENT_ANSWER, "insufficient", [])
                stage_record.update(status="skipped", reason="no_supported_components")
                record["projection"] = {"status": "completed", "evidence_status": "insufficient"}
                if persist is not None:
                    await persist()
                return answer, record
            for attempt in range(2):
                started = time.monotonic()
                stage_record.update(attempt=attempt + 1, status="preflight")
                stage_record.pop("result", None)
                stage_record.pop("model", None)
                try:
                    if stage == "selection":
                        packet = prepare_selection_packet(
                            reservation.question,
                            resolved_query,
                            catalog,
                            settings,
                            repair=bool(attempt),
                        )
                    else:
                        assert plan is not None
                        packet = prepare_composition_packet(plan, settings, repair=bool(attempt))
                    stage_record.update(
                        prompt_version=packet.version,
                        request_sha256=hashlib.sha256(packet.content.encode()).hexdigest(),
                        input_tokens=packet.input_tokens,
                        reserved_input_tokens=packet.reserved_input_tokens,
                        input_limit=packet.input_limit,
                        max_output_tokens=packet.max_output_tokens,
                        status="requested",
                    )
                    generated = await self._generate(
                        reservation,
                        packet.schema,
                        packet.system,
                        packet.content,
                        packet.max_output_tokens,
                        f"chat.source_{stage}",
                        settings.openai_generation_timeout_seconds,
                        provider=self.source_answerer,
                    )
                    raw = generated.result.model_dump(mode="json")
                    stage_record.update(result=raw, model=generated.model)
                    record["model"] = generated.model
                    try:
                        if stage == "selection":
                            selected = SourceSelectionOutput.model_validate(raw)
                            plan = validate_selection(
                                selected,
                                catalog,
                                question=reservation.question,
                                resolved_query=resolved_query,
                                complete_scope=complete_scope,
                                max_output_tokens=packet.max_output_tokens,
                            )
                            record["quote_bindings"] = {
                                identifier: {"source_id": str(ref.source_id), "quote": ref.quote}
                                for identifier, ref in plan.passages
                            }
                        else:
                            assert plan is not None
                            answer = project_composition(
                                plan,
                                CompositionOutput.model_validate(raw),
                                evidence,
                                max_output_tokens=packet.max_output_tokens,
                            )
                            record["projection"] = {
                                "status": "completed",
                                "evidence_status": answer.evidence_status,
                                "retained_block_count": len(answer.blocks),
                                "citation_count": len(answer.citations),
                            }
                    except ValidationError:
                        raise ProcessingError(
                            "ai_schema_invalid",
                            "Source-selected answer did not match its output contract.",
                            diagnostics=AIFailureDetails(reason="schema_validation"),
                        ) from None
                    stage_record.update(status="completed")
                    stage_record.pop("error", None)
                    if stage == "composition":
                        return answer, record
                    break
                except ProcessingError as error:
                    details = error.diagnostics.as_dict() if error.diagnostics is not None else {}
                    stage_record.update(
                        status="failed", error={"code": error.code, "details": details}
                    )
                    if repaired or attempt or error.code not in repairable:
                        raise
                    repaired = True
                    stage_record["repair"] = {
                        "first_error_code": error.code,
                        "first_error": details,
                        "first_output_sha256": hashlib.sha256(
                            json.dumps(stage_record.get("result"), sort_keys=True).encode()
                        ).hexdigest(),
                    }
                finally:
                    stage_record["duration_ms"] = (time.monotonic() - started) * 1000
                    if persist is not None:
                        await persist()
        raise AssertionError("Source answer stages completed without an outcome")

    async def _review_answer(
        self,
        reservation: TurnReservation,
        draft: AnswerOutput,
        evidence: tuple[Evidence, ...],
        *,
        overview: bool,
        complete_scope: bool,
        review_record: dict[str, Any] | None = None,
        allow_repair: bool = True,
        resolved_query: str | None = None,
    ) -> tuple[GroundedAnswer, dict[str, Any]]:
        record = review_record if review_record is not None else {}
        for attempt in range(2 if allow_repair else 1):
            try:
                return await self._review_answer_once(
                    reservation,
                    draft,
                    evidence,
                    overview=overview,
                    complete_scope=complete_scope,
                    review_record=record,
                    repair=bool(attempt),
                    resolved_query=resolved_query,
                )
            except ProcessingError as error:
                if attempt or not allow_repair or error.code not in REVIEW_REPAIRABLE_CODES:
                    raise
                record["repair"] = {
                    "attempted": True,
                    "first_error_code": error.code,
                    "first_error": error.diagnostics.as_dict() if error.diagnostics else {},
                }
        raise AssertionError("Review loop exhausted without outcome")

    async def _review_answer_once(
        self,
        reservation: TurnReservation,
        draft: AnswerOutput,
        evidence: tuple[Evidence, ...],
        *,
        overview: bool,
        complete_scope: bool,
        review_record: dict[str, Any] | None = None,
        repair: bool = False,
        resolved_query: str | None = None,
    ) -> tuple[GroundedAnswer, dict[str, Any]]:
        """Review prose before rendering; a failed audit never publishes the draft."""
        packet = prepare_review(
            {"answer": [block.model_dump(mode="json") for block in draft.blocks]},
            answer_quote_catalog(evidence),
            question=reservation.question,
        )
        if resolved_query is not None:
            content = json.loads(packet.content)
            content["retrieval_query"] = resolved_query
            packet = replace(
                packet, content=json.dumps(content, ensure_ascii=False, separators=(",", ":"))
            )
        packet = compact_chat_packet(packet)
        settings = self.settings
        system = CHAT_REVIEW_SYSTEM + (REVIEW_REPAIR if repair else "")
        input_tokens = generation_prompt_tokens(ChatReviewOutput, system, packet.content)
        input_limit = chat_review_input_limit(settings)
        if review_record is not None:
            review_record.update(
                prompt_version=CHAT_REVIEW_PROMPT_VERSION,
                projection_version=PROJECTION_VERSION,
                projection_status="preflight",
                input_tokens=input_tokens,
                input_limit=input_limit,
                max_output_tokens=settings.evidence_review_max_output_tokens,
                passage_count=len(packet.catalog),
                source_count=len({ref.source_id for ref in packet.catalog.values()}),
                request_sha256=hashlib.sha256(packet.content.encode()).hexdigest(),
                claims=[
                    {"id": c.id, "text": c.text, "block_index": c.block_index}
                    for c in packet.claims
                ],
                wire_version="source-index-v1",
            )
        if input_tokens > input_limit:
            if review_record is not None:
                review_record["projection_status"] = "input_rejected"
            raise ProcessingError(
                "EVIDENCE_REVIEW_INPUT_LIMIT",
                f"Answer review exceeds its configured input budget "
                f"({input_tokens} > {input_limit} tokens).",
                diagnostics=AIFailureDetails(
                    reason="input_token_limit",
                    input_tokens=input_tokens,
                    max_output_tokens=settings.evidence_review_max_output_tokens,
                ),
            )
        reviewed = await self._generate(
            reservation,
            ChatReviewOutput,
            system,
            packet.content,
            settings.evidence_review_max_output_tokens,
            "chat.evidence_review",
            settings.openai_generation_timeout_seconds,
            provider=self.reviewer,
        )
        selected_quote_ids = {
            q for decision in reviewed.result.decisions for q in decision.quote_ids
        } | {q for group in reviewed.result.required_aspects for q in group.quote_ids}
        metadata = {
            "prompt_version": CHAT_REVIEW_PROMPT_VERSION,
            "projection_version": PROJECTION_VERSION,
            "model": reviewed.model,
            "scope_complete": reviewed.result.scope_complete,
            "answerability": reviewed.result.answerability,
            "relevant_claim_ids": reviewed.result.relevant_claim_ids,
            "draft_evidence_status": draft.evidence_status,
            "unanswered_parts": reviewed.result.unanswered_parts,
            "force_qualification_quotes": False,
            "qualification_policy": "required_aspect_source_bindings",
            "projection_status": "pending",
            "review_attempt": 2 if repair else 1,
            "review": reviewed.result.model_dump(mode="json"),
            "request_sha256": hashlib.sha256(packet.content.encode()).hexdigest(),
            "claims": [
                {"id": c.id, "text": c.text, "block_index": c.block_index} for c in packet.claims
            ],
            "quote_bindings": {
                q: {"source_id": str(ref.source_id), "quote": ref.quote}
                for q, ref in packet.catalog.items()
                if q in selected_quote_ids
            },
        }
        # Retain the bounded original audit in the turn context even when its
        # projection fails. Never log draft, question, source or provider bodies.
        if review_record is not None:
            review_record.update(metadata)
            metadata = review_record
        try:
            projection = project_chat_review(packet, reviewed.result)
            blocks = projection.payload["answer"]
            status = (
                "insufficient"
                if not blocks
                else "partial"
                if not projection.scope_complete
                else "answered"
            )
            answer = validate_answer(
                AnswerOutput.model_validate({"evidence_status": status, "blocks": blocks}),
                evidence,
                complete_scope=complete_scope,
            )
            answer = append_evidence_gaps(answer, reviewed.result.unanswered_parts)
        except ProcessingError as error:
            metadata["projection_status"] = "failed"
            metadata["projection_error"] = (
                error.diagnostics.as_dict() if error.diagnostics is not None else {}
            )
            raise
        metadata.pop("projection_error", None)
        metadata.update(
            projection_status="completed",
            scope_complete=projection.scope_complete,
            dropped_claim_ids=list(projection.dropped_claim_ids),
            appended_quote_ids=list(projection.appended_quote_ids),
        )
        return answer, metadata

    async def run(self, reservation: TurnReservation) -> Replay:
        started = time.monotonic()
        try:
            remaining = (reservation.deadline_at - utcnow()).total_seconds()
            async with asyncio.timeout(max(0, remaining)), self.semaphore:
                return await self._answer(reservation, started)
        except TimeoutError:
            return await self.store.fail(
                reservation,
                code="CHAT_DEADLINE",
                message="Question deadline exceeded. Send a new question to try again.",
                status=504,
            )
        except ConversationError as error:
            return await self.store.fail(
                reservation, code=error.code, message=error.message, status=error.status
            )
        except ProcessingError as error:
            return await self.store.fail(
                reservation,
                code=error.code,
                message=error.message,
                status=502,
                diagnostics=error.diagnostics,
            )
        except Exception as error:
            logging.getLogger(__name__).error(
                "Unexpected chat failure [%s] for turn %s",
                type(error).__name__,
                reservation.turn_id,
            )
            return await self.store.fail(
                reservation,
                code="CHAT_FAILED",
                message="Question failed. Inspect service health and send a new question to retry.",
                status=503,
            )

    async def _answer(self, reservation: TurnReservation, started: float) -> Replay:
        settings = self.settings
        history = await self.store.history(reservation)
        timings: dict[str, float] = {}
        phase = time.monotonic()
        planned = await self._generate(
            reservation,
            PlannerOutput,
            PLANNER_SYSTEM,
            json.dumps({"question": reservation.question, "history": history}, ensure_ascii=False),
            settings.planner_max_output_tokens,
            "chat.planner",
            10,
        )
        planner = planned.result
        timings["planner_ms"] = (time.monotonic() - phase) * 1000
        async with async_sessionmaker(self.engine, expire_on_commit=False)() as session:
            doc = await session.get(
                Document, reservation.document_id, options=[defer(Document.extracted_text)]
            )
            if doc is None or doc.status != "COMPLETED":
                raise ConversationError(
                    409, "DOCUMENT_NOT_READY", "Document is no longer ready for chat"
                )
            insights = (
                await session.get(DocumentInsights, doc.id)
                if planner.intent == "overview"
                else None
            )
        saved_analysis = insights.payload if insights else None
        phase = time.monotonic()
        if planner.intent == "overview":
            source_ids = []
            from uuid import UUID

            for block in (saved_analysis or {}).get("summary", []) + (saved_analysis or {}).get(
                "insights", []
            ):
                for source_id in block.get("source_ids", []):
                    identifier = UUID(source_id)
                    if identifier not in source_ids:
                        source_ids.append(identifier)
            candidates = await self.retriever.get_chunks(
                [doc.id], reservation.profile_id, source_ids
            )
            complete_scope = bool(
                insights
                and insights.covered_chunks == insights.total_chunks
                and (saved_analysis or {}).get("analysis_scope_complete", True)
            )
        else:
            remaining = await self.store.before_call(reservation)
            embed_started = time.monotonic()
            try:
                query = await self.embedder.embed_many(
                    [planner.query],
                    timeout_seconds=min(remaining, settings.openai_embedding_timeout_seconds),
                )
            except ProcessingError as error:
                details = error.diagnostics.as_dict() if error.diagnostics is not None else {}
                known_input = details.get("input_tokens")
                await self._sample(
                    reservation,
                    "embedding.query",
                    embed_started,
                    error=error.code,
                    input_tokens=known_input if isinstance(known_input, int) else None,
                )
                raise
            await self._sample(
                reservation, "embedding.query", embed_started, input_tokens=query.input_tokens
            )
            timings["embedding_ms"] = (time.monotonic() - embed_started) * 1000
            phase = time.monotonic()
            candidates = await self.retriever.search(
                [doc.id], reservation.profile_id, query.vectors[0], settings.retrieval_candidates
            )
            complete_scope = planner.intent != "exhaustive" or len(candidates) == doc.chunk_count
        timings["retrieval_ms"] = (time.monotonic() - phase) * 1000
        prepared = prepare_context(
            reservation.question,
            history,
            candidates,
            settings,
            intent=planner.intent,
            complete_scope=complete_scope,
            saved_analysis=saved_analysis,
            review_fits=(
                None
                if planner.intent == "factual"
                or (planner.intent == "exhaustive" and not complete_scope)
                else lambda sources: (
                    review_input_reservation(
                        answer_quote_catalog(sources), reservation.question, planner.query, settings
                    )
                    <= chat_review_input_limit(settings)
                )
            ),
            source_answer_query=planner.query if planner.intent == "factual" else None,
        )
        extractive_inventory = planner.intent == "exhaustive" and not prepared.complete_scope
        context = {
            "request_id": reservation.request_id,
            "answer_mode": (
                "source_selected"
                if planner.intent == "factual"
                else "partial_source_excerpts"
                if extractive_inventory
                else "synthesis"
            ),
            "payload": json.loads(prepared.content),
            "candidate_ids": [str(item.id) for item in candidates],
            "selected_ids": [str(item.id) for item in prepared.evidence],
            "input_token_estimate_with_repair_reserve": prepared.token_count,
        }
        await self.store.save_context(
            reservation,
            query=planner.query,
            intent=planner.intent,
            evidence_scope=prepared.evidence_scope,
            context=context,
            timings=timings,
        )
        phase = time.monotonic()
        if not prepared.evidence:
            answer = GroundedAnswer(INSUFFICIENT_ANSWER, "insufficient", [])
        elif planner.intent == "factual":
            context["source_answer"] = {}

            async def persist_source() -> None:
                timings["source_answer_ms"] = (time.monotonic() - phase) * 1000
                await self.store.save_context(
                    reservation,
                    query=planner.query,
                    intent=planner.intent,
                    evidence_scope=prepared.evidence_scope,
                    context=context,
                    timings=timings,
                )

            answer, context["source_answer"] = await self._source_answer(
                reservation,
                prepared.evidence,
                resolved_query=planner.query,
                complete_scope=prepared.complete_scope,
                source_record=context["source_answer"],
                persist=persist_source,
            )
        else:
            for repair in range(2):
                try:
                    schema = (
                        ExhaustiveSelectionOutput if extractive_inventory else AnswerSelectionOutput
                    )
                    system = (
                        EXHAUSTIVE_SYSTEM
                        + (
                            (
                                "\nPrevious selection invalid. "
                                "Select only unique supplied quote_ids; no prose."
                            )
                            if repair
                            else ""
                        )
                        if extractive_inventory
                        else ANSWER_SYSTEM + (ANSWER_REPAIR if repair else "")
                    )
                    generated = await self._generate(
                        reservation,
                        schema,
                        system,
                        prepared.content,
                        settings.chat_max_output_tokens,
                        "chat.answer",
                        settings.openai_generation_timeout_seconds,
                        provider=self.answerer,
                    )
                    selection = (
                        exhaustive_selection_to_answer(
                            ExhaustiveSelectionOutput.model_validate(generated.result),
                            prepared.evidence,
                        )
                        if extractive_inventory
                        else AnswerSelectionOutput.model_validate(generated.result)
                    )
                    logging.getLogger(__name__).info(
                        "Chat answer generated turn_id=%s document_id=%s repair=%s "
                        "evidence_status=%s block_count=%s quote_selection_count=%s",
                        reservation.turn_id,
                        reservation.document_id,
                        repair,
                        selection.evidence_status,
                        len(selection.blocks),
                        sum(len(block.quote_ids) for block in selection.blocks),
                    )
                    draft = resolve_answer_selection(selection, prepared.evidence)
                    answer = validate_answer(
                        draft,
                        prepared.evidence,
                        complete_scope=prepared.complete_scope,
                    )
                    if extractive_inventory and answer.evidence_status == "partial":
                        answer = GroundedAnswer(
                            (
                                "Relevant source excerpts "
                                "(partial coverage, not a complete inventory):\n\n"
                            )
                            + answer.answer,
                            answer.evidence_status,
                            answer.citations,
                            answer.blocks,
                        )
                    logging.getLogger(__name__).info(
                        "Chat answer validated turn_id=%s document_id=%s repair=%s "
                        "evidence_status=%s retained_block_count=%s citation_count=%s",
                        reservation.turn_id,
                        reservation.document_id,
                        repair,
                        answer.evidence_status,
                        len(answer.blocks),
                        len(answer.citations),
                    )
                    break
                except ProcessingError as error:
                    reason = (
                        error.diagnostics.as_dict()["reason"]
                        if error.diagnostics is not None
                        else "source_validation_failed"
                    )
                    logging.getLogger(__name__).warning(
                        "Chat answer validation failed turn_id=%s document_id=%s repair=%s "
                        "reason=%s",
                        reservation.turn_id,
                        reservation.document_id,
                        repair,
                        reason,
                    )
                    if repair or error.code not in {
                        "ai_invalid_output",
                        "ai_output_truncated",
                        "ai_invalid_json",
                        "ai_schema_invalid",
                        "CHAT_INVALID_OUTPUT",
                    }:
                        raise
            if (
                not extractive_inventory
                and draft.blocks
                and draft.evidence_status != "insufficient"
            ):
                review_started = time.monotonic()
                context["evidence_review"] = {}
                try:
                    answer, context["evidence_review"] = await self._review_answer(
                        reservation,
                        draft,
                        prepared.evidence,
                        overview=planner.intent == "overview",
                        complete_scope=prepared.complete_scope,
                        review_record=context["evidence_review"],
                        allow_repair=not repair,
                        resolved_query=planner.query,
                    )
                finally:
                    timings["evidence_review_ms"] = (time.monotonic() - review_started) * 1000
                    await self.store.save_context(
                        reservation,
                        query=planner.query,
                        intent=planner.intent,
                        evidence_scope=prepared.evidence_scope,
                        context=context,
                        timings=timings,
                    )

        timings["answer_ms"] = (time.monotonic() - phase) * 1000
        timings["total_ms"] = (time.monotonic() - started) * 1000
        body = {
            "turn_id": str(reservation.turn_id),
            "message_id": str(reservation.assistant_message_id),
            "status": "COMPLETED",
            "answer": answer.answer,
            "evidence_status": answer.evidence_status,
            "citations": answer.citations,
            "evidence_scope": prepared.evidence_scope,
            "extraction_coverage": doc.extraction_coverage,
            "warnings": doc.warnings,
            "model": (
                context.get("source_answer", {}).get("model", settings.openai_source_answer_model)
                if planner.intent == "factual"
                else settings.openai_chat_model
            ),
            "prompt_version": (
                SOURCE_COMPOSITION_VERSION if planner.intent == "factual" else ANSWER_PROMPT_VERSION
            ),
            "planner_prompt_version": PLANNER_PROMPT_VERSION,
        }
        return await self.store.complete(reservation, body, timings=timings)
