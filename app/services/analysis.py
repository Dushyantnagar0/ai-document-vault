"""Bounded source-preserving analysis of every accepted extracted chunk."""

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, TypeVar
from uuid import UUID

import tiktoken
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.config import Settings
from app.domain.documents import (
    AIFailureDetails,
    AnalysisResult,
    BeforeCall,
    Chunk,
    Generated,
    GenerationProvider,
    ProcessingError,
)
from app.prompts.analysis import (
    ANALYSIS_PROMPT_VERSION,
    ANALYSIS_SYSTEM,
    REPAIR_REASONS,
    analysis_budget_instruction,
    analysis_repair_instruction,
    generation_prompt_tokens,
)
from app.services.analysis_evidence import (
    MAX_FINAL_SUMMARY_BLOCKS,
    QUOTE_CATALOG_VERSION,
    QuoteReference,
    SelectionOutput,
    build_quote_catalog,
    selection_to_payload,
)
from app.services.analysis_review import (
    ANALYSIS_REVIEW_PROMPT_VERSION,
    ANALYSIS_REVIEW_REPAIR,
    ANALYSIS_REVIEW_SYSTEM,
    INSIGHT_VALUE_PROJECTION_VERSION,
    AnalysisReviewOutput,
    project_analysis_review,
)
from app.services.evidence_review import (
    PROJECTION_VERSION,
    ReviewPacket,
    prepare_review,
)
from app.services.insight_generation import (
    INSIGHT_PROMPT_VERSION,
    INSIGHT_SYSTEM,
    InsightCandidates,
    candidate_content,
    generate_candidates,
)
from app.services.prompt_evidence import COMPACT_EVIDENCE_VERSION, compact_evidence

T = TypeVar("T", bound=BaseModel)

MAX_REDUCTION_DEPTH = 8
REPAIRABLE_CODES = {
    "ai_invalid_output",
    "ai_output_truncated",
    "ai_invalid_json",
    "ai_schema_invalid",
    "analysis_citation_invalid",
    "analysis_output_budget",
}


class AnalysisCitationError(ProcessingError):
    """Safe reason codes distinguish validation failures without retaining raw output."""

    def __init__(self, reason: str) -> None:
        self.reason = reason if reason in REPAIR_REASONS else "source_validation_failed"
        super().__init__(
            "analysis_citation_invalid",
            f"Analysis source validation failed ({self.reason}).",
            diagnostics=AIFailureDetails(reason=self.reason),
        )


def _repair_reason(error: ProcessingError) -> str:
    if isinstance(error, AnalysisCitationError):
        return error.reason
    reason = getattr(error, "reason", None)
    if error.code == "analysis_citation_invalid" and reason in REPAIR_REASONS:
        return str(reason)
    if error.diagnostics is not None:
        reason = error.diagnostics.as_dict()["reason"]
        if reason == "unknown_quote_id":
            return "unknown_quote_ids"
        if reason == "duplicate_quote_id":
            return "duplicate_quote_ids"
        if reason in REPAIR_REASONS:
            return str(reason)
        if reason == "output_token_limit":
            return "output_budget_exceeded"
        if reason == "invalid_json":
            return "invalid_json"
    return {
        "ai_invalid_output": "schema_invalid",
        "ai_schema_invalid": "schema_invalid",
        "ai_invalid_json": "invalid_json",
        "ai_output_truncated": "output_budget_exceeded",
        "analysis_output_budget": "output_budget_exceeded",
    }.get(error.code, "source_validation_failed")


class SourceQuote(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    source_id: UUID
    quote: str = Field(min_length=1, max_length=300)


class FactualBlock(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    text: str = Field(min_length=1, max_length=4000)
    source_ids: list[UUID] = Field(min_length=1, max_length=8)
    support_quotes: list[SourceQuote] = Field(min_length=1, max_length=8)


class AnalysisOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    summary: list[FactualBlock] = Field(min_length=1, max_length=MAX_FINAL_SUMMARY_BLOCKS)
    insights: list[FactualBlock] = Field(max_length=5)


@dataclass(frozen=True)
class _Node:
    item: dict[str, Any]
    covered: tuple[UUID, ...]
    evidence: dict[UUID, tuple[str, ...]]


def _evidence(nodes: Sequence[_Node]) -> dict[UUID, tuple[str, ...]]:
    evidence: dict[UUID, tuple[str, ...]] = {}
    for node in nodes:
        for source, excerpts in node.evidence.items():
            evidence[source] = (*evidence.get(source, ()), *excerpts)
    return evidence


def _content(stage: str, nodes: Sequence[_Node]) -> str:
    evidence = _evidence(nodes)
    catalog = build_quote_catalog(evidence)
    source_groups = {source: f"s{index}" for index, source in enumerate(evidence, start=1)}
    request: dict[str, Any] = {
        "stage": stage,
        "evidence": [
            {
                "quote_id": quote_id,
                "source_group": source_groups[reference.source_id],
                "text": reference.quote,
            }
            for quote_id, reference in catalog.items()
        ],
    }
    drafts = [
        block["text"]
        for node in nodes
        if "analysis" in node.item
        for name in ("summary", "insights")
        for block in node.item["analysis"][name]
    ]
    if drafts:
        request["draft_summaries"] = drafts
    return json.dumps(
        request,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _writing_units(text: str) -> tuple[str, ...]:
    """Estimate writing substance only; never remove source evidence or its offsets."""
    return tuple(
        dict.fromkeys(
            normalized
            for part in re.split(r"(?<=[.!?。！？])\s+|\n+", text)
            if (normalized := " ".join(part.split()))
        )
    )


def _tokens(value: str) -> int:
    return len(tiktoken.get_encoding("o200k_base").encode(value, disallowed_special=()))


def quality_input_limit(settings: Settings, max_output_tokens: int) -> int:
    """Intersect the existing application, provider-input and context budgets."""
    return max(
        0,
        min(
            settings.analysis_group_max_tokens
            + settings.summary_max_output_tokens
            + settings.insight_max_visible_tokens
            + settings.INSTRUCTION_RESERVE_TOKENS,
            settings.CHAT_MODEL_MAX_INPUT_TOKENS,
            settings.CHAT_CONTEXT_TOKENS - max_output_tokens,
        ),
    )


def prepare_analysis_review(
    draft: dict[str, Any], catalog: dict[str, QuoteReference]
) -> ReviewPacket:
    """Compact only the wire packet; retain the original catalog for projection."""
    packet = prepare_review(draft, catalog)
    content = json.loads(packet.content)
    content.update(compact_evidence(catalog))
    return replace(packet, content=json.dumps(content, ensure_ascii=False, separators=(",", ":")))


def quality_input_reservations(
    catalog: dict[str, QuoteReference], settings: Settings
) -> tuple[int, int]:
    """Reserve complete downstream prompts before committing to final synthesis.

    Include all original passages, source identities, instructions and schemas,
    all 64 permitted claim frames and the possible review repair. Reserve two
    generated-output budgets for prose plus encoding/framing expansion, as for
    chat review. This conservative estimate is not a tokenizer upper-bound proof;
    actual assembled prompts still pass the authoritative guard before any call.
    """
    insight_tokens = (
        generation_prompt_tokens(
            InsightCandidates,
            INSIGHT_SYSTEM,
            candidate_content([{"text": ""} for _ in range(8)], catalog),
        )
        + 2 * settings.summary_max_output_tokens
    )
    review_content = {
        "question": None,
        "claims": [
            {"claim_id": f"s8.{index + 1}", "kind": "summary", "block_index": 7, "text": ""}
            for index in range(64)
        ],
        **compact_evidence(catalog),
    }
    review_tokens = generation_prompt_tokens(
        AnalysisReviewOutput,
        ANALYSIS_REVIEW_SYSTEM + ANALYSIS_REVIEW_REPAIR,
        json.dumps(review_content, ensure_ascii=False, separators=(",", ":")),
    ) + 2 * (settings.summary_max_output_tokens + settings.insight_max_visible_tokens)
    return insight_tokens, review_tokens


def _record_failed_call(call: dict[str, Any], error: ProcessingError, output_budget: int) -> None:
    """Preserve known usage without retaining failed output or provider exception text."""
    call.update(outcome="failed", error_code=error.code)
    if error.diagnostics is not None:
        error.diagnostics = replace(
            error.diagnostics,
            input_tokens=(
                error.diagnostics.input_tokens
                if error.diagnostics.input_tokens is not None
                else call.get("input_tokens")
            ),
            output_tokens=(
                error.diagnostics.output_tokens
                if error.diagnostics.output_tokens is not None
                else call.get("output_tokens")
            ),
            max_output_tokens=(
                error.diagnostics.max_output_tokens
                if error.diagnostics.max_output_tokens is not None
                else output_budget
            ),
        )
        diagnostic = error.diagnostics.as_dict()
        call["diagnostics"] = diagnostic
        for token_key in ("input_tokens", "output_tokens"):
            if diagnostic[token_key] is not None:
                call[token_key] = diagnostic[token_key]


def _groups(nodes: Sequence[_Node], stage: str, budget: int) -> list[list[_Node]]:
    groups: list[list[_Node]] = []
    current: list[_Node] = []
    for node in nodes:
        if _tokens(_content(stage, [*current, node])) <= budget:
            current.append(node)
            continue
        if not current or _tokens(_content(stage, [node])) > budget:
            raise ProcessingError(
                "analysis_input_limit", "A source item cannot fit analysis input limits."
            )
        groups.append(current)
        current = [node]
    if current:
        groups.append(current)
    return groups


def _validate_sources(
    result: AnalysisOutput,
    evidence: dict[UUID, tuple[str, ...]],
    originals: dict[UUID, str],
    *,
    final: bool,
) -> None:
    for block in [*result.summary, *result.insights]:
        ids = set(block.source_ids)
        if not block.text.strip():
            raise AnalysisCitationError("blank_factual_text")
        if len(ids) != len(block.source_ids):
            raise AnalysisCitationError("duplicate_source_ids")
        if not ids.issubset(evidence):
            raise AnalysisCitationError("unknown_source_ids")
        quote_ids = {quote.source_id for quote in block.support_quotes}
        if ids - quote_ids:
            raise AnalysisCitationError("missing_source_quote")
        if quote_ids - ids:
            raise AnalysisCitationError("uncited_quote_source")
        for support in block.support_quotes:
            if not support.quote.strip():
                raise AnalysisCitationError("blank_support_quote")
            if support.quote not in originals[support.source_id]:
                raise AnalysisCitationError("quote_not_in_original")
            if not any(support.quote in excerpt for excerpt in evidence[support.source_id]):
                raise AnalysisCitationError("quote_not_in_supplied_evidence")


class _QualityGeneration:
    """One bounded final-stage request; no retry and no unreserved provider call."""

    def __init__(
        self,
        provider: GenerationProvider,
        settings: Settings,
        before_call: BeforeCall,
        calls: list[dict[str, Any]],
        stage: str,
    ) -> None:
        self.provider, self.settings, self.before_call = provider, settings, before_call
        self.calls, self.stage = calls, stage

    def generate(
        self,
        *,
        schema: type[T],
        system: str,
        content: str,
        max_output_tokens: int,
        timeout_seconds: float,
    ) -> Generated[T]:
        settings = self.settings
        prompt_tokens = generation_prompt_tokens(schema, system, content)
        if prompt_tokens > quality_input_limit(settings, max_output_tokens):
            raise ProcessingError(
                "analysis_prompt_limit", "Complete quality-review prompt exceeds its budget."
            )
        if len(self.calls) >= settings.max_document_ai_calls:
            raise ProcessingError("ai_call_budget", "Document AI-call budget is exhausted.")
        remaining = self.before_call()
        if not math.isfinite(remaining) or remaining <= 0:
            raise ProcessingError(
                "processing_deadline", "Processing deadline expired before review."
            )
        call: dict[str, Any] = {"stage": self.stage, "repair": False}
        self.calls.append(call)
        try:
            result = self.provider.generate(
                schema=schema,
                system=system,
                content=content,
                max_output_tokens=max_output_tokens,
                timeout_seconds=min(
                    remaining, timeout_seconds, settings.openai_generation_timeout_seconds
                ),
            )
            call.update(
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                model=result.model,
            )
            if result.model != settings.openai_chat_model:
                raise ProcessingError("ai_model_mismatch", "Analysis model profile does not match.")
            if result.output_tokens is not None and result.output_tokens > max_output_tokens:
                raise ProcessingError(
                    "analysis_output_budget", "Quality output exceeds its budget."
                )
        except ProcessingError as error:
            _record_failed_call(call, error, max_output_tokens)
            raise
        call["outcome"] = "completed"
        return result


def analyse_document(
    chunks: Sequence[Chunk],
    settings: Settings,
    generator: GenerationProvider,
    before_call: BeforeCall,
    *,
    insight_generator: GenerationProvider | None = None,
    reviewer: GenerationProvider | None = None,
) -> AnalysisResult:
    """Process every chunk, optionally refining the final draft with paired providers.

    Production supplies both quality providers; omission supports draft component checks.
    Coverage records participation, not semantic completeness or visual extraction.
    Every external attempt/repair reserves the persisted call/deadline budget.
    """
    if (insight_generator is None) != (reviewer is None):
        raise ProcessingError(
            "analysis_configuration_invalid", "Insight generation and review must be paired."
        )
    if (
        not chunks
        or len(chunks) > settings.max_document_chunks
        or len({chunk.id for chunk in chunks}) != len(chunks)
        or len({chunk.ordinal for chunk in chunks}) != len(chunks)
        or any(not chunk.text.strip() for chunk in chunks)
    ):
        raise ProcessingError(
            "analysis_input_invalid", "Analysis requires unique nonempty source chunks."
        )
    ordered = sorted(chunks, key=lambda chunk: chunk.ordinal)
    originals = {chunk.id: chunk.text for chunk in ordered}
    original_units = {source: _writing_units(text) for source, text in originals.items()}
    nodes = [
        _Node(
            item={},
            covered=(chunk.id,),
            evidence={chunk.id: (chunk.text,)},
        )
        for chunk in ordered
    ]
    lineage: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []

    def quality_fits(catalog: dict[str, QuoteReference]) -> bool:
        insight_tokens, review_tokens = quality_input_reservations(catalog, settings)
        return insight_tokens <= quality_input_limit(
            settings, settings.insight_max_output_tokens
        ) and review_tokens <= quality_input_limit(
            settings, settings.evidence_review_max_output_tokens
        )

    def final_fits(group: Sequence[_Node]) -> bool:
        return _tokens(_content("final", group)) <= settings.analysis_group_max_tokens and (
            reviewer is None or quality_fits(build_quote_catalog(_evidence(group)))
        )

    if reviewer is not None and not quality_fits({}):
        raise ProcessingError(
            "analysis_prompt_limit", "Quality output reservations cannot fit the input budget."
        )

    def generate(group: Sequence[_Node], stage: str, *, final: bool = False) -> _Node:
        content = _content(stage, group)
        if _tokens(content) > settings.analysis_group_max_tokens:
            raise ProcessingError(
                "analysis_input_limit", "Analysis input exceeds its group budget."
            )
        evidence = _evidence(group)
        catalog = build_quote_catalog(evidence)
        covered = tuple(dict.fromkeys(source for node in group for source in node.covered))
        evidence_tokens = _tokens(
            "\n".join(dict.fromkeys(unit for source in covered for unit in original_units[source]))
        )
        output_budget = (
            settings.summary_max_output_tokens if final else settings.analysis_max_output_tokens
        )
        repair_reason = "source_validation_failed"
        for repair in range(2):
            system = (
                ANALYSIS_SYSTEM
                + analysis_budget_instruction(
                    output_budget,
                    evidence_tokens=evidence_tokens,
                    final=final,
                    repair=bool(repair),
                )
                + (analysis_repair_instruction(repair_reason) if repair else "")
            )
            prompt_tokens = generation_prompt_tokens(SelectionOutput, system, content)
            if (
                prompt_tokens
                > settings.analysis_group_max_tokens + settings.INSTRUCTION_RESERVE_TOKENS
                or prompt_tokens > settings.CHAT_MODEL_MAX_INPUT_TOKENS
                or prompt_tokens + output_budget > settings.CHAT_CONTEXT_TOKENS
            ):
                raise ProcessingError(
                    "analysis_prompt_limit", "Complete analysis prompt exceeds its budget."
                )
            if len(calls) >= settings.max_document_ai_calls:
                raise ProcessingError("ai_call_budget", "Document AI-call budget is exhausted.")
            remaining = before_call()
            if not math.isfinite(remaining) or remaining <= 0:
                raise ProcessingError(
                    "processing_deadline", "Processing deadline expired before analysis."
                )
            call: dict[str, Any] = {"stage": stage, "repair": bool(repair)}
            calls.append(call)
            try:
                response = generator.generate(
                    schema=SelectionOutput,
                    system=system,
                    content=content,
                    max_output_tokens=output_budget,
                    timeout_seconds=min(remaining, settings.openai_generation_timeout_seconds),
                )
                call.update(
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    model=response.model,
                )
                if response.model != settings.openai_chat_model:
                    raise ProcessingError(
                        "ai_model_mismatch", "Analysis model profile does not match."
                    )
                try:
                    selection = SelectionOutput.model_validate(response.result)
                except ValidationError:
                    raise ProcessingError(
                        "ai_schema_invalid",
                        "Analysis schema validation failed.",
                        diagnostics=AIFailureDetails(
                            reason="schema_validation",
                            input_tokens=response.input_tokens,
                            output_tokens=response.output_tokens,
                            max_output_tokens=output_budget,
                        ),
                    ) from None
                if _tokens(selection.model_dump_json()) > output_budget or (
                    response.output_tokens is not None and response.output_tokens > output_budget
                ):
                    raise ProcessingError(
                        "analysis_output_budget", "Analysis output exceeds its budget."
                    )
                output = AnalysisOutput.model_validate(selection_to_payload(selection, catalog))
                _validate_sources(output, evidence, originals, final=final)
            except ProcessingError as exc:
                _record_failed_call(call, exc, output_budget)
                if exc.code in REPAIRABLE_CODES:
                    repair_reason = _repair_reason(exc)
                    call["validation_reason"] = repair_reason
                if repair == 0 and exc.code in REPAIRABLE_CODES:
                    continue
                raise
            call["outcome"] = "completed"
            item = output.model_dump(mode="json")
            quotes: dict[UUID, tuple[str, ...]] = {}
            for block in [*output.summary, *output.insights]:
                for support in block.support_quotes:
                    quotes[support.source_id] = (*quotes.get(support.source_id, ()), support.quote)
            lineage.append(
                {
                    "stage": stage,
                    "covered_chunk_ids": [str(source) for source in covered],
                    "cited_chunk_ids": [str(source) for source in quotes],
                    "call_number": len(calls),
                }
            )
            return _Node(item={"analysis": item}, covered=covered, evidence=quotes)
        raise AssertionError("Bounded repair loop must return or raise")

    if final_fits(nodes):
        final_evidence = _evidence(nodes)
        final = generate(nodes, "final", final=True)
    else:
        nodes = [
            generate(group, "section")
            for group in _groups(nodes, "section", settings.analysis_group_max_tokens)
        ]
        for _depth in range(MAX_REDUCTION_DEPTH):
            if final_fits(nodes):
                break
            previous_size = _tokens(_content("reduce", nodes))
            reduced = [
                generate(group, "reduce")
                for group in _groups(nodes, "reduce", settings.analysis_group_max_tokens)
            ]
            if _tokens(_content("reduce", reduced)) >= previous_size:
                raise ProcessingError(
                    "analysis_reduction_stalled", "Analysis reduction did not shrink."
                )
            nodes = reduced
        else:
            if not final_fits(nodes):
                raise ProcessingError(
                    "analysis_reduction_limit", "Analysis reduction depth was exhausted."
                )
        final_evidence = _evidence(nodes)
        final = generate(nodes, "final", final=True)
    expected = tuple(chunk.id for chunk in ordered)
    if set(final.covered) != set(expected):
        raise ProcessingError(
            "analysis_coverage_invalid", "Analysis did not process every extracted chunk."
        )
    payload = dict(final.item["analysis"])
    if insight_generator is not None and reviewer is not None:
        catalog = build_quote_catalog(final_evidence)
        active_stage = "insights"
        try:
            candidates = generate_candidates(
                payload["summary"],
                catalog,
                _QualityGeneration(insight_generator, settings, before_call, calls, "insights"),
                # The proxy reserves immediately before the actual outbound call.
                lambda: settings.openai_generation_timeout_seconds,
                max_visible_tokens=settings.insight_max_visible_tokens,
                max_completion_tokens=settings.insight_max_output_tokens,
            )
            active_stage = "preparing_review"
            draft = {
                "summary": payload["summary"],
                "insights": [block.model_dump(mode="json") for block in candidates.result.insights],
            }
            packet = prepare_analysis_review(draft, catalog)
            active_stage = "evidence_review"
            for review_attempt in range(2):
                previous_calls = len(calls)
                try:
                    response = _QualityGeneration(
                        reviewer, settings, before_call, calls, "evidence_review"
                    ).generate(
                        schema=AnalysisReviewOutput,
                        system=ANALYSIS_REVIEW_SYSTEM
                        + (ANALYSIS_REVIEW_REPAIR if review_attempt else ""),
                        content=packet.content,
                        max_output_tokens=settings.evidence_review_max_output_tokens,
                        timeout_seconds=settings.openai_generation_timeout_seconds,
                    )
                    calls[-1]["repair"] = bool(review_attempt)
                    reviewed = AnalysisReviewOutput.model_validate(response.result)
                    projection = project_analysis_review(packet, reviewed)
                    break
                except (ProcessingError, ValidationError) as failure:
                    error = (
                        failure
                        if isinstance(failure, ProcessingError)
                        else ProcessingError(
                            "ai_schema_invalid",
                            "Quality result failed schema validation.",
                            diagnostics=AIFailureDetails(reason="schema_validation"),
                        )
                    )
                    if len(calls) > previous_calls:
                        calls[-1]["repair"] = bool(review_attempt)
                        if calls[-1].get("outcome") == "completed":
                            _record_failed_call(
                                calls[-1], error, settings.evidence_review_max_output_tokens
                            )
                    if review_attempt or error.code not in {
                        "EVIDENCE_REVIEW_INVALID",
                        "ai_invalid_output",
                        "ai_output_truncated",
                        "ai_invalid_json",
                        "ai_schema_invalid",
                    }:
                        raise error from None
            output = AnalysisOutput.model_validate(projection.payload)
            _validate_sources(output, final_evidence, originals, final=True)
            visible_tokens = _tokens(
                json.dumps(
                    {
                        name: [block.text for block in getattr(output, name)]
                        for name in ("summary", "insights")
                    },
                    ensure_ascii=False,
                )
            )
            if (
                visible_tokens
                > settings.summary_max_output_tokens + settings.insight_max_visible_tokens
            ):
                raise ProcessingError(
                    "analysis_output_budget", "Reviewed analysis text exceeds its visible budget."
                )
        except (ProcessingError, ValidationError) as failure:
            error = (
                failure
                if isinstance(failure, ProcessingError)
                else ProcessingError(
                    "ai_schema_invalid", "Quality result failed schema validation."
                )
            )
            if (
                calls
                and calls[-1].get("stage") == active_stage
                and calls[-1].get("outcome") == "completed"
            ):
                _record_failed_call(
                    calls[-1],
                    error,
                    settings.evidence_review_max_output_tokens
                    if calls[-1]["stage"] == "evidence_review"
                    else settings.insight_max_output_tokens,
                )
            raise error from None
        payload = output.model_dump(mode="json")
        selected_ids = {key for decision in reviewed.decisions for key in decision.quote_ids} | {
            key for group in reviewed.required_qualifications for key in group.quote_ids
        }
        payload.update(
            analysis_scope_complete=projection.scope_complete,
            analysis_warnings=[]
            if projection.scope_complete
            else [
                {
                    "code": "ANALYSIS_SCOPE_PARTIAL",
                    "message": "Some analysis claims or requested scope could not be supported.",
                }
            ],
            evidence_review={
                "prompt_version": ANALYSIS_REVIEW_PROMPT_VERSION,
                "projection_version": PROJECTION_VERSION,
                "insight_value_projection_version": INSIGHT_VALUE_PROJECTION_VERSION,
                "insight_prompt_version": INSIGHT_PROMPT_VERSION,
                "quote_catalog_version": QUOTE_CATALOG_VERSION,
                "evidence_format_version": COMPACT_EVIDENCE_VERSION,
                "evidence_basis": "original_passages_supplied_to_final_synthesis",
                "catalog_sha256": hashlib.sha256(
                    json.dumps(
                        {
                            key: {"source_id": str(ref.source_id), "quote": ref.quote}
                            for key, ref in catalog.items()
                        },
                        sort_keys=True,
                        ensure_ascii=False,
                    ).encode()
                ).hexdigest(),
                "claims": [
                    {
                        "claim_id": claim.id,
                        "kind": claim.kind,
                        "block_index": claim.block_index,
                        "text": claim.text,
                    }
                    for claim in packet.claims
                ],
                "decisions": reviewed.model_dump(mode="json"),
                "quote_bindings": {
                    key: {"source_id": str(catalog[key].source_id), "quote": catalog[key].quote}
                    for key in catalog
                    if key in selected_ids
                },
                "dropped_claim_ids": list(projection.dropped_claim_ids),
                "appended_quote_ids": list(projection.appended_quote_ids),
                "scope_complete": projection.scope_complete,
            },
        )
    payload.update(
        analysis_coverage=1.0,
        covered_extracted_chunks=len(expected),
        total_extracted_chunks=len(expected),
        covered_chunk_ids=[str(source) for source in expected],
        coverage_basis="all_accepted_extracted_chunks_processed",
        prompt_version=ANALYSIS_PROMPT_VERSION,
        model=settings.openai_chat_model,
        lineage=lineage,
        usage={
            "calls": len(calls),
            "known_input_tokens": sum(call.get("input_tokens") or 0 for call in calls),
            "known_output_tokens": sum(call.get("output_tokens") or 0 for call in calls),
            "all_calls_reported_usage": all(
                call.get("input_tokens") is not None and call.get("output_tokens") is not None
                for call in calls
            ),
            "call_history": calls,
        },
    )
    return AnalysisResult(
        payload=payload,
        model=settings.openai_chat_model,
        prompt_version=ANALYSIS_PROMPT_VERSION,
        covered_chunk_ids=expected,
    )
