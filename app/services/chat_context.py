"""Budget complete answer/repair prompts while preserving exact source excerpts."""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any
from uuid import UUID

import tiktoken
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.config import Settings
from app.domain.conversation import Evidence
from app.domain.documents import AIFailureDetails, Locator, ProcessingError
from app.prompts.analysis import generation_prompt_tokens
from app.prompts.chat import ANSWER_REPAIR, ANSWER_SYSTEM
from app.services.analysis import FactualBlock
from app.services.analysis_evidence import (
    QuoteReference,
    SelectionBlock,
    build_quote_catalog,
    iter_literal_passages,
)
from app.services.grounding import AnswerOutput, EvidenceStatus, _quote_windows


class AnswerSelectionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    evidence_status: EvidenceStatus
    blocks: list[SelectionBlock] = Field(max_length=8)


class ExhaustiveSelectionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    quote_ids: list[str] = Field(max_length=8)


def exhaustive_selection_to_answer(
    output: ExhaustiveSelectionOutput, evidence: Sequence[Evidence]
) -> AnswerSelectionOutput:
    """Restrict an incomplete inventory to selected literal source text."""
    catalog = answer_quote_catalog(evidence)
    reason = (
        "duplicate_quote_id"
        if len(set(output.quote_ids)) != len(output.quote_ids)
        else "unknown_quote_id"
        if any(identifier not in catalog for identifier in output.quote_ids)
        else None
    )
    if reason:
        raise ProcessingError(
            "CHAT_INVALID_OUTPUT",
            "Chat selected invalid evidence references.",
            diagnostics=AIFailureDetails(reason=reason),
        )
    return AnswerSelectionOutput(
        evidence_status="partial" if output.quote_ids else "insufficient",
        blocks=[
            SelectionBlock(text=catalog[identifier].quote, quote_ids=[identifier])
            for identifier in output.quote_ids
        ],
    )


def answer_quote_catalog(sources: Sequence[Evidence]) -> dict[str, QuoteReference]:
    return build_quote_catalog({source.id: (source.text,) for source in sources})


def resolve_answer_selection(
    output: AnswerSelectionOutput, evidence: Sequence[Evidence]
) -> AnswerOutput:
    """Restore exact source UUIDs/quotes; never repair or normalize model references."""
    catalog = answer_quote_catalog(evidence)
    blocks = []
    all_sources: set[UUID] = set()
    for block in output.blocks:
        reason = None
        if not block.text.strip():
            reason = "blank_factual_text"
        elif not block.quote_ids:
            reason = "missing_source_quote"
        elif len(set(block.quote_ids)) != len(block.quote_ids):
            reason = "duplicate_quote_id"
        elif any(identifier not in catalog for identifier in block.quote_ids):
            reason = "unknown_quote_id"
        if reason:
            raise ProcessingError(
                "CHAT_INVALID_OUTPUT",
                "Chat selected invalid evidence references.",
                diagnostics=AIFailureDetails(reason=reason),
            )
        references = [catalog[identifier] for identifier in block.quote_ids]
        all_sources.update(item.source_id for item in references)
        if len(all_sources) > 8:
            raise ProcessingError(
                "CHAT_INVALID_OUTPUT",
                "Chat selected too many evidence sources.",
                diagnostics=AIFailureDetails(reason="final_source_limit"),
            )
        blocks.append(
            FactualBlock.model_validate(
                {
                    "text": block.text,
                    "source_ids": list(dict.fromkeys(item.source_id for item in references)),
                    "support_quotes": [
                        {"source_id": item.source_id, "quote": item.quote} for item in references
                    ],
                }
            )
        )
    return AnswerOutput(evidence_status=output.evidence_status, blocks=blocks)


@dataclass(frozen=True)
class PreparedContext:
    content: str
    evidence: tuple[Evidence, ...]
    history: tuple[dict[str, str], ...]
    complete_scope: bool
    evidence_scope: str
    token_count: int


def _tokens(text: str) -> int:
    return len(tiktoken.get_encoding("o200k_base").encode(text, disallowed_special=()))


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _locators(source: Evidence) -> list[Locator]:
    start, end = source.text_offset, source.text_offset + len(source.text)
    spans = [
        span for span in source.source_spans if span.chunk_start < end and span.chunk_end > start
    ]
    if not spans:
        return []
    first, last = dict(spans[0].locator), dict(spans[-1].locator)
    return [first] if first == last else [first, last]


def _evidence_json(sources: Sequence[Evidence]) -> list[dict[str, Any]]:
    catalog = answer_quote_catalog(sources)
    identifiers = {
        (reference.source_id, reference.quote): identifier
        for identifier, reference in catalog.items()
    }
    return [
        {
            "source_id": str(source.id),
            "filename": source.filename,
            "locators": _locators(source),
            "passages": [
                {
                    "text": passage,
                    **({"quote_id": identifiers[(source.id, passage)]} if passage.strip() else {}),
                }
                for passage in iter_literal_passages(source.text)
            ],
        }
        for source in sources
    ]


def _evidence_tokens(sources: Sequence[Evidence]) -> int:
    return _tokens(_json(_evidence_json(sources)))


def _prefix(source: Evidence, fits: Callable[[Evidence], bool]) -> Evidence | None:
    """Search character prefixes; never decode a partial multibyte token into U+FFFD."""
    low, high, accepted = 1, len(source.text), None
    while low <= high:
        middle = (low + high) // 2
        candidate = replace(source, text=source.text[:middle])
        if fits(candidate):
            accepted, low = candidate, middle + 1
        else:
            high = middle - 1
    return accepted if accepted is not None and accepted.text.strip() and fits(accepted) else None


def _supported_window(
    source: Evidence,
    quotes: Sequence[str],
    fits: Callable[[Evidence], bool],
) -> Evidence | None:
    """Keep every quote, shrinking surrounding context before omitting its source."""
    try:
        windows = _quote_windows(source.text, quotes)
    except ProcessingError:
        return None
    if not windows:
        return None
    left = min(start for group in windows for start, _ in group.window.supports)
    right = max(end for group in windows for _, end in group.window.supports)
    left_padding = left - min(group.window.start for group in windows)
    right_padding = max(group.window.end for group in windows) - right

    def expanded(padding: int) -> Evidence:
        before = min(left_padding, padding // 2)
        after = min(right_padding, padding - before)
        before += min(left_padding - before, padding - before - after)
        return replace(
            source,
            text=source.text[left - before : right + after],
            text_offset=source.text_offset + left - before,
        )

    accepted = expanded(0)
    if not fits(accepted):
        return None
    low, high = 1, left_padding + right_padding
    while low <= high:
        middle = (low + high) // 2
        candidate = expanded(middle)
        if fits(candidate):
            accepted, low = candidate, middle + 1
        else:
            high = middle - 1
    return accepted if fits(accepted) else None


def _claims(
    saved: dict[str, Any] | None,
    candidates: Sequence[Evidence],
) -> tuple[dict[str, list[FactualBlock]], dict[UUID, list[str]], bool]:
    supplied = {source.id: source for source in candidates}
    retained: dict[str, list[FactualBlock]] = {"summary": [], "insights": []}
    accepted_quotes: dict[UUID, list[str]] = {}
    omitted = False
    for field, limit in (("summary", 8), ("insights", 5)):
        values = saved.get(field, []) if saved is not None else []
        if not isinstance(values, list):
            omitted = True
            continue
        omitted |= len(values) > limit
        for value in values[:limit]:
            try:
                block = FactualBlock.model_validate(value)
            except (ValidationError, ValueError, TypeError):
                omitted = True
                continue
            ids = set(block.source_ids)
            if (
                not block.text.strip()
                or len(ids) != len(block.source_ids)
                or not ids.issubset(supplied)
                or ids != {support.source_id for support in block.support_quotes}
                or any(not support.quote.strip() for support in block.support_quotes)
            ):
                omitted = True
                continue
            proposed = {source: list(accepted_quotes.get(source, [])) for source in ids}
            for support in block.support_quotes:
                proposed[support.source_id].append(support.quote)
            fits = all(
                quote in supplied[source].text
                for source, quotes in proposed.items()
                for quote in quotes
            )
            if not fits:
                omitted = True
                continue
            retained[field].append(block)
            accepted_quotes.update(proposed)
    return retained, accepted_quotes, omitted


def _retain_claims(
    claims: dict[str, list[FactualBlock]],
    sources: Sequence[Evidence],
) -> tuple[dict[str, list[FactualBlock]], bool]:
    supplied = {source.id: source.text for source in sources}
    retained: dict[str, list[FactualBlock]] = {"summary": [], "insights": []}
    omitted = False
    for field, blocks in claims.items():
        for block in blocks:
            if all(source in supplied for source in block.source_ids) and all(
                support.quote in supplied.get(support.source_id, "")
                for support in block.support_quotes
            ):
                retained[field].append(block)
            else:
                omitted = True
    return retained, omitted


def prepare_context(
    question: str,
    history: Sequence[dict[str, str]],
    candidates: Sequence[Evidence],
    settings: Settings,
    *,
    intent: str,
    complete_scope: bool,
    saved_analysis: dict[str, Any] | None = None,
    review_fits: Callable[[Sequence[Evidence]], bool] | None = None,
) -> PreparedContext:
    """Keep question/instructions intact and reserve an unchanged-evidence repair."""
    if (
        intent not in {"factual", "overview", "exhaustive"}
        or len(question) > settings.question_max_chars
        or _tokens(question) > settings.question_max_tokens
        or len({source.document_id for source in candidates}) > 1
        or len({source.profile_id for source in candidates}) > 1
        or any(source.text_offset < 0 for source in candidates)
    ):
        raise ProcessingError(
            "CHAT_INPUT_LIMIT", "Question or source context cannot be supplied safely."
        )
    retained_history = [
        dict(message) for message in history[-2 * settings.chat_history_max_turns :]
    ]
    # TurnStore supplies ordered completed user/assistant pairs. Both history
    # and full-prompt trimming must preserve each answer's original question.
    while retained_history and _tokens(_json(retained_history)) > settings.chat_history_max_tokens:
        del retained_history[:2]
    unique: list[Evidence] = []
    seen_ids: set[UUID] = set()
    seen_text: set[str] = set()
    for source in candidates:
        if source.id in seen_ids or source.text in seen_text or not source.text.strip():
            continue
        seen_ids.add(source.id)
        seen_text.add(source.text)
        unique.append(source)
    scope = complete_scope
    if intent == "exhaustive" and len({source.id for source in candidates}) != len(unique):
        scope = False
    claims: dict[str, list[FactualBlock]] = {"summary": [], "insights": []}
    quotes: dict[UUID, list[str]] = {}
    if intent == "overview":
        claims, quotes, omitted = _claims(saved_analysis, unique)
        scope &= not omitted
        unique = [source for source in unique if source.id in quotes]
    selected: list[Evidence] = []
    for source in unique:
        if len(selected) >= min(8, settings.retrieval_max_chunks):
            break

        def fits(value: Evidence) -> bool:
            return _evidence_tokens([*selected, value]) <= settings.retrieval_max_tokens

        if intent == "overview":
            fitted = _supported_window(source, quotes[source.id], fits)
        else:
            fitted = source if fits(source) else _prefix(source, fits)
        if fitted is not None:
            selected.append(fitted)
            if intent == "exhaustive" and fitted.text != source.text:
                scope = False
    if intent == "exhaustive" and len(selected) != len(unique):
        scope = False

    def synchronize() -> None:
        nonlocal selected, claims, scope
        if intent == "overview":
            claims, omitted = _retain_claims(claims, selected)
            scope &= not omitted
            used = {
                source
                for values in claims.values()
                for block in values
                for source in block.source_ids
            }
            selected = [source for source in selected if source.id in used]
        if not selected:
            claims = {"summary": [], "insights": []}
            scope = False

    def render(sources: Sequence[Evidence], messages: Sequence[dict[str, str]]) -> str:
        payload: dict[str, Any] = {
            "question": question,
            "history": list(messages),
            "intent": intent,
            "complete_scope": scope,
            "evidence_scope": "document_summary" if intent == "overview" else "retrieved_excerpts",
            "evidence": _evidence_json(sources),
        }
        if intent == "overview" and any(claims.values()):
            payload["saved_analysis"] = {
                key: [block.model_dump(mode="json") for block in values]
                for key, values in claims.items()
            }
        return _json(payload)

    def prompt_size(sources: Sequence[Evidence], messages: Sequence[dict[str, str]]) -> int:
        return generation_prompt_tokens(
            AnswerSelectionOutput, ANSWER_SYSTEM + ANSWER_REPAIR, render(sources, messages)
        )

    def fits_both(sources: Sequence[Evidence], messages: Sequence[dict[str, str]]) -> bool:
        return prompt_size(sources, messages) <= settings.chat_prompt_max_tokens and (
            review_fits is None or review_fits(sources)
        )

    synchronize()
    while not fits_both(selected, retained_history):
        if retained_history:
            del retained_history[:2]
            continue
        if not selected:
            raise ProcessingError(
                "CHAT_INPUT_LIMIT", "Required question and instructions exceed chat input limits."
            )
        scope = False
        if len(selected) > 1:
            selected.pop()
        else:
            source = selected[0]

            def fits_prompt(value: Evidence) -> bool:
                return fits_both([value], ())

            if intent == "overview":
                needed = [
                    support.quote
                    for values in claims.values()
                    for block in values
                    for support in block.support_quotes
                    if support.source_id == source.id
                ]
                fitted = _supported_window(source, needed, fits_prompt)
            else:
                fitted = _prefix(source, fits_prompt)
            selected = [fitted] if fitted is not None else []
        synchronize()
    content = render(selected, retained_history)
    return PreparedContext(
        content=content,
        evidence=tuple(selected),
        history=tuple(retained_history),
        complete_scope=scope,
        evidence_scope="document_summary" if intent == "overview" else "retrieved_excerpts",
        token_count=generation_prompt_tokens(
            AnswerSelectionOutput, ANSWER_SYSTEM + ANSWER_REPAIR, content
        ),
    )
