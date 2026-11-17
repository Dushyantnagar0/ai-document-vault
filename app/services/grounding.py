"""Validate answer provenance; semantic source support still needs live evaluation."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.domain.conversation import Evidence
from app.domain.documents import AIFailureDetails, Locator, ProcessingError
from app.services.analysis import FactualBlock

EvidenceStatus = Literal["answered", "partial", "insufficient"]
INSUFFICIENT_ANSWER = "I cannot find the answer to this question in the uploaded document."
MAX_SOURCES = 8
MAX_CITATIONS = 8
MAX_EXCERPT_CHARS = 1200


class PlannerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    query: str = Field(min_length=1, max_length=4000)
    intent: Literal["factual", "overview", "exhaustive"]

    @field_validator("query")
    @classmethod
    def meaningful_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("The retrieval query must contain text.")
        return value.strip()


class AnswerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    evidence_status: EvidenceStatus
    blocks: list[FactualBlock] = Field(max_length=8)


class Citation(TypedDict):
    source_id: str
    document_id: str
    filename: str
    locators: list[Locator]
    excerpt: str


@dataclass(frozen=True)
class GroundedAnswer:
    answer: str
    evidence_status: EvidenceStatus
    citations: list[Citation]
    blocks: tuple[FactualBlock, ...] = ()


def append_evidence_gaps(answer: GroundedAnswer, unanswered_parts: Sequence[str]) -> GroundedAnswer:
    """Render scoped missing-request metadata separately from cited facts."""
    if answer.evidence_status != "partial" or not unanswered_parts:
        return answer
    gaps = "\n".join(f"- {part}" for part in unanswered_parts)
    return GroundedAnswer(
        answer.answer + "\n\nNot established by the supplied evidence:\n" + gaps,
        answer.evidence_status,
        answer.citations,
        answer.blocks,
    )


@dataclass(frozen=True)
class _Window:
    start: int
    end: int
    supports: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _QuoteWindow:
    quotes: tuple[str, ...]
    window: _Window


def _invalid(reason: str | None = None) -> ProcessingError:
    return ProcessingError(
        "CHAT_INVALID_OUTPUT",
        "Chat source references or supporting quotes are invalid.",
        diagnostics=AIFailureDetails(reason=reason) if reason is not None else None,
    )


def _support_window(text: str, quotes: Sequence[str]) -> _Window | None:
    """Find quote occurrences that fit one bounded excerpt using bounded memory.

    Advancing the leftmost occurrence avoids retaining a distant first occurrence
    when an identical quote also occurs beside the other supporting quotes.
    """
    needles = tuple(dict.fromkeys(quotes))
    if not needles:
        return None
    positions = [text.find(quote) for quote in needles]
    if any(position < 0 for position in positions):
        raise _invalid()
    while True:
        left = min(positions)
        right = max(
            position + len(quote) for position, quote in zip(positions, needles, strict=True)
        )
        if right - left <= MAX_EXCERPT_CHARS:
            length = min(MAX_EXCERPT_CHARS, len(text))
            start = max(0, left - (length - (right - left)) // 2)
            end = min(len(text), start + length)
            start = max(0, end - length)
            return _Window(
                start,
                end,
                tuple(
                    (position, position + len(quote))
                    for position, quote in zip(positions, needles, strict=True)
                ),
            )
        index = positions.index(left)
        following = text.find(needles[index], left + 1)
        if following < 0:
            return None
        positions[index] = following


def _locators(source: Evidence, window: _Window) -> list[Locator]:
    locators: list[Locator] = []
    for start, end in window.supports:
        for span in source.source_spans:
            if (
                span.chunk_start < source.text_offset + end
                and span.chunk_end > source.text_offset + start
                and span.locator not in locators
            ):
                locators.append(dict(span.locator))
    return locators


def _quote_windows(text: str, quotes: Sequence[str]) -> list[_QuoteWindow]:
    """Greedily group literal support into bounded contiguous display windows."""
    groups: list[_QuoteWindow] = []
    for quote in sorted(dict.fromkeys(quotes), key=text.find):
        for index, group in enumerate(groups):
            combined = (*group.quotes, quote)
            window = _support_window(text, combined)
            if window is not None:
                groups[index] = _QuoteWindow(combined, window)
                break
        else:
            window = _support_window(text, [quote])
            if window is None:
                raise _invalid("support_window_exceeded")
            groups.append(_QuoteWindow((quote,), window))
    return groups


def validate_answer(
    result: AnswerOutput,
    evidence: Sequence[Evidence],
    *,
    complete_scope: bool = True,
) -> GroundedAnswer:
    """Reject invented references; omit claims whose support cannot be displayed.

    V1 evidence must share a document/profile. Retrieval/orchestration enforces
    session membership because it owns the expected document ID.
    """
    if result.evidence_status == "insufficient" and result.blocks:
        raise _invalid("inconsistent_evidence_status")
    if not result.blocks:
        return GroundedAnswer(INSUFFICIENT_ANSWER, "insufficient", [])
    supplied = {source.id: source for source in evidence}
    if (
        len(supplied) != len(evidence)
        or len({source.document_id for source in evidence}) > 1
        or len({source.profile_id for source in evidence}) > 1
        or any(source.text_offset < 0 for source in evidence)
    ):
        raise _invalid()
    all_sources: set[UUID] = set()
    useful: list[FactualBlock] = []
    for block in result.blocks:
        if not block.text.strip():
            continue
        ids = set(block.source_ids)
        if (
            len(ids) != len(block.source_ids)
            or not ids.issubset(supplied)
            or ids != {support.source_id for support in block.support_quotes}
        ):
            raise _invalid()
        all_sources.update(ids)
        for support in block.support_quotes:
            if (
                not support.quote.strip()
                or len(support.quote) > 300
                or support.quote not in supplied[support.source_id].text
            ):
                raise _invalid()
        useful.append(block)
    if len(all_sources) > MAX_SOURCES:
        raise _invalid()

    retained: list[FactualBlock] = []
    accepted_quotes: dict[UUID, list[str]] = {}
    windows: dict[UUID, list[_QuoteWindow]] = {}
    omitted = len(useful) != len(result.blocks)
    for block in useful:
        proposed = {
            source_id: list(accepted_quotes.get(source_id, [])) for source_id in block.source_ids
        }
        for support in block.support_quotes:
            proposed[support.source_id].append(support.quote)
        proposed_windows = {
            source_id: _quote_windows(supplied[source_id].text, quotes)
            for source_id, quotes in proposed.items()
        }
        combined_windows = {**windows, **proposed_windows}
        if sum(len(groups) for groups in combined_windows.values()) > MAX_CITATIONS:
            omitted = True
            continue
        retained.append(block)
        accepted_quotes.update(proposed)
        windows = combined_windows
    if not retained:
        if useful:
            # Supplied support exists, but cannot fit the display contract. Give
            # orchestration its bounded repair instead of claiming it is absent.
            raise _invalid("support_window_exceeded")
        return GroundedAnswer(INSUFFICIENT_ANSWER, "insufficient", [])

    source_order = list(
        dict.fromkeys(source_id for block in retained for source_id in block.source_ids)
    )
    citations: list[Citation] = []
    references: dict[tuple[UUID, str], int] = {}
    for source_id in source_order:
        source = supplied[source_id]
        for group in windows[source_id]:
            window = group.window
            citations.append(
                {
                    "source_id": str(source.id),
                    "document_id": str(source.document_id),
                    "filename": source.filename,
                    "locators": _locators(source, window),
                    "excerpt": source.text[window.start : window.end],
                }
            )
            for quote in group.quotes:
                references[(source_id, quote)] = len(citations)
    answer = "\n\n".join(
        block.text.strip()
        + " "
        + " ".join(
            f"[{reference}]"
            for reference in dict.fromkeys(
                references[(source_id, support.quote)]
                for source_id in block.source_ids
                for support in block.support_quotes
                if support.source_id == source_id
            )
        )
        for block in retained
    )
    status: EvidenceStatus = (
        "answered"
        if complete_scope and not omitted and result.evidence_status == "answered"
        else "partial"
    )
    if omitted:
        answer += (
            "\n\nSome detail could not be included with its supporting sources; "
            "this answer covers only the supported points shown here."
        )
    if not complete_scope:
        answer += (
            "\n\nThe available excerpts do not establish complete coverage of all "
            "the material requested."
        )
    return GroundedAnswer(answer, status, citations, tuple(retained))
