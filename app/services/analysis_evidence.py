"""Build literal evidence passages; generated output selects IDs without rewriting quotes."""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.domain.documents import AIFailureDetails, ProcessingError

MAX_QUOTE_CHARACTERS = 300
QUOTE_CATALOG_VERSION = "literal-sentences-v2"


@dataclass(frozen=True)
class QuoteReference:
    source_id: UUID
    quote: str


class SelectionBlock(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    quote_ids: list[str] = Field(
        min_length=1,
        max_length=8,
        description=(
            "Select exact supporting passages FIRST; cover every material claim, "
            "including late facts."
        ),
    )
    text: str = Field(
        min_length=1,
        max_length=4000,
        description=(
            "Write only claims supported by selected quote_ids; split or shorten "
            "when support cannot fit."
        ),
    )


class SelectionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    insights: list[SelectionBlock] = Field(
        max_length=5,
        description=(
            "Select useful supported implications first; do not restate facts. "
            "Restated facts or disclaimers add nothing; return [] when none "
            "remains."
        ),
    )

    summary: list[SelectionBlock] = Field(min_length=1, max_length=8)


class SelectionReferenceError(ProcessingError):
    """Static repair reason; no original text, generated text or unknown IDs retained."""

    def __init__(self, reason: str = "invalid_quote_selection") -> None:
        self.reason = (
            reason
            if reason in {"unknown_quote_ids", "duplicate_quote_ids"}
            else "invalid_quote_selection"
        )
        diagnostic_reason = {
            "unknown_quote_ids": "unknown_quote_id",
            "duplicate_quote_ids": "duplicate_quote_id",
        }.get(self.reason, "missing_source_quote")
        super().__init__(
            "analysis_citation_invalid",
            "Analysis selected invalid evidence references.",
            diagnostics=AIFailureDetails(reason=diagnostic_reason),
        )


def _invalid_selection(reason: str = "invalid_quote_selection") -> SelectionReferenceError:
    return SelectionReferenceError(reason)


def iter_literal_passages(text: str) -> Iterator[str]:
    """Preserve complete sentences where possible, with exact literal character coverage.

    Sentence/paragraph boundaries are English-first heuristics, not linguistic proof.
    Long sentences fall back to the existing newline/whitespace character cap.
    """
    boundaries = [
        0,
        *(match.end() for match in re.finditer(r"(?<=[.!?。！？])\s+|\n[ \t\r]*\n", text)),
        len(text),
    ]
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        start = left
        while start < right:
            end = min(start + MAX_QUOTE_CHARACTERS, right)
            if end < right:
                newline = text.rfind("\n", start, end)
                if newline >= start:
                    end = newline + 1
                else:
                    for index in range(end - 1, start - 1, -1):
                        if text[index].isspace():
                            end = index + 1
                            break
            yield text[start:end]
            start = end


def build_quote_catalog(evidence: dict[UUID, tuple[str, ...]]) -> dict[str, QuoteReference]:
    """Assign deterministic IDs to distinct nonblank literal passages per source."""
    catalog: dict[str, QuoteReference] = {}
    seen: set[tuple[UUID, str]] = set()
    for source_id, excerpts in evidence.items():
        for excerpt in excerpts:
            for quote in iter_literal_passages(excerpt):
                identity = (source_id, quote)
                if quote.strip() and identity not in seen:
                    catalog[f"q{len(catalog) + 1}"] = QuoteReference(source_id, quote)
                    seen.add(identity)
    return catalog


def selection_to_payload(
    output: SelectionOutput, catalog: dict[str, QuoteReference]
) -> dict[str, Any]:
    """Resolve only exact catalog IDs into the existing source-backed public shape.

    IDs may support multiple factual blocks, but cannot repeat within one block.
    Public-schema/source-scope/final-source-count validation remains the caller's job.
    """
    payload: dict[str, Any] = {"summary": [], "insights": []}
    if not output.summary or len(output.summary) > 8 or len(output.insights) > 5:
        raise _invalid_selection()
    for field in ("summary", "insights"):
        for block in getattr(output, field):
            if (
                not block.text.strip()
                or len(block.text) > 4000
                or not 1 <= len(block.quote_ids) <= 8
            ):
                raise _invalid_selection()
            if len(set(block.quote_ids)) != len(block.quote_ids):
                raise _invalid_selection("duplicate_quote_ids")
            source_ids: list[str] = []
            support_quotes: list[dict[str, str]] = []
            for identifier in block.quote_ids:
                reference = catalog.get(identifier)
                if reference is None:
                    raise _invalid_selection("unknown_quote_ids")
                if not reference.quote.strip() or len(reference.quote) > MAX_QUOTE_CHARACTERS:
                    raise _invalid_selection()
                source_id = str(reference.source_id)
                if source_id not in source_ids:
                    source_ids.append(source_id)
                support_quotes.append({"source_id": source_id, "quote": reference.quote})
            payload[field].append(
                {"text": block.text, "source_ids": source_ids, "support_quotes": support_quotes}
            )
    return payload
