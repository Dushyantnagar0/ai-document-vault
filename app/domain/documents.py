"""Format/provider independent document-processing values."""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeVar
from uuid import UUID

from pydantic import BaseModel

Coverage = Literal["complete_within_capability", "partial", "unknown"]
Locator = dict[str, str | int]


@dataclass(frozen=True)
class AIFailureDetails:
    """Allowlisted provider diagnostics; never response text, prompts or exception bodies."""

    reason: str
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    max_output_tokens: int | None = None
    provider_request_id: str | None = None

    def as_dict(self) -> dict[str, str | int | None]:
        reasons = {
            "output_token_limit",
            "input_token_limit",
            "invalid_json",
            "schema_validation",
            "structured_parse_failure",
            "missing_raw_response",
            "missing_parsed_response",
            "blank_factual_text",
            "duplicate_source_ids",
            "unknown_source_ids",
            "missing_source_quote",
            "uncited_quote_source",
            "blank_support_quote",
            "quote_not_in_original",
            "quote_not_in_supplied_evidence",
            "final_source_limit",
            "unknown_quote_id",
            "duplicate_quote_id",
            "unknown_insight_id",
            "duplicate_insight_id",
            "support_window_exceeded",
            "inconsistent_evidence_status",
            "claim_identity_mismatch",
            "invalid_support_selection",
            "invalid_overlap_selection",
            "invalid_qualification_selection",
            "empty_summary",
            "claim_limit",
            "invalid_draft_fields",
            "invalid_draft_text",
        }
        result: dict[str, str | int | None] = {
            "reason": self.reason
            if isinstance(self.reason, str) and self.reason in reasons
            else "structured_parse_failure",
            "finish_reason": self.finish_reason
            if isinstance(self.finish_reason, str)
            and self.finish_reason
            in {"stop", "length", "content_filter", "tool_calls", "function_call"}
            else None,
            "provider_request_id": self.provider_request_id
            if (
                isinstance(self.provider_request_id, str)
                and re.fullmatch(r"req_[A-Za-z0-9_-]{1,120}", self.provider_request_id)
            )
            else None,
        }
        for key in ("input_tokens", "output_tokens", "max_output_tokens"):
            value = getattr(self, key)
            result[key] = value if type(value) is int and 0 <= value <= 1_000_000_000 else None
        return result


class ProcessingError(Exception):
    """Safe public diagnostic, with an application-owned retry classification."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        retry_after_seconds: float | None = None,
        diagnostics: AIFailureDetails | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class ExtractionWarning:
    code: str
    message: str
    locator: Locator | None = None


@dataclass(frozen=True)
class TextBlock:
    block_id: str
    text: str
    locator: Locator
    start: int
    end: int


@dataclass(frozen=True)
class ExtractionResult:
    text: str
    blocks: tuple[TextBlock, ...]
    extractor_name: str
    extractor_version: str
    coverage: Coverage
    warnings: tuple[ExtractionWarning, ...]
    page_count: int | None


@dataclass(frozen=True)
class SourceSpan:
    block_id: str
    locator: Locator
    document_start: int
    document_end: int
    chunk_start: int
    chunk_end: int


@dataclass(frozen=True)
class Chunk:
    id: UUID
    ordinal: int
    text: str
    token_count: int
    source_spans: tuple[SourceSpan, ...]


class Extractor(Protocol):
    def extract(self, content: bytes) -> ExtractionResult: ...


@dataclass(frozen=True)
class EmbeddingBatch:
    vectors: list[list[float]]
    model: str
    input_tokens: int | None = None


T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class Generated[T: BaseModel]:
    result: T
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class EmbeddingProvider(Protocol):
    def embed_many(self, texts: Sequence[str], *, timeout_seconds: float) -> EmbeddingBatch: ...


class GenerationProvider(Protocol):
    def generate(
        self,
        *,
        schema: type[T],
        system: str,
        content: str,
        max_output_tokens: int,
        timeout_seconds: float,
    ) -> Generated[T]: ...


# Invoked immediately before each outbound call, including repairs. Returns the
# remaining deadline after atomically reserving the persisted call budget.
BeforeCall = Callable[[], float]


@dataclass(frozen=True)
class AnalysisResult:
    payload: dict[str, Any]
    model: str
    prompt_version: str
    covered_chunk_ids: tuple[UUID, ...]
