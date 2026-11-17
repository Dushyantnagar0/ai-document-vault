"""Single-call OpenAI boundaries with safe errors and no hidden retry/batching."""

import asyncio
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeVar, cast

import openai
import tiktoken
from celery.exceptions import SoftTimeLimitExceeded
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import tracing_context
from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.domain.documents import AIFailureDetails, EmbeddingBatch, Generated, ProcessingError
from app.domain.vectors import float32_vector
from app.prompts.analysis import generation_prompt_tokens

T = TypeVar("T", bound=BaseModel)


def provider_error(exc: Exception, *, max_output_tokens: int | None = None) -> ProcessingError:
    """Map known provider categories without retaining their possibly private body."""
    if isinstance(exc, SoftTimeLimitExceeded):
        raise exc
    # The SDK's transport catch wraps a soft signal directly as a connection
    # error; the real-SDK mock-transport regression covers this specific path.
    if isinstance(exc, openai.APIConnectionError) and isinstance(
        exc.__cause__, (SoftTimeLimitExceeded, asyncio.CancelledError)
    ):
        raise exc.__cause__
    if isinstance(exc, openai.LengthFinishReasonError):
        completion = exc.completion
        usage = getattr(completion, "usage", None)
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", {})
        request_id = headers.get("x-request-id") or getattr(completion, "_request_id", None)
        return ProcessingError(
            "ai_output_truncated",
            "AI response exceeded its output-token limit.",
            diagnostics=_safe_details(
                AIFailureDetails(
                    reason="output_token_limit",
                    finish_reason="length",
                    input_tokens=_usage(getattr(usage, "prompt_tokens", None)),
                    output_tokens=_usage(getattr(usage, "completion_tokens", None)),
                    max_output_tokens=max_output_tokens,
                    provider_request_id=request_id,
                )
            ),
        )
    if isinstance(
        exc,
        (
            ValidationError,
            json.JSONDecodeError,
            OutputParserException,
            openai.APIResponseValidationError,
        ),
    ):
        reason, code = (
            ("invalid_json", "ai_invalid_json")
            if isinstance(exc, json.JSONDecodeError)
            else ("schema_validation", "ai_schema_invalid")
            if isinstance(exc, ValidationError)
            else ("structured_parse_failure", "ai_invalid_output")
        )
        return ProcessingError(
            code,
            "AI returned an invalid structured response.",
            diagnostics=AIFailureDetails(reason=reason, max_output_tokens=max_output_tokens),
        )
    if isinstance(exc, openai.ContentFilterFinishReasonError):
        return ProcessingError("ai_refused", "AI could not analyse this content.")
    if isinstance(exc, openai.APITimeoutError):
        return ProcessingError("ai_timeout", "AI request timed out.", retryable=True)
    if isinstance(exc, openai.APIConnectionError):
        return ProcessingError("ai_connection", "AI service connection failed.", retryable=True)
    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        if status == 429:
            body = exc.body if isinstance(exc.body, dict) else {}
            error = body.get("error", body)
            error = error if isinstance(error, dict) else {}
            failure_code = error.get("code") or error.get("type") or getattr(exc, "code", None)
            if isinstance(failure_code, str) and failure_code in {
                "insufficient_quota",
                "billing_hard_limit_reached",
                "quota_exceeded",
            }:
                return ProcessingError("ai_quota", "AI account quota is unavailable.")
            error_result = ProcessingError(
                "ai_rate_limited", "AI service rate limited the request.", retryable=True
            )
            try:
                retry_after = float(exc.response.headers.get("retry-after", ""))
                if math.isfinite(retry_after) and retry_after >= 0:
                    error_result.retry_after_seconds = min(retry_after, 60)
            except ValueError:
                pass
            return error_result
        if status in {401, 403}:
            return ProcessingError(
                "ai_authentication", "AI credentials or model access were rejected."
            )
        if status in {400, 404, 422}:
            return ProcessingError(
                "ai_configuration", "AI model or request configuration was rejected."
            )
        if status in {408, 409} or status >= 500:
            return ProcessingError(
                "ai_unavailable", "AI service is temporarily unavailable.", retryable=True
            )
    return ProcessingError("ai_provider_error", "AI request could not be completed.")


def _timeout(requested: float, configured: float) -> float:
    if not math.isfinite(requested) or requested <= 0:
        raise ProcessingError("processing_deadline", "No time remains for an AI request.")
    return min(requested, configured)


def _usage(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _generation_failure(
    raw: AIMessage | None,
    reason: str,
    max_output_tokens: int,
    *,
    code: str = "ai_invalid_output",
    message: str = "AI returned an invalid structured response.",
) -> ProcessingError:
    metadata = raw.response_metadata if raw is not None else {}
    usage: Mapping[str, Any] = (raw.usage_metadata or {}) if raw is not None else {}
    headers = metadata.get("headers")
    request_id = metadata.get("request_id")
    if request_id is None and isinstance(headers, dict):
        request_id = headers.get("x-request-id")
    details = AIFailureDetails(
        reason=reason,
        finish_reason=metadata.get("finish_reason"),
        input_tokens=_usage(usage.get("input_tokens")),
        output_tokens=_usage(usage.get("output_tokens")),
        max_output_tokens=max_output_tokens,
        provider_request_id=request_id,
    )
    return ProcessingError(code, message, diagnostics=_safe_details(details))


def _safe_details(details: AIFailureDetails) -> AIFailureDetails:
    # Sanitize fields before attaching them, not only when persisting as_dict().
    clean = details.as_dict()
    return AIFailureDetails(
        reason=str(clean["reason"]),
        finish_reason=cast(str | None, clean["finish_reason"]),
        input_tokens=cast(int | None, clean["input_tokens"]),
        output_tokens=cast(int | None, clean["output_tokens"]),
        max_output_tokens=cast(int | None, clean["max_output_tokens"]),
        provider_request_id=cast(str | None, clean["provider_request_id"]),
    )


def _parse_generation[T: BaseModel](
    result: dict[str, Any], schema: type[T], model: str, output: int
) -> Generated[T]:
    parsing_error = result.get("parsing_error")
    if isinstance(parsing_error, (SoftTimeLimitExceeded, asyncio.CancelledError)):
        raise parsing_error
    if isinstance(parsing_error, openai.APIConnectionError) and isinstance(
        parsing_error.__cause__, (SoftTimeLimitExceeded, asyncio.CancelledError)
    ):
        raise parsing_error.__cause__
    raw = result.get("raw")
    if not isinstance(raw, AIMessage):
        raise _generation_failure(None, "missing_raw_response", output)
    # A complete-looking partial result must never hide a token-limit finish.
    if raw.response_metadata.get("finish_reason") == "length":
        raise _generation_failure(
            raw,
            "output_token_limit",
            output,
            code="ai_output_truncated",
            message="AI response exceeded its output-token limit.",
        )
    if (
        raw.additional_kwargs.get("refusal")
        or raw.response_metadata.get("finish_reason") == "content_filter"
    ):
        raise _generation_failure(
            raw,
            "structured_parse_failure",
            output,
            code="ai_refused",
            message="AI could not analyse this content.",
        )
    if raw.response_metadata.get("model_name") != model:
        raise _generation_failure(
            raw,
            "structured_parse_failure",
            output,
            code="ai_model_mismatch",
            message="AI response used an unexpected model profile.",
        )
    if not isinstance(raw.content, str) or not raw.content:
        raise _generation_failure(raw, "missing_parsed_response", output)
    try:
        # LangChain's JSON parser can repair partial JSON. Only original complete JSON is accepted.
        value = json.loads(raw.content)
    except (json.JSONDecodeError, ValueError, TypeError):
        raise _generation_failure(raw, "invalid_json", output, code="ai_invalid_json") from None
    try:
        parsed = schema.model_validate(value)
    except (ValidationError, ValueError, TypeError):
        raise _generation_failure(
            raw, "schema_validation", output, code="ai_schema_invalid"
        ) from None
    usage: Mapping[str, Any] = raw.usage_metadata or {}
    return Generated(
        result=parsed,
        model=model,
        input_tokens=_usage(usage.get("input_tokens")),
        output_tokens=_usage(usage.get("output_tokens")),
    )


class OpenAIEmbeddingProvider:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            dimensions=settings.openai_embedding_dimensions,
            openai_api_key=settings.openai_api_key,
            max_retries=0,
            request_timeout=settings.openai_embedding_timeout_seconds,
            check_embedding_ctx_length=False,
            chunk_size=settings.embedding_batch_size,
        )

    def embed_many(self, texts: Sequence[str], *, timeout_seconds: float) -> EmbeddingBatch:
        timeout = _timeout(timeout_seconds, self.settings.openai_embedding_timeout_seconds)
        encoding = tiktoken.get_encoding("cl100k_base")
        counts = [len(encoding.encode(text, disallowed_special=())) for text in texts]
        if (
            not texts
            or len(texts) > self.settings.embedding_batch_size
            or any(not text.strip() for text in texts)
            or any(count > self.settings.EMBEDDING_INPUT_TOKENS for count in counts)
            or sum(counts) > self.settings.embedding_batch_max_tokens
        ):
            raise ProcessingError(
                "embedding_input_limit", "Embedding batch exceeds validated limits."
            )
        try:
            # Use LangChain's configured endpoint directly: preserve model/usage and
            # guarantee one request, without embed_documents' internal batch loop.
            response = self._embeddings.client.create(
                input=list(texts),
                model=self.settings.openai_embedding_model,
                dimensions=self.settings.openai_embedding_dimensions,
                timeout=timeout,
            )
        except SoftTimeLimitExceeded:
            raise
        except Exception as exc:
            raise provider_error(exc) from None
        data: Any = None
        try:
            data = response if isinstance(response, dict) else response.model_dump()
            if data["model"] != self.settings.openai_embedding_model:
                raise ValueError
            rows = data["data"]
            if len(rows) != len(texts) or sorted(row["index"] for row in rows) != list(
                range(len(texts))
            ):
                raise ValueError
            vectors = []
            for row in sorted(rows, key=lambda item: item["index"]):
                vectors.append(
                    float32_vector(row["embedding"], self.settings.openai_embedding_dimensions)
                )
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            usage = data.get("usage") if isinstance(data, dict) else None
            raise ProcessingError(
                "embedding_invalid_output",
                "AI returned invalid embedding data.",
                diagnostics=AIFailureDetails(
                    reason="schema_validation",
                    input_tokens=_usage(usage.get("prompt_tokens"))
                    if isinstance(usage, dict)
                    else None,
                ),
            ) from None
        usage = data.get("usage")
        return EmbeddingBatch(
            vectors=vectors,
            model=data["model"],
            input_tokens=_usage(usage.get("prompt_tokens")) if isinstance(usage, dict) else None,
        )


class OpenAIGenerationProvider:
    def __init__(
        self, settings: Settings, *, reasoning_effort: Literal["none", "low"] = "none"
    ) -> None:
        self.settings = settings
        self.reasoning_effort = reasoning_effort

    def generate(
        self,
        *,
        schema: type[T],
        system: str,
        content: str,
        max_output_tokens: int,
        timeout_seconds: float,
    ) -> Generated[T]:
        timeout = _timeout(timeout_seconds, self.settings.openai_generation_timeout_seconds)
        input_tokens = generation_prompt_tokens(schema, system, content)
        if (
            max_output_tokens <= 0
            or max_output_tokens > self.settings.CHAT_MODEL_MAX_OUTPUT_TOKENS
            or input_tokens > self.settings.CHAT_MODEL_MAX_INPUT_TOKENS
            or input_tokens + max_output_tokens > self.settings.CHAT_CONTEXT_TOKENS
        ):
            raise ProcessingError(
                "generation_input_limit", "Generation request exceeds model limits."
            )
        try:
            model = ChatOpenAI(
                model=self.settings.openai_chat_model,
                api_key=self.settings.openai_api_key,
                max_retries=0,
                timeout=timeout,
                max_completion_tokens=max_output_tokens,
                reasoning_effort=self.reasoning_effort,
                use_responses_api=False,
                store=False,
                cache=False,
                include_response_headers=True,
            )
            structured = model.with_structured_output(
                schema.model_json_schema(),
                method="json_schema",
                strict=True,
                include_raw=True,
            )
            with tracing_context(enabled=False):
                result = cast(
                    dict[str, Any],
                    structured.invoke(
                        [
                            SystemMessage(content=system),
                            HumanMessage(content=content),
                        ]
                    ),
                )
        except SoftTimeLimitExceeded:
            raise
        except Exception as exc:
            raise provider_error(exc, max_output_tokens=max_output_tokens) from None
        return _parse_generation(result, schema, self.settings.openai_chat_model, max_output_tokens)
