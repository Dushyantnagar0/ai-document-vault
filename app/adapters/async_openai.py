"""Native async, single-request provider boundaries for interactive conversation."""

import asyncio
from collections.abc import Sequence
from typing import Any, Literal, TypeVar, cast

import tiktoken
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import tracing_context
from pydantic import BaseModel

from app.adapters.openai import _parse_generation, _timeout, _usage, provider_error
from app.core.config import Settings
from app.domain.documents import AIFailureDetails, EmbeddingBatch, Generated, ProcessingError
from app.domain.vectors import float32_vector
from app.prompts.analysis import generation_prompt_tokens

T = TypeVar("T", bound=BaseModel)


class AsyncOpenAIEmbeddingProvider:
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

    async def embed_many(self, texts: Sequence[str], *, timeout_seconds: float) -> EmbeddingBatch:
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
            async with asyncio.timeout(timeout):
                # Direct native async endpoint preserves model/usage and guarantees
                # one request, without LangChain's splitting/internal batch loops.
                response = await self._embeddings.async_client.create(
                    input=list(texts),
                    model=self.settings.openai_embedding_model,
                    dimensions=self.settings.openai_embedding_dimensions,
                    timeout=timeout,
                )
        except TimeoutError:
            raise ProcessingError("ai_timeout", "AI request timed out.", retryable=True) from None
        except Exception as exc:
            raise provider_error(exc) from None
        data: Any = None
        try:
            data = response if isinstance(response, dict) else response.model_dump()
            if data["model"] != self.settings.openai_embedding_model:
                raise ValueError
            rows = data["data"]
            if (
                len(rows) != len(texts)
                or any(type(row["index"]) is not int for row in rows)
                or sorted(row["index"] for row in rows) != list(range(len(texts)))
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


class AsyncOpenAIGenerationProvider:
    def __init__(
        self, settings: Settings, *, reasoning_effort: Literal["none", "low"] = "none"
    ) -> None:
        self.settings = settings
        self.reasoning_effort = reasoning_effort

    async def generate(
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
            async with asyncio.timeout(timeout):
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
                        await structured.ainvoke(
                            [
                                SystemMessage(content=system),
                                HumanMessage(content=content),
                            ]
                        ),
                    )
        except TimeoutError:
            raise ProcessingError("ai_timeout", "AI request timed out.", retryable=True) from None
        except Exception as exc:
            raise provider_error(exc, max_output_tokens=max_output_tokens) from None
        return _parse_generation(result, schema, self.settings.openai_chat_model, max_output_tokens)
