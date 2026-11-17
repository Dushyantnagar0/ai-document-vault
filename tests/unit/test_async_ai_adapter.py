"""Native async SDK boundaries through mock transports, without sockets or charges."""

import asyncio
import json
import time
import traceback
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import httpx2
import openai
import pytest
from celery.exceptions import SoftTimeLimitExceeded
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, ConfigDict

import app.adapters.async_openai as adapter
from app.core.config import Settings
from app.domain.documents import ProcessingError

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
Handler = Callable[[httpx2.Request], Awaitable[httpx2.Response]]


def settings() -> Settings:
    return Settings(_env_file=None, openai_api_key="async-unit-private-placeholder")


class Output(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str


class ForbiddenSyncEndpoint:
    def create(self, **kwargs: Any) -> None:
        pytest.fail("Native async adapter must never invoke a synchronous endpoint")


class Wire:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.monkeypatch = monkeypatch
        self.async_clients: list[httpx2.AsyncClient] = []
        self.sync_clients: list[httpx2.Client] = []
        self.options: list[dict[str, Any]] = []

    def embedding(self, handler: Handler) -> None:
        client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
        self.async_clients.append(client)

        def factory(**kwargs: Any) -> OpenAIEmbeddings:
            self.options.append(kwargs)
            root = openai.AsyncOpenAI(
                api_key="unit-placeholder",
                max_retries=kwargs["max_retries"],
                http_client=client,
            )
            return OpenAIEmbeddings(
                **kwargs, client=ForbiddenSyncEndpoint(), async_client=root.embeddings
            )

        self.monkeypatch.setattr(adapter, "OpenAIEmbeddings", factory)

    def generation(self, handler: Handler) -> None:
        async_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
        self.async_clients.append(async_client)

        def forbidden(request: httpx2.Request) -> httpx2.Response:
            pytest.fail("Native async generation must not use synchronous HTTP")

        sync_client = httpx2.Client(transport=httpx2.MockTransport(forbidden))
        self.sync_clients.append(sync_client)

        def factory(**kwargs: Any) -> ChatOpenAI:
            self.options.append(kwargs)
            return ChatOpenAI(
                **kwargs,
                http_socket_options=(),
                http_client=sync_client,
                http_async_client=async_client,
            )

        self.monkeypatch.setattr(adapter, "ChatOpenAI", factory)

    async def close(self) -> None:
        for client in self.async_clients:
            await client.aclose()
        for client in self.sync_clients:
            client.close()


@pytest.fixture
async def wire(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Wire]:
    configured = Wire(monkeypatch)
    yield configured
    await configured.close()


def completion(
    *,
    content: str = '{"answer":"source backed"}',
    model: str = Settings.CHAT_MODEL,
    refusal: str | None = None,
) -> dict[str, Any]:
    return {
        "id": "async-unit-response",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content, "refusal": refusal},
            }
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 8, "total_tokens": 38},
    }


async def invoke(kind: str, remaining_seconds: float) -> Any:
    if kind == "embedding":
        return await adapter.AsyncOpenAIEmbeddingProvider(settings()).embed_many(
            ["evidence"], timeout_seconds=remaining_seconds
        )
    return await adapter.AsyncOpenAIGenerationProvider(settings()).generate(
        schema=Output,
        system="Source only.",
        content="evidence",
        max_output_tokens=99,
        timeout_seconds=remaining_seconds,
    )


async def test_native_async_embedding_one_request_model_dimensions_usage_and_timeout(
    wire: Wire,
) -> None:
    requests: list[httpx2.Request] = []

    async def respond(request: httpx2.Request) -> httpx2.Response:
        assert asyncio.current_task() is not None
        requests.append(request)
        return httpx2.Response(
            200,
            json={
                "object": "list",
                "model": Settings.EMBEDDING_MODEL,
                "data": [
                    {"object": "embedding", "index": 1, "embedding": [0.2] * 1536},
                    {"object": "embedding", "index": 0, "embedding": [0.1] * 1536},
                ],
                "usage": {"prompt_tokens": 7, "total_tokens": 7},
            },
        )

    wire.embedding(respond)
    result = await adapter.AsyncOpenAIEmbeddingProvider(settings()).embed_many(
        ["first", "second"], timeout_seconds=100
    )
    assert len(requests) == 1
    assert wire.options[0]["max_retries"] == 0
    assert wire.options[0]["check_embedding_ctx_length"] is False
    assert requests[0].extensions["timeout"]["read"] == 15
    body = json.loads(requests[0].content)
    assert body["input"] == ["first", "second"] and body["dimensions"] == 1536
    assert body["model"] == Settings.EMBEDDING_MODEL
    assert result.model == Settings.EMBEDDING_MODEL and result.input_tokens == 7
    assert [vector[0] for vector in result.vectors] == pytest.approx([0.1, 0.2])


@pytest.mark.parametrize("effort", ["none", "low"])
async def test_native_async_generation_strict_schema_snapshot_and_usage(
    wire: Wire, effort: str
) -> None:
    requests: list[httpx2.Request] = []

    async def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json=completion())

    wire.generation(respond)
    result = await adapter.AsyncOpenAIGenerationProvider(
        settings(), reasoning_effort=effort
    ).generate(
        schema=Output,
        system="Source only.",
        content="evidence",
        max_output_tokens=99,
        timeout_seconds=4,
    )
    assert len(requests) == 1 and result.result.answer == "source backed"
    body = json.loads(requests[0].content)
    assert body["model"] == Settings.CHAT_MODEL and body["reasoning_effort"] == effort
    assert body["max_completion_tokens"] == 99 and body["store"] is False
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert wire.options[0]["max_retries"] == 0 and wire.options[0]["cache"] is False
    assert requests[0].extensions["timeout"]["read"] == 4
    assert result.model == Settings.CHAT_MODEL
    assert result.input_tokens == 30 and result.output_tokens == 8


@pytest.mark.parametrize("kind", ["embedding", "generation"])
@pytest.mark.parametrize("quota", [False, True])
async def test_async_429_one_attempt_classifies_quota_and_retry_after(
    wire: Wire, kind: str, quota: bool
) -> None:
    requests: list[httpx2.Request] = []
    code = "insufficient_quota" if quota else "rate_limit_exceeded"

    async def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            429,
            headers={"retry-after": "120"},
            json={
                "error": {"message": "async-unit-private-placeholder", "code": code, "type": code}
            },
        )

    getattr(wire, kind)(respond)
    with pytest.raises(ProcessingError) as captured:
        await invoke(kind, 2)
    assert len(requests) == 1
    assert captured.value.code == ("ai_quota" if quota else "ai_rate_limited")
    assert captured.value.retryable is not quota
    assert captured.value.retry_after_seconds == (None if quota else 60)
    assert "async-unit-private-placeholder" not in "".join(
        traceback.format_exception(captured.value)
    )


@pytest.mark.parametrize("kind", ["embedding", "generation"])
async def test_whole_await_deadline_cancels_inflight_request_without_retry(
    wire: Wire, kind: str
) -> None:
    requests: list[httpx2.Request] = []
    cancelled = asyncio.Event()

    async def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        raise AssertionError("Unreachable")

    getattr(wire, kind)(respond)
    started = time.monotonic()
    with pytest.raises(ProcessingError) as captured:
        await asyncio.wait_for(invoke(kind, 0.05), timeout=1)
    assert captured.value.code == "ai_timeout" and captured.value.retryable is True
    assert time.monotonic() - started < 0.5
    assert len(requests) == 1 and cancelled.is_set()


@pytest.mark.parametrize("kind", ["embedding", "generation"])
async def test_caller_cancellation_propagates_and_does_not_leave_background_http(
    wire: Wire, kind: str
) -> None:
    requests: list[httpx2.Request] = []
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        raise AssertionError("Unreachable")

    getattr(wire, kind)(respond)
    task = asyncio.create_task(invoke(kind, 10))
    await asyncio.wait_for(entered.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(requests) == 1 and cancelled.is_set()


@pytest.mark.parametrize("problem", ["dimension", "count", "nan", "boolean", "model", "index"])
async def test_async_embedding_rejects_invalid_vectors_and_profile(
    monkeypatch: pytest.MonkeyPatch,
    problem: str,
) -> None:
    calls: list[bool] = []

    class Endpoint:
        async def create(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(True)
            values: list[Any] = [0.1] * (2 if problem == "dimension" else 1536)
            if problem == "nan":
                values[0] = float("nan")
            if problem == "boolean":
                values[0] = True
            return {
                "model": "wrong" if problem == "model" else Settings.EMBEDDING_MODEL,
                "data": []
                if problem == "count"
                else [{"index": False if problem == "index" else 0, "embedding": values}],
            }

    class Wrapper:
        async_client = Endpoint()
        client = ForbiddenSyncEndpoint()

    monkeypatch.setattr(adapter, "OpenAIEmbeddings", lambda **kwargs: Wrapper())
    with pytest.raises(ProcessingError) as captured:
        await invoke("embedding", 1)
    assert captured.value.code == "embedding_invalid_output" and not captured.value.retryable
    assert len(calls) == 1


@pytest.mark.parametrize("texts", [[], [" "], ["x"] * 33, ["x " * 8192], ["x " * 2000] * 17])
async def test_async_embedding_input_limits_prevent_outbound_calls(
    monkeypatch: pytest.MonkeyPatch,
    texts: list[str],
) -> None:
    class Endpoint:
        async def create(self, **kwargs: Any) -> None:
            pytest.fail("Invalid embedding input must not make an outbound call")

    class Wrapper:
        async_client = Endpoint()

    monkeypatch.setattr(adapter, "OpenAIEmbeddings", lambda **kwargs: Wrapper())
    with pytest.raises(ProcessingError) as captured:
        await adapter.AsyncOpenAIEmbeddingProvider(settings()).embed_many(texts, timeout_seconds=1)
    assert captured.value.code == "embedding_input_limit"


@pytest.mark.parametrize(
    "problem,code",
    [
        ("schema", "ai_schema_invalid"),
        ("refusal", "ai_refused"),
        ("model", "ai_model_mismatch"),
    ],
)
async def test_async_generation_invalid_schema_refusal_or_profile_is_safe(
    wire: Wire,
    problem: str,
    code: str,
) -> None:
    requests: list[httpx2.Request] = []

    async def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json=completion(
                content='{"answer":["async-unit-private-placeholder"]}'
                if problem == "schema"
                else '{"answer":"fact"}',
                refusal="async-unit-private-placeholder" if problem == "refusal" else None,
                model="wrong-profile" if problem == "model" else Settings.CHAT_MODEL,
            ),
        )

    wire.generation(respond)
    with pytest.raises(ProcessingError) as captured:
        await invoke("generation", 1)
    assert captured.value.code == code and not captured.value.retryable
    assert len(requests) == 1
    assert "async-unit-private-placeholder" not in "".join(
        traceback.format_exception(captured.value)
    )


async def test_async_generation_output_limit_prevents_sdk_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(**kwargs: Any) -> None:
        pytest.fail("Out-of-bounds output must not construct a provider client")

    monkeypatch.setattr(adapter, "ChatOpenAI", forbidden)
    with pytest.raises(ProcessingError) as captured:
        await adapter.AsyncOpenAIGenerationProvider(settings()).generate(
            schema=Output,
            system="source",
            content="evidence",
            max_output_tokens=128001,
            timeout_seconds=1,
        )
    assert captured.value.code == "generation_input_limit"


@pytest.mark.parametrize("kind", ["embedding", "generation"])
@pytest.mark.parametrize("remaining_seconds", [0, -1, float("nan")])
async def test_invalid_deadline_prevents_outbound_calls(
    wire: Wire,
    kind: str,
    remaining_seconds: float,
) -> None:
    async def forbidden(request: httpx2.Request) -> httpx2.Response:
        pytest.fail("Invalid timeout must not make an outbound call")

    getattr(wire, kind)(forbidden)
    with pytest.raises(ProcessingError) as captured:
        await invoke(kind, remaining_seconds)
    assert captured.value.code == "processing_deadline"


@pytest.mark.parametrize("kind", ["embedding", "generation"])
async def test_native_async_sdk_timeouts_are_redacted_and_never_retried(
    wire: Wire, kind: str
) -> None:
    requests: list[httpx2.Request] = []

    async def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        raise httpx2.ReadTimeout("async-unit-private-placeholder", request=request)

    getattr(wire, kind)(respond)
    with pytest.raises(ProcessingError) as captured:
        await invoke(kind, 1)
    assert captured.value.code == "ai_timeout" and captured.value.retryable
    assert len(requests) == 1
    assert "async-unit-private-placeholder" not in "".join(
        traceback.format_exception(captured.value)
    )


async def test_async_generation_disables_hosted_tracing_even_when_environment_enables_it(
    wire: Wire,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langsmith.run_helpers import get_tracing_context

    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    async def respond(request: httpx2.Request) -> httpx2.Response:
        assert get_tracing_context()["enabled"] is False
        return httpx2.Response(200, json=completion())

    wire.generation(respond)
    result = await invoke("generation", 1)
    assert result.result.answer == "source backed"


async def test_async_structured_quote_schema_limit_is_validated_without_retry(wire: Wire) -> None:
    from uuid import uuid4

    from app.services.analysis import SourceQuote

    requests: list[httpx2.Request] = []

    async def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json=completion(content=json.dumps({"source_id": str(uuid4()), "quote": "x" * 301})),
        )

    wire.generation(respond)
    with pytest.raises(ProcessingError) as captured:
        await adapter.AsyncOpenAIGenerationProvider(settings()).generate(
            schema=SourceQuote,
            system="source only",
            content="evidence",
            max_output_tokens=500,
            timeout_seconds=1,
        )
    assert captured.value.code == "ai_schema_invalid" and not captured.value.retryable
    assert len(requests) == 1


@pytest.mark.parametrize(
    "problem,code,reason",
    [
        ("length_complete", "ai_output_truncated", "output_token_limit"),
        ("length_partial", "ai_output_truncated", "output_token_limit"),
        ("partial_json", "ai_invalid_json", "invalid_json"),
        ("invalid_uuid", "ai_schema_invalid", "schema_validation"),
        ("nested_quote", "ai_schema_invalid", "schema_validation"),
    ],
)
async def test_async_strict_raw_failure_diagnostics_and_no_json_salvage(
    wire: Wire,
    problem: str,
    code: str,
    reason: str,
) -> None:
    from app.services.analysis import SourceQuote

    payload = {"source_id": "00000000-0000-0000-0000-000000000001", "quote": "CANARY_PRIVATE_TEXT"}
    if problem == "invalid_uuid":
        payload["source_id"] = "CANARY_PRIVATE_TEXT"
    if problem == "nested_quote":
        payload["quote"] = "x" * 301
    content = json.dumps(payload)
    if problem in {"length_partial", "partial_json"}:
        content = content[:-1]
    requests = []

    async def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        response = completion(content=content)
        response["choices"][0]["finish_reason"] = (
            "length" if problem.startswith("length") else "stop"
        )
        return httpx2.Response(200, headers={"x-request-id": "req_async-test_123"}, json=response)

    wire.generation(respond)
    with pytest.raises(ProcessingError) as error:
        await adapter.AsyncOpenAIGenerationProvider(settings()).generate(
            schema=SourceQuote,
            system="source only",
            content="evidence",
            max_output_tokens=99,
            timeout_seconds=2,
        )
    assert len(requests) == 1 and error.value.code == code
    assert error.value.diagnostics is not None
    details = error.value.diagnostics.as_dict()
    assert details["reason"] == reason
    assert details["input_tokens"] == 30 and details["output_tokens"] == 8
    assert details["max_output_tokens"] == 99
    assert details["provider_request_id"] == "req_async-test_123"
    assert "CANARY_PRIVATE_TEXT" not in json.dumps(details)
    assert "CANARY_PRIVATE_TEXT" not in "".join(traceback.format_exception(error.value))


@pytest.mark.parametrize("interruption", [asyncio.CancelledError, SoftTimeLimitExceeded])
async def test_async_returned_parser_interruption_propagates(
    monkeypatch: pytest.MonkeyPatch,
    interruption: type[BaseException],
) -> None:
    class Structured:
        async def ainvoke(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"parsing_error": interruption()}

    class Model:
        def with_structured_output(self, *args: Any, **kwargs: Any) -> Structured:
            return Structured()

    monkeypatch.setattr(adapter, "ChatOpenAI", lambda **kwargs: Model())
    with pytest.raises(interruption):
        await invoke("generation", 1)
