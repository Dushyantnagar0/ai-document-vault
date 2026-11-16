"""Mock HTTP transports exercise the real SDK without sockets/provider charges."""

import asyncio
import json
import traceback
from typing import Any

import httpx2
import openai
import pytest
from celery.exceptions import SoftTimeLimitExceeded
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, ConfigDict

import app.adapters.openai as adapter
from app.core.config import Settings
from app.domain.documents import ProcessingError

pytestmark = pytest.mark.unit


def settings() -> Settings:
    return Settings(_env_file=None, openai_api_key="unit-private-placeholder")


class Output(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str


def test_embedding_single_actual_request_no_automatic_splitting_or_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx2.Request] = []
    options: list[dict[str, Any]] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        body = json.loads(request.content)
        return httpx2.Response(
            200,
            json={
                "object": "list",
                "model": Settings.EMBEDDING_MODEL,
                "data": [
                    {"object": "embedding", "index": index, "embedding": [0.25] * 1536}
                    for index, _ in enumerate(body["input"])
                ],
                "usage": {"prompt_tokens": 7, "total_tokens": 7},
            },
        )

    client = openai.OpenAI(
        api_key="unit-placeholder",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
    )

    def factory(**kwargs: Any) -> OpenAIEmbeddings:
        options.append(kwargs)
        return OpenAIEmbeddings(**kwargs, client=client.embeddings, async_client=object())

    monkeypatch.setattr(adapter, "OpenAIEmbeddings", factory)
    result = adapter.OpenAIEmbeddingProvider(settings()).embed_many(
        ["first", "second"], timeout_seconds=3
    )
    assert len(requests) == 1
    assert options[0]["max_retries"] == 0
    assert options[0]["check_embedding_ctx_length"] is False
    assert len(result.vectors) == 2 and len(result.vectors[0]) == 1536
    assert result.input_tokens == 7
    assert requests[0].extensions["timeout"]["read"] == 3
    assert json.loads(requests[0].content)["dimensions"] == 1536


@pytest.mark.parametrize("effort", ["none", "low"])
def test_generation_real_sdk_strict_schema_reasoning_setting_and_no_retry(
    monkeypatch: pytest.MonkeyPatch,
    effort: str,
) -> None:
    requests: list[httpx2.Request] = []
    options: list[dict[str, Any]] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json={
                "id": "unit-response",
                "object": "chat.completion",
                "created": 1,
                "model": Settings.CHAT_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": '{"answer":"source backed"}',
                            "refusal": None,
                        },
                    }
                ],
                "usage": {"prompt_tokens": 30, "completion_tokens": 8, "total_tokens": 38},
            },
        )

    def factory(**kwargs: Any) -> ChatOpenAI:
        options.append(kwargs)
        return ChatOpenAI(
            **kwargs,
            # MockTransport has no sockets; skip LangChain's local TCP-option probe.
            http_socket_options=(),
            http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
            http_async_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)),
        )

    monkeypatch.setattr(adapter, "ChatOpenAI", factory)
    result = adapter.OpenAIGenerationProvider(settings(), reasoning_effort=effort).generate(
        schema=Output,
        system="Source-only.",
        content="evidence",
        max_output_tokens=99,
        timeout_seconds=4,
    )
    assert len(requests) == 1 and result.result.answer == "source backed"
    body = json.loads(requests[0].content)
    assert body["model"] == Settings.CHAT_MODEL
    assert body["reasoning_effort"] == effort
    assert body["max_completion_tokens"] == 99
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert options[0]["max_retries"] == 0
    assert requests[0].extensions["timeout"]["read"] == 4
    assert result.input_tokens == 30 and result.output_tokens == 8


@pytest.mark.parametrize(
    "status,body,retry_after,code,retryable,delay",
    [
        (429, {"code": "insufficient_quota"}, "5", "ai_quota", False, None),
        (429, {"type": "rate_limit_exceeded"}, "120", "ai_rate_limited", True, 60),
        (429, {"code": "rate_limit_exceeded"}, "NaN", "ai_rate_limited", True, None),
        (401, {}, None, "ai_authentication", False, None),
        (403, {}, None, "ai_authentication", False, None),
        (400, {}, None, "ai_configuration", False, None),
        (404, {}, None, "ai_configuration", False, None),
        (503, {}, None, "ai_unavailable", True, None),
    ],
)
def test_safe_provider_classification_and_capped_retry_after(
    status: int,
    body: dict[str, Any],
    retry_after: str | None,
    code: str,
    retryable: bool,
    delay: float | None,
) -> None:
    headers = {} if retry_after is None else {"retry-after": retry_after}
    response = httpx2.Response(
        status, request=httpx2.Request("POST", "https://invalid.example"), headers=headers
    )
    failure = openai.APIStatusError("unit-private-placeholder", response=response, body=body)
    translated = adapter.provider_error(failure)
    assert translated.code == code and translated.retryable is retryable
    assert translated.retry_after_seconds == delay
    assert "unit-private-placeholder" not in str(translated)


def test_connection_and_timeout_categories() -> None:
    request = httpx2.Request("POST", "https://invalid.example")
    assert (
        adapter.provider_error(openai.APIConnectionError(request=request)).code == "ai_connection"
    )
    assert adapter.provider_error(openai.APITimeoutError(request=request)).code == "ai_timeout"


@pytest.mark.parametrize("problem", ["dimension", "count", "nan", "model"])
def test_invalid_embedding_response_is_terminal_and_safe(
    monkeypatch: pytest.MonkeyPatch,
    problem: str,
) -> None:
    class Endpoint:
        def create(self, **kwargs: Any) -> dict[str, Any]:
            vector = [0.1] * (2 if problem == "dimension" else 1536)
            if problem == "nan":
                vector[0] = float("nan")
            return {
                "model": "wrong" if problem == "model" else Settings.EMBEDDING_MODEL,
                "data": [] if problem == "count" else [{"index": 0, "embedding": vector}],
                "usage": {"prompt_tokens": 1},
            }

    class Wrapper:
        client = Endpoint()

    monkeypatch.setattr(adapter, "OpenAIEmbeddings", lambda **kwargs: Wrapper())
    with pytest.raises(ProcessingError) as captured:
        adapter.OpenAIEmbeddingProvider(settings()).embed_many(["evidence"], timeout_seconds=1)
    assert captured.value.code == "embedding_invalid_output"
    assert captured.value.retryable is False


def test_embedding_validation_prevents_outbound_call(monkeypatch: pytest.MonkeyPatch) -> None:
    class Endpoint:
        def create(self, **kwargs: Any) -> None:
            pytest.fail("Invalid input must not make an outbound call")

    class Wrapper:
        client = Endpoint()

    monkeypatch.setattr(adapter, "OpenAIEmbeddings", lambda **kwargs: Wrapper())
    provider = adapter.OpenAIEmbeddingProvider(settings())
    for texts in ([], [" "], ["x"] * 33):
        with pytest.raises(ProcessingError) as captured:
            provider.embed_many(texts, timeout_seconds=1)
        assert captured.value.code == "embedding_input_limit"


def test_outbound_exception_is_redacted_without_automatic_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    class Endpoint:
        def create(self, **kwargs: Any) -> None:
            calls.append(True)
            raise ValueError("unit-private-placeholder provider body")

    class Wrapper:
        client = Endpoint()

    monkeypatch.setattr(adapter, "OpenAIEmbeddings", lambda **kwargs: Wrapper())
    with pytest.raises(ProcessingError) as captured:
        adapter.OpenAIEmbeddingProvider(settings()).embed_many(["evidence"], timeout_seconds=1)
    assert len(calls) == 1
    assert "unit-private-placeholder" not in "".join(traceback.format_exception(captured.value))


@pytest.mark.parametrize(
    "code,retryable", [("insufficient_quota", False), ("rate_limit_exceeded", True)]
)
def test_real_sdk_rate_errors_make_one_outbound_attempt(
    monkeypatch: pytest.MonkeyPatch,
    code: str,
    retryable: bool,
) -> None:
    requests: list[httpx2.Request] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            429,
            headers={"retry-after": "2"},
            json={
                "error": {"message": "unit-private-placeholder", "code": code, "type": code},
            },
        )

    def factory(**kwargs: Any) -> ChatOpenAI:
        return ChatOpenAI(
            **kwargs,
            http_socket_options=(),
            http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
            http_async_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)),
        )

    monkeypatch.setattr(adapter, "ChatOpenAI", factory)
    with pytest.raises(ProcessingError) as captured:
        adapter.OpenAIGenerationProvider(settings()).generate(
            schema=Output,
            system="source only",
            content="evidence",
            max_output_tokens=20,
            timeout_seconds=3,
        )
    assert len(requests) == 1
    assert captured.value.retryable is retryable
    assert "unit-private-placeholder" not in "".join(traceback.format_exception(captured.value))


def test_structured_parse_failure_is_safe_and_eligible_for_service_repair() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as captured:
        Output.model_validate({"answer": ["unit-private-placeholder"]})
    mapped = adapter.provider_error(captured.value)
    assert mapped.code == "ai_schema_invalid"
    assert mapped.retryable is False
    assert "unit-private-placeholder" not in str(mapped)


@pytest.mark.parametrize("interruption", [SoftTimeLimitExceeded, asyncio.CancelledError])
def test_embedding_boundary_propagates_worker_and_cancellation_interruptions(
    monkeypatch: pytest.MonkeyPatch,
    interruption: type[BaseException],
) -> None:
    calls = []

    class Endpoint:
        def create(self, **kwargs: Any) -> None:
            calls.append(True)
            raise interruption()

    class Wrapper:
        client = Endpoint()

    monkeypatch.setattr(adapter, "OpenAIEmbeddings", lambda **kwargs: Wrapper())
    provider = adapter.OpenAIEmbeddingProvider(settings())
    with pytest.raises(interruption):
        provider.embed_many(["evidence"], timeout_seconds=1)
    assert len(calls) == 1


@pytest.mark.parametrize("interruption", [SoftTimeLimitExceeded, asyncio.CancelledError])
def test_generation_invoke_propagates_worker_and_cancellation_interruptions(
    monkeypatch: pytest.MonkeyPatch,
    interruption: type[BaseException],
) -> None:
    calls = []

    class Structured:
        def invoke(self, *args: Any, **kwargs: Any) -> None:
            calls.append(True)
            raise interruption()

    class Model:
        def with_structured_output(self, *args: Any, **kwargs: Any) -> Structured:
            return Structured()

    monkeypatch.setattr(adapter, "ChatOpenAI", lambda **kwargs: Model())
    with pytest.raises(interruption):
        adapter.OpenAIGenerationProvider(settings()).generate(
            schema=Output,
            system="Source only",
            content="evidence",
            max_output_tokens=30,
            timeout_seconds=1,
        )
    assert len(calls) == 1


def test_real_sdk_wrapped_soft_limit_is_unwrapped_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def interrupted(request: httpx2.Request) -> httpx2.Response:
        calls.append(True)
        raise SoftTimeLimitExceeded()

    with openai.OpenAI(
        api_key="unit-placeholder",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(interrupted)),
    ) as client:

        def factory(**kwargs: Any) -> OpenAIEmbeddings:
            return OpenAIEmbeddings(**kwargs, client=client.embeddings, async_client=object())

        monkeypatch.setattr(adapter, "OpenAIEmbeddings", factory)
        with pytest.raises(SoftTimeLimitExceeded):
            adapter.OpenAIEmbeddingProvider(settings()).embed_many(["evidence"], timeout_seconds=1)
    assert len(calls) == 1


def test_langchain_returned_soft_limit_sentinel_is_not_a_repairable_parse_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class Structured:
        def invoke(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            calls.append(True)
            return {
                "raw": adapter.AIMessage(
                    content="", response_metadata={"model_name": Settings.CHAT_MODEL}
                ),
                "parsed": None,
                "parsing_error": SoftTimeLimitExceeded(),
            }

    class Model:
        def with_structured_output(self, *args: Any, **kwargs: Any) -> Structured:
            return Structured()

    monkeypatch.setattr(adapter, "ChatOpenAI", lambda **kwargs: Model())
    with pytest.raises(SoftTimeLimitExceeded):
        adapter.OpenAIGenerationProvider(settings()).generate(
            schema=Output,
            system="Source only",
            content="evidence",
            max_output_tokens=30,
            timeout_seconds=1,
        )
    assert len(calls) == 1


def _completion(
    content: str,
    *,
    finish: str = "stop",
    model: str = Settings.CHAT_MODEL,
    refusal: str | None = None,
) -> dict[str, Any]:
    return {
        "id": "offline-response",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": finish,
                "message": {"role": "assistant", "content": content, "refusal": refusal},
            }
        ],
        "usage": {"prompt_tokens": 71, "completion_tokens": 29, "total_tokens": 100},
    }


@pytest.mark.parametrize(
    "problem,code,reason",
    [
        ("length_complete", "ai_output_truncated", "output_token_limit"),
        ("length_partial", "ai_output_truncated", "output_token_limit"),
        ("partial_json", "ai_invalid_json", "invalid_json"),
        ("invalid_uuid", "ai_schema_invalid", "schema_validation"),
        ("nested_quote", "ai_schema_invalid", "schema_validation"),
        ("extra_field", "ai_schema_invalid", "schema_validation"),
        ("model", "ai_model_mismatch", "structured_parse_failure"),
        ("refusal", "ai_refused", "structured_parse_failure"),
    ],
)
def test_strict_raw_generation_failures_preserve_only_safe_diagnostics_one_request(
    monkeypatch: pytest.MonkeyPatch,
    problem: str,
    code: str,
    reason: str,
) -> None:
    from app.services.analysis import AnalysisOutput

    source = "00000000-0000-0000-0000-000000000001"
    block = {
        "text": "CANARY_PRIVATE_TEXT",
        "source_ids": [source],
        "support_quotes": [{"source_id": source, "quote": "CANARY_PRIVATE_TEXT"}],
    }
    payload = {"summary": [block], "insights": []}
    if problem == "invalid_uuid":
        block["source_ids"] = ["CANARY_PRIVATE_TEXT"]
    elif problem == "nested_quote":
        block["support_quotes"][0]["quote"] = "x" * 301
    elif problem == "extra_field":
        block["private_field"] = "CANARY_PRIVATE_TEXT"
    content = json.dumps(payload)
    if problem in {"length_partial", "partial_json"}:
        content = content[:-2]  # LangChain can salvage this; the application must reject it.
    requests = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        body = json.loads(request.content)
        assert body["response_format"]["json_schema"]["strict"] is True
        assert (
            body["response_format"]["json_schema"]["schema"]["$defs"]["SourceQuote"]["properties"][
                "source_id"
            ]["format"]
            == "uuid"
        )
        return httpx2.Response(
            200,
            headers={"x-request-id": "req_safe-test_123"},
            json=_completion(
                content,
                finish="length" if problem.startswith("length") else "stop",
                model="wrong-model" if problem == "model" else Settings.CHAT_MODEL,
                refusal="CANARY_PRIVATE_TEXT" if problem == "refusal" else None,
            ),
        )

    def factory(**kwargs: Any) -> ChatOpenAI:
        assert kwargs["max_retries"] == 0
        return ChatOpenAI(
            **kwargs,
            http_socket_options=(),
            http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
            http_async_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)),
        )

    monkeypatch.setattr(adapter, "ChatOpenAI", factory)
    with pytest.raises(ProcessingError) as error:
        adapter.OpenAIGenerationProvider(settings()).generate(
            schema=AnalysisOutput,
            system="Source only",
            content="synthetic evidence",
            max_output_tokens=100,
            timeout_seconds=2,
        )
    assert len(requests) == 1
    assert error.value.code == code and not error.value.retryable
    assert error.value.diagnostics is not None
    details = error.value.diagnostics.as_dict()
    assert details == {
        "reason": reason,
        "finish_reason": "length" if problem.startswith("length") else "stop",
        "input_tokens": 71,
        "output_tokens": 29,
        "max_output_tokens": 100,
        "provider_request_id": "req_safe-test_123",
    }
    assert "CANARY_PRIVATE_TEXT" not in json.dumps(details)
    assert "CANARY_PRIVATE_TEXT" not in "".join(traceback.format_exception(error.value))


@pytest.mark.parametrize(
    "raw,reason",
    [
        (None, "missing_raw_response"),
        (
            adapter.AIMessage(content="", response_metadata={"model_name": Settings.CHAT_MODEL}),
            "missing_parsed_response",
        ),
    ],
)
def test_missing_generation_payload_is_safe(
    monkeypatch: pytest.MonkeyPatch, raw: Any, reason: str
) -> None:
    class Structured:
        def invoke(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {
                "raw": raw,
                "parsed": {"answer": "MUST_NOT_TRUST_PARSED"},
                "parsing_error": None,
            }

    class Model:
        def with_structured_output(self, *args: Any, **kwargs: Any) -> Structured:
            return Structured()

    monkeypatch.setattr(adapter, "ChatOpenAI", lambda **kwargs: Model())
    with pytest.raises(ProcessingError) as error:
        adapter.OpenAIGenerationProvider(settings()).generate(
            schema=Output,
            system="source",
            content="evidence",
            max_output_tokens=99,
            timeout_seconds=1,
        )
    assert error.value.diagnostics.reason == reason


def test_direct_sdk_length_exception_retains_usage() -> None:
    completion = openai.types.chat.ChatCompletion.model_validate(
        _completion('{"answer":"partial"}', finish="length")
    )
    error = adapter.provider_error(
        openai.LengthFinishReasonError(completion=completion), max_output_tokens=100
    )
    assert error.code == "ai_output_truncated"
    assert error.diagnostics.as_dict()["input_tokens"] == 71
    assert error.diagnostics.as_dict()["output_tokens"] == 29
    assert error.diagnostics.as_dict()["max_output_tokens"] == 100


def test_wrapped_caller_cancellation_is_never_a_retryable_provider_error() -> None:
    failure = openai.APIConnectionError(request=httpx2.Request("POST", "https://invalid.example"))
    failure.__cause__ = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        adapter.provider_error(failure)
