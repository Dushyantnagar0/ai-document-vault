"""Float32 vector validity is consistent before provider return, publish and query."""

import struct
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from app.adapters.async_openai import AsyncOpenAIEmbeddingProvider
from app.adapters.openai import OpenAIEmbeddingProvider
from app.adapters.retrieval import _query_vector
from app.adapters.vectors import PgVectorPublisher
from app.core.config import Settings
from app.domain.documents import Chunk, ProcessingError
from app.domain.vectors import float32_vector

pytestmark = pytest.mark.unit
DIMENSIONS = 1536


def vector(first: Any = 1.0) -> list[Any]:
    return [first] + [0.0] * (DIMENSIONS - 1)


INVALID = [
    [],
    [1.0],
    [0.0] * DIMENSIONS,
    [-0.0] * DIMENSIONS,
    [1e-46] * DIMENSIONS,
    vector(float("nan")),
    vector(float("inf")),
    vector(float("-inf")),
    vector(1e100),
    vector(10**500),
    vector(True),
    vector(False),
    vector("1.0"),
    vector(None),
    vector([]),
    "1" * DIMENSIONS,
]


@pytest.mark.parametrize("values", INVALID)
def test_unusable_vectors_fail_shared_validator_and_query_without_content(values: Any) -> None:
    with pytest.raises(ValueError, match="finite nonzero float32") as error:
        float32_vector(values, DIMENSIONS)
    assert "nan" not in str(error.value).lower()
    with pytest.raises(ValueError, match="1536-dimensional"):
        _query_vector(values)


class Capture:
    def __init__(self) -> None:
        self.rows: list[Any] = []
        self.calls = 0

    def add_all(self, rows: list[Any]) -> None:
        self.rows.extend(rows)
        self.calls += 1


@pytest.mark.parametrize("values", INVALID)
def test_invalid_later_vector_prevents_every_row_being_scheduled(values: Any) -> None:
    session = Capture()
    chunks = [Chunk(uuid4(), i, "Original evidence.", 3, ()) for i in range(2)]
    with pytest.raises(ProcessingError) as error:
        PgVectorPublisher(cast(Session, session)).publish(
            uuid4(), uuid4(), uuid4(), chunks, [vector(), values]
        )
    assert error.value.code == "INVALID_VECTORS" and not error.value.retryable
    assert session.calls == 0 and session.rows == []


@pytest.mark.parametrize("kind", ["sync", "async"])
@pytest.mark.parametrize("values", INVALID)
async def test_embedding_invalid_float32_data_is_one_terminal_observed_call(
    kind: str, values: Any
) -> None:
    configured = Settings(_env_file=None, openai_api_key="unit-placeholder")
    calls: list[bool] = []

    def respond(**kwargs: Any) -> dict[str, Any]:
        calls.append(True)
        return {
            "model": configured.openai_embedding_model,
            "data": [{"index": 0, "embedding": values}],
            "usage": {"prompt_tokens": 7},
        }

    async def respond_async(**kwargs: Any) -> dict[str, Any]:
        return respond(**kwargs)

    provider: Any = object.__new__(
        OpenAIEmbeddingProvider if kind == "sync" else AsyncOpenAIEmbeddingProvider
    )
    provider.settings = configured
    provider._embeddings = SimpleNamespace(
        client=SimpleNamespace(create=respond), async_client=SimpleNamespace(create=respond_async)
    )
    with pytest.raises(ProcessingError) as error:
        if kind == "sync":
            provider.embed_many(["Evidence."], timeout_seconds=1)
        else:
            await provider.embed_many(["Evidence."], timeout_seconds=1)
    assert error.value.code == "embedding_invalid_output" and not error.value.retryable
    assert calls == [True]
    assert error.value.diagnostics is not None
    assert error.value.diagnostics.input_tokens == 7
    assert error.value.diagnostics.output_tokens is None


@pytest.mark.parametrize(
    "values", [vector(1), vector(-1.0), vector(0.1), [1e-46, 0.1] + [0.0] * 1534]
)
def test_valid_values_are_rounded_at_publication_precision_without_normalizing(
    values: list[float],
) -> None:
    expected = [struct.unpack("!f", struct.pack("!f", value))[0] for value in values]
    assert float32_vector(values, DIMENSIONS) == expected
    assert _query_vector(values) == expected
    session = Capture()
    chunk = Chunk(uuid4(), 0, "Original evidence.", 3, ())
    PgVectorPublisher(cast(Session, session)).publish(uuid4(), uuid4(), uuid4(), [chunk], [values])
    assert session.calls == 1 and len(session.rows) == 1
    assert session.rows[0].embedding == expected
