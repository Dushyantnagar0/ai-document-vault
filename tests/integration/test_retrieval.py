"""Actual pgvector ranking/scope checks in a private migrated PostgreSQL schema."""

import math
import os
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import Session
from test_processing_claims import ClaimDatabase, require_claim
from test_processing_claims import claim_database as claim_database
from test_processing_claims import new_document as new_document

from app.adapters.retrieval import PgVectorRetriever
from app.db.models import Document, DocumentChunk, DocumentInsights, IndexProfile, ProcessingAttempt
from app.db.profiles import profile_values
from app.domain.documents import SourceSpan

pytestmark = [
    pytest.mark.integration,
    pytest.mark.enable_socket,
    pytest.mark.skipif(
        os.environ.get("VAULT_INTEGRATION_TESTS") != "1", reason="requires isolated Compose stack"
    ),
]


def vector(first: float = 1, second: float = 0) -> list[float]:
    return [first, second] + [0.0] * 1534


@dataclass(frozen=True)
class Published:
    document_id: UUID
    chunk_ids: tuple[UUID, ...]
    texts: tuple[str, ...]
    filename: str


@pytest.fixture
async def retrieval_engine(claim_database: ClaimDatabase) -> AsyncIterator[AsyncEngine]:
    assert claim_database.settings.postgres_db == "vault_test"
    with Session(claim_database.engine) as session:
        schema = session.scalar(text("SELECT current_schema()"))
    assert isinstance(schema, str) and schema.startswith("claims_test_")
    assert schema.removeprefix("claims_test_").isalnum()
    engine = create_async_engine(
        claim_database.settings.async_database_url,
        hide_parameters=True,
        pool_size=2,
        max_overflow=0,
        connect_args={
            "connect_timeout": 3,
            "options": f"-c search_path={schema},public -c statement_timeout=5000",
        },
    )
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
def publish(
    claim_database: ClaimDatabase, new_document: Callable[..., UUID]
) -> Callable[[Sequence[Sequence[float]]], Published]:
    def create(vectors: Sequence[Sequence[float]]) -> Published:
        identifier = new_document()
        claim = require_claim(claim_database, identifier)
        ids = tuple(uuid4() for _ in vectors)
        texts = tuple(f"Known original source passage {index}." for index in range(len(vectors)))
        filename = f"source-{identifier}.pdf"
        now = datetime.now(UTC)
        with Session(claim_database.engine) as session, session.begin():
            for index, (chunk_id, embedding, passage) in enumerate(
                zip(ids, vectors, texts, strict=True)
            ):
                span = SourceSpan(
                    block_id=f"page-{index + 1}",
                    locator={"page": index + 1},
                    document_start=index * 100,
                    document_end=index * 100 + len(passage),
                    chunk_start=0,
                    chunk_end=len(passage),
                )
                session.add(
                    DocumentChunk(
                        id=chunk_id,
                        document_id=identifier,
                        attempt_id=claim.attempt_id,
                        profile_id=claim.profile_id,
                        ordinal=index,
                        text=passage,
                        token_count=10,
                        source_spans=[asdict(span)],
                        embedding=list(embedding),
                    )
                )
            session.add(
                DocumentInsights(
                    document_id=identifier,
                    attempt_id=claim.attempt_id,
                    payload={
                        "summary": [{"text": texts[0], "source_ids": [str(ids[0])]}],
                        "insights": [],
                    },
                    covered_chunks=len(ids),
                    total_chunks=len(ids),
                    extraction_coverage="complete_within_capability",
                    warnings=[],
                    model=claim_database.settings.openai_chat_model,
                    prompt_version="retrieval-fixture",
                )
            )
            document = session.get(Document, identifier)
            attempt = session.get(ProcessingAttempt, claim.attempt_id)
            assert document is not None and attempt is not None
            document.filename = filename
            document.extracted_text, document.extracted_tokens = "\n".join(texts), len(ids) * 10
            document.page_count, document.block_count, document.chunk_count = (
                len(ids),
                len(ids),
                len(ids),
            )
            document.extractor_version = "fixture-v1"
            document.extraction_coverage = "complete_within_capability"
            document.status, document.stage = "COMPLETED", "COMPLETED"
            document.published_attempt_id = claim.attempt_id
            document.claim_token, document.lease_until = None, None
            document.updated_at, document.terminal_at = now, now
            attempt.outcome, attempt.stage, attempt.ended_at = "SUCCEEDED", "COMPLETED", now
        return Published(identifier, ids, texts, filename)

    return create


async def test_exact_cosine_ranking_and_allowlist_isolation(
    claim_database: ClaimDatabase,
    retrieval_engine: AsyncEngine,
    publish: Callable[[Sequence[Sequence[float]]], Published],
) -> None:
    document = publish([vector(), vector(0, 1), vector(1, 1)])
    other = publish([vector()])
    retriever = PgVectorRetriever(retrieval_engine)
    evidence = await retriever.search(
        [document.document_id], claim_database.profile_id, vector(), 10
    )
    assert [item.id for item in evidence] == [document.chunk_ids[i] for i in (0, 2, 1)]
    assert [item.score for item in evidence] == pytest.approx([1, 1 / math.sqrt(2), 0])
    assert all(item.filename == document.filename for item in evidence)
    assert all(item.document_id == document.document_id for item in evidence)
    assert evidence[0].text == document.texts[0]
    assert evidence[0].source_spans[0].locator == {"page": 1}
    assert evidence[0].source_spans[0].chunk_end == len(document.texts[0])
    both = await retriever.search(
        [document.document_id, other.document_id], claim_database.profile_id, vector(), 10
    )
    assert {item.document_id for item in both} == {document.document_id, other.document_id}


async def test_equal_distance_order_and_limit_are_deterministic(
    claim_database: ClaimDatabase,
    retrieval_engine: AsyncEngine,
    publish: Callable[[Sequence[Sequence[float]]], Published],
) -> None:
    document = publish([vector(), vector()])
    retriever = PgVectorRetriever(retrieval_engine)
    evidence = await retriever.search(
        [document.document_id], claim_database.profile_id, vector(), 1
    )
    assert [item.id for item in evidence] == [min(document.chunk_ids)]


async def test_get_chunks_preserves_requested_order_and_filters_hidden_ids(
    claim_database: ClaimDatabase,
    retrieval_engine: AsyncEngine,
    publish: Callable[[Sequence[Sequence[float]]], Published],
) -> None:
    document = publish([vector(), vector(0, 1)])
    other = publish([vector()])
    retriever = PgVectorRetriever(retrieval_engine)
    evidence = await retriever.get_chunks(
        [document.document_id],
        claim_database.profile_id,
        [
            document.chunk_ids[1],
            other.chunk_ids[0],
            uuid4(),
            document.chunk_ids[0],
            document.chunk_ids[1],
        ],
    )
    assert [item.id for item in evidence] == [document.chunk_ids[1], document.chunk_ids[0]]
    assert [item.text for item in evidence] == [document.texts[1], document.texts[0]]
    assert all(item.score is None for item in evidence)


async def test_empty_filters_return_no_evidence(
    claim_database: ClaimDatabase,
    retrieval_engine: AsyncEngine,
    publish: Callable[[Sequence[Sequence[float]]], Published],
) -> None:
    document = publish([vector()])
    retriever = PgVectorRetriever(retrieval_engine)
    assert await retriever.search([], claim_database.profile_id, vector(), 10) == ()
    assert await retriever.get_chunks([], claim_database.profile_id, document.chunk_ids) == ()
    assert await retriever.get_chunks([document.document_id], claim_database.profile_id, []) == ()


async def test_profile_mismatch_on_either_document_or_chunk_is_hidden(
    claim_database: ClaimDatabase,
    retrieval_engine: AsyncEngine,
    publish: Callable[[Sequence[Sequence[float]]], Published],
) -> None:
    document = publish([vector()])
    settings = claim_database.settings.model_copy(update={"chunk_size_tokens": 700})
    with Session(claim_database.engine) as session, session.begin():
        profile = IndexProfile(**profile_values(settings))
        session.add(profile)
        session.flush()
        other_profile = profile.id
        chunk = session.get(DocumentChunk, document.chunk_ids[0])
        assert chunk is not None
        chunk.profile_id = other_profile
    retriever = PgVectorRetriever(retrieval_engine)
    for profile_id in (claim_database.profile_id, other_profile, uuid4()):
        assert await retriever.search([document.document_id], profile_id, vector(), 10) == ()
        assert (
            await retriever.get_chunks([document.document_id], profile_id, document.chunk_ids) == ()
        )


@pytest.mark.parametrize("status", ["PENDING", "PROCESSING", "FAILED"])
async def test_unready_document_never_returns_retained_chunks(
    claim_database: ClaimDatabase,
    retrieval_engine: AsyncEngine,
    publish: Callable[[Sequence[Sequence[float]]], Published],
    status: str,
) -> None:
    document = publish([vector()])
    with Session(claim_database.engine) as session, session.begin():
        row = session.get(Document, document.document_id)
        assert row is not None
        row.status = status
    retriever = PgVectorRetriever(retrieval_engine)
    assert (
        await retriever.search([document.document_id], claim_database.profile_id, vector(), 10)
        == ()
    )
    assert (
        await retriever.get_chunks(
            [document.document_id], claim_database.profile_id, document.chunk_ids
        )
        == ()
    )


async def test_only_published_attempt_can_supply_evidence(
    claim_database: ClaimDatabase,
    retrieval_engine: AsyncEngine,
    publish: Callable[[Sequence[Sequence[float]]], Published],
) -> None:
    document = publish([vector()])
    current_id, current_attempt = uuid4(), uuid4()
    now = datetime.now(UTC)
    with Session(claim_database.engine) as session, session.begin():
        session.add(
            ProcessingAttempt(
                id=current_attempt,
                document_id=document.document_id,
                attempt_number=2,
                claim_token=uuid4(),
                requested_at=now,
                started_at=now,
                heartbeat_at=now,
                ended_at=now,
                outcome="SUCCEEDED",
                stage="COMPLETED",
                stage_started_at=now,
                stage_durations={},
            )
        )
        session.flush()
        session.add(
            DocumentChunk(
                id=current_id,
                document_id=document.document_id,
                attempt_id=current_attempt,
                profile_id=claim_database.profile_id,
                ordinal=1,
                text="Current evidence.",
                token_count=3,
                source_spans=[asdict(SourceSpan("current", {"page": 1}, 0, 17, 0, 17))],
                embedding=vector(0, 1),
            )
        )
        row = session.get(Document, document.document_id)
        analysis = session.get(DocumentInsights, document.document_id)
        assert row is not None and analysis is not None
        row.current_attempt_id = row.published_attempt_id = current_attempt
        row.attempt_count = 2
        analysis.attempt_id = current_attempt
        analysis.payload = {
            "summary": [{"text": "Current evidence.", "source_ids": [str(current_id)]}],
            "insights": [],
        }
    retriever = PgVectorRetriever(retrieval_engine)
    evidence = await retriever.search(
        [document.document_id], claim_database.profile_id, vector(), 10
    )
    assert [item.id for item in evidence] == [current_id]
    resolved = await retriever.get_chunks(
        [document.document_id], claim_database.profile_id, [*document.chunk_ids, current_id]
    )
    assert [item.id for item in resolved] == [current_id]


@pytest.mark.parametrize("limit", [0, 101, -1, True, 1.5])
async def test_invalid_limit_is_rejected(
    claim_database: ClaimDatabase,
    retrieval_engine: AsyncEngine,
    limit: Any,
) -> None:
    with pytest.raises(ValueError, match="limit"):
        await PgVectorRetriever(retrieval_engine).search(
            [uuid4()], claim_database.profile_id, vector(), limit
        )


@pytest.mark.parametrize(
    "query",
    [
        [1.0],
        [0.0] * 1536,
        [float("nan")] + [0.0] * 1535,
        [float("inf")] + [0.0] * 1535,
        [1e100] + [0.0] * 1535,
        [1e-100] + [0.0] * 1535,
    ],
)
async def test_invalid_query_is_rejected(
    claim_database: ClaimDatabase,
    retrieval_engine: AsyncEngine,
    query: list[float],
) -> None:
    with pytest.raises(ValueError, match="1536-dimensional"):
        await PgVectorRetriever(retrieval_engine).search(
            [uuid4()], claim_database.profile_id, query, 10
        )
