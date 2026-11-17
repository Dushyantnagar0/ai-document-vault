"""Public chat HTTP contracts against isolated real services and a provider fixture."""

import asyncio
import hashlib
import json
import os
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from uuid import UUID, uuid4

import httpx
import pymupdf
import pytest
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session

from app.adapters.async_openai import AsyncOpenAIEmbeddingProvider, AsyncOpenAIGenerationProvider
from app.adapters.retrieval import PgVectorRetriever
from app.core.config import get_settings
from app.db.models import ChatTurn, Document, DocumentChunk, PerformanceSample
from app.db.session import api_engine
from app.domain.conversation import Replay, TurnReservation
from app.domain.documents import Generated
from app.services.chat import ChatProcessor
from app.services.chat_review import CHAT_REVIEW_PROMPT_VERSION, ChatReviewOutput
from app.services.conversations import TurnStore
from app.services.grounding import INSUFFICIENT_ANSWER

pytestmark = [
    pytest.mark.integration,
    pytest.mark.enable_socket,
    pytest.mark.skipif(
        os.environ.get("VAULT_INTEGRATION_TESTS") != "1",
        reason="Requires isolated Compose stack and provider fixture",
    ),
]
TEST_AUTHORIZATION = "Bearer test-placeholder-never-a-real-key"


@pytest.fixture(scope="module")
def database() -> Iterator[Engine]:
    configured = get_settings()
    if (
        configured.postgres_db != "vault_test"
        or configured.openai_api_key.get_secret_value() != "test-placeholder-never-a-real-key"
        or os.environ.get("OPENAI_BASE_URL") != "http://provider-stub:4010/v1"
    ):
        pytest.fail("Refusing non-test database/provider configuration", pytrace=False)
    engine = create_engine(configured.database_url)
    yield engine
    engine.dispose()


@pytest.fixture(scope="module")
def api(database: Engine) -> Iterator[httpx.Client]:
    with httpx.Client(base_url="http://api:8000", timeout=10) as client:
        yield client


def _wait_document(api: httpx.Client, identifier: str) -> dict[str, Any]:
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        response = api.get(f"/api/v1/documents/{identifier}")
        assert response.status_code == 200
        body: dict[str, Any] = response.json()
        if body["status"] in {"COMPLETED", "FAILED"}:
            assert body["status"] == "COMPLETED", body
            return body
        time.sleep(0.1)
    pytest.fail("Fixture document did not finish within40s")


@pytest.fixture(scope="module")
def documents(api: httpx.Client) -> dict[str, str]:
    """Exactly two uploads per module provide intentionally conflicting document facts."""
    records = {
        "juniper": (
            "Project Juniper agreement\nOwner: Priya Shah\n"
            "Renewal date: 12 December 2027\nAnnual fee: 240 USD\n"
        ),
        "maple": (
            "Project Maple agreement\nOwner: Alex Rivers\n"
            "Renewal date: 20 January 2028\nAnnual fee: 360 USD\n"
        ),
    }
    identifiers = {}
    for name, text in records.items():
        with pymupdf.open() as document:
            document.new_page().insert_text((72, 72), text)
            content = document.tobytes()
        response = api.post(
            "/api/v1/documents/upload",
            files={"file": (f"chat-{name}.pdf", content, "application/pdf")},
            headers={"Idempotency-Key": str(uuid4())},
        )
        assert response.status_code == 202
        identifier = response.json()["document_id"]
        state = _wait_document(api, identifier)
        assert state["chunk_count"] == 1 and state["ai_calls_reserved"] == 4
        identifiers[name] = identifier
    return identifiers


def _session(api: httpx.Client, document_id: str, title: str | None = None) -> str:
    payload = {"document_id": document_id}
    if title:
        payload["title"] = title
    response = api.post("/api/v1/chat/sessions", json=payload)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["document_ids"] == [document_id]
    return str(body["id"])


def _ask(
    api: httpx.Client, session_id: str, question: str, request_id: str | None = None
) -> httpx.Response:
    return api.post(
        f"/api/v1/chat/sessions/{session_id}/messages",
        json={"question": question, "client_request_id": request_id or str(uuid4())},
    )


def _history(api: httpx.Client, session_id: str, **params: Any) -> dict[str, Any]:
    response = api.get(f"/api/v1/chat/sessions/{session_id}/messages", params=params)
    assert response.status_code == 200
    return dict(response.json())


def _stats() -> dict[str, Any]:
    response = httpx.get(
        "http://provider-stub:4010/stats", headers={"Authorization": TEST_AUTHORIZATION}, timeout=5
    )
    assert response.status_code == 200
    return dict(response.json())


def _count(stats: dict[str, Any], operation: str) -> int:
    return int(stats["counts"].get(operation, 0))


def _question_count(stats: dict[str, Any], question: str) -> int:
    return int(stats["planner_questions"].get(hashlib.sha256(question.encode()).hexdigest(), 0))


def _turns(database: Engine, session_id: str) -> list[dict[str, Any]]:
    with Session(database) as session:
        rows = session.scalars(
            select(ChatTurn)
            .where(ChatTurn.session_id == UUID(session_id))
            .order_by(ChatTurn.turn_number)
        ).all()
        return [
            {
                "id": str(turn.id),
                "number": turn.turn_number,
                "status": turn.status,
                "calls": turn.ai_calls_reserved,
                "query": turn.retrieval_query,
                "context": turn.context_snapshot,
                "response": turn.response_body,
            }
            for turn in rows
        ]


def _document_snapshot(database: Engine, identifier: str) -> tuple[int, tuple[str, ...]]:
    with Session(database) as session:
        document = session.get(Document, UUID(identifier))
        assert document is not None
        ids = session.scalars(
            select(DocumentChunk.id)
            .where(DocumentChunk.document_id == document.id)
            .order_by(DocumentChunk.ordinal)
        ).all()
        return document.ai_calls_reserved, tuple(str(item) for item in ids)


def _citations(database: Engine, body: dict[str, Any], document_id: str) -> None:
    assert body["citations"]
    with Session(database) as session:
        for citation in body["citations"]:
            assert citation["document_id"] == document_id
            chunk = session.get(DocumentChunk, UUID(citation["source_id"]))
            assert chunk is not None and str(chunk.document_id) == document_id
            assert citation["excerpt"] in chunk.text and len(citation["excerpt"]) <= 1200
            assert citation["locators"] and all(
                locator.get("page") == 1 for locator in citation["locators"]
            )


def _pending(database: Engine, session_id: str) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        turns = _turns(database, session_id)
        if turns and turns[-1]["status"] == "PENDING" and turns[-1]["calls"] >= 1:
            return
        time.sleep(0.01)
    pytest.fail("Slow fixture turn was not reserved within3s")


def test_grounded_followup_history_replay_and_persisted_vectors_reused(
    api: httpx.Client,
    database: Engine,
    documents: dict[str, str],
) -> None:
    identifier = documents["juniper"]
    session_id = _session(api, identifier, "Juniper conversation")
    before_document, before = _document_snapshot(database, identifier), _stats()
    request_id, question = str(uuid4()), "Who owns this agreement?"
    first = _ask(api, session_id, question, request_id)
    assert first.status_code == 200, first.text
    assert "Priya Shah" in first.json()["answer"]
    _citations(database, first.json(), identifier)
    followup = _ask(api, session_id, "And when is its renewal date?")
    assert followup.status_code == 200 and "12 December 2027" in followup.json()["answer"]
    _citations(database, followup.json(), identifier)
    after_calls = _stats()
    replay = _ask(api, session_id, question, request_id)
    assert replay.status_code == 200 and replay.json() == first.json()
    conflict = _ask(api, session_id, "A different question", request_id)
    assert (
        conflict.status_code == 409 and conflict.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"
    )
    after_replay = _stats()
    assert after_replay == after_calls
    assert _count(after_calls, "embeddings") - _count(before, "embeddings") == 2
    assert _count(after_calls, "PlannerOutput") - _count(before, "PlannerOutput") == 2
    assert (
        _count(after_calls, "AnswerSelectionOutput") - _count(before, "AnswerSelectionOutput") == 2
    )
    assert _count(after_calls, "ChatReviewOutput") - _count(before, "ChatReviewOutput") == 2
    assert _document_snapshot(database, identifier) == before_document
    turns = _turns(database, session_id)
    assert len(turns) == 2 and [turn["calls"] for turn in turns] == [4, 4]
    for turn in turns:
        review = turn["context"]["evidence_review"]
        assert review["prompt_version"] == CHAT_REVIEW_PROMPT_VERSION
        assert review["scope_complete"] and not review["force_qualification_quotes"]
        assert review["qualification_policy"] == "retained_claim_source_bindings"
        assert review["review"]["decisions"] and review["quote_bindings"]
        assert review["claims"] and len(review["request_sha256"]) == 64
        with Session(database) as session:
            samples = session.scalars(
                select(PerformanceSample).where(
                    PerformanceSample.turn_id == UUID(turn["id"]),
                    PerformanceSample.operation == "chat.evidence_review",
                )
            ).all()
            assert len(samples) == 1 and samples[0].outcome == "succeeded"
            assert samples[0].input_tokens == 100 and samples[0].output_tokens == 60
    assert len(turns[1]["context"]["payload"]["history"]) == 2
    assert question in turns[1]["query"]
    messages = _history(api, session_id)["items"]
    assert [message["sequence"] for message in messages] == [1, 2, 3, 4]
    assert [message["role"] for message in messages] == ["user", "assistant", "user", "assistant"]
    assert all(message["turn_status"] == "COMPLETED" for message in messages)


def test_session_filter_pagination_and_wrong_document_isolation(
    api: httpx.Client,
    database: Engine,
    documents: dict[str, str],
) -> None:
    juniper_a = _session(api, documents["juniper"], "First Juniper")
    juniper_b = _session(api, documents["juniper"], "Second Juniper")
    maple = _session(api, documents["maple"], "Maple scope")
    for session_id, key, owner, excluded in (
        (juniper_a, "juniper", "Priya Shah", "Alex Rivers"),
        (maple, "maple", "Alex Rivers", "Priya Shah"),
    ):
        answer = _ask(api, session_id, "Who owns this agreement?")
        assert answer.status_code == 200
        assert owner in answer.json()["answer"] and excluded not in answer.json()["answer"]
        _citations(database, answer.json(), documents[key])
    found, cursor = [], None
    for _ in range(30):
        params = {"document_id": documents["juniper"], "limit": 1}
        if cursor:
            params["cursor"] = cursor
        response = api.get("/api/v1/chat/sessions", params=params)
        assert response.status_code == 200
        page = response.json()
        assert all(item["document_ids"] == [documents["juniper"]] for item in page["items"])
        found.extend(item["id"] for item in page["items"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
    else:
        pytest.fail("Session pagination did not terminate")
    assert {juniper_a, juniper_b}.issubset(found) and maple not in found
    assert len(found) == len(set(found))


def test_insufficient_overview_and_tiny_exhaustive_answers_use_original_sources(
    api: httpx.Client,
    database: Engine,
    documents: dict[str, str],
) -> None:
    identifier = documents["juniper"]
    session_id = _session(api, identifier)
    unsupported = _ask(api, session_id, "What cancellation penalty applies?")
    assert unsupported.status_code == 200
    assert unsupported.json()["evidence_status"] == "insufficient"
    assert unsupported.json()["citations"] == [] and "cannot find" in unsupported.json()["answer"]
    unsupported_turn = _turns(database, session_id)[0]
    assert unsupported_turn["calls"] == 3
    assert "evidence_review" not in unsupported_turn["context"]
    before = _stats()
    overview = _ask(api, session_id, "Give an overview of this document.")
    assert overview.status_code == 200, overview.text
    body = overview.json()
    assert body["evidence_scope"] == "document_summary" and body["evidence_status"] == "answered"
    assert "Priya Shah" in body["answer"] and "240 USD" in body["answer"]
    assert "Deterministic fixture analysis" not in body["answer"]
    _citations(database, body, identifier)
    assert _count(_stats(), "embeddings") == _count(before, "embeddings")
    overview_turn = _turns(database, session_id)[1]
    assert overview_turn["calls"] == 3
    assert not overview_turn["context"]["evidence_review"]["force_qualification_quotes"]
    exhaustive = _ask(api, session_id, "List all facts in this document.")
    assert exhaustive.status_code == 200
    assert exhaustive.json()["evidence_status"] == "answered"
    assert exhaustive.json()["evidence_scope"] == "retrieved_excerpts"
    assert "12 December 2027" in exhaustive.json()["answer"]
    _citations(database, exhaustive.json(), identifier)


def test_quota_failure_persistent_replay_and_new_turn_after_failure(
    api: httpx.Client,
    database: Engine,
    documents: dict[str, str],
) -> None:
    session_id = _session(api, documents["juniper"])
    question, request_id = "[fixture quota] Who owns this document?", str(uuid4())
    before = _stats()
    failed = _ask(api, session_id, question, request_id)
    assert failed.status_code == 502 and failed.json()["error"]["code"] == "ai_quota"
    assert "test-placeholder-never-a-real-key" not in failed.text
    replay = _ask(api, session_id, question, request_id)
    assert replay.status_code == 502 and replay.json() == failed.json()
    after = _stats()
    assert _question_count(after, question) - _question_count(before, question) == 1
    assert _count(after, "AnswerSelectionOutput") == _count(before, "AnswerSelectionOutput")
    assert _count(after, "embeddings") == _count(before, "embeddings")
    failed_turn = _turns(database, session_id)[0]
    assert failed_turn["status"] == "FAILED" and failed_turn["calls"] == 1
    answered = _ask(api, session_id, "Who owns this agreement?")
    assert answered.status_code == 200 and "Priya Shah" in answered.json()["answer"]
    turns = _turns(database, session_id)
    assert turns[1]["context"]["payload"]["history"] == []
    messages = _history(api, session_id)["items"]
    assert [message["sequence"] for message in messages] == [1, 2, 3, 4]
    assert messages[1]["status"] == "FAILED" and messages[3]["status"] == "COMPLETED"


@pytest.mark.parametrize("recover", [False, True], ids=["terminal-failure", "repaired-review"])
def test_invalid_review_retains_bounded_audit_and_replay_spends_no_calls(
    api: httpx.Client,
    database: Engine,
    documents: dict[str, str],
    recover: bool,
) -> None:
    """Actual service audit repair/failure persists; HTTP replays without new work.

    Planner, embedding and draft use the isolated HTTP provider fixture. A local
    reviewer deliberately selects a nonexistent qualification passage, then either
    repairs it or repeats the defect. This does not establish live AI quality.
    """
    session_id = _session(api, documents["juniper"])
    question, request_id = "Who owns this agreement?", str(uuid4())
    review_requests: list[dict[str, Any]] = []

    class InvalidQualificationReviewer:
        async def generate(self, **kwargs: Any) -> Generated[ChatReviewOutput]:
            review_requests.append(kwargs)
            packet = json.loads(kwargs["content"])
            valid = recover and len(review_requests) == 2
            return Generated(
                ChatReviewOutput.model_validate(
                    {
                        "required_qualifications": []
                        if valid
                        else [
                            {"quote_ids": ["missing-source-canary"], "conveyed_by_claim_ids": []}
                        ],
                        "decisions": [
                            {
                                "claim_id": claim["claim_id"],
                                "supported": True,
                                "quote_ids": [
                                    next(
                                        source["quote_id"]
                                        for source in packet["evidence"]
                                        if claim["text"].strip() in source["text"]
                                    )
                                ],
                                "redundant_with": [],
                            }
                            for claim in packet["claims"]
                        ],
                        "scope_complete": True,
                        "answerability": "answered",
                        "unanswered_parts": [],
                        "relevant_claim_ids": [claim["claim_id"] for claim in packet["claims"]],
                    }
                ),
                "deterministic-invalid-review",
                100,
                60,
            )

    async def execute() -> Replay:
        settings = get_settings()
        engine = api_engine(settings)
        try:
            store = TurnStore(engine, settings)
            processor = ChatProcessor(
                engine,
                settings,
                store,
                PgVectorRetriever(engine),
                AsyncOpenAIGenerationProvider(settings),
                AsyncOpenAIEmbeddingProvider(settings),
                reviewer=InvalidQualificationReviewer(),
            )
            reservation = await store.reserve(
                UUID(session_id), UUID(request_id), question, str(uuid4())
            )
            assert isinstance(reservation, TurnReservation)
            return await processor.run(reservation)
        finally:
            await engine.dispose()

    result = asyncio.run(execute())
    assert len(review_requests) == 2
    assert review_requests[0]["content"] == review_requests[1]["content"]
    assert review_requests[1]["system"].startswith(review_requests[0]["system"])
    assert review_requests[1]["system"] != review_requests[0]["system"]
    assert "missing-source-canary" not in review_requests[1]["content"]
    assert "missing-source-canary" not in review_requests[1]["system"]
    turns = _turns(database, session_id)
    assert len(turns) == 1 and turns[0]["calls"] == 5
    context = turns[0]["context"]
    review = context["evidence_review"]
    assert review["review_attempt"] == 2
    assert review["repair"]["attempted"] is True
    assert review["repair"]["first_error_code"] == "EVIDENCE_REVIEW_INVALID"
    assert review["repair"]["first_error"]["reason"] == "invalid_qualification_selection"
    assert review["claims"] and len(review["claims"]) <= 64
    assert review["quote_bindings"] and "missing-source-canary" not in review["quote_bindings"]
    assert len(review["request_sha256"]) == 64
    assert "evidence_review" not in context["payload"]
    history = _history(api, session_id)["items"]
    assert len(history) == 2
    if recover:
        assert result.status == 200 and "Priya Shah" in result.body["answer"]
        assert turns[0]["status"] == "COMPLETED" and history[-1]["status"] == "COMPLETED"
        assert review["projection_status"] == "completed"
        assert "projection_error" not in review
        assert review["review"]["required_qualifications"] == []
        assert history[-1]["content"] == result.body["answer"]
        _citations(database, result.body, documents["juniper"])
    else:
        assert result.status == 502
        assert result.body["error"]["code"] == "EVIDENCE_REVIEW_INVALID"
        assert result.body["error"]["details"]["reason"] == "invalid_qualification_selection"
        assert "missing-source-canary" not in json.dumps(result.body)
        assert turns[0]["status"] == "FAILED" and history[-1]["status"] == "FAILED"
        assert review["projection_status"] == "failed"
        assert review["projection_error"]["reason"] == "invalid_qualification_selection"
        assert review["review"]["required_qualifications"][0]["quote_ids"] == [
            "missing-source-canary"
        ]
        assert history[-1]["content"] == result.body["error"]["message"]
    before_replay = _stats()
    replay = _ask(api, session_id, question, request_id)
    assert replay.status_code == result.status and replay.json() == result.body
    assert _stats() == before_replay
    assert len(review_requests) == 2
    assert _turns(database, session_id) == turns


def test_irrelevant_but_source_valid_draft_abstains_and_replays_without_calls(
    api: httpx.Client,
    database: Engine,
    documents: dict[str, str],
) -> None:
    """Injected relevance decisions exercise real persistence, not model semantics.

    The source-backed draft deliberately answers an ownership question even though
    the user asks about a different topic. A source-valid qualification must not
    be appended after the chat reviewer finds no answerable requested content.
    """
    session_id = _session(api, documents["juniper"])
    request_id = str(uuid4())
    question = "Can you help me to understand about kafka?"
    review_requests: list[dict[str, Any]] = []

    class UnrelatedDraftGenerator:
        async def generate(self, **kwargs: Any) -> Generated[Any]:
            if kwargs["schema"].__name__ != "AnswerSelectionOutput":
                return await AsyncOpenAIGenerationProvider(get_settings()).generate(**kwargs)
            packet = json.loads(kwargs["content"])
            passage = next(
                passage
                for source in packet["evidence"]
                for passage in source["passages"]
                if "Owner:" in passage["text"]
            )
            return Generated(
                kwargs["schema"].model_validate(
                    {
                        "evidence_status": "answered",
                        "blocks": [
                            {"text": passage["text"].strip(), "quote_ids": [passage["quote_id"]]}
                        ],
                    }
                ),
                "deterministic-unrelated-draft",
                100,
                60,
            )

    class UnanswerableReviewer:
        async def generate(self, **kwargs: Any) -> Generated[ChatReviewOutput]:
            review_requests.append(kwargs)
            packet = json.loads(kwargs["content"])
            return Generated(
                ChatReviewOutput.model_validate(
                    {
                        "answerability": "insufficient",
                        "unanswered_parts": [],
                        "relevant_claim_ids": [],
                        "scope_complete": False,
                        "required_qualifications": [
                            {
                                "quote_ids": [packet["evidence"][0]["quote_id"]],
                                "conveyed_by_claim_ids": [],
                            }
                        ],
                        "decisions": [
                            {
                                "claim_id": claim["claim_id"],
                                "supported": True,
                                "quote_ids": [
                                    next(
                                        source["quote_id"]
                                        for source in packet["evidence"]
                                        if claim["text"].strip() in source["text"]
                                    )
                                ],
                                "redundant_with": [],
                            }
                            for claim in packet["claims"]
                        ],
                    }
                ),
                "deterministic-topic-review",
                100,
                60,
            )

    async def execute() -> Replay:
        settings = get_settings()
        engine = api_engine(settings)
        try:
            store = TurnStore(engine, settings)
            processor = ChatProcessor(
                engine,
                settings,
                store,
                PgVectorRetriever(engine),
                UnrelatedDraftGenerator(),
                AsyncOpenAIEmbeddingProvider(settings),
                reviewer=UnanswerableReviewer(),
            )
            reservation = await store.reserve(
                UUID(session_id), UUID(request_id), question, str(uuid4())
            )
            assert isinstance(reservation, TurnReservation)
            return await processor.run(reservation)
        finally:
            await engine.dispose()

    result = asyncio.run(execute())
    assert result.status == 200, result.body
    assert result.body["evidence_status"] == "insufficient"
    assert result.body["answer"] == INSUFFICIENT_ANSWER
    assert result.body["citations"] == []
    assert len(review_requests) == 1
    turns = _turns(database, session_id)
    assert len(turns) == 1 and turns[0]["calls"] == 4
    assert turns[0]["status"] == "COMPLETED"
    assert turns[0]["context"]["evidence_review"]["review"]["answerability"] == "insufficient"
    messages = _history(api, session_id)["items"]
    assert [message["sequence"] for message in messages] == [1, 2]
    assert messages[-1]["content"] == INSUFFICIENT_ANSWER
    assert messages[-1]["citations"] == []
    before_replay = _stats()
    replay = _ask(api, session_id, question, request_id)
    assert replay.status_code == 200 and replay.json() == result.body
    assert _stats() == before_replay and len(review_requests) == 1


def test_slow_turn_busy_inprogress_and_mismatch_are_ordered_before_admission(
    api: httpx.Client,
    database: Engine,
    documents: dict[str, str],
) -> None:
    session_id = _session(api, documents["juniper"])
    old_id, old_question = str(uuid4()), "Who owns this agreement?"
    old = _ask(api, session_id, old_question, old_id)
    assert old.status_code == 200
    request_id, question = str(uuid4()), "[fixture slow] Who owns this agreement?"
    before = _stats()
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(_ask, api, session_id, question, request_id)
        _pending(database, session_id)
        replay = _ask(api, session_id, old_question, old_id)
        assert replay.status_code == 200 and replay.json() == old.json()
        busy = _ask(api, session_id, "What is its renewal date?")
        in_progress = _ask(api, session_id, question, request_id)
        mismatch = _ask(api, session_id, "A changed question", request_id)
        assert busy.status_code == 409 and busy.json()["error"]["code"] == "CHAT_BUSY"
        assert (
            in_progress.status_code == 409
            and in_progress.json()["error"]["code"] == "CHAT_IN_PROGRESS"
        )
        assert (
            mismatch.status_code == 409
            and mismatch.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"
        )
        completed = pending.result(timeout=5)
    assert completed.status_code == 200 and "Priya Shah" in completed.json()["answer"]
    turns = _turns(database, session_id)
    assert len(turns) == 2 and all(turn["status"] == "COMPLETED" for turn in turns)
    assert _question_count(_stats(), question) - _question_count(before, question) == 1


def test_client_read_timeout_recovers_history_and_replay_without_extra_provider_calls(
    api: httpx.Client,
    database: Engine,
    documents: dict[str, str],
) -> None:
    session_id = _session(api, documents["juniper"])
    question, request_id = "[fixture slow] What is the annual fee?", str(uuid4())
    before = _stats()
    with httpx.Client(base_url="http://api:8000", timeout=httpx.Timeout(5, read=0.1)) as impatient:
        with pytest.raises(httpx.ReadTimeout):
            _ask(impatient, session_id, question, request_id)
    _pending(database, session_id)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        messages = _history(api, session_id)["items"]
        if len(messages) == 2 and messages[-1]["status"] == "COMPLETED":
            break
        time.sleep(0.1)
    else:
        pytest.fail("Disconnected accepted turn did not complete in history")
    assert "240 USD" in messages[-1]["content"]
    after_completion = _stats()
    replay = _ask(api, session_id, question, request_id)
    assert replay.status_code == 200
    turns = _turns(database, session_id)
    assert len(turns) == 1 and turns[0]["calls"] == 4
    assert replay.json() == turns[0]["response"]
    assert _stats() == after_completion
    assert _question_count(after_completion, question) - _question_count(before, question) == 1
    assert _count(after_completion, "embeddings") - _count(before, "embeddings") == 1
    assert (
        _count(after_completion, "AnswerSelectionOutput") - _count(before, "AnswerSelectionOutput")
        == 1
    )


def test_history_cursor_pagination_is_ordered_and_session_scoped(
    api: httpx.Client,
    documents: dict[str, str],
) -> None:
    session_id = _session(api, documents["juniper"])
    other = _session(api, documents["maple"])
    for question in ("Who owns this agreement?", "When is its renewal date?"):
        assert _ask(api, session_id, question).status_code == 200
    other_result = _ask(api, other, "What is the annual fee?")
    assert other_result.status_code == 200 and "360 USD" in other_result.json()["answer"]
    collected, cursor = [], None
    for _ in range(8):
        page = _history(api, session_id, limit=1, cursor=cursor or 0)
        assert len(page["items"]) == 1
        collected.extend(page["items"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
    else:
        pytest.fail("History pagination did not terminate")
    assert [message["sequence"] for message in collected] == [1, 2, 3, 4]
    assert len({message["id"] for message in collected}) == 4
    assert all(
        "Alex Rivers" not in message["content"] and "360 USD" not in message["content"]
        for message in collected
    )
    assert collected[-1]["turn_id"] != other_result.json()["turn_id"]
