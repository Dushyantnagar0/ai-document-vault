"""Offline durable/concurrent request accounting; never calls a provider."""

import json
import multiprocessing
from pathlib import Path
from uuid import uuid4

import pytest

from scripts.live_budget import BudgetError, RequestBudget

FINGERPRINT = "a" * 64


def reserve_worker(path: str, limit: int, barrier, queue) -> None:
    budget = RequestBudget(path, limit)
    barrier.wait(timeout=10)
    accepted = 0
    for _ in range(limit):
        try:
            budget.reserve("parallel.probe", FINGERPRINT)
            accepted += 1
        except BudgetError:
            break
    queue.put(accepted)


def test_two_processes_cannot_exceed_shared_budget(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    barrier, queue = context.Barrier(2), context.Queue()
    path, limit = tmp_path / "requests.json", 7
    workers = [
        context.Process(target=reserve_worker, args=(str(path), limit, barrier, queue))
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=15)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)
            pytest.fail("Concurrent budget worker did not finish")
        assert worker.exitcode == 0
    assert sum(queue.get(timeout=2) for _ in workers) == limit
    assert RequestBudget(path, limit).snapshot() == {"used": limit, "remaining": 0}
    queue.close()
    queue.join_thread()
    state = json.loads(path.read_text())
    assert len({entry["id"] for entry in state["entries"]}) == limit


def test_interrupted_reservations_survive_resume_and_completion_never_refunds(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested" / "requests.json"
    budget = RequestBudget(path, limit=2)
    first = budget.reserve("review.long", FINGERPRINT)
    resumed = RequestBudget(path, limit=2)
    assert resumed.snapshot() == {"used": 1, "remaining": 1}
    second = resumed.reserve("review.long", FINGERPRINT)
    assert first != second  # Repeating the request fingerprint still spends a new slot.
    metadata = {"outcome": "uncertain", "input_tokens": None, "model": "gpt-test"}
    resumed.complete(first, metadata)
    resumed.complete(first, metadata)
    assert RequestBudget(path, limit=2).snapshot() == {"used": 2, "remaining": 0}
    saved = path.read_bytes()
    with pytest.raises(BudgetError, match="EXHAUSTED"):
        resumed.reserve("review.third", FINGERPRINT)
    assert path.read_bytes() == saved


@pytest.mark.parametrize("corrupt", ["{broken", "[]", '{"version":1,"limit":2,"entries":{}}'])
def test_corrupt_ledger_is_preserved_and_fails_closed(tmp_path: Path, corrupt: str) -> None:
    path = tmp_path / "requests.json"
    path.write_text(corrupt)
    saved = path.read_bytes()
    with pytest.raises(BudgetError):
        RequestBudget(path, limit=2)
    assert path.read_bytes() == saved


def test_existing_limit_and_completed_observations_cannot_be_rewritten(tmp_path: Path) -> None:
    path = tmp_path / "requests.json"
    budget = RequestBudget(path, limit=2)
    identifier = budget.reserve("analysis.review", FINGERPRINT)
    budget.complete(identifier, {"outcome": "failed", "output_tokens": 10})
    saved = path.read_bytes()
    with pytest.raises(BudgetError):
        RequestBudget(path, limit=3)
    with pytest.raises(BudgetError):
        budget.complete(identifier, {"outcome": "succeeded"})
    with pytest.raises(BudgetError):
        budget.complete(str(uuid4()), {"outcome": "succeeded"})
    assert path.read_bytes() == saved


@pytest.mark.parametrize("link_target", ["ledger", "lock"])
def test_symlinks_are_refused_without_changing_the_target(tmp_path: Path, link_target: str) -> None:
    original = tmp_path / "original.json"
    original.write_text("private original remains untouched")
    ledger = tmp_path / "budget.json"
    (ledger if link_target == "ledger" else Path(str(ledger) + ".lock")).symlink_to(original)
    with pytest.raises(BudgetError):
        RequestBudget(ledger)
    assert original.read_text() == "private original remains untouched"


@pytest.mark.parametrize(
    "metadata",
    [
        {"outcome": "succeeded", "prompt": "private source"},
        {"outcome": "succeeded", "api_key": "private placeholder"},
        {"outcome": "succeeded", "input_tokens": -1},
        {"outcome": "succeeded", "output_tokens": True},
        {"outcome": "succeeded", "model": "unsafe model with spaces"},
        {"outcome": {}},
    ],
)
def test_metadata_rejects_private_fields_and_invalid_usage(tmp_path: Path, metadata: dict) -> None:
    path = tmp_path / "budget.json"
    budget = RequestBudget(path)
    identifier = budget.reserve("safe.label", FINGERPRINT)
    saved = path.read_bytes()
    with pytest.raises(BudgetError):
        budget.complete(identifier, metadata)
    assert path.read_bytes() == saved
    assert budget.snapshot() == {"used": 1, "remaining": 99}


def test_failed_atomic_replacement_preserves_existing_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import live_budget

    path = tmp_path / "budget.json"
    budget = RequestBudget(path, limit=3)
    budget.reserve("first.call", FINGERPRINT)
    saved = path.read_bytes()

    def unavailable(*args, **kwargs) -> None:
        raise OSError("Controlled atomic replacement failure")

    monkeypatch.setattr(live_budget.os, "replace", unavailable)
    with pytest.raises(OSError):
        budget.reserve("second.call", FINGERPRINT)
    assert path.read_bytes() == saved
    assert budget.snapshot() == {"used": 1, "remaining": 2}
    assert {p.name for p in tmp_path.iterdir()} == {"budget.json", "budget.json.lock"}


@pytest.mark.parametrize("kind", ["over_limit", "duplicate_id", "bad_status"])
def test_tampered_accounting_cannot_be_loaded_or_overwritten(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "budget.json"
    budget = RequestBudget(path, limit=2)
    budget.reserve("first.call", FINGERPRINT)
    state = json.loads(path.read_text())
    if kind == "over_limit":
        state["entries"] *= 3
    elif kind == "duplicate_id":
        state["entries"] *= 2
    else:
        state["entries"][0]["status"] = "refunded"
    path.write_text(json.dumps(state))
    saved = path.read_bytes()
    with pytest.raises(BudgetError):
        budget.reserve("additional.call", FINGERPRINT)
    assert path.read_bytes() == saved


def generation_request() -> dict:
    from pydantic import BaseModel

    class Output(BaseModel):
        answer: str

    return {
        "schema": Output,
        "system": "private synthetic instruction",
        "content": "private synthetic source",
        "max_output_tokens": 100,
        "timeout_seconds": 5,
    }


def test_sync_wrapper_reserves_durably_before_one_delegate_call_and_only_records_usage(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace
    from unittest.mock import Mock

    from scripts.live_budget import BudgetedGenerationProvider

    path = tmp_path / "budget.json"
    budget = RequestBudget(path, limit=2)
    response = SimpleNamespace(
        model="gpt-test",
        input_tokens=12,
        output_tokens=4,
        reasoning_tokens=2,
        body="private provider body",
    )

    def generate(**kwargs):
        assert budget.snapshot() == {"used": 1, "remaining": 1}
        assert json.loads(path.read_text())["entries"][0]["status"] == "reserved"
        return response

    delegate = Mock()
    delegate.generate.side_effect = generate
    wrapper = BudgetedGenerationProvider(delegate, budget, "probe.answer", profile="low")
    request = generation_request()
    assert wrapper.generate(**request) is response
    delegate.generate.assert_called_once_with(**request)
    entry = json.loads(path.read_text())["entries"][0]
    assert entry["status"] == "completed"
    assert entry["metadata"] == {
        "outcome": "succeeded",
        "model": "gpt-test",
        "input_tokens": 12,
        "output_tokens": 4,
        "reasoning_tokens": 2,
    }
    assert "private" not in path.read_text()


@pytest.mark.parametrize("unexpected", [False, True])
def test_sync_wrapper_failure_counts_once_and_never_retries(
    tmp_path: Path,
    unexpected: bool,
) -> None:
    from unittest.mock import Mock

    from app.domain.documents import AIFailureDetails, ProcessingError
    from scripts.live_budget import BudgetedGenerationProvider

    path = tmp_path / "budget.json"
    budget = RequestBudget(path, limit=2)
    error = (
        RuntimeError("private unknown failure")
        if unexpected
        else ProcessingError(
            "ai_timeout",
            "private provider failure",
            diagnostics=AIFailureDetails(
                reason="output_token_limit", input_tokens=12, output_tokens=3
            ),
        )
    )
    delegate = Mock()
    delegate.generate.side_effect = error
    wrapper = BudgetedGenerationProvider(delegate, budget, "probe.failure")
    with pytest.raises(type(error)) as caught:
        wrapper.generate(**generation_request())
    assert caught.value is error
    delegate.generate.assert_called_once()
    entry = json.loads(path.read_text())["entries"][0]
    assert entry["status"] == ("reserved" if unexpected else "completed")
    if unexpected:
        assert entry["metadata"] == {}
    else:
        assert entry["metadata"]["error_code"] == "ai_timeout"
        assert entry["metadata"]["input_tokens"] == 12
    assert budget.snapshot() == {"used": 1, "remaining": 1}
    assert "private" not in path.read_text()


def test_explicit_retries_spend_distinct_slots_and_exhaustion_blocks_delegate(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace
    from unittest.mock import Mock

    from scripts.live_budget import BudgetedGenerationProvider

    path = tmp_path / "budget.json"
    budget = RequestBudget(path, limit=2)
    delegate = Mock()
    delegate.generate.return_value = SimpleNamespace()
    wrapper = BudgetedGenerationProvider(delegate, budget, "probe.retry")
    request = generation_request()
    wrapper.generate(**request)
    wrapper.generate(**request)
    with pytest.raises(BudgetError, match="EXHAUSTED"):
        wrapper.generate(**request)
    assert delegate.generate.call_count == 2
    entries = json.loads(path.read_text())["entries"]
    assert entries[0]["id"] != entries[1]["id"]
    assert entries[0]["fingerprint"] == entries[1]["fingerprint"]


def test_profile_changes_request_fingerprint_without_storing_request_text(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from unittest.mock import Mock

    from scripts.live_budget import BudgetedGenerationProvider

    path = tmp_path / "budget.json"
    delegate = Mock()
    delegate.generate.return_value = SimpleNamespace()
    budget = RequestBudget(path)
    request = generation_request()
    for profile in ("none", "low"):
        BudgetedGenerationProvider(delegate, budget, "probe.profile", profile).generate(**request)
    entries = json.loads(path.read_text())["entries"]
    assert entries[0]["fingerprint"] != entries[1]["fingerprint"]
    assert "private" not in path.read_text()


@pytest.mark.asyncio
async def test_async_wrapper_preserves_result_and_failure_accounting(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from app.domain.documents import ProcessingError
    from scripts.live_budget import AsyncBudgetedGenerationProvider

    path = tmp_path / "budget.json"
    budget = RequestBudget(path, limit=2)
    response = SimpleNamespace(model="gpt-test", input_tokens=7, output_tokens=2)

    async def generate(**kwargs):
        assert budget.snapshot() == {"used": 1, "remaining": 1}
        assert json.loads(path.read_text())["entries"][0]["status"] == "reserved"
        return response

    delegate = SimpleNamespace(generate=AsyncMock(side_effect=generate))
    wrapper = AsyncBudgetedGenerationProvider(delegate, budget, "probe.async")
    request = generation_request()
    assert await wrapper.generate(**request) is response
    delegate.generate.assert_awaited_once_with(**request)
    delegate.generate.side_effect = ProcessingError("ai_timeout", "private failure")
    with pytest.raises(ProcessingError):
        await wrapper.generate(**request)
    assert delegate.generate.await_count == 2
    assert budget.snapshot() == {"used": 2, "remaining": 0}
    assert [e["metadata"]["outcome"] for e in json.loads(path.read_text())["entries"]] == [
        "succeeded",
        "failed",
    ]
    with pytest.raises(BudgetError, match="EXHAUSTED"):
        await wrapper.generate(**request)
    assert delegate.generate.await_count == 2


@pytest.mark.asyncio
async def test_async_cancellation_keeps_uncertain_reserved_slot(tmp_path: Path) -> None:
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from scripts.live_budget import AsyncBudgetedGenerationProvider

    path = tmp_path / "budget.json"
    budget = RequestBudget(path)
    delegate = SimpleNamespace(generate=AsyncMock(side_effect=asyncio.CancelledError()))
    wrapper = AsyncBudgetedGenerationProvider(delegate, budget, "probe.cancel")
    with pytest.raises(asyncio.CancelledError):
        await wrapper.generate(**generation_request())
    assert budget.snapshot()["used"] == 1
    assert json.loads(path.read_text())["entries"][0]["status"] == "reserved"


@pytest.mark.asyncio
async def test_embedding_wrapper_reserves_before_request_hides_text_and_counts_failures(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from app.domain.documents import ProcessingError
    from scripts.live_budget import AsyncBudgetedEmbeddingProvider

    path = tmp_path / "budget.json"
    budget = RequestBudget(path, limit=2)
    result = SimpleNamespace(model="embedding-test", input_tokens=17)

    async def embed_many(texts, *, timeout_seconds):
        assert budget.snapshot()["used"] == 1
        assert json.loads(path.read_text())["entries"][0]["status"] == "reserved"
        return result

    delegate = SimpleNamespace(embed_many=AsyncMock(side_effect=embed_many))
    wrapper = AsyncBudgetedEmbeddingProvider(delegate, budget, "embedding.query", "test.1536")
    assert await wrapper.embed_many(["private query"], timeout_seconds=5) is result
    delegate.embed_many.assert_awaited_once_with(["private query"], timeout_seconds=5)
    delegate.embed_many.side_effect = ProcessingError("ai_timeout", "private error")
    with pytest.raises(ProcessingError):
        await wrapper.embed_many(["private query"], timeout_seconds=5)
    assert budget.snapshot() == {"used": 2, "remaining": 0}
    entries = json.loads(path.read_text())["entries"]
    assert [entry["metadata"]["outcome"] for entry in entries] == ["succeeded", "failed"]
    assert entries[0]["metadata"]["input_tokens"] == 17
    assert "private" not in path.read_text()
    with pytest.raises(BudgetError, match="EXHAUSTED"):
        await wrapper.embed_many(["private query"], timeout_seconds=5)
    assert delegate.embed_many.await_count == 2


@pytest.mark.asyncio
async def test_embedding_cancellation_is_counted_and_profile_changes_fingerprint(
    tmp_path: Path,
) -> None:
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from scripts.live_budget import AsyncBudgetedEmbeddingProvider

    path = tmp_path / "budget.json"
    budget = RequestBudget(path)
    delegate = SimpleNamespace(embed_many=AsyncMock(side_effect=asyncio.CancelledError()))
    for profile in ("small.1536", "large.3072"):
        wrapper = AsyncBudgetedEmbeddingProvider(delegate, budget, "embedding.query", profile)
        with pytest.raises(asyncio.CancelledError):
            await wrapper.embed_many(["private query"], timeout_seconds=5)
    entries = json.loads(path.read_text())["entries"]
    assert all(entry["status"] == "reserved" for entry in entries)
    assert entries[0]["fingerprint"] != entries[1]["fingerprint"]
    assert budget.snapshot()["used"] == 2
