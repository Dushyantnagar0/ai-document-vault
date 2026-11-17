"""Source-answer orchestration with scripted outputs; no semantic or provider claims."""

import json
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import pytest

from app.core.config import Settings
from app.domain.conversation import Evidence, TurnReservation
from app.domain.documents import Generated, ProcessingError
from app.services.chat import ChatProcessor
from app.services.source_answer import CompositionOutput, SourceSelectionOutput

pytestmark = pytest.mark.unit


def setup() -> tuple[ChatProcessor, TurnReservation, tuple[Evidence, ...]]:
    processor = object.__new__(ChatProcessor)
    processor.settings = Settings(_env_file=None, openai_api_key="unit-placeholder")
    processor.source_answerer = object()  # type: ignore[assignment]
    now = datetime.now(UTC)
    reservation = TurnReservation(
        UUID(int=1),
        UUID(int=2),
        UUID(int=3),
        UUID(int=4),
        UUID(int=5),
        UUID(int=6),
        UUID(int=7),
        "Who is responsible, and what is the fee?",
        now + timedelta(seconds=90),
        now + timedelta(seconds=120),
        "test",
        1,
    )
    evidence = Evidence(
        UUID(int=8), UUID(int=6), UUID(int=7), "rules.txt", "Owner: Mira Stone.", ()
    )
    return processor, reservation, (evidence,)


def selected(
    *, missing: bool = False, insufficient: bool = False, invalid: bool = False
) -> SourceSelectionOutput:
    components = (
        []
        if insufficient
        else [{"label": "Responsible person", "quote_ids": ["q999" if invalid else "q1"]}]
    )
    if missing:
        components.append({"label": "Fee", "quote_ids": []})
    return SourceSelectionOutput.model_validate(
        {
            "evidence_status": "insufficient"
            if insufficient
            else "partial"
            if missing
            else "answered",
            "components": components,
        }
    )


def composed(*, missing: bool = False, invalid: bool = False) -> CompositionOutput:
    parts = [
        {
            "component_id": "c1",
            "status": "supported",
            "text": "Mira Stone is responsible.",
            "quote_ids": ["q999" if invalid else "q1"],
        }
    ]
    if missing:
        parts.append({"component_id": "c2", "status": "unsupported", "text": "", "quote_ids": []})
    return CompositionOutput.model_validate({"parts": parts})


@pytest.mark.parametrize("status", ["answered", "partial", "insufficient"])
async def test_source_stages_persist_selection_first_and_keep_history_out(
    monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    processor, reservation, evidence = setup()
    calls: list[dict[str, Any]] = []
    snapshots = []
    record: dict[str, Any] = {}

    async def persist() -> None:
        snapshots.append(json.loads(json.dumps(record)))

    async def generate(*args: Any, **kwargs: Any) -> Generated[Any]:
        calls.append({"schema": args[1], "system": args[2], "content": args[3], **kwargs})
        assert kwargs["provider"] is processor.source_answerer
        payload = json.loads(args[3])
        assert payload["question"] == reservation.question
        assert payload["retrieval_query"] == "Responsible person and fee"
        assert "history" not in payload
        if args[1] is SourceSelectionOutput:
            result = selected(missing=status == "partial", insufficient=status == "insufficient")
        else:
            assert snapshots[-1]["selection"]["status"] == "completed"
            assert "composition" not in snapshots[-1]
            result = composed(missing=status == "partial")
        return Generated(result, processor.settings.openai_source_answer_model, 100, 60)

    monkeypatch.setattr(processor, "_generate", generate)
    answer, audit = await processor._source_answer(
        reservation,
        evidence,
        resolved_query="Responsible person and fee",
        source_record=record,
        persist=persist,
    )
    assert answer.evidence_status == status
    assert len(calls) == (1 if status == "insufficient" else 2)
    assert audit is record and audit["reasoning_effort"] == "low"
    assert audit["projection"]["evidence_status"] == status
    assert "original_catalog" not in json.dumps(audit)
    if status == "partial":
        assert "Not established by the supplied evidence:\n- Fee" in answer.answer
    if status == "insufficient":
        assert audit["composition"]["status"] == "skipped"
    else:
        assert audit["quote_bindings"]["q1"]["quote"] == evidence[0].text
        assert len(audit["composition"]["request_sha256"]) == 64


@pytest.mark.parametrize("invalid_stage", ["selection", "composition"])
async def test_single_structural_repair_reuses_identical_input(
    monkeypatch: pytest.MonkeyPatch, invalid_stage: str
) -> None:
    processor, reservation, evidence = setup()
    calls: dict[str, list[tuple[str, str]]] = {"selection": [], "composition": []}

    async def generate(*args: Any, **kwargs: Any) -> Generated[Any]:
        stage = "selection" if args[1] is SourceSelectionOutput else "composition"
        calls[stage].append((args[2], args[3]))
        invalid = stage == invalid_stage and len(calls[stage]) == 1
        result = selected(invalid=invalid) if stage == "selection" else composed(invalid=invalid)
        return Generated(result, processor.settings.openai_source_answer_model, 100, 60)

    monkeypatch.setattr(processor, "_generate", generate)
    answer, audit = await processor._source_answer(reservation, evidence)
    assert answer.evidence_status == "answered"
    assert sum(map(len, calls.values())) == 3
    initial, repaired = calls[invalid_stage]
    assert initial[1] == repaired[1]
    assert repaired[0].startswith(initial[0]) and repaired[0] != initial[0]
    assert "q999" not in repaired[1]
    assert audit[invalid_stage]["attempt"] == 2
    assert audit[invalid_stage]["repair"]["first_error_code"] == "SOURCE_ANSWER_INVALID"
    assert audit[invalid_stage]["status"] == "completed"
    assert "error" not in audit[invalid_stage]


async def test_selection_repair_consumes_the_composition_repair_allowance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor, reservation, evidence = setup()
    calls = []
    record: dict[str, Any] = {}
    snapshots = []

    async def persist() -> None:
        snapshots.append(json.loads(json.dumps(record)))

    async def generate(*args: Any, **kwargs: Any) -> Generated[Any]:
        calls.append(args[1])
        result = (
            selected(invalid=len(calls) == 1)
            if args[1] is SourceSelectionOutput
            else composed(invalid=True)
        )
        return Generated(result, processor.settings.openai_source_answer_model, 100, 60)

    monkeypatch.setattr(processor, "_generate", generate)
    with pytest.raises(ProcessingError) as failure:
        await processor._source_answer(reservation, evidence, source_record=record, persist=persist)
    assert failure.value.code == "SOURCE_ANSWER_INVALID"
    assert calls == [SourceSelectionOutput, SourceSelectionOutput, CompositionOutput]
    assert snapshots[-1]["selection"]["status"] == "completed"
    assert snapshots[-1]["composition"]["status"] == "failed"
    assert snapshots[-1]["composition"]["attempt"] == 1
    assert snapshots[-1]["quote_bindings"]


@pytest.mark.parametrize("failure_stage", ["selection", "composition"])
async def test_transport_failure_is_not_retried_and_retains_completed_selection(
    monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    processor, reservation, evidence = setup()
    calls = []
    record: dict[str, Any] = {}

    async def generate(*args: Any, **kwargs: Any) -> Generated[Any]:
        stage = "selection" if args[1] is SourceSelectionOutput else "composition"
        calls.append(stage)
        if stage == failure_stage:
            raise ProcessingError("ai_timeout", "Provider request timed out.")
        return Generated(selected(), processor.settings.openai_source_answer_model, 100, 60)

    monkeypatch.setattr(processor, "_generate", generate)
    with pytest.raises(ProcessingError):
        await processor._source_answer(reservation, evidence, source_record=record)
    assert calls.count(failure_stage) == 1
    assert record[failure_stage]["error"]["code"] == "ai_timeout"
    if failure_stage == "composition":
        assert record["selection"]["status"] == "completed" and record["quote_bindings"]


async def test_complete_input_limit_fails_before_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor, reservation, evidence = setup()
    processor.settings = processor.settings.model_copy(update={"chat_prompt_max_tokens": 100})
    record: dict[str, Any] = {}

    async def generate(*args: Any, **kwargs: Any) -> Generated[Any]:
        pytest.fail("Budget rejection must happen before the provider")

    monkeypatch.setattr(processor, "_generate", generate)
    with pytest.raises(ProcessingError) as failure:
        await processor._source_answer(reservation, evidence, source_record=record)
    assert failure.value.code == "SOURCE_ANSWER_INPUT_LIMIT"
    assert record["selection"]["attempt"] == 1
    assert record["selection"]["status"] == "failed"


async def test_insufficient_selection_does_not_charge_synthetic_composition_to_output_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor, reservation, evidence = setup()
    processor.settings = processor.settings.model_copy(update={"chat_max_output_tokens": 64})
    calls = []

    async def generate(*args: Any, **kwargs: Any) -> Generated[Any]:
        calls.append(args[1])
        return Generated(
            SourceSelectionOutput.model_validate(
                {
                    "evidence_status": "insufficient",
                    "components": [
                        {"label": f"Missing item {index}", "quote_ids": []} for index in range(4)
                    ],
                }
            ),
            processor.settings.openai_source_answer_model,
            100,
            60,
        )

    monkeypatch.setattr(processor, "_generate", generate)
    answer, audit = await processor._source_answer(reservation, evidence)
    assert answer.evidence_status == "insufficient" and calls == [SourceSelectionOutput]
    assert audit["composition"]["status"] == "skipped"


async def test_failed_repair_does_not_attribute_earlier_output_to_new_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor, reservation, evidence = setup()
    calls = []
    record: dict[str, Any] = {}

    async def generate(*args: Any, **kwargs: Any) -> Generated[Any]:
        calls.append(args[1])
        if len(calls) == 2:
            raise ProcessingError("ai_timeout", "Provider request timed out.")
        return Generated(
            selected(invalid=True), processor.settings.openai_source_answer_model, 100, 60
        )

    monkeypatch.setattr(processor, "_generate", generate)
    with pytest.raises(ProcessingError):
        await processor._source_answer(reservation, evidence, source_record=record)
    assert record["selection"]["attempt"] == 2
    assert "result" not in record["selection"] and "model" not in record["selection"]
    assert len(record["selection"]["repair"]["first_output_sha256"]) == 64
