"""Offline provider-fixture wire behavior; no model or semantic-quality evidence."""

import importlib.util
import io
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import Mock
from uuid import UUID

import pytest

from app.core.config import Settings
from app.domain.conversation import Evidence
from app.services.analysis_evidence import QuoteReference, SelectionOutput
from app.services.source_answer import (
    CompositionOutput,
    SourceSelectionOutput,
    prepare_composition_packet,
    project_composition,
    validate_selection,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "fixtures" / "provider_stub.py"
    spec = importlib.util.spec_from_file_location("selection_provider_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.BARRIERS, "wait", Mock())
    return module


def post(
    provider: ModuleType,
    schema: str,
    content: dict[str, Any],
    *,
    authorized: bool = True,
) -> tuple[int, dict[str, Any]]:
    payload = json.dumps(
        {
            "model": "fixture-generation",
            "messages": [{"role": "user", "content": json.dumps(content)}],
            "response_format": {"json_schema": {"name": schema}},
        }
    ).encode()
    handler = object.__new__(provider.Handler)
    handler.path = "/v1/chat/completions"
    handler.headers = {
        "Content-Length": str(len(payload)),
        "Authorization": provider.TEST_AUTHORIZATION
        if authorized
        else "Bearer rejected-fixture-key",
    }
    handler.rfile = io.BytesIO(payload)
    handler.reply = Mock()
    handler.do_POST()
    handler.reply.assert_called_once()
    status, response = handler.reply.call_args.args
    return status, response


@pytest.mark.parametrize("stage", ["final", "section", "reduce"])
def test_selection_schema_selects_actual_catalog_id_and_literal_passage(
    provider: ModuleType,
    stage: str,
) -> None:
    passage = "Owner: Test Operator.\nRenewal: 12 December 2027. Unicode: café 東京."
    supplied = {
        "stage": stage,
        "evidence": [
            {"quote_id": "q7", "text": passage},
            {"quote_id": "q44", "text": "Other fact."},
        ],
        "draft_summaries": ["Unsupported draft claim must not become fixture evidence."],
    }
    status, response = post(provider, "SelectionOutput", supplied)
    assert status == 200
    result = json.loads(response["choices"][0]["message"]["content"])
    assert SelectionOutput.model_validate(result).summary[0].quote_ids == ["q7"]
    assert result == {"summary": [{"text": passage, "quote_ids": ["q7"]}], "insights": []}
    selected = {quote_id for block in result["summary"] for quote_id in block["quote_ids"]}
    assert selected <= {item["quote_id"] for item in supplied["evidence"]}
    assert "source_ids" not in json.dumps(result) and "support_quotes" not in json.dumps(result)
    provider.BARRIERS.wait.assert_called_once_with("analysis", supplied)
    assert provider.COUNTS["SelectionOutput"] == 1
    assert response["model"] == "fixture-generation"


@pytest.mark.parametrize(
    "evidence",
    [None, [], [{"text": "No ID"}], [{"quote_id": "q1"}], [{"quote_id": "q1", "text": " "}]],
)
def test_selection_without_actual_catalog_entry_returns_controlled_error(
    provider: ModuleType,
    evidence: Any,
) -> None:
    status, response = post(
        provider,
        "SelectionOutput",
        {
            "stage": "final",
            "evidence": evidence,
            "draft_summaries": ["Do not invent q1 from this."],
        },
    )
    assert status == 400
    assert response == {"error": {"message": "Invalid fixture analysis evidence"}}


def test_legacy_analysis_fixture_remains_supported(provider: ModuleType) -> None:
    identifier = str(UUID(int=1))
    status, response = post(
        provider,
        "AnalysisOutput",
        {
            "items": [{"source_id": identifier, "text": "Owner: Legacy Test Operator."}],
        },
    )
    assert status == 200
    result = json.loads(response["choices"][0]["message"]["content"])
    assert result["summary"][0]["support_quotes"] == [
        {"source_id": identifier, "quote": "Owner: Legacy Test Operator."}
    ]
    assert provider.COUNTS["AnalysisOutput"] == 1


def test_selection_still_requires_fixed_fixture_authorization(provider: ModuleType) -> None:
    status, _ = post(
        provider,
        "SelectionOutput",
        {
            "stage": "final",
            "evidence": [{"quote_id": "q7", "text": "A fact."}],
        },
        authorized=False,
    )
    assert status == 403
    assert not provider.COUNTS
    provider.BARRIERS.wait.assert_not_called()


@pytest.mark.parametrize(
    "question,expected",
    [("Who is the owner?", "answered"), ("What is the cancellation penalty?", "insufficient")],
)
def test_answer_selection_wire_contract_uses_supplied_ids(
    provider: ModuleType, question: str, expected: str
) -> None:
    supplied = {
        "question": question,
        "intent": "factual",
        "complete_scope": True,
        "evidence": [
            {
                "source_id": str(UUID(int=1)),
                "text": "Owner: Priya Shah",
                "passages": [{"quote_id": "q7", "text": "Owner: Priya Shah"}],
            }
        ],
    }
    status, response = post(provider, "AnswerSelectionOutput", supplied)
    assert status == 200
    result = json.loads(response["choices"][0]["message"]["content"])
    assert result["evidence_status"] == expected
    if expected == "answered":
        assert result["blocks"] == [{"text": "Owner: Priya Shah", "quote_ids": ["q7"]}]
    else:
        assert result["blocks"] == []
    provider.BARRIERS.wait.assert_called_once_with("answer", supplied)


def test_answer_fixture_skips_unselectable_whitespace_in_overview(provider: ModuleType) -> None:
    content = {
        "question": "Summarize the document",
        "intent": "overview",
        "complete_scope": True,
        "evidence": [
            {
                "source_id": str(UUID(int=1)),
                "passages": [
                    {"text": "\n"},
                    {"quote_id": "q1", "text": "Owner: Priya Shah\n"},
                    {"quote_id": "q2", "text": "Renewal date: 12 December 2027"},
                ],
            }
        ],
    }
    status, response = post(provider, "AnswerSelectionOutput", content)
    assert status == 200
    result = json.loads(response["choices"][0]["message"]["content"])
    assert result["blocks"][0]["quote_ids"] == ["q1"]


@pytest.mark.parametrize(
    "question,intent",
    [
        ("Summarize only the retention rules and their exceptions.", "factual"),
        ("Give an overview of this document.", "overview"),
        ("List every condition.", "exhaustive"),
    ],
)
def test_fixture_keeps_explicit_focused_summary_out_of_overview(
    provider: ModuleType, question: str, intent: str
) -> None:
    status, response = post(provider, "PlannerOutput", {"question": question, "history": []})
    assert status == 200
    result = json.loads(response["choices"][0]["message"]["content"])
    assert result["intent"] == intent


@pytest.mark.parametrize(
    "question,expected,expected_text",
    [
        ("Who is the owner?", "answered", "Owner: Priya Shah"),
        ("And when does it renew?", "answered", "Renewal date: 12 December 2027"),
        ("How much is the annual fee?", "answered", "Annual fee: 240 USD"),
        ("Who owns it and what are the cancellation terms?", "partial", "Cancellation terms"),
        ("What is the cancellation penalty?", "insufficient", "cannot find"),
        ("Explain quantum error correction.", "insufficient", "cannot find"),
    ],
)
def test_source_fixture_two_stage_literals_partial_and_refusal(
    provider: ModuleType, question: str, expected: str, expected_text: str
) -> None:
    source_id = UUID(int=1)
    passages = {
        "q7": "Owner: Priya Shah.",
        "q19": "Renewal date: 12 December 2027.",
        "q28": "Annual fee: 240 USD.",
    }
    supplied = {
        "question": question,
        "evidence": [
            {"quote_id": quote_id, "source_id": "s1", "text": text}
            for quote_id, text in passages.items()
        ],
        "source_index": {"s1": str(source_id)},
    }
    status, response = post(provider, "SourceSelectionOutput", supplied)
    assert status == 200
    selected = SourceSelectionOutput.model_validate_json(
        response["choices"][0]["message"]["content"]
    )
    assert selected.evidence_status == expected
    catalog = {key: QuoteReference(source_id, text) for key, text in passages.items()}
    frozen = validate_selection(selected, catalog, question=question)
    packet = prepare_composition_packet(
        frozen, Settings(_env_file=None, openai_api_key="fixture-placeholder")
    )
    status, response = post(provider, "CompositionOutput", json.loads(packet.content))
    assert status == 200
    composed = CompositionOutput.model_validate_json(response["choices"][0]["message"]["content"])
    sources = (
        Evidence(
            source_id, UUID(int=2), UUID(int=3), "fixture.txt", "\n".join(passages.values()), ()
        ),
    )
    result = project_composition(frozen, composed, sources)
    assert result.evidence_status == expected and expected_text in result.answer
    assert all(
        part.text in passages[quote_id] for part in composed.parts for quote_id in part.quote_ids
    )
    provider.BARRIERS.wait.assert_called_once_with("answer", supplied)
    assert provider.COUNTS["SourceSelectionOutput"] == 1
    assert provider.COUNTS["CompositionOutput"] == 1
    assert provider.ANSWER_QUESTIONS[provider._fingerprint(question)] == 1
