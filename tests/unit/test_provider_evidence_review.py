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

from app.services.analysis_evidence import QuoteReference
from app.services.analysis_review import AnalysisReviewOutput, project_analysis_review
from app.services.chat_review import ChatReviewOutput, compact_chat_packet, project_chat_review
from app.services.evidence_review import ReviewOutput, prepare_review, project_review
from app.services.insight_generation import InsightCandidates

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


def test_review_wire_preserves_literal_claims_and_actual_named_source(provider: ModuleType) -> None:
    source_id = UUID(int=1)
    catalog = {
        "q1": QuoteReference(source_id, "Condition 40 is owned by Mira Stone. "),
        "q2": QuoteReference(source_id, "Condition 39 is owned by Pavel Reed. "),
    }
    packet = prepare_review({"answer": [{"text": "Condition 40 is owned by Mira Stone."}]}, catalog)
    status, response = post(provider, "ReviewOutput", json.loads(packet.content))
    assert status == 200
    result = ReviewOutput.model_validate_json(response["choices"][0]["message"]["content"])
    assert result.decisions[0].quote_ids == ["q1"]
    projected = project_review(packet, result)
    assert projected.scope_complete
    assert not projected.dropped_claim_ids
    assert projected.payload["answer"][0]["support_quotes"] == [
        {"source_id": str(source_id), "quote": catalog["q1"].quote}
    ]


def test_review_fixture_refuses_to_fake_inference_support(provider: ModuleType) -> None:
    packet = prepare_review(
        {"answer": [{"text": "The team can process six exports per hour."}]},
        {"q1": QuoteReference(UUID(int=1), "Two workers each process three exports per hour.")},
    )
    status, response = post(provider, "ReviewOutput", json.loads(packet.content))
    assert status == 200
    result = ReviewOutput.model_validate_json(response["choices"][0]["message"]["content"])
    assert not result.decisions[0].supported
    assert not result.scope_complete


def test_insight_fixture_does_not_pad_summary(provider: ModuleType) -> None:
    status, response = post(
        provider,
        "InsightCandidates",
        {"frozen_summary": ["Owner: Priya Shah."], "evidence": []},
    )
    assert status == 200
    result = InsightCandidates.model_validate_json(response["choices"][0]["message"]["content"])
    assert not result.insights


def test_analysis_review_wire_assesses_every_candidate_conservatively(provider: ModuleType) -> None:
    packet = prepare_review(
        {
            "summary": [{"text": "Owner: Priya Shah."}],
            "insights": [
                {"text": "Owner: Priya Shah."},
                {"text": "Annual fee: 240 USD."},
                {"text": "The team can process six exports per hour."},
            ],
        },
        {"q1": QuoteReference(UUID(int=1), "Owner: Priya Shah. Annual fee: 240 USD.")},
    )
    status, response = post(provider, "AnalysisReviewOutput", json.loads(packet.content))
    assert status == 200
    result = AnalysisReviewOutput.model_validate_json(response["choices"][0]["message"]["content"])
    assert [item.classification for item in result.insight_assessments] == [
        "repeats_meaning",
        "trivial_repackaging",
        "no_supported_conclusion",
    ]
    assert [item.block_index for item in result.insight_assessments] == [0, 1, 2]
    projected = project_analysis_review(packet, result)
    assert projected.payload["insights"] == []
    assert projected.payload["summary"][0]["text"] == "Owner: Priya Shah."


def test_analysis_review_wire_explicitly_records_no_candidates(provider: ModuleType) -> None:
    packet = prepare_review(
        {"summary": [{"text": "Owner: Priya Shah."}], "insights": []},
        {"q1": QuoteReference(UUID(int=1), "Owner: Priya Shah.")},
    )
    status, response = post(provider, "AnalysisReviewOutput", json.loads(packet.content))
    assert status == 200
    result = AnalysisReviewOutput.model_validate_json(response["choices"][0]["message"]["content"])
    assert result.insight_assessments == []
    assert project_analysis_review(packet, result).scope_complete


@pytest.mark.parametrize(
    "question,draft",
    [
        ("Who is the owner?", "Owner: Priya Shah."),
        ("Give a document overview", "Owner: Priya Shah. Annual fee: 240 USD."),
    ],
)
def test_chat_review_wire_binds_source_fact_inventory_and_literal_coverage(
    provider: ModuleType, question: str, draft: str
) -> None:
    packet = compact_chat_packet(
        prepare_review(
            {"answer": [{"text": draft}]},
            {"q1": QuoteReference(UUID(int=1), "Owner: Priya Shah. Annual fee: 240 USD.")},
            question=question,
        )
    )
    status, response = post(provider, "ChatReviewOutput", json.loads(packet.content))
    assert status == 200
    raw = json.loads(response["choices"][0]["message"]["content"])
    assert "required_qualifications" not in raw
    result = ChatReviewOutput.model_validate(raw)
    assert result.answerability == "answered"
    assert len(result.required_aspects) == 1
    assert result.required_aspects[0].quote_ids == ["q1"]
    assert result.required_aspects[0].conveyed_by_claim_ids == [c.id for c in packet.claims]
    assert "Priya Shah" in result.required_aspects[0].aspect
    projected = project_chat_review(packet, result)
    assert projected.scope_complete
    assert projected.payload["answer"][0]["text"] == draft


def test_chat_review_wire_abstains_when_supported_literal_is_unrelated(
    provider: ModuleType,
) -> None:
    packet = prepare_review(
        {"answer": [{"text": "Annual fee: 240 USD."}]},
        {"q1": QuoteReference(UUID(int=1), "Owner: Priya Shah. Annual fee: 240 USD.")},
        question="Who is the owner?",
    )
    status, response = post(provider, "ChatReviewOutput", json.loads(packet.content))
    assert status == 200
    result = ChatReviewOutput.model_validate_json(response["choices"][0]["message"]["content"])
    assert result.answerability == "insufficient"
    assert result.required_aspects == []
    assert result.relevant_claim_ids == []
    assert project_chat_review(packet, result).payload == {"answer": []}
