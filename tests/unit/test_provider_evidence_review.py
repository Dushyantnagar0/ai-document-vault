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
