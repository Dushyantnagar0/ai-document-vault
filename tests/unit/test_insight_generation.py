"""Focused generation preserves frozen input and fails within one provider call."""

import json
from typing import Any
from uuid import uuid4

import pytest

from app.domain.documents import Generated, ProcessingError
from app.services.analysis_evidence import QuoteReference
from app.services.insight_generation import generate_candidates


class Generator:
    def __init__(self, insights: list[dict[str, Any]]) -> None:
        self.insights = insights
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> Generated[Any]:
        self.calls.append(kwargs)
        return Generated(
            result=kwargs["schema"].model_validate({"insights": self.insights}),
            model="test-model",
            input_tokens=100,
            output_tokens=200,
        )


def test_frozen_summary_and_original_passages_are_supplied_once() -> None:
    summary = [{"text": "Two workers each complete three exports per hour."}]
    original = json.dumps(summary)
    catalog = {"q1": QuoteReference(uuid4(), "Two workers each complete three exports per hour.")}
    generator = Generator(
        [
            {
                "quote_ids": ["q1"],
                "text": "Under the stated assumptions, combined capacity is six per hour.",
            }
        ]
    )
    result = generate_candidates(summary, catalog, generator, lambda: 30)
    assert len(generator.calls) == 1 and result.output_tokens == 200
    request = generator.calls[0]
    packet = json.loads(request["content"])
    assert packet["frozen_summary"] == [summary[0]["text"]]
    assert packet["evidence"][0]["text"] == catalog["q1"].quote
    assert request["max_output_tokens"] == 3072
    assert json.dumps(summary) == original


@pytest.mark.parametrize("ids", [["unknown"], ["q1", "q1"]])
def test_invalid_references_do_not_trigger_an_automatic_repair(ids: list[str]) -> None:
    generator = Generator([{"text": "A conclusion.", "quote_ids": ids}])
    with pytest.raises(ProcessingError, match="Insight candidates failed"):
        generate_candidates(
            [{"text": "A fact."}],
            {"q1": QuoteReference(uuid4(), "A fact.")},
            generator,
            lambda: 30,
        )
    assert len(generator.calls) == 1


def test_sparse_source_can_have_no_candidates() -> None:
    generator = Generator([])
    result = generate_candidates(
        [{"text": "Owner: Example."}],
        {"q1": QuoteReference(uuid4(), "Owner: Example.")},
        generator,
        lambda: 30,
    )
    assert result.result.insights == [] and len(generator.calls) == 1


def test_literal_tokenizer_markers_are_document_text() -> None:
    text = "The <|endoftext|> marker remains ordinary quoted document content."
    generator = Generator([{"text": text, "quote_ids": ["q1"]}])
    result = generate_candidates(
        [{"text": "Tokenizer notation."}],
        {"q1": QuoteReference(uuid4(), text)},
        generator,
        lambda: 30,
    )
    assert result.result.insights[0].text == text


def test_deadline_or_call_budget_failure_prevents_the_provider_request() -> None:
    generator = Generator([])

    def exhausted() -> float:
        raise ProcessingError("BUDGET_EXHAUSTED", "No calls left.")

    with pytest.raises(ProcessingError, match="No calls left"):
        generate_candidates([], {}, generator, exhausted)
    assert generator.calls == []
