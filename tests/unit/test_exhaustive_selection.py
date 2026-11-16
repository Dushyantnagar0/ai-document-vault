"""Incomplete inventories expose selected literal evidence, never generated facts."""

from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from app.domain.conversation import Evidence
from app.domain.documents import ProcessingError, SourceSpan
from app.services.chat_context import (
    ExhaustiveSelectionOutput,
    answer_quote_catalog,
    exhaustive_selection_to_answer,
    resolve_answer_selection,
)
from app.services.grounding import INSUFFICIENT_ANSWER, validate_answer

DOCUMENT = UUID(int=1)
PROFILE = UUID(int=2)


def source(text: str) -> Evidence:
    return Evidence(
        id=uuid4(),
        document_id=DOCUMENT,
        profile_id=PROFILE,
        filename="terms.pdf",
        text=text,
        source_spans=(SourceSpan("body", {"page": 3}, 0, len(text), 0, len(text)),),
    )


def test_known_passages_become_literal_partial_blocks_with_original_sources() -> None:
    evidence = [source("Condition α requires 117 units.\n"), source("The owner is Zoë.\n")]
    catalog = answer_quote_catalog(evidence)
    selected = list(catalog)
    output = exhaustive_selection_to_answer(ExhaustiveSelectionOutput(quote_ids=selected), evidence)
    assert output.evidence_status == "partial"
    assert len(output.blocks) == len(selected)
    for block, identifier in zip(output.blocks, selected, strict=True):
        assert block.text == catalog[identifier].quote
        assert block.quote_ids == [identifier]
    hydrated = resolve_answer_selection(output, evidence)
    for block, identifier in zip(hydrated.blocks, selected, strict=True):
        reference = catalog[identifier]
        assert block.source_ids == [reference.source_id]
        assert len(block.support_quotes) == 1
        assert block.support_quotes[0].source_id == reference.source_id
        assert block.support_quotes[0].quote == reference.quote
    result = validate_answer(hydrated, evidence, complete_scope=False)
    assert result.evidence_status == "partial"
    assert "Condition α requires 117 units." in result.answer
    assert "The owner is Zoë." in result.answer
    assert {citation["document_id"] for citation in result.citations} == {str(DOCUMENT)}
    assert {citation["source_id"] for citation in result.citations} == {str(x.id) for x in evidence}


@pytest.mark.parametrize("kind", ["unknown", "duplicate"])
def test_unknown_or_duplicate_passages_fail_with_static_diagnostics(kind: str) -> None:
    evidence = [source("Only this original fact is available.")]
    known = next(iter(answer_quote_catalog(evidence)))
    identifiers = [known, "unknown-untrusted-id"] if kind == "unknown" else [known, known]
    with pytest.raises(ProcessingError) as caught:
        exhaustive_selection_to_answer(ExhaustiveSelectionOutput(quote_ids=identifiers), evidence)
    assert caught.value.code == "CHAT_INVALID_OUTPUT"
    assert "unknown-untrusted-id" not in str(caught.value)
    assert caught.value.diagnostics is not None
    assert caught.value.diagnostics.as_dict()["reason"] == (
        "unknown_quote_id" if kind == "unknown" else "duplicate_quote_id"
    )


def test_empty_selection_is_insufficient_and_carries_no_factual_blocks() -> None:
    evidence = [source("Unrelated source text.")]
    output = exhaustive_selection_to_answer(ExhaustiveSelectionOutput(quote_ids=[]), evidence)
    assert output.evidence_status == "insufficient"
    assert output.blocks == []
    result = validate_answer(
        resolve_answer_selection(output, evidence), evidence, complete_scope=False
    )
    assert result.answer == INSUFFICIENT_ANSWER
    assert result.evidence_status == "insufficient"
    assert result.citations == []


def test_schema_cannot_accept_generated_prose_or_more_than_eight_passages() -> None:
    for extra in ({"text": "An invented threshold is 999."}, {"blocks": []}):
        with pytest.raises(ValidationError):
            ExhaustiveSelectionOutput.model_validate({"quote_ids": ["q1"], **extra})
    with pytest.raises(ValidationError):
        ExhaustiveSelectionOutput(quote_ids=[f"q{index}" for index in range(9)])


def test_distant_selected_passages_from_one_source_survive_as_separate_citations() -> None:
    first = "Early condition requires 101 units.\n"
    last = "Late condition requires 140 units.\n"
    item = source(first + "Repeated neutral context.\n" * 100 + last)
    catalog = answer_quote_catalog([item])
    selected = [
        next(q for q, ref in catalog.items() if first.strip() in ref.quote),
        next(q for q, ref in catalog.items() if last.strip() in ref.quote),
    ]
    output = exhaustive_selection_to_answer(ExhaustiveSelectionOutput(quote_ids=selected), [item])
    result = validate_answer(resolve_answer_selection(output, [item]), [item], complete_scope=False)
    assert result.evidence_status == "partial"
    assert len(result.blocks) == 2
    assert len(result.citations) == 2
    assert first.strip() in result.answer and last.strip() in result.answer
    for identifier in selected:
        quote = catalog[identifier].quote
        assert any(quote in citation["excerpt"] for citation in result.citations)
    assert {citation["source_id"] for citation in result.citations} == {str(item.id)}
