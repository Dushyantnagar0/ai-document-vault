"""Literal quote catalogs and exact ID resolution, without models or providers."""

import json
import traceback
from dataclasses import FrozenInstanceError
from uuid import UUID

import pytest
from pydantic import ValidationError

from app.domain.documents import ProcessingError
from app.services.analysis_evidence import (
    MAX_QUOTE_CHARACTERS,
    QuoteReference,
    SelectionBlock,
    SelectionOutput,
    build_quote_catalog,
    iter_literal_passages,
    selection_to_payload,
)

pytestmark = pytest.mark.unit
FIRST, SECOND = UUID(int=1), UUID(int=2)


@pytest.mark.parametrize(
    "text",
    ["", " \t\r\n" * 100, "Same line.\n" * 100, "café e\u0301 中文👩🏽‍💻" * 100],
)
def test_literal_segmentation_preserves_order_repetitions_and_whitespace(text: str) -> None:
    passages = list(iter_literal_passages(text))
    assert "".join(passages) == text
    assert all(0 < len(passage) <= MAX_QUOTE_CHARACTERS for passage in passages)
    assert [reference.quote for reference in build_quote_catalog({FIRST: (text,)}).values()] == (
        list(dict.fromkeys(passage for passage in passages if passage.strip()))
    )


def test_catalog_covers_entire_unicode_excerpt_without_normalization_or_truncation() -> None:
    text = ("Cafe\u0301 中文 🦋\u00a0literal  spaces\r\n" * 35) + "LATE_DECISIVE_FACT"
    # Use a non-repeating suffix per passage to examine full coverage without dedup.
    unique = "".join(f"Section {i}: {text[:91]} Marker {i}.\n" for i in range(40))
    catalog = build_quote_catalog({FIRST: (unique,)})
    quotes = [reference.quote for reference in catalog.values()]
    assert "".join(quotes) == unique
    assert all(0 < len(quote) <= MAX_QUOTE_CHARACTERS and quote.strip() for quote in quotes)
    assert all(quote in unique for quote in quotes)
    assert "Cafe\u0301" in "".join(quotes)
    assert "Café" not in "".join(quotes)
    assert "\ufffd" not in "".join(quotes)
    long_catalog = build_quote_catalog({FIRST: (text,)})
    assert any("LATE_DECISIVE_FACT" in item.quote for item in long_catalog.values())
    assert len(long_catalog) > 1


@pytest.mark.parametrize("text", ["漢" * 721, "🙂" * 541, "e\u0301" * 377, "x" * 901])
def test_no_whitespace_long_runs_cover_every_character(text: str) -> None:
    # Distinct markers avoid deliberately deduplicated repeated slices.
    text = text[:179] + "A" + text[179:358] + "B" + text[358:537] + "C" + text[537:]
    catalog = build_quote_catalog({FIRST: (text,)})
    unique_quotes = [ref.quote for ref in catalog.values()]
    assert all(len(quote) <= MAX_QUOTE_CHARACTERS for quote in unique_quotes)
    for index in range(0, len(text), MAX_QUOTE_CHARACTERS):
        assert text[index : index + MAX_QUOTE_CHARACTERS] in unique_quotes
    assert all(quote in text for quote in unique_quotes)


def test_newlines_whitespace_and_excerpt_boundaries_remain_literal() -> None:
    excerpts = (" \t\r\n", "First\r\nSecond\tthird\u00a0value", "last\n", "")
    catalog = build_quote_catalog({FIRST: excerpts})
    assert [ref.quote for ref in catalog.values()] == [excerpts[1], excerpts[2]]
    assert all(ref.quote in excerpts for ref in catalog.values())
    assert build_quote_catalog({FIRST: (" \t\n" * 200,)}) == {}
    assert build_quote_catalog({}) == {}


def test_catalog_prefers_literal_newline_then_whitespace_at_boundaries() -> None:
    newline_text = "first line\n" + "word " * 70
    quoted = list(build_quote_catalog({FIRST: (newline_text,)}).values())
    assert quoted[0].quote == "first line\n"
    assert all(ref.quote.endswith(" ") for ref in quoted[1:-1])
    assert "".join(ref.quote for ref in quoted) == newline_text


def test_ids_are_deterministic_and_deduplication_is_source_specific() -> None:
    forward = build_quote_catalog({FIRST: ("same", "same", "different"), SECOND: ("same",)})
    reverse = build_quote_catalog({SECOND: ("same",), FIRST: ("same", "same", "different")})
    assert forward == build_quote_catalog({FIRST: ("same", "same", "different"), SECOND: ("same",)})
    assert reverse["q1"].source_id == SECOND
    assert forward == {
        "q1": QuoteReference(FIRST, "same"),
        "q2": QuoteReference(FIRST, "different"),
        "q3": QuoteReference(SECOND, "same"),
    }
    with pytest.raises(FrozenInstanceError):
        forward["q1"].quote = "changed"  # type: ignore[misc]


def test_selection_rehydrates_verbatim_quotes_unique_sources_and_preserves_claims() -> None:
    catalog = build_quote_catalog(
        {
            FIRST: ("Quoted\r\n literal Cafe\u0301.", "Another exact quote."),
            SECOND: ("Other source.",),
        }
    )
    output = SelectionOutput(
        summary=[SelectionBlock(text="Qualified fact.", quote_ids=["q2", "q1", "q3"])], insights=[]
    )
    payload = selection_to_payload(output, catalog)
    block = payload["summary"][0]
    assert block["text"] == "Qualified fact."
    assert block["source_ids"] == [str(FIRST), str(SECOND)]
    assert block["support_quotes"] == [
        {"source_id": str(catalog[key].source_id), "quote": catalog[key].quote}
        for key in ["q2", "q1", "q3"]
    ]
    assert payload["insights"] == []
    assert "quote_ids" not in json.dumps(payload)


@pytest.mark.parametrize("ids", [["q999"], ["q1", "q1"], ["Q1"], [" q1"], [""]])
def test_invalid_references_fail_without_private_content(ids: list[str]) -> None:
    catalog = {"q1": QuoteReference(FIRST, "PRIVATE_ORIGINAL_CANARY")}
    output = SelectionOutput(
        summary=[SelectionBlock(text="PRIVATE_GENERATED_CANARY", quote_ids=ids)], insights=[]
    )
    with pytest.raises(ProcessingError) as captured:
        selection_to_payload(output, catalog)
    assert captured.value.code == "analysis_citation_invalid"
    assert captured.value.diagnostics is not None
    assert captured.value.diagnostics.as_dict()["reason"] == (
        "duplicate_quote_id" if len(ids) != len(set(ids)) else "unknown_quote_id"
    )
    assert captured.value.reason == (
        "duplicate_quote_ids" if len(ids) != len(set(ids)) else "unknown_quote_ids"
    )
    rendered = "".join(traceback.format_exception(captured.value))
    assert "CANARY" not in rendered
    assert not any(identifier in str(captured.value) for identifier in ids if identifier)


@pytest.mark.parametrize("quote", ["", " \n\t", "x" * (MAX_QUOTE_CHARACTERS + 1)])
def test_invalid_catalog_passage_cannot_be_published(quote: str) -> None:
    output = SelectionOutput(summary=[SelectionBlock(text="fact", quote_ids=["q1"])], insights=[])
    with pytest.raises(ProcessingError, match="Analysis selected invalid evidence references"):
        selection_to_payload(output, {"q1": QuoteReference(FIRST, quote)})


@pytest.mark.parametrize(
    "payload",
    [
        {"summary": [], "insights": []},
        {"summary": [{"text": "fact", "quote_ids": []}], "insights": []},
        {"summary": [{"text": "", "quote_ids": ["q1"]}], "insights": []},
        {"summary": [{"text": "fact", "quote_ids": ["q1"] * 9}], "insights": []},
        {
            "summary": [{"text": "fact", "quote_ids": ["q1"], "source_ids": [str(FIRST)]}],
            "insights": [],
        },
        {"summary": [{"text": "fact", "quote_ids": ["q1"]}], "insights": [], "raw_quotes": []},
    ],
)
def test_selection_schema_rejects_empty_out_of_bounds_or_extra_fields(payload: dict) -> None:
    with pytest.raises(ValidationError):
        SelectionOutput.model_validate(payload)


def test_blank_factual_claim_rejected_after_schema_validation() -> None:
    output = SelectionOutput(summary=[SelectionBlock(text=" \n", quote_ids=["q1"])], insights=[])
    with pytest.raises(ProcessingError):
        selection_to_payload(output, {"q1": QuoteReference(FIRST, "literal")})


def test_same_quote_may_support_distinct_blocks_without_duplicate_source_ids() -> None:
    block = SelectionBlock(text="fact", quote_ids=["q1"])
    output = SelectionOutput(summary=[block], insights=[block])
    payload = selection_to_payload(output, {"q1": QuoteReference(FIRST, "literal")})
    assert payload["summary"][0] == payload["insights"][0]


def test_complete_qualification_sentence_is_not_cut_into_unrelated_fact_fragments() -> None:
    text = (
        "A legal hold blocks deletion. Release of a legal hold does not create a "
        "universal new thirty-day clock: the ordinary or dispute rule still applies. "
        "Operators inspect the saved dates before scheduling deletion."
    )
    passages = list(iter_literal_passages(text))
    assert "".join(passages) == text
    qualification = next(part for part in passages if "Release of a legal hold" in part)
    assert "ordinary or dispute rule still applies." in qualification
    assert "Operators inspect" not in qualification
