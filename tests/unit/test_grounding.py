"""Offline provenance/coverage checks; quote membership is not semantic proof."""

from dataclasses import replace
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from app.domain.conversation import Evidence
from app.domain.documents import ProcessingError, SourceSpan
from app.services.analysis import FactualBlock, SourceQuote
from app.services.grounding import (
    INSUFFICIENT_ANSWER,
    AnswerOutput,
    PlannerOutput,
    validate_answer,
)

DOCUMENT = UUID("11111111-1111-4111-8111-111111111111")
PROFILE = UUID("22222222-2222-4222-8222-222222222222")


def source(text: str, *, filename: str = "source.pdf") -> Evidence:
    return Evidence(
        id=uuid4(),
        document_id=DOCUMENT,
        profile_id=PROFILE,
        filename=filename,
        text=text,
        source_spans=(SourceSpan("block-1", {"page": 3}, 100, 100 + len(text), 0, len(text)),),
    )


def block(evidence: Evidence, text: str, *quotes: str) -> FactualBlock:
    return FactualBlock(
        text=text,
        source_ids=[evidence.id],
        support_quotes=[SourceQuote(source_id=evidence.id, quote=quote) for quote in quotes],
    )


def test_server_metadata_and_numbered_sources_are_preserved() -> None:
    evidence = source("The limit is 72. The term is annual.", filename="Café report.pdf")
    result = validate_answer(
        AnswerOutput(
            evidence_status="answered",
            blocks=[
                block(evidence, "The limit is 72.", "The limit is 72."),
                block(evidence, "The term is annual.", "The term is annual."),
            ],
        ),
        [evidence],
    )
    assert result.evidence_status == "answered"
    assert result.answer == "The limit is 72. [1]\n\nThe term is annual. [1]"
    assert result.citations == [
        {
            "source_id": str(evidence.id),
            "document_id": str(DOCUMENT),
            "filename": "Café report.pdf",
            "locators": [{"page": 3}],
            "excerpt": evidence.text,
        }
    ]
    assert len(result.blocks) == 2


@pytest.mark.parametrize(
    "status,complete,expected",
    [
        ("answered", True, "answered"),
        ("answered", False, "partial"),
        ("partial", True, "partial"),
        ("partial", False, "partial"),
    ],
)
def test_partial_scope_cannot_be_promoted(status: str, complete: bool, expected: str) -> None:
    evidence = source("One obligation is payment.")
    parsed = AnswerOutput.model_validate(
        {
            "evidence_status": status,
            "blocks": [block(evidence, "Payment is one obligation.", "One obligation is payment.")],
        }
    )
    result = validate_answer(parsed, [evidence], complete_scope=complete)
    assert result.evidence_status == expected
    assert result.citations
    if not complete:
        assert "do not establish complete coverage" in result.answer


@pytest.mark.parametrize("status", ["answered", "partial", "insufficient"])
def test_empty_blocks_and_zero_evidence_are_explicitly_insufficient(status: str) -> None:
    parsed = AnswerOutput.model_validate({"evidence_status": status, "blocks": []})
    result = validate_answer(parsed, [], complete_scope=False)
    assert result.answer == INSUFFICIENT_ANSWER
    assert result.evidence_status == "insufficient"
    assert result.citations == []


def test_insufficient_status_with_nonempty_blocks_is_repairable_inconsistency() -> None:
    evidence = source("The fee is 100.")
    with pytest.raises(ProcessingError) as caught:
        validate_answer(
            AnswerOutput(
                evidence_status="insufficient",
                blocks=[block(evidence, "The fee is 100.", "The fee is 100.")],
            ),
            [evidence],
        )
    assert caught.value.code == "CHAT_INVALID_OUTPUT"
    assert caught.value.diagnostics is not None
    assert caught.value.diagnostics.as_dict()["reason"] == "inconsistent_evidence_status"
    assert "100" not in str(caught.value)


@pytest.mark.parametrize(
    "kind",
    ["unknown_source", "duplicate_ids", "quote_id_mismatch", "absent_quote", "whitespace_quote"],
)
def test_invalid_source_membership_and_quotes_are_safe_repairable_errors(kind: str) -> None:
    evidence = source("The private-source-marker limit is 72.")
    candidate = block(evidence, "The limit is 72.", "limit is 72.")
    if kind == "unknown_source":
        candidate = candidate.model_copy(update={"source_ids": [uuid4()]})
    elif kind == "duplicate_ids":
        candidate = candidate.model_copy(update={"source_ids": [evidence.id, evidence.id]})
    elif kind == "quote_id_mismatch":
        candidate = candidate.model_copy(
            update={"support_quotes": [SourceQuote(source_id=uuid4(), quote="limit is 72.")]}
        )
    elif kind == "absent_quote":
        candidate = block(evidence, "The limit is 72.", "private-source-marker absent text")
    else:
        candidate = block(evidence, "The limit is 72.", " ")
    with pytest.raises(ProcessingError) as captured:
        validate_answer(AnswerOutput(evidence_status="answered", blocks=[candidate]), [evidence])
    assert captured.value.code == "CHAT_INVALID_OUTPUT"
    assert "private-source-marker" not in str(captured.value)


@pytest.mark.parametrize("kind", ["document", "profile", "duplicate_source"])
def test_ambiguous_or_wrong_document_evidence_is_rejected(kind: str) -> None:
    evidence = source("The limit is 72.")
    if kind == "document":
        other = replace(source("Different document."), document_id=uuid4())
    elif kind == "profile":
        other = replace(source("Different profile."), profile_id=uuid4())
    else:
        other = evidence
    with pytest.raises(ProcessingError) as captured:
        validate_answer(
            AnswerOutput(
                evidence_status="answered",
                blocks=[block(evidence, "The limit is 72.", "The limit is 72.")],
            ),
            [evidence, other],
        )
    assert captured.value.code == "CHAT_INVALID_OUTPUT"


def test_quotes_must_come_from_trimmed_evidence_not_the_original_chunk() -> None:
    original = source("Omitted renewal date. The supplied limit is 72.")
    supplied = replace(original, text="The supplied limit is 72.", text_offset=22)
    with pytest.raises(ProcessingError) as captured:
        validate_answer(
            AnswerOutput(
                evidence_status="answered",
                blocks=[block(supplied, "There is a renewal date.", "Omitted renewal date.")],
            ),
            [supplied],
        )
    assert captured.value.code == "CHAT_INVALID_OUTPUT"


def test_trimmed_text_offsets_select_only_intersecting_real_docx_locator() -> None:
    original = "Earlier section " * 10 + "Renewal is July 2027. " + "Later section " * 10
    boundary = original.index("Renewal")
    quote = "Renewal is July 2027."
    evidence = replace(
        source(original, filename="source.docx"),
        source_spans=(
            SourceSpan("first", {"path": "body/paragraph-1"}, 0, boundary, 0, boundary),
            SourceSpan(
                "second",
                {"path": "body/table-1/row-2/cell-1/paragraph-1"},
                boundary,
                len(original),
                boundary,
                len(original),
            ),
        ),
    )
    supplied = replace(
        evidence, text=original[boundary - 5 : boundary + 50], text_offset=boundary - 5
    )
    result = validate_answer(
        AnswerOutput(
            evidence_status="answered",
            blocks=[block(supplied, "The stated renewal is July 2027.", quote)],
        ),
        [supplied],
    )
    assert result.citations[0]["locators"] == [{"path": "body/table-1/row-2/cell-1/paragraph-1"}]
    assert quote in result.citations[0]["excerpt"]
    assert "page" not in result.citations[0]["locators"][0]


def test_bounded_excerpt_contains_all_support_and_is_an_exact_supplied_substring() -> None:
    text = "x" * 2000 + "First support." + "y" * 700 + "Second support." + "z" * 2000
    evidence = source(text)
    result = validate_answer(
        AnswerOutput(
            evidence_status="answered",
            blocks=[block(evidence, "Both reported facts.", "First support.", "Second support.")],
        ),
        [evidence],
    )
    excerpt = result.citations[0]["excerpt"]
    assert len(excerpt) <= 1200
    assert excerpt in text
    assert "First support." in excerpt and "Second support." in excerpt
    assert result.evidence_status == "answered"


def test_repeated_quote_uses_nearby_occurrence_to_keep_all_support() -> None:
    evidence = source("Repeated support." + "x" * 1800 + "Repeated support. Nearby fact.")
    result = validate_answer(
        AnswerOutput(
            evidence_status="answered",
            blocks=[
                block(
                    evidence,
                    "Both facts are supported nearby.",
                    "Repeated support.",
                    "Nearby fact.",
                )
            ],
        ),
        [evidence],
    )
    assert result.evidence_status == "answered"
    assert "Repeated support. Nearby fact." in result.citations[0]["excerpt"]


def test_distant_support_keeps_claims_with_separate_exact_citation_windows() -> None:
    evidence = source("Early fact." + "x" * 1800 + "Distant fact.")
    result = validate_answer(
        AnswerOutput(
            evidence_status="answered",
            blocks=[
                block(evidence, "An early fact is stated.", "Early fact."),
                block(evidence, "A distant fact is also stated.", "Distant fact."),
            ],
        ),
        [evidence],
    )
    assert result.evidence_status == "answered"
    assert "early fact" in result.answer and "distant fact" in result.answer
    assert "Early fact." in result.citations[0]["excerpt"]
    assert "Distant fact." in result.citations[1]["excerpt"]
    assert len(result.blocks) == 2 and len(result.citations) == 2
    assert result.answer.endswith("A distant fact is also stated. [2]")
    assert all(citation["excerpt"] in evidence.text for citation in result.citations)
    assert all(len(citation["excerpt"]) <= 1200 for citation in result.citations)


@pytest.mark.parametrize("status", ["answered", "partial"])
def test_one_claim_with_distant_support_gets_each_exact_window(status: str) -> None:
    evidence = source("Early fact." + "x" * 1800 + "Distant fact.")
    result = validate_answer(
        AnswerOutput.model_validate(
            {
                "evidence_status": status,
                "blocks": [block(evidence, "Both facts together.", "Early fact.", "Distant fact.")],
            }
        ),
        [evidence],
    )
    assert result.evidence_status == status
    assert result.answer == "Both facts together. [1] [2]"
    assert len(result.citations) == 2
    assert {citation["source_id"] for citation in result.citations} == {str(evidence.id)}
    assert all(citation["excerpt"] in evidence.text for citation in result.citations)
    assert all(len(citation["excerpt"]) <= 1200 for citation in result.citations)


def test_total_source_cap_is_enforced_across_blocks() -> None:
    evidence = [source(f"Fact {index}.") for index in range(9)]
    first = FactualBlock(
        text="Eight facts are supplied.",
        source_ids=[item.id for item in evidence[:8]],
        support_quotes=[SourceQuote(source_id=item.id, quote=item.text) for item in evidence[:8]],
    )
    with pytest.raises(ProcessingError) as captured:
        validate_answer(
            AnswerOutput(
                evidence_status="answered",
                blocks=[first, block(evidence[-1], "Ninth fact.", evidence[-1].text)],
            ),
            evidence,
        )
    assert captured.value.code == "CHAT_INVALID_OUTPUT"


def test_conflict_qualifications_and_literal_tags_remain_source_backed_text() -> None:
    earlier = source("Draft says limit 72. <note>Not effective.</note>")
    later = source("Revision says limit 80; approval is unknown.")
    result = validate_answer(
        AnswerOutput(
            evidence_status="partial",
            blocks=[
                block(
                    earlier,
                    "Draft: 72; <note>Not effective.</note>",
                    "Draft says limit 72.",
                    "<note>Not effective.</note>",
                ),
                block(
                    later,
                    "Revision: 80; approval remains unknown.",
                    "Revision says limit 80; approval is unknown.",
                ),
            ],
        ),
        [earlier, later],
    )
    assert result.evidence_status == "partial"
    assert "<note>Not effective.</note>" in result.answer
    assert "approval remains unknown" in result.answer
    assert len(result.citations) == 2


@pytest.mark.parametrize("query", ["", " ", "x" * 4001])
def test_planner_rejects_empty_or_unbounded_queries(query: str) -> None:
    with pytest.raises(ValidationError):
        PlannerOutput(query=query, intent="factual")


def test_generated_schema_rejects_oversized_quotes_or_answer_block_count() -> None:
    evidence = source("x" * 301)
    with pytest.raises(ValidationError):
        block(evidence, "A statement.", evidence.text)
    valid = block(source("A fact."), "A fact.", "A fact.")
    with pytest.raises(ValidationError):
        AnswerOutput(evidence_status="answered", blocks=[valid] * 9)


def test_literal_passage_hydration_cannot_silently_lose_an_answer_due_to_display_window() -> None:
    from app.services.chat_context import (
        AnswerSelectionOutput,
        answer_quote_catalog,
        resolve_answer_selection,
    )

    evidence = source(
        "First direct description.\n"
        + "".join(f"Other line {index}. " + "context " * 20 + "\n" for index in range(20))
        + "Last direct description."
    )
    catalog = answer_quote_catalog([evidence])
    identifiers = [next(iter(catalog)), next(reversed(catalog))]
    selected = AnswerSelectionOutput.model_validate(
        {
            "evidence_status": "answered",
            "blocks": [{"text": "Two descriptions are provided.", "quote_ids": identifiers}],
        }
    )
    restored = resolve_answer_selection(selected, [evidence])
    result = validate_answer(restored, [evidence])
    assert result.evidence_status == "answered" and len(result.citations) == 2
    assert result.answer == "Two descriptions are provided. [1] [2]"
    assert all(citation["excerpt"] in evidence.text for citation in result.citations)
    # Nearby support still shares one exact-substring citation window.
    narrowed = selected.model_copy(
        update={"blocks": [selected.blocks[0].model_copy(update={"quote_ids": identifiers[:1]})]}
    )
    repaired = validate_answer(resolve_answer_selection(narrowed, [evidence]), [evidence])
    assert repaired.evidence_status == "answered"
    assert len(repaired.blocks) == 1 and len(repaired.citations) == 1
    assert repaired.citations[0]["excerpt"] in evidence.text
    assert len(repaired.citations[0]["excerpt"]) <= 1200


def test_global_citation_window_limit_keeps_supported_subset_and_marks_partial() -> None:
    quotes = [f"Distinct fact {index}." for index in range(9)]
    evidence = source(("x" * 1800).join(quotes))
    result = validate_answer(
        AnswerOutput(
            evidence_status="answered",
            blocks=[
                block(evidence, "First eight supported facts.", *quotes[:8]),
                block(evidence, "A ninth fact.", quotes[-1]),
            ],
        ),
        [evidence],
    )
    assert result.evidence_status == "partial"
    assert len(result.citations) == 8 and len(result.blocks) == 1
    assert "ninth" not in result.answer
    assert "Some detail could not be included with its supporting sources" in result.answer
    assert all(citation["excerpt"] in evidence.text for citation in result.citations)
    assert all(len(citation["excerpt"]) <= 1200 for citation in result.citations)
    for quote in quotes[:8]:
        assert any(quote in citation["excerpt"] for citation in result.citations)


def test_window_budget_cannot_silently_turn_supported_answer_into_insufficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.services.grounding as grounding

    monkeypatch.setattr(grounding, "MAX_CITATIONS", 1)
    evidence = source("Early fact." + "x" * 1800 + "Distant fact.")
    with pytest.raises(ProcessingError) as caught:
        validate_answer(
            AnswerOutput(
                evidence_status="answered",
                blocks=[block(evidence, "Both facts.", "Early fact.", "Distant fact.")],
            ),
            [evidence],
        )
    assert caught.value.code == "CHAT_INVALID_OUTPUT"
    assert caught.value.diagnostics is not None
    assert caught.value.diagnostics.as_dict()["reason"] == "support_window_exceeded"


def test_final_window_order_updates_each_block_reference_and_actual_page_locators() -> None:
    first, last = "Early fact.", "Distant fact."
    text = first + "x" * 1800 + last
    evidence = replace(
        source(text),
        source_spans=(
            SourceSpan("first", {"page": 1}, 0, len(first), 0, len(first)),
            SourceSpan(
                "last",
                {"page": 9},
                len(text) - len(last),
                len(text),
                len(text) - len(last),
                len(text),
            ),
        ),
    )
    result = validate_answer(
        AnswerOutput(
            evidence_status="answered",
            blocks=[
                block(evidence, "The later fact.", last),
                block(evidence, "The earlier fact.", first),
            ],
        ),
        [evidence],
    )
    assert result.answer == "The later fact. [2]\n\nThe earlier fact. [1]"
    assert result.citations[0]["locators"] == [{"page": 1}]
    assert result.citations[1]["locators"] == [{"page": 9}]
    assert first in result.citations[0]["excerpt"]
    assert last in result.citations[1]["excerpt"]
