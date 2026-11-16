"""Deterministic complete-prompt budgets and saved-analysis source preservation."""

import json
from dataclasses import replace
from typing import Any
from uuid import UUID, uuid4

import pytest
import tiktoken

from app.core.config import Settings
from app.domain.conversation import Evidence
from app.domain.documents import ProcessingError, SourceSpan
from app.prompts.analysis import generation_prompt_tokens
from app.prompts.chat import ANSWER_REPAIR, ANSWER_SYSTEM
from app.services.chat_context import AnswerSelectionOutput, prepare_context

pytestmark = pytest.mark.unit
DOCUMENT, PROFILE = UUID(int=1), UUID(int=2)


def settings(**overrides: Any) -> Settings:
    # Tight per-stage budgets exercise assembly guards independently of startup validation.
    return Settings(_env_file=None, openai_api_key="unit-placeholder").model_copy(update=overrides)


def source(
    text: str, *, filename: str = "source.pdf", spans: tuple[SourceSpan, ...] = ()
) -> Evidence:
    return Evidence(
        id=uuid4(),
        document_id=DOCUMENT,
        profile_id=PROFILE,
        filename=filename,
        text=text,
        source_spans=spans,
    )


def block(item: Evidence, quote: str, text: str = "Supported fact.") -> dict[str, Any]:
    return {
        "text": text,
        "source_ids": [str(item.id)],
        "support_quotes": [{"source_id": str(item.id), "quote": quote}],
    }


def tokens(value: Any) -> int:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return len(tiktoken.get_encoding("o200k_base").encode(encoded, disallowed_special=()))


def test_highest_ordered_candidates_dedup_and_factual_top_k_preserves_scope() -> None:
    candidates = [source(f"Distinct fact {index}.") for index in range(16)]
    supplied = [candidates[0], candidates[0], replace(candidates[0], id=uuid4()), *candidates[1:]]
    result = prepare_context(
        "What is stated?", [], supplied, settings(), intent="factual", complete_scope=True
    )
    assert [item.id for item in result.evidence] == [item.id for item in candidates[:8]]
    assert result.complete_scope is True and result.evidence_scope == "retrieved_excerpts"
    assert result.token_count == generation_prompt_tokens(
        AnswerSelectionOutput, ANSWER_SYSTEM + ANSWER_REPAIR, result.content
    )


def test_full_metadata_and_labels_count_against_evidence_budget() -> None:
    candidates = [source("owner " * 500, filename='📁 rules "quoted".pdf'), source("Short fact.")]
    configured = settings(retrieval_max_tokens=180)
    result = prepare_context(
        "Who owns it?", [], candidates, configured, intent="factual", complete_scope=True
    )
    packet = json.loads(result.content)
    assert tokens(packet["evidence"]) <= 180
    assert packet["evidence"][0]["filename"] == candidates[0].filename
    assert len(result.evidence[0].text) < len(candidates[0].text)
    assert result.complete_scope is True


def test_unicode_prefix_and_literal_special_tokens_remain_exact_original_text() -> None:
    original = "👩🏽‍💻 中文é e\u0301 <|endoftext|> " * 200
    item = source(original)
    question = "Explain <|endoftext|> without changing it."
    result = prepare_context(
        question,
        [],
        [item],
        settings(retrieval_max_tokens=200),
        intent="factual",
        complete_scope=True,
    )
    assert result.evidence
    actual = result.evidence[0].text
    assert original.startswith(actual) and "\ufffd" not in actual
    assert actual.encode("utf-8", errors="strict")
    assert json.loads(result.content)["question"] == question
    assert tokens(json.loads(result.content)["evidence"]) <= 200


def test_compact_locators_are_actual_boundaries_while_internal_spans_are_retained() -> None:
    text = "alpha " * 300
    spans = tuple(
        SourceSpan(str(page), {"page": page}, start, end, start, end)
        for page, start, end in ((1, 0, 600), (2, 600, 1200), (3, 1200, len(text)))
    )
    item = source(text, spans=spans)
    full = prepare_context("What?", [], [item], settings(), intent="factual", complete_scope=True)
    assert json.loads(full.content)["evidence"][0]["locators"] == [{"page": 1}, {"page": 3}]
    trimmed = prepare_context(
        "What?",
        [],
        [item],
        settings(retrieval_max_tokens=80),
        intent="factual",
        complete_scope=True,
    )
    assert trimmed.evidence[0].source_spans == spans
    assert json.loads(trimmed.content)["evidence"][0]["locators"] == [{"page": 1}]


def test_oldest_history_is_removed_before_changing_actual_evidence() -> None:
    history = [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"Turn {index}: " + "old " * 100,
        }
        for index in range(8)
    ]
    item = source("Owner Priya Shah.")
    configured = settings(chat_history_max_tokens=10000)
    baseline = prepare_context(
        "Who owns it?", history, [item], configured, intent="factual", complete_scope=True
    )
    result = prepare_context(
        "Who owns it?",
        history,
        [item],
        configured.model_copy(update={"chat_prompt_max_tokens": baseline.token_count - 100}),
        intent="factual",
        complete_scope=True,
    )
    assert result.history == tuple(history[1:])
    assert result.evidence == (item,) and result.complete_scope is True
    assert json.loads(result.content)["question"] == "Who owns it?"


def test_history_json_is_bounded_and_only_latest_completed_turns_are_used() -> None:
    history = [
        {"role": "user", "content": f"Turn {index} " + "detail " * 30} for index in range(20)
    ]
    result = prepare_context(
        "What?",
        history,
        [],
        settings(chat_history_max_tokens=90),
        intent="factual",
        complete_scope=True,
    )
    assert tokens(list(result.history)) <= 90
    assert result.history[-1] == history[-1]
    assert len(result.history) <= 8 and not result.complete_scope


def test_current_question_and_system_instructions_are_never_truncated() -> None:
    question = "  Explain the precise current question, including 中文 and <|endoftext|>.  "
    baseline = prepare_context(question, [], [], settings(), intent="factual", complete_scope=True)
    with pytest.raises(ProcessingError) as captured:
        prepare_context(
            question,
            [],
            [],
            settings(chat_prompt_max_tokens=baseline.token_count - 1),
            intent="factual",
            complete_scope=True,
        )
    assert captured.value.code == "CHAT_INPUT_LIMIT"
    assert json.loads(baseline.content)["question"] == question


def test_complete_prompt_repair_reserve_trims_lowest_evidence_and_marks_partial() -> None:
    candidates = [source(f"Fact {index}: " + "detail " * 400) for index in range(2)]
    baseline = prepare_context(
        "What?", [], candidates, settings(), intent="factual", complete_scope=True
    )
    configured = settings(chat_prompt_max_tokens=baseline.token_count - 50)
    result = prepare_context(
        "What?", [], candidates, configured, intent="factual", complete_scope=True
    )
    assert result.evidence == (candidates[0],)
    assert result.complete_scope is False
    assert result.token_count <= configured.chat_prompt_max_tokens
    assert result.token_count == generation_prompt_tokens(
        AnswerSelectionOutput, ANSWER_SYSTEM + ANSWER_REPAIR, result.content
    )


def test_single_factual_source_can_shrink_for_full_prompt_without_corrupting_unicode() -> None:
    item = source("事实 👩🏽‍💻 detail " * 100)
    baseline = prepare_context(
        "What?", [], [item], settings(), intent="factual", complete_scope=True
    )
    result = prepare_context(
        "What?",
        [],
        [item],
        settings(chat_prompt_max_tokens=baseline.token_count - 30),
        intent="factual",
        complete_scope=True,
    )
    assert result.evidence and item.text.startswith(result.evidence[0].text)
    assert len(result.evidence[0].text) < len(item.text)
    assert "\ufffd" not in result.evidence[0].text and result.complete_scope is False


def test_overview_uses_original_quote_windows_and_only_retained_analysis_fields() -> None:
    item = source("prefix " * 1000 + "Owner Priya Shah. Annual fee 240 USD." + " suffix" * 1000)
    saved = {
        "summary": [block(item, "Owner Priya Shah.", "Priya owns the agreement.")],
        "insights": [block(item, "Annual fee 240 USD.", "The annual fee is 240 USD.")],
        "usage": {"private": "NEVER_SEND_USAGE"},
        "lineage": ["NEVER_SEND_LINEAGE"],
    }
    result = prepare_context(
        "Summarize.",
        [],
        [item],
        settings(),
        intent="overview",
        complete_scope=True,
        saved_analysis=saved,
    )
    packet = json.loads(result.content)
    assert result.complete_scope is True and result.evidence_scope == "document_summary"
    assert 0 < result.evidence[0].text_offset and len(result.evidence[0].text) <= 1200
    assert {"summary", "insights"} == set(packet["saved_analysis"])
    assert "NEVER_SEND" not in result.content
    for values in packet["saved_analysis"].values():
        for claim in values:
            for support in claim["support_quotes"]:
                assert support["quote"] in result.evidence[0].text


def test_overview_shrinks_surroundings_while_preserving_all_quotes_in_budget() -> None:
    item = source(
        "intro " * 150 + "Owner Priya Shah." + " detail " * 40 + "Fee 240 USD." + " suffix" * 150
    )
    saved = {
        "summary": [block(item, "Owner Priya Shah.")],
        "insights": [block(item, "Fee 240 USD.")],
    }
    result = prepare_context(
        "Summarize.",
        [],
        [item],
        settings(retrieval_max_tokens=180),
        intent="overview",
        complete_scope=True,
        saved_analysis=saved,
    )
    assert result.evidence and result.complete_scope is True
    assert (
        "Owner Priya Shah." in result.evidence[0].text and "Fee 240 USD." in result.evidence[0].text
    )
    assert tokens(json.loads(result.content)["evidence"]) <= 180


def test_overview_keeps_distant_shared_source_support_within_context_budget() -> None:
    item = source("First proof." + " x" * 1500 + "Last proof.")
    saved = {
        "summary": [block(item, "First proof.", "Retained claim.")],
        "insights": [block(item, "Last proof.", "Omitted claim.")],
    }
    result = prepare_context(
        "Summarize.",
        [],
        [item],
        settings(),
        intent="overview",
        complete_scope=True,
        saved_analysis=saved,
    )
    assert result.complete_scope is True
    assert "Retained claim." in result.content and "Omitted claim." in result.content
    assert "First proof." in result.evidence[0].text
    assert "Last proof." in result.evidence[0].text


def test_missing_source_or_quote_cannot_make_summary_stand_in_for_evidence() -> None:
    item = source("Actual source text.")
    other = source("Other document's text.")
    for saved in (
        {"summary": [block(other, "Other document's text.")]},
        {"summary": [block(item, "An invented quote.")]},
    ):
        result = prepare_context(
            "Summarize.",
            [],
            [item],
            settings(),
            intent="overview",
            complete_scope=True,
            saved_analysis=saved,
        )
        assert not result.evidence and not result.complete_scope
        assert "saved_analysis" not in json.loads(result.content)


def test_overview_revalidates_claims_after_full_prompt_evidence_drop() -> None:
    first, second = source("First proof. " + "a " * 500), source("Second proof. " + "b " * 500)
    saved = {
        "summary": [block(first, "First proof.", "First retained claim.")],
        "insights": [block(second, "Second proof.", "Second omitted claim.")],
    }
    baseline = prepare_context(
        "Summarize.",
        [],
        [first, second],
        settings(),
        intent="overview",
        complete_scope=True,
        saved_analysis=saved,
    )
    result = prepare_context(
        "Summarize.",
        [],
        [first, second],
        settings(chat_prompt_max_tokens=baseline.token_count - 50),
        intent="overview",
        complete_scope=True,
        saved_analysis=saved,
    )
    assert [item.id for item in result.evidence] == [first.id]
    assert result.complete_scope is False
    assert "Second omitted claim." not in result.content
    assert "First retained claim." in result.content


def test_overview_prompt_locators_intersect_only_actual_window_but_keep_full_spans() -> None:
    text = "a" * 1500 + "Original proof." + "b" * 1500
    spans = tuple(
        SourceSpan(str(page), {"page": page}, start, end, start, end)
        for page, start, end in ((1, 0, 800), (2, 800, 2400), (3, 2400, len(text)))
    )
    item = source(text, spans=spans)
    result = prepare_context(
        "Summarize.",
        [],
        [item],
        settings(),
        intent="overview",
        complete_scope=True,
        saved_analysis={"summary": [block(item, "Original proof.")]},
    )
    assert json.loads(result.content)["evidence"][0]["locators"] == [{"page": 2}]
    assert result.evidence[0].source_spans == spans


@pytest.mark.parametrize("initial", [False, True])
def test_exhaustive_scope_preserved_only_when_all_candidates_survive(initial: bool) -> None:
    candidates = [source(f"Fact {index}.") for index in range(2)]
    result = prepare_context(
        "List every fact.", [], candidates, settings(), intent="exhaustive", complete_scope=initial
    )
    assert result.complete_scope is initial
    many = candidates + [source(f"Extra fact {index}.") for index in range(12)]
    limited = prepare_context(
        "List every fact.", [], many, settings(), intent="exhaustive", complete_scope=initial
    )
    assert limited.complete_scope is False


def test_exhaustive_prefix_trim_downgrades_scope() -> None:
    item = source("Fact. " * 1000)
    result = prepare_context(
        "List every fact.",
        [],
        [item],
        settings(retrieval_max_tokens=100),
        intent="exhaustive",
        complete_scope=True,
    )
    assert result.evidence and result.complete_scope is False


def test_empty_evidence_drops_all_saved_claims_and_cannot_be_complete() -> None:
    item = source("Original proof.")
    result = prepare_context(
        "Summarize.",
        [{"role": "assistant", "content": "An old assertion."}],
        [],
        settings(),
        intent="overview",
        complete_scope=True,
        saved_analysis={"summary": [block(item, "Original proof.")]},
    )
    packet = json.loads(result.content)
    assert packet["evidence"] == [] and "saved_analysis" not in packet
    assert result.complete_scope is False


def test_metadata_that_cannot_fit_is_omitted_without_leaking_unsupported_summary() -> None:
    item = source("Original proof.", filename="unbounded metadata " * 2000)
    result = prepare_context(
        "Summarize.",
        [],
        [item],
        settings(retrieval_max_tokens=100),
        intent="overview",
        complete_scope=True,
        saved_analysis={"summary": [block(item, "Original proof.")]},
    )
    assert result.evidence == () and result.complete_scope is False
    assert "saved_analysis" not in json.loads(result.content)


def test_context_rejects_cross_document_or_profile_sources() -> None:
    first = source("First fact.")
    for foreign in (
        replace(source("Other fact."), document_id=uuid4()),
        replace(source("Other fact."), profile_id=uuid4()),
    ):
        with pytest.raises(ProcessingError) as captured:
            prepare_context(
                "What?", [], [first, foreign], settings(), intent="factual", complete_scope=True
            )
        assert captured.value.code == "CHAT_INPUT_LIMIT"


def test_catalog_answer_restores_literal_pdf_whitespace_unicode_and_source_ids() -> None:
    from app.services.chat_context import resolve_answer_selection
    from app.services.grounding import validate_answer

    original = "A database stores structured\ninformation.  café e\u0301 👩🏽‍💻\n" * 20
    item = source(original)
    prepared = prepare_context(
        "What is a database?", [], [item], settings(), intent="factual", complete_scope=True
    )
    packet = json.loads(prepared.content)
    passages = packet["evidence"][0]["passages"]
    assert "text" not in packet["evidence"][0]
    assert "".join(passage["text"] for passage in passages) == prepared.evidence[0].text
    assert all(0 < len(passage["text"]) <= 180 for passage in passages)
    selected = AnswerSelectionOutput.model_validate(
        {
            "evidence_status": "answered",
            "blocks": [{"text": "It stores information.", "quote_ids": [passages[0]["quote_id"]]}],
        }
    )
    restored = resolve_answer_selection(selected, prepared.evidence)
    assert restored.blocks[0].source_ids == [item.id]
    assert restored.blocks[0].support_quotes[0].quote == passages[0]["text"]
    grounded = validate_answer(restored, prepared.evidence)
    assert grounded.evidence_status == "answered" and len(grounded.citations) == 1
    assert passages[0]["text"] in grounded.citations[0]["excerpt"]


@pytest.mark.parametrize(
    ("quote_ids", "reason"),
    [(["invented"], "unknown_quote_id"), (["q1", "q1"], "duplicate_quote_id")],
)
def test_catalog_answer_rejects_unknown_or_duplicate_selections_without_source_leaks(
    quote_ids: list[str], reason: str
) -> None:
    from app.services.chat_context import resolve_answer_selection

    item = source("PRIVATE_CONTENT_MUST_NOT_APPEAR")
    selected = AnswerSelectionOutput.model_validate(
        {
            "evidence_status": "answered",
            "blocks": [{"text": "A supported fact.", "quote_ids": quote_ids}],
        }
    )
    with pytest.raises(ProcessingError) as caught:
        resolve_answer_selection(selected, [item])
    assert caught.value.code == "CHAT_INVALID_OUTPUT"
    assert caught.value.diagnostics is not None
    assert caught.value.diagnostics.as_dict()["reason"] == reason
    assert "PRIVATE_CONTENT" not in str(caught.value)
    assert "invented" not in str(caught.value)


def test_catalog_answer_empty_insufficient_uses_existing_response_contract() -> None:
    from app.services.chat_context import resolve_answer_selection
    from app.services.grounding import INSUFFICIENT_ANSWER, validate_answer

    result = resolve_answer_selection(
        AnswerSelectionOutput(evidence_status="insufficient", blocks=[]), []
    )
    grounded = validate_answer(result, [])
    assert grounded.answer == INSUFFICIENT_ANSWER
    assert grounded.evidence_status == "insufficient" and grounded.citations == []


def test_catalog_answer_rejects_more_than_eight_distinct_sources() -> None:
    from app.services.chat_context import resolve_answer_selection

    evidence = [source(f"Fact {index}.") for index in range(9)]
    output = AnswerSelectionOutput.model_validate(
        {
            "evidence_status": "answered",
            "blocks": [
                {"text": "First facts.", "quote_ids": [f"q{i}" for i in range(1, 9)]},
                {"text": "Last fact.", "quote_ids": ["q9"]},
            ],
        }
    )
    with pytest.raises(ProcessingError) as caught:
        resolve_answer_selection(output, evidence)
    assert caught.value.diagnostics is not None
    assert caught.value.diagnostics.as_dict()["reason"] == "final_source_limit"


def test_wire_passages_preserve_repetitions_order_and_unselectable_whitespace() -> None:
    from app.services.chat_context import answer_quote_catalog

    item = source(" \t\r\n" * 100 + "Repeated line.\n" * 100 + "Last fact.")
    result = prepare_context(
        "What repeats?", [], [item], settings(), intent="factual", complete_scope=True
    )
    wire = json.loads(result.content)["evidence"][0]
    passages = wire["passages"]
    assert "text" not in wire
    assert "".join(passage["text"] for passage in passages) == result.evidence[0].text
    blank = [passage for passage in passages if not passage["text"].strip()]
    assert blank and all("quote_id" not in passage for passage in blank)
    ids = [passage["quote_id"] for passage in passages if passage["text"].strip()]
    assert len(ids) > len(set(ids))
    catalog = answer_quote_catalog(result.evidence)
    for passage in passages:
        if "quote_id" in passage:
            assert catalog[passage["quote_id"]].quote == passage["text"]
            assert catalog[passage["quote_id"]].source_id == item.id
    assert tokens([wire]) <= settings().retrieval_max_tokens


def test_catalog_ids_are_rebuilt_after_an_earlier_source_is_omitted() -> None:
    from app.services.chat_context import answer_quote_catalog, resolve_answer_selection

    omitted = source("An omitted fact.", filename="oversized metadata " * 2000)
    first, last = source("First retained fact."), source("Last retained fact.")
    original = answer_quote_catalog([omitted, first, last])
    assert original["q2"].source_id == first.id and original["q3"].source_id == last.id
    result = prepare_context(
        "Which facts?",
        [],
        [omitted, first, last],
        settings(retrieval_max_tokens=500),
        intent="factual",
        complete_scope=True,
    )
    assert result.evidence == (first, last)
    packet = json.loads(result.content)["evidence"]
    assert [entry["passages"][0]["quote_id"] for entry in packet] == ["q1", "q2"]
    resolved = resolve_answer_selection(
        AnswerSelectionOutput.model_validate(
            {
                "evidence_status": "answered",
                "blocks": [{"text": "Both retained facts.", "quote_ids": ["q1", "q2"]}],
            }
        ),
        result.evidence,
    )
    assert resolved.blocks[0].source_ids == [first.id, last.id]
    assert [quote.quote for quote in resolved.blocks[0].support_quotes] == [first.text, last.text]


def test_catalog_ids_are_rebuilt_after_an_earlier_source_window_is_trimmed() -> None:
    from app.services.chat_context import answer_quote_catalog, resolve_answer_selection

    first = source(
        "First decisive proof.\n"
        + "".join(f"Section {index}: " + "nearby detail " * 8 + "\n" for index in range(30))
    )
    last = source("Last decisive proof.")
    original_id = next(
        identifier
        for identifier, reference in answer_quote_catalog([first, last]).items()
        if reference.source_id == last.id
    )
    result = prepare_context(
        "Summarize.",
        [],
        [first, last],
        settings(),
        intent="overview",
        complete_scope=True,
        saved_analysis={
            "summary": [block(first, "First decisive proof.")],
            "insights": [block(last, last.text)],
        },
    )
    assert len(result.evidence) == 2
    assert len(result.evidence[0].text) < len(first.text)
    wire = json.loads(result.content)["evidence"]
    current_id = wire[1]["passages"][0]["quote_id"]
    assert current_id != original_id
    for entry, evidence in zip(wire, result.evidence, strict=True):
        assert "".join(piece["text"] for piece in entry["passages"]) == evidence.text
    resolved = resolve_answer_selection(
        AnswerSelectionOutput.model_validate(
            {
                "evidence_status": "answered",
                "blocks": [{"text": "Last fact.", "quote_ids": [current_id]}],
            }
        ),
        result.evidence,
    )
    assert resolved.blocks[0].source_ids == [last.id]
    assert resolved.blocks[0].support_quotes[0].quote == last.text


@pytest.mark.parametrize(
    "question",
    [
        "Summarize the staffing recommendations.",
        "Why is the proposed transition risky?",
        "How should the rollout be organized?",
        "Compare the two proposed rollout options.",
    ],
)
def test_focused_retrieval_context_keeps_topic_evidence_outside_saved_general_summary(
    question: str,
) -> None:
    intro = source("The report considers a service transition over the next year.")
    topic = source(
        "Staffing recommendation: assign a named coordinator for each team.\n"
        "The transition is risky because training overlaps the busiest service period.\n"
        "Option A rolls out one team at a time; option B moves all teams on one date.\n"
        "The report recommends option A to leave time to resolve early problems."
    )
    saved = {"summary": [block(intro, intro.text, "The report discusses a transition.")]}
    focused = prepare_context(
        question,
        [],
        [topic, intro],
        settings(),
        intent="factual",
        complete_scope=True,
        saved_analysis=saved,
    )
    packet = json.loads(focused.content)
    assert focused.evidence_scope == "retrieved_excerpts"
    assert focused.complete_scope is True
    assert [item.id for item in focused.evidence] == [topic.id, intro.id]
    assert packet["question"] == question and packet["intent"] == "factual"
    assert "saved_analysis" not in packet
    assert "".join(piece["text"] for piece in packet["evidence"][0]["passages"]) == topic.text
    assert focused.token_count <= settings().chat_prompt_max_tokens
    # This is the distinct whole-document route. Its general saved summary does
    # not supply focused evidence absent from the original summary citations.
    overview = prepare_context(
        "Summarize this document.",
        [],
        [topic, intro],
        settings(),
        intent="overview",
        complete_scope=True,
        saved_analysis=saved,
    )
    assert overview.evidence_scope == "document_summary"
    assert [item.id for item in overview.evidence] == [intro.id]
    assert "Staffing recommendation" not in overview.content


def test_review_reservation_fits_evidence_before_draft_without_changing_question() -> None:
    first = source("First complete evidence sentence. " * 20)
    second = source("Second complete evidence sentence. " * 20)
    calls: list[int] = []

    def review_fits(sources: Any) -> bool:
        calls.append(len(sources))
        return len(sources) <= 1

    result = prepare_context(
        "What does the first source say?",
        [],
        [first, second],
        settings(),
        intent="factual",
        complete_scope=True,
        review_fits=review_fits,
    )
    assert [e.id for e in result.evidence] == [first.id]
    assert json.loads(result.content)["question"] == "What does the first source say?"
    assert not result.complete_scope
    assert 2 in calls and calls[-1] == 1


def test_review_reservation_can_fit_a_single_source_prefix_before_drafting() -> None:
    item = source("A sentence with preserved source text. " * 100)
    result = prepare_context(
        "Explain the source.",
        [],
        [item],
        settings(),
        intent="factual",
        complete_scope=True,
        review_fits=lambda sources: sum(len(s.text) for s in sources) < 500,
    )
    assert len(result.evidence) == 1
    assert item.text.startswith(result.evidence[0].text)
    assert len(result.evidence[0].text) < 500
    assert not result.complete_scope
