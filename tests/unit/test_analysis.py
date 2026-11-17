"""Deterministic quote-selection analysis coverage, grounding and cost boundaries."""

import json
import re
from typing import Any
from uuid import uuid4

import pytest
import tiktoken

from app.core.config import Settings
from app.domain.documents import AIFailureDetails, Chunk, Generated, ProcessingError, SourceSpan
from app.prompts.analysis import (
    ANALYSIS_SYSTEM,
    REPAIR_INSTRUCTION,
    REPAIR_REASONS,
    analysis_budget_instruction,
    analysis_repair_instruction,
    generation_prompt_tokens,
)
from app.services.analysis import (
    AnalysisOutput,
    _validate_sources,
    analyse_document,
)
from app.services.analysis_evidence import MAX_QUOTE_CHARACTERS, SelectionOutput

pytestmark = pytest.mark.unit


def settings(**values: Any) -> Settings:
    return Settings(_env_file=None, openai_api_key="unit-placeholder", **values)


def chunks(count: int, *, words: int = 5) -> list[Chunk]:
    return [
        Chunk(
            uuid4(),
            index,
            f"Fact {index}: renewal is annual.\n"
            + " ".join(f"detail{word}" for word in range(words))
            + " "
            + "Hidden qualifier.",
            words + 15,
            (),
        )
        for index in range(count)
    ]


class FakeGenerator:
    def __init__(self, *, padding: int = 0, problem: str | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.padding, self.problem = padding, problem

    def generate(self, **values: Any) -> Generated[Any]:
        self.calls.append(values)
        request = json.loads(values["content"])
        passage = request["evidence"][-1]
        reference = passage["quote_id"]
        if self.problem == "timeout":
            raise ProcessingError("ai_timeout", "Timed out.", retryable=True)
        if self.problem == "unknown" or (self.problem == "first_unknown" and len(self.calls) == 1):
            reference = "unknown_reference"
        quote_ids = [reference]
        if self.problem == "duplicate" or (
            self.problem == "first_duplicate" and len(self.calls) == 1
        ):
            quote_ids *= 2
        output = values["schema"].model_validate(
            {
                "summary": [
                    {"text": "Supported fact." + " detail" * self.padding, "quote_ids": quote_ids}
                ],
                "insights": [],
            }
        )
        return Generated(
            result=output,
            model=Settings.CHAT_MODEL,
            input_tokens=100,
            output_tokens=len(tiktoken.get_encoding("o200k_base").encode(output.model_dump_json())),
        )


def test_small_document_one_final_call_all_chunks_and_late_evidence() -> None:
    source = chunks(5)
    generator = FakeGenerator()
    reservations: list[bool] = []

    def reserve() -> float:
        reservations.append(True)
        return 7

    result = analyse_document(source, settings(), generator, reserve)
    assert len(generator.calls) == len(reservations) == 1
    call = generator.calls[0]
    request = json.loads(call["content"])
    assert call["schema"] is SelectionOutput
    assert request["stage"] == "final" and "draft_summaries" not in request
    assert call["timeout_seconds"] == 7
    assert result.covered_chunk_ids == tuple(chunk.id for chunk in source)
    assert result.payload["summary"][0]["source_ids"] == [str(source[-1].id)]
    assert result.payload["summary"][0]["support_quotes"][0]["quote"] in source[-1].text
    assert result.payload["analysis_coverage"] == 1.0
    assert result.prompt_version == result.payload["prompt_version"] == "analysis-v13"
    assert result.payload["usage"]["all_calls_reported_usage"] is True
    for chunk in source:
        assert str(chunk.id) not in call["content"]
    assert "source_id" not in call["content"] and "locators" not in call["content"]


def test_every_literal_character_is_supplied_without_first_n_truncation() -> None:
    source = chunks(2, words=100)
    generator = FakeGenerator()
    analyse_document(source, settings(), generator, lambda: 30)
    evidence = json.loads(generator.calls[0]["content"])["evidence"]
    assert "".join(item["text"] for item in evidence) == "".join(chunk.text for chunk in source)
    assert all(len(item["text"]) <= MAX_QUOTE_CHARACTERS for item in evidence)


def test_large_document_reduction_retains_lineage_and_only_validated_quotes() -> None:
    source = chunks(20, words=300)
    generator = FakeGenerator(padding=20)
    result = analyse_document(
        source, settings(analysis_group_max_tokens=1800), generator, lambda: 30
    )
    requests = [json.loads(call["content"]) for call in generator.calls]
    assert any(request["stage"] == "section" for request in requests)
    assert requests[-1]["stage"] == "final" and requests[-1]["draft_summaries"]
    assert all("Supported fact." not in item["text"] for item in requests[-1]["evidence"])
    initial_literals = "".join(
        item["text"]
        for request in requests
        if request["stage"] == "section"
        for item in request["evidence"]
    )
    assert initial_literals == "".join(chunk.text for chunk in source)
    assert result.covered_chunk_ids == tuple(chunk.id for chunk in source)
    for block in result.payload["summary"]:
        for support in block["support_quotes"]:
            original = next(chunk.text for chunk in source if str(chunk.id) == support["source_id"])
            assert support["quote"] in original
    assert all(str(chunk.id) not in call["content"] for call in generator.calls for chunk in source)


@pytest.mark.parametrize(
    "problem,reason",
    [("first_unknown", "unknown_quote_ids"), ("first_duplicate", "duplicate_quote_ids")],
)
def test_invalid_references_get_one_safe_repair_with_identical_evidence(
    problem: str, reason: str
) -> None:
    generator = FakeGenerator(problem=problem)
    result = analyse_document(chunks(2), settings(), generator, lambda: 30)
    assert len(generator.calls) == 2
    assert generator.calls[0]["content"] == generator.calls[1]["content"]
    assert REPAIR_INSTRUCTION in generator.calls[1]["system"]
    assert f"Validation reason: {reason}." in generator.calls[1]["system"]
    assert result.payload["usage"]["call_history"][0]["validation_reason"] == reason


@pytest.mark.parametrize("problem", ["unknown", "duplicate"])
def test_persistent_invalid_reference_stops_after_two_calls(problem: str) -> None:
    generator = FakeGenerator(problem=problem)
    with pytest.raises(ProcessingError) as captured:
        analyse_document(chunks(2), settings(), generator, lambda: 30)
    assert captured.value.code == "analysis_citation_invalid"
    assert len(generator.calls) == 2


def test_transport_failure_is_not_repaired() -> None:
    generator = FakeGenerator(problem="timeout")
    with pytest.raises(ProcessingError) as captured:
        analyse_document(chunks(1), settings(), generator, lambda: 30)
    assert captured.value.retryable and len(generator.calls) == 1


@pytest.mark.parametrize("remaining", [0, -1, float("nan")])
def test_expired_deadline_prevents_generation(remaining: float) -> None:
    generator = FakeGenerator()
    with pytest.raises(ProcessingError) as captured:
        analyse_document(chunks(1), settings(), generator, lambda: remaining)
    assert captured.value.code == "processing_deadline" and not generator.calls


def test_repair_cannot_escape_call_budget() -> None:
    generator = FakeGenerator(problem="unknown")
    with pytest.raises(ProcessingError) as captured:
        analyse_document(chunks(1), settings(max_document_ai_calls=1), generator, lambda: 30)
    assert captured.value.code == "ai_call_budget" and len(generator.calls) == 1


def test_exhausted_persisted_budget_prevents_generation() -> None:
    generator = FakeGenerator()

    def reserve() -> float:
        raise ProcessingError("ai_call_budget", "Budget exhausted.")

    with pytest.raises(ProcessingError):
        analyse_document(chunks(1), settings(), generator, reserve)
    assert not generator.calls


def test_output_limit_applies_to_actual_selection_not_hydrated_uuid_payload() -> None:
    generator = FakeGenerator()
    result = analyse_document(
        chunks(1, words=30), settings(summary_max_output_tokens=70), generator, lambda: 30
    )
    assert (
        len(
            tiktoken.get_encoding("o200k_base").encode(
                AnalysisOutput.model_validate(
                    {"summary": result.payload["summary"], "insights": result.payload["insights"]}
                ).model_dump_json()
            )
        )
        > 70
    )
    assert generator.calls[0]["max_output_tokens"] == 70


def test_actual_oversized_selection_still_fails_after_single_repair() -> None:
    generator = FakeGenerator(padding=300)
    with pytest.raises(ProcessingError) as captured:
        analyse_document(chunks(1), settings(summary_max_output_tokens=100), generator, lambda: 30)
    assert captured.value.code == "analysis_output_budget" and len(generator.calls) == 2


def test_oversized_item_rejected_without_truncation() -> None:
    generator = FakeGenerator()
    with pytest.raises(ProcessingError) as captured:
        analyse_document(
            chunks(1, words=3000), settings(analysis_group_max_tokens=2000), generator, lambda: 30
        )
    assert captured.value.code == "analysis_input_limit" and not generator.calls


def test_empty_and_duplicate_input_rejected() -> None:
    source = chunks(1)
    for supplied in ([], [source[0], source[0]]):
        with pytest.raises(ProcessingError) as captured:
            analyse_document(supplied, settings(), FakeGenerator(), lambda: 30)
        assert captured.value.code == "analysis_input_invalid"


class StructuredFailure(FakeGenerator):
    def __init__(self, code: str, reason: str, *, persistent: bool = False) -> None:
        super().__init__()
        self.code, self.persistent = code, persistent
        self.diagnostic = AIFailureDetails(
            reason=reason,
            finish_reason="length",
            input_tokens=321,
            output_tokens=1500,
            max_output_tokens=1500,
            provider_request_id="req_synthetic_test",
        )

    def generate(self, **values: Any) -> Generated[Any]:
        result = super().generate(**values)
        if self.persistent or len(self.calls) == 1:
            raise ProcessingError(
                self.code, "UNTRUSTED_FAILED_OUTPUT_CANARY", diagnostics=self.diagnostic
            )
        return result


@pytest.mark.parametrize(
    "code,reason,repair_reason",
    [
        ("ai_output_truncated", "output_token_limit", "output_budget_exceeded"),
        ("ai_invalid_json", "invalid_json", "invalid_json"),
        ("ai_schema_invalid", "schema_validation", "schema_invalid"),
    ],
)
def test_structured_failure_compact_repair_preserves_known_usage(
    code: str, reason: str, repair_reason: str
) -> None:
    generator = StructuredFailure(code, reason)
    result = analyse_document(chunks(2), settings(), generator, lambda: 30)
    first, repair = generator.calls
    assert first["content"] == repair["content"]
    assert first["max_output_tokens"] == repair["max_output_tokens"] == 1500
    assert writing_target(repair["system"])[1] < writing_target(first["system"])[1]
    assert "ceilings, not required counts" in first["system"]
    assert f"Validation reason: {repair_reason}." in repair["system"]
    assert "UNTRUSTED_FAILED_OUTPUT_CANARY" not in repair["system"]
    assert (
        result.payload["usage"]["call_history"][0]["diagnostics"] == generator.diagnostic.as_dict()
    )
    assert result.payload["usage"]["all_calls_reported_usage"] is True


def test_persistent_structured_failure_retains_diagnostics_no_third_call() -> None:
    generator = StructuredFailure("ai_output_truncated", "output_token_limit", persistent=True)
    with pytest.raises(ProcessingError) as captured:
        analyse_document(chunks(1), settings(), generator, lambda: 30)
    assert captured.value.diagnostics == generator.diagnostic and len(generator.calls) == 2


def test_old_uuid_quote_schema_is_rejected_and_repaired() -> None:
    class LegacyResponse(FakeGenerator):
        def generate(self, **values: Any) -> Generated[Any]:
            result = super().generate(**values)
            if len(self.calls) == 1:
                return Generated(
                    result={
                        "summary": [
                            {"text": "Claim", "source_ids": ["bad UUID"], "support_quotes": []}
                        ],
                        "insights": [],
                    },
                    model=result.model,
                    input_tokens=100,
                    output_tokens=50,
                )
            return result

    generator = LegacyResponse()
    result = analyse_document(chunks(1), settings(), generator, lambda: 30)
    assert result.payload["usage"]["call_history"][0]["error_code"] == "ai_schema_invalid"
    assert "bad UUID" not in generator.calls[1]["system"]


def test_dynamic_prompt_and_selection_schema_counted_before_repair_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = chunks(2, words=100)
    configured = settings(analysis_group_max_tokens=800)
    baseline = FakeGenerator(problem="first_unknown")
    analyse_document(source, configured, baseline, lambda: 30)
    first = baseline.calls[0]
    exact = generation_prompt_tokens(SelectionOutput, first["system"], first["content"])
    monkeypatch.setattr(
        Settings, "INSTRUCTION_RESERVE_TOKENS", exact - configured.analysis_group_max_tokens
    )
    reserved: list[bool] = []

    def reserve() -> float:
        reserved.append(True)
        return 30

    generator = FakeGenerator(problem="first_unknown")
    with pytest.raises(ProcessingError) as captured:
        analyse_document(source, configured, generator, reserve)
    assert captured.value.code == "analysis_prompt_limit"
    assert len(generator.calls) == len(reserved) == 1


def test_synthetic_ten_chunks_fifteen_pages_195_blocks_keeps_full_evidence() -> None:
    source = chunks(10, words=320)
    decorated = []
    block = 0
    for index, chunk in enumerate(source):
        spans = []
        for _offset in range(20 if index < 5 else 19):
            spans.append(
                SourceSpan(
                    f"block-{block}",
                    {"page": block // 13 + 1, "block": block % 13 + 1},
                    0,
                    len(chunk.text),
                    0,
                    len(chunk.text),
                )
            )
            block += 1
        decorated.append(
            Chunk(chunk.id, chunk.ordinal, chunk.text, chunk.token_count, tuple(spans))
        )
    generator = StructuredFailure("ai_output_truncated", "output_token_limit")
    result = analyse_document(decorated, settings(), generator, lambda: 30)
    request = json.loads(generator.calls[0]["content"])
    assert "".join(item["text"] for item in request["evidence"]) == "".join(c.text for c in source)
    assert sum(len(c.source_spans) for c in decorated) == 195
    assert {span.locator["page"] for c in decorated for span in c.source_spans} == set(range(1, 16))
    assert "locators" not in generator.calls[0]["content"]
    assert result.covered_chunk_ids == tuple(c.id for c in source)
    assert (
        result.payload["covered_extracted_chunks"] == result.payload["total_extracted_chunks"] == 10
    )


@pytest.mark.parametrize(
    "problem,reason",
    [
        ("duplicate", "duplicate_source_ids"),
        ("unknown", "unknown_source_ids"),
        ("missing", "missing_source_quote"),
        ("uncited", "uncited_quote_source"),
        ("blank", "blank_support_quote"),
        ("invented", "quote_not_in_original"),
        ("unsupplied", "quote_not_in_supplied_evidence"),
        ("blank_text", "blank_factual_text"),
    ],
)
def test_public_source_validation_remains_strict(problem: str, reason: str) -> None:
    source, second = uuid4(), uuid4()
    block: dict[str, Any] = {
        "text": "Fact",
        "source_ids": [str(source)],
        "support_quotes": [{"source_id": str(source), "quote": "Original"}],
    }
    evidence = {source: ("Original",), second: ("Second",)}
    if problem == "duplicate":
        block["source_ids"] *= 2
    if problem == "unknown":
        block["source_ids"] = [str(uuid4())]
    if problem == "missing":
        block["source_ids"].append(str(second))
    if problem == "uncited":
        block["support_quotes"].append({"source_id": str(second), "quote": "Second"})
    if problem == "blank":
        block["support_quotes"][0]["quote"] = " "
    if problem == "invented":
        block["support_quotes"][0]["quote"] = "Invented"
    if problem == "unsupplied":
        block["support_quotes"][0]["quote"] = "Hidden"
    if problem == "blank_text":
        block["text"] = " "
    output = AnalysisOutput.model_validate({"summary": [block], "insights": []})
    with pytest.raises(ProcessingError) as captured:
        _validate_sources(
            output, evidence, {source: "Original Hidden", second: "Second"}, final=True
        )
    assert reason in captured.value.message


def test_final_analysis_can_support_blocks_from_more_than_eight_original_chunks() -> None:
    sources = [uuid4() for _ in range(9)]
    blocks = [
        {
            "text": f"Original fact {index}",
            "source_ids": [str(source)],
            "support_quotes": [{"source_id": str(source), "quote": f"Original fact {index}"}],
        }
        for index, source in enumerate(sources)
    ]
    output = AnalysisOutput.model_validate({"summary": blocks[:8], "insights": blocks[8:]})
    originals = {source: f"Original fact {index}" for index, source in enumerate(sources)}
    _validate_sources(
        output,
        {source: (text,) for source, text in originals.items()},
        originals,
        final=True,
    )
    assert output.insights[0].source_ids == [sources[-1]]


@pytest.mark.parametrize("depth", [0, 1])
def test_reduction_depth_is_bounded_and_last_permitted_depth_can_finish(
    monkeypatch: pytest.MonkeyPatch, depth: int
) -> None:
    import app.services.analysis as module

    monkeypatch.setattr(module, "MAX_REDUCTION_DEPTH", depth)
    generator = FakeGenerator(padding=130)
    source = chunks(45, words=550)
    configured = settings(analysis_group_max_tokens=2000, analysis_max_output_tokens=300)
    if depth == 0:
        with pytest.raises(ProcessingError) as captured:
            analyse_document(source, configured, generator, lambda: 30)
        assert captured.value.code == "analysis_reduction_limit"
    else:
        result = analyse_document(source, configured, generator, lambda: 30)
        assert result.payload["analysis_coverage"] == 1.0
        assert any(node["stage"] == "reduce" for node in result.payload["lineage"])


def test_failed_reference_usage_survives_diagnostics_enrichment() -> None:
    generator = FakeGenerator(problem="first_unknown")
    result = analyse_document(chunks(2), settings(), generator, lambda: 30)
    call = result.payload["usage"]["call_history"][0]
    assert call["input_tokens"] == call["diagnostics"]["input_tokens"] == 100
    assert call["output_tokens"] == call["diagnostics"]["output_tokens"] > 0
    assert result.payload["usage"]["all_calls_reported_usage"] is True


def writing_target(system: str) -> tuple[int, int]:
    match = re.search(r"Writing target: about (\d+)-(\d+) tokens", system)
    assert match is not None
    return int(match[1]), int(match[2])


def test_longer_content_receives_useful_depth_within_unchanged_generation_cap() -> None:
    short, long = FakeGenerator(), FakeGenerator()
    analyse_document(chunks(1, words=10), settings(), short, lambda: 30)
    substantive = [
        Chunk(
            item.id,
            item.ordinal,
            item.text.replace("detail", f"section{item.ordinal}detail"),
            item.token_count,
            item.source_spans,
        )
        for item in chunks(10, words=200)
    ]
    analyse_document(substantive, settings(), long, lambda: 30)
    short_floor, short_target = writing_target(short.calls[0]["system"])
    long_floor, long_target = writing_target(long.calls[0]["system"])
    assert 0 < short_floor <= short_target < long_floor <= long_target < 1500
    # The previous 200-character/block guidance encouraged only a few lines even
    # for a large document. Substantive content now has room for a fuller account.
    assert long_target >= 750
    assert short.calls[0]["max_output_tokens"] == long.calls[0]["max_output_tokens"] == 1500
    assert "within 200 characters" not in long.calls[0]["system"]
    assert "Do not pad" in long.calls[0]["system"]


@pytest.mark.parametrize("output_tokens", [70, 128, 300, 768, 1500])
def test_small_budgets_and_repairs_keep_prose_below_actual_cap(output_tokens: int) -> None:
    first = analysis_budget_instruction(
        output_tokens, evidence_tokens=10000, final=True, repair=False
    )
    repaired = analysis_budget_instruction(
        output_tokens, evidence_tokens=10000, final=True, repair=True
    )
    first_floor, first_target = writing_target(first)
    repair_floor, repair_target = writing_target(repaired)
    assert 0 < repair_floor <= repair_target <= first_target < output_tokens
    assert first_floor <= first_target
    assert f"{output_tokens} tokens TOTAL" in first
    assert f"{output_tokens} tokens TOTAL" in repaired


def test_reduction_retains_length_guidance_from_represented_originals() -> None:
    source = [
        Chunk(
            item.id,
            item.ordinal,
            item.text.replace("detail", f"section{item.ordinal}detail"),
            item.token_count,
            item.source_spans,
        )
        for item in chunks(12, words=200)
    ]
    generator = FakeGenerator()
    result = analyse_document(
        source, settings(analysis_group_max_tokens=2000), generator, lambda: 30
    )
    final = generator.calls[-1]
    assert json.loads(final["content"])["stage"] == "final"
    assert len(json.loads(final["content"])["evidence"]) < 20
    # Reduced literal support is short, but it represents a substantive document.
    # Its final synthesis must not inherit a tiny writing target from quote length.
    assert writing_target(final["system"])[1] >= 750
    assert result.covered_chunk_ids == tuple(chunk.id for chunk in source)


def test_quality_guidance_and_every_repair_fit_existing_instruction_reserve() -> None:
    for reason in REPAIR_REASONS:
        system = (
            ANALYSIS_SYSTEM
            + analysis_budget_instruction(1500, evidence_tokens=100000, final=True, repair=True)
            + analysis_repair_instruction(reason)
        )
        assert (
            generation_prompt_tokens(SelectionOutput, system, "")
            <= Settings.INSTRUCTION_RESERVE_TOKENS
        )


def test_catalog_groups_original_chunks_without_exposing_ids_or_losing_text() -> None:
    source = chunks(2, words=100)
    generator = FakeGenerator()
    analyse_document(source, settings(), generator, lambda: 30)
    evidence = json.loads(generator.calls[0]["content"])["evidence"]
    assert {item["source_group"] for item in evidence} == {"s1", "s2"}
    for index, chunk in enumerate(source, start=1):
        passages = [item["text"] for item in evidence if item["source_group"] == f"s{index}"]
        assert len(passages) > 1
        assert "".join(passages) == chunk.text
        assert str(chunk.id) not in generator.calls[0]["content"]


def test_synthesis_can_cite_more_than_eight_passages_across_blocks() -> None:
    class SynthesisGenerator(FakeGenerator):
        def generate(self, **values: Any) -> Generated[Any]:
            self.calls.append(values)
            evidence = json.loads(values["content"])["evidence"]
            output = values["schema"].model_validate(
                {
                    "summary": [
                        {
                            "text": (
                                "The document outlines an annually renewed Standard plan for "
                                "monitoring stored documents. It combines monthly usage reports, "
                                "retention of originals and conversation history, CSV export and "
                                "local backups. Responsibility for keeping the API key remains "
                                "with the owner."
                            ),
                            "quote_ids": [item["quote_id"] for item in evidence[:8]],
                        }
                    ],
                    "insights": [
                        {
                            "text": (
                                "Annual renewal alone does not increase storage; additional "
                                "storage requires a separate purchase."
                            ),
                            "quote_ids": [evidence[index]["quote_id"] for index in (0, 8, 9)],
                        }
                    ],
                }
            )
            return Generated(
                result=output,
                model=Settings.CHAT_MODEL,
                input_tokens=500,
                output_tokens=len(
                    tiktoken.get_encoding("o200k_base").encode(output.model_dump_json())
                ),
            )

    facts = [
        "The Standard plan renews annually.",
        "The service monitors stored documents.",
        "Usage reports are available each month.",
        "The service retains document originals.",
        "The service retains conversation history.",
        "Data export is available in CSV format.",
        "The plan supports local backups.",
        "The owner is responsible for keeping the API key.",
        "Renewal does not activate additional storage.",
        "Additional storage requires a separate purchase.",
    ]
    text = "\n".join(
        fact + " This statement applies to the Standard plan described in this document."
        for fact in facts
    )
    source = [Chunk(uuid4(), 0, text, 300, ())]
    generator = SynthesisGenerator()
    result = analyse_document(source, settings(), generator, lambda: 30)
    summary, insight = result.payload["summary"][0], result.payload["insights"][0]
    assert len(summary["text"]) > 200
    assert summary["text"] != insight["text"]
    assert len(summary["support_quotes"]) == 8
    quotes = {
        support["quote"] for block in (summary, insight) for support in block["support_quotes"]
    }
    assert len(quotes) == 10
    assert summary["source_ids"] == insight["source_ids"] == [str(source[0].id)]
    assert all(quote in source[0].text for quote in quotes)
    assert result.payload["usage"]["calls"] == 1
    assert len(generator.calls) == 1


def test_writing_depth_estimate_deduplicates_repetition_without_dropping_distinct_facts() -> None:
    from app.services.analysis import _writing_units

    boilerplate = "Routine operations use the checklist and retain supporting records."
    assert _writing_units((boilerplate + "\n") * 100) == (boilerplate,)
    facts = "Threshold is 101 units. Threshold is 102 units."
    units = _writing_units(boilerplate + "\n" + facts + "\n" + boilerplate)
    assert len(units) == 3
    assert "Threshold is 101 units." in units
    assert "Threshold is 102 units." in units
