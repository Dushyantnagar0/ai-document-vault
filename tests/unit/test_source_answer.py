"""Offline source-answer contracts, not generated-answer quality proof."""

import ast
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
import tiktoken
from pydantic import ValidationError

from app.core.config import Settings
from app.domain.conversation import Evidence
from app.domain.documents import ProcessingError
from app.prompts.analysis import generation_prompt_tokens
from app.prompts.chat import ANSWER_PROMPT_VERSION
from app.services.analysis_evidence import QuoteReference
from app.services.chat_context import answer_quote_catalog, prepare_context
from app.services.chat_review import CHAT_REVIEW_PROMPT_VERSION, ChatReviewOutput
from app.services.source_answer import (
    SOURCE_COMPOSITION_REPAIR,
    SOURCE_COMPOSITION_SYSTEM,
    SOURCE_SELECTION_REPAIR,
    SOURCE_SELECTION_SYSTEM,
    CompositionOutput,
    SourceAnswerPlan,
    SourceSelectionOutput,
    composition_input_reservation,
    prepare_composition_packet,
    prepare_selection_packet,
    project_composition,
    source_answer_input_limit,
    validate_selection,
)

pytestmark = pytest.mark.unit
DOCUMENT, PROFILE, SOURCE = UUID(int=1), UUID(int=2), UUID(int=3)


def settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, openai_api_key="unit-placeholder").model_copy(update=overrides)


def evidence(text: str, **overrides: Any) -> Evidence:
    return replace(
        Evidence(SOURCE, DOCUMENT, PROFILE, "synthetic.txt", text, ()),
        **overrides,
    )


def fixture(
    *, tolerance: str = "6 hours", rate: str = "EUR 3", cap: str = "EUR 15"
) -> tuple[dict[str, QuoteReference], tuple[Evidence, ...]]:
    # Nonconsecutive IDs deliberately represent a saved, previously built catalog.
    passages = {
        "q11": "Written dispatch approval replaces the delivery target.",
        "q18": f"The {tolerance} tolerance is measured from the revised target.",
        "q31": "Requests should reach dispatch two business days in advance.",
        "q50": "Approval must be recorded before the original target.",
        "q61": f"A late delivery costs {rate} per started day, capped at {cap} per delivery.",
        "q74": "The next dispatch needs four hours for inspection and loading.",
        "q83": "An account with an unpaid invoice cannot place a new order.",
    }
    catalog = {key: QuoteReference(SOURCE, text) for key, text in passages.items()}
    return catalog, (evidence("\n\n".join(passages.values())),)


def selection(*, missing: bool = False) -> SourceSelectionOutput:
    components = [
        {"label": "Approved rescheduling and its consequences", "quote_ids": ["q11", "q18", "q50"]},
        {"label": "Charges for an unapproved delay", "quote_ids": ["q61"]},
        {"label": "Reason for checking the next dispatch", "quote_ids": ["q74"]},
    ]
    if missing:
        components.append({"label": "Weekend delivery arrangements", "quote_ids": []})
    return SourceSelectionOutput.model_validate(
        {"evidence_status": "partial" if missing else "answered", "components": components}
    )


def composition(
    *, tolerance: str = "6 hours", rate: str = "EUR 3", cap: str = "EUR 15", missing: bool = False
) -> CompositionOutput:
    parts: list[dict[str, Any]] = [
        {
            "component_id": "c1",
            "status": "supported",
            "quote_ids": ["q11", "q18", "q50"],
            "text": (
                "Written approval before the original target replaces that target; "
                f"the {tolerance} tolerance then runs from the revised target."
            ),
        },
        {
            "component_id": "c2",
            "status": "supported",
            "quote_ids": ["q61"],
            "text": f"A late delivery costs {rate} for each started day, up to {cap} per delivery.",
        },
        {
            "component_id": "c3",
            "status": "supported",
            "quote_ids": ["q74"],
            "text": "The next dispatch needs four hours for inspection and loading.",
        },
    ]
    if missing:
        parts.append({"component_id": "c4", "status": "unsupported", "quote_ids": [], "text": ""})
    return CompositionOutput.model_validate({"parts": parts})


def plan(*, missing: bool = False, complete_scope: bool = True) -> SourceAnswerPlan:
    catalog, _ = fixture()
    return validate_selection(
        selection(missing=missing),
        catalog,
        question="Compare a reschedule with a delay and explain the dispatch check.",
        resolved_query=(
            "Compare approved rescheduling and delay consequences; explain dispatch check."
        ),
        complete_scope=complete_scope,
    )


def test_source_only_selection_and_selected_only_composition_preserve_original_ids() -> None:
    catalog, _ = fixture()
    packet = prepare_selection_packet(
        "How do these options differ?", "Compare the options.", catalog, settings()
    )
    content = json.loads(packet.content)
    assert packet.schema is SourceSelectionOutput
    assert set(content) == {"question", "retrieval_query", "evidence", "source_index"}
    assert [item["quote_id"] for item in content["evidence"]] == list(catalog)
    assert content["source_index"] == {"s1": str(SOURCE)}
    assert packet.max_output_tokens == 4096

    composed = prepare_composition_packet(plan(), settings())
    content = json.loads(composed.content)
    assert composed.schema is CompositionOutput and composed.max_output_tokens == 1200
    assert [item["component_id"] for item in content["components"]] == ["c1", "c2", "c3"]
    assert {item["quote_id"] for item in content["evidence"]} == {"q11", "q18", "q50", "q61", "q74"}
    assert "unpaid invoice" not in composed.content
    assert "q31" not in composed.content and "q83" not in composed.content
    assert "history" not in content and "claims" not in content and "draft" not in content
    for item in content["evidence"]:
        assert item["text"] == catalog[item["quote_id"]].quote


@pytest.mark.parametrize(
    ("tolerance", "rate", "cap"), [("6 hours", "EUR 3", "EUR 15"), ("9 hours", "$7", "$35")]
)
def test_valid_paraphrase_keeps_supplied_relationships_and_limits_without_appending_sources(
    tolerance: str, rate: str, cap: str
) -> None:
    catalog, sources = fixture(tolerance=tolerance, rate=rate, cap=cap)
    frozen = validate_selection(selection(), catalog, question="Compare the delivery arrangements.")
    generated = composition(tolerance=tolerance, rate=rate, cap=cap)
    result = project_composition(frozen, generated, sources)
    assert result.evidence_status == "answered"
    assert [block.text for block in result.blocks] == [part.text for part in generated.parts]
    assert tolerance in result.answer and rate in result.answer and cap in result.answer
    assert result.answer.count("revised target") == 1
    assert "unpaid invoice" not in result.answer
    assert len(result.citations) <= 8
    assert all(len(item["excerpt"]) <= 1200 for item in result.citations)
    assert all(item["locators"] == [] for item in result.citations)
    assert all(item["document_id"] == str(DOCUMENT) for item in result.citations)


def test_component_selection_does_not_masquerade_as_semantic_coverage_validation() -> None:
    # Deliberately incomplete prose can still pass identity checks. This test
    # documents the boundary; quality acceptance must inspect generated meaning.
    _, sources = fixture()
    generated = composition().model_dump()
    generated["parts"][0]["text"] = "Approval changes the target."
    result = project_composition(plan(), CompositionOutput.model_validate(generated), sources)
    assert result.blocks[0].text == "Approval changes the target."
    assert "6 hours" not in result.answer


@pytest.mark.parametrize("select_other_passage", [False, True])
@pytest.mark.parametrize(
    ("first_passage", "other_passage", "claim"),
    [
        ("Owner: Alice Wong.", "Final contact: Mira Stone.", "Mira Stone is the final contact."),
        (
            "See section 7 for allocation details.",
            "See section 12 for final approval.",
            "Section 12 describes final approval.",
        ),
    ],
)
def test_full_catalog_copied_identifiers_require_actual_selected_support(
    select_other_passage: bool, first_passage: str, other_passage: str, claim: str
) -> None:
    other_source = UUID(int=4)
    catalog = {
        "q1": QuoteReference(SOURCE, first_passage),
        "q2": QuoteReference(other_source, other_passage),
    }
    selected = SourceSelectionOutput.model_validate(
        {
            "evidence_status": "answered",
            "components": [
                {
                    "label": "Requested information",
                    "quote_ids": ["q1", "q2"] if select_other_passage else ["q1"],
                }
            ],
        }
    )
    frozen = validate_selection(selected, catalog, question="What does the source establish?")
    sources = (evidence(first_passage), evidence(other_passage, id=other_source))
    packet = prepare_composition_packet(frozen, settings())
    if not select_other_passage:
        assert other_passage not in packet.content
        assert "q2" not in packet.content
        assert len(frozen.passages) == 1
    assert len(frozen.original_catalog) == 2
    # Mutating the caller's catalog cannot erase an original copied identifier.
    catalog.pop("q2")
    generated = CompositionOutput.model_validate(
        {
            "parts": [
                {
                    "component_id": "c1",
                    "status": "supported",
                    "quote_ids": ["q1"],
                    "text": claim,
                }
            ]
        }
    )
    result = project_composition(frozen, generated, sources)
    assert result.evidence_status == "insufficient"
    assert not result.blocks and not result.citations and claim not in result.answer


def test_copied_name_rejection_retains_other_useful_components_as_partial() -> None:
    passages = {
        "q1": QuoteReference(SOURCE, "Owner: Alice Wong."),
        "q2": QuoteReference(UUID(int=4), "Final contact: Mira Stone."),
        "q3": QuoteReference(UUID(int=5), "Inspection requires four hours."),
    }
    sources = tuple(evidence(ref.quote, id=ref.source_id) for ref in passages.values())
    frozen = validate_selection(
        SourceSelectionOutput.model_validate(
            {
                "evidence_status": "answered",
                "components": [
                    {"label": "Final contact", "quote_ids": ["q1", "q2"]},
                    {"label": "Inspection duration", "quote_ids": ["q3"]},
                ],
            }
        ),
        passages,
        question="Who is the final contact and how long does inspection take?",
    )
    generated = CompositionOutput.model_validate(
        {
            "parts": [
                {
                    "component_id": "c1",
                    "status": "supported",
                    "text": "Mira Stone is the final contact.",
                    "quote_ids": ["q1"],
                },
                {
                    "component_id": "c2",
                    "status": "supported",
                    "text": "Inspection requires four hours.",
                    "quote_ids": ["q3"],
                },
            ]
        }
    )
    result = project_composition(frozen, generated, sources)
    assert result.evidence_status == "partial"
    assert [block.text for block in result.blocks] == ["Inspection requires four hours."]
    assert "- Final contact" in result.answer
    assert "Mira Stone" not in result.answer
    assert [citation["source_id"] for citation in result.citations] == [str(UUID(int=5))]

    corrected = generated.model_dump()
    corrected["parts"][0]["quote_ids"] = ["q2"]
    assert (
        project_composition(
            frozen, CompositionOutput.model_validate(corrected), sources
        ).evidence_status
        == "answered"
    )


def test_copied_identifier_guard_keeps_ordinary_lowercase_words() -> None:
    catalog = {
        "q1": QuoteReference(
            SOURCE, "Stage materials according to the planned assembly timetable."
        ),
        "q2": QuoteReference(UUID(int=4), "Assembly Schedule"),
    }
    frozen = validate_selection(
        SourceSelectionOutput.model_validate(
            {
                "evidence_status": "answered",
                "components": [{"label": "Material staging", "quote_ids": ["q1"]}],
            }
        ),
        catalog,
        question="How should materials be staged?",
    )
    output = CompositionOutput.model_validate(
        {
            "parts": [
                {
                    "component_id": "c1",
                    "status": "supported",
                    "text": "The assembly schedule guides material staging.",
                    "quote_ids": ["q1"],
                }
            ]
        }
    )
    sources = tuple(evidence(ref.quote, id=ref.source_id) for ref in catalog.values())
    result = project_composition(frozen, output, sources)
    assert result.evidence_status == "answered"
    assert result.blocks[0].text == output.parts[0].text


def test_partial_result_names_missing_requested_information_separately_from_facts() -> None:
    _, sources = fixture()
    result = project_composition(plan(missing=True), composition(missing=True), sources)
    assert result.evidence_status == "partial"
    assert (
        "Not established by the supplied evidence:\n- Weekend delivery arrangements"
        in result.answer
    )
    assert all("Weekend delivery" not in block.text for block in result.blocks)


def test_composer_can_decline_a_selected_component_and_all_selected_components() -> None:
    _, sources = fixture()
    output = composition().model_dump()
    output["parts"][1].update(status="unsupported", text="", quote_ids=[])
    partial = project_composition(plan(), CompositionOutput.model_validate(output), sources)
    assert partial.evidence_status == "partial"
    assert "- Charges for an unapproved delay" in partial.answer
    for part in output["parts"]:
        part.update(status="unsupported", text="", quote_ids=[])
    refused = project_composition(plan(), CompositionOutput.model_validate(output), sources)
    assert (
        refused.evidence_status == "insufficient" and not refused.citations and not refused.blocks
    )


def test_composition_downgrades_keep_the_existing_four_missing_component_limit() -> None:
    catalog, sources = fixture()
    selected = SourceSelectionOutput.model_validate(
        {
            "evidence_status": "answered",
            "components": [
                {"label": f"Requested component {index}", "quote_ids": ["q11"]}
                for index in range(8)
            ],
        }
    )
    frozen = validate_selection(selected, catalog, question="Explain the requested components.")
    parts = [
        {
            "component_id": f"c{index + 1}",
            "status": "unsupported" if index < 4 else "supported",
            "quote_ids": [] if index < 4 else ["q11"],
            "text": "" if index < 4 else "Written approval replaces the delivery target.",
        }
        for index in range(8)
    ]
    output = CompositionOutput.model_validate({"parts": parts})
    assert project_composition(frozen, output, sources).evidence_status == "partial"
    parts[4].update(status="unsupported", text="", quote_ids=[])
    with pytest.raises(ProcessingError) as failure:
        project_composition(frozen, CompositionOutput.model_validate({"parts": parts}), sources)
    assert failure.value.diagnostics is not None
    assert failure.value.diagnostics.reason == "invalid_qualification_selection"


@pytest.mark.parametrize("components", [[], [{"label": "Requested information", "quote_ids": []}]])
def test_no_supported_components_remains_insufficient(components: list[dict[str, Any]]) -> None:
    catalog, sources = fixture()
    output = SourceSelectionOutput(evidence_status="insufficient", components=components)
    frozen = validate_selection(output, catalog, question="What does the unavailable section say?")
    composed = CompositionOutput.model_validate(
        {
            "parts": [
                {"component_id": component.id, "status": "unsupported", "quote_ids": [], "text": ""}
                for component in frozen.components
            ]
        }
    )
    assert not frozen.passages
    assert project_composition(frozen, composed, sources).evidence_status == "insufficient"


@pytest.mark.parametrize("bad_id", ["q0", "q01", "q11,q18", "q11 q18", "s1", "Q11", "q-1", "q1\n"])
def test_quote_id_grammar_rejects_malformed_individual_selections(bad_id: str) -> None:
    with pytest.raises(ValidationError):
        SourceSelectionOutput.model_validate(
            {
                "evidence_status": "answered",
                "components": [{"label": "Requested rule", "quote_ids": [bad_id]}],
            }
        )


@pytest.mark.parametrize(
    "change",
    [
        {"evidence_status": "answered", "components": []},
        {"evidence_status": "partial", "components": [{"label": "Rule", "quote_ids": ["q11"]}]},
        {
            "evidence_status": "insufficient",
            "components": [{"label": "Rule", "quote_ids": ["q11"]}],
        },
        {"evidence_status": "answered", "components": [{"label": "Rule", "quote_ids": []}]},
        {
            "evidence_status": "answered",
            "components": [{"label": "Rule", "quote_ids": ["q11", "q11"]}],
        },
        {
            "evidence_status": "answered",
            "components": [{"label": "Rule", "quote_ids": ["q11"]}] * 2,
        },
        {"evidence_status": "answered", "components": [{"label": " ", "quote_ids": ["q11"]}]},
        {
            "evidence_status": "answered",
            "components": [{"label": "Hidden\tlabel", "quote_ids": ["q11"]}],
        },
        {
            "evidence_status": "answered",
            "components": [{"label": "Citation [1]", "quote_ids": ["q11"]}],
        },
        {"evidence_status": "answered", "components": [{"label": "x" * 61, "quote_ids": ["q11"]}]},
    ],
)
def test_selection_status_labels_and_duplicate_contracts(change: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        SourceSelectionOutput.model_validate(change)


def test_selection_bounds_eight_components_eight_quotes_and_four_missing_labels() -> None:
    components = [
        {"label": f"Requested component {index}", "quote_ids": ["q11"]} for index in range(8)
    ]
    assert (
        len(SourceSelectionOutput(evidence_status="answered", components=components).components)
        == 8
    )
    with pytest.raises(ValidationError):
        SourceSelectionOutput(
            evidence_status="answered",
            components=components + [{"label": "Ninth", "quote_ids": ["q11"]}],
        )
    components[0]["quote_ids"] = [f"q{index + 1}" for index in range(9)]
    with pytest.raises(ValidationError):
        SourceSelectionOutput(evidence_status="answered", components=components)
    with pytest.raises(ValidationError):
        SourceSelectionOutput(
            evidence_status="insufficient",
            components=[{"label": f"Missing {index}", "quote_ids": []} for index in range(5)],
        )


def test_unknown_catalog_ids_and_too_many_source_identities_fail_before_composition() -> None:
    catalog, _ = fixture()
    output = SourceSelectionOutput.model_validate(
        {"evidence_status": "answered", "components": [{"label": "A rule", "quote_ids": ["q999"]}]}
    )
    with pytest.raises(ProcessingError, match="could not be validated") as failure:
        validate_selection(output, catalog, question="A question")
    assert failure.value.diagnostics is not None
    assert failure.value.diagnostics.reason == "unknown_quote_id"
    many = {
        f"q{index + 1}": QuoteReference(UUID(int=100 + index), f"Rule {index}.")
        for index in range(9)
    }
    output = SourceSelectionOutput.model_validate(
        {
            "evidence_status": "answered",
            "components": [
                {"label": "First group", "quote_ids": list(many)[:8]},
                {"label": "Second group", "quote_ids": ["q9"]},
            ],
        }
    )
    with pytest.raises(ProcessingError) as failure:
        validate_selection(output, many, question="A question")
    assert failure.value.diagnostics is not None
    assert failure.value.diagnostics.reason == "final_source_limit"


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "unknown", "cross_component", "unselected"]
)
def test_composition_cannot_drop_repeat_invent_or_cross_bind_components(mutation: str) -> None:
    _, sources = fixture()
    output = composition().model_dump()
    if mutation == "missing":
        output["parts"].pop()
    elif mutation == "duplicate":
        output["parts"].append(output["parts"][0])
    elif mutation == "unknown":
        output["parts"][0]["component_id"] = "c8"
    elif mutation == "cross_component":
        output["parts"][0]["quote_ids"] = ["q61"]
    else:
        output["parts"][0]["quote_ids"] = ["q83"]
    with pytest.raises(ProcessingError):
        project_composition(plan(), CompositionOutput.model_validate(output), sources)


def test_unselected_component_cannot_be_resurrected_from_another_components_sources() -> None:
    _, sources = fixture()
    output = composition(missing=True).model_dump()
    output["parts"][-1].update(
        status="supported", text="An invented weekend claim.", quote_ids=["q11"]
    )
    with pytest.raises(ProcessingError):
        project_composition(plan(missing=True), CompositionOutput.model_validate(output), sources)


@pytest.mark.parametrize(
    "change",
    [
        {"status": "unsupported"},
        {"status": "supported", "text": ""},
        {"status": "supported", "quote_ids": []},
        {"quote_ids": ["q11", "q11"]},
        {"quote_ids": ["q11,q18"]},
        {"component_id": "c9"},
    ],
)
def test_composition_support_contract_rejects_malformed_parts(change: dict[str, Any]) -> None:
    output = composition().model_dump()
    output["parts"][0].update(change)
    with pytest.raises(ValidationError):
        CompositionOutput.model_validate(output)


def test_composition_order_is_frozen_plan_order_and_repairs_freeze_original_evidence() -> None:
    catalog, sources = fixture()
    selected = selection()
    frozen = validate_selection(selected, catalog, question="Compare the options.")
    first = prepare_composition_packet(frozen, settings())
    catalog["q18"] = QuoteReference(SOURCE, "An altered later passage.")
    selected.components[0].quote_ids.clear()
    repaired = prepare_composition_packet(frozen, settings(), repair=True)
    assert first.content == repaired.content
    assert "An altered later passage" not in repaired.content
    assert repaired.system == first.system + SOURCE_COMPOSITION_REPAIR
    with pytest.raises(FrozenInstanceError):
        frozen.question = "Changed"  # type: ignore[misc]
    output = composition().model_dump()
    output["parts"].reverse()
    rendered = project_composition(frozen, CompositionOutput.model_validate(output), sources)
    assert rendered.blocks[0].text == composition().parts[0].text


def test_scope_and_citation_window_overflow_downgrade_without_inventing_evidence() -> None:
    _, sources = fixture()
    partial = project_composition(plan(complete_scope=False), composition(), sources)
    assert partial.evidence_status == "partial"
    assert "do not establish complete coverage" in partial.answer

    quotes = [f"Distinct rule number {index}." for index in range(9)]
    long_source = evidence(("z" * 1300).join(quotes))
    catalog = {f"q{index + 1}": QuoteReference(SOURCE, quote) for index, quote in enumerate(quotes)}
    selected = SourceSelectionOutput.model_validate(
        {
            "evidence_status": "answered",
            "components": [
                {"label": "First group", "quote_ids": list(catalog)[:8]},
                {"label": "Last group", "quote_ids": ["q9"]},
            ],
        }
    )
    frozen = validate_selection(selected, catalog, question="What are the requested rules?")
    composed = CompositionOutput.model_validate(
        {
            "parts": [
                {
                    "component_id": "c1",
                    "status": "supported",
                    "quote_ids": list(catalog)[:8],
                    "text": "First rules.",
                },
                {
                    "component_id": "c2",
                    "status": "supported",
                    "quote_ids": ["q9"],
                    "text": "Last rule.",
                },
            ]
        }
    )
    result = project_composition(frozen, composed, (long_source,))
    assert result.evidence_status == "partial" and len(result.citations) == 8
    assert "Last rule." not in result.answer and "could not be included" in result.answer


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_source",
        "missing_quote",
        "duplicate",
        "different_document",
        "different_profile",
        "negative_offset",
    ],
)
def test_projection_rechecks_supplied_evidence_identity_and_literal_presence(mutation: str) -> None:
    _, original = fixture()
    sources: tuple[Evidence, ...] = original
    if mutation == "wrong_source":
        sources = (replace(original[0], id=UUID(int=33)),)
    elif mutation == "missing_quote":
        sources = (replace(original[0], text="No selected passage."),)
    elif mutation == "duplicate":
        sources = (*original, original[0])
    elif mutation == "different_document":
        sources = (*original, replace(original[0], id=UUID(int=33), document_id=UUID(int=44)))
    elif mutation == "different_profile":
        sources = (*original, replace(original[0], id=UUID(int=33), profile_id=UUID(int=44)))
    else:
        sources = (replace(original[0], text_offset=-1),)
    with pytest.raises(ProcessingError):
        project_composition(plan(), composition(), sources)


def test_actual_full_prompts_schema_repair_and_configured_output_limits_are_counted() -> None:
    catalog, _ = fixture()
    configured = settings(chat_max_output_tokens=900, evidence_review_max_output_tokens=3000)
    selected = prepare_selection_packet("A question", None, catalog, configured)
    composed = prepare_composition_packet(plan(), configured)
    assert selected.max_output_tokens == 3000 and composed.max_output_tokens == 900
    assert selected.input_tokens == generation_prompt_tokens(
        SourceSelectionOutput, SOURCE_SELECTION_SYSTEM, selected.content
    )
    assert selected.reserved_input_tokens == generation_prompt_tokens(
        SourceSelectionOutput, SOURCE_SELECTION_SYSTEM + SOURCE_SELECTION_REPAIR, selected.content
    )
    assert composed.input_tokens == generation_prompt_tokens(
        CompositionOutput, SOURCE_COMPOSITION_SYSTEM, composed.content
    )
    assert composed.reserved_input_tokens == generation_prompt_tokens(
        CompositionOutput, SOURCE_COMPOSITION_SYSTEM + SOURCE_COMPOSITION_REPAIR, composed.content
    )
    tight = settings(chat_prompt_max_tokens=composed.reserved_input_tokens - 1)
    with pytest.raises(ProcessingError) as failure:
        prepare_composition_packet(plan(), tight)
    assert failure.value.code == "SOURCE_ANSWER_INPUT_LIMIT"
    assert "A question" not in failure.value.message


def test_selection_reserves_composition_before_work_and_final_guard_remains_authoritative() -> None:
    catalog, _ = fixture()
    question = "A focused question"
    reserve = composition_input_reservation(question, None, catalog, settings())
    with pytest.raises(ProcessingError) as failure:
        prepare_selection_packet(
            question, None, catalog, settings(chat_prompt_max_tokens=reserve - 1)
        )
    assert failure.value.code == "SOURCE_ANSWER_INPUT_LIMIT"
    prepared = prepare_selection_packet(
        question, None, catalog, settings(chat_prompt_max_tokens=reserve)
    )
    assert prepared.reserved_input_tokens <= reserve
    frozen = validate_selection(selection(), catalog, question=question)
    assert prepare_composition_packet(frozen, settings()).reserved_input_tokens <= reserve


def test_near_full_existing_context_fits_without_mutation_or_larger_input_limit() -> None:
    # Bounded neutral labels leave room for this existing retrieval boundary;
    # no passage is silently dropped to make the candidate's reservation fit.
    configured = settings()
    question = "Compare transport conditions."
    original = evidence(
        " ".join(
            f"Transport entry {index} has measured value {index + 21} under condition blue."
            for index in range(1000)
        )
    )
    current = prepare_context(
        question, [], [original], configured, intent="factual", complete_scope=True
    )
    frozen_content = current.content
    catalog = answer_quote_catalog(current.evidence)
    frozen_catalog = dict(catalog)
    encoded = tiktoken.get_encoding("o200k_base")
    source_tokens = len(
        encoded.encode(
            json.dumps(
                json.loads(current.content)["evidence"],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    )
    query = " detail" * 256
    assert 6350 <= source_tokens <= 6400
    assert len(encoded.encode(query)) == 256
    assert current.token_count < configured.chat_prompt_max_tokens == 12000
    reservation = composition_input_reservation(question, query, catalog, configured)
    assert reservation <= configured.chat_prompt_max_tokens
    packet = prepare_selection_packet(question, query, catalog, configured)
    assert packet.reserved_input_tokens <= configured.chat_prompt_max_tokens
    assert len(json.loads(packet.content)["evidence"]) == len(catalog)
    assert catalog == frozen_catalog and current.content == frozen_content
    assert configured.chat_prompt_max_tokens == 12000


def test_reservation_covers_maximum_unicode_labels_and_longest_individual_identifiers() -> None:
    catalog = {"q" + "9" * 29: QuoteReference(SOURCE, "A supported rule.")}
    components = [
        {"label": chr(0x10000 + index) * 60, "quote_ids": list(catalog)} for index in range(8)
    ]
    frozen = validate_selection(
        SourceSelectionOutput(evidence_status="answered", components=components),
        catalog,
        question="A question",
    )
    packet = prepare_composition_packet(frozen, settings())
    reservation = composition_input_reservation("A question", None, catalog, settings())
    assert packet.reserved_input_tokens <= reservation
    components[0]["label"] += chr(0x10000)
    with pytest.raises(ValidationError):
        SourceSelectionOutput(evidence_status="answered", components=components)


def test_provider_input_context_and_output_caps_are_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured = settings()
    monkeypatch.setattr(Settings, "CHAT_MODEL_MAX_INPUT_TOKENS", 1000)
    monkeypatch.setattr(Settings, "CHAT_CONTEXT_TOKENS", 1500)
    assert source_answer_input_limit(configured, max_output_tokens=1200) == 300
    with pytest.raises(ProcessingError):
        prepare_composition_packet(plan(), configured)
    monkeypatch.setattr(Settings, "CHAT_MODEL_MAX_OUTPUT_TOKENS", 1000)
    with pytest.raises(ProcessingError):
        prepare_composition_packet(plan(), configured)


def test_output_budget_rejects_oversized_visible_json_without_source_content_in_errors() -> None:
    catalog, sources = fixture()
    with pytest.raises(ProcessingError) as failure:
        validate_selection(selection(), catalog, question="A question", max_output_tokens=1)
    assert failure.value.code == "SOURCE_ANSWER_OUTPUT_LIMIT"
    assert "dispatch" not in failure.value.message
    with pytest.raises(ProcessingError) as failure:
        project_composition(plan(), composition(), sources, max_output_tokens=1)
    assert failure.value.code == "SOURCE_ANSWER_OUTPUT_LIMIT"


def test_source_helpers_remain_pure_and_preserve_historical_review_contract() -> None:
    service_dir = Path(__file__).parents[2] / "app" / "services"
    candidate = service_dir / "source_answer.py"
    assert all(
        not isinstance(node, (ast.AsyncFunctionDef, ast.Await))
        for node in ast.walk(ast.parse(candidate.read_text()))
    )
    assert ANSWER_PROMPT_VERSION == "answer-v13"
    assert CHAT_REVIEW_PROMPT_VERSION == "chat-evidence-review-v7"
    assert "required_aspects" in ChatReviewOutput.model_fields
