"""Cross-stage prompt budgets preserve accepted input and original provenance.

All generation is deterministic; these checks do not establish semantic quality.
"""

import json
from typing import Any
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.domain.documents import Chunk, Generated, ProcessingError
from app.prompts.analysis import generation_prompt_tokens
from app.services.analysis import (
    _content,
    _Node,
    _tokens,
    analyse_document,
    prepare_analysis_review,
    quality_input_limit,
    quality_input_reservations,
)
from app.services.analysis_evidence import QuoteReference, SelectionOutput, build_quote_catalog
from app.services.analysis_review import (
    ANALYSIS_REVIEW_REPAIR_VARIANTS,
    ANALYSIS_REVIEW_SYSTEM,
    AnalysisReviewOutput,
    analysis_review_schema,
)
from app.services.insight_generation import INSIGHT_SYSTEM, InsightCandidates, candidate_content


def settings(**values: Any) -> Settings:
    return Settings(_env_file=None, openai_api_key="unit-placeholder", **values)


class LiteralProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> Generated[Any]:
        self.calls.append(kwargs)
        request = json.loads(kwargs["content"])
        first = request["evidence"][0]
        if kwargs["schema"] is InsightCandidates:
            result: dict[str, Any] = {"insights": []}
        elif issubclass(kwargs["schema"], AnalysisReviewOutput):
            result = {
                "decisions": [
                    {
                        "claim_id": claim["claim_id"],
                        "supported": True,
                        "quote_ids": [first["quote_id"]],
                        "redundant_with": [],
                    }
                    for claim in request["claims"]
                ],
                "required_qualifications": [],
                "scope_complete": True,
                "insight_assessments": [],
            }
        else:
            result = {
                "summary": [{"text": first["text"].strip(), "quote_ids": [first["quote_id"]]}],
                "insights": [],
            }
        model = (
            Settings.ANALYSIS_REVIEW_MODEL
            if issubclass(kwargs["schema"], AnalysisReviewOutput)
            else Settings.CHAT_MODEL
        )
        return Generated(kwargs["schema"].model_validate(result), model, 100, 100)


def short_passages(count: int) -> list[Chunk]:
    return [
        Chunk(
            uuid4(),
            ordinal,
            " ".join(
                f"Valve {index + 1} is red." for index in range(start, min(start + 100, count))
            ),
            600,
            (),
        )
        for ordinal, start in enumerate(range(0, count, 100))
    ]


def test_400_short_passages_reduce_before_final_draft_and_complete_quality_stages() -> None:
    configured = settings()
    chunks = short_passages(400)
    nodes = [_Node({}, (chunk.id,), {chunk.id: (chunk.text,)}) for chunk in chunks]
    catalog = build_quote_catalog({chunk.id: (chunk.text,) for chunk in chunks})
    # The synthesis packet fits, but its complete downstream allowance does not.
    assert _tokens(_content("final", nodes)) < configured.analysis_group_max_tokens
    assert quality_input_reservations(catalog, configured)[1] > quality_input_limit(
        configured, configured.evidence_review_max_output_tokens
    )
    provider = LiteralProvider()
    result = analyse_document(
        chunks, configured, provider, lambda: 30, insight_generator=provider, reviewer=provider
    )
    assert result.covered_chunk_ids == tuple(chunk.id for chunk in chunks)
    assert result.payload["analysis_coverage"] == 1.0
    history = result.payload["usage"]["call_history"]
    assert history[0]["stage"] == "section"
    assert [entry["stage"] for entry in history[-3:]] == ["final", "insights", "evidence_review"]
    final_evidence = json.loads(provider.calls[-3]["content"])["evidence"]
    for call in provider.calls[-2:]:
        request = json.loads(call["content"])
        assert [item["text"] for item in request["evidence"]] == [
            item["text"] for item in final_evidence
        ]
        assert generation_prompt_tokens(call["schema"], call["system"], call["content"]) <= (
            quality_input_limit(configured, call["max_output_tokens"])
        )
    bindings = result.payload["evidence_review"]["quote_bindings"]
    assert bindings and all(item["source_id"] == str(chunks[0].id) for item in bindings.values())


@pytest.mark.parametrize("spare_tokens", [0, -1])
def test_final_path_reserves_complete_review_at_exact_input_boundary(
    monkeypatch: pytest.MonkeyPatch, spare_tokens: int
) -> None:
    configured = settings()
    chunks = short_passages(200)
    catalog = build_quote_catalog({chunk.id: (chunk.text,) for chunk in chunks})
    reserved = quality_input_reservations(catalog, configured)[1]
    monkeypatch.setattr(Settings, "CHAT_MODEL_MAX_INPUT_TOKENS", reserved + spare_tokens)
    provider = LiteralProvider()
    result = analyse_document(
        chunks, configured, provider, lambda: 30, insight_generator=provider, reviewer=provider
    )
    stages = [entry["stage"] for entry in result.payload["usage"]["call_history"]]
    assert stages[0] == ("final" if spare_tokens == 0 else "section")
    assert result.covered_chunk_ids == tuple(chunk.id for chunk in chunks)


def test_reservation_includes_maximum_claim_frames_generated_text_and_repair() -> None:
    configured = settings()
    catalog = {"q1": QuoteReference(uuid4(), "A valve remains red.")}
    # Fill both generated JSON allowances, with 49 summary and 15 insight sentences
    # across all three permitted candidates.
    # Quotes and backslashes also exercise escaped text in the reserialized claims.
    draft: dict[str, Any] = {"summary": [], "insights": []}
    for kind, count, sentences, budget in (
        ("summary", 7, 7, configured.summary_max_output_tokens),
        ("insights", 3, 5, configured.insight_max_visible_tokens),
    ):
        words = 0
        while True:
            blocks = [
                {
                    "text": ('A valve says "red\\blue"' + " qualified" * words + ". ") * sentences,
                    "quote_ids": ["q1"],
                }
                for _ in range(count)
            ]
            selection = (
                SelectionOutput.model_validate({"summary": blocks, "insights": []})
                if kind == "summary"
                else InsightCandidates.model_validate({"insights": blocks})
            )
            if _tokens(selection.model_dump_json()) > budget:
                break
            draft[kind] = blocks
            words += 1
        assert words > 1
    packet = prepare_analysis_review(draft, catalog)
    insight_reserve, review_reserve = quality_input_reservations(catalog, configured)
    assert len(packet.claims) == 64
    for repair in ANALYSIS_REVIEW_REPAIR_VARIANTS:
        assert review_reserve >= generation_prompt_tokens(
            analysis_review_schema(packet), ANALYSIS_REVIEW_SYSTEM + repair, packet.content
        )
    assert insight_reserve >= generation_prompt_tokens(
        InsightCandidates, INSIGHT_SYSTEM, candidate_content(draft["summary"], catalog)
    )
    assert packet.catalog == catalog
    assert json.loads(packet.content)["source_index"] == {"s1": str(catalog["q1"].source_id)}


def test_maximum_candidate_assessments_fit_representative_review_output_budget() -> None:
    # All 64 claim decisions and all three assessments fit for this representative
    # one-passage-per-claim response. This is not a worst-case bound for every
    # possible eight-reference decision or the provider's reasoning-token usage.
    review = AnalysisReviewOutput.model_validate(
        {
            "required_qualifications": [],
            "decisions": [
                {
                    "claim_id": f"s1.{index + 1}",
                    "supported": True,
                    "quote_ids": [f"q{index + 1}"],
                    "redundant_with": [],
                }
                for index in range(61)
            ]
            + [
                {
                    "claim_id": f"i{index + 1}.1",
                    "supported": True,
                    "quote_ids": [f"q{number + 1}" for number in range(8)],
                    "redundant_with": [],
                }
                for index in range(3)
            ],
            "scope_complete": True,
            "insight_assessments": [
                {
                    "block_index": index,
                    "classification": "adds_conclusion",
                    "conclusion_claim_ids": [f"i{index + 1}.1"],
                    "premise_quote_ids": [f"q{number + 1}" for number in range(8)],
                }
                for index in range(3)
            ],
        }
    )
    assert _tokens(review.model_dump_json()) < settings().evidence_review_max_output_tokens


def test_unusable_quality_reservations_fail_before_any_provider_reservation() -> None:
    provider = LiteralProvider()
    reservations: list[bool] = []

    def reserve() -> float:
        reservations.append(True)
        return 30

    with pytest.raises(ProcessingError) as caught:
        analyse_document(
            short_passages(1),
            settings(analysis_group_max_tokens=1800),
            provider,
            reserve,
            insight_generator=provider,
            reviewer=provider,
        )
    assert caught.value.code == "analysis_prompt_limit"
    assert provider.calls == reservations == []
