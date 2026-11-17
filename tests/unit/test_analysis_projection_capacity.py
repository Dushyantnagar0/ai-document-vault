"""Bounded final analysis capacity after corrected source selections split drafts.

Synthetic review decisions exercise projection, not semantic source entailment.
"""

from typing import Any
from uuid import UUID

import pytest
from pydantic import ValidationError

from app.api.documents import SavedAnalysis
from app.services.analysis import AnalysisOutput
from app.services.analysis_evidence import QuoteReference, SelectionOutput
from app.services.evidence_review import ReviewOutput, ReviewPacket, prepare_review, project_review


def reviewed_draft(*, claims_per_block: int = 2) -> tuple[ReviewPacket, dict[str, Any]]:
    catalog = {
        f"q{index}": QuoteReference(UUID(int=index), f"Original evidence passage {index}.")
        for index in range(1, 8 * claims_per_block * 5 + 2)
    }
    # Eight valid generated blocks can need more than eight references each after
    # independent review corrects individual claims' original support selections.
    draft = {
        "summary": [
            {
                "text": " ".join(
                    f"Supported statement {block}-{claim}." for claim in range(claims_per_block)
                ),
                "quote_ids": ["q1"],
            }
            for block in range(8)
        ],
        "insights": [],
    }
    SelectionOutput.model_validate(draft)
    packet = prepare_review(draft, catalog)
    raw = {
        "decisions": [
            {
                "claim_id": claim.id,
                "supported": True,
                "quote_ids": [f"q{index * 5 + offset}" for offset in range(1, 6)],
                "redundant_with": [],
            }
            for index, claim in enumerate(packet.claims)
        ],
        "required_qualifications": [],
        "scope_complete": True,
    }
    return packet, raw


def test_eight_drafts_retain_all_ordered_claims_after_review_citation_splits() -> None:
    packet, raw = reviewed_draft()
    projected = project_review(packet, ReviewOutput.model_validate(raw))
    final = AnalysisOutput.model_validate(projected.payload)
    public = SavedAnalysis.model_validate(final.model_dump(mode="json"))
    assert len(public.summary) == 16
    assert "".join(block.text for block in final.summary) == "".join(
        claim.text for claim in packet.claims
    )
    assert projected.dropped_claim_ids == ()
    assert projected.scope_complete
    for block, decision in zip(final.summary, raw["decisions"], strict=True):
        assert len(block.text) <= 4000
        assert len(block.source_ids) == len(block.support_quotes) == 5
        assert [(quote.source_id, quote.quote) for quote in block.support_quotes] == [
            (packet.catalog[key].source_id, packet.catalog[key].quote)
            for key in decision["quote_ids"]
        ]


def test_final_cap_still_drops_overflow_and_discloses_partial_scope() -> None:
    packet, raw = reviewed_draft(claims_per_block=3)
    projected = project_review(packet, ReviewOutput.model_validate(raw))
    final = AnalysisOutput.model_validate(projected.payload)
    assert len(final.summary) == 16
    assert projected.dropped_claim_ids == tuple(claim.id for claim in packet.claims[16:])
    assert not projected.scope_complete
    assert "".join(block.text for block in final.summary) == "".join(
        claim.text for claim in packet.claims[:16]
    )
    with pytest.raises(ValidationError):
        AnalysisOutput.model_validate(
            {"summary": [*final.summary, final.summary[-1]], "insights": []}
        )


def test_qualification_supplements_share_the_final_summary_capacity() -> None:
    packet, raw = reviewed_draft()
    # Removing one unsupported final claim leaves exactly one supplement slot.
    raw["decisions"][-1].update(supported=False, quote_ids=[])
    raw["required_qualifications"] = [
        {"quote_ids": ["q80"], "conveyed_by_claim_ids": []},
        {"quote_ids": ["q81"], "conveyed_by_claim_ids": []},
    ]
    projected = project_review(packet, ReviewOutput.model_validate(raw))
    final = AnalysisOutput.model_validate(projected.payload)
    assert len(final.summary) == 16
    assert final.summary[-1].text == packet.catalog["q80"].quote
    assert projected.appended_quote_ids == ("q80",)
    assert not projected.scope_complete


def test_generation_and_per_block_reference_limits_remain_unchanged() -> None:
    with pytest.raises(ValidationError):
        SelectionOutput.model_validate(
            {"summary": [{"text": "A fact.", "quote_ids": ["q1"]}] * 9, "insights": []}
        )
    packet, raw = reviewed_draft()
    raw["decisions"][0]["quote_ids"] = [f"q{index}" for index in range(1, 10)]
    with pytest.raises(ValidationError):
        ReviewOutput.model_validate(raw)
