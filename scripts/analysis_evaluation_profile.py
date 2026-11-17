"""Pure analysis provenance helpers; never infer a saved review from today's settings."""

from typing import Any

from app.core.config import Settings
from app.services.analysis_evidence import QUOTE_CATALOG_VERSION
from app.services.analysis_review import (
    ANALYSIS_REVIEW_PROMPT_VERSION,
    INSIGHT_VALUE_PROJECTION_VERSION,
)
from app.services.evidence_review import PROJECTION_VERSION
from app.services.insight_generation import INSIGHT_PROMPT_VERSION


def expected_review_profile(settings: Settings | None = None) -> dict[str, Any]:
    return {
        "prompt_version": ANALYSIS_REVIEW_PROMPT_VERSION,
        "model": (
            settings.openai_analysis_review_model if settings else Settings.ANALYSIS_REVIEW_MODEL
        ),
        "reasoning_effort": Settings.ANALYSIS_REVIEW_REASONING_EFFORT,
        "projection_version": PROJECTION_VERSION,
        "insight_value_projection_version": INSIGHT_VALUE_PROJECTION_VERSION,
        "insight_prompt_version": INSIGHT_PROMPT_VERSION,
        "quote_catalog_version": QUOTE_CATALOG_VERSION,
    }


def observed_review_profile(payload: dict[str, Any]) -> dict[str, Any]:
    review = payload.get("evidence_review")
    if not isinstance(review, dict):
        return {}
    return {key: review[key] for key in expected_review_profile() if key in review}


def review_stale_reasons(payload: dict[str, Any]) -> list[str]:
    observed = observed_review_profile(payload)
    return [
        f"analysis_review_{key}_missing"
        if key not in observed
        else f"analysis_review_{key}_changed"
        for key, value in expected_review_profile().items()
        if observed.get(key) != value
    ]
