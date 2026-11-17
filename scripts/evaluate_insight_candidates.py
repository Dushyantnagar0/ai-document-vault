"""Budgeted focused insight generation/audit on frozen and held-out sources."""

import argparse
import json
import re
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.adapters.openai import OpenAIGenerationProvider
from app.core.config import get_settings
from app.db.models import DocumentInsights
from app.domain.documents import ProcessingError
from app.services.analysis import prepare_analysis_review
from app.services.analysis_evidence import QuoteReference, build_quote_catalog
from app.services.analysis_review import (
    ANALYSIS_REVIEW_PROMPT_VERSION,
    ANALYSIS_REVIEW_SYSTEM,
    INSIGHT_VALUE_PROJECTION_VERSION,
    analysis_review_schema,
    project_analysis_review,
)
from app.services.insight_generation import INSIGHT_PROMPT_VERSION, generate_candidates
from scripts.analysis_evaluation_profile import expected_review_profile
from scripts.evaluate_ai import _atomic_write, evaluation_lock
from scripts.evaluate_review import BUDGET_FILE, document_catalog, fingerprint, load_case
from scripts.live_budget import BudgetedGenerationProvider, RequestBudget

CASES = ("narrative", "held-out", "sparse", "long")


def fixture(
    directory: Path, database: Session, key: str
) -> tuple[list[dict[str, Any]], dict[str, QuoteReference]]:
    if key in {"narrative", "long"}:
        draft, catalog, _ = load_case(directory, database, key)
        return draft["summary"], catalog
    if key == "sparse":
        state = json.loads((directory / "eval-core-v1-prompts-v11-20261116.json").read_text())
        identifier = state["documents"]["juniper"]["id"]
        analysis = database.get(DocumentInsights, UUID(identifier))
        if analysis is None:
            raise ValueError("FROZEN_ANALYSIS_UNAVAILABLE")
        return analysis.payload["summary"], document_catalog(database, identifier)
    # Independently authored before generation; no generated answer defines these facts.
    sources = (
        "In the planning example, two packing stations each process four parcels per hour. "
        "Assume both stations remain staffed and parcels have the stated routine workload.",
        "Fragile parcels require inspection before packing. One inspector can inspect three "
        "fragile parcels per hour under the example's assumptions. Standard parcels bypass "
        "inspection. Both kinds share the packing stations.",
        "The carrier collects packed parcels at 16:00 UTC on weekdays. A printed label only "
        "confirms that intake recorded the parcel; it does not mean inspection, packing or "
        "carrier collection has finished.",
        "Label acknowledgement has a two-minute target during staffed hours. This is not a "
        "same-day collection promise. Actual workload or interruptions can change throughput.",
    )
    catalog = build_quote_catalog(
        {
            uuid5(NAMESPACE_URL, f"held-out-fulfilment:{i}"): (value,)
            for i, value in enumerate(sources)
        }
    )
    return [
        {
            "text": (
                "The example has two staffed packing stations, each rated at four parcels per hour."
            )
        },
        {
            "text": "Fragile parcels need inspection by one inspector rated at three per hour; "
            "standard parcels bypass inspection, and both kinds share packing stations."
        },
        {
            "text": "Weekday carrier collection is at 16:00 UTC. A printed label records intake, "
            "not completed inspection, packing or collection. Its two-minute staffed-hours "
            "acknowledgement target is not a same-day collection promise. Throughput varies "
            "with workload and interruptions."
        },
    ], catalog


def run(
    keys: list[str], run_id: str, *, audit_only: bool = False, audit_revision: int = 1
) -> dict[str, Any]:
    if audit_revision < 1 or (audit_only and audit_revision <= 1):
        raise ValueError("INVALID_AUDIT_REVISION")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,39}", run_id):
        raise ValueError("INVALID_RUN_ID")
    settings = get_settings()
    directory = settings.local_storage_dir / "demo"
    path = directory / f"insight-experiment-{run_id}.json"
    budget = RequestBudget(directory / BUDGET_FILE, limit=100)
    generation_profile = {
        "model": settings.openai_chat_model,
        "reasoning_effort": "low",
        "prompt_version": INSIGHT_PROMPT_VERSION,
    }
    with evaluation_lock(directory):
        state = json.loads(path.read_text()) if path.exists() else {"cases": {}}
        engine = create_engine(settings.database_url, hide_parameters=True)
        try:
            for key in keys:
                with Session(engine) as database:
                    database.execute(text("SET TRANSACTION READ ONLY"))
                    summary, catalog = fixture(directory, database, key)
                binding = fingerprint(
                    {
                        "summary": summary,
                        "evidence": {
                            q: {"source_id": str(ref.source_id), "text": ref.quote}
                            for q, ref in catalog.items()
                        },
                    }
                )
                case_key = f"{key}:audit-{audit_revision}" if audit_only else key
                original_case = state["cases"].get(key)
                if audit_only and (not original_case or "generation" not in original_case):
                    raise ValueError("SAVED_CANDIDATES_REQUIRED")
                existing = state["cases"].get(case_key)
                if existing:
                    if existing["binding"] != binding:
                        raise ValueError("FROZEN_INPUT_CHANGED")
                    if existing.get("review_profile") != expected_review_profile(settings):
                        raise ValueError("INSIGHT_REVIEW_RECIPE_CHANGED")
                    if not audit_only and existing.get("generation_profile") != generation_profile:
                        raise ValueError("INSIGHT_GENERATION_RECIPE_CHANGED")
                    continue
                record: dict[str, Any] = {
                    "binding": binding,
                    "summary": summary,
                    "status": "started",
                    "review_prompt_version": ANALYSIS_REVIEW_PROMPT_VERSION,
                    "review_profile": expected_review_profile(settings),
                    "insight_projection_version": INSIGHT_VALUE_PROJECTION_VERSION,
                    "catalog": {
                        q: {"source_id": str(ref.source_id), "quote": ref.quote}
                        for q, ref in catalog.items()
                    },
                }
                if audit_only:
                    record["generation"] = original_case["generation"]
                    record["generation_source"] = key
                    record["generation_new_calls"] = 0
                    record["generation_profile"] = original_case.get("generation_profile")
                else:
                    record["generation_profile"] = generation_profile
                state["cases"][case_key] = record
                _atomic_write(path, json.dumps(state, ensure_ascii=False, indent=2).encode())
                try:
                    if not audit_only:
                        provider = BudgetedGenerationProvider(
                            OpenAIGenerationProvider(settings, reasoning_effort="low"),
                            budget,
                            f"insightgen.{key}",
                            profile="low",
                        )
                        generated = generate_candidates(
                            summary,
                            catalog,
                            provider,
                            lambda: settings.openai_generation_timeout_seconds,
                        )
                        record["generation"] = generated.result.model_dump(mode="json")
                        record["generation_model"] = generated.model
                        record["generation_usage"] = [
                            generated.input_tokens,
                            generated.output_tokens,
                        ]
                        _atomic_write(
                            path, json.dumps(state, ensure_ascii=False, indent=2).encode()
                        )
                    packet = prepare_analysis_review(
                        {"summary": summary, "insights": record["generation"]["insights"]}, catalog
                    )
                    audit_provider = BudgetedGenerationProvider(
                        OpenAIGenerationProvider(
                            settings,
                            model=settings.openai_analysis_review_model,
                            reasoning_effort=settings.ANALYSIS_REVIEW_REASONING_EFFORT,
                        ),
                        budget,
                        f"insightaudit.{key}",
                        profile=(
                            f"{settings.openai_analysis_review_model}:"
                            f"{settings.ANALYSIS_REVIEW_REASONING_EFFORT}"
                        ),
                    )
                    audited = audit_provider.generate(
                        schema=analysis_review_schema(packet),
                        system=ANALYSIS_REVIEW_SYSTEM,
                        content=packet.content,
                        max_output_tokens=4096,
                        timeout_seconds=settings.openai_generation_timeout_seconds,
                    )
                    projection = project_analysis_review(packet, audited.result)
                    record.update(
                        status="completed",
                        review=audited.result.model_dump(mode="json"),
                        review_model=audited.model,
                        payload=projection.payload,
                        scope_complete=projection.scope_complete,
                        dropped_claim_ids=projection.dropped_claim_ids,
                        audit_usage=[audited.input_tokens, audited.output_tokens],
                    )
                except ProcessingError as error:
                    record.update(
                        status="failed",
                        error_code=error.code,
                        diagnostics=error.diagnostics.as_dict() if error.diagnostics else None,
                    )
                _atomic_write(path, json.dumps(state, ensure_ascii=False, indent=2).encode())
                print(
                    json.dumps({"case": case_key, "status": record["status"], **budget.snapshot()}),
                    flush=True,
                )
        finally:
            engine.dispose()
        return {"artifact": str(path), **budget.snapshot()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", required=True, action="store_true")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--audit-revision", type=int, default=1)
    args = parser.parse_args()
    keys = args.cases.split(",")
    if any(key not in CASES for key in keys) or len(keys) != len(set(keys)):
        parser.error("Select unique known cases.")
    print(
        json.dumps(
            run(keys, args.run_id, audit_only=args.audit_only, audit_revision=args.audit_revision)
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
