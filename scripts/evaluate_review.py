"""Explicit paid review experiments on frozen synthetic evidence; no publication."""

import argparse
import hashlib
import json
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, defer

from app.adapters.openai import OpenAIGenerationProvider
from app.core.config import get_settings
from app.db.models import ChatTurn, Document, DocumentChunk, DocumentInsights
from app.domain.documents import ProcessingError
from app.services.analysis import prepare_analysis_review
from app.services.analysis_evidence import QuoteReference, build_quote_catalog
from app.services.analysis_review import (
    ANALYSIS_REVIEW_PROMPT_VERSION,
    ANALYSIS_REVIEW_SYSTEM,
    AnalysisReviewOutput,
    analysis_review_schema,
    project_analysis_review,
)
from app.services.chat_review import (
    CHAT_REVIEW_PROMPT_VERSION,
    CHAT_REVIEW_SYSTEM,
    ChatReviewOutput,
    compact_chat_packet,
    project_chat_review,
)
from app.services.evidence_review import prepare_review
from scripts.analysis_evaluation_profile import expected_review_profile
from scripts.evaluate_ai import _atomic_write, _supplied_source_text, evaluation_lock
from scripts.live_budget import BudgetError, RequestBudget

CASES = (
    "long",
    "narrative",
    "q3",
    "q4",
    "capacity-control",
    "inventory-control",
    "retention-control",
)
BUDGET_FILE = "quality-budget-100-20261116.json"


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def document_catalog(database: Session, identifier: str) -> dict[str, QuoteReference]:
    document = database.get(Document, UUID(identifier))
    if document is None or document.status != "COMPLETED":
        raise ValueError("FROZEN_DOCUMENT_UNAVAILABLE")
    chunks = database.scalars(
        select(DocumentChunk)
        .options(defer(DocumentChunk.embedding))
        .where(
            DocumentChunk.document_id == document.id,
            DocumentChunk.attempt_id == document.published_attempt_id,
        )
        .order_by(DocumentChunk.ordinal)
    ).all()
    return build_quote_catalog({chunk.id: (chunk.text,) for chunk in chunks})


def load_case(
    directory: Path, database: Session, key: str
) -> tuple[dict[str, Any], dict[str, QuoteReference], str | None]:
    if key.endswith("-control"):
        capacity = key == "capacity-control"
        original = (
            "The planning example has two workers. Each worker completes three exports per hour. "
            "Assume both workers remain available and exports have the stated routine workload."
            if capacity
            else "The inventory contains twelve components. Each kit requires three components. "
            "Assume no other shortages, wastage or reserved inventory."
        )
        summary = (
            "Two available workers each complete three routine exports per hour."
            if capacity
            else "There are twelve available components and each kit uses three components."
        )
        useful = (
            (
                "Under those planning assumptions, "
                "combined completion capacity is six exports per hour."
            )
            if capacity
            else "Under those assumptions, the available inventory can supply four complete kits."
        )
        duplicate = (
            "Each of the two workers completes three routine exports per hour."
            if capacity
            else "Each kit consumes three of the twelve available components."
        )
        if key == "retention-control":
            original = (
                "Closed records are retained for thirty days after closure. "
                "A legal hold prevents deletion until release. Releasing the hold does not "
                "create a new thirty-day clock; the ordinary closure-based rule still applies."
            )
            summary = (
                "Closed records are retained for thirty days after closure, "
                "and a legal hold prevents deletion until release."
            )
            useful = (
                "After a hold is released, the retention deadline remains closure-based "
                "rather than restarting thirty days from release."
            )
            duplicate = "A legal hold blocks deletion while it remains in force."
        return (
            {"summary": [{"text": summary}], "insights": [{"text": useful}, {"text": duplicate}]},
            build_quote_catalog({uuid5(NAMESPACE_URL, key): (original,)}),
            None,
        )
    filename = (
        "eval-core-v1-analysis-v12-20261116.json" if key == "long" else "eval-quality-v1.json"
    )
    state = json.loads((directory / filename).read_text())
    if key in {"long", "narrative"}:
        name = "long" if key == "long" else "quality-request-guide"
        identifier = state["documents"][name]["id"]
        analysis = database.get(DocumentInsights, UUID(identifier))
        if analysis is None:
            raise ValueError("FROZEN_ANALYSIS_UNAVAILABLE")
        return (
            {field: analysis.payload[field] for field in ("summary", "insights")},
            document_catalog(database, identifier),
            None,
        )
    case = state["cases"][key.upper()]
    turn = database.get(ChatTurn, UUID(case["response"]["turn_id"]))
    if turn is None:
        raise ValueError("FROZEN_TURN_UNAVAILABLE")
    catalog = build_quote_catalog(
        {
            UUID(item["source_id"]): (_supplied_source_text(item),)
            for item in turn.context_snapshot["payload"]["evidence"]
        }
    )
    return (
        {
            "answer": [
                {"text": re.sub(r"\s*\[\d+\]", "", paragraph).strip()}
                for paragraph in case["response"]["answer"].split("\n\n")
                if paragraph.strip()
            ]
        },
        catalog,
        case["question"],
    )


def run(cases: Sequence[str], effort: Literal["none", "low"], run_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,39}", run_id):
        raise ValueError("INVALID_RUN_ID")
    settings = get_settings()
    directory = settings.local_storage_dir / "demo"
    budget = RequestBudget(directory / BUDGET_FILE, limit=100)
    path = directory / f"review-experiment-{run_id}.json"
    with evaluation_lock(directory):
        state: dict[str, Any] = (
            json.loads(path.read_text()) if path.exists() else {"run_id": run_id, "cases": {}}
        )
        engine = create_engine(settings.database_url, hide_parameters=True)
        try:
            for key in cases:
                task_key = f"{key}:{effort}"
                with Session(engine) as database:
                    database.execute(text("SET TRANSACTION READ ONLY"))
                    draft, catalog, question = load_case(directory, database, key)
                packet = prepare_review(draft, catalog, question=question)
                is_chat = "answer" in packet.kinds
                if not is_chat and len(draft.get("insights", [])) > 3:
                    raise ValueError("FROZEN_INSIGHT_COUNT_EXCEEDS_CURRENT_REVIEW_LIMIT")
                if is_chat:
                    packet = compact_chat_packet(packet)
                else:
                    if effort != settings.ANALYSIS_REVIEW_REASONING_EFFORT:
                        raise ValueError("ANALYSIS_REVIEW_EFFORT_MISMATCH")
                    packet = prepare_analysis_review(draft, catalog)
                model = (
                    settings.openai_chat_model if is_chat else settings.openai_analysis_review_model
                )
                schema = ChatReviewOutput if is_chat else analysis_review_schema(packet)
                system = CHAT_REVIEW_SYSTEM if is_chat else ANALYSIS_REVIEW_SYSTEM
                prompt_version = (
                    CHAT_REVIEW_PROMPT_VERSION if is_chat else ANALYSIS_REVIEW_PROMPT_VERSION
                )
                request = {
                    "schema": schema.model_json_schema(),
                    "system": system,
                    "prompt_version": prompt_version,
                    "content": packet.content,
                    "max_output_tokens": 4096,
                    "reasoning_effort": effort,
                    "model": model,
                }
                if not is_chat:
                    request["review_profile"] = expected_review_profile(settings)
                digest = fingerprint(request)
                if task_key in state["cases"]:
                    if state["cases"][task_key]["request_sha256"] != digest:
                        raise ValueError("EXPERIMENT_RECIPE_CHANGED")
                    continue
                reservation = budget.reserve(f"review.{key}.{effort}", digest)
                record: dict[str, Any] = {
                    "request_sha256": digest,
                    "reservation_id": reservation,
                    "request": request,
                    "draft": draft,
                    "status": "reserved",
                }
                state["cases"][task_key] = record
                _atomic_write(path, json.dumps(state, ensure_ascii=False, indent=2).encode())
                started = time.monotonic()
                try:
                    response = OpenAIGenerationProvider(
                        settings, model=model, reasoning_effort=effort
                    ).generate(
                        schema=schema,
                        system=system,
                        content=packet.content,
                        max_output_tokens=4096,
                        timeout_seconds=settings.openai_generation_timeout_seconds,
                    )
                    record.update(
                        review=response.result.model_dump(mode="json"),
                        model=response.model,
                        input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens,
                    )
                    projection = (
                        project_chat_review(packet, response.result)
                        if isinstance(response.result, ChatReviewOutput)
                        else project_analysis_review(
                            packet, AnalysisReviewOutput.model_validate(response.result)
                        )
                    )
                    record.update(
                        status="completed",
                        payload=projection.payload,
                        scope_complete=projection.scope_complete,
                        dropped_claim_ids=projection.dropped_claim_ids,
                        appended_quote_ids=projection.appended_quote_ids,
                    )
                    if key.endswith("-control"):
                        kept = [b["text"] for b in projection.payload.get("insights", [])]
                        record["control_checks"] = {
                            "useful_inference_retained": draft["insights"][0]["text"] in kept,
                            "repeated_fact_removed": draft["insights"][1]["text"] not in kept,
                        }
                    elif key == "narrative":
                        record["known_duplicate_claims_removed"] = all(
                            identifier in projection.dropped_claim_ids
                            for identifier in ("i1.1", "i4.1")
                        )
                    elif key == "q3":
                        projected_text = " ".join(
                            b["text"] for b in projection.payload.get("answer", [])
                        ).lower()
                        record["hold_release_qualification_present"] = all(
                            word in projected_text
                            for word in ("release", "hold", "universal", "clock")
                        )
                    budget.complete(
                        reservation,
                        {
                            "outcome": "succeeded",
                            "model": response.model,
                            "input_tokens": response.input_tokens,
                            "output_tokens": response.output_tokens,
                        },
                    )
                except ProcessingError as error:
                    details = error.diagnostics.as_dict() if error.diagnostics else {}
                    record.update(status="failed", error_code=error.code, diagnostics=details)
                    budget.complete(
                        reservation,
                        {
                            "outcome": "failed",
                            "error_code": error.code,
                            "input_tokens": details.get("input_tokens"),
                            "output_tokens": details.get("output_tokens"),
                        },
                    )
                record["duration_ms"] = (time.monotonic() - started) * 1000
                _atomic_write(path, json.dumps(state, ensure_ascii=False, indent=2).encode())
                print(
                    json.dumps({"case": task_key, "status": record["status"], **budget.snapshot()}),
                    flush=True,
                )
        finally:
            engine.dispose()
        return {
            "artifact": str(path),
            "cases": {k: v["status"] for k, v in state["cases"].items()},
            **budget.snapshot(),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", required=True, action="store_true")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--effort", choices=("none", "low"), required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    cases = args.cases.split(",")
    if any(case not in CASES for case in cases) or len(cases) != len(set(cases)):
        parser.error("Select unique known frozen cases.")
    try:
        print(json.dumps(run(cases, args.effort, args.run_id)))
    except (BudgetError, ValueError):
        print(json.dumps({"error_code": "REVIEW_EXPERIMENT_BLOCKED"}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
