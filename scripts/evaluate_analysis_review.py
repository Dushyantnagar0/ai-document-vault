"""Opt-in budgeted analysis replay on saved original chunks; no publication/embedding."""

import argparse
import json
import time
from collections.abc import Callable
from typing import Any, Literal
from uuid import UUID

from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, defer

from app.adapters.openai import OpenAIGenerationProvider
from app.core.config import get_settings
from app.db.models import DocumentChunk
from app.domain.documents import Chunk, ProcessingError, SourceSpan
from app.prompts.analysis import ANALYSIS_PROMPT_VERSION
from app.services.analysis import analyse_document
from app.services.analysis_evidence import QUOTE_CATALOG_VERSION
from app.services.evidence_review import PROJECTION_VERSION, REVIEW_PROMPT_VERSION
from scripts.evaluate_ai import _atomic_write, evaluation_lock, validate_run_id
from scripts.evaluate_quality import analysis_indicators
from scripts.evaluate_review import BUDGET_FILE, fingerprint
from scripts.live_budget import BudgetedGenerationProvider, RequestBudget

SOURCES = {
    "narrative": ("eval-quality-v1.json", "quality-request-guide"),
    "long": ("eval-core-v1-analysis-v12-20261116.json", "long"),
}


class RecordingProvider:
    """Retain experimental inputs/results before application validation can fail."""

    def __init__(
        self, delegate: BudgetedGenerationProvider, record: dict[str, Any], save: Callable[[], None]
    ) -> None:
        self.delegate, self.record, self.save = delegate, record, save

    def generate(self, **kwargs: Any) -> Any:
        request = {
            "schema": kwargs["schema"].__name__,
            "system": kwargs["system"],
            "content": kwargs["content"],
            "max_output_tokens": kwargs["max_output_tokens"],
            "status": "reserved",
        }
        self.record.setdefault("requests", []).append(request)
        self.save()
        result = self.delegate.generate(**kwargs)
        request.update(
            status="completed",
            result=result.result.model_dump(mode="json"),
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
        self.save()
        return result


def run(run_id: str, cases: list[str]) -> dict[str, Any]:
    validate_run_id(run_id)
    settings = get_settings()
    directory = settings.local_storage_dir / "demo"
    path = directory / f"analysis-review-{run_id}.json"
    budget = RequestBudget(directory / BUDGET_FILE, limit=100)
    recipe = {
        "analysis": ANALYSIS_PROMPT_VERSION,
        "review": REVIEW_PROMPT_VERSION,
        "projection": PROJECTION_VERSION,
        "quote_catalog": QUOTE_CATALOG_VERSION,
        "model": settings.openai_chat_model,
        "group_input": settings.analysis_group_max_tokens,
        "summary_output": settings.summary_max_output_tokens,
        "insight_output": settings.insight_max_output_tokens,
        "insight_visible": settings.insight_max_visible_tokens,
        "review_output": settings.evidence_review_max_output_tokens,
        "review_effort": settings.openai_review_reasoning_effort,
    }
    engine = create_engine(settings.database_url, hide_parameters=True)
    try:
        with evaluation_lock(directory):
            state = (
                json.loads(path.read_text()) if path.exists() else {"recipe": recipe, "cases": {}}
            )
            if state["recipe"] != recipe:
                raise ValueError("ANALYSIS_RECIPE_CHANGED")
            for key in cases:
                if key in state["cases"]:
                    continue
                filename, document_key = SOURCES[key]
                frozen = json.loads((directory / filename).read_text())
                identifier = UUID(frozen["documents"][document_key]["id"])
                with Session(engine) as database:
                    database.execute(text("SET TRANSACTION READ ONLY"))
                    rows = database.scalars(
                        select(DocumentChunk)
                        .where(DocumentChunk.document_id == identifier)
                        .options(defer(DocumentChunk.embedding))
                        .order_by(DocumentChunk.ordinal)
                    ).all()
                    chunks = [
                        Chunk(
                            row.id,
                            row.ordinal,
                            row.text,
                            row.token_count,
                            tuple(SourceSpan(**span) for span in row.source_spans),
                        )
                        for row in rows
                    ]
                if not chunks:
                    raise ValueError("SAVED_CHUNKS_MISSING")
                record: dict[str, Any] = {
                    "status": "reserved",
                    "document_id": str(identifier),
                    "input_chunks": len(chunks),
                    "source_sha256": fingerprint([(str(c.id), c.text) for c in chunks]),
                    "evaluation_type": "read_only_saved_chunk_analysis_replay",
                    "published": False,
                    "new_embeddings": False,
                }
                state["cases"][key] = record

                def save() -> None:
                    _atomic_write(path, json.dumps(state, ensure_ascii=False, indent=2).encode())

                save()
                deadline = time.monotonic() + settings.processing_deadline_seconds

                def before_call(*, call_deadline: float = deadline) -> float:
                    return call_deadline - time.monotonic()

                def provider(
                    phase: str,
                    effort: Literal["none", "low"] = "none",
                    *,
                    case_key: str = key,
                    case_record: dict[str, Any] = record,
                    save_record: Callable[[], None] = save,
                ) -> RecordingProvider:
                    return RecordingProvider(
                        BudgetedGenerationProvider(
                            OpenAIGenerationProvider(settings, reasoning_effort=effort),
                            budget,
                            f"analysis.{case_key}.{phase}",
                            profile=effort,
                        ),
                        case_record,
                        save_record,
                    )

                try:
                    result = analyse_document(
                        chunks,
                        settings,
                        provider("draft"),
                        before_call,
                        insight_generator=provider(
                            "insights", settings.openai_review_reasoning_effort
                        ),
                        reviewer=provider("review", settings.openai_review_reasoning_effort),
                    )
                    record.update(
                        status="completed",
                        analysis=result.payload,
                        all_chunks_participated=set(result.covered_chunk_ids)
                        == {c.id for c in chunks},
                        semantic_review="required",
                    )
                    if key == "narrative":
                        record["indicators"] = analysis_indicators(result.payload)
                except ProcessingError as error:
                    record.update(
                        status="failed",
                        error_code=error.code,
                        diagnostics=error.diagnostics.as_dict() if error.diagnostics else None,
                    )
                _atomic_write(path, json.dumps(state, ensure_ascii=False, indent=2).encode())
                print(
                    json.dumps({"case": key, "status": record["status"], **budget.snapshot()}),
                    flush=True,
                )
        return {"artifact": str(path), **budget.snapshot()}
    finally:
        engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cases", default="narrative,long")
    args = parser.parse_args()
    cases = args.cases.split(",")
    if not cases or any(c not in SOURCES for c in cases) or len(set(cases)) != len(cases):
        parser.error("Select unique narrative,long cases")
    print(json.dumps(run(args.run_id, cases), indent=2))


if __name__ == "__main__":
    main()
