"""Apply current deterministic projection to saved frozen chat audits: zero AI calls."""

import argparse
import json
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.services.evidence_review import PROJECTION_VERSION
from app.services.grounding import AnswerOutput, validate_answer
from scripts.evaluate_ai import _atomic_write, automatic_checks
from scripts.evaluate_review import fingerprint
from scripts.evaluation_fixtures import EvaluationCase
from scripts.reproject_review import replay_case, saved_evidence


def audit_record(record: dict[str, Any]) -> dict[str, Any] | None:
    requests = record.get("requests", [])
    reviews = [r for r in requests if r["phase"] == "review" and r["status"] == "completed"]
    if not reviews:
        return None
    if len(reviews) != 1:
        raise ValueError("AMBIGUOUS_SAVED_REVIEW")
    drafts = [r for r in requests if r["phase"] == "draft" and r["status"] == "completed"]
    if len(drafts) != 1 or reviews[0]["result"] != record["audit"]["review"]:
        raise ValueError("SAVED_REVIEW_BINDING_CHANGED")
    request = reviews[0]
    draft = drafts[0]["result"]
    return {
        "request": request,
        "request_sha256": fingerprint(request),
        "draft": {"answer": [{"text": b["text"]} for b in draft["blocks"]]},
        "draft_evidence_status": draft["evidence_status"],
        "complete_scope": record["frozen_payload"]["complete_scope"],
        "review": request["result"],
    }


def run(source: Path, output: Path) -> dict[str, Any]:
    if output.exists() or output.resolve() == source.resolve():
        raise ValueError("REPLAY_OUTPUT_MUST_BE_NEW")
    state = json.loads(source.read_text())
    engine = create_engine(get_settings().database_url, hide_parameters=True)
    records: dict[str, Any] = {}
    try:
        with Session(engine) as database:
            database.execute(text("SET TRANSACTION READ ONLY"))
            for key, record in state["cases"].items():
                if record["status"] != "completed":
                    records[key] = {"status": "skipped", "reason": "INCOMPLETE_SOURCE_CASE"}
                    continue
                evidence = saved_evidence(database, record["source_turn_id"])
                audit = audit_record(record)
                if audit is not None:
                    result = replay_case(audit, evidence, force_qualification_quotes=False)
                else:
                    answer = validate_answer(
                        AnswerOutput.model_validate(
                            {
                                "evidence_status": record["response"]["evidence_status"],
                                "blocks": record["retained_blocks"],
                            }
                        ),
                        evidence,
                        complete_scope=record["frozen_payload"]["complete_scope"],
                    )
                    result = {
                        "answer": answer.answer,
                        "evidence_status": answer.evidence_status,
                        "citations": answer.citations,
                        "retained_blocks": [b.model_dump(mode="json") for b in answer.blocks],
                        "audit_skipped": record["audit"]["audit_skipped"],
                    }
                expectations = record["expectations"]
                case = EvaluationCase(
                    key,
                    "frozen",
                    record["question"],
                    tuple(expectations["patterns"]),
                    status=expectations["status"],
                    forbidden=tuple(expectations["forbidden"]),
                )
                allowed = {str(e.document_id) for e in evidence}
                snapshot = {
                    "membership_valid": all(
                        c["document_id"] in allowed for c in result["citations"]
                    ),
                    "selected_membership_valid": len(allowed) <= 1,
                    "supplied_text": "\n".join(e.text for e in evidence),
                }
                result.update(
                    status="completed",
                    checks=automatic_checks(case, result, snapshot),
                    before_words=len(record["response"]["answer"].split()),
                    after_words=len(result["answer"].split()),
                    semantic_review="required",
                )
                records[key] = result
    finally:
        engine.dispose()
    result = {
        "source": str(source),
        "source_sha256": fingerprint(state),
        "projection_version": PROJECTION_VERSION,
        "qualification_policy": "retained_claim_source_bindings",
        "paid_calls": 0,
        "evaluation_type": "frozen_evidence_offline_reprojection",
        "full_api_e2e": False,
        "cases": records,
    }
    _atomic_write(output, json.dumps(result, ensure_ascii=False, indent=2).encode())
    return {
        "artifact": str(output),
        "paid_calls": 0,
        "automatic_pass": sum(
            bool(r.get("checks", {}).get("automatic_pass")) for r in records.values()
        ),
        "cases": {
            key: {
                name: r.get(name)
                for name in ("evidence_status", "before_words", "after_words", "appended_quote_ids")
            }
            for key, r in records.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
