"""Offline service-audit projection; reconstruct full catalog, never call AI."""

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import ChatTurn
from app.domain.conversation import Evidence
from app.services.chat_context import answer_quote_catalog
from app.services.chat_review import ChatReviewOutput, compact_chat_packet
from app.services.evidence_review import PROJECTION_VERSION, prepare_review
from app.services.grounding import AnswerOutput, validate_answer
from scripts.evaluate_ai import _atomic_write
from scripts.evaluate_review import fingerprint
from scripts.reproject_review import replay_case, saved_evidence


def _stored_audit_record(
    turn: ChatTurn, record: dict[str, Any], evidence: Sequence[Evidence]
) -> dict[str, Any]:
    """Rebuild the exact versioned wire packet before comparing its saved hash."""
    metadata = turn.context_snapshot["evidence_review"]
    if "review" not in metadata:
        raise ValueError("SAVED_TURN_HAS_NO_REVIEW")
    groups: dict[int, list[str]] = {}
    for claim in metadata["claims"]:
        groups.setdefault(claim["block_index"], []).append(claim["text"])
    draft = {"answer": [{"text": "".join(parts)} for _, parts in sorted(groups.items())]}
    packet = prepare_review(draft, answer_quote_catalog(evidence), question=record["question"])
    if metadata.get("prompt_version", "").startswith("chat-") and turn.retrieval_query is not None:
        content = json.loads(packet.content)
        content["retrieval_query"] = turn.retrieval_query
        packet = replace(
            packet, content=json.dumps(content, ensure_ascii=False, separators=(",", ":"))
        )
    if metadata.get("wire_version") == "source-index-v1":
        packet = compact_chat_packet(packet)
    if hashlib.sha256(packet.content.encode()).hexdigest() != metadata["request_sha256"]:
        raise ValueError("SAVED_REVIEW_CONTEXT_CHANGED")
    request = {
        "content": packet.content,
        "schema": ChatReviewOutput.model_json_schema()
        if "unanswered_parts" in metadata["review"]
        else {},
    }
    return {
        "draft": draft,
        "request": request,
        "request_sha256": fingerprint(request),
        "review": metadata["review"],
        "draft_evidence_status": metadata.get("draft_evidence_status", "answered"),
        "complete_scope": turn.context_snapshot["payload"]["complete_scope"],
    }


def run(source: Path, output: Path) -> dict[str, Any]:
    if output.exists() or source.resolve() == output.resolve():
        raise ValueError("REPLAY_OUTPUT_MUST_BE_NEW")
    state = json.loads(source.read_text())
    results = []
    engine = create_engine(get_settings().database_url, hide_parameters=True)
    try:
        with Session(engine) as database:
            database.execute(text("SET TRANSACTION READ ONLY"))
            for record in state["cases"]:
                turn_id = record["turn_id"]
                turn = database.get(ChatTurn, UUID(turn_id))
                if turn is None:
                    raise ValueError("SAVED_TURN_MISSING")
                evidence = saved_evidence(database, turn_id)
                metadata = turn.context_snapshot.get("evidence_review")
                if metadata:
                    # Source bindings may be compact/selected-only in metadata;
                    # the full saved catalog remains necessary for all guards.
                    replay_record = _stored_audit_record(turn, record, evidence)
                    result = replay_case(replay_record, evidence, force_qualification_quotes=False)
                    result["draft_status_assumption"] = (
                        "recorded original draft status: " + metadata["draft_evidence_status"]
                        if "draft_evidence_status" in metadata
                        else "answered: original draft status absent from service audit; "
                        "projection-only evidence, not a service rerun"
                    )
                    result["previous_dropped_claim_ids"] = metadata["dropped_claim_ids"]
                else:
                    if record["response"]["evidence_status"] != "insufficient":
                        raise ValueError("UNAUDITED_SUPPORTED_SOURCE_NOT_REPLAYABLE")
                    answer = validate_answer(
                        AnswerOutput(evidence_status="insufficient", blocks=[]), evidence
                    )
                    result = {
                        "answer": answer.answer,
                        "evidence_status": answer.evidence_status,
                        "citations": answer.citations,
                        "dropped_claim_ids": [],
                    }
                result.update(question=record["question"], source_turn_id=turn_id)
                results.append(result)
    finally:
        engine.dispose()
    artifact = {
        "source": str(source),
        "source_sha256": fingerprint(state),
        "projection": PROJECTION_VERSION,
        "paid_calls": 0,
        "evidence_type": "saved_service_audit_offline_projection",
        "cases": results,
    }
    _atomic_write(output, json.dumps(artifact, ensure_ascii=False, indent=2).encode())
    return {
        "artifact": str(output),
        "paid_calls": 0,
        "cases": [
            {
                key: r.get(key)
                for key in (
                    "question",
                    "evidence_status",
                    "dropped_claim_ids",
                    "appended_quote_ids",
                    "all_displayed_support_in_citations",
                )
            }
            for r in results
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
