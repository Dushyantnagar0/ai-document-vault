"""Replay saved review verdicts without provider calls or business-state writes."""

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast
from uuid import UUID

from pydantic import Field
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, defer

from app.core.config import get_settings
from app.db.models import ChatTurn, DocumentChunk
from app.domain.conversation import Evidence
from app.domain.documents import SourceSpan
from app.services.analysis_evidence import QuoteReference
from app.services.chat_review import ChatReviewOutput, compact_chat_packet, project_chat_review
from app.services.evidence_review import (
    PROJECTION_VERSION,
    ReviewOutput,
    ReviewPacket,
    prepare_review,
    project_review,
)
from app.services.grounding import AnswerOutput, append_evidence_gaps, validate_answer
from scripts.evaluate_ai import _atomic_write, _supplied_source_text
from scripts.evaluate_review import fingerprint


class _LegacyChatReviewOutput(ReviewOutput):
    """Strict frozen v1/v2 contract, used only to replay historical records."""

    answerability: Literal["answered", "partial", "insufficient"]
    relevant_claim_ids: list[str] = Field(max_length=64)


def _review_schema_properties(record: dict[str, Any]) -> dict[str, Any]:
    schema = record["request"].get("schema", {})
    if not isinstance(schema, dict):
        raise ValueError("INVALID_SAVED_REVIEW_SCHEMA")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise ValueError("INVALID_SAVED_REVIEW_SCHEMA")
    return cast(dict[str, Any], properties)


def _chat_review_for_replay(record: dict[str, Any]) -> tuple[ChatReviewOutput, bool]:
    properties = _review_schema_properties(record)
    if "unanswered_parts" in record["review"] or "unanswered_parts" in properties:
        return ChatReviewOutput.model_validate(record["review"]), True
    legacy = _LegacyChatReviewOutput.model_validate(record["review"])
    # Historical partial verdicts did not record a missing-component label.
    # Reuse their strictly parsed typed selections for offline projection only;
    # never invent a label or weaken the current provider validation contract.
    return ChatReviewOutput.model_construct(
        required_qualifications=legacy.required_qualifications,
        decisions=legacy.decisions,
        scope_complete=legacy.scope_complete,
        answerability=legacy.answerability,
        relevant_claim_ids=legacy.relevant_claim_ids,
        unanswered_parts=[],
    ), False


def restore_packet(record: dict[str, Any]) -> ReviewPacket:
    """Bind replay to the saved request, draft and exact selected source catalog."""
    request = record["request"]
    if fingerprint(request) != record["request_sha256"]:
        raise ValueError("SAVED_REQUEST_CHANGED")
    content = json.loads(request["content"])
    items = content["evidence"]
    if not isinstance(items, list):
        raise ValueError("INVALID_SAVED_EVIDENCE")
    if any(
        not isinstance(item, dict)
        or not isinstance(item.get("source_id"), str)
        or not isinstance(item.get("quote_id"), str)
        or not isinstance(item.get("text"), str)
        for item in items
    ):
        raise ValueError("INVALID_SAVED_EVIDENCE")
    source_index: dict[str, UUID] | None = None
    if "source_index" in content:
        raw_index = content["source_index"]
        if not isinstance(raw_index, dict):
            raise ValueError("INVALID_SAVED_SOURCE_INDEX")
        source_index = {}
        for alias, source in raw_index.items():
            if not isinstance(alias, str) or not isinstance(source, str):
                raise ValueError("INVALID_SAVED_SOURCE_INDEX")
            try:
                identifier = UUID(source)
            except ValueError:
                raise ValueError("INVALID_SAVED_SOURCE_INDEX") from None
            if str(identifier) != source or identifier in source_index.values():
                raise ValueError("INVALID_SAVED_SOURCE_INDEX")
            source_index[alias] = identifier
        if any(item.get("source_id") not in source_index for item in items):
            # A compact packet never interprets an unknown alias as a raw UUID.
            raise ValueError("UNKNOWN_SAVED_SOURCE_ALIAS")
        if set(source_index) != {item["source_id"] for item in items}:
            raise ValueError("INVALID_SAVED_SOURCE_INDEX")
    catalog = {
        item["quote_id"]: QuoteReference(
            source_index[item["source_id"]]
            if source_index is not None
            else UUID(item["source_id"]),
            item["text"],
        )
        for item in items
    }
    if len(catalog) != len(items):
        raise ValueError("DUPLICATE_SAVED_QUOTE_ID")
    packet = prepare_review(record["draft"], catalog, question=content["question"])
    if "retrieval_query" in content:
        query = content["retrieval_query"]
        if not isinstance(query, str) or len(query) > 4000:
            raise ValueError("INVALID_SAVED_RETRIEVAL_QUERY")
        restored = json.loads(packet.content)
        restored["retrieval_query"] = query
        packet = replace(
            packet, content=json.dumps(restored, ensure_ascii=False, separators=(",", ":"))
        )
    if source_index is not None:
        packet = compact_chat_packet(packet)
    if json.loads(packet.content) != content:
        raise ValueError("SAVED_CLAIMS_CHANGED")
    return packet


def replay_case(
    record: dict[str, Any],
    evidence: Sequence[Evidence] | None = None,
    *,
    force_qualification_quotes: bool = False,
) -> dict[str, Any]:
    packet = restore_packet(record)
    properties = _review_schema_properties(record)
    has_relevance = any(
        name in record["review"] or name in properties
        for name in ("answerability", "unanswered_parts")
    )
    missing_parts_available = False
    if has_relevance:
        if force_qualification_quotes:
            raise ValueError("FORCE_MODE_CANNOT_BYPASS_CHAT_RELEVANCE")
        reviewed, missing_parts_available = _chat_review_for_replay(record)
        projection = project_chat_review(packet, reviewed)
    else:
        projection = project_review(
            packet,
            ReviewOutput.model_validate(record["review"]),
            force_qualification_quotes=force_qualification_quotes and "answer" in packet.kinds,
        )
    result: dict[str, Any] = {
        "projection_version": PROJECTION_VERSION,
        "paid_calls": 0,
        "relevance_enforced": has_relevance,
        "missing_parts_available": missing_parts_available,
        "force_qualification_quotes": force_qualification_quotes,
        "scope_complete": projection.scope_complete,
        "dropped_claim_ids": projection.dropped_claim_ids,
        "appended_quote_ids": projection.appended_quote_ids,
        "payload": projection.payload,
        "renderer_checked": False,
    }
    if "answer" in packet.kinds and evidence is not None:
        status = (
            (
                "insufficient"
                if not projection.payload["answer"]
                else "answered"
                if projection.scope_complete
                else "partial"
            )
            if has_relevance
            else record.get("draft_evidence_status", "answered")
        )
        answer = validate_answer(
            AnswerOutput.model_validate(
                {
                    "evidence_status": status,
                    "blocks": projection.payload["answer"],
                }
            ),
            evidence,
            complete_scope=projection.scope_complete and record.get("complete_scope", True),
        )
        if has_relevance and missing_parts_available:
            answer = append_evidence_gaps(answer, reviewed.unanswered_parts)
        displayed = [
            (str(quote.source_id), quote.quote)
            for block in answer.blocks
            for quote in block.support_quotes
        ]
        result.update(
            renderer_checked=True,
            answer=answer.answer,
            evidence_status=answer.evidence_status,
            citations=answer.citations,
            retained_blocks=[block.model_dump(mode="json") for block in answer.blocks],
            appended_qualifications_retained=all(
                (str(packet.catalog[q].source_id), packet.catalog[q].quote) in displayed
                for q in projection.appended_quote_ids
            ),
            all_displayed_support_in_citations=all(
                any(c["source_id"] == source and quote in c["excerpt"] for c in answer.citations)
                for source, quote in displayed
            ),
        )
    return result


def saved_evidence(database: Session, turn_id: str) -> tuple[Evidence, ...]:
    turn = database.get(ChatTurn, UUID(turn_id))
    if turn is None:
        raise ValueError("FROZEN_TURN_UNAVAILABLE")
    sources = []
    for item in turn.context_snapshot["payload"]["evidence"]:
        chunk = database.get(
            DocumentChunk, UUID(item["source_id"]), options=[defer(DocumentChunk.embedding)]
        )
        if chunk is None:
            raise ValueError("FROZEN_SOURCE_UNAVAILABLE")
        supplied = _supplied_source_text(item)
        offset = chunk.text.find(supplied)
        if offset < 0 or chunk.text.find(supplied, offset + 1) >= 0:
            raise ValueError("FROZEN_SOURCE_OFFSET_UNRESOLVED")
        sources.append(
            Evidence(
                chunk.id,
                chunk.document_id,
                chunk.profile_id,
                item["filename"],
                supplied,
                tuple(SourceSpan(**span) for span in chunk.source_spans),
                text_offset=offset,
            )
        )
    return tuple(sources)


def run(input_path: Path, output_path: Path, quality_state: Path | None = None) -> dict[str, Any]:
    if input_path.resolve() == output_path.resolve() or output_path.exists():
        raise ValueError("REPLAY_OUTPUT_MUST_BE_NEW")
    state = json.loads(input_path.read_text())
    quality = json.loads(quality_state.read_text()) if quality_state else None
    results = {}
    engine = create_engine(get_settings().database_url, hide_parameters=True) if quality else None
    try:
        for key, record in state["cases"].items():
            if record.get("status") != "completed":
                continue
            evidence = None
            force_qualifications = False
            case_name = key.split(":", 1)[0].upper()
            if engine is not None and quality and case_name in quality["cases"]:
                with Session(engine) as database:
                    database.execute(text("SET TRANSACTION READ ONLY"))
                    turn_id = quality["cases"][case_name]["response"]["turn_id"]
                    evidence = saved_evidence(database, turn_id)

            results[key] = replay_case(
                record, evidence, force_qualification_quotes=force_qualifications
            )
    finally:
        if engine is not None:
            engine.dispose()
    output = {
        "source_artifact": str(input_path),
        "source_sha256": fingerprint(state),
        "quality_state": str(quality_state) if quality_state else None,
        "paid_calls": 0,
        "cases": results,
    }
    _atomic_write(output_path, json.dumps(output, ensure_ascii=False, indent=2).encode())
    return {
        "artifact": str(output_path),
        "paid_calls": 0,
        "cases": {
            key: {
                name: value.get(name)
                for name in (
                    "renderer_checked",
                    "evidence_status",
                    "scope_complete",
                    "appended_qualifications_retained",
                    "all_displayed_support_in_citations",
                )
            }
            for key, value in results.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--quality-state", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output, args.quality_state), indent=2))


if __name__ == "__main__":
    main()
