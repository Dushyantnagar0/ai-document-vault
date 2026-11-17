"""Compact wire references with complete original identities retained once per source."""

from typing import Any

from app.services.analysis_evidence import QuoteReference

COMPACT_EVIDENCE_VERSION = "source-aliases-v1"


def compact_evidence(catalog: dict[str, QuoteReference]) -> dict[str, Any]:
    """Preserve every quote, its ID, order and source grouping without repeated UUIDs.

    The caller's catalog remains the authority for validation and public citations.
    Only the provider-facing representation uses aliases.
    """
    aliases: dict[str, str] = {}
    evidence = []
    for quote_id, reference in catalog.items():
        source_id = str(reference.source_id)
        alias = aliases.setdefault(source_id, f"s{len(aliases) + 1}")
        evidence.append({"quote_id": quote_id, "source_id": alias, "text": reference.quote})
    return {
        "evidence": evidence,
        "source_index": {alias: source_id for source_id, alias in aliases.items()},
    }
