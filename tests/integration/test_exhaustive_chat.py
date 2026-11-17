"""Real async ingestion/chat services with an offline provider, including replay."""

import os
import re
from uuid import UUID, uuid4

import httpx
import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session
from test_chat import _ask, _history, _session, _stats, _turns, _wait_document
from test_chat import api as api
from test_chat import database as database

from app.db.models import DocumentChunk
from scripts.evaluation_fixtures import additional_fixture

pytestmark = [
    pytest.mark.integration,
    pytest.mark.enable_socket,
    pytest.mark.skipif(
        os.environ.get("VAULT_INTEGRATION_TESTS") != "1",
        reason="Requires isolated Compose services and offline provider",
    ),
]


def test_incomplete_inventory_uses_literal_excerpts_persists_mode_and_replays(
    api: httpx.Client, database: Engine
) -> None:
    accepted = api.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "exhaustive-forty-conditions.docx",
                additional_fixture("long"),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
        headers={"Idempotency-Key": str(uuid4())},
    )
    assert accepted.status_code == 202, accepted.text
    document_id = accepted.json()["document_id"]
    document = _wait_document(api, document_id)
    assert document["chunk_count"] > 8
    session_id = _session(api, document_id, "Incomplete inventory regression")
    question = "List all 40 numbered conditions and every approval threshold; is this complete?"
    request_id = str(uuid4())
    response = _ask(api, session_id, question, request_id)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["evidence_status"] == "partial"
    prefix = "Relevant source excerpts (partial coverage, not a complete inventory):\n\n"
    assert body["answer"].startswith(prefix)
    assert body["evidence_scope"] == "retrieved_excerpts"
    assert 1 <= len(body["citations"]) <= 8
    with Session(database) as session:
        for citation in body["citations"]:
            assert citation["document_id"] == document_id
            chunk = session.get(DocumentChunk, UUID(citation["source_id"]))
            assert chunk is not None and str(chunk.document_id) == document_id
            assert citation["excerpt"] in chunk.text
            assert len(citation["excerpt"]) <= 1200
            assert citation["locators"]

    # Excluding fixed scope explanations and citation labels, every returned block
    # must be verbatim text from the citation displayed beside it.
    excerpts = body["answer"][len(prefix) :].removesuffix(
        "\n\nThe available excerpts do not establish complete coverage of all "
        "the material requested."
    )
    blocks = list(re.finditer(r"(.*?) \[(\d+)\](?:\n\n|$)", excerpts, re.DOTALL))
    assert blocks and blocks[-1].end() == len(excerpts)
    previous = 0
    for match in blocks:
        assert match.start() == previous
        literal, reference = match.group(1), int(match.group(2))
        assert literal.strip()
        assert literal in body["citations"][reference - 1]["excerpt"]
        previous = match.end()

    turns = _turns(database, session_id)
    assert len(turns) == 1 and turns[0]["status"] == "COMPLETED"
    assert turns[0]["context"]["answer_mode"] == "partial_source_excerpts"
    assert turns[0]["calls"] == 3
    assert "evidence_review" not in turns[0]["context"]
    assert turns[0]["response"] == body
    history = _history(api, session_id)["items"]
    assert len(history) == 2
    assert history[1]["content"] == body["answer"]
    assert history[1]["evidence_status"] == "partial"
    after = _stats()
    replay = _ask(api, session_id, question, request_id)
    assert replay.status_code == 200 and replay.json() == body
    assert _stats() == after
    assert len(_turns(database, session_id)) == 1
