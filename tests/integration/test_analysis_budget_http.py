"""Short-paragraph DOCX reaches publication through real HTTP/queue/storage.

Generation uses the local provider substitute, not a live semantic-quality judge.
"""

import io
import os
from uuid import UUID, uuid4

import httpx
import pytest
from docx import Document as DocxDocument
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import Document, DocumentChunk, DocumentInsights, ProcessingAttempt
from tests.integration.test_ingestion import await_terminal

pytestmark = [
    pytest.mark.integration,
    pytest.mark.enable_socket,
    pytest.mark.skipif(
        os.environ.get("VAULT_INTEGRATION_TESTS") != "1", reason="Requires isolated Compose stack"
    ),
]


def test_400_short_docx_paragraphs_complete_with_full_chunk_participation() -> None:
    fixture = DocxDocument()
    for index in range(1, 401):
        fixture.add_paragraph(f"Valve {index} is red.")
    output = io.BytesIO()
    fixture.save(output)
    content = output.getvalue()
    with httpx.Client(base_url="http://api:8000", timeout=10) as client:
        response = client.post(
            "/api/v1/documents/upload",
            files={
                "file": (
                    "short-paragraph-budget.docx",
                    content,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
            headers={"Idempotency-Key": str(uuid4())},
        )
        assert response.status_code == 202, response.text
        identifier = response.json()["document_id"]
        state = await_terminal(identifier)
        assert state["status"] == "COMPLETED", state
        assert state["chunk_count"] > 1
        insights = client.get(f"/api/v1/documents/{identifier}/insights")
        assert insights.status_code == 200, insights.text
        assert (
            insights.json()["analysis_coverage"]["covered_extracted_chunks"] == state["chunk_count"]
        )
        assert client.get(f"/api/v1/documents/{identifier}/file").content == content

    engine = create_engine(get_settings().database_url)
    try:
        with Session(engine) as session:
            document = session.get(Document, UUID(identifier))
            saved = session.get(DocumentInsights, UUID(identifier))
            assert document is not None and saved is not None
            assert document.attempt_count == 1
            chunks = list(
                session.scalars(
                    select(DocumentChunk).where(DocumentChunk.document_id == UUID(identifier))
                )
            )
            assert len(chunks) == state["chunk_count"]
            source_ids = {str(chunk.id) for chunk in chunks}
            assert set(saved.payload["covered_chunk_ids"]) == source_ids
            assert saved.payload["analysis_coverage"] == 1.0
            assert (
                saved.payload["evidence_review"]["evidence_format_version"] == "source-aliases-v1"
            )
            assert all(
                quote["source_id"] in source_ids
                for quote in saved.payload["evidence_review"]["quote_bindings"].values()
            )
            calls = saved.payload["usage"]["call_history"]
            assert calls[0]["stage"] == "section"
            assert [call["stage"] for call in calls[-3:]] == [
                "final",
                "insights",
                "evidence_review",
            ]
            assert all(call["outcome"] == "completed" for call in calls)
            attempts = list(
                session.scalars(
                    select(ProcessingAttempt).where(
                        ProcessingAttempt.document_id == UUID(identifier)
                    )
                )
            )
            assert len(attempts) == 1
    finally:
        engine.dispose()
