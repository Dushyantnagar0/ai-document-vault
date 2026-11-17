"""Real HTTP/PG/Redis/Celery with a deterministic provider HTTP substitute."""

import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from uuid import UUID, uuid4

import httpx
import pymupdf
import pytest
from docx import Document as DocxDocument
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import Document, DocumentChunk, DocumentInsights, ProcessingAttempt
from app.services.analysis_review import ANALYSIS_REVIEW_PROMPT_VERSION

pytestmark = [
    pytest.mark.integration,
    pytest.mark.enable_socket,
    pytest.mark.skipif(
        os.environ.get("VAULT_INTEGRATION_TESTS") != "1", reason="Requires isolated Compose stack"
    ),
]


def fixture_bytes(format: str) -> tuple[bytes, str, str]:
    text = "Project Juniper renews on 12 December 2027. The annual fee is 240 dollars."
    if format == "pdf":
        document = pymupdf.open()
        document.new_page().insert_text((72, 72), text)
        content = document.tobytes()
        document.close()
        return content, "sample.pdf", "application/pdf"
    document = DocxDocument()
    document.add_paragraph(text)
    result = io.BytesIO()
    document.save(result)
    return (
        result.getvalue(),
        "sample.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


def await_terminal(identifier: str) -> dict[str, Any]:
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        response = httpx.get(f"http://api:8000/api/v1/documents/{identifier}", timeout=5)
        response.raise_for_status()
        state: dict[str, Any] = response.json()
        if state["status"] in {"FAILED", "COMPLETED"}:
            return state
        time.sleep(0.2)
    pytest.fail("Document did not reach a terminal state within40s")


@pytest.mark.parametrize("format", ["pdf", "docx"])
def test_real_ingestion_atomic_outputs_and_replay(format: str) -> None:
    content, filename, media_type = fixture_bytes(format)
    request_id = str(uuid4())
    with httpx.Client(base_url="http://api:8000", timeout=10) as client:
        result = client.post(
            "/api/v1/documents/upload",
            files={"file": (filename, content, media_type)},
            headers={"Idempotency-Key": request_id},
        )
        assert result.status_code == 202, result.text
        identifier = result.json()["document_id"]
        state = await_terminal(identifier)
        assert state["status"] == "COMPLETED", state
        assert state["chunk_count"] > 0
        insights = client.get(f"/api/v1/documents/{identifier}/insights")
        assert insights.status_code == 200
        assert (
            insights.json()["analysis_coverage"]["covered_extracted_chunks"] == state["chunk_count"]
        )
        assert client.get(f"/api/v1/documents/{identifier}/file").content == content
        replay = client.post(
            "/api/v1/documents/upload",
            files={"file": (filename, content, media_type)},
            headers={"Idempotency-Key": request_id},
        )
        assert replay.status_code == 202 and replay.json()["document_id"] == identifier
        conflict = client.post(
            "/api/v1/documents/upload",
            files={"file": (f"other-{filename}", content, media_type)},
            headers={"Idempotency-Key": request_id},
        )
        assert conflict.status_code == 409
    engine = create_engine(get_settings().database_url)
    try:
        with Session(engine) as session:
            document = session.get(Document, UUID(identifier))
            assert document is not None and document.attempt_count == 1
            assert document.ai_calls_reserved == 4
            assert (
                session.scalar(
                    select(func.count())
                    .select_from(DocumentChunk)
                    .where(DocumentChunk.document_id == UUID(identifier))
                )
                == state["chunk_count"]
            )
            saved = session.get(DocumentInsights, UUID(identifier))
            assert saved is not None
            review = saved.payload["evidence_review"]
            assert review["prompt_version"] == ANALYSIS_REVIEW_PROMPT_VERSION
            assert review["insight_prompt_version"] == "focused-insights-v2"
            assert review["quote_catalog_version"] == "literal-sentences-v2"
            assert review["decisions"]["decisions"] and review["quote_bindings"]
            assert review["scope_complete"] and saved.payload["analysis_scope_complete"]
            assert saved.payload["analysis_warnings"] == []
            usage = saved.payload["usage"]
            assert usage["calls"] == 3
            assert [call["stage"] for call in usage["call_history"]] == [
                "final",
                "insights",
                "evidence_review",
            ]
            assert usage["known_input_tokens"] == 300
            assert usage["known_output_tokens"] == 180
            assert usage["all_calls_reported_usage"]
            assert (
                session.scalar(
                    select(func.count())
                    .select_from(ProcessingAttempt)
                    .where(ProcessingAttempt.document_id == UUID(identifier))
                )
                == 1
            )
    finally:
        engine.dispose()


def test_concurrent_identical_upload_creates_one_document() -> None:
    content, filename, media_type = fixture_bytes("pdf")
    request_id = str(uuid4())

    def upload() -> str:
        response = httpx.post(
            "http://api:8000/api/v1/documents/upload",
            files={"file": (filename, content, media_type)},
            headers={"Idempotency-Key": request_id},
            timeout=10,
        )
        assert response.status_code == 202, response.text
        return response.json()["document_id"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        identifiers = list(pool.map(lambda _: upload(), range(2)))
    assert identifiers[0] == identifiers[1]
    assert await_terminal(identifiers[0])["status"] == "COMPLETED"


def test_corrupt_pdf_accepted_then_fails_without_calling_provider() -> None:
    response = httpx.post(
        "http://api:8000/api/v1/documents/upload",
        files={"file": ("corrupt.pdf", b"%PDF-broken-content", "application/pdf")},
        timeout=10,
    )
    assert response.status_code == 202
    state = await_terminal(response.json()["document_id"])
    assert state["status"] == "FAILED"
    assert state["error"]["code"]
