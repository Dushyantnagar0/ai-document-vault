"""Provider-free HTTP rejection contracts before any database access."""

from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import httpx
import pytest

import app.api.documents as documents_api
from app.core.config import Settings, get_settings
from app.main import create_app

UPLOAD = "/api/v1/documents/upload"


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    # A personal installation's ignored dotenv must not tune these rejection cases.
    monkeypatch.chdir(tmp_path)
    for field in Settings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)
        monkeypatch.delenv(field, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder-never-a-real-key")
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    # Deliberately omit lifespan: no engine, broker or provider is available.
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app()), base_url="http://test"
    ) as instance:
        yield instance


def assert_error(response: httpx.Response, status: int, code: str) -> None:
    assert response.status_code == status, response.text
    error = response.json()["error"]
    assert error["code"] == code
    assert str(UUID(error["request_id"])) == response.headers["X-Request-ID"]
    assert error["message"]


def assert_no_uploads(directory: Path) -> None:
    assert not [path for path in directory.rglob("*") if path.is_file()]


@pytest.mark.parametrize("filename", ["   ", "/", "a" * 256 + ".pdf"])
async def test_invalid_filename_is_rejected_before_storage_or_database(
    client: httpx.AsyncClient, filename: str, tmp_path: Path
) -> None:
    response = await client.post(
        UPLOAD, files={"file": (filename, b"%PDF-1.7\nprivate document", "application/pdf")}
    )
    assert_error(response, 422, "INVALID_FILENAME")
    assert_no_uploads(tmp_path)


@pytest.mark.parametrize(
    "filename,media_type",
    [("report.exe", "application/octet-stream"), ("report.txt", "text/plain")],
)
async def test_unsupported_format_has_specific_status_and_cleans_staging(
    client: httpx.AsyncClient, filename: str, media_type: str, tmp_path: Path
) -> None:
    response = await client.post(
        UPLOAD, files={"file": (filename, b"private document body", media_type)}
    )
    assert_error(response, 415, "UNSUPPORTED_FORMAT")
    assert "private document body" not in response.text
    assert_no_uploads(tmp_path)


@pytest.mark.parametrize(
    "filename,body,media_type,code",
    [
        ("report.pdf", b"%PDF-1.7\nbody", "text/plain", "MEDIA_TYPE_MISMATCH"),
        ("report.docx", b"PK\x03\x04body", "application/pdf", "MEDIA_TYPE_MISMATCH"),
        ("report.pdf", b"not a PDF signature", "application/pdf", "INVALID_DOCUMENT"),
        ("report.docx", b"%PDF-1.7\nbody", "application/zip", "INVALID_DOCUMENT"),
    ],
)
async def test_media_hints_and_signatures_must_match(
    client: httpx.AsyncClient,
    filename: str,
    body: bytes,
    media_type: str,
    code: str,
    tmp_path: Path,
) -> None:
    response = await client.post(UPLOAD, files={"file": (filename, body, media_type)})
    assert_error(response, 422, code)
    assert_no_uploads(tmp_path)


async def multipart_stream(
    body: bytes, *, filename: str = "report.pdf", chunk_bytes: int = 8192
) -> AsyncIterator[bytes]:
    yield (
        '--vault-boundary\r\nContent-Disposition: form-data; name="file"; '
        f'filename="{filename}"\r\nContent-Type: application/pdf\r\n\r\n'
    ).encode()
    for offset in range(0, len(body), chunk_bytes):
        yield body[offset : offset + chunk_bytes]
    yield b"\r\n--vault-boundary--\r\n"


async def test_file_limit_applies_to_stream_without_content_length(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "128")
    request = client.build_request(
        "POST",
        UPLOAD,
        headers={"Content-Type": "multipart/form-data; boundary=vault-boundary"},
        content=multipart_stream(b"%PDF-1.7\n" + b"x" * 128, chunk_bytes=7),
    )
    assert "Content-Length" not in request.headers
    response = await client.send(request)
    assert_error(response, 413, "UPLOAD_TOO_LARGE")
    assert_no_uploads(tmp_path)


async def test_total_multipart_limit_applies_without_content_length(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "128")
    request = client.build_request(
        "POST",
        UPLOAD,
        headers={"Content-Type": "multipart/form-data; boundary=vault-boundary"},
        content=multipart_stream(b"%PDF-1.7\n" + b"x" * (1024 * 1024 + 128)),
    )
    assert "Content-Length" not in request.headers
    response = await client.send(request)
    assert_error(response, 413, "UPLOAD_TOO_LARGE")
    assert_no_uploads(tmp_path)


@pytest.mark.parametrize("suffix", ["", "/file", "/insights"])
async def test_invalid_document_uuid_is_field_validation_before_database(
    client: httpx.AsyncClient, suffix: str
) -> None:
    response = await client.get(f"/api/v1/documents/not-a-uuid{suffix}")
    assert_error(response, 422, "VALIDATION_ERROR")


@pytest.mark.parametrize("query", ["limit=0", "limit=101", "limit=word", "status=UNKNOWN"])
async def test_invalid_list_parameters_are_rejected_before_database(
    client: httpx.AsyncClient, query: str
) -> None:
    response = await client.get(f"/api/v1/documents?{query}")
    assert_error(response, 422, "VALIDATION_ERROR")


async def test_invalid_request_uuid_is_rejected_before_upload_side_effects(
    client: httpx.AsyncClient, tmp_path: Path
) -> None:
    response = await client.post(
        UPLOAD,
        headers={"Idempotency-Key": "not-a-uuid"},
        files={"file": ("report.pdf", b"%PDF-1.7\nprivate", "application/pdf")},
    )
    assert_error(response, 422, "VALIDATION_ERROR")
    assert_no_uploads(tmp_path)


async def test_insights_openapi_describes_prose_support_coverage_and_usage(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    operation = schema["paths"]["/api/v1/documents/{document_id}/insights"]["get"]
    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/DocumentInsightsView"
    }
    models = schema["components"]["schemas"]
    assert models["SavedAnalysis"]["properties"]["summary"]["items"] == {
        "$ref": "#/components/schemas/InsightBlock"
    }
    assert models["SavedAnalysis"]["properties"]["insights"]["items"] == {
        "$ref": "#/components/schemas/InsightBlock"
    }
    assert models["InsightBlock"]["properties"]["support_quotes"]["items"] == {
        "$ref": "#/components/schemas/InsightQuote"
    }
    assert {"source_id", "quote"} <= models["InsightQuote"]["properties"].keys()
    assert {"covered_extracted_chunks", "total_extracted_chunks"} == set(
        models["AnalysisCoverage"]["properties"]
    )
    assert {"call_history", "known_input_tokens", "known_output_tokens"} <= models["AnalysisUsage"][
        "properties"
    ].keys()
    assert {"lineage", "usage", "coverage_basis"} <= models["SavedAnalysis"]["properties"].keys()


@pytest.mark.parametrize("with_metadata", [False, True])
async def test_typed_insights_preserves_saved_public_payload_without_added_nulls(
    monkeypatch: pytest.MonkeyPatch, with_metadata: bool
) -> None:
    document_id, source_id = uuid4(), str(uuid4())
    analysis = {
        "summary": [
            {
                "text": "Annual renewal does not increase storage.",
                "source_ids": [source_id],
                "support_quotes": [{"source_id": source_id, "quote": "Renewal adds no storage."}],
            }
        ],
        "insights": [],
    }
    if with_metadata:
        analysis.update(
            analysis_coverage=1.0,
            analysis_scope_complete=False,
            analysis_warnings=[
                {"code": "ANALYSIS_SCOPE_PARTIAL", "message": "Some claims were omitted."}
            ],
            covered_extracted_chunks=1,
            total_extracted_chunks=1,
            covered_chunk_ids=[source_id],
            coverage_basis="all_accepted_extracted_chunks_processed",
            model="saved-model",
            prompt_version="analysis-v6",
            lineage=[
                {
                    "stage": "final",
                    "covered_chunk_ids": [source_id],
                    "cited_chunk_ids": [source_id],
                    "call_number": 2,
                }
            ],
            usage={
                "calls": 2,
                "known_input_tokens": 500,
                "known_output_tokens": 80,
                "all_calls_reported_usage": True,
                "call_history": [
                    {
                        "stage": "final",
                        "repair": False,
                        "outcome": "failed",
                        "error_code": "ai_invalid_json",
                        "input_tokens": 250,
                        "output_tokens": 20,
                        "model": "saved-model",
                        "validation_reason": "invalid_json",
                        "diagnostics": {"reason": "invalid_json", "finish_reason": None},
                    },
                    {
                        "stage": "final",
                        "repair": True,
                        "outcome": "completed",
                        "input_tokens": 250,
                        "output_tokens": 60,
                        "model": "saved-model",
                    },
                ],
                "saved_extension": {"count": 1},
            },
            saved_extension={"description": "Already persisted public metadata"},
        )
    warnings = [
        {"code": "VISUAL_CONTENT", "message": "Images are not interpreted.", "locator": None}
    ]
    stored = SimpleNamespace(
        payload=analysis,
        covered_chunks=1,
        total_chunks=1,
        extraction_coverage="unknown",
        warnings=warnings,
        model="saved-model",
        prompt_version="analysis-v6",
    )
    session = AsyncMock()
    session.__aenter__.return_value = session
    session.get.return_value = stored
    monkeypatch.setattr(
        documents_api, "get_document", AsyncMock(return_value=SimpleNamespace(status="COMPLETED"))
    )
    monkeypatch.setattr(
        documents_api, "async_sessionmaker", lambda *args, **kwargs: lambda: session
    )
    application = create_app()
    application.state.engine = object()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://test"
    ) as instance:
        response = await instance.get(f"/api/v1/documents/{document_id}/insights")
    assert response.status_code == 200, response.text
    assert response.json() == {
        "document_id": str(document_id),
        "analysis": analysis,
        "analysis_coverage": {"covered_extracted_chunks": 1, "total_extracted_chunks": 1},
        "extraction_coverage": "unknown",
        "warnings": [*warnings, *analysis.get("analysis_warnings", [])],
        "model": "saved-model",
        "prompt_version": "analysis-v6",
    }
    session.get.assert_awaited_once_with(documents_api.DocumentInsights, document_id)
