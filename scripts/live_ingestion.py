"""Explicit, idempotent live-ingestion smoke check inside the API container.

Run with --live to opt into paid processing. Existing demo bytes and upload UUIDs
are reused, so rerunning an unchanged completed smoke check does not request AI.
"""

import argparse
import io
import json
import os
import re
import tempfile
import time
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, cast
from uuid import NAMESPACE_URL, UUID, uuid5

import httpx
import pymupdf
from docx import Document

from app.core.config import ConfigurationError, get_settings

BASE_URL = "http://localhost:8000"
POLL_SECONDS = 180
MIME_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
SAFE_CODE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")


class SmokeFailure(Exception):
    def __init__(self, code: str, *, http_status: int | None = None) -> None:
        self.code = code
        self.http_status = http_status
        super().__init__(code)


def _safe_code(value: Any, fallback: str) -> str:
    return value if isinstance(value, str) and SAFE_CODE.fullmatch(value) else fallback


def _response(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        raise SmokeFailure("INVALID_API_RESPONSE", http_status=response.status_code) from None
    if not response.is_success:
        error = payload.get("error") if isinstance(payload, dict) else None
        code = error.get("code") if isinstance(error, dict) else None
        raise SmokeFailure(_safe_code(code, "HTTP_ERROR"), http_status=response.status_code)
    if not isinstance(payload, dict):
        raise SmokeFailure("INVALID_API_RESPONSE", http_status=response.status_code)
    return payload


def _make_bytes(format_name: str) -> bytes:
    if format_name == "pdf":
        with pymupdf.open() as document:  # type: ignore[no-untyped-call]
            page = document.new_page()
            page.insert_text(
                (72, 72),
                "Juniper service agreement\nOwner: Priya Shah\n"
                "Renewal date: 12 December 2027\nAnnual fee: 240 USD\n",
            )
            return cast(bytes, document.tobytes(garbage=4, deflate=True))
    document = Document()
    document.add_heading("Maple service agreement", 0)
    document.add_paragraph("Owner: Alex Rivers")
    document.add_paragraph("Renewal date: 20 January 2028")
    document.add_paragraph("Annual fee: 360 USD")
    output = io.BytesIO()
    document.save(output)
    return output.getvalue()


def _demo_file(root: Path, format_name: str, fixture_version: int) -> Path:
    directory = root / "demo"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"live-smoke-v{fixture_version}.{format_name}"
    if destination.exists():
        return destination
    content = _make_bytes(format_name)
    # Publish complete bytes without overwriting a concurrent winner. The demo
    # directory is separate from API-owned opaque original-file storage keys.
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=directory, prefix=".live-smoke-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        with suppress(FileExistsError):
            os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def _run(format_name: str, fixture_version: int) -> tuple[dict[str, Any], int]:
    sample = _demo_file(get_settings().local_storage_dir, format_name, fixture_version)
    request_id = uuid5(NAMESPACE_URL, f"document-vault:live-smoke-v{fixture_version}:{format_name}")
    with httpx.Client(base_url=BASE_URL, timeout=10, follow_redirects=False) as client:
        with sample.open("rb") as stream:
            accepted = _response(
                client.post(
                    "/api/v1/documents/upload",
                    files={"file": (sample.name, stream, MIME_TYPES[format_name])},
                    headers={"Idempotency-Key": str(request_id)},
                )
            )
        try:
            identifier = str(UUID(accepted["document_id"]))
        except (KeyError, TypeError, ValueError):
            raise SmokeFailure("INVALID_API_RESPONSE") from None
        result: dict[str, Any] = {"document_id": identifier, "format": format_name}
        deadline = time.monotonic() + POLL_SECONDS
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                result["error_code"] = "POLL_TIMEOUT"
                return result, 1
            state = _response(
                client.get(
                    f"/api/v1/documents/{identifier}",
                    timeout=min(10, remaining),
                )
            )
            status = state.get("status")
            if status not in {"PENDING", "PROCESSING", "COMPLETED", "FAILED"}:
                raise SmokeFailure("INVALID_API_RESPONSE")
            result.update(status=status, ai_calls_reserved=state.get("ai_calls_reserved"))
            if status == "FAILED":
                error = state.get("error")
                code = error.get("code") if isinstance(error, dict) else None
                result["error_code"] = _safe_code(code, "PROCESSING_FAILED")
                return result, 1
            if status == "COMPLETED":
                insights = _response(client.get(f"/api/v1/documents/{identifier}/insights"))
                analysis = insights.get("analysis")
                if not isinstance(analysis, dict):
                    raise SmokeFailure("INVALID_API_RESPONSE")
                result.update(
                    summary=analysis.get("summary"),
                    insights=analysis.get("insights"),
                    coverage={
                        "extraction": insights.get("extraction_coverage"),
                        "analysis": insights.get("analysis_coverage"),
                    },
                    provider_usage=analysis.get("usage"),
                )
                return result, 0
            time.sleep(min(1, max(0, deadline - time.monotonic())))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explicit opt-in live ingestion smoke check.")
    parser.add_argument(
        "--live", action="store_true", required=True, help="Allow paid AI processing."
    )
    parser.add_argument("--format", choices=("pdf", "docx"), default="pdf")
    parser.add_argument("--fixture-version", type=int, choices=(1, 2), default=None)
    args = parser.parse_args(argv)
    # Parse the mandatory opt-in before reading configuration, creating files or HTTP.
    try:
        payload, exit_code = _run(
            args.format, args.fixture_version or (2 if args.format == "docx" else 1)
        )
    except SmokeFailure as exc:
        payload = {"format": args.format, "error_code": exc.code}
        if exc.http_status is not None:
            payload["http_status"] = exc.http_status
        exit_code = 1
    except httpx.HTTPError:
        payload, exit_code = {"format": args.format, "error_code": "HTTP_REQUEST_FAILED"}, 1
    except ConfigurationError:
        payload, exit_code = {"format": args.format, "error_code": "INVALID_CONFIGURATION"}, 1
    except OSError:
        payload, exit_code = {"format": args.format, "error_code": "DEMO_FILE_ERROR"}, 1
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
