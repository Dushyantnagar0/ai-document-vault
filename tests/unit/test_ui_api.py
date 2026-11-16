"""UI HTTP contracts use MockTransport only; no backend or provider requests."""

import json
from uuid import UUID, uuid4

import httpx
import pytest

from ui.api import APIClient, APIError

pytestmark = pytest.mark.unit
DOCUMENT, SESSION, REQUEST = str(uuid4()), str(uuid4()), str(uuid4())


def test_upload_filename_bytes_mime_and_stable_key_are_sent_unchanged() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(202, json={"document_id": DOCUMENT, "status": "PENDING"})

    client = APIClient(
        client=httpx.Client(base_url="http://api:8000", transport=httpx.MockTransport(handler))
    )
    for _ in range(2):
        assert (
            client.upload("named.pdf", b"%PDF-exact-bytes", "application/pdf", REQUEST)[
                "document_id"
            ]
            == DOCUMENT
        )
    assert len(requests) == 2
    for request in requests:
        assert request.headers["Idempotency-Key"] == REQUEST
        assert b'filename="named.pdf"' in request.content
        assert b"Content-Type: application/pdf" in request.content
        assert b"%PDF-exact-bytes" in request.content


def test_question_payload_uuid_timeout_and_no_automatic_retry() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise httpx.ReadTimeout("private header data", request=request)

    client = APIClient(
        client=httpx.Client(base_url="http://api:8000", transport=httpx.MockTransport(handler))
    )
    for _ in range(2):
        with pytest.raises(APIError) as caught:
            client.ask(SESSION, "Exact question?", REQUEST)
        assert caught.value.code == "REQUEST_TIMEOUT" and "private" not in str(caught.value)
    assert len(requests) == 2
    assert (
        json.loads(requests[0].content)
        == json.loads(requests[1].content)
        == {"question": "Exact question?", "client_request_id": REQUEST}
    )
    assert requests[0].extensions["timeout"]["read"] == 105


def test_cursor_pages_and_read_routes_are_get_only() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"items": [], "next_cursor": None})

    client = APIClient(
        client=httpx.Client(base_url="http://api:8000", transport=httpx.MockTransport(handler))
    )
    client.documents("document-cursor")
    client.sessions(DOCUMENT, "session-cursor")
    client.history(SESSION, "50")
    client.document(DOCUMENT)
    client.insights(DOCUMENT)
    client.document_metrics()
    client.processing_metrics()
    assert all(request.method == "GET" for request in requests)
    assert requests[0].url.params["cursor"] == "document-cursor"
    assert requests[1].url.params["document_id"] == DOCUMENT
    assert requests[1].url.params["cursor"] == "session-cursor"
    assert requests[2].url.params["cursor"] == "50"
    assert requests[-2].url.path == "/api/v1/metrics/documents"


def test_safe_error_envelope_keeps_code_request_and_turn_without_raw_headers() -> None:
    request_id, turn_id = str(uuid4()), str(uuid4())

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            502,
            headers={"x-private": "do-not-show"},
            json={
                "error": {
                    "code": "ai_quota",
                    "message": "AI account quota is unavailable.",
                    "request_id": request_id,
                    "details": {"turn_id": turn_id},
                }
            },
        )

    client = APIClient(
        client=httpx.Client(base_url="http://api:8000", transport=httpx.MockTransport(handler))
    )
    with pytest.raises(APIError) as caught:
        client.ask(SESSION, "Question?", REQUEST)
    error = caught.value
    assert error.status == 502 and error.code == "ai_quota"
    assert error.request_id == request_id and error.turn_id == turn_id
    assert "do-not-show" not in repr(error)


@pytest.mark.parametrize(
    "message",
    ["<script>alert('private')</script>", "Authorization: Bearer private", "password=private"],
)
def test_untrusted_error_message_is_replaced(message: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500, json={"error": {"code": "<html>", "message": message, "request_id": "not-an-id"}}
        )

    client = APIClient(
        client=httpx.Client(base_url="http://api:8000", transport=httpx.MockTransport(handler))
    )
    with pytest.raises(APIError) as caught:
        client.document(DOCUMENT)
    assert caught.value.code == "HTTP_ERROR"
    assert caught.value.request_id is None and "private" not in str(caught.value)


def test_public_original_link_and_environment_only_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_BASE_URL", "http://api:8000")
    monkeypatch.setenv("PUBLIC_API_BASE_URL", "http://localhost:4000")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-read")
    client = APIClient()
    assert (
        client.original_url(DOCUMENT) == f"http://localhost:4000/api/v1/documents/{DOCUMENT}/file"
    )
    assert UUID(REQUEST)
    with pytest.raises(APIError):
        APIClient(base_url="https://user:password@example.invalid")
    with pytest.raises(APIError):
        APIClient(public_url="javascript:alert(1)")
