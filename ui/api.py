"""UI HTTP boundary. No backend settings, database, parsers or provider credentials."""

import os
import re
from functools import lru_cache
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID

import httpx

ANSWER_TIMEOUT = httpx.Timeout(105, connect=5, write=20, pool=5)


class APIError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status: int | None = None,
        request_id: str | None = None,
        turn_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code, self.message, self.status = code, message, status
        self.request_id, self.turn_id = request_id, turn_id


def _uuid(value: str) -> str:
    try:
        return str(UUID(value))
    except (ValueError, TypeError, AttributeError):
        raise APIError(
            "INVALID_REFERENCE", "The selected saved item has an invalid reference."
        ) from None


def _optional_uuid(value: Any) -> str | None:
    try:
        return str(UUID(value)) if isinstance(value, str) else None
    except ValueError:
        return None


def _base(value: str) -> str:
    try:
        parsed = urlsplit(value)
        valid_port = parsed.port is None or 1 <= parsed.port <= 65535
    except ValueError:
        raise APIError("INVALID_API_URL", "Configure a valid HTTP API URL.") from None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not valid_port
    ):
        raise APIError(
            "INVALID_API_URL", "Configure an HTTP API URL without credentials or query parameters."
        )
    return value.rstrip("/")


def _safe_message(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "<" in value
        or ">" in value
        or re.search(r"(?i)\b(authorization|bearer|password|traceback)\b|sk-[A-Za-z0-9_-]+", value)
    ):
        return "The request could not be completed. Check service status and try again."
    return re.sub(r"[\x00-\x1f\x7f]", " ", value).strip()[:300]


class APIClient:
    def __init__(
        self,
        base_url: str | None = None,
        public_url: str | None = None,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = _base(
            base_url if base_url is not None else os.getenv("API_BASE_URL", "http://api:8000")
        )
        self.public_url = _base(
            public_url
            if public_url is not None
            else os.getenv("PUBLIC_API_BASE_URL", "http://localhost:4000")
        )
        self.client = client or httpx.Client(
            base_url=self.base_url, timeout=10, follow_redirects=False
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self.client.request(method, path, **kwargs)
        except httpx.TimeoutException:
            raise APIError(
                "REQUEST_TIMEOUT",
                "The response timed out. Check saved status or history "
                "before retrying the same request.",
            ) from None
        except httpx.HTTPError:
            raise APIError(
                "CONNECTION_UNAVAILABLE",
                "Cannot reach the API. Retry when the service is available.",
            ) from None
        try:
            payload = response.json()
        except ValueError:
            raise APIError(
                "INVALID_RESPONSE",
                "The API returned an unreadable response.",
                status=response.status_code,
            ) from None
        if not response.is_success:
            error = payload.get("error") if isinstance(payload, dict) else None
            error = error if isinstance(error, dict) else {}
            code = error.get("code")
            code = (
                code
                if isinstance(code, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", code)
                else "HTTP_ERROR"
            )
            details = error.get("details")
            raise APIError(
                code,
                _safe_message(error.get("message")),
                status=response.status_code,
                request_id=_optional_uuid(error.get("request_id")),
                turn_id=_optional_uuid(details.get("turn_id"))
                if isinstance(details, dict)
                else None,
            )
        if not isinstance(payload, dict):
            raise APIError(
                "INVALID_RESPONSE",
                "The API returned an unexpected response.",
                status=response.status_code,
            )
        return payload

    def documents(self, cursor: str | None = None) -> dict[str, Any]:
        return self._request(
            "GET",
            "/api/v1/documents",
            params={"limit": 20, **({"cursor": cursor} if cursor else {})},
        )

    def document(self, document_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/documents/{_uuid(document_id)}")

    def insights(self, document_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/documents/{_uuid(document_id)}/insights")

    def upload(
        self, filename: str, content: bytes, media_type: str, request_id: str
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/documents/upload",
            files={"file": (filename, content, media_type)},
            headers={"Idempotency-Key": _uuid(request_id)},
            timeout=30,
        )

    def sessions(self, document_id: str, cursor: str | None = None) -> dict[str, Any]:
        return self._request(
            "GET",
            "/api/v1/chat/sessions",
            params={
                "document_id": _uuid(document_id),
                "limit": 20,
                **({"cursor": cursor} if cursor else {}),
            },
        )

    def new_session(self, document_id: str) -> dict[str, Any]:
        return self._request(
            "POST", "/api/v1/chat/sessions", json={"document_id": _uuid(document_id)}
        )

    def history(self, session_id: str, cursor: str | None = None) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/api/v1/chat/sessions/{_uuid(session_id)}/messages",
            params={"limit": 50, "cursor": cursor or "0"},
        )

    def ask(self, session_id: str, question: str, request_id: str) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/api/v1/chat/sessions/{_uuid(session_id)}/messages",
            json={"question": question, "client_request_id": _uuid(request_id)},
            timeout=ANSWER_TIMEOUT,
        )

    def document_metrics(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/metrics/documents")

    def processing_metrics(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/metrics/processing")

    def original_url(self, document_id: str) -> str:
        return f"{self.public_url}/api/v1/documents/{_uuid(document_id)}/file"


@lru_cache(maxsize=1)
def get_client() -> APIClient:
    return APIClient()
