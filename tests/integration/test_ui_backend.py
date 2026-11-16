"""Streamlit AppTest emulation over actual isolated HTTP/DB/worker/provider-stub services.

This verifies UI-to-backend integration, not a real browser or live AI quality.
"""

import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pymupdf
import pytest
from streamlit.testing.v1 import AppTest

from ui.api import APIClient, get_client

pytestmark = [
    pytest.mark.integration,
    pytest.mark.enable_socket,
    pytest.mark.skipif(
        os.environ.get("VAULT_INTEGRATION_TESTS") != "1",
        reason="Requires isolated Compose API/worker/provider fixture",
    ),
]
APP = Path(__file__).resolve().parents[2] / "ui" / "app.py"


@pytest.fixture
def actual_api(monkeypatch: pytest.MonkeyPatch) -> Iterator[APIClient]:
    if (
        os.environ.get("POSTGRES_DB") != "vault_test"
        or os.environ.get("OPENAI_API_KEY") != "test-placeholder-never-a-real-key"
        or os.environ.get("OPENAI_BASE_URL") != "http://provider-stub:4010/v1"
    ):
        pytest.fail("Refusing non-test backend/provider configuration", pytrace=False)
    monkeypatch.setenv("API_BASE_URL", "http://api:8000")
    monkeypatch.setenv("PUBLIC_API_BASE_URL", "http://api:8000")
    get_client.cache_clear()
    client = get_client()
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            response = client.client.get("/health/ready", timeout=3)
            if (
                response.status_code == 200
                and response.json()["components"].get("worker") == "ready"
            ):
                break
        except (httpx.HTTPError, ValueError, KeyError):
            pass
        time.sleep(0.2)
    else:
        client.client.close()
        get_client.cache_clear()
        pytest.fail("Isolated API/worker did not become ready within60s")
    try:
        yield client
    finally:
        client.client.close()
        get_client.cache_clear()


def _clean(result: AppTest) -> None:
    assert not result.exception


def _select_document(result: AppTest, identifier: str) -> None:
    for _ in range(100):
        if any(row["id"] == identifier for row in result.session_state["vault"]["documents"]):
            result.selectbox(key="selected_document").set_value(identifier).run()
            _clean(result)
            return
        result.button(key="more_documents").click().run()
        _clean(result)
    pytest.fail("Uploaded document was not reachable through UI pagination")


def _messages(api: APIClient, session_id: str) -> list[dict[str, Any]]:
    response = api.history(session_id)
    assert response["next_cursor"] is None
    return list(response["items"])


def test_streamlit_upload_to_real_backend_chat_followup_saved_reload_and_metrics(
    actual_api: APIClient,
) -> None:
    with pymupdf.open() as document:
        document.new_page().insert_text(
            (72, 72),
            "Project Juniper agreement\nOwner: Priya Shah\n"
            "Renewal date: 12 December 2027\nAnnual fee: 240 USD\n",
        )
        content = document.tobytes()
    filename = f"ui-backend-{uuid4().hex[:12]}.pdf"
    result = AppTest.from_file(APP, default_timeout=20).run()
    _clean(result)
    result.file_uploader[0].set_value((filename, content, "application/pdf")).run()
    _clean(result)
    result.button(key="upload_submit").click().run()
    _clean(result)
    identifier = str(result.session_state["selected_document"])
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        state = actual_api.document(identifier)
        assert state["filename"] == filename
        assert state["status"] != "FAILED", state.get("error")
        if state["status"] == "COMPLETED":
            break
        # AppTest explicitly emulates reruns; browser timer scheduling is tested separately.
        result.run()
        _clean(result)
        time.sleep(0.2)
    else:
        pytest.fail("UI-uploaded document did not complete within40s")
    result.run()
    _clean(result)
    assert actual_api.document(identifier)["ai_calls_reserved"] == 4
    assert any("Extraction coverage:" in element.value for element in result.text)
    assert any("Priya Shah" in element.value for element in result.text)
    assert not result.chat_input[0].disabled
    assert len(actual_api.sessions(identifier)["items"]) == 1
    session_id = str(result.selectbox(key="selected_session").value)
    assert not result.chat_input[0].disabled
    result.chat_input[0].set_value("Who owns this agreement?").run()
    _clean(result)
    first = _messages(actual_api, session_id)
    assert len(first) == 2 and "Priya Shah" in first[-1]["content"]
    assert first[-1]["status"] == "COMPLETED"
    assert any(element.label == "Source 1" for element in result.expander)
    assert any("PDF page 1" in element.value for element in result.text)
    assert first[-1]["citations"][0]["document_id"] == identifier
    result.chat_input[0].set_value("And when does it renew?").run()
    _clean(result)
    saved = _messages(actual_api, session_id)
    assert [message["sequence"] for message in saved] == [1, 2, 3, 4]
    assert "12 December 2027" in saved[-1]["content"]
    assert all(message["status"] == "COMPLETED" for message in saved)
    assert all(
        citation["document_id"] == identifier
        for message in saved
        for citation in message["citations"]
    )
    result.run()
    _clean(result)
    assert _messages(actual_api, session_id) == saved
    metrics = {metric.label: metric.value for metric in result.metric}
    assert metrics["Saved documents"] == str(actual_api.document_metrics()["document_count"])
    assert metrics["Completed in processing window"] != "Unavailable"
    assert metrics["Failed in processing window"] != "Unavailable"
    # A completely fresh Streamlit session restores persisted state through HTTP only.
    restored = AppTest.from_file(APP, default_timeout=20).run()
    _clean(restored)
    _select_document(restored, identifier)
    restored.selectbox(key="selected_session").set_value(session_id).run()
    _clean(restored)
    assert any("Who owns this agreement?" == element.value for element in restored.text)
    assert any("12 December 2027" in element.value for element in restored.text)
    assert any("PDF page 1" in element.value for element in restored.text)
    assert _messages(actual_api, session_id) == saved
    assert actual_api.document(identifier)["ai_calls_reserved"] == 4
