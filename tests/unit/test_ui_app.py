"""Streamlit AppTest emulation with a fake HTTP boundary, not a real browser/E2E."""

import time
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from streamlit.testing.v1 import AppTest

import ui.api
from ui.api import APIError
from ui.app import poll_interval, upload_fingerprint

pytestmark = pytest.mark.unit
DOCUMENT, SECOND_DOCUMENT, SESSION, SECOND_SESSION = (str(UUID(int=index)) for index in range(1, 5))
APP = Path(__file__).resolve().parents[2] / "ui" / "app.py"


class FakeAPI:
    public_url = "http://localhost:4000"

    def __init__(self, *, empty: bool = False) -> None:
        self.docs = {} if empty else {DOCUMENT: self.make_document(DOCUMENT, "COMPLETED")}
        self.conversations = {DOCUMENT: [{"id": SESSION, "title": "Saved conversation"}]}
        self.messages: dict[str, list[dict[str, Any]]] = {SESSION: []}
        self.uploads: list[tuple[Any, ...]] = []
        self.questions: list[tuple[str, str, str]] = []
        self.session_writes = 0
        self.gets: list[tuple[Any, ...]] = []
        self.fail_upload = False
        self.upload_status = "PENDING"
        self.fail_question = False
        self.accept_timeout_question = True
        self.fail_terminal = False
        self.pagination = False
        self.metrics_unavailable = False

    @staticmethod
    def make_document(identifier: str, status: str) -> dict[str, Any]:
        return {
            "id": identifier,
            "filename": "saved.pdf",
            "status": status,
            "stage": "COMPLETED" if status == "COMPLETED" else "EXTRACTING",
            "error": {"code": "ai_configuration", "message": "Fix model access."}
            if status == "FAILED"
            else None,
        }

    def documents(self, cursor: str | None = None) -> dict[str, Any]:
        self.gets.append(("documents", cursor))
        rows = list(self.docs.values())
        if self.pagination and len(rows) > 1:
            return {
                "items": rows[1:] if cursor else rows[:1],
                "next_cursor": None if cursor else "documents-2",
            }
        return {"items": rows, "next_cursor": None}

    def document(self, identifier: str) -> dict[str, Any]:
        self.gets.append(("document", identifier))
        return dict(self.docs[identifier])

    def upload(
        self, filename: str, content: bytes, media_type: str, request_id: str
    ) -> dict[str, Any]:
        self.uploads.append((filename, content, media_type, request_id))
        if self.fail_upload:
            self.fail_upload = False
            raise APIError("REQUEST_TIMEOUT", "The response timed out.")
        identifier = str(uuid4())
        self.docs[identifier] = {
            **self.make_document(identifier, self.upload_status),
            "filename": filename,
        }
        self.conversations[identifier] = []
        return {"document_id": identifier, "status": self.upload_status}

    def insights(self, identifier: str) -> dict[str, Any]:
        return {
            "extraction_coverage": "partial",
            "analysis_coverage": {"covered_extracted_chunks": 1, "total_extracted_chunks": 1},
            "warnings": [{"message": "Images were not interpreted."}],
            "analysis": {
                "summary": [
                    {"text": "Saved summary.", "support_quotes": [{"quote": "Owner Priya Shah."}]}
                ],
                "insights": [],
            },
        }

    def sessions(self, identifier: str, cursor: str | None = None) -> dict[str, Any]:
        self.gets.append(("sessions", identifier, cursor))
        rows = self.conversations.get(identifier, [])
        if self.pagination and len(rows) > 1:
            return {
                "items": rows[1:] if cursor else rows[:1],
                "next_cursor": None if cursor else "sessions-2",
            }
        return {"items": list(rows), "next_cursor": None}

    def new_session(self, identifier: str) -> dict[str, Any]:
        self.session_writes += 1
        row = {"id": str(uuid4()), "title": "New conversation"}
        self.conversations.setdefault(identifier, []).insert(0, row)
        self.messages[row["id"]] = []
        return row

    def history(self, identifier: str, cursor: str | None = None) -> dict[str, Any]:
        self.gets.append(("history", identifier, cursor))
        rows = self.messages.get(identifier, [])
        start = int(cursor or 0)
        return {
            "items": rows[start : start + 50],
            "next_cursor": str(start + 50) if len(rows) > start + 50 else None,
        }

    def append_turn(
        self,
        identifier: str,
        question: str,
        *,
        status: str = "COMPLETED",
        turn_id: str | None = None,
        evidence_status: str | None = "answered",
    ) -> str:
        rows = self.messages.setdefault(identifier, [])
        turn = turn_id or str(uuid4())
        rows.extend(
            [
                {
                    "id": str(uuid4()),
                    "turn_id": turn,
                    "sequence": len(rows) + 1,
                    "role": "user",
                    "content": question,
                    "status": "COMPLETED",
                    "citations": [],
                },
                {
                    "id": str(uuid4()),
                    "turn_id": turn,
                    "sequence": len(rows) + 2,
                    "role": "assistant",
                    "content": "Owner Priya Shah." if status != "PENDING" else None,
                    "status": status,
                    "evidence_status": evidence_status,
                    "evidence_scope": "retrieved_excerpts",
                    "citations": [
                        {
                            "filename": "saved.pdf",
                            "locators": [{"page": 1}],
                            "excerpt": "Owner Priya Shah.",
                        }
                    ]
                    if status == "COMPLETED"
                    else [],
                },
            ]
        )
        return turn

    def ask(self, identifier: str, question: str, request_id: str) -> dict[str, Any]:
        self.questions.append((identifier, question, request_id))
        rows = self.messages.setdefault(identifier, [])
        if self.fail_question:
            self.fail_question = False
            if self.accept_timeout_question:
                self.append_turn(identifier, question, status="PENDING")
            raise APIError("REQUEST_TIMEOUT", "The response timed out.")
        if self.fail_terminal:
            self.fail_terminal = False
            turn = self.append_turn(identifier, question, status="FAILED")
            raise APIError("ai_quota", "AI account quota is unavailable.", status=502, turn_id=turn)
        pending = next(
            (row for row in rows if row["role"] == "assistant" and row["status"] == "PENDING"), None
        )
        if pending is not None:
            pending.update(status="COMPLETED", content="Owner Priya Shah.", citations=[])
            turn = pending["turn_id"]
        else:
            turn = self.append_turn(identifier, question)
        return {"turn_id": turn, "status": "COMPLETED"}

    def original_url(self, identifier: str) -> str:
        return f"{self.public_url}/api/v1/documents/{identifier}/file"

    def document_metrics(self) -> dict[str, Any]:
        if self.metrics_unavailable:
            raise APIError("UNAVAILABLE", "Monitoring is unavailable.")
        return {"document_count": len(self.docs)}

    def processing_metrics(self) -> dict[str, Any]:
        return {
            "documents": {"completed": 2, "failed": 1},
            "window": {"start": "2026-11-15T00:00:00Z", "end": "2026-11-16T00:00:00Z"},
        }


def run_app(monkeypatch: pytest.MonkeyPatch, fake: FakeAPI) -> AppTest:
    monkeypatch.setattr(ui.api, "get_client", lambda: fake)
    result = AppTest.from_file(APP, default_timeout=10).run()
    assert not result.exception
    return result


def test_saved_views_restore_backend_history_citations_and_coverage_without_paid_posts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeAPI()
    fake.append_turn(SESSION, "Who owns it?")
    result = run_app(monkeypatch, fake)
    assert any("Owner Priya Shah." in item.value for item in result.text)
    assert any("PDF page 1" in item.value for item in result.text)
    assert any("Extraction coverage: partial" in item.value for item in result.text)
    assert not fake.questions and not fake.uploads and fake.session_writes == 0
    again = AppTest.from_file(APP, default_timeout=10).run()
    assert not again.exception and any("Owner Priya Shah." in item.value for item in again.text)
    assert [item.value for item in again.metric] == ["1", "2", "1"]


def test_default_screen_places_chat_beside_document_information(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeAPI()
    fake.append_turn(SESSION, "Who owns it?")
    result = run_app(monkeypatch, fake)
    assert result.sidebar.selectbox(key="selected_document").value == DOCUMENT
    assert result.main.subheader[0].value == "Chat with your document"
    assert not result.main.chat_input[0].disabled
    assert len(result.main.metric) == 0
    assert len(result.sidebar.metric) == 3
    expanders = {item.label: item for item in result.expander}
    for label in (
        "Upload a document",
        "Saved conversations",
        "Document details and original",
        "Source 1",
        "Advanced: diagnostics and metrics",
    ):
        assert not expanders[label].proto.expanded
    assert expanders["Document overview"].proto.expanded
    columns = result.main.get("column")
    assert len(columns) == 2
    assert columns[0].subheader[0].value == "Chat with your document"
    assert any(item.label == "Document overview" for item in columns[1].expander)
    assert not any(item.label == "Refresh history" for item in result.button)
    assert any("Saved summary." == item.value for item in expanders["Document overview"].text)
    assert any("may be missing" in item.value for item in result.main.warning)
    assert all(
        DOCUMENT[-8:] not in label for label in result.selectbox(key="selected_document").options
    )
    assert all(
        SESSION[-8:] not in label for label in result.selectbox(key="selected_session").options
    )
    assert not fake.questions and not fake.uploads and fake.session_writes == 0


def test_same_named_documents_and_conversations_remain_distinguishable_without_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeAPI()
    fake.docs[SECOND_DOCUMENT] = fake.make_document(SECOND_DOCUMENT, "COMPLETED")
    fake.conversations[DOCUMENT].append({"id": SECOND_SESSION, "title": "Saved conversation"})
    result = run_app(monkeypatch, fake)
    for key in ("selected_document", "selected_session"):
        options = result.selectbox(key=key).options
        assert len(options) == len(set(options)) == 2
    result.selectbox(key="selected_document").set_value(SECOND_DOCUMENT).run()
    assert not result.exception
    assert not result.chat_input[0].disabled
    assert fake.session_writes == 1
    result.button(key="new_session").click().run()
    assert not result.exception and not result.chat_input[0].disabled
    assert fake.session_writes == 2
    assert not fake.questions and not fake.uploads


def test_message_reload_is_only_offered_after_a_load_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeAPI()
    fake.append_turn(SESSION, "Who owns it?")
    history = fake.history

    def fail_once(identifier: str, cursor: str | None = None) -> dict[str, Any]:
        monkeypatch.setattr(fake, "history", history)
        raise APIError("SERVICE_UNAVAILABLE", "Messages could not be loaded.", status=503)

    monkeypatch.setattr(fake, "history", fail_once)
    result = run_app(monkeypatch, fake)
    assert result.chat_input[0].disabled
    result.button(key="retry_history_load").click().run()
    assert not result.exception and not result.chat_input[0].disabled
    assert any(item.value == "Owner Priya Shah." for item in result.text)
    assert not any(item.key == "retry_history_load" for item in result.button)
    assert not fake.questions and not fake.uploads and fake.session_writes == 0


def test_saved_conversation_creation_times_identify_same_titles_after_fresh_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeAPI()
    fake.conversations[DOCUMENT] = [
        {
            "id": SESSION,
            "title": "saved.pdf",
            "created_at": "2026-11-15T09:30:45+05:30",
        },
        {
            "id": SECOND_SESSION,
            "title": "saved.pdf",
            "created_at": "2026-11-14T16:20:30Z",
        },
    ]
    result = run_app(monkeypatch, fake)
    options = result.selectbox(key="selected_session").options
    assert options == [
        "1. saved.pdf · 15 Nov 2026, 04:00:45 UTC",
        "2. saved.pdf · 14 Nov 2026, 16:20:30 UTC",
    ]
    reloaded = AppTest.from_file(APP, default_timeout=10).run()
    assert not reloaded.exception
    assert reloaded.selectbox(key="selected_session").options == options
    reloaded.selectbox(key="selected_session").set_value(SECOND_SESSION).run()
    assert not reloaded.exception
    assert reloaded.selectbox(key="selected_session").value == SECOND_SESSION
    assert not fake.questions and not fake.uploads and fake.session_writes == 0


def test_duplicate_document_names_show_persisted_times_while_unique_names_stay_compact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeAPI()
    fake.docs[DOCUMENT]["created_at"] = "2026-11-15T09:30:45+05:30"
    fake.docs[SECOND_DOCUMENT] = {
        **fake.make_document(SECOND_DOCUMENT, "COMPLETED"),
        "created_at": "2026-11-14T16:20:30Z",
    }
    unique = str(UUID(int=5))
    fake.docs[unique] = {
        **fake.make_document(unique, "COMPLETED"),
        "filename": "unique.pdf",
        "created_at": "2026-11-13T12:00:00Z",
    }
    expected = [
        "1. saved.pdf · Ready to chat · 15 Nov 2026, 04:00:45 UTC",
        "2. saved.pdf · Ready to chat · 14 Nov 2026, 16:20:30 UTC",
        "3. unique.pdf · Ready to chat",
    ]
    result = run_app(monkeypatch, fake)
    assert result.selectbox(key="selected_document").options == expected
    result.button(key="refresh_documents").click().run()
    assert not result.exception
    assert result.selectbox(key="selected_document").options == expected
    reloaded = AppTest.from_file(APP, default_timeout=10).run()
    assert not reloaded.exception
    assert reloaded.selectbox(key="selected_document").options == expected
    assert all(identifier not in label for identifier in fake.docs for label in expected)
    assert not fake.questions and not fake.uploads and fake.session_writes == 0


def test_complete_native_extraction_keeps_capability_qualification_in_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeAPI()
    insights = fake.insights(DOCUMENT)
    insights.update(extraction_coverage="complete_within_capability", warnings=[])
    monkeypatch.setattr(fake, "insights", lambda identifier: insights)
    result = run_app(monkeypatch, fake)
    assert not any("may be missing" in item.value for item in result.warning)
    details = next(
        item for item in result.expander if item.label == "Document details and original"
    )
    assert not details.proto.expanded
    assert any("complete within capability" in item.value for item in details.text)
    assert any(
        "does not imply complete visual extraction" in item.value for item in details.caption
    )


def test_processing_error_is_actionable_with_collapsed_technical_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeAPI()
    fake.docs[DOCUMENT].update(
        status="FAILED",
        error={"code": "analysis_call_budget", "message": "Attempt reached 64 call budget."},
    )
    result = run_app(monkeypatch, fake)
    assert result.chat_input[0].disabled
    assert any("Try a smaller file" in item.value for item in result.error)
    assert not any("64" in item.value for item in result.error)
    details = next(item for item in result.expander if item.label == "Technical details")
    assert not details.proto.expanded
    assert any("64 call budget" in item.value for item in details.text)
    assert not fake.questions and not fake.uploads


def test_partial_saved_answer_warns_after_reload_without_paid_posts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeAPI()
    fake.append_turn(SESSION, "List every condition.", evidence_status="partial")
    result = run_app(monkeypatch, fake)
    assert any(
        "Partial answer:" in item.value and "does not cover the full question" in item.value
        for item in result.warning
    )
    assert any("PDF page 1" in item.value for item in result.text)
    reloaded = AppTest.from_file(APP, default_timeout=10).run()
    assert not reloaded.exception
    assert any("Partial answer:" in item.value for item in reloaded.warning)
    assert not fake.questions and not fake.uploads and fake.session_writes == 0


@pytest.mark.parametrize("evidence_status", ["answered", "insufficient", None])
def test_partial_warning_uses_assistant_classification_only(
    monkeypatch: pytest.MonkeyPatch, evidence_status: str | None
) -> None:
    fake = FakeAPI()
    fake.append_turn(SESSION, "A partial answer?", evidence_status=evidence_status)
    fake.messages[SESSION][0]["evidence_status"] = "partial"
    fake.messages[SESSION][1]["content"] = "Partial information appears in this source."
    result = run_app(monkeypatch, fake)
    assert not any("Partial answer:" in item.value for item in result.warning)
    assert not fake.questions and not fake.uploads


@pytest.mark.parametrize("status", ["PENDING", "FAILED"])
def test_partial_warning_requires_completed_assistant_message(
    monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    fake = FakeAPI()
    fake.append_turn(SESSION, "List every condition.", status=status, evidence_status="partial")
    result = run_app(monkeypatch, fake)
    assert not any("Partial answer:" in item.value for item in result.warning)
    assert not fake.questions and not fake.uploads


def test_upload_timeout_reuses_key_bytes_and_deduplicates_reruns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeAPI(empty=True)
    fake.fail_upload = True
    result = run_app(monkeypatch, fake)
    result.file_uploader[0].set_value(("tiny.pdf", b"%PDF-stable", "application/pdf")).run()
    result.button(key="upload_submit").click().run()
    assert not result.exception and len(fake.uploads) == 1
    first_key = fake.uploads[0][-1]
    result.run()
    assert len(fake.uploads) == 1
    result.button(key="upload_submit").click().run()
    assert not result.exception and len(fake.uploads) == 2
    assert fake.uploads[0] == fake.uploads[1]
    assert fake.uploads[1][-1] == first_key
    assert result.chat_input[0].disabled
    result.run()
    assert len(fake.uploads) == 2 and not fake.questions
    identifier = next(iter(fake.docs))
    fake.docs[identifier].update(status="COMPLETED", stage="COMPLETED")
    result.run()
    assert not result.exception
    assert len(fake.uploads) == 2 and any("Ready to chat" == item.value for item in result.caption)
    assert not result.chat_input[0].disabled
    assert result.chat_input[0].placeholder == "Ask about this document"
    assert fake.session_writes == 1 and not fake.questions
    selected_session = result.selectbox(key="selected_session").value
    result.run()
    assert not result.exception and not result.chat_input[0].disabled
    assert result.selectbox(key="selected_session").value == selected_session
    assert fake.session_writes == 1 and len(fake.uploads) == 2 and not fake.questions


def test_timeout_retry_keeps_question_uuid_and_followup_gets_new_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeAPI()
    fake.fail_question = True
    result = run_app(monkeypatch, fake)
    result.chat_input[0].set_value("Who owns it?").run()
    assert not result.exception and len(fake.questions) == 1
    request_id = fake.questions[0][-1]
    assert result.chat_input[0].disabled
    result.run()
    assert len(fake.questions) == 1
    result.button(key="retry_question").click().run()
    assert not result.exception and len(fake.questions) == 2
    assert fake.questions[1] == fake.questions[0]
    assert not result.chat_input[0].disabled
    result.chat_input[0].set_value("And the renewal date?").run()
    assert not result.exception and len(fake.questions) == 3
    assert fake.questions[-1][-1] != request_id
    result.run()
    assert len(fake.questions) == 3


def test_old_identical_question_does_not_clear_uncertain_new_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeAPI()
    fake.append_turn(SESSION, "Who owns it?")
    fake.fail_question = True
    fake.accept_timeout_question = False
    result = run_app(monkeypatch, fake)
    result.chat_input[0].set_value("Who owns it?").run()
    assert not result.exception and result.chat_input[0].disabled
    assert SESSION in result.session_state["vault"]["pending"]
    assert len(fake.questions) == 1
    result.run()
    assert SESSION in result.session_state["vault"]["pending"] and len(fake.questions) == 1


def test_terminal_failure_restores_history_and_allows_new_intent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeAPI()
    fake.fail_terminal = True
    result = run_app(monkeypatch, fake)
    result.chat_input[0].set_value("Who owns it?").run()
    assert not result.exception and not result.chat_input[0].disabled
    assert SESSION not in result.session_state["vault"]["pending"]
    assert any("AI account quota is unavailable." == item.value for item in result.error)
    result.chat_input[0].set_value("Try another question.").run()
    assert not result.exception and len(fake.questions) == 2
    assert fake.questions[0][-1] != fake.questions[1][-1]


def test_all_document_session_and_history_pages_are_reachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeAPI()
    fake.pagination = True
    fake.docs[SECOND_DOCUMENT] = fake.make_document(SECOND_DOCUMENT, "COMPLETED")
    fake.conversations[DOCUMENT].append({"id": SECOND_SESSION, "title": "Older conversation"})
    fake.messages[SECOND_SESSION] = []
    for index in range(26):
        fake.append_turn(SECOND_SESSION, f"Earlier question {index}")
    result = run_app(monkeypatch, fake)
    result.button(key="more_documents").click().run()
    assert not result.exception
    result.button(key="more_sessions").click().run()
    assert not result.exception
    result.selectbox(key="selected_session").set_value(SECOND_SESSION).run()
    assert not result.exception
    assert ("documents", "documents-2") in fake.gets
    assert ("sessions", DOCUMENT, "sessions-2") in fake.gets
    assert ("history", SECOND_SESSION, "50") in fake.gets
    assert not any(item.value == "Earlier question 0" for item in result.text)
    result.button(key="earlier_messages").click().run()
    assert any(item.value == "Earlier question 0" for item in result.text)
    assert not fake.questions and not fake.uploads


def test_polling_intervals_pause_and_manual_resume_never_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert poll_interval(100, 100) == 2
    assert poll_interval(100, 129.9) == 2
    assert poll_interval(100, 130) == 5
    assert poll_interval(100, 700) is None
    fake = FakeAPI()
    fake.docs[DOCUMENT].update(status="PROCESSING", stage="EMBEDDING")
    result = run_app(monkeypatch, fake)
    assert result.chat_input[0].disabled
    result.session_state["vault"]["poll"][DOCUMENT] = time.monotonic() - 601
    result.run()
    assert not result.exception
    result.button(key="resume_document").click().run()
    assert not result.exception and result.chat_input[0].disabled
    assert result.session_state["vault"]["poll"][DOCUMENT] > time.monotonic() - 10
    assert not fake.questions and not fake.uploads and fake.session_writes == 0


def test_failed_upload_needs_explicit_consent_and_new_request_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeAPI(empty=True)
    fake.upload_status = "FAILED"
    result = run_app(monkeypatch, fake)
    result.file_uploader[0].set_value(("failed.pdf", b"%PDF-failed", "application/pdf")).run()
    result.button(key="upload_submit").click().run()
    assert not result.exception and len(fake.uploads) == 1
    assert result.button(key="upload_again").disabled
    fake.upload_status = "PENDING"
    result.checkbox(key="reupload_confirm").check().run()
    result.button(key="upload_again").click().run()
    assert not result.exception and len(fake.uploads) == 2
    assert fake.uploads[0][-1] != fake.uploads[1][-1]
    assert fake.uploads[0][:3] == fake.uploads[1][:3]


def test_untrusted_saved_content_is_plain_text_and_unavailable_metrics_are_not_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeAPI()
    fake.metrics_unavailable = True
    fake.append_turn(SESSION, "Question?")
    fake.messages[SESSION][-1]["content"] = '<script>alert("untrusted")</script>'
    fake.messages[SESSION][-1]["citations"][0]["excerpt"] = "<b>untrusted excerpt</b>"
    result = run_app(monkeypatch, fake)
    assert any("<script>" in item.value for item in result.text)
    assert any("<b>untrusted excerpt</b>" == item.value for item in result.text)
    assert not any("<script>" in item.value for item in result.markdown)
    assert len(result.metric) == 0
    assert any("Metrics are unavailable" in item.value for item in result.info)


def test_new_conversation_is_explicit_and_restored_selection_does_not_create_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeAPI()
    result = run_app(monkeypatch, fake)
    assert fake.session_writes == 0
    result.button(key="new_session").click().run()
    assert not result.exception and fake.session_writes == 1
    result.run()
    assert fake.session_writes == 1 and not fake.questions
    assert result.selectbox(key="selected_session").value != SESSION
    assert upload_fingerprint("same.pdf", b"same", "application/pdf") == upload_fingerprint(
        "same.pdf", b"same", "application/pdf"
    )
    assert upload_fingerprint("same.pdf", b"same", "application/pdf") != upload_fingerprint(
        "other.pdf", b"same", "application/pdf"
    )


def test_restored_pending_assistant_blocks_new_question_until_saved_terminal_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeAPI()
    fake.append_turn(SESSION, "Earlier accepted question", status="PENDING")
    result = run_app(monkeypatch, fake)
    assert result.chat_input[0].disabled
    assert SESSION not in result.session_state["vault"]["pending"]
    result.session_state["vault"]["chat_poll"][SESSION] = time.monotonic() - 601
    result.run()
    result.button(key="resume_history").click().run()
    assert not result.exception and result.chat_input[0].disabled
    fake.messages[SESSION][-1].update(status="COMPLETED", content="A saved answer.")
    result.run()
    assert not result.exception and not result.chat_input[0].disabled
    assert any(item.value == "A saved answer." for item in result.text)
    assert not fake.questions


def test_rejected_question_can_be_corrected_without_trapping_an_unsaved_intent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeAPI()
    original = fake.ask

    def reject_once(identifier: str, question: str, request_id: str) -> dict[str, Any]:
        if not fake.questions:
            fake.questions.append((identifier, question, request_id))
            raise APIError("INVALID_QUESTION", "Shorten the question and try again.", status=422)
        return original(identifier, question, request_id)

    monkeypatch.setattr(fake, "ask", reject_once)
    result = run_app(monkeypatch, fake)
    result.chat_input[0].set_value("Rejected question.").run()
    assert not result.exception and not result.chat_input[0].disabled
    assert SESSION not in result.session_state["vault"]["pending"]
    result.chat_input[0].set_value("Corrected question.").run()
    assert not result.exception and len(fake.questions) == 2
    assert fake.questions[0][-1] != fake.questions[1][-1]


def test_ready_document_starts_first_conversation_once_without_ai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeAPI()
    fake.conversations[DOCUMENT] = []
    result = run_app(monkeypatch, fake)
    assert not result.chat_input[0].disabled
    assert result.chat_input[0].placeholder == "Ask about this document"
    selected = result.selectbox(key="selected_session").value
    assert fake.session_writes == 1
    result.run()
    assert not result.exception and not result.chat_input[0].disabled
    assert result.selectbox(key="selected_session").value == selected
    assert fake.session_writes == 1 and not fake.questions and not fake.uploads


def test_first_conversation_failure_requires_explicit_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeAPI()
    fake.conversations[DOCUMENT] = []
    create = fake.new_session
    attempts: list[str] = []

    def fail_once(identifier: str) -> dict[str, Any]:
        attempts.append(identifier)
        monkeypatch.setattr(fake, "new_session", create)
        raise APIError("REQUEST_TIMEOUT", "Could not confirm conversation creation", status=504)

    monkeypatch.setattr(fake, "new_session", fail_once)
    result = run_app(monkeypatch, fake)
    assert attempts == [DOCUMENT] and not fake.session_writes
    result.run()
    assert attempts == [DOCUMENT] and not fake.session_writes
    assert result.chat_input[0].disabled
    result.button(key="new_session").click().run()
    assert not result.exception and not result.chat_input[0].disabled
    assert fake.session_writes == 1 and not fake.questions and not fake.uploads


def test_uncertain_first_conversation_recovers_saved_session_without_reposting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeAPI()
    fake.conversations[DOCUMENT] = []
    create = fake.new_session

    def saved_but_lost(identifier: str) -> dict[str, Any]:
        create(identifier)
        monkeypatch.setattr(fake, "new_session", create)
        raise APIError("REQUEST_TIMEOUT", "Creation response was lost", status=504)

    monkeypatch.setattr(fake, "new_session", saved_but_lost)
    result = run_app(monkeypatch, fake)
    assert fake.session_writes == 1
    result.run()
    assert not result.exception and not result.chat_input[0].disabled
    assert result.selectbox(key="selected_session").value == fake.conversations[DOCUMENT][0]["id"]
    assert fake.session_writes == 1 and not fake.questions and not fake.uploads
