"""Minimal Streamlit HTTP client; all business state remains in FastAPI."""

import hashlib
import time
import unicodedata
from collections import Counter
from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

import streamlit as st

from ui.api import APIClient, APIError, get_client

MIME_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
TERMINAL = {"COMPLETED", "FAILED"}


def _state() -> dict[str, Any]:
    if "vault" not in st.session_state:
        st.session_state["vault"] = {
            "documents": None,
            "document_cursor": None,
            "document_cache": {},
            "uploads": {},
            "sessions": {},
            "preferred_sessions": {},
            "session_bootstrap_attempted": {},
            "pending": {},
            "chat_errors": {},
            "poll": {},
            "chat_poll": {},
            "chat_pending": {},
            "visible": {},
        }
    return cast(dict[str, Any], st.session_state["vault"])


def poll_interval(started: float, now: float) -> int | None:
    elapsed = now - started
    return None if elapsed >= 600 else (2 if elapsed < 30 else 5)


def upload_fingerprint(filename: str, content: bytes, media_type: str) -> str:
    name = unicodedata.normalize("NFC", filename).replace("\\", "/").rsplit("/", 1)[-1].strip()
    return hashlib.sha256(name.encode() + b"\0" + media_type.encode() + b"\0" + content).hexdigest()


def _error(error: APIError) -> None:
    code = error.code.lower()
    message = error.message
    if code in {
        "ai_invalid_output",
        "ai_invalid_json",
        "ai_schema_invalid",
        "ai_output_truncated",
        "analysis_citation_invalid",
        "analysis_insight_review_invalid",
        "chat_invalid_output",
    }:
        message = (
            "The AI response could not be verified. Check the technical details before retrying."
        )
    elif "budget" in code or code in {"processing_deadline", "processing_attempts_exhausted"}:
        message = (
            "Processing could not finish within this installation's limits. Try a smaller file."
        )
    st.error(message)
    if code == "ai_quota":
        st.caption("Check the installation owner's OpenAI billing or usage limits before retrying.")
    elif code in {"ai_configuration", "ai_model_mismatch", "ai_authentication"}:
        st.caption("Check the installation's OpenAI key and model access before retrying.")
    with st.expander("Technical details"):
        st.text(error.message)
        st.text("Code: " + error.code)
        if error.request_id:
            st.text("Request: " + error.request_id)


def _merge(old: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = {row["id"]: row for row in old}
    merged.update({row["id"]: row for row in new})
    return list(merged.values())


def _load_documents(api: APIClient, *, more: bool = False) -> None:
    state = _state()
    page = api.documents(state["document_cursor"] if more else None)
    previous = state["documents"] or []
    rows = _merge(previous if more else [], page["items"])
    selected = st.session_state.get("selected_document")
    if selected and not any(row["id"] == selected for row in rows):
        rows += [row for row in previous if row["id"] == selected]
    state["documents"], state["document_cursor"] = rows, page.get("next_cursor")


def _select_document(api: APIClient, identifier: str) -> None:
    _load_documents(api)
    st.session_state["selected_document"] = identifier


def _upload(api: APIClient) -> None:
    file = st.file_uploader("PDF or DOCX", type=["pdf", "docx"], key="upload_file")
    if file is None:
        return
    content = file.getvalue()
    media_type = MIME_TYPES.get(file.name.rsplit(".", 1)[-1].lower(), "application/octet-stream")
    fingerprint = upload_fingerprint(file.name, content, media_type)
    state = _state()
    intent = state["uploads"].setdefault(
        fingerprint, {"request_id": str(uuid4()), "attempted": False}
    )
    if intent.get("document_id"):
        st.caption("This file already has a saved upload in this browser session.")
        if st.button(
            "Open saved upload",
            key="open_upload",
            help=(
                "Select the upload already saved in this browser session without "
                "uploading it again."
            ),
        ):
            _select_document(api, intent["document_id"])
            st.rerun()
        if intent.get("status") != "FAILED":
            return
        consent_for = fingerprint + ":" + intent["request_id"]
        if state.get("consent_for") != consent_for:
            state["consent_for"] = consent_for
            st.session_state["reupload_confirm"] = False
        fixed = st.checkbox(
            "I fixed the cause; a new upload may incur AI charges.", key="reupload_confirm"
        )
        submit = st.button(
            "Upload as a new attempt",
            disabled=not fixed,
            key="upload_again",
            help=(
                "Start a separate processing attempt after fixing the failure. This "
                "may incur new AI charges."
            ),
        )
        if submit:
            intent.clear()
            intent.update(request_id=str(uuid4()), attempted=False)
    else:
        submit = st.button(
            "Retry upload (same request)" if intent["attempted"] else "Upload document",
            key="upload_submit",
            help=(
                "Upload and analyze this document. Retrying an uncertain upload "
                "reuses the same request to avoid duplicate work. AI charges may "
                "apply."
            ),
        )
    if submit:
        intent["attempted"] = True
        try:
            with st.spinner("Submitting document…"):
                result = api.upload(file.name, content, media_type, intent["request_id"])
            intent.update(document_id=result["document_id"], status=result["status"])
            _select_document(api, result["document_id"])
            st.rerun()
        except APIError as error:
            _error(error)
            st.info(
                "You can retry safely with the same file. "
                "We will check for your original upload first."
            )


def _choose_document(api: APIClient) -> str | None:
    state = _state()
    if state["documents"] is None:
        try:
            _load_documents(api)
        except APIError as error:
            _error(error)
            return None
    rows = state["documents"] or []
    if not rows:
        st.info("Upload a document to begin.")
        return None
    by_id = {row["id"]: row for row in rows}
    filenames = Counter(row["filename"] for row in rows)
    labels = {
        value: f"{index}. {row['filename']} · {_status_label(row['status'])}"
        + (
            " · " + _session_created(row.get("created_at"))
            if filenames[row["filename"]] > 1
            else ""
        )
        for index, (value, row) in enumerate(by_id.items(), 1)
    }
    if st.session_state.get("selected_document") not in by_id:
        st.session_state["selected_document"] = rows[0]["id"]
    identifier = st.selectbox(
        "Your documents",
        list(by_id),
        key="selected_document",
        format_func=lambda value: labels[value],
    )
    if st.button(
        "Refresh documents",
        key="refresh_documents",
        help=(
            "Reload the document list and readiness labels, including uploads "
            "made in another tab. Does not call AI."
        ),
    ):
        try:
            _load_documents(api)
            st.rerun()
        except APIError as error:
            _error(error)
    if state["document_cursor"] and st.button(
        "Load more documents", key="more_documents", help="Load the next page of saved documents."
    ):
        try:
            _load_documents(api, more=True)
            st.rerun()
        except APIError as error:
            _error(error)
    return str(identifier)


def _warnings(warnings: list[dict[str, Any]]) -> None:
    for warning in warnings:
        st.warning(
            str(warning.get("message", "Inspect the original document for unsupported content."))
        )


def _status_label(status: str) -> str:
    return {
        "COMPLETED": "Ready to chat",
        "FAILED": "Needs attention",
        "PENDING": "Waiting to process",
        "PROCESSING": "Processing",
    }.get(status, "Processing")


def _document(api: APIClient, identifier: str) -> dict[str, Any] | None:
    state = _state()
    started = state["poll"].setdefault(identifier, time.monotonic())
    cached = state["document_cache"].get(identifier, {})
    interval = (
        None if cached.get("status") in TERMINAL else poll_interval(started, time.monotonic())
    )

    @st.fragment(run_every=interval)
    def watch() -> None:
        if (
            cached.get("status") not in TERMINAL
            and poll_interval(started, time.monotonic()) != interval
        ):
            st.rerun()
        try:
            document = api.document(identifier)
        except APIError as error:
            _error(error)
            if interval is None and cached.get("status") not in TERMINAL:
                st.info("Automatic status checks paused after 10 minutes.")
                if st.button(
                    "Resume status checks",
                    key="resume_document",
                    help=(
                        "Resume automatic processing-status checks for another 10 minutes. "
                        "Does not restart processing or call AI."
                    ),
                ):
                    state["poll"][identifier] = time.monotonic()
                    st.rerun()
            return
        state["document_cache"][identifier] = document
        for row in state["documents"] or []:
            if row["id"] == identifier:
                row.update(document)
        for intent in state["uploads"].values():
            if intent.get("document_id") == identifier:
                intent["status"] = document["status"]
        actual_interval = (
            None if document["status"] in TERMINAL else poll_interval(started, time.monotonic())
        )
        if actual_interval != interval:
            st.rerun()
        st.caption(_status_label(document["status"]))
        if document["status"] == "FAILED":
            st.error("Document processing failed.")
            failure = document.get("error") or {}
            _error(
                APIError(
                    str(failure.get("code", "PROCESSING_FAILED")),
                    str(failure.get("message", "Fix the cause before uploading again.")),
                )
            )
        elif document["status"] != "COMPLETED":
            if actual_interval is None:
                st.info(
                    "Automatic status checks paused after 10 minutes. Resume when you are ready."
                )
                if st.button(
                    "Resume status checks",
                    key="resume_document",
                    help=(
                        "Resume automatic processing-status checks for another 10 minutes. "
                        "Does not restart processing or call AI."
                    ),
                ):
                    state["poll"][identifier] = time.monotonic()
                    st.rerun()
            else:
                st.info("Preparing your document. This page updates automatically.")

    watch()
    return cast(dict[str, Any] | None, state["document_cache"].get(identifier))


def _analysis(api: APIClient, identifier: str) -> dict[str, Any] | None:
    try:
        result = api.insights(identifier)
    except APIError as error:
        _error(error)
        return None
    extraction = str(result.get("extraction_coverage", "unavailable")).replace("_", " ")
    if extraction not in {"complete", "complete within capability"}:
        st.warning(
            "Some original content may be missing from extracted text. "
            "Answers use the text we could read; check the original when needed."
        )
    _warnings(result.get("warnings") or [])
    return result


def _analysis_details(api: APIClient, identifier: str, result: dict[str, Any]) -> None:
    with st.expander("Document details and original"):
        st.link_button(
            "Open original file",
            api.original_url(identifier),
            key="original_file",
            help="Open the uploaded PDF or DOCX in your browser. Does not call AI.",
        )
        st.text(
            "Extraction coverage: "
            + str(result.get("extraction_coverage", "unavailable")).replace("_", " ")
        )
        _coverage_details(result)
    with st.expander("Document overview", expanded=True):
        _analysis_content(result)


def _coverage_details(result: dict[str, Any]) -> None:
    coverage = result.get("analysis_coverage") or {}
    if (
        coverage.get("covered_extracted_chunks") is not None
        and coverage.get("total_extracted_chunks") is not None
    ):
        st.caption(
            f"Analysis covers {coverage['covered_extracted_chunks']} "
            f"of {coverage['total_extracted_chunks']} extracted chunks."
        )
    st.caption(
        "Analysis of extracted text does not imply complete visual extraction of the original."
    )


def _analysis_content(result: dict[str, Any]) -> None:
    st.caption("A condensed overview. Ask in chat to explore a topic or its implications.")
    for heading in ("summary", "insights"):
        st.markdown("**" + ("Summary" if heading == "summary" else "Key insights") + "**")
        for block in (result.get("analysis") or {}).get(heading, []):
            st.text(str(block.get("text", "")))
            if block.get("support_quotes"):
                with st.expander("Supporting source text"):
                    for support in block["support_quotes"]:
                        st.text(str(support.get("quote", "")))


def _load_sessions(api: APIClient, document_id: str, *, more: bool = False) -> None:
    state = _state()
    old = state["sessions"].get(document_id, {"items": [], "next_cursor": None})
    page = api.sessions(document_id, old["next_cursor"] if more else None)
    rows = _merge(old["items"] if more else [], page["items"])
    preferred = state["preferred_sessions"].get(document_id)
    if preferred and not any(row["id"] == preferred for row in rows):
        rows += [row for row in old["items"] if row["id"] == preferred]
    state["sessions"][document_id] = {"items": rows, "next_cursor": page.get("next_cursor")}


def _history(api: APIClient, session_id: str) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    cursor, seen = None, set()
    for _ in range(100):
        page = api.history(session_id, cursor)
        rows.update({message["id"]: message for message in page["items"]})
        following = page.get("next_cursor")
        if following is None:
            return sorted(rows.values(), key=lambda row: row["sequence"])
        if following in seen:
            break
        seen.add(following)
        cursor = following
    raise APIError(
        "HISTORY_PAGINATION",
        "History could not be fully loaded. Try loading messages again before asking a question.",
    )


def _reconcile(session_id: str, messages: list[dict[str, Any]]) -> None:
    state = _state()
    intent = state["pending"].get(session_id)
    if intent is None:
        return
    turn_id = intent.get("turn_id")
    if turn_id is None:
        for message in reversed(messages):
            if (
                message["role"] == "user"
                and message.get("content") == intent["question"]
                and message["turn_id"] not in intent["known_turns"]
            ):
                turn_id = message["turn_id"]
                intent["turn_id"] = turn_id
                break
    if any(
        message["turn_id"] == turn_id
        and message["role"] == "assistant"
        and message["status"] in TERMINAL
        for message in messages
    ):
        state["pending"].pop(session_id, None)


def _send(api: APIClient, session_id: str) -> None:
    state = _state()
    intent = state["pending"][session_id]
    try:
        with st.spinner("Waiting for the saved answer…"):
            result = api.ask(session_id, intent["question"], intent["request_id"])
        intent["turn_id"] = result.get("turn_id")
        state["chat_errors"].pop(session_id, None)
    except APIError as error:
        state["chat_errors"][session_id] = {
            "code": error.code,
            "message": error.message,
            "status": error.status,
            "request_id": error.request_id,
        }
        if error.turn_id and (error.code == "CHAT_IN_PROGRESS" or (error.status or 0) >= 500):
            intent["turn_id"] = error.turn_id
        elif error.status in {400, 404, 413, 415, 422} or error.code == "IDEMPOTENCY_CONFLICT":
            # Definite admission rejection ends this intent; uncertain HTTP outcomes
            # retain the same UUID until saved history or an explicit replay resolves it.
            state["pending"].pop(session_id, None)
    st.rerun()


def _citations(citations: list[dict[str, Any]]) -> None:
    for index, citation in enumerate(citations, 1):
        with st.expander(f"Source {index}"):
            st.text(str(citation.get("filename", "Original document")))
            for locator in citation.get("locators", []):
                if "page" in locator:
                    label = f"PDF page {locator['page']}"
                    if "block" in locator:
                        label += f" · text block {locator['block']}"
                    st.text(label)
                elif "path" in locator:
                    st.text("DOCX location: " + str(locator["path"]))
            st.text(str(citation.get("excerpt", "")))


def _chat(api: APIClient, session_id: str) -> None:
    state = _state()
    started = state["chat_poll"].setdefault(session_id, time.monotonic())
    interval = (
        poll_interval(started, time.monotonic()) if state["chat_pending"].get(session_id) else None
    )

    view: dict[str, Any] = {"ready": False, "pending": True, "messages": []}
    readiness = state.setdefault("chat_history_ready", {})

    @st.fragment(run_every=interval)
    def transcript() -> None:
        messages_panel = st.container(height=500, border=False, key="chat_panel")
        with messages_panel:
            if (
                state["chat_pending"].get(session_id)
                and poll_interval(started, time.monotonic()) != interval
            ):
                st.rerun()
            try:
                messages = _history(api, session_id)
            except APIError as error:
                _error(error)
                was_ready = readiness.get(session_id)
                readiness[session_id] = False
                if was_ready:
                    st.rerun()
                if state["chat_pending"].get(session_id) and interval is None:
                    st.info("Automatic history checks paused after 10 minutes.")
                    if st.button(
                        "Resume history checks",
                        key="resume_history",
                        help=(
                            "Resume automatic checks for the pending saved answer for another 10 "
                            "minutes. Does not submit a new question."
                        ),
                    ):
                        state["chat_poll"][session_id] = time.monotonic()
                        st.rerun()
                if st.button(
                    "Try loading messages again",
                    key="retry_history_load",
                    help="Fetch saved messages again without submitting a question or calling AI.",
                ):
                    st.rerun()
                return
            was_unavailable = readiness.get(session_id) is False
            readiness[session_id] = True
            if was_unavailable:
                st.rerun()
            _reconcile(session_id, messages)
            pending = session_id in state["pending"] or any(
                message["role"] == "assistant" and message["status"] == "PENDING"
                for message in messages
            )
            if pending and not state["chat_pending"].get(session_id):
                state["chat_poll"][session_id] = time.monotonic()
                state["chat_pending"][session_id] = True
                st.rerun()
            actual = (
                poll_interval(state["chat_poll"][session_id], time.monotonic()) if pending else None
            )
            state["chat_pending"][session_id] = pending
            if actual != interval:
                st.rerun()
            view.update(ready=True, pending=pending, messages=messages)
            visible = state["visible"].setdefault(session_id, 50)
            if len(messages) > visible and st.button(
                "Show earlier messages",
                key="earlier_messages",
                help="Show up to 50 more earlier messages from this conversation.",
            ):
                state["visible"][session_id] += 50
                st.rerun()
            if messages:
                st.caption(
                    f"Showing {min(visible, len(messages))} of {len(messages)} saved messages."
                )
            for message in messages[-visible:]:
                with st.chat_message("user" if message["role"] == "user" else "assistant"):
                    if message["status"] == "FAILED":
                        st.error("This question failed.")
                    if (
                        message["role"] == "assistant"
                        and message["status"] == "COMPLETED"
                        and message.get("evidence_status") == "partial"
                    ):
                        st.warning(
                            "Partial answer: this response does not cover the full question. "
                            "Check the sources and stated limitations."
                        )
                    st.text(str(message.get("content") or "Waiting for the saved response…"))
                    _citations(message.get("citations") or [])
            if session_id in state["chat_errors"]:
                _error(APIError(**state["chat_errors"][session_id]))
            if pending:
                st.info("Your answer is on its way. We'll keep checking for it automatically.")
                if actual is None:
                    st.caption("Automatic history checks paused after 10 minutes.")
                    if st.button(
                        "Resume history checks",
                        key="resume_history",
                        help=(
                            "Resume automatic checks for the pending saved answer for another 10 "
                            "minutes. Does not submit a new question."
                        ),
                    ):
                        state["chat_poll"][session_id] = time.monotonic()
                        st.rerun()
                if session_id in state["pending"] and st.button(
                    "Check or retry this question",
                    key="retry_question",
                    help=(
                        "Check the original question using the same request ID. A saved "
                        "result is reused; if it was never accepted, processing may start and "
                        "incur AI charges."
                    ),
                ):
                    _send(api, session_id)

    transcript()
    # Keep the composer outside the polling fragment so its sticky containing
    # block includes the conversation controls, even when those are expanded.
    with st.container(key="chat_composer"):
        prompt = st.chat_input(
            "Ask about this document"
            if view["ready"]
            else "History must load before asking a question",
            disabled=not view["ready"] or view["pending"],
            key="question",
        )
    if prompt and view["ready"] and not view["pending"]:
        state["pending"][session_id] = {
            "request_id": str(uuid4()),
            "question": str(prompt),
            "known_turns": list({message["turn_id"] for message in view["messages"]}),
        }
        state["chat_poll"][session_id] = time.monotonic()
        _send(api, session_id)


def _conversation(api: APIClient, document: dict[str, Any]) -> None:
    st.subheader("Chat with your document")
    if document["status"] != "COMPLETED":
        st.info("Chat becomes available when document processing completes.")
        st.chat_input("Wait for document processing", disabled=True, key="question")
        return
    st.caption(
        "Ask follow-up questions in the same conversation. New questions may incur AI charges."
    )
    state, identifier = _state(), document["id"]
    try:
        if identifier not in state["sessions"]:
            _load_sessions(api, identifier)
        create_requested = st.button(
            "New conversation",
            key="new_session",
            help=(
                "Start a separate chat for this document. Existing conversations stay "
                "saved; creating an empty chat does not call AI."
            ),
        )
        page = state["sessions"][identifier]
        bootstrap = state.setdefault("session_bootstrap_attempted", {})
        first_conversation = not page["items"] and not bootstrap.get(identifier)
        if create_requested or first_conversation:
            # Creating an empty conversation never calls AI. Attempt automatic
            # setup only once: an uncertain HTTP outcome must not POST again on
            # unrelated reruns; the explicit button remains available to retry.
            bootstrap[identifier] = True
            result = api.new_session(identifier)
            state["preferred_sessions"][identifier] = result["id"]
            _load_sessions(api, identifier)
            st.session_state["selected_session"] = result["id"]
            st.rerun()
        page = state["sessions"][identifier]
    except APIError as error:
        # A creation response may be lost after the server saved the session.
        # Re-list on the next rerun before any explicit retry; never auto-POST
        # again merely because the previous browser cache was empty.
        state["sessions"].pop(identifier, None)
        _error(error)
        return
    by_id = {row["id"]: row for row in page["items"]}
    if not by_id:
        st.info("Start a conversation to ask a question.")
        st.chat_input("Start a conversation first", disabled=True, key="question")
        return
    if st.session_state.get("selected_session") not in by_id:
        preferred = state["preferred_sessions"].get(identifier)
        st.session_state["selected_session"] = (
            preferred if preferred in by_id else next(iter(by_id))
        )
    labels = {
        value: f"{index}. {row['title']} · {_session_created(row.get('created_at'))}"
        for index, (value, row) in enumerate(by_id.items(), 1)
    }
    with st.expander("Saved conversations"):
        session_id = st.selectbox(
            "Continue a conversation",
            list(by_id),
            key="selected_session",
            format_func=lambda value: labels[value],
        )
        if st.button(
            "Refresh conversations",
            key="refresh_sessions",
            help=(
                "Reload the saved conversation list, including chats created in "
                "another tab. Does not regenerate answers or call AI."
            ),
        ):
            _load_sessions(api, identifier)
            st.rerun()
        if page["next_cursor"] and st.button(
            "Load more conversations",
            key="more_sessions",
            help="Load the next page of saved conversations for this document.",
        ):
            _load_sessions(api, identifier, more=True)
            st.rerun()
    state["preferred_sessions"][identifier] = session_id
    _chat(api, str(session_id))


def _session_created(value: Any) -> str:
    try:
        created = datetime.fromisoformat(value)
        if created.tzinfo is None:
            return "Creation time unavailable"
        return created.astimezone(UTC).strftime("%d %b %Y, %H:%M:%S UTC")
    except (TypeError, ValueError):
        return "Creation time unavailable"


def _metrics(api: APIClient) -> None:
    with st.expander("Advanced: diagnostics and metrics"):
        st.link_button(
            "API documentation",
            api.public_url + "/docs",
            key="api_docs",
            help="Open Swagger to explore and try the backend APIs.",
        )
        st.caption(
            "Uploads and new questions can incur AI charges. "
            "Viewing saved documents and refreshing status do not call AI."
        )
        selected = st.session_state.get("selected_document")
        document = _state()["document_cache"].get(selected)
        if document:
            st.text("Document reference: " + document["id"])
            st.text("Processing stage: " + document["stage"])
        try:
            documents, processing = api.document_metrics(), api.processing_metrics()
        except APIError as error:
            st.info("Metrics are unavailable.")
            _error(error)
            return
        total = documents.get("document_count")
        outcomes = processing.get("documents") or {}
        st.metric("Saved documents", str(total) if total is not None else "Unavailable")
        completed, failed = outcomes.get("completed"), outcomes.get("failed")
        st.metric(
            "Completed in processing window",
            str(completed) if completed is not None else "Unavailable",
        )
        st.metric(
            "Failed in processing window", str(failed) if failed is not None else "Unavailable"
        )
        window = processing.get("window") or {}
        if window.get("start") and window.get("end"):
            st.text(f"Window: {window['start']} to {window['end']}")
        st.caption(
            "Processing outcomes count documents; retries are separate attempts "
            "in the monitoring API."
        )


def main() -> None:
    st.set_page_config(page_title="AI Document Vault", page_icon="📄", layout="wide")
    st.html(
        """<style>
        [data-testid="stLayoutWrapper"]:has(> .st-key-chat_composer) {
            position: sticky;
            bottom: 1rem;
            z-index: 5;
            padding-top: 0.5rem;
            background: var(--st-background-color, var(--background-color, Canvas));
        }
        </style>"""
    )
    st.title("AI Document Vault")
    st.caption("Choose a document, ask a question, and explore the sources behind each answer.")
    try:
        api = get_client()
    except APIError as error:
        _error(error)
        return
    state = _state()
    if state["documents"] is None:
        try:
            _load_documents(api)
        except APIError as error:
            _error(error)
            return
    with st.sidebar:
        st.header("Your library")
        with st.expander("Upload a document", expanded=not bool(state["documents"])):
            st.caption("PDF or DOCX with selectable text. Uploads may incur AI charges.")
            _upload(api)
        identifier = _choose_document(api)
    if identifier is not None:
        document = _document(api, identifier)
        if document is not None:
            st.text(str(document["filename"]))
            chat_panel, details_panel = st.columns([2, 1], gap="large")
            with chat_panel:
                _conversation(api, document)
            with details_panel, st.container(height=760, border=False):
                analysis = _analysis(api, identifier) if document["status"] == "COMPLETED" else None
                if analysis is not None:
                    _analysis_details(api, identifier, analysis)
                else:
                    with st.expander("Document details and original"):
                        st.link_button(
                            "Open original file",
                            api.original_url(identifier),
                            key="original_file",
                            help=(
                                "Open the uploaded PDF or DOCX in your browser. Does not call AI."
                            ),
                        )
    with st.sidebar:
        _metrics(api)


if __name__ == "__main__":
    main()
