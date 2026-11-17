"""Independently authored synthetic content and expectations for core AI quality."""

import io
from uuid import NAMESPACE_URL, UUID, uuid5

from docx import Document

from scripts.evaluation_fixtures import EvaluationCase

QUALITY_VERSION = "eval-quality-v1"
DOCUMENT_KEY = "quality-request-guide"

# These facts/expectations are authored inputs, never inferred from generated answers.
GUIDE_SECTIONS = (
    (
        "Intake and acceptance",
        "A service request is a recorded instruction to carry out a defined piece of work. "
        "The intake service validates the submitted fields, writes the request to a durable log, "
        "and places its identifier on the work queue. Only after the log and queue intent are "
        "durable does it acknowledge the request as accepted. Accepted therefore means that the "
        "system has recorded the request and arranged for later processing. It does not mean "
        "that a worker has started, that an external message has been delivered, or that the "
        "requested work has completed. The acknowledgement includes a request identifier that "
        "the requester can use to inspect status. Retrying the same submission identifier "
        "returns the recorded request rather than intentionally creating another request.",
    ),
    (
        "Workers and completion",
        "Workers process queued requests asynchronously. A worker takes an eligible request, "
        "records a processing claim, performs the required work, and saves the outcome before "
        "the request can be shown as complete. The intake acknowledgement and the worker's "
        "completion record are separate events. A busy queue can therefore contain accepted "
        "requests for which work has not begun. The status page distinguishes queued, processing, "
        "completed, and failed requests. Operators should use the saved status and outcome to "
        "answer progress questions. An acknowledgement is useful evidence of durable receipt, "
        "but cannot substitute for a completion record. A failed request remains visible with "
        "its attempt history so that a requester can understand what happened without guessing "
        "from a missing item or a disconnected browser.",
    ),
    (
        "Leases and recovery",
        "Each processing claim has a renewable lease. A worker renews its lease while healthy, "
        "and only the current unexpired claim may publish the saved outcome. If a worker "
        "crashes, its lease eventually expires and recovery can make the request eligible for "
        "another attempt. This protects the saved request state from a stale worker publishing "
        "over its replacement. It does not make every external side effect happen exactly once. "
        "For example, a worker might send an external email and crash before recording that "
        "delivery. A replacement attempt may send the email again. Duplicate external email "
        "is therefore possible even when only one final outcome is published. Operators retain "
        "the interrupted attempt and the replacement attempt rather than deleting the earlier "
        "history to make recovery look like a single uninterrupted execution.",
    ),
    (
        "Capacity and backlog",
        "For routine requests, planning assumes two workers and eight completed requests per "
        "worker per hour. The simple combined planning capacity is sixteen routine requests "
        "per hour. At an arrival rate of twelve routine requests per hour, this leaves four "
        "requests per hour of nominal spare capacity under those assumptions. If arrivals "
        "remain above sixteen per hour while capacity stays unchanged, the queue backlog "
        "grows. Accepting requests more quickly does not increase the workers' completion "
        "capacity. These figures describe a planning example, not a guarantee for every mix "
        "of work. Manual review, external service delays, and interruptions may change actual "
        "throughput. Operators inspect queue depth and completion rates before interpreting "
        "a fast acknowledgement as evidence that the service can sustain a higher workload.",
    ),
    (
        "Manual review rules",
        "Automation may process a routine request only when its confidence score is at least "
        "0.80 and no mandatory review rule applies. A confidence score below 0.80 requires "
        "manual review. Independently, a request value greater than 1,000 units always requires "
        "manual review, even when confidence is high. A value of exactly 1,000 units does not "
        "trigger that value rule by itself; other review rules still apply. These are separate "
        "conditions, so improving confidence cannot bypass the value threshold. The review "
        "decision and its reason are saved with the request. A request waiting for a reviewer "
        "must remain distinguishable from one completed automatically. The planning example "
        "does not assign a fixed duration to manual review or promise that every reviewed "
        "request can be completed in the same hour.",
    ),
    (
        "Retention and exceptions",
        "The ordinary retention rule keeps a closed request for thirty days after its closure. "
        "Active requests are excluded from that ordinary deletion schedule because their "
        "work has not closed. For a request subject to a dispute, the applicable retention "
        "period is ninety days after the dispute closes, rather than thirty days after the "
        "original request closes. A legal hold prevents deletion until the hold is released, "
        "even when an otherwise applicable retention date has passed. Release of a legal hold "
        "does not create a universal new thirty-day clock: the applicable ordinary or dispute "
        "rule must still be considered. Operators check active status, dispute status, and "
        "legal holds before scheduling deletion. They record the relevant closure date and "
        "any hold release so that a retention decision can be explained from saved facts.",
    ),
    (
        "Urgent acknowledgement",
        "Urgent requests have a target acknowledgement time of five minutes during staffed "
        "hours: 09:00 to 17:00 UTC on weekdays. This is an acknowledgement target, not a "
        "resolution guarantee. A request can receive a timely acknowledgement and still wait "
        "for an available worker, manual review, or an external response. The five-minute "
        "target is not stated to apply outside those staffed hours. Operators communicate the "
        "current saved status and the relevant service target rather than promising completion "
        "within five minutes. Acceptance, completion, confidence, capacity, and retention answer "
        "different operational questions. Reading those rules together helps explain why a "
        "quick intake response can coexist with a queue, why a recovered request may produce "
        "a repeated email, and why an old closed request can remain stored under an exception.",
    ),
)
GUIDE_TEXT = "\n\n".join(f"{title}\n{text}" for title, text in GUIDE_SECTIONS)

QUALITY_CASES = (
    EvaluationCase(
        "Q1",
        DOCUMENT_KEY,
        "What does an accepted request mean in this guide?",
        (r"durable|persist|record|log", r"queue|queued"),
    ),
    EvaluationCase(
        "Q2",
        DOCUMENT_KEY,
        "Why doesn't that mean the work is complete?",
        (r"worker|asynchronous", r"acknowledg|accept", r"complet|finish"),
        ("Q1",),
    ),
    EvaluationCase(
        "Q3",
        DOCUMENT_KEY,
        "Summarize the retention rules and exceptions: include the clocks for ordinary and "
        "disputed requests, active requests, and legal holds.",
        (r"30|thirty", r"90|ninety", r"clos", r"active", r"legal hold|hold.*releas"),
    ),
    EvaluationCase(
        "Q4",
        DOCUMENT_KEY,
        "Give a whole-document overview covering intake, workers and recovery, capacity, "
        "review rules, retention, and the urgent-service qualification.",
        (
            r"intake|accept|durable",
            r"worker",
            r"recover|lease|crash",
            r"capacity|sixteen|16",
            r"retention|retain|thirty|30",
            r"manual|confidence",
            r"acknowledg",
            r"guarantee|resolut",
        ),
    ),
)
EXPECTED_INTENTS = {"Q1": "factual", "Q2": "factual", "Q3": "factual", "Q4": "overview"}


def quality_request_id(case_id: str, revision: int = 1) -> UUID:
    return uuid5(NAMESPACE_URL, f"document-vault:{QUALITY_VERSION}:{case_id}:revision:{revision}")


def guide_docx() -> bytes:
    document = Document()
    document.add_heading("Request handling guide", 0)
    for title, text in GUIDE_SECTIONS:
        document.add_heading(title, 1)
        document.add_paragraph(text)
    output = io.BytesIO()
    document.save(output)
    return output.getvalue()
