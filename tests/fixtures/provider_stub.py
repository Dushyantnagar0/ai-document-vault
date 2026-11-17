"""Deterministic provider fixture; orchestration evidence, never AI quality evidence."""

import hashlib
import json
import math
import re
import threading
import time
from collections import Counter
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from uuid import UUID

TEST_AUTHORIZATION = "Bearer test-placeholder-never-a-real-key"
LOCK = threading.Lock()
COUNTS: Counter[str] = Counter()
REASONING_COUNTS: Counter[str] = Counter()
REASONING_SCHEMAS = frozenset(
    {
        "PlannerOutput",
        "AnswerOutput",
        "AnswerSelectionOutput",
        "SourceSelectionOutput",
        "CompositionOutput",
        "ExhaustiveSelectionOutput",
        "SelectionOutput",
        "InsightCandidates",
        "ChatReviewOutput",
        "AnalysisReviewOutput",
        "ReviewOutput",
        "AnalysisOutput",
    }
)
PLANNER_QUESTIONS: Counter[str] = Counter()
ANSWER_QUESTIONS: Counter[str] = Counter()
MARKER = re.compile(r"\[operational:([0-9a-f-]{36})\]")


@dataclass
class Barrier:
    nonce: str
    phase: str
    expires_at: float
    release: threading.Event = field(default_factory=threading.Event)
    consumed: bool = False
    entered_at: float | None = None
    active: int = 0


class BarrierRegistry:
    """Test-only one-shot gates; no request may block longer than sixty seconds."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.lock = threading.Lock()
        self.barriers: dict[str, Barrier] = {}

    @staticmethod
    def nonce(value: Any) -> str:
        if not isinstance(value, str) or len(value) != 36:
            raise ValueError("Invalid barrier nonce")
        if str(UUID(value)) != value:
            raise ValueError("Invalid barrier nonce")
        return value

    def arm(self, nonce: str, phase: str, ttl_seconds: float = 60) -> dict[str, Any]:
        nonce = self.nonce(nonce)
        if (
            phase not in {"analysis", "answer"}
            or isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, (int, float))
            or not 1 <= ttl_seconds <= 60
        ):
            raise ValueError("Invalid barrier phase or lifetime")
        with self.lock:
            existing = self.barriers.get(nonce)
            if existing is not None:
                if existing.phase != phase:
                    raise ValueError("Barrier nonce already identifies another phase")
                return self._snapshot(existing)
            if len(self.barriers) >= 256:
                self.barriers = {
                    key: value
                    for key, value in self.barriers.items()
                    if self.clock() < value.expires_at + 300 or value.active
                }
            if len(self.barriers) >= 256:
                raise ValueError("Barrier registry is full")
            barrier = Barrier(nonce, phase, self.clock() + ttl_seconds)
            self.barriers[nonce] = barrier
            return self._snapshot(barrier)

    def _snapshot(self, barrier: Barrier) -> dict[str, Any]:
        now = self.clock()
        return {
            "nonce": barrier.nonce,
            "phase": barrier.phase,
            "entered": barrier.entered_at is not None,
            "consumed": barrier.consumed,
            "active": barrier.active,
            "released": barrier.release.is_set(),
            "expired": now >= barrier.expires_at,
            "elapsed_seconds": now - barrier.entered_at if barrier.entered_at is not None else None,
            "remaining_seconds": max(0.0, barrier.expires_at - now),
        }

    def snapshot(self, nonce: str) -> dict[str, Any] | None:
        nonce = self.nonce(nonce)
        with self.lock:
            barrier = self.barriers.get(nonce)
            return self._snapshot(barrier) if barrier is not None else None

    def release(self, nonce: str) -> dict[str, Any] | None:
        nonce = self.nonce(nonce)
        with self.lock:
            barrier = self.barriers.get(nonce)
            if barrier is None:
                return None
            barrier.release.set()
            return self._snapshot(barrier)

    def wait(self, phase: str, content: dict[str, Any]) -> None:
        matches = MARKER.findall(json.dumps(content, ensure_ascii=False))
        with self.lock:
            barrier = next(
                (
                    self.barriers[nonce]
                    for nonce in matches
                    if nonce in self.barriers
                    and self.barriers[nonce].phase == phase
                    and not self.barriers[nonce].consumed
                    and not self.barriers[nonce].release.is_set()
                    and self.clock() < self.barriers[nonce].expires_at
                ),
                None,
            )
            if barrier is None:
                return
            barrier.consumed, barrier.entered_at, barrier.active = True, self.clock(), 1
            remaining = max(0.0, barrier.expires_at - self.clock())
        try:
            barrier.release.wait(remaining)
        finally:
            with self.lock:
                barrier.active = 0


BARRIERS = BarrierRegistry()


def _fingerprint(question: str) -> str:
    return hashlib.sha256(question.encode()).hexdigest()


def _block(source_id: str, quote: str, text: str | None = None) -> dict[str, Any]:
    return {
        "text": text or quote,
        "source_ids": [source_id],
        "support_quotes": [{"source_id": source_id, "quote": quote}],
    }


def _analysis(content: dict[str, Any]) -> dict[str, Any]:
    item = content["items"][0]
    if "analysis" in item:
        block = item["analysis"]["summary"][0]
    else:
        block = _block(
            item["source_id"],
            item["text"].strip()[:300],
            "Deterministic fixture analysis of the supplied source.",
        )
    return {"summary": [block], "insights": []}


def _selection(content: dict[str, Any]) -> dict[str, Any]:
    """Select a supplied catalog entry; this fixture does not establish AI quality."""
    evidence = content.get("evidence")
    if not isinstance(evidence, list) or not evidence or not isinstance(evidence[0], dict):
        raise ValueError("Fixture requires supplied analysis evidence")
    first = evidence[0]
    quote_id, passage = first.get("quote_id"), first.get("text")
    if (
        not isinstance(quote_id, str)
        or not quote_id.strip()
        or not isinstance(passage, str)
        or not passage.strip()
    ):
        raise ValueError("Fixture requires a quote ID and literal passage")
    return {"summary": [{"text": passage, "quote_ids": [quote_id]}], "insights": []}


def _planner(content: dict[str, Any]) -> dict[str, Any]:
    question = content["question"]
    lowered = question.lower()
    if re.search(r"\b(all|every|exhaustive|inventory)\b", lowered):
        intent = "exhaustive"
    elif re.search(r"\b(summary|overview)\b|summari[sz]e", lowered) and not re.search(
        r"\b(only|focused|specifically)\b", lowered
    ):
        intent = "overview"
    else:
        intent = "factual"
    recent = " ".join(message.get("content", "") for message in content.get("history", [])[-2:])
    query = (question + (" Recent conversation: " + recent if recent else ""))[:600]
    return {"query": query, "intent": intent}


def _answer(content: dict[str, Any]) -> dict[str, Any]:
    question = content["question"].lower()
    evidence = content.get("evidence", [])
    if re.search(r"cancel|penalt|termination", question):
        return {"evidence_status": "insufficient", "blocks": []}
    overview = content.get("intent") in {"overview", "exhaustive"}
    patterns = (
        r"Owner:\s*[^\n.]+" if re.search(r"\b(owner|owns|owned|who)\b", question) else None,
        r"(?:Renewal date:\s*|renews on\s+)\d{1,2}\s+[A-Za-z]+\s+\d{4}"
        if re.search(r"renew|\bwhen\b", question)
        else None,
        r"(?:Annual fee:\s*|annual fee is\s+)\d+(?:\.\d+)?\s+(?:USD|dollars)"
        if re.search(r"fee|cost|price", question)
        else None,
    )
    for source in evidence:
        text = source["text"]
        if overview:
            quote = text.strip()[:300]
            if quote:
                return {
                    "evidence_status": "answered" if content["complete_scope"] else "partial",
                    "blocks": [_block(source["source_id"], quote)],
                }
        for pattern in patterns:
            if pattern and (matched := re.search(pattern, text, re.IGNORECASE)):
                quote = matched.group().strip()
                return {
                    "evidence_status": "answered" if content["complete_scope"] else "partial",
                    "blocks": [_block(source["source_id"], quote)],
                }
    return {"evidence_status": "insufficient", "blocks": []}


def _answer_selection(content: dict[str, Any]) -> dict[str, Any]:
    """Exercise the quote-selection wire contract, without claiming model quality."""
    flattened = dict(content)
    flattened["evidence"] = [
        {**source, "text": "".join(passage["text"] for passage in source["passages"])}
        for source in content.get("evidence", [])
    ]
    legacy = _answer(flattened)
    blocks = []
    for block in legacy["blocks"]:
        source_id = block["source_ids"][0]
        source = next(item for item in content["evidence"] if item["source_id"] == source_id)
        quote = block["support_quotes"][0]["quote"]
        selected = next((item for item in source["passages"] if quote in item["text"]), None)
        if selected is None:
            # Overview fixture text can span boundaries; a literal first passage
            # is sufficient for this deterministic orchestration fixture.
            selected = next(item for item in source["passages"] if "quote_id" in item)
            text = selected["text"]
        else:
            text = block["text"]
        blocks.append({"text": text, "quote_ids": [selected["quote_id"]]})
    return {"evidence_status": legacy["evidence_status"], "blocks": blocks}


# Deliberately limited fixture vocabulary: these literal rules exercise stage
# wiring, requested gaps and references, not general retrieval or semantic quality.
SOURCE_COMPONENTS = (
    ("Owner", r"\b(owner|owns|owned|who)\b", r"Owner:\s*[^\n.]+"),
    (
        "Renewal date",
        r"renew|\bwhen\b",
        r"(?:Renewal date:\s*|renews on\s+)\d{1,2}\s+[A-Za-z]+\s+\d{4}",
    ),
    (
        "Annual fee",
        r"fee|cost|price",
        r"(?:Annual fee:\s*|annual fee is\s+)\d+(?:\.\d+)?\s+(?:USD|dollars)",
    ),
    ("Cancellation terms", r"cancel|penalt|termination", None),
)


def _source_selection(content: dict[str, Any]) -> dict[str, Any]:
    """Select only the fixture's recognized requested source literals."""
    components = []
    for label, requested, literal in SOURCE_COMPONENTS:
        if not re.search(requested, content["question"], re.IGNORECASE):
            continue
        selected = next(
            (
                item["quote_id"]
                for item in content.get("evidence", [])
                if literal is not None and re.search(literal, item["text"], re.IGNORECASE)
            ),
            None,
        )
        components.append({"label": label, "quote_ids": [selected] if selected else []})
    supported = sum(bool(item["quote_ids"]) for item in components)
    status = (
        "insufficient"
        if not supported
        else "partial"
        if supported < len(components)
        else "answered"
    )
    return {"evidence_status": status, "components": components}


def _source_composition(content: dict[str, Any]) -> dict[str, Any]:
    """Copy the requested fixture literal from each frozen component selection."""
    evidence = {item["quote_id"]: item["text"] for item in content.get("evidence", [])}
    patterns = {label: literal for label, _, literal in SOURCE_COMPONENTS}
    parts = []
    for component in content["components"]:
        literal = patterns.get(component["label"])
        selected = next(
            (
                (quote_id, match.group().strip())
                for quote_id in component["quote_ids"]
                if literal is not None
                and (match := re.search(literal, evidence.get(quote_id, ""), re.IGNORECASE))
            ),
            None,
        )
        parts.append(
            {
                "component_id": component["component_id"],
                "status": "supported" if selected else "unsupported",
                "quote_ids": [selected[0]] if selected else [],
                "text": selected[1] if selected else "",
            }
        )
    return {"parts": parts}


def _review(content: dict[str, Any]) -> dict[str, Any]:
    """Bind literal fixture claims only; never simulate a semantic quality judge.

    Fixture generation returns source substrings, so matching those substrings
    exercises claim identities, provider wiring and reference projection. A real
    inference deliberately needs live-provider evaluation rather than this stub.
    """
    decisions = []
    for claim in content["claims"]:
        text = claim["text"].strip()
        matches = [
            item["quote_id"] for item in content["evidence"] if text and text in item["text"]
        ][:8]
        decisions.append(
            {
                "claim_id": claim["claim_id"],
                "supported": bool(matches),
                "quote_ids": matches,
                "redundant_with": [],
            }
        )
    return {
        "required_qualifications": [],
        "decisions": decisions,
        "scope_complete": all(item["supported"] for item in decisions),
    }


def _analysis_review(content: dict[str, Any]) -> dict[str, Any]:
    """Assess fixture copies conservatively without inventing useful conclusions."""
    result = _review(content)
    decisions = {item["claim_id"]: item for item in result["decisions"]}
    summary = " ".join(item["text"] for item in content["claims"] if item["kind"] == "summary")
    candidates: dict[int, list[dict[str, Any]]] = {}
    for claim in content["claims"]:
        if claim["kind"] == "insights":
            candidates.setdefault(claim["block_index"], []).append(claim)
    result["insight_assessments"] = [
        {
            "block_index": index,
            "classification": "no_supported_conclusion"
            if any(not decisions[item["claim_id"]]["supported"] for item in claims)
            else "repeats_meaning"
            if all(item["text"].strip() in summary for item in claims)
            else "trivial_repackaging",
            "conclusion_claim_ids": [],
            "premise_quote_ids": [],
        }
        for index, claims in sorted(candidates.items())
    ]
    return result


def _chat_review(content: dict[str, Any]) -> dict[str, Any]:
    """Use the same small source-fact fixture vocabulary as answer generation.

    Identify expected source facts before mapping literal draft coverage. This
    exercises the current wire contract, not general semantic relevance or AI
    coverage judgments. Unsupported fixture questions conservatively abstain.
    """
    expected = _answer({**content, "intent": _planner(content)["intent"], "complete_scope": True})
    result = _review(content)
    result.pop("required_qualifications")
    claims = {item["claim_id"]: item for item in content["claims"]}
    aspects = []
    relevant = []
    for block in expected["blocks"]:
        quote = block["support_quotes"][0]["quote"]
        passage = next(
            item
            for item in content["evidence"]
            if item["source_id"] == block["source_ids"][0] and quote in item["text"]
        )
        covering = [
            item["claim_id"]
            for item in result["decisions"]
            if item["supported"]
            and passage["quote_id"] in item["quote_ids"]
            and (
                quote in claims[item["claim_id"]]["text"]
                or claims[item["claim_id"]]["text"].strip() in quote
            )
        ][:8]
        relevant.extend(covering)
        label = re.sub(r"[\r\n\[\]]", " ", block["text"]).strip()[:120]
        aspects.append(
            {
                "aspect": label,
                "quote_ids": [passage["quote_id"]],
                "conveyed_by_claim_ids": covering,
            }
        )
    # An aspect supplement cannot rescue a draft without supported relevant prose.
    answerable = bool(relevant)
    result.update(
        required_aspects=aspects if answerable else [],
        scope_complete=answerable,
        answerability="answered" if answerable else "insufficient",
        relevant_claim_ids=list(dict.fromkeys(relevant)),
        unanswered_parts=[],
    )
    return result


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass

    def reply(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body, ensure_ascii=False).encode()
        with suppress(BrokenPipeError, ConnectionResetError):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    def authorized(self) -> bool:
        if self.headers.get("Authorization") == TEST_AUTHORIZATION:
            return True
        self.reply(
            403,
            {
                "error": {
                    "message": "Fixture requires its test-only key",
                    "type": "invalid_request_error",
                }
            },
        )
        return False

    def do_GET(self) -> None:
        if self.path == "/health":
            self.reply(200, {"fixture": True, "barrier_version": 1})
        elif self.path.startswith("/barriers/"):
            if not self.authorized():
                return
            try:
                barrier = BARRIERS.snapshot(self.path.removeprefix("/barriers/"))
            except ValueError:
                self.reply(422, {"error": {"message": "Invalid barrier nonce"}})
                return
            self.reply(200 if barrier is not None else 404, {"fixture": True, "barrier": barrier})
        elif self.path == "/stats" and self.authorized():
            with LOCK:
                body = {
                    "counts": dict(COUNTS),
                    "reasoning_efforts": dict(REASONING_COUNTS),
                    "planner_questions": dict(PLANNER_QUESTIONS),
                    "answer_questions": dict(ANSWER_QUESTIONS),
                }
            self.reply(200, body)
        elif self.path != "/stats":
            self.reply(404, {"fixture": True})

    def do_POST(self) -> None:
        if not self.authorized():
            return
        size = int(self.headers.get("Content-Length", "0"))
        if size > 2 * 1024 * 1024:
            self.reply(413, {"error": {"message": "Fixture body too large"}})
            return
        try:
            request = json.loads(self.rfile.read(size))
        except (ValueError, UnicodeError):
            self.reply(400, {"error": {"message": "Invalid fixture JSON"}})
            return
        if self.path == "/barriers/arm":
            try:
                if not isinstance(request, dict) or set(request) - {
                    "nonce",
                    "phase",
                    "ttl_seconds",
                }:
                    raise ValueError
                nonce, phase = request.get("nonce"), request.get("phase")
                if not isinstance(nonce, str) or not isinstance(phase, str):
                    raise ValueError
                armed = BARRIERS.arm(nonce, phase, request.get("ttl_seconds", 60))
            except (ValueError, TypeError):
                self.reply(422, {"error": {"message": "Invalid fixture barrier"}})
                return
            self.reply(200, {"fixture": True, "barrier": armed})
            return
        if self.path.startswith("/barriers/") and self.path.endswith("/release"):
            try:
                barrier = BARRIERS.release(self.path[len("/barriers/") : -len("/release")])
            except ValueError:
                self.reply(422, {"error": {"message": "Invalid barrier nonce"}})
                return
            self.reply(200 if barrier is not None else 404, {"fixture": True, "barrier": barrier})
            return
        if self.path == "/v1/embeddings":
            with LOCK:
                COUNTS["embeddings"] += 1
            rows = []
            for index, text in enumerate(request["input"]):
                vector = [0.0] * 1536
                for word in re.findall(r"\w+", text.lower()):
                    bucket = (
                        int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big") % 1536
                    )
                    vector[bucket] += 1
                norm = math.sqrt(sum(value * value for value in vector)) or 1
                rows.append(
                    {
                        "index": index,
                        "object": "embedding",
                        "embedding": [value / norm for value in vector],
                    }
                )
            self.reply(
                200,
                {
                    "object": "list",
                    "model": request["model"],
                    "data": rows,
                    "usage": {"prompt_tokens": 20, "total_tokens": 20},
                },
            )
            return
        if self.path == "/v1/chat/completions":
            content = json.loads(request["messages"][-1]["content"])
            schema = request.get("response_format", {}).get("json_schema", {}).get("name", "")
            effort = request.get("reasoning_effort")
            safe_effort = (
                effort
                if isinstance(effort, str) and effort in {"none", "low"}
                else "missing"
                if effort is None
                else "other"
            )
            safe_schema = schema if schema in REASONING_SCHEMAS else "other"
            with LOCK:
                COUNTS[schema] += 1
                # Fixed schema/effort categories bound cardinality and retain no content.
                REASONING_COUNTS[f"{safe_schema}:{safe_effort}"] += 1
                if schema == "PlannerOutput":
                    PLANNER_QUESTIONS[_fingerprint(content["question"])] += 1
                elif schema in {
                    "AnswerOutput",
                    "AnswerSelectionOutput",
                    "ExhaustiveSelectionOutput",
                    "SourceSelectionOutput",
                }:
                    ANSWER_QUESTIONS[_fingerprint(content["question"])] += 1
            if schema == "PlannerOutput":
                if "[fixture quota]" in content["question"]:
                    self.reply(
                        429,
                        {
                            "error": {
                                "message": "Fixture account quota exhausted",
                                "type": "insufficient_quota",
                                "code": "insufficient_quota",
                            }
                        },
                    )
                    return
                if "[fixture slow]" in content["question"]:
                    time.sleep(0.5)
                result = _planner(content)
            elif schema == "AnswerOutput":
                BARRIERS.wait("answer", content)
                result = _answer(content)
            elif schema == "AnswerSelectionOutput":
                BARRIERS.wait("answer", content)
                result = _answer_selection(content)
            elif schema == "SourceSelectionOutput":
                BARRIERS.wait("answer", content)
                result = _source_selection(content)
            elif schema == "CompositionOutput":
                result = _source_composition(content)
            elif schema == "ExhaustiveSelectionOutput":
                BARRIERS.wait("answer", content)
                result = {
                    "quote_ids": [
                        passage["quote_id"]
                        for source in content.get("evidence", [])
                        for passage in source.get("passages", [])
                    ][:3]
                }
            elif schema == "SelectionOutput":
                BARRIERS.wait("analysis", content)
                try:
                    result = _selection(content)
                except ValueError:
                    self.reply(400, {"error": {"message": "Invalid fixture analysis evidence"}})
                    return
            elif schema == "InsightCandidates":
                # Literal fixture summaries already state the source facts. Do
                # not fabricate a novel inference just to populate this field.
                result = {"insights": []}
            elif schema == "ChatReviewOutput":
                result = _chat_review(content)
            elif schema == "AnalysisReviewOutput":
                result = _analysis_review(content)
            elif schema == "ReviewOutput":
                result = _review(content)
            elif schema == "AnalysisOutput" or "items" in content:
                BARRIERS.wait("analysis", content)
                result = _analysis(content)
            else:
                self.reply(
                    400,
                    {
                        "error": {
                            "message": "Unsupported fixture schema",
                            "type": "invalid_request_error",
                        }
                    },
                )
                return
            self.reply(
                200,
                {
                    "id": "fixture-response",
                    "object": "chat.completion",
                    "created": 0,
                    "model": request["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(result),
                                "refusal": None,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 60, "total_tokens": 160},
                },
            )
            return
        self.reply(404, {"error": {"message": "Fixture does not implement this route"}})


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 4010), Handler).serve_forever()
