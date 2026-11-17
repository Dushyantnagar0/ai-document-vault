"""Durable request reservations for opt-in experiments, never provider execution."""

import fcntl
import json
import os
import re
import stat
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

SAFE_LABEL = re.compile(r"[A-Za-z0-9_.:-]{1,80}\Z")
FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
TOKEN_KEYS = {"input_tokens", "output_tokens", "reasoning_tokens"}
METADATA_KEYS = TOKEN_KEYS | {"model", "outcome", "error_code"}


class BudgetError(ValueError):
    """Static messages deliberately exclude input, credentials and file contents."""


def _safe(value: Any) -> bool:
    return isinstance(value, str) and SAFE_LABEL.fullmatch(value) is not None


def _metadata(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and not (value.keys() - METADATA_KEYS)
        and all(
            (item is None or (type(item) is int and item >= 0))
            if key in TOKEN_KEYS
            else isinstance(item, str) and item in {"succeeded", "failed", "uncertain"}
            if key == "outcome"
            else _safe(item)
            for key, item in value.items()
        )
    )


def _require(condition: bool) -> None:
    if not condition:
        raise BudgetError("INVALID_BUDGET_STATE_OR_LIMIT")


class RequestBudget:
    def __init__(self, path: Path | str, limit: int = 100) -> None:
        if type(limit) is not int or limit < 1:
            raise BudgetError("INVALID_BUDGET_LIMIT")
        self.path, self.limit = Path(path), limit
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked():
            self._read()

    @contextmanager
    def _locked(self) -> Iterator[None]:
        try:
            descriptor = os.open(
                str(self.path) + ".lock",
                os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK,
                0o600,
            )
        except OSError:
            raise BudgetError("INVALID_BUDGET_LOCK") from None
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise BudgetError("INVALID_BUDGET_LOCK")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            os.close(descriptor)

    def _read(self) -> dict[str, Any]:
        try:
            descriptor = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            return {"version": 1, "limit": self.limit, "entries": []}
        except OSError:
            raise BudgetError("INVALID_BUDGET_PATH") from None
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_size > 1024 * 1024:
                raise BudgetError("INVALID_BUDGET_STATE")
            with os.fdopen(descriptor, "r", closefd=False) as stream:
                state = json.load(stream)
            _require(set(state) == {"version", "limit", "entries"})
            _require(type(state["version"]) is int and state["version"] == 1)
            _require(type(state["limit"]) is int and state["limit"] == self.limit)
            entries = state["entries"]
            _require(isinstance(entries, list) and len(entries) <= self.limit)
            identities = []
            for entry in entries:
                _require(set(entry) == {"id", "label", "fingerprint", "status", "metadata"})
                identities.append(str(UUID(entry["id"])))
                _require(identities[-1] == entry["id"] and _safe(entry["label"]))
                _require(FINGERPRINT.fullmatch(entry["fingerprint"]) is not None)
                _require(entry["status"] in {"reserved", "completed"})
                _require(_metadata(entry["metadata"]))
                _require(entry["status"] != "reserved" or not entry["metadata"])
                _require(entry["status"] != "completed" or "outcome" in entry["metadata"])
            _require(len(identities) == len(set(identities)))
            return cast(dict[str, Any], state)
        except (ValueError, KeyError, TypeError, AttributeError):
            raise BudgetError("INVALID_BUDGET_STATE_OR_LIMIT") from None
        finally:
            os.close(descriptor)

    def _write(self, state: dict[str, Any]) -> None:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(json.dumps(state, separators=(",", ":")).encode())
                stream.flush()
                os.fsync(stream.fileno())
            if self.path.is_symlink():
                raise BudgetError("INVALID_BUDGET_PATH")
            os.replace(temporary, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def reserve(self, label: str, request_fingerprint: str) -> str:
        if (
            not _safe(label)
            or not isinstance(request_fingerprint, str)
            or not FINGERPRINT.fullmatch(request_fingerprint)
        ):
            raise BudgetError("INVALID_BUDGET_RESERVATION")
        with self._locked():
            state = self._read()
            if len(state["entries"]) >= self.limit:
                raise BudgetError("REQUEST_BUDGET_EXHAUSTED")
            identifier = str(uuid4())
            state["entries"].append(
                {
                    "id": identifier,
                    "label": label,
                    "fingerprint": request_fingerprint,
                    "status": "reserved",
                    "metadata": {},
                }
            )
            self._write(state)
            return identifier

    def complete(self, identifier: str, metadata: dict[str, Any]) -> None:
        if not _metadata(metadata) or "outcome" not in metadata:
            raise BudgetError("INVALID_BUDGET_METADATA")
        with self._locked():
            state = self._read()
            entry = next((item for item in state["entries"] if item["id"] == identifier), None)
            if entry is None or (entry["status"] == "completed" and entry["metadata"] != metadata):
                raise BudgetError("UNKNOWN_OR_COMPLETED_RESERVATION")
            entry.update(status="completed", metadata=dict(metadata))
            self._write(state)

    def snapshot(self) -> dict[str, int]:
        with self._locked():
            used = len(self._read()["entries"])
            return {"used": used, "remaining": self.limit - used}


def _request_fingerprint(kwargs: dict[str, Any], profile: str) -> str:
    import hashlib

    recipe = {key: kwargs[key] for key in ("system", "content", "max_output_tokens")}
    recipe.update(schema=kwargs["schema"].model_json_schema(), profile=profile)
    return hashlib.sha256(
        json.dumps(recipe, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _observation(outcome: str, value: Any, *, failure: bool = False) -> dict[str, Any]:
    observed: dict[str, Any] = {"outcome": outcome}
    details = value.diagnostics.as_dict() if failure and value.diagnostics is not None else {}
    for key in TOKEN_KEYS:
        usage = details.get(key) if failure else getattr(value, key, None)
        if usage is None or (type(usage) is int and usage >= 0):
            observed[key] = usage
    model = getattr(value, "model", None)
    if _safe(model):
        observed["model"] = model
    if failure and _safe(value.code):
        observed["error_code"] = value.code
    return observed


class _BudgetedProvider:
    def __init__(
        self, delegate: Any, budget: RequestBudget, label_prefix: str, profile: str = "none"
    ) -> None:
        if not _safe(label_prefix) or not _safe(profile):
            raise BudgetError("INVALID_BUDGET_PROVIDER_LABEL")
        self.delegate, self.budget = delegate, budget
        self.label, self.profile = label_prefix, profile

    def _reserve(self, kwargs: dict[str, Any]) -> str:
        return self.budget.reserve(self.label, _request_fingerprint(kwargs, self.profile))


class BudgetedGenerationProvider(_BudgetedProvider):
    """One reservation per explicit generation invocation, including retries."""

    def generate(self, **kwargs: Any) -> Any:
        from app.domain.documents import ProcessingError

        identifier = self._reserve(kwargs)
        try:
            result = self.delegate.generate(**kwargs)
        except ProcessingError as error:
            self.budget.complete(identifier, _observation("failed", error, failure=True))
            raise
        self.budget.complete(identifier, _observation("succeeded", result))
        return result


class AsyncBudgetedGenerationProvider(_BudgetedProvider):
    """Async counterpart; cancellation leaves a conservatively reserved slot."""

    async def generate(self, **kwargs: Any) -> Any:
        from app.domain.documents import ProcessingError

        identifier = self._reserve(kwargs)
        try:
            result = await self.delegate.generate(**kwargs)
        except ProcessingError as error:
            self.budget.complete(identifier, _observation("failed", error, failure=True))
            raise
        self.budget.complete(identifier, _observation("succeeded", result))
        return result


class AsyncBudgetedEmbeddingProvider(_BudgetedProvider):
    """Count each embedding request, retaining hashes and safe usage only."""

    async def embed_many(self, texts: Sequence[str], *, timeout_seconds: float) -> Any:
        import hashlib

        from app.domain.documents import ProcessingError

        request_hash = hashlib.sha256(
            json.dumps(
                {"texts": list(texts), "profile": self.profile},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        identifier = self.budget.reserve(self.label, request_hash)
        try:
            result = await self.delegate.embed_many(texts, timeout_seconds=timeout_seconds)
        except ProcessingError as error:
            self.budget.complete(identifier, _observation("failed", error, failure=True))
            raise
        self.budget.complete(identifier, _observation("succeeded", result))
        return result
