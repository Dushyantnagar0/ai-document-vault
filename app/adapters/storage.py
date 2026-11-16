"""Blocking local storage and bounded upload-container validation.

Callers offload filesystem operations and stream uploads with their own byte/time
limits. No original filename is used as a storage path; PDF parsing and actual
DOCX expansion checks belong to the worker.
"""

import errno
import fcntl
import math
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import BinaryIO, Literal, Self
from uuid import UUID, uuid4
from zipfile import BadZipFile, ZipFile

from app.core.config import Settings
from app.domain.documents import ProcessingError

PDF_MEDIA_TYPE = "application/pdf"
DOCX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
StorageArea = Literal["staging", "originals"]


@dataclass(frozen=True)
class StaleFile:
    area: StorageArea
    key: str
    mtime: float
    device: int
    inode: int
    # Preserve the scan's age qualification and precise timestamp across races.
    older_than: float = field(kw_only=True, repr=False)
    mtime_ns: int = field(kw_only=True, repr=False)


def _matches_candidate(metadata: os.stat_result, candidate: StaleFile) -> bool:
    return (
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_dev == candidate.device
        and metadata.st_ino == candidate.inode
        and metadata.st_mtime == candidate.mtime
        and metadata.st_mtime_ns == candidate.mtime_ns
        and metadata.st_mtime < candidate.older_than
    )


class LockedFile:
    """Exclusive candidate lock; the caller must recheck DB references before delete."""

    def __init__(self, candidate: StaleFile, descriptor: int, directory_descriptor: int) -> None:
        self.candidate = candidate
        self._descriptor: int | None = descriptor
        self._directory_descriptor: int | None = directory_descriptor

    def delete(self) -> bool:
        """Unlink only the unchanged aged inode; return false if closed/changed/missing."""
        if self._descriptor is None or self._directory_descriptor is None:
            return False
        with _storage_errors():
            try:
                named = os.stat(
                    self.candidate.key,
                    dir_fd=self._directory_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return False
            if not _matches_candidate(named, self.candidate) or not _matches_candidate(
                os.fstat(self._descriptor), self.candidate
            ):
                return False
            try:
                os.unlink(self.candidate.key, dir_fd=self._directory_descriptor)
            except FileNotFoundError:
                return False
            return True

    def close(self) -> None:
        descriptor, directory = self._descriptor, self._directory_descriptor
        self._descriptor = self._directory_descriptor = None
        if descriptor is not None:
            with suppress(OSError):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            with suppress(OSError):
                os.close(descriptor)
        if directory is not None:
            with suppress(OSError):
                os.close(directory)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class InputProcessingError(ProcessingError):
    """An upload's format/content/limits failed before durable acceptance."""


def _key(value: str) -> str:
    if not isinstance(value, str) or len(value) != 36:
        raise ProcessingError("INVALID_STORAGE_KEY", "Stored file key is invalid.")
    try:
        valid = str(UUID(value)) == value
    except ValueError:
        valid = False
    if not valid:
        raise ProcessingError("INVALID_STORAGE_KEY", "Stored file key is invalid.")
    return value


@contextmanager
def _storage_errors() -> Iterator[None]:
    try:
        yield
    except FileNotFoundError:
        raise ProcessingError("STORAGE_NOT_FOUND", "Stored file is unavailable.") from None
    except OSError:
        raise ProcessingError(
            "STORAGE_ERROR", "Local file storage is unavailable.", retryable=True
        ) from None


class LocalFileStorage:
    """Generated UUID keys in private staging/originals directories.

    The returned staging Path is for trusted in-process validation only. Use
    open_staging to stream bytes safely; public open/read/delete address originals.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        if not self.root.is_absolute():
            raise ValueError("Local storage root must be absolute")
        with _storage_errors():
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
            root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                for area in ("staging", "originals"):
                    with suppress(FileExistsError):
                        os.mkdir(area, mode=0o700, dir_fd=root_fd)
                    child_fd = os.open(
                        area, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd
                    )
                    os.close(child_fd)
            finally:
                os.close(root_fd)

    @contextmanager
    def _directory(self, area: str) -> Iterator[int]:
        with _storage_errors():
            root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                directory_fd = os.open(
                    area, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd
                )
                try:
                    yield directory_fd
                finally:
                    os.close(directory_fd)
            finally:
                os.close(root_fd)

    def create_staging(self) -> tuple[str, Path]:
        key = str(uuid4())
        with self._directory("staging") as directory_fd:
            descriptor = os.open(
                key,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory_fd,
            )
            os.close(descriptor)
        return key, self.root / "staging" / key

    def open_staging(self, key: str) -> BinaryIO:
        """Lock before writing; retain the stream through finalization and DB commit.

        Closing the stream releases its inode lock, including after rename into
        originals. Opening itself neither truncates nor recreates the file.
        """
        return self._open("staging", key, writable=True)

    def _open(self, area: str, key: str, *, writable: bool = False) -> BinaryIO:
        _key(key)
        with self._directory(area) as directory_fd:
            descriptor = os.open(
                key,
                (os.O_RDWR if writable else os.O_RDONLY) | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=directory_fd,
            )
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    raise ProcessingError("INVALID_STORAGE_FILE", "Stored file must be regular.")
                if writable:
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                    # A collector may have won before this open acquired its lock.
                    named = os.stat(key, dir_fd=directory_fd, follow_symlinks=False)
                    if (
                        not stat.S_ISREG(named.st_mode)
                        or named.st_dev != metadata.st_dev
                        or named.st_ino != metadata.st_ino
                    ):
                        raise ProcessingError("STORAGE_NOT_FOUND", "Staged file was replaced.")
                return os.fdopen(descriptor, "r+b" if writable else "rb")
            except BaseException:
                os.close(descriptor)
                raise

    def finalize(self, staging_key: str) -> str:
        key = _key(staging_key)
        with self._directory("staging") as source_fd, self._directory("originals") as target_fd:
            if not stat.S_ISREG(os.stat(key, dir_fd=source_fd, follow_symlinks=False).st_mode):
                raise ProcessingError("INVALID_STORAGE_FILE", "Staged file must be regular.")
            # Reserve the opaque destination exclusively so an existing original
            # can never be overwritten, then atomically replace this reservation.
            try:
                reservation_fd = os.open(
                    key,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=target_fd,
                )
            except FileExistsError:
                raise ProcessingError(
                    "STORAGE_CONFLICT", "Stored file key is already finalized."
                ) from None
            os.close(reservation_fd)
            try:
                os.replace(key, key, src_dir_fd=source_fd, dst_dir_fd=target_fd)
            except OSError:
                with suppress(OSError):
                    os.unlink(key, dir_fd=target_fd)
                raise
        return key

    def open(self, key: str) -> BinaryIO:
        return self._open("originals", key)

    def read(self, key: str) -> bytes:
        with _storage_errors(), self.open(key) as stream:
            return stream.read()

    def _delete(self, area: str, key: str) -> None:
        _key(key)
        with self._directory(area) as directory_fd, suppress(FileNotFoundError):
            # unlink never follows a final symlink and cannot traverse a UUID key.
            os.unlink(key, dir_fd=directory_fd)

    def delete(self, key: str) -> None:
        self._delete("originals", key)

    def discard_staging(self, key: str) -> None:
        self._delete("staging", key)

    def old_candidates(self, area: StorageArea, older_than: float) -> tuple[StaleFile, ...]:
        """List metadata only; unknown names and nonregular entries are preserved."""
        if area not in {"staging", "originals"}:
            raise ValueError("Cleanup area must be staging or originals")
        if not math.isfinite(older_than):
            raise ValueError("Cleanup cutoff must be a finite Unix timestamp")
        candidates = []
        with self._directory(area) as directory_fd, os.scandir(directory_fd) as entries:
            for entry in entries:
                try:
                    key = _key(entry.name)
                    metadata = entry.stat(follow_symlinks=False)
                except (ProcessingError, OSError):
                    continue
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_mtime >= older_than:
                    continue
                candidates.append(
                    StaleFile(
                        area,
                        key,
                        metadata.st_mtime,
                        metadata.st_dev,
                        metadata.st_ino,
                        older_than=older_than,
                        mtime_ns=metadata.st_mtime_ns,
                    )
                )
        return tuple(sorted(candidates, key=lambda candidate: (candidate.mtime, candidate.key)))

    def lock_candidate(self, candidate: StaleFile) -> LockedFile | None:
        """Take a nonblocking inode lock and revalidate scan identity/age/name."""
        if candidate.area not in {"staging", "originals"}:
            raise ValueError("Cleanup area must be staging or originals")
        key = _key(candidate.key)
        if not math.isfinite(candidate.older_than):
            raise ValueError("Cleanup cutoff must be a finite Unix timestamp")
        with self._directory(candidate.area) as directory_fd:
            try:
                descriptor = os.open(
                    key, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd
                )
            except OSError as error:
                if error.errno in {errno.ENOENT, errno.ENOTDIR, errno.ELOOP}:
                    return None
                raise
            retained = False
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    return None
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return None
                if not _matches_candidate(os.fstat(descriptor), candidate):
                    return None
                try:
                    named = os.stat(key, dir_fd=directory_fd, follow_symlinks=False)
                except FileNotFoundError:
                    return None
                if not _matches_candidate(named, candidate):
                    return None
                locked = LockedFile(candidate, descriptor, os.dup(directory_fd))
                retained = True
                return locked
            finally:
                if not retained:
                    os.close(descriptor)


def validate_upload(path: Path, filename: str, media_type: str | None, settings: Settings) -> str:
    """Check hints/signature and ZIP metadata without parsing document content."""
    suffix = Path(filename).suffix.lower()
    expected = {".pdf": PDF_MEDIA_TYPE, ".docx": DOCX_MEDIA_TYPE}.get(suffix)
    if expected is None:
        raise InputProcessingError("UNSUPPORTED_FORMAT", "Upload a PDF or DOCX document.")
    hint = (media_type or "").partition(";")[0].strip().lower()
    allowed = {"", "application/octet-stream", expected}
    if expected == DOCX_MEDIA_TYPE:
        allowed.add("application/zip")
    if hint not in allowed:
        raise InputProcessingError(
            "MEDIA_TYPE_MISMATCH", "File media type does not match its PDF/DOCX extension."
        )
    with _storage_errors():
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise InputProcessingError("INVALID_DOCUMENT", "Upload must be a regular file.")
            if metadata.st_size > settings.max_upload_bytes:
                raise InputProcessingError(
                    "UPLOAD_TOO_LARGE", "Upload exceeds the file-size limit."
                )
            signature = stream.read(8)
            stream.seek(0)
            if expected == PDF_MEDIA_TYPE:
                if not signature.startswith(b"%PDF-"):
                    raise InputProcessingError(
                        "INVALID_DOCUMENT", "File content does not have a PDF signature."
                    )
            else:
                if not signature.startswith(b"PK\x03\x04"):
                    raise InputProcessingError(
                        "INVALID_DOCUMENT", "File content is not a DOCX ZIP container."
                    )
                _validate_docx_container(stream, settings)
    return expected


def _validate_docx_container(stream: BinaryIO, settings: Settings) -> None:
    try:
        with ZipFile(stream) as archive:
            entries = archive.infolist()
            if len(entries) > settings.max_docx_entries:
                raise InputProcessingError("DOCX_ENTRY_LIMIT", "DOCX has too many archive entries.")
            if sum(entry.file_size for entry in entries) > settings.max_docx_expanded_bytes:
                raise InputProcessingError(
                    "DOCX_EXPANSION_LIMIT", "DOCX expanded content exceeds the configured limit."
                )
            names = [entry.filename for entry in entries]
            if len(set(names)) != len(names):
                raise InputProcessingError("INVALID_DOCUMENT", "DOCX contains duplicate parts.")
            for entry in entries:
                part = PurePosixPath(entry.filename)
                if part.is_absolute() or ".." in part.parts or "\\" in entry.filename:
                    raise InputProcessingError(
                        "INVALID_DOCUMENT", "DOCX contains invalid part paths."
                    )
                if entry.flag_bits & 1:
                    raise InputProcessingError(
                        "INVALID_DOCUMENT", "Encrypted DOCX archives are unsupported."
                    )
            required = {"[Content_Types].xml", "_rels/.rels", "word/document.xml"}
            if not required.issubset(names):
                raise InputProcessingError(
                    "INVALID_DOCUMENT", "ZIP container is missing required DOCX document parts."
                )
    except (BadZipFile, EOFError, UnicodeError, NotImplementedError):
        raise InputProcessingError("INVALID_DOCUMENT", "DOCX ZIP container is corrupt.") from None
