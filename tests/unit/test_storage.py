"""Real temporary filesystem and container checks; no parsers or services."""

import io
import os
import stat
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from app.adapters.storage import (
    DOCX_MEDIA_TYPE,
    PDF_MEDIA_TYPE,
    InputProcessingError,
    LocalFileStorage,
    validate_upload,
)
from app.core.config import Settings
from app.domain.documents import ProcessingError

pytestmark = pytest.mark.unit


@pytest.fixture
def settings() -> Callable[..., Settings]:
    def create(**overrides: Any) -> Settings:
        return Settings(_env_file=None, openai_api_key="test-placeholder", **overrides)

    return create


def docx_bytes(extra: dict[str, bytes] | None = None) -> bytes:
    output = io.BytesIO()
    parts = {
        "[Content_Types].xml": b"<Types/>",
        "_rels/.rels": b"<Relationships/>",
        "word/document.xml": b"<document/>",
        **(extra or {}),
    }
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name, value in parts.items():
            archive.writestr(name, value)
    return output.getvalue()


def test_staging_finalize_round_trip_and_cleanup(tmp_path: Path) -> None:
    storage = LocalFileStorage(tmp_path)
    key, path = storage.create_staging()
    assert str(UUID(key)) == key
    assert path == tmp_path / "staging" / key
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    with storage.open_staging(key) as stream:
        stream.write(b"original\x00\xff")
    assert storage.finalize(key) == key
    assert not path.exists()
    assert storage.read(key) == b"original\x00\xff"
    with storage.open(key) as stream:
        assert stream.read() == b"original\x00\xff"
    with pytest.raises(ProcessingError, match="unavailable"):
        storage.finalize(key)
    storage.delete(key)
    storage.delete(key)
    second_key, second_path = storage.create_staging()
    storage.discard_staging(second_key)
    storage.discard_staging(second_key)
    assert not second_path.exists()


@pytest.mark.parametrize("key", ["../escape", "/tmp/escape", "", "a" * 36, "x" * 1000])
def test_storage_rejects_nonopaque_keys(tmp_path: Path, key: str) -> None:
    storage = LocalFileStorage(tmp_path)
    for operation in (
        storage.open,
        storage.open_staging,
        storage.finalize,
        storage.delete,
        storage.discard_staging,
    ):
        with pytest.raises(ProcessingError) as captured:
            operation(key)
        assert captured.value.code == "INVALID_STORAGE_KEY"


def test_original_and_staging_symlinks_never_followed(tmp_path: Path) -> None:
    storage = LocalFileStorage(tmp_path)
    outside = tmp_path / "outside"
    outside.write_bytes(b"unchanged")
    key = str(uuid4())
    (tmp_path / "originals" / key).symlink_to(outside)
    with pytest.raises(ProcessingError):
        storage.read(key)
    storage.delete(key)
    assert outside.read_bytes() == b"unchanged"
    staged_key, staged_path = storage.create_staging()
    staged_path.unlink()
    staged_path.symlink_to(outside)
    with pytest.raises(ProcessingError):
        storage.open_staging(staged_key)
    with pytest.raises(ProcessingError):
        storage.finalize(staged_key)
    storage.discard_staging(staged_key)
    assert outside.read_bytes() == b"unchanged"


def test_directory_symlinks_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    (root / "staging").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ProcessingError):
        LocalFileStorage(root)
    assert list(outside.iterdir()) == []


def test_finalize_never_replaces_existing_original(tmp_path: Path) -> None:
    storage = LocalFileStorage(tmp_path)
    key, path = storage.create_staging()
    path.write_bytes(b"new")
    original = tmp_path / "originals" / key
    original.write_bytes(b"existing")
    with pytest.raises(ProcessingError) as captured:
        storage.finalize(key)
    assert captured.value.code == "STORAGE_CONFLICT"
    assert original.read_bytes() == b"existing"
    assert path.read_bytes() == b"new"


def test_failed_finalize_removes_reservation_and_preserves_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = LocalFileStorage(tmp_path)
    key, path = storage.create_staging()
    path.write_bytes(b"original")

    def fail_replace(*args: Any, **kwargs: Any) -> None:
        raise OSError("Simulated filesystem failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(ProcessingError) as captured:
        storage.finalize(key)
    assert captured.value.code == "STORAGE_ERROR"
    assert captured.value.retryable is True
    assert path.read_bytes() == b"original"
    assert not (tmp_path / "originals" / key).exists()


def test_storage_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    storage = LocalFileStorage(tmp_path)
    key = str(uuid4())
    os.mkfifo(tmp_path / "originals" / key)
    with pytest.raises(ProcessingError) as captured:
        storage.open(key)
    assert captured.value.code == "INVALID_STORAGE_FILE"


@pytest.mark.parametrize(
    "filename,hint,payload,expected",
    [
        ("report.PDF", "application/pdf", b"%PDF-1.7\n", PDF_MEDIA_TYPE),
        ("report.pdf", "application/octet-stream", b"%PDF-1.7\n", PDF_MEDIA_TYPE),
        ("report.pdf", None, b"%PDF-1.7\n", PDF_MEDIA_TYPE),
        ("report.docx", DOCX_MEDIA_TYPE, docx_bytes(), DOCX_MEDIA_TYPE),
        ("report.docx", "application/zip", docx_bytes(), DOCX_MEDIA_TYPE),
    ],
)
def test_supported_hints_and_signatures(
    tmp_path: Path,
    settings: Callable[..., Settings],
    filename: str,
    hint: str | None,
    payload: bytes,
    expected: str,
) -> None:
    path = tmp_path / "staged"
    path.write_bytes(payload)
    assert validate_upload(path, filename, hint, settings()) == expected


@pytest.mark.parametrize(
    "filename,hint,payload,code",
    [
        ("file.html", "text/html", b"<html/>", "UNSUPPORTED_FORMAT"),
        ("file.pdf", DOCX_MEDIA_TYPE, b"%PDF-1.7", "MEDIA_TYPE_MISMATCH"),
        ("file.pdf", PDF_MEDIA_TYPE, docx_bytes(), "INVALID_DOCUMENT"),
        ("file.docx", DOCX_MEDIA_TYPE, b"%PDF-1.7", "INVALID_DOCUMENT"),
        ("file.pdf", PDF_MEDIA_TYPE, b"", "INVALID_DOCUMENT"),
        ("file.docx", DOCX_MEDIA_TYPE, b"PK\x03\x04broken", "INVALID_DOCUMENT"),
    ],
)
def test_disguised_and_corrupt_uploads(
    tmp_path: Path,
    settings: Callable[..., Settings],
    filename: str,
    hint: str,
    payload: bytes,
    code: str,
) -> None:
    path = tmp_path / "staged"
    path.write_bytes(payload)
    with pytest.raises(InputProcessingError) as captured:
        validate_upload(path, filename, hint, settings())
    assert captured.value.code == code
    assert captured.value.retryable is False


def test_docx_structure_and_metadata_limits(
    tmp_path: Path, settings: Callable[..., Settings]
) -> None:
    path = tmp_path / "staged"
    path.write_bytes(docx_bytes())
    with pytest.raises(InputProcessingError) as captured:
        validate_upload(path, "file.docx", DOCX_MEDIA_TYPE, settings(max_docx_entries=2))
    assert captured.value.code == "DOCX_ENTRY_LIMIT"
    forged = bytearray(docx_bytes())
    central_entry = forged.index(b"PK\x01\x02")
    struct.pack_into("<I", forged, central_entry + 24, 200_000_000)
    path.write_bytes(forged)
    with pytest.raises(InputProcessingError) as captured:
        validate_upload(path, "file.docx", DOCX_MEDIA_TYPE, settings())
    assert captured.value.code == "DOCX_EXPANSION_LIMIT"
    path.write_bytes(docx_bytes({"../outside": b"bad"}))
    with pytest.raises(InputProcessingError) as captured:
        validate_upload(path, "file.docx", DOCX_MEDIA_TYPE, settings())
    assert captured.value.code == "INVALID_DOCUMENT"


def test_zip_renamed_docx_is_rejected(tmp_path: Path, settings: Callable[..., Settings]) -> None:
    path = tmp_path / "staged"
    output = io.BytesIO()
    with ZipFile(output, "w") as archive:
        archive.writestr("not-a-document.txt", "hello")
    path.write_bytes(output.getvalue())
    with pytest.raises(InputProcessingError) as captured:
        validate_upload(path, "file.docx", DOCX_MEDIA_TYPE, settings())
    assert captured.value.code == "INVALID_DOCUMENT"


def test_docx_validation_does_not_decompress_parts(
    tmp_path: Path, settings: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "staged"
    path.write_bytes(docx_bytes())

    def forbidden_open(*args: Any, **kwargs: Any) -> None:
        pytest.fail("API container validation must not decompress document parts")

    monkeypatch.setattr(ZipFile, "open", forbidden_open)
    assert validate_upload(path, "file.docx", DOCX_MEDIA_TYPE, settings()) == DOCX_MEDIA_TYPE


def test_file_size_limit(tmp_path: Path, settings: Callable[..., Settings]) -> None:
    path = tmp_path / "staged"
    path.write_bytes(b"%PDF-1.7")
    assert validate_upload(path, "file.pdf", PDF_MEDIA_TYPE, settings(max_upload_bytes=8))
    with pytest.raises(InputProcessingError) as captured:
        validate_upload(path, "file.pdf", PDF_MEDIA_TYPE, settings(max_upload_bytes=7))
    assert captured.value.code == "UPLOAD_TOO_LARGE"
