"""Exercise storage readiness against real temporary filesystem operations."""

from pathlib import Path

import pytest

from app.core import readiness
from app.core.readiness import storage_ready

pytestmark = pytest.mark.unit


def test_writable_empty_directory_is_ready_without_leftover_probe(tmp_path: Path) -> None:
    assert storage_ready(tmp_path) is True
    assert list(tmp_path.iterdir()) == []


def test_missing_directory_is_not_ready_and_is_not_created(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    assert storage_ready(missing) is False
    assert not missing.exists()
    assert list(tmp_path.iterdir()) == []


def test_failed_read_reports_unready_and_removes_created_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed_probes: list[Path] = []

    def fail_read(probe: Path) -> bytes:
        assert probe.parent == tmp_path
        assert probe.name.startswith(".health-")
        assert probe.is_file()
        observed_probes.append(probe)
        raise OSError("Simulated storage read failure")

    monkeypatch.setattr(Path, "read_bytes", fail_read)
    assert storage_ready(tmp_path) is False
    assert len(observed_probes) == 1
    assert list(tmp_path.iterdir()) == []


def test_existing_unrelated_file_is_preserved(tmp_path: Path) -> None:
    existing = tmp_path / "existing-original.pdf"
    original = b"original document bytes\x00\xff"
    existing.write_bytes(original)

    assert storage_ready(tmp_path) is True
    assert existing.read_bytes() == original
    assert list(tmp_path.iterdir()) == [existing]


def test_failed_delete_is_not_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_delete(probe: Path, missing_ok: bool = False) -> None:
        raise OSError("Simulated delete failure")

    monkeypatch.setattr(Path, "unlink", fail_delete)
    assert storage_ready(tmp_path) is False


def test_failed_exclusive_create_never_deletes_existing_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(readiness, "uuid4", lambda: "fixed-probe-id")
    existing = tmp_path / ".health-fixed-probe-id"
    existing.write_bytes(b"preserve existing bytes")
    assert storage_ready(tmp_path) is False
    assert existing.read_bytes() == b"preserve existing bytes"
