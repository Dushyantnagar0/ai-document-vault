"""Cooperative writer/collector locks on real temporary files, without any DB/GC."""

import os
import time
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

from app.adapters.storage import LocalFileStorage
from app.domain.documents import ProcessingError

pytestmark = pytest.mark.unit


def age(path: Path) -> float:
    cutoff = time.time() - 3600
    old = cutoff - 3600
    os.utime(path, (old, old), follow_symlinks=False)
    return cutoff


def test_only_canonical_old_regular_files_are_candidates(tmp_path: Path) -> None:
    storage = LocalFileStorage(tmp_path)
    old_key, old_path = storage.create_staging()
    cutoff = age(old_path)
    _, fresh_path = storage.create_staging()
    unknown = old_path.parent / "not-a-storage-key.pdf"
    unknown.write_text("preserve")
    age(unknown)
    directory = old_path.parent / str(uuid4())
    directory.mkdir()
    age(directory)
    link = old_path.parent / str(uuid4())
    link.symlink_to(old_path)
    fifo = old_path.parent / str(uuid4())
    os.mkfifo(fifo)
    age(fifo)
    candidates = storage.old_candidates("staging", cutoff)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.area == "staging" and candidate.key == old_key
    assert candidate.device == old_path.stat().st_dev
    assert candidate.inode == old_path.stat().st_ino
    assert candidate.mtime < cutoff
    assert fresh_path.exists() and unknown.read_text() == "preserve"
    assert link.is_symlink() and directory.is_dir()


@pytest.mark.parametrize("finalize", [False, True])
def test_active_writer_is_skipped_until_close_even_after_finalize(
    tmp_path: Path,
    finalize: bool,
) -> None:
    storage = LocalFileStorage(tmp_path)
    key, path = storage.create_staging()
    stream = storage.open_staging(key)
    try:
        stream.write(b"original")
        stream.flush()
        cutoff = age(path)
        area = "staging"
        if finalize:
            storage.finalize(key)
            path = tmp_path / "originals" / key
            area = "originals"
        (candidate,) = storage.old_candidates(area, cutoff)
        assert storage.lock_candidate(candidate) is None
        assert path.read_bytes() == b"original"
    finally:
        stream.close()
    locked = storage.lock_candidate(candidate)
    assert locked is not None
    with locked:
        assert storage.lock_candidate(candidate) is None
        assert locked.delete() is True
        assert locked.delete() is False
    assert not path.exists()


def test_changed_inode_is_preserved_at_lock_and_at_delete(tmp_path: Path) -> None:
    storage = LocalFileStorage(tmp_path)
    key, path = storage.create_staging()
    cutoff = age(path)
    (candidate,) = storage.old_candidates("staging", cutoff)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"new inode")
    os.replace(replacement, path)
    assert storage.lock_candidate(candidate) is None
    cutoff = age(path)
    (current,) = storage.old_candidates("staging", cutoff)
    locked = storage.lock_candidate(current)
    assert locked is not None
    try:
        another = tmp_path / "another"
        another.write_bytes(b"keep this")
        os.replace(another, path)
        assert locked.delete() is False
    finally:
        locked.close()
    assert path.read_bytes() == b"keep this"


def test_changed_timestamp_is_preserved_even_with_same_inode(tmp_path: Path) -> None:
    storage = LocalFileStorage(tmp_path)
    _, path = storage.create_staging()
    cutoff = age(path)
    (candidate,) = storage.old_candidates("staging", cutoff)
    os.utime(path, ns=(candidate.mtime_ns + 1, candidate.mtime_ns + 1))
    assert storage.lock_candidate(candidate) is None


def test_delete_rechecks_timestamp_and_close_is_idempotent(tmp_path: Path) -> None:
    storage = LocalFileStorage(tmp_path)
    _, path = storage.create_staging()
    cutoff = age(path)
    (candidate,) = storage.old_candidates("staging", cutoff)
    locked = storage.lock_candidate(candidate)
    assert locked is not None
    os.utime(path, None)
    assert locked.delete() is False
    locked.close()
    locked.close()
    assert locked.delete() is False
    assert path.exists()
    cutoff = age(path)
    (current,) = storage.old_candidates("staging", cutoff)
    again = storage.lock_candidate(current)
    assert again is not None
    again.close()


@pytest.mark.parametrize("replacement", ["missing", "symlink", "fifo"])
def test_candidate_replaced_with_missing_symlink_or_fifo_is_skipped(
    tmp_path: Path,
    replacement: str,
) -> None:
    storage = LocalFileStorage(tmp_path)
    _, path = storage.create_staging()
    cutoff = age(path)
    (candidate,) = storage.old_candidates("staging", cutoff)
    path.unlink()
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    if replacement == "symlink":
        path.symlink_to(outside)
    elif replacement == "fifo":
        os.mkfifo(path)
    assert storage.lock_candidate(candidate) is None
    assert outside.read_bytes() == b"outside"


def test_symlinked_storage_area_is_not_traversed(tmp_path: Path) -> None:
    storage = LocalFileStorage(tmp_path)
    _, path = storage.create_staging()
    cutoff = age(path)
    (candidate,) = storage.old_candidates("staging", cutoff)
    staging = tmp_path / "staging"
    staging.rename(tmp_path / "moved-staging")
    staging.symlink_to(tmp_path / "moved-staging", target_is_directory=True)
    with pytest.raises(ProcessingError):
        storage.old_candidates("staging", cutoff)
    with pytest.raises(ProcessingError):
        storage.lock_candidate(candidate)
    assert (tmp_path / "moved-staging" / candidate.key).exists()


def test_scan_and_lock_validate_area_key_and_age(tmp_path: Path) -> None:
    storage = LocalFileStorage(tmp_path)
    _, path = storage.create_staging()
    cutoff = age(path)
    (candidate,) = storage.old_candidates("staging", cutoff)
    with pytest.raises(ValueError, match="area"):
        storage.old_candidates("../outside", cutoff)
    with pytest.raises(ValueError, match="finite"):
        storage.old_candidates("staging", float("inf"))
    with pytest.raises(ProcessingError):
        storage.lock_candidate(replace(candidate, key="../outside"))
    assert storage.lock_candidate(replace(candidate, older_than=candidate.mtime)) is None
