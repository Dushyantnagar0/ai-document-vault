"""Bounded telemetry retention and cooperative, DB-checked orphan cleanup."""

import asyncio
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import partial

from sqlalchemy import delete, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine

from app.adapters.storage import LocalFileStorage, LockedFile, StaleFile, StorageArea
from app.db.models import Document, PerformanceSample
from app.domain.documents import ProcessingError

MAX_PERFORMANCE_DELETIONS = 1000
MAX_FILE_DELETIONS = 1000


@dataclass
class MaintenanceResult:
    """Confirmed deletions; unavailable DB stops all further filesystem deletion."""

    performance_samples_deleted: int = 0
    staging_deleted: int = 0
    originals_deleted: int = 0
    referenced_skipped: int = 0
    busy_or_changed_skipped: int = 0
    file_errors: int = 0
    database_available: bool = True
    file_limit_reached: bool = False


async def _offload[T](operation: Callable[[], T]) -> T:
    """Drain filesystem operations before cancellation can release their locks."""
    pending = asyncio.create_task(asyncio.to_thread(operation))
    try:
        return await asyncio.shield(pending)
    except asyncio.CancelledError:
        with suppress(Exception):
            await pending
        raise


async def _lock(storage: LocalFileStorage, candidate: StaleFile) -> LockedFile | None:
    pending = asyncio.create_task(asyncio.to_thread(storage.lock_candidate, candidate))
    try:
        return await asyncio.shield(pending)
    except asyncio.CancelledError:
        # Acquiring a descriptor in a background thread can finish after the
        # awaiting task is cancelled. Reclaim that descriptor before unwinding.
        locked = None
        with suppress(Exception):
            locked = await pending
        if locked is not None:
            await _offload(locked.close)
        raise


async def _is_referenced(engine: AsyncEngine, key: str) -> bool:
    async with engine.connect() as connection:
        return (
            await connection.scalar(select(Document.id).where(Document.storage_key == key).limit(1))
            is not None
        )


async def run_maintenance(
    engine: AsyncEngine,
    storage: LocalFileStorage,
    now: datetime | None = None,
) -> MaintenanceResult:
    """Expire at most 1,000 samples and 1,000 unreferenced aged regular files.

    Metadata scans may examine all current keys; referenced/busy entries do not
    consume the deletion budget or hide an orphan later in the scan. Original
    upload writers must retain their staging stream lock through DB commit.
    """
    measured = now if now is not None else datetime.now(UTC)
    if measured.tzinfo is None or measured.utcoffset() is None:
        raise ValueError("Maintenance time must include a UTC offset")
    measured = measured.astimezone(UTC)
    result = MaintenanceResult()
    expired_samples = (
        select(PerformanceSample.id)
        .where(PerformanceSample.recorded_at < measured - timedelta(days=7))
        .order_by(PerformanceSample.recorded_at, PerformanceSample.id)
        .limit(MAX_PERFORMANCE_DELETIONS)
        .with_for_update(skip_locked=True)
        .cte("expired_performance_samples")
    )
    try:
        async with engine.begin() as connection:
            referenced = set((await connection.scalars(select(Document.storage_key))).all())
            removed = (
                await connection.scalars(
                    delete(PerformanceSample)
                    .where(PerformanceSample.id.in_(select(expired_samples.c.id)))
                    .returning(PerformanceSample.id)
                )
            ).all()
        result.performance_samples_deleted = len(removed)
    except SQLAlchemyError:
        result.database_available = False
        return result

    areas: tuple[tuple[StorageArea, timedelta], ...] = (
        ("staging", timedelta(hours=1)),
        ("originals", timedelta(hours=24)),
    )
    for area, minimum_age in areas:
        try:
            cutoff = (measured - minimum_age).timestamp()
            candidates = await _offload(partial(storage.old_candidates, area, cutoff))
        except (ProcessingError, OSError):
            result.file_errors += 1
            continue
        for candidate in candidates:
            if candidate.key in referenced:
                result.referenced_skipped += 1
                continue
            if result.staging_deleted + result.originals_deleted >= MAX_FILE_DELETIONS:
                result.file_limit_reached = True
                return result
            try:
                locked = await _lock(storage, candidate)
            except (ProcessingError, OSError):
                result.file_errors += 1
                continue
            if locked is None:
                result.busy_or_changed_skipped += 1
                continue
            try:
                try:
                    # Check both areas for safety. The lock excludes an upload
                    # committing this same inode while its reference is checked.
                    if await _is_referenced(engine, candidate.key):
                        referenced.add(candidate.key)
                        result.referenced_skipped += 1
                        continue
                except SQLAlchemyError:
                    result.database_available = False
                    return result
                try:
                    deleted = await _offload(locked.delete)
                except (ProcessingError, OSError):
                    result.file_errors += 1
                    continue
                if not deleted:
                    result.busy_or_changed_skipped += 1
                elif area == "staging":
                    result.staging_deleted += 1
                else:
                    result.originals_deleted += 1
            finally:
                await _offload(locked.close)
    result.file_limit_reached = (
        result.staging_deleted + result.originals_deleted >= MAX_FILE_DELETIONS
    )
    return result
