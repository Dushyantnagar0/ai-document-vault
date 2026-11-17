"""Immediate publication plus periodic repair of missed/stale durable work."""

import asyncio
from datetime import timedelta
from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.adapters.dispatch import TaskDispatcher
from app.core.config import Settings
from app.db.models import Document, ProcessingAttempt
from app.services.ingestion import utcnow


async def dispatch_document(
    engine: AsyncEngine, dispatcher: TaskDispatcher, document_id: UUID
) -> None:
    factory = async_sessionmaker(engine, expire_on_commit=False)
    now = utcnow()
    reservation = now + timedelta(seconds=10)
    async with factory() as session, session.begin():
        doc = await session.get(Document, document_id, with_for_update=True)
        now = utcnow()
        reservation = now + timedelta(seconds=10)
        if doc is None or doc.status != "PENDING" or doc.next_dispatch_at > now:
            return
        if doc.dispatch_reserved_until is not None and doc.dispatch_reserved_until > now:
            return
        if doc.last_dispatched_at is not None and doc.last_dispatched_at > now - timedelta(
            seconds=60
        ):
            return
        doc.dispatch_reserved_until = reservation
    published = False
    try:
        await asyncio.to_thread(dispatcher.submit, document_id)
        published = True
    except Exception:
        pass
    async with factory() as session, session.begin():
        doc = await session.get(Document, document_id, with_for_update=True)
        if doc is not None and doc.dispatch_reserved_until == reservation:
            doc.dispatch_reserved_until = None
            if published:
                doc.last_dispatched_at = now
                if (
                    doc.status == "PENDING"
                    and doc.error
                    and doc.error.get("code") == "DISPATCH_UNAVAILABLE"
                ):
                    doc.error = None
                    if doc.stage == "DISPATCH_PENDING":
                        doc.stage = "QUEUED"
            elif doc.status == "PENDING":
                doc.stage = "DISPATCH_PENDING"
                doc.error = {
                    "code": "DISPATCH_UNAVAILABLE",
                    "message": "Accepted; waiting for background transport recovery",
                }


async def recover_once(engine: AsyncEngine, settings: Settings, dispatcher: TaskDispatcher) -> None:
    now = utcnow()
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session, session.begin():
        documents = (
            await session.scalars(
                select(Document)
                .where(
                    Document.status.in_(["PENDING", "PROCESSING"]),
                    or_(
                        Document.deadline_at <= now,
                        and_(
                            Document.status == "PROCESSING",
                            or_(Document.lease_until.is_(None), Document.lease_until <= now),
                        ),
                        and_(
                            Document.status == "PENDING",
                            Document.attempt_count == 0,
                            Document.created_at <= now - timedelta(seconds=900),
                        ),
                    ),
                )
                .order_by(Document.created_at)
                .limit(100)
                .with_for_update(skip_locked=True)
            )
        ).all()
        for doc in documents:
            now = utcnow()
            expired = doc.status == "PROCESSING" and (
                doc.lease_until is None or doc.lease_until <= now
            )
            deadline = doc.deadline_at <= now
            waiting = (
                doc.status == "PENDING"
                and doc.attempt_count == 0
                and doc.created_at <= now - timedelta(seconds=900)
            )
            exhausted = (
                doc.attempt_count >= settings.processing_max_attempts
                or doc.ai_calls_reserved >= settings.max_document_ai_calls
            )
            if not (expired or deadline or waiting):
                continue
            error = {
                "code": "PROCESSING_DEADLINE" if deadline or waiting else "WORKER_INTERRUPTED",
                "message": "Processing deadline exhausted; upload again"
                if deadline or waiting
                else "Worker ownership expired during processing",
            }
            if doc.status == "PROCESSING" and doc.current_attempt_id is not None:
                attempt = await session.get(ProcessingAttempt, doc.current_attempt_id)
                if attempt is not None and attempt.outcome == "RUNNING":
                    attempt.outcome, attempt.ended_at, attempt.error = "INTERRUPTED", now, error
            terminal = deadline or waiting or exhausted
            doc.status, doc.stage = ("FAILED", "FAILED") if terminal else ("PENDING", "RETRY_WAIT")
            doc.error, doc.updated_at = error, now
            doc.claim_token, doc.lease_until, doc.dispatch_reserved_until = None, None, None
            if terminal:
                doc.terminal_at = now
            else:
                doc.next_dispatch_at, doc.last_dispatched_at = now + timedelta(seconds=15), None
    async with factory() as session:
        ids = (
            await session.scalars(
                select(Document.id)
                .where(
                    Document.status == "PENDING",
                    Document.next_dispatch_at <= now,
                    or_(
                        Document.last_dispatched_at.is_(None),
                        Document.last_dispatched_at <= now - timedelta(seconds=60),
                    ),
                    or_(
                        Document.dispatch_reserved_until.is_(None),
                        Document.dispatch_reserved_until <= now,
                    ),
                )
                .order_by(Document.next_dispatch_at)
                .limit(100)
            )
        ).all()
    for document_id in ids:
        await dispatch_document(engine, dispatcher, document_id)


async def recovery_loop(
    engine: AsyncEngine, settings: Settings, dispatcher: TaskDispatcher
) -> None:
    while True:
        try:
            await recover_once(engine, settings, dispatcher)
        except Exception:
            pass
        await asyncio.sleep(15)
