"""Selected bounded Celery publisher; PostgreSQL claims fence duplicate deliveries."""

from typing import Protocol
from uuid import UUID

from app.domain.documents import ProcessingError


class TaskDispatcher(Protocol):
    def submit(self, document_id: UUID) -> None: ...


class CeleryTaskDispatcher:
    def submit(self, document_id: UUID) -> None:
        from app.tasks.celery_app import INGEST_TASK_NAME, app

        try:
            with app.connection_for_write(
                connect_timeout=1,
                transport_options={
                    "socket_connect_timeout": 1,
                    "socket_timeout": 1,
                    "retry_on_timeout": False,
                },
            ) as connection:
                connection.ensure_connection(max_retries=0)
                app.send_task(
                    INGEST_TASK_NAME,
                    args=[str(document_id)],
                    connection=connection,
                    queue="ingestion",
                    retry=False,
                )
        except Exception:
            raise ProcessingError(
                "DISPATCH_UNAVAILABLE",
                "Background transport is unavailable; accepted work will be recovered",
                retryable=True,
            ) from None
