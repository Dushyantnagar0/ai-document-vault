"""Charge-free core readiness and cached ingestion dependency diagnostics."""

import asyncio
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

from alembic.config import Config
from alembic.script import ScriptDirectory
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.config import Settings


def storage_ready(directory: Path) -> bool:
    """Verify read/write/delete as the running user, without touching user files."""
    probe = directory / f".health-{uuid4()}"
    created = False
    try:
        with probe.open("xb") as stream:
            created = True
            stream.write(b"ready")
        readable = probe.read_bytes() == b"ready"
        probe.unlink()
        created = False
        return readable
    except OSError:
        return False
    finally:
        if created:
            with suppress(OSError):
                probe.unlink(missing_ok=True)


class Readiness:
    def __init__(self, settings: Settings, engine: AsyncEngine) -> None:
        self.settings = settings
        self.engine = engine
        self.head = ScriptDirectory.from_config(Config("alembic.ini")).get_current_head()
        self.dependencies: dict[str, str] = {"broker": "unknown", "worker": "unknown"}

    async def core(self) -> dict[str, str]:
        components = {"database": "unavailable", "schema": "unavailable", "storage": "unavailable"}
        try:
            async with asyncio.timeout(5), self.engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
                components["database"] = "ready"
                revision = await connection.scalar(text("SELECT version_num FROM alembic_version"))
                extension = await connection.scalar(
                    text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                )
                if revision == self.head and extension:
                    components["schema"] = "ready"
        except Exception:
            # HTTP diagnostics never serialize connection strings or SQL exceptions.
            pass
        if await asyncio.to_thread(storage_ready, self.settings.local_storage_dir):
            components["storage"] = "ready"
        return components

    async def refresh_dependencies(self) -> None:
        broker = Redis.from_url(
            self.settings.redis_url.get_secret_value(),
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        try:
            await broker.ping()
            self.dependencies["broker"] = "ready"
        except Exception:
            self.dependencies = {"broker": "unavailable", "worker": "unavailable"}
            return
        finally:
            await broker.aclose()
        self.dependencies["worker"] = (
            "ready" if await asyncio.to_thread(self._worker_ready) else "unavailable"
        )

    @staticmethod
    def _worker_ready() -> bool:
        from app.tasks.celery_app import INGEST_TASK_NAME, PROBE_TASK_NAME, QUEUE_NAME, app

        try:
            inspector = app.control.inspect(timeout=2)
            registered = inspector.registered() or {}
            queues = inspector.active_queues() or {}
            return any(
                PROBE_TASK_NAME in tasks
                and INGEST_TASK_NAME in tasks
                and any(queue.get("name") == QUEUE_NAME for queue in queues.get(node, []))
                for node, tasks in registered.items()
            )
        except Exception:
            return False

    async def monitor(self) -> None:
        while True:
            await self.refresh_dependencies()
            await asyncio.sleep(10)
