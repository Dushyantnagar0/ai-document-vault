"""Bounded own-node task/queue readiness; processing quality is tested separately."""

import socket
from typing import Any

from app.tasks.celery_app import INGEST_TASK_NAME, PROBE_TASK_NAME, QUEUE_NAME, app


def worker_ready(application: Any, node: str | None = None, timeout: float = 2) -> bool:
    """Require own-node ping, probe/ingestion registration and queue consumption."""
    destination = node if node is not None else f"worker@{socket.gethostname()}"
    inspector = application.control.inspect(destination=[destination], timeout=timeout)
    ping = inspector.ping() or {}
    registered = inspector.registered() or {}
    queues = inspector.active_queues() or {}
    node_ping = ping.get(destination)
    node_tasks = registered.get(destination)
    node_queues = queues.get(destination)
    return (
        isinstance(node_ping, dict)
        and node_ping.get("ok") == "pong"
        and isinstance(node_tasks, list)
        and PROBE_TASK_NAME in node_tasks
        and INGEST_TASK_NAME in node_tasks
        and isinstance(node_queues, list)
        and any(
            isinstance(queue, dict) and queue.get("name") == QUEUE_NAME for queue in node_queues
        )
    )


def main() -> int:
    try:
        ready = worker_ready(app)
    except Exception:
        # Broker/SDK exceptions may contain credential-bearing URLs or payloads.
        print("Worker readiness unavailable")
        return 1
    if not ready:
        print("Worker transport is not ready")
        return 1
    print("Worker ready: probe and ingestion registered on the configured queue")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
