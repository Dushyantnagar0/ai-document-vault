"""Best-effort route timing without request bodies, query values, or secrets."""

import time

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.db.models import PerformanceSample


async def record_http_sample(
    engine: AsyncEngine, operation: str, started: float, status: int
) -> None:
    duration_ms = max(0.0, (time.monotonic() - started) * 1000)
    try:
        factory = async_sessionmaker(engine)
        async with factory() as session, session.begin():
            session.add(
                PerformanceSample(
                    operation=operation[:120],
                    duration_ms=duration_ms,
                    outcome="SUCCEEDED" if status < 400 else "FAILED",
                    error_class=f"HTTP_{status}" if status >= 400 else None,
                )
            )
    except Exception:
        pass  # Diagnostics must not turn a delivered application response into a failure.
