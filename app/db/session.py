"""Bounded engines; callers own short sessions and transaction boundaries."""

from sqlalchemy import Engine, create_engine
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.core.config import Settings


def api_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(
        settings.async_database_url,
        pool_size=5,
        max_overflow=0,
        pool_timeout=3,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 3, "options": "-c statement_timeout=3000"},
        hide_parameters=True,
    )


def worker_engine(settings: Settings) -> Engine:
    return create_engine(
        settings.database_url,
        pool_size=2,
        max_overflow=0,
        pool_timeout=3,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 3},
        hide_parameters=True,
    )
