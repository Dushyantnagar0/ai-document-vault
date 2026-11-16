"""FastAPI composition root for document APIs, health and durable recovery."""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from uuid import UUID, uuid4

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from app.adapters.async_openai import AsyncOpenAIEmbeddingProvider, AsyncOpenAIGenerationProvider
from app.adapters.dispatch import CeleryTaskDispatcher
from app.adapters.retrieval import PgVectorRetriever
from app.adapters.storage import LocalFileStorage
from app.api.conversations import router as conversations_router
from app.api.documents import APIError
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.metrics import router as metrics_router
from app.core.body_limit import RequestBodyLimit
from app.core.config import get_settings
from app.core.readiness import Readiness
from app.db.session import api_engine
from app.domain.conversation import ConversationError
from app.domain.documents import ProcessingError
from app.services.chat import ChatProcessor
from app.services.conversations import TurnStore
from app.services.maintenance import run_maintenance
from app.services.recovery import recovery_loop
from app.services.telemetry import record_http_sample


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    # Configure only application events; provider/HTTP libraries keep their
    # existing levels and must not emit request bodies or credentials.
    application_logger = logging.getLogger("app")
    if not application_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        application_logger.addHandler(handler)
    application_logger.setLevel(logging.INFO)
    settings = get_settings()
    engine = api_engine(settings)
    application.state.engine = engine
    application.state.dispatcher = CeleryTaskDispatcher()
    store = TurnStore(engine, settings)
    application.state.turn_store = store
    application.state.turn_tasks = {}
    application.state.chat_processor = ChatProcessor(
        engine,
        settings,
        store,
        PgVectorRetriever(engine),
        AsyncOpenAIGenerationProvider(settings),
        AsyncOpenAIEmbeddingProvider(settings),
        reviewer=AsyncOpenAIGenerationProvider(
            settings, reasoning_effort=settings.openai_review_reasoning_effort
        ),
    )
    application.state.readiness = Readiness(settings, engine)
    monitor = asyncio.create_task(application.state.readiness.monitor())
    recovery = asyncio.create_task(recovery_loop(engine, settings, application.state.dispatcher))

    async def expire_turns() -> None:
        while True:
            try:
                await store.expire_pending()
            except Exception:
                pass
            await asyncio.sleep(15)

    conversation_recovery = asyncio.create_task(expire_turns())

    async def maintain_storage() -> None:
        while True:
            try:
                storage = await asyncio.to_thread(LocalFileStorage, settings.local_storage_dir)
                await run_maintenance(engine, storage)
            except Exception as error:
                logging.getLogger(__name__).warning(
                    "Maintenance deferred [%s]", type(error).__name__
                )
            await asyncio.sleep(3600)

    maintenance = asyncio.create_task(maintain_storage())
    try:
        yield
    finally:
        monitor.cancel()
        recovery.cancel()
        conversation_recovery.cancel()
        maintenance.cancel()
        with suppress(asyncio.CancelledError):
            await monitor
        with suppress(asyncio.CancelledError):
            await recovery
        with suppress(asyncio.CancelledError):
            await conversation_recovery
        with suppress(asyncio.CancelledError):
            await maintenance
        tasks = list(application.state.turn_tasks.values())
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=95)
            for task in pending:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        await engine.dispose()


def error_response(request: Request, status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": getattr(request.state, "request_id", str(uuid4())),
                "details": [],
            }
        },
    )


def create_app() -> FastAPI:
    application = FastAPI(
        title="Document Vault API",
        version="0.1.0",
        description=(
            "Asynchronous PDF/DOCX analysis with persisted source evidence. "
            "Grounded multi-turn conversation and persisted history."
        ),
        lifespan=lifespan,
    )
    application.add_middleware(RequestBodyLimit)

    @application.middleware("http")
    async def request_identity(request: Request, call_next: RequestResponseEndpoint) -> Response:
        started = time.monotonic()
        try:
            request_id = str(UUID(request.headers.get("X-Request-ID", "")))
        except ValueError:
            request_id = str(uuid4())
        request.state.request_id = request_id
        try:
            response = await call_next(request)
        except Exception:
            response = error_response(
                request, 500, "INTERNAL_ERROR", "An unexpected error occurred"
            )
        response.headers["X-Request-ID"] = request_id
        route = request.scope.get("route")
        engine = getattr(request.app.state, "engine", None)
        if engine is not None and route is not None:
            background = BackgroundTasks()
            if response.background is not None:
                background.add_task(response.background)
            background.add_task(
                record_http_sample,
                engine,
                f"http.{request.method} {route.path}",
                started,
                response.status_code,
            )
            response.background = background
        return response

    @application.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return error_response(
            request,
            exc.status_code,
            "NOT_FOUND"
            if exc.status_code == 404
            else (
                "UPLOAD_TOO_LARGE"
                if request.url.path == "/api/v1/documents/upload"
                else "REQUEST_TOO_LARGE"
            )
            if exc.status_code == 413
            else "HTTP_ERROR",
            "Resource not found"
            if exc.status_code == 404
            else "Request exceeds the configured input limit"
            if exc.status_code == 413
            else "Request could not be completed",
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        malformed = any(error.get("type") == "json_invalid" for error in exc.errors())
        response = error_response(
            request,
            400 if malformed else 422,
            "INVALID_JSON" if malformed else "VALIDATION_ERROR",
            "Malformed JSON request" if malformed else "Request validation failed",
        )
        payload = json.loads(bytes(response.body))
        payload["error"]["details"] = [
            {
                "field": ".".join(str(part) for part in error.get("loc", ())),
                "code": error.get("type", "invalid"),
                "message": "Check this field's format and limits",
            }
            for error in exc.errors()[:20]
        ]
        return JSONResponse(status_code=response.status_code, content=payload)

    @application.exception_handler(APIError)
    async def application_error(request: Request, exc: APIError) -> JSONResponse:
        return error_response(request, exc.status, exc.code, exc.message)

    @application.exception_handler(ProcessingError)
    async def processing_error(request: Request, exc: ProcessingError) -> JSONResponse:
        return error_response(request, 503, exc.code, exc.message)

    @application.exception_handler(SQLAlchemyError)
    async def database_error(request: Request, exc: SQLAlchemyError) -> JSONResponse:
        return error_response(
            request,
            503,
            "DATABASE_UNAVAILABLE",
            "Database operation is unavailable; retry with the same request ID",
        )

    @application.exception_handler(ConversationError)
    async def conversation_error(request: Request, exc: ConversationError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "request_id": request.state.request_id,
                    "details": exc.details,
                }
            },
        )

    application.include_router(health_router)
    application.include_router(documents_router)
    application.include_router(conversations_router)
    application.include_router(metrics_router)
    return application


app = create_app()
