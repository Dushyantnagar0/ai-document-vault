"""Bound upload bodies before FastAPI's form parser spools file data."""

from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.config import get_settings


class RequestBodyLimit:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "")
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return
        if path == "/api/v1/documents/upload":
            limit = get_settings().max_upload_bytes + 1024 * 1024
        elif path.startswith("/api/v1/chat/"):
            limit = 4 * get_settings().question_max_chars + 4096
        else:
            await self.app(scope, receive, send)
            return
        consumed = 0

        async def bounded_receive() -> Message:
            nonlocal consumed
            message = await receive()
            if message["type"] == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > limit:
                    raise HTTPException(
                        status_code=413,
                        detail="Request exceeds the configured input-body limit",
                    )
            return message

        await self.app(scope, bounded_receive, send)
