"""Pure-ASGI middleware: request context, request-size limit, rate limit.

They are implemented at the ASGI level (not ``BaseHTTPMiddleware``) so that
multipart uploads are never buffered by the middleware and so the size limit
also applies to chunked requests that carry no ``Content-Length``.
"""

from __future__ import annotations

import logging
import math
import re
import time
import uuid
from collections import defaultdict, deque

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.errors import RateLimitExceededError, RequestTooLargeError, error_body

logger = logging.getLogger("scorematrix.http")
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _request_id(scope: Scope) -> str | None:
    return scope.get("state", {}).get("request_id")


class RequestContextMiddleware:
    """Assigns an ``X-Request-ID`` and logs one line per request (never bodies or queries)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = Headers(scope=scope).get("x-request-id", "")
        request_id = incoming if _REQUEST_ID_RE.match(incoming) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        started = time.perf_counter()
        status = {"code": 0}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            logger.info(
                "%s %s -> %s in %.0f ms [%s]",
                scope["method"],
                scope["path"],
                status["code"] or "-",
                (time.perf_counter() - started) * 1000,
                request_id,
            )


class RequestSizeLimitMiddleware:
    """Rejects oversized bodies: early via Content-Length, and while streaming otherwise."""

    def __init__(self, app: ASGIApp, max_bytes: int, max_json_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.max_json_bytes = max_json_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] in ("GET", "HEAD", "OPTIONS"):
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        content_type = headers.get("content-type", "").split(";")[0].strip().lower()
        limit = self.max_json_bytes if content_type == "application/json" else self.max_bytes
        declared = headers.get("content-length")
        if declared is not None:
            try:
                too_big = int(declared) > limit
            except ValueError:
                too_big = False
            if too_big:
                await self._too_large(scope, receive, send, limit)
                return

        state = {"received": 0, "exceeded": False, "replaced": False}

        async def limited_receive() -> Message:
            message = await receive()
            if message["type"] == "http.request":
                state["received"] += len(message.get("body", b""))
                if state["received"] > limit:
                    state["exceeded"] = True
                    raise RequestTooLargeError()
            return message

        async def limited_send(message: Message) -> None:
            if not state["exceeded"]:
                await send(message)
                return
            # The handler failed because we cut the body off: answer with a clean 413 instead.
            if message["type"] == "http.response.start" and not state["replaced"]:
                state["replaced"] = True
                await self._too_large(scope, receive, send, limit)

        await self.app(scope, limited_receive, limited_send)

    @staticmethod
    async def _too_large(scope: Scope, receive: Receive, send: Send, limit: int) -> None:
        message = f"The request body is larger than the {limit / (1024 * 1024):g} MB limit."
        response = JSONResponse(
            status_code=413,
            content=error_body(
                code=RequestTooLargeError.code,
                message=message,
                source="client",
                details={"max_size_mb": round(limit / (1024 * 1024), 3)},
                request_id=_request_id(scope),
            ),
        )
        await response(scope, receive, send)


class RateLimitMiddleware:
    """In-memory sliding-window limiter per client address for ``/api/*``.

    State is per process: with several workers/instances each enforces its own
    window. Put a shared limiter (reverse proxy / Redis) in front for strict limits.
    """

    _MAX_TRACKED_CLIENTS = 10_000

    def __init__(
        self,
        app: ASGIApp,
        *,
        enabled: bool,
        limit: int,
        window_seconds: float,
        trust_proxy_headers: bool = False,
        clock=time.monotonic,
    ) -> None:
        self.app = app
        self.enabled = enabled
        self.limit = limit
        self.window = window_seconds
        self.trust_proxy = trust_proxy_headers
        self._clock = clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def _client_key(self, scope: Scope) -> str:
        if self.trust_proxy:
            forwarded = Headers(scope=scope).get("x-forwarded-for", "")
            if forwarded:
                return forwarded.split(",")[0].strip()
        client = scope.get("client")
        return client[0] if client else "unknown"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            not self.enabled
            or scope["type"] != "http"
            or scope["method"] == "OPTIONS"
            or not scope["path"].startswith("/api/")
        ):
            await self.app(scope, receive, send)
            return
        now = self._clock()
        key = self._client_key(scope)
        hits = self._hits[key]
        while hits and now - hits[0] >= self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            retry_after = max(1, math.ceil(self.window - (now - hits[0])))
            response = JSONResponse(
                status_code=429,
                content=error_body(
                    code=RateLimitExceededError.code,
                    message=f"Rate limit exceeded ({self.limit} requests per {self.window:g}s). Retry in {retry_after}s.",
                    source="client",
                    details={"retry_after_seconds": retry_after},
                    request_id=_request_id(scope),
                ),
                headers={"Retry-After": str(retry_after)},
            )
            await response(scope, receive, send)
            return
        hits.append(now)
        if len(self._hits) > self._MAX_TRACKED_CLIENTS:
            for stale in [k for k, v in self._hits.items() if not v or now - v[-1] >= self.window]:
                del self._hits[stale]
        await self.app(scope, receive, send)
