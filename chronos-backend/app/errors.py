"""Application error hierarchy and the consistent JSON error envelope.

Every error response has this shape::

    {
      "detail": "<human readable message>",          # kept for FastAPI-style clients
      "error": {
        "code": "pdf_encrypted",                     # stable machine-readable code
        "message": "<human readable message>",
        "source": "client" | "server" | "upstream_ai",
        "details": {...},                            # optional, never contains secrets
        "request_id": "..."
      }
    }

``source`` lets callers distinguish their own mistakes (``client``), faults in
this service (``server``) and failures of the Gemini API (``upstream_ai``).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("scorematrix.errors")


class AppError(Exception):
    status_code: int = 500
    code: str = "internal_error"
    source: str = "server"
    default_message: str = "An unexpected error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.message = message or self.default_message
        self.details = details
        self.headers = headers
        super().__init__(self.message)


# ------------------------------------------------------------------ client errors
class BadRequestError(AppError):
    status_code, code, source = 400, "bad_request", "client"
    default_message = "The request was malformed."


class ValidationAppError(AppError):
    status_code, code, source = 422, "validation_error", "client"
    default_message = "The request contains invalid values."


class UnsupportedMediaTypeError(AppError):
    status_code, code, source = 415, "unsupported_media_type", "client"
    default_message = "Unsupported request content type."


class UnsupportedFileTypeError(AppError):
    status_code, code, source = 415, "unsupported_file_type", "client"
    default_message = "Unsupported file type."


class FileTooLargeError(AppError):
    status_code, code, source = 413, "file_too_large", "client"
    default_message = "The uploaded file is too large."


class RequestTooLargeError(AppError):
    status_code, code, source = 413, "request_too_large", "client"
    default_message = "The request body is too large."


class TooManyFilesError(AppError):
    status_code, code, source = 413, "too_many_files", "client"
    default_message = "Too many files were uploaded."


class RateLimitExceededError(AppError):
    status_code, code, source = 429, "rate_limit_exceeded", "client"
    default_message = "Too many requests. Please slow down."


class PdfError(AppError):
    status_code, code, source = 422, "pdf_error", "client"
    default_message = "The PDF could not be processed."


class CorruptPdfError(PdfError):
    code = "pdf_corrupt"
    default_message = "The PDF is corrupt or malformed and could not be read."


class EncryptedPdfError(PdfError):
    code = "pdf_encrypted"
    default_message = "The PDF is password-protected. Remove the password and upload it again."


class EmptyPdfError(PdfError):
    code = "pdf_empty"
    default_message = "The PDF contains no pages."


class NoReadableTextError(PdfError):
    code = "pdf_no_readable_text"
    default_message = "No readable text could be extracted from the document."


class EmptyFileError(AppError):
    status_code, code, source = 422, "file_empty", "client"
    default_message = "The uploaded file is empty."


class InsufficientTextError(AppError):
    status_code, code, source = 422, "insufficient_text", "client"
    default_message = "The document does not contain enough text for this operation."


# ------------------------------------------------------------ server-side errors
class OCRUnavailableError(AppError):
    status_code, code, source = 503, "ocr_unavailable", "server"
    default_message = "OCR is required for this document but is not available."


class EmbeddingsUnavailableError(AppError):
    status_code, code, source = 503, "embeddings_unavailable", "server"
    default_message = "The embedding model is not available."


class ProcessingTimeoutError(AppError):
    status_code, code, source = 504, "processing_timeout", "server"
    default_message = "Processing took too long and was stopped."


# ------------------------------------------------------------ upstream AI errors
class AIError(AppError):
    status_code, code, source = 502, "ai_error", "upstream_ai"
    default_message = "The AI provider returned an error."


class AINotConfiguredError(AIError):
    status_code, code = 503, "ai_not_configured"
    source = "server"
    default_message = "The AI provider is not configured on the server (GEMINI_API_KEY is missing)."


class AIAuthenticationError(AIError):
    code = "ai_authentication_failed"
    default_message = "The AI provider rejected the server's credentials."


class AIRateLimitError(AIError):
    status_code, code = 503, "ai_rate_limited"
    default_message = "The AI provider is rate-limiting or out of quota. Try again shortly."


class AITimeoutError(AIError):
    status_code, code = 504, "ai_timeout"
    default_message = "The AI provider did not respond in time."


class AIUnavailableError(AIError):
    status_code, code = 503, "ai_unavailable"
    default_message = "The AI provider is temporarily unavailable."


class AIRequestRejectedError(AIError):
    code = "ai_request_rejected"
    default_message = "The AI provider rejected the request."


class AIContentBlockedError(AIError):
    code = "ai_content_blocked"
    default_message = "The AI provider declined to process this content."


class AIResponseError(AIError):
    code = "ai_malformed_response"
    default_message = "The AI provider returned a response that could not be validated."


class AIIncompleteResponseError(AIError):
    code = "ai_incomplete_response"
    default_message = "The AI response was cut off before completion."


# ------------------------------------------------------------------- envelope
def _request_id(request: Request) -> str | None:
    state = getattr(request, "state", None)
    return getattr(state, "request_id", None) if state is not None else None


def error_body(
    *,
    code: str,
    message: str,
    source: str,
    details: Any = None,
    request_id: str | None = None,
    detail: Any = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message, "source": source}
    if details:
        error["details"] = details
    if request_id:
        error["request_id"] = request_id
    return {"detail": detail if detail is not None else message, "error": error}


def sanitize_validation_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop ``input`` / ``ctx`` so submitted content is never echoed back."""
    return [
        {"loc": list(e.get("loc", ())), "msg": str(e.get("msg", "")), "type": str(e.get("type", ""))}
        for e in errors
    ]


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.warning("%s on %s (%s)", exc.code, request.url.path, type(exc).__name__)
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(
                code=exc.code,
                message=exc.message,
                source=exc.source,
                details=exc.details,
                request_id=_request_id(request),
            ),
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = sanitize_validation_errors(list(exc.errors()))
        return JSONResponse(
            status_code=422,
            content=error_body(
                code="validation_error",
                message="The request contains invalid or missing fields.",
                source="client",
                details={"errors": errors},
                request_id=_request_id(request),
                detail=errors,  # FastAPI-compatible shape for existing clients
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        source = "client" if exc.status_code < 500 else "server"
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(
                code=code,
                message=str(exc.detail),
                source=source,
                request_id=_request_id(request),
            ),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # The traceback goes to the server log only; clients get a generic message.
        logger.error("Unhandled %s on %s", type(exc).__name__, request.url.path, exc_info=True)
        return JSONResponse(
            status_code=500,
            content=error_body(
                code="internal_error",
                message="An internal error occurred. Please try again or contact support.",
                source="server",
                request_id=_request_id(request),
            ),
        )
