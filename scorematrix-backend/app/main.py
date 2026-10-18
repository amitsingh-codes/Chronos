"""ScoreMatrix API application factory."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from pydantic.json_schema import models_json_schema

from app.config import Settings
from app.dependencies import build_container
from app.errors import register_exception_handlers
from app.logging_config import configure_logging
from app.middleware.core import RateLimitMiddleware, RequestContextMiddleware, RequestSizeLimitMiddleware
from app.routers import evaluation, health, plagiarism, questions, rag
from app.schemas.evaluation import BatchEvaluationRequest, EvaluationRequest
from app.schemas.plagiarism import PlagiarismCheckRequest
from app.schemas.rag import RagAskRequest

DESCRIPTION = """
AI-assisted assessment backend for teachers and evaluators.

**Important:** all AI output (generated questions, marks, feedback, similarity scores) is advisory.
Evaluation results are provisional and must be reviewed by a teacher.

PDF uploads (selectable text, scanned/OCR and mixed) are supported on question generation,
evaluation (single + batch), plagiarism checking and RAG.
"""

# JSON request models referenced by hand-written multi-format request bodies.
_EXTRA_SCHEMA_MODELS = [EvaluationRequest, BatchEvaluationRequest, PlagiarismCheckRequest, RagAskRequest]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    app.state.settings = settings
    app.state.container = build_container(settings)

    # Last added = outermost. CORS is outermost so even 413/429 responses carry CORS headers.
    app.add_middleware(
        RequestSizeLimitMiddleware,
        max_bytes=settings.max_request_size_bytes,
        max_json_bytes=settings.max_json_body_bytes,
    )
    app.add_middleware(
        RateLimitMiddleware,
        enabled=settings.rate_limit_enabled,
        limit=settings.rate_limit_requests,
        window_seconds=settings.rate_limit_window_seconds,
        trust_proxy_headers=settings.trust_proxy_headers,
    )
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
        expose_headers=["X-Request-ID", "Retry-After"],
    )

    register_exception_handlers(app)
    for module in (health, questions, evaluation, plagiarism, rag):
        app.include_router(module.router)

    def custom_openapi() -> dict:
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title, version=app.version, description=app.description, routes=app.routes
        )
        _, definitions = models_json_schema(
            [(m, "validation") for m in _EXTRA_SCHEMA_MODELS],
            ref_template="#/components/schemas/{model}",
        )
        components = schema.setdefault("components", {}).setdefault("schemas", {})
        for name, definition in definitions.get("$defs", {}).items():
            components.setdefault(name, definition)
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi  # type: ignore[method-assign]
    return app


app = create_app()
