"""Health and diagnostics. Reports only safe information (no keys, paths or stack traces)."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends

from app.dependencies import (
    Container,
    get_ai_service,
    get_container,
    get_embedding_service,
    get_ocr_engine,
    get_pdf_service,
)
from app.schemas.health import (
    AICapability,
    EmbeddingCapability,
    HealthResponse,
    OCRCapability,
    PdfCapability,
)
from app.services.ai_service import AIService
from app.services.embeddings import EmbeddingService
from app.services.ocr import OcrEngine
from app.services.pdf_service import PdfService

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health and optional capabilities",
    description=(
        "`status` describes the application itself. Optional capabilities (Gemini credentials, OCR, "
        "embeddings) are reported separately; a missing capability degrades specific features but does "
        "not make the application unhealthy."
    ),
)
async def health(
    container: Container = Depends(get_container),
    ai: AIService = Depends(get_ai_service),
    ocr: OcrEngine = Depends(get_ocr_engine),
    embedder: EmbeddingService = Depends(get_embedding_service),
    pdf: PdfService = Depends(get_pdf_service),
) -> HealthResponse:
    s = container.settings
    ai_status = ai.status()
    ocr_status = await asyncio.to_thread(ocr.status)
    emb_status = embedder.status()
    pdf_caps = pdf.capabilities()

    unavailable: list[str] = []
    if not ai_status.configured:
        unavailable += ["question_generation", "evaluation", "batch_evaluation", "rag (answer generation)"]
    if not s.ocr_enabled:
        unavailable.append("scanned_pdf_ocr (disabled by OCR_ENABLED=false)")
    elif not ocr_status.available:
        unavailable.append("scanned_pdf_ocr")
    if not emb_status.available:
        unavailable += ["rag (retrieval)", "plagiarism_semantic_similarity"]
    if not pdf_caps["page_rendering"]:
        unavailable.append("scanned_pdf_ocr (page rendering)")

    return HealthResponse(
        status="ok",
        app=s.app_name,
        version=s.app_version,
        environment=s.environment,
        ai=AICapability(provider=ai_status.provider, configured=ai_status.configured, model=ai_status.model),
        ocr=OCRCapability(
            enabled=s.ocr_enabled,
            available=s.ocr_enabled and ocr_status.available,
            engine=ocr_status.engine,
            version=ocr_status.version,
            languages_requested=ocr_status.languages_requested,
            languages_available=ocr_status.languages_available,
            detail="OCR is disabled by configuration (OCR_ENABLED=false)." if not s.ocr_enabled else ocr_status.detail,
        ),
        embeddings=EmbeddingCapability(
            backend=embedder.backend,
            model=embedder.model_name,
            available=emb_status.available,
            loaded=emb_status.loaded,
            semantic=embedder.semantic,
            detail=emb_status.detail,
        ),
        pdf=PdfCapability(**pdf_caps),
        unavailable_features=unavailable,
        limits={
            "max_request_size_mb": s.max_request_size_mb,
            "max_pdf_size_mb": s.max_pdf_size_mb,
            "max_pdf_pages": s.max_pdf_pages,
            "ocr_dpi": s.ocr_dpi,
            "ocr_max_pages": s.ocr_max_pages,
            "batch_max_files": s.batch_max_files,
            "batch_concurrency": s.batch_concurrency,
            "plagiarism_max_documents": s.plagiarism_max_documents,
            "rag_max_documents": s.rag_max_documents,
            "rate_limit": f"{s.rate_limit_requests}/{s.rate_limit_window_seconds:g}s"
            if s.rate_limit_enabled
            else "disabled",
        },
    )
