"""Dependency-injection providers.

Long-lived, expensive or stateful objects (AI client, OCR probe, embedding
model, concurrency limiters) live in a :class:`Container` on ``app.state``.
Per-request services are built from them through ``Depends`` so tests can swap
any layer with ``app.dependency_overrides`` (e.g. a fake AI service).
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Request

from app.config import Settings
from app.services.ai_service import AIService, GeminiAIService
from app.services.batch import BatchEvaluator
from app.services.documents import DocumentLoader
from app.services.embeddings import EmbeddingService, build_embedding_service
from app.services.evaluation_service import EvaluationService
from app.services.ocr import OcrEngine, TesseractOcrEngine
from app.services.pdf_service import PdfService
from app.services.plagiarism_service import PlagiarismService
from app.services.question_service import QuestionService
from app.services.rag_service import RagService
from app.utils.concurrency import BoundedConcurrency


@dataclass
class Container:
    settings: Settings
    ai: AIService
    ocr: OcrEngine
    embedder: EmbeddingService
    pdf_limiter: BoundedConcurrency


def build_container(settings: Settings) -> Container:
    return Container(
        settings=settings,
        ai=GeminiAIService(settings),
        ocr=TesseractOcrEngine(settings),
        embedder=build_embedding_service(settings),
        pdf_limiter=BoundedConcurrency(settings.pdf_max_concurrent_extractions),
    )


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_settings(container: Container = Depends(get_container)) -> Settings:
    return container.settings


def get_ai_service(container: Container = Depends(get_container)) -> AIService:
    return container.ai


def get_ocr_engine(container: Container = Depends(get_container)) -> OcrEngine:
    return container.ocr


def get_embedding_service(container: Container = Depends(get_container)) -> EmbeddingService:
    return container.embedder


def get_pdf_service(
    settings: Settings = Depends(get_settings),
    ocr: OcrEngine = Depends(get_ocr_engine),
    container: Container = Depends(get_container),
) -> PdfService:
    return PdfService(settings, ocr, limiter=container.pdf_limiter)


def get_document_loader(
    settings: Settings = Depends(get_settings), pdf: PdfService = Depends(get_pdf_service)
) -> DocumentLoader:
    return DocumentLoader(settings, pdf)


def get_question_service(
    ai: AIService = Depends(get_ai_service), settings: Settings = Depends(get_settings)
) -> QuestionService:
    return QuestionService(ai, settings)


def get_evaluation_service(
    ai: AIService = Depends(get_ai_service), settings: Settings = Depends(get_settings)
) -> EvaluationService:
    return EvaluationService(ai, settings)


def get_batch_evaluator(
    evaluator: EvaluationService = Depends(get_evaluation_service), settings: Settings = Depends(get_settings)
) -> BatchEvaluator:
    return BatchEvaluator(evaluator, settings)


def get_plagiarism_service(
    settings: Settings = Depends(get_settings), embedder: EmbeddingService = Depends(get_embedding_service)
) -> PlagiarismService:
    return PlagiarismService(settings, embedder)


def get_rag_service(
    ai: AIService = Depends(get_ai_service),
    embedder: EmbeddingService = Depends(get_embedding_service),
    settings: Settings = Depends(get_settings),
) -> RagService:
    return RagService(ai, embedder, settings)
