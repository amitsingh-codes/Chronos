"""Question generation: JSON (existing contract) and PDF upload."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import ValidationError

from app.dependencies import get_document_loader, get_question_service
from app.routers._request import load_named, raise_validation
from app.schemas.questions import QuestionGenerationRequest, QuestionGenerationResponse, QuestionOptions
from app.services.documents import DocumentLoader
from app.services.question_service import QuestionService

router = APIRouter(prefix="/api/questions", tags=["Question generation"])

_ERRORS = {
    413: {"description": "PDF or request too large."},
    415: {"description": "Not a PDF (signature check failed) or unsupported media type."},
    422: {"description": "Invalid fields, corrupt/encrypted/empty PDF, no readable text, or insufficient text."},
    429: {"description": "Rate limit exceeded."},
    502: {"description": "The AI provider returned an invalid or rejected response."},
    503: {"description": "AI not configured / rate-limited, or OCR required but unavailable."},
    504: {"description": "Processing or AI call timed out."},
}


@router.post(
    "/generate",
    response_model=QuestionGenerationResponse,
    summary="Generate questions from text (JSON)",
    responses=_ERRORS,
)
async def generate_questions(
    request: QuestionGenerationRequest,
    service: QuestionService = Depends(get_question_service),
    loader: DocumentLoader = Depends(get_document_loader),
) -> QuestionGenerationResponse:
    source = loader.from_text(request.source_text, name="source_text")
    return await service.generate(request, source)


@router.post(
    "/generate-pdf",
    response_model=QuestionGenerationResponse,
    summary="Generate questions from an uploaded PDF (multipart/form-data)",
    description=(
        "Upload a PDF (selectable text, scanned, or mixed). Pages without readable text are OCR'd when "
        "Tesseract is available. Only `application/pdf` content with a valid `%PDF-` signature is accepted; "
        "the size/page limits are configured by `MAX_PDF_SIZE_MB` and `MAX_PDF_PAGES`."
    ),
    responses=_ERRORS,
)
async def generate_questions_from_pdf(
    file: UploadFile = File(..., description="PDF file."),
    num_questions: int = Form(10, ge=1, le=50),
    difficulty: str = Form("medium", description="easy | medium | hard"),
    question_types: list[str] | None = Form(
        None,
        description="Repeat the field or use commas: mcq, short_answer, long_answer, fill_in_the_blanks. Default: all.",
    ),
    total_marks: int | None = Form(None, ge=1),
    subject: str | None = Form(None, max_length=200),
    instructions: str | None = Form(None, max_length=2000),
    service: QuestionService = Depends(get_question_service),
    loader: DocumentLoader = Depends(get_document_loader),
) -> QuestionGenerationResponse:
    try:
        options = QuestionOptions(
            num_questions=num_questions,
            difficulty=difficulty,
            question_types=[t for t in (question_types or []) if t.strip()] or None,
            total_marks=total_marks,
            subject=subject,
            instructions=instructions,
        )
    except ValidationError as exc:
        raise_validation(exc, "form")
        raise  # pragma: no cover
    source = await load_named(loader, file, allow_text=False)
    return await service.generate(options, source)
