"""Rubric evaluation (single + batch). Both accept JSON or multipart at the same path."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import Field, TypeAdapter
from starlette.datastructures import FormData

from app.config import Settings
from app.dependencies import (
    get_batch_evaluator,
    get_document_loader,
    get_evaluation_service,
    get_settings,
)
from app.errors import TooManyFilesError, ValidationAppError
from app.routers._request import (
    FILE_SCHEMA,
    dual_request_body,
    field_error,
    form_file,
    form_files,
    form_float,
    form_json,
    form_text,
    load_named,
    multipart_body,
    open_form,
    read_json_model,
    request_kind,
)
from app.schemas.documents import ExtractionResult
from app.schemas.evaluation import (
    BatchEvaluationRequest,
    BatchEvaluationResponse,
    EvaluationRequest,
    EvaluationResponse,
    RubricCriterion,
    check_rubric_total,
)
from app.services.batch import BatchEvaluator, TextSource, UploadSource
from app.services.documents import DocumentLoader
from app.services.evaluation_service import EvaluationService
from app.utils.text import sanitize_filename

router = APIRouter(prefix="/api/evaluation", tags=["Evaluation"])

_RubricList = Annotated[list[RubricCriterion], Field(min_length=1, max_length=30)]
_RUBRIC_ADAPTER: TypeAdapter[_RubricList] = TypeAdapter(_RubricList)
_LABELS_ADAPTER: TypeAdapter[list[str]] = TypeAdapter(list[str])

_RUBRIC_DOC = (
    "JSON array, e.g. `[{\"name\": \"Accuracy\", \"description\": \"...\", \"max_marks\": 5}]`. "
    "Alternatively upload `rubric_file`; the AI then extracts the rubric and the result is flagged for review."
)
_ERRORS = {
    413: {"description": "File, batch or request too large."},
    415: {"description": "Unsupported file type or content type."},
    422: {"description": "Invalid fields, unreadable PDF, or no text."},
    429: {"description": "Rate limit exceeded."},
    502: {"description": "The AI provider returned an invalid or rejected response."},
    503: {"description": "AI not configured / rate-limited, or OCR required but unavailable."},
    504: {"description": "Processing or AI call timed out."},
}

_EVAL_MULTIPART = multipart_body(
    {
        "answer_file": {**FILE_SCHEMA, "description": "Student answer (PDF or UTF-8 .txt). Or send student_answer."},
        "student_answer": {"type": "string", "description": "Student answer as text."},
        "question": {"type": "string", "description": "Question text. Or upload question_file."},
        "question_file": {**FILE_SCHEMA, "description": "Question paper (PDF or .txt)."},
        "model_answer": {"type": "string"},
        "model_answer_file": {**FILE_SCHEMA, "description": "Model answer (PDF or .txt)."},
        "rubric": {"type": "string", "description": _RUBRIC_DOC},
        "rubric_file": {**FILE_SCHEMA, "description": "Rubric document (PDF or .txt); AI-extracted."},
        "total_marks": {"type": "number", "description": "Optional; must equal the rubric total."},
        "subject": {"type": "string"},
    }
)
_BATCH_MULTIPART = multipart_body(
    {
        "files": {
            "type": "array",
            "items": FILE_SCHEMA,
            "description": "Student answer files (PDF or .txt), one per student.",
        },
        "student_labels": {
            "type": "string",
            "description": 'Optional JSON array of labels aligned with `files`, e.g. ["Asha", "Ravi"]. Default: file names.',
        },
        "question": {"type": "string"},
        "question_file": {**FILE_SCHEMA},
        "model_answer": {"type": "string"},
        "model_answer_file": {**FILE_SCHEMA},
        "rubric": {"type": "string", "description": _RUBRIC_DOC},
        "rubric_file": {**FILE_SCHEMA},
        "total_marks": {"type": "number"},
    },
    required=["files"],
)


# ------------------------------------------------------------------- form helpers
async def _text_or_file(
    form: FormData,
    loader: DocumentLoader,
    settings: Settings,
    *,
    text_field: str,
    file_field: str,
    required: bool,
    max_chars: int,
) -> ExtractionResult | None:
    upload = form_file(form, file_field)
    if upload is not None:
        return await load_named(loader, upload)
    text = form_text(form, text_field)
    if text is not None:
        if len(text) > max_chars:
            raise ValidationAppError(f"'{text_field}' is longer than {max_chars} characters.", details={"field": text_field})
        return loader.from_text(text, name=text_field)
    if required:
        raise field_error("form", text_field, f"provide '{text_field}' or upload '{file_field}'")
    return None


def _clip(doc: ExtractionResult, limit: int) -> str:
    return doc.text if len(doc.text) <= limit else doc.text[:limit]


async def _read_common(
    form: FormData, loader: DocumentLoader, evaluator: EvaluationService, settings: Settings
) -> tuple[str, str | None, list[RubricCriterion], str, list[str]]:
    """question, model_answer, rubric, rubric_source, warnings"""
    warnings: list[str] = []
    question_doc = await _text_or_file(
        form, loader, settings, text_field="question", file_field="question_file",
        required=True, max_chars=settings.eval_max_question_chars,
    )
    assert question_doc is not None
    question = _clip(question_doc, settings.eval_max_question_chars)
    if len(question_doc.text) > settings.eval_max_question_chars:
        warnings.append("The question paper was truncated to the configured length.")
    warnings.extend(question_doc.warnings)
    model_doc = await _text_or_file(
        form, loader, settings, text_field="model_answer", file_field="model_answer_file",
        required=False, max_chars=settings.eval_max_question_chars,
    )
    model_answer = _clip(model_doc, settings.eval_max_question_chars) if model_doc else None
    if model_doc:
        warnings.extend(model_doc.warnings)

    rubric = form_json(form, "rubric", _RUBRIC_ADAPTER)
    source = "provided"
    if rubric is None:
        rubric_file = form_file(form, "rubric_file")
        if rubric_file is None:
            raise field_error("form", "rubric", "provide 'rubric' (JSON) or upload 'rubric_file'")
        rubric = await evaluator.extract_rubric(await load_named(loader, rubric_file))
        source = "extracted_from_pdf"
        warnings.append("The rubric was extracted by AI from the uploaded file. Verify it before relying on the marks.")
    try:
        check_rubric_total(rubric, form_float(form, "total_marks"))
    except ValueError as exc:
        raise field_error("form", "rubric", str(exc)) from None
    return question, model_answer, rubric, source, warnings


# ------------------------------------------------------------------- single
@router.post(
    "/evaluate",
    response_model=EvaluationResponse,
    summary="Evaluate one answer against a rubric (JSON or multipart with PDF)",
    description=(
        "Send `application/json` (text answer) or `multipart/form-data` (answer as PDF/.txt upload, "
        "optionally with the question paper, model answer and rubric as files). The result is a "
        "**provisional** AI assessment: totals are computed server-side, evidence quotes are verified "
        "against the answer, and anything unverifiable is flagged for teacher review."
    ),
    openapi_extra=dual_request_body("EvaluationRequest", _EVAL_MULTIPART),
    responses=_ERRORS,
)
async def evaluate(
    request: Request,
    loader: DocumentLoader = Depends(get_document_loader),
    evaluator: EvaluationService = Depends(get_evaluation_service),
    settings: Settings = Depends(get_settings),
) -> EvaluationResponse:
    if request_kind(request) == "json":
        body = await read_json_model(request, EvaluationRequest)
        answer = loader.from_text(body.student_answer, name="student_answer")
        return await evaluator.evaluate(
            question=body.question,
            answer=answer,
            rubric=body.rubric,
            model_answer=body.model_answer,
            subject=body.subject,
        )
    async with open_form(request) as form:
        question, model_answer, rubric, source, warnings = await _read_common(form, loader, evaluator, settings)
        answer = await _text_or_file(
            form, loader, settings, text_field="student_answer", file_field="answer_file",
            required=True, max_chars=settings.eval_max_answer_chars,
        )
        assert answer is not None
        subject = form_text(form, "subject")
    result = await evaluator.evaluate(
        question=question, answer=answer, rubric=rubric, model_answer=model_answer,
        subject=subject, rubric_source=source,
    )
    result.warnings = warnings + result.warnings
    return result


# --------------------------------------------------------------------- batch
@router.post(
    "/evaluate-batch",
    response_model=BatchEvaluationResponse,
    summary="Evaluate many answers (JSON list or multiple uploaded files)",
    description=(
        "Each submission is validated and processed independently with bounded concurrency; one failure "
        "never discards the others. Results keep the order of submission. Identical files are evaluated "
        "once and reported as `skipped` with `duplicate_of`. The request is synchronous: keep batches "
        "within `BATCH_MAX_FILES` and the per-file time limit."
    ),
    openapi_extra=dual_request_body("BatchEvaluationRequest", _BATCH_MULTIPART),
    responses=_ERRORS,
)
async def evaluate_batch(
    request: Request,
    loader: DocumentLoader = Depends(get_document_loader),
    evaluator: EvaluationService = Depends(get_evaluation_service),
    batch: BatchEvaluator = Depends(get_batch_evaluator),
    settings: Settings = Depends(get_settings),
) -> BatchEvaluationResponse:
    if request_kind(request) == "json":
        body = await read_json_model(request, BatchEvaluationRequest)
        sources = [
            TextSource(sub.student_id or f"student-{n}", sub.student_answer, loader)
            for n, sub in enumerate(body.submissions, start=1)
        ]
        return await batch.run(sources, question=body.question, rubric=body.rubric, model_answer=body.model_answer)

    async with open_form(request, max_files=settings.batch_max_files + 5) as form:
        files = form_files(form, "files")
        if not files:
            raise field_error("form", "files", "upload at least one answer file")
        if len(files) > settings.batch_max_files:
            raise TooManyFilesError(
                f"A batch may contain at most {settings.batch_max_files} files.",
                details={"max_files": settings.batch_max_files, "received": len(files)},
            )
        labels = form_json(form, "student_labels", _LABELS_ADAPTER)
        if labels is not None and len(labels) != len(files):
            raise field_error("form", "student_labels", "must contain exactly one label per uploaded file")
        question, model_answer, rubric, source, warnings = await _read_common(form, loader, evaluator, settings)
        sources = [
            UploadSource(
                f,
                loader,
                label=(labels[i] if labels else None) or _stem(f.filename, i),
            )
            for i, f in enumerate(files)
        ]
        return await batch.run(
            sources, question=question, rubric=rubric, model_answer=model_answer,
            rubric_source=source, warnings=warnings,
        )


def _stem(filename: str | None, index: int) -> str:
    name = sanitize_filename(filename, f"student-{index + 1}")
    return name.rsplit(".", 1)[0] if "." in name else name
