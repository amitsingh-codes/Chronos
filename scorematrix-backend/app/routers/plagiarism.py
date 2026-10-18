"""Plagiarism / similarity checking across texts and PDFs."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.config import Settings
from app.dependencies import get_document_loader, get_plagiarism_service, get_settings
from app.errors import NoReadableTextError, ValidationAppError
from app.routers._request import (
    FILE_SCHEMA,
    dual_request_body,
    form_bool,
    form_files,
    form_text,
    form_texts,
    load_named,
    multipart_body,
    open_form,
    read_json_model,
    request_kind,
)
from app.schemas.documents import ExtractionResult
from app.schemas.plagiarism import PlagiarismCheckRequest, PlagiarismCheckResponse
from app.services.documents import DocumentLoader
from app.services.plagiarism_service import InputDocument, PlagiarismService
from app.utils.text import sanitize_filename

router = APIRouter(prefix="/api/plagiarism", tags=["Plagiarism"])

_MULTIPART = multipart_body(
    {
        "files": {
            "type": "array",
            "items": FILE_SCHEMA,
            "description": "PDF or UTF-8 .txt documents to compare.",
        },
        "texts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Additional documents as plain text (repeat the field).",
        },
        "exclude_text": {"type": "string", "description": "Text to ignore, e.g. the assignment prompt."},
        "use_semantic": {"type": "boolean", "default": True},
    }
)


def _empty(name: str, reason: str) -> ExtractionResult:
    return ExtractionResult(
        filename=name,
        source_type="pdf" if name.lower().endswith(".pdf") else "text",
        text="",
        pages=[],
        page_count=0,
        pages_processed=0,
        ocr_status="not_needed",
        warnings=[reason],
    )


async def _text_document(loader: DocumentLoader, doc_id: str, name: str, text: str) -> InputDocument:
    try:
        extraction = loader.from_text(text, name=name)
    except NoReadableTextError:
        extraction = _empty(name, "The text is empty.")
    return InputDocument(id=doc_id, name=name, extraction=extraction)


async def _file_document(loader: DocumentLoader, upload, doc_id: str) -> InputDocument:
    name = sanitize_filename(upload.filename, doc_id)
    try:
        extraction = await load_named(loader, upload)
    except NoReadableTextError as exc:  # unreadable/blank files are reported per document, not fatal
        extraction = _empty(name, exc.message)
    return InputDocument(id=doc_id, name=name, extraction=extraction)


@router.post(
    "/check",
    response_model=PlagiarismCheckResponse,
    summary="Compare two or more documents (JSON, or multipart with PDFs)",
    description=(
        "Compares the submitted documents **with each other only**. Reports exact/near-duplicate sentences, "
        "word n-gram overlap and (separately) semantic embedding similarity, with page references for PDFs. "
        "No internet or academic database is searched, and a score is never a verdict of plagiarism."
    ),
    openapi_extra=dual_request_body("PlagiarismCheckRequest", _MULTIPART),
    responses={
        413: {"description": "File or request too large."},
        415: {"description": "Unsupported file type."},
        422: {"description": "Invalid input, too many documents, or OCR required but unavailable."},
        429: {"description": "Rate limit exceeded."},
    },
)
async def check_plagiarism(
    request: Request,
    loader: DocumentLoader = Depends(get_document_loader),
    service: PlagiarismService = Depends(get_plagiarism_service),
    settings: Settings = Depends(get_settings),
) -> PlagiarismCheckResponse:
    if request_kind(request) == "json":
        body = await read_json_model(request, PlagiarismCheckRequest)
        docs: list[InputDocument] = []
        for n, d in enumerate(body.documents or [], start=1):
            doc_id = d.id or f"doc-{n}"
            docs.append(await _text_document(loader, doc_id, d.name or doc_id, d.text))
        exclude, semantic = body.exclude_text, body.use_semantic
    else:
        limit = settings.plagiarism_max_documents
        async with open_form(request, max_files=limit + 5) as form:
            files = form_files(form, "files")
            texts = form_texts(form, "texts")
            if len(files) + len(texts) < 2:
                raise ValidationAppError("Provide at least two documents (files and/or texts).")
            if len(files) + len(texts) > limit:
                raise ValidationAppError(
                    f"At most {limit} documents can be compared per request.", details={"max_documents": limit}
                )
            docs = []
            for n, f in enumerate(files, start=1):
                docs.append(await _file_document(loader, f, f"doc-{n}"))
            for n, t in enumerate(texts, start=len(files) + 1):
                docs.append(await _text_document(loader, f"doc-{n}", f"text-{n - len(files)}", t))
            exclude = form_text(form, "exclude_text")
            semantic = form_bool(form, "use_semantic", True)

    ids = [d.id for d in docs]
    if len(set(ids)) != len(ids):
        raise ValidationAppError("Document ids must be unique.")
    return await service.check(docs, exclude_text=exclude, use_semantic=semantic)
