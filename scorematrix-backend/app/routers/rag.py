"""Retrieval-augmented question answering over request-supplied documents."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import ValidationError

from app.config import Settings
from app.dependencies import get_document_loader, get_rag_service, get_settings
from app.errors import ValidationAppError
from app.routers._request import (
    FILE_SCHEMA,
    dual_request_body,
    field_error,
    form_files,
    form_int,
    form_text,
    form_texts,
    load_named,
    multipart_body,
    open_form,
    raise_validation,
    read_json_model,
    request_kind,
)
from app.schemas.rag import RagAskRequest, RagAskResponse, RagOptions
from app.services.documents import DocumentLoader
from app.services.rag_service import RagInputDocument, RagService
from app.utils.text import sanitize_filename

router = APIRouter(prefix="/api/rag", tags=["RAG"])

_MULTIPART = multipart_body(
    {
        "question": {"type": "string"},
        "files": {"type": "array", "items": FILE_SCHEMA, "description": "PDF or UTF-8 .txt source documents."},
        "texts": {"type": "array", "items": {"type": "string"}, "description": "Source documents as plain text."},
        "chunk_size": {"type": "integer", "description": "Characters per chunk (default from RAG_CHUNK_SIZE)."},
        "chunk_overlap": {"type": "integer", "description": "Overlap in characters (default RAG_CHUNK_OVERLAP)."},
        "top_k": {"type": "integer", "description": "Chunks to retrieve (default RAG_TOP_K)."},
    },
    required=["question"],
)


@router.post(
    "/ask",
    response_model=RagAskResponse,
    summary="Ask a question grounded in supplied documents (JSON, or multipart with PDFs)",
    description=(
        "Documents (text or PDF, OCR'd when needed) are chunked, embedded and searched **in memory for this "
        "request only**; nothing is stored. The answer carries page-aware citations whose quotes are verified "
        "against the retrieved text. If the evidence is insufficient the response says so rather than guessing. "
        "Document content is treated as untrusted data; embedded instructions are ignored."
    ),
    openapi_extra=dual_request_body("RagAskRequest", _MULTIPART),
    responses={
        413: {"description": "File or request too large."},
        415: {"description": "Unsupported file type."},
        422: {"description": "Invalid input or unreadable document."},
        429: {"description": "Rate limit exceeded."},
        502: {"description": "The AI provider returned an invalid or rejected response."},
        503: {"description": "AI/embeddings/OCR unavailable."},
    },
)
async def ask(
    request: Request,
    loader: DocumentLoader = Depends(get_document_loader),
    service: RagService = Depends(get_rag_service),
    settings: Settings = Depends(get_settings),
) -> RagAskResponse:
    if request_kind(request) == "json":
        body = await read_json_model(request, RagAskRequest)
        docs = []
        for n, d in enumerate(body.documents or [], start=1):
            doc_id = d.id or f"doc-{n}"
            name = d.name or doc_id
            docs.append(RagInputDocument(doc_id, name, loader.from_text(d.text, name=name)))
        question, size, overlap, top_k = body.question, body.chunk_size, body.chunk_overlap, body.top_k
    else:
        limit = settings.rag_max_documents
        async with open_form(request, max_files=limit + 5) as form:
            try:
                options = RagOptions(
                    question=form_text(form, "question") or "",
                    chunk_size=form_int(form, "chunk_size"),
                    chunk_overlap=form_int(form, "chunk_overlap"),
                    top_k=form_int(form, "top_k"),
                )
            except ValidationError as exc:
                raise_validation(exc, "form")
                raise  # pragma: no cover
            files = form_files(form, "files")
            texts = form_texts(form, "texts")
            if not files and not texts:
                raise field_error("form", "files", "upload at least one document or send 'texts'")
            if len(files) + len(texts) > limit:
                raise ValidationAppError(
                    f"At most {limit} documents are allowed per request.", details={"max_documents": limit}
                )
            docs = []
            for n, f in enumerate(files, start=1):
                extraction = await load_named(loader, f)
                docs.append(RagInputDocument(f"doc-{n}", sanitize_filename(f.filename, f"doc-{n}"), extraction))
            for n, t in enumerate(texts, start=len(files) + 1):
                name = f"text-{n - len(files)}"
                docs.append(RagInputDocument(f"doc-{n}", name, loader.from_text(t, name=name)))
        question, size, overlap, top_k = options.question, options.chunk_size, options.chunk_overlap, options.top_k

    return await service.ask(question, docs, chunk_size=size, chunk_overlap=overlap, top_k=top_k)
