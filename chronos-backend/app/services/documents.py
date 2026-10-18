"""Turn uploads and raw text into :class:`ExtractionResult` objects.

All endpoints go through this loader so file validation (size, signature,
type, filename sanitising) and PDF/OCR handling are identical everywhere.
"""

from __future__ import annotations

import os

from starlette.datastructures import UploadFile

from app.config import Settings
from app.errors import (
    EmptyFileError,
    FileTooLargeError,
    NoReadableTextError,
    UnsupportedFileTypeError,
)
from app.schemas.documents import ExtractionResult, PageContent
from app.services.pdf_service import PdfService
from app.utils.text import sanitize_filename, sha256_hex, strip_control_chars

_READ_CHUNK = 1024 * 1024
_TEXT_EXTENSIONS = {".txt", ".text", ".md"}


class DocumentLoader:
    def __init__(self, settings: Settings, pdf_service: PdfService) -> None:
        self.settings = settings
        self.pdf = pdf_service

    # ------------------------------------------------------------------ uploads
    async def read_upload(self, upload: UploadFile, max_bytes: int) -> bytes:
        """Read an upload in chunks, aborting as soon as it exceeds ``max_bytes``."""
        buf = bytearray()
        while True:
            chunk = await upload.read(_READ_CHUNK)
            if not chunk:
                break
            buf.extend(chunk)
            if len(buf) > max_bytes:
                raise FileTooLargeError(
                    f"The file is larger than the {max_bytes / (1024 * 1024):g} MB limit.",
                    details={"max_size_mb": round(max_bytes / (1024 * 1024), 3)},
                )
        return bytes(buf)

    async def load_upload(self, upload: UploadFile, *, allow_text: bool = True) -> ExtractionResult:
        """Validate and extract an uploaded PDF (or UTF-8 text file when ``allow_text``)."""
        filename = sanitize_filename(upload.filename, "upload")
        try:
            data = await self.read_upload(upload, self.settings.max_pdf_size_bytes)
        finally:
            await upload.close()
        return await self.load_bytes(
            data, filename=filename, content_type=upload.content_type, allow_text=allow_text
        )

    async def load_bytes(
        self,
        data: bytes,
        *,
        filename: str,
        content_type: str | None = None,
        allow_text: bool = True,
    ) -> ExtractionResult:
        if not data:
            raise EmptyFileError("The uploaded file is empty (0 bytes).")
        kind = self.detect_kind(data, filename, content_type, allow_text=allow_text)
        if kind == "pdf":
            return await self.pdf.extract_async(data, filename)
        if len(data) > self.settings.max_text_file_size_bytes:
            raise FileTooLargeError(
                f"Text files are limited to {self.settings.max_text_file_size_mb:g} MB.",
                details={"max_size_mb": self.settings.max_text_file_size_mb},
            )
        text = self._decode_text(data)
        return self.from_text(text, name=filename, sha=sha256_hex(data))

    @staticmethod
    def detect_kind(
        data: bytes, filename: str, content_type: str | None, *, allow_text: bool = True
    ) -> str:
        """Decide the file type from its *content*; names and MIME types are only hints."""
        if b"%PDF-" in data[:1024]:
            return "pdf"
        ext = os.path.splitext(filename)[1].lower()
        declared_pdf = ext == ".pdf" or (content_type or "").lower() == "application/pdf"
        if declared_pdf:
            raise UnsupportedFileTypeError(
                "The file is named or labelled as a PDF but does not have a valid PDF signature."
            )
        if not allow_text:
            raise UnsupportedFileTypeError("Only PDF files are accepted for this field.")
        if b"\x00" in data[:8192]:
            raise UnsupportedFileTypeError("Binary files are not supported. Upload a PDF or a UTF-8 .txt file.")
        if ext in _TEXT_EXTENSIONS or (content_type or "").lower().startswith("text/"):
            return "text"
        raise UnsupportedFileTypeError("Unsupported file type. Upload a PDF or a UTF-8 .txt file.")

    @staticmethod
    def _decode_text(data: bytes) -> str:
        for encoding in ("utf-8-sig", "utf-16"):
            if encoding == "utf-16" and not data.startswith((b"\xff\xfe", b"\xfe\xff")):
                continue
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise UnsupportedFileTypeError("Text files must be UTF-8 encoded.")

    # --------------------------------------------------------------------- text
    def from_text(self, text: str, name: str = "text", sha: str | None = None) -> ExtractionResult:
        cleaned = strip_control_chars(text.replace("\r\n", "\n").replace("\r", "\n")).strip()
        if not cleaned:
            raise NoReadableTextError("The text is empty.")
        return ExtractionResult(
            filename=sanitize_filename(name, "text"),
            source_type="text",
            text=cleaned,
            pages=[PageContent(page_number=1, text=cleaned, method="text", char_count=len(cleaned))],
            page_count=1,
            pages_processed=1,
            ocr_status="not_needed",
            sha256=sha or sha256_hex(cleaned),
        )
