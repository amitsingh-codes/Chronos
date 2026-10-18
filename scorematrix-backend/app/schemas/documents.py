"""Shared structures describing extracted documents (PDF or plain text)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

PageMethod = Literal["text", "ocr", "none"]
OcrStatusValue = Literal["not_needed", "used", "partial", "unavailable", "disabled", "failed"]


class PageContent(BaseModel):
    """Text extracted from a single page, with provenance."""

    page_number: int = Field(ge=1)
    text: str = ""
    method: PageMethod = "text"
    char_count: int = 0
    ocr_confidence: float | None = Field(default=None, description="Mean OCR word confidence, 0-100.")
    warnings: list[str] = Field(default_factory=list)


class PageSummary(BaseModel):
    """Page-level metadata (no text) that is safe to return to clients."""

    page_number: int
    method: PageMethod
    char_count: int
    ocr_confidence: float | None = None


class DocumentInfo(BaseModel):
    """Public description of how a document was read."""

    filename: str
    source_type: Literal["pdf", "text"]
    page_count: int = Field(description="Total pages in the file.")
    pages_processed: int = Field(description="Pages that were actually read (may be < page_count).")
    char_count: int
    ocr_status: OcrStatusValue
    ocr_used: bool
    ocr_pages: list[int] = Field(default_factory=list)
    truncated: bool = False
    pages: list[PageSummary] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """Reusable extraction result: combined text, per-page text, OCR status, warnings."""

    filename: str
    source_type: Literal["pdf", "text"]
    text: str
    pages: list[PageContent]
    page_count: int
    pages_processed: int
    ocr_status: OcrStatusValue = "not_needed"
    ocr_used: bool = False
    ocr_pages: list[int] = Field(default_factory=list)
    truncated: bool = False
    warnings: list[str] = Field(default_factory=list)
    sha256: str | None = None

    @property
    def has_pages(self) -> bool:
        """Whether page numbers are meaningful for citations."""
        return self.source_type == "pdf"

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def is_complete(self) -> bool:
        """False if any page may be missing or unread (limits, OCR unavailable/partial/failed)."""
        return not self.truncated and self.ocr_status in ("not_needed", "used")

    def page_pairs(self) -> list[tuple[int | None, str]]:
        """(page_number-or-None, text) pairs; ``None`` for non-paginated sources."""
        return [(p.page_number if self.has_pages else None, p.text) for p in self.pages]

    def marked_text(self) -> str:
        """Text with explicit page markers, for prompts. Plain text has no markers."""
        if not self.has_pages:
            return self.text
        return "\n\n".join(f"=== Page {p.page_number} ===\n{p.text}" for p in self.pages if p.text.strip())

    def info(self) -> DocumentInfo:
        return DocumentInfo(
            filename=self.filename,
            source_type=self.source_type,
            page_count=self.page_count,
            pages_processed=self.pages_processed,
            char_count=self.char_count,
            ocr_status=self.ocr_status,
            ocr_used=self.ocr_used,
            ocr_pages=self.ocr_pages,
            truncated=self.truncated,
            pages=[
                PageSummary(
                    page_number=p.page_number,
                    method=p.method,
                    char_count=p.char_count,
                    ocr_confidence=p.ocr_confidence,
                )
                for p in self.pages
            ],
            warnings=self.warnings,
        )
