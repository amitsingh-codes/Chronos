"""Reusable PDF text extraction with page-aware OCR fallback.

Pipeline (per request):

1. Validate size and ``%PDF-`` signature, open with pypdf, reject corrupt /
   encrypted / zero-page files.
2. Extract selectable text page by page.
3. Decide per page whether the text is readable. *Only* pages that are not
   (image-only pages, garbled text) are rendered with pypdfium2 and OCR'd.
4. Combine pages, preserving page numbers and provenance, and report whether
   OCR was used, unavailable, partial or failed - never claiming success it
   did not achieve.
"""

from __future__ import annotations

import asyncio
import io
import logging
import math
import re
import time
from typing import Any

from app.config import Settings
from app.errors import (
    CorruptPdfError,
    EmptyFileError,
    EmptyPdfError,
    EncryptedPdfError,
    FileTooLargeError,
    NoReadableTextError,
    OCRUnavailableError,
    ProcessingTimeoutError,
    UnsupportedFileTypeError,
)
from app.schemas.documents import ExtractionResult, OcrStatusValue, PageContent
from app.services.ocr import OCR_HELP, OcrEngine
from app.utils.concurrency import BoundedConcurrency
from app.utils.text import alnum_count, is_readable, sanitize_filename, sha256_hex, strip_control_chars

logger = logging.getLogger("scorematrix.pdf")

_BLANK_STDDEV = 2.0
_LOW_OCR_CONFIDENCE = 60.0


def clean_extracted_text(raw: str) -> str:
    text = strip_control_chars(raw.replace("\r\n", "\n").replace("\r", "\n"))
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class PdfService:
    def __init__(
        self,
        settings: Settings,
        ocr: OcrEngine,
        limiter: BoundedConcurrency | None = None,
        clock=time.monotonic,
    ) -> None:
        self.settings = settings
        self.ocr = ocr
        self.limiter = limiter or BoundedConcurrency(settings.pdf_max_concurrent_extractions)
        self._clock = clock

    # ------------------------------------------------------------------ public
    async def extract_async(self, data: bytes, filename: str = "document.pdf") -> ExtractionResult:
        """Run extraction in a worker thread under the shared concurrency limit."""
        async with self.limiter:
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(self.extract, data, filename),
                    timeout=self.settings.pdf_processing_timeout_seconds + 10,
                )
            except asyncio.TimeoutError:
                raise ProcessingTimeoutError(
                    f"PDF processing exceeded {self.settings.pdf_processing_timeout_seconds:g} seconds."
                ) from None

    def extract(self, data: bytes, filename: str = "document.pdf") -> ExtractionResult:
        s = self.settings
        name = sanitize_filename(filename, "document.pdf")
        if not data:
            raise EmptyFileError("The uploaded file is empty (0 bytes).")
        if len(data) > s.max_pdf_size_bytes:
            raise FileTooLargeError(
                f"The PDF is larger than the {s.max_pdf_size_mb:g} MB limit.",
                details={"max_size_mb": s.max_pdf_size_mb},
            )
        if b"%PDF-" not in data[:1024]:
            raise UnsupportedFileTypeError("The file does not look like a PDF (missing %PDF header).")

        deadline = self._clock() + s.pdf_processing_timeout_seconds
        reader = self._open_reader(data)
        try:
            page_count = len(reader.pages)
        except Exception as exc:
            logger.info("pdf page tree unreadable: %s", type(exc).__name__)
            raise CorruptPdfError() from None
        if page_count == 0:
            raise EmptyPdfError()

        warnings: list[str] = []
        truncated = False
        limit = min(page_count, s.max_pdf_pages)
        if page_count > limit:
            truncated = True
            warnings.append(
                f"The PDF has {page_count} pages; only the first {limit} were processed "
                f"(MAX_PDF_PAGES={s.max_pdf_pages})."
            )

        pages: list[PageContent] = []
        needs_ocr: list[int] = []
        for i in range(limit):
            self._check_deadline(deadline)
            page_warnings: list[str] = []
            try:
                raw = reader.pages[i].extract_text() or ""
            except Exception as exc:
                logger.info("text extraction failed on page %d: %s", i + 1, type(exc).__name__)
                raw = ""
                page_warnings.append("Selectable-text extraction failed for this page.")
            text = clean_extracted_text(raw)
            pages.append(
                PageContent(
                    page_number=i + 1,
                    text=text,
                    method="text" if text else "none",
                    char_count=len(text),
                    warnings=page_warnings,
                )
            )
            if not is_readable(text, s.min_chars_per_page):
                needs_ocr.append(i)

        ocr_status: OcrStatusValue = "not_needed"
        ocr_pages: list[int] = []
        if needs_ocr:
            ocr_status, ocr_pages, ocr_truncated = self._run_ocr(data, pages, needs_ocr, deadline, warnings)
            truncated = truncated or ocr_truncated

        total_alnum = sum(alnum_count(p.text) for p in pages)
        if total_alnum == 0:
            self._raise_no_text(ocr_status, len(needs_ocr))

        unread = [p.page_number for p in pages if not p.text.strip()]
        if unread and ocr_status in ("used", "not_needed", "partial", "failed"):
            warnings.append(
                f"{len(unread)} page(s) yielded no text: {_fmt_pages(unread)}. They may be blank or illegible."
            )

        return ExtractionResult(
            filename=name,
            source_type="pdf",
            text="\n\n".join(p.text for p in pages if p.text),
            pages=pages,
            page_count=page_count,
            pages_processed=limit,
            ocr_status=ocr_status,
            ocr_used=bool(ocr_pages),
            ocr_pages=ocr_pages,
            truncated=truncated,
            warnings=warnings,
            sha256=sha256_hex(data),
        )

    def capabilities(self) -> dict[str, bool]:
        try:
            import pypdfium2  # noqa: F401

            rendering = True
        except ImportError:
            rendering = False
        return {"text_extraction": True, "page_rendering": rendering}

    # ---------------------------------------------------------------- internals
    def _check_deadline(self, deadline: float) -> None:
        if self._clock() > deadline:
            raise ProcessingTimeoutError(
                f"PDF processing exceeded {self.settings.pdf_processing_timeout_seconds:g} seconds."
            )

    def _open_reader(self, data: bytes) -> Any:
        from pypdf import PdfReader
        from pypdf import PasswordType

        try:
            reader = PdfReader(io.BytesIO(data), strict=False)
        except Exception as exc:
            logger.info("pypdf could not open file: %s", type(exc).__name__)
            raise CorruptPdfError() from None
        try:
            if reader.is_encrypted:
                try:
                    result = reader.decrypt("")
                except Exception as exc:  # e.g. missing 'cryptography' for AES
                    logger.info("pdf decrypt failed: %s", type(exc).__name__)
                    raise EncryptedPdfError(
                        "The PDF is encrypted and could not be opened. Remove the password or "
                        "encryption and upload it again."
                    ) from None
                if result == PasswordType.NOT_DECRYPTED:
                    raise EncryptedPdfError()
        except EncryptedPdfError:
            raise
        except Exception as exc:
            logger.info("pdf encryption check failed: %s", type(exc).__name__)
            raise CorruptPdfError() from None
        return reader

    def _raise_no_text(self, ocr_status: OcrStatusValue, pages_needing_ocr: int) -> None:
        if ocr_status == "unavailable":
            raise OCRUnavailableError(
                "This PDF has no selectable text, so OCR is required, but OCR is not available on the server. "
                + OCR_HELP,
                details={"pages_needing_ocr": pages_needing_ocr},
            )
        if ocr_status == "disabled":
            raise OCRUnavailableError(
                "This PDF has no selectable text, so OCR is required, but OCR is disabled "
                "(OCR_ENABLED=false or OCR_MAX_PAGES=0). Enable it in the server configuration.",
                details={"pages_needing_ocr": pages_needing_ocr},
            )
        if ocr_status in ("failed", "partial", "used"):
            raise NoReadableTextError(
                "OCR ran but found no readable text. The scan may be blank, too low-resolution or handwritten "
                "in a way OCR cannot read.",
                details={"ocr_status": ocr_status, "pages_needing_ocr": pages_needing_ocr},
            )
        raise NoReadableTextError("The PDF contains no readable text (it appears to be blank).")

    def _run_ocr(
        self,
        data: bytes,
        pages: list[PageContent],
        needs_ocr: list[int],
        deadline: float,
        warnings: list[str],
    ) -> tuple[OcrStatusValue, list[int], bool]:
        s = self.settings
        numbers = [i + 1 for i in needs_ocr]
        if not s.ocr_enabled or s.ocr_max_pages == 0:
            warnings.append(
                f"{len(needs_ocr)} page(s) have little or no selectable text ({_fmt_pages(numbers)}) "
                "and OCR is disabled, so they were not read."
            )
            return "disabled", [], False
        status = self.ocr.status()
        if not status.available:
            warnings.append(
                f"{len(needs_ocr)} page(s) need OCR ({_fmt_pages(numbers)}) but OCR is unavailable: {status.detail}"
            )
            return "unavailable", [], False

        truncated = False
        candidates = needs_ocr[: s.ocr_max_pages]
        skipped = needs_ocr[s.ocr_max_pages :]
        if skipped:
            truncated = True
            warnings.append(
                f"OCR page limit reached (OCR_MAX_PAGES={s.ocr_max_pages}); "
                f"pages {_fmt_pages([i + 1 for i in skipped])} were not OCR'd."
            )

        try:
            import pypdfium2 as pdfium

            pdf = pdfium.PdfDocument(data)
        except Exception as exc:
            logger.warning("pdfium could not open document for OCR: %s", type(exc).__name__)
            warnings.append("The PDF could not be rendered for OCR.")
            return "failed", [], truncated

        ocr_pages: list[int] = []
        blank: list[int] = []
        failures = 0
        try:
            for i in candidates:
                self._check_deadline(deadline)
                page = pages[i]
                image = None
                try:
                    image = self._render(pdf, i)
                    if _is_blank(image):
                        blank.append(i)
                        page.warnings.append("Page appears blank; OCR skipped.")
                        continue
                    remaining = max(1.0, deadline - self._clock())
                    result = self.ocr.recognize(
                        image,
                        dpi=int(round(getattr(image, "info", {}).get("dpi", (s.ocr_dpi,))[0])),
                        timeout=min(s.ocr_timeout_seconds_per_page, remaining),
                    )
                except ProcessingTimeoutError:
                    raise
                except Exception as exc:
                    failures += 1
                    logger.info("OCR failed on page %d: %s", i + 1, type(exc).__name__)
                    page.warnings.append("OCR failed for this page.")
                    continue
                finally:
                    if image is not None:
                        image.close()
                text = clean_extracted_text(result.text)
                if alnum_count(text) > alnum_count(page.text):
                    page.text = text
                    page.method = "ocr"
                    page.char_count = len(text)
                    page.ocr_confidence = result.confidence
                    ocr_pages.append(i + 1)
                    if result.confidence is not None and result.confidence < _LOW_OCR_CONFIDENCE:
                        page.warnings.append(
                            f"Low OCR confidence ({result.confidence:.0f}/100); text may contain errors."
                        )
                else:
                    page.warnings.append("OCR found no additional readable text on this page.")
        finally:
            pdf.close()

        if blank:
            warnings.append(f"Blank page(s) skipped: {_fmt_pages([i + 1 for i in blank])}.")
        if failures:
            warnings.append(f"OCR failed on {failures} page(s).")
        if ocr_pages:
            warnings.append(
                f"OCR was used for page(s) {_fmt_pages(ocr_pages)}; OCR text can contain recognition errors."
            )

        pending = [i for i in needs_ocr if i not in blank]
        if not pending:
            return "not_needed", ocr_pages, truncated
        produced = len(ocr_pages)
        if produced == len(pending):
            return "used", ocr_pages, truncated
        if produced > 0:
            return "partial", ocr_pages, truncated
        return "failed", ocr_pages, truncated

    def _render(self, pdf: Any, index: int) -> Any:
        s = self.settings
        page = pdf[index]
        try:
            width, height = page.get_size()
            scale = s.ocr_dpi / 72.0
            if width * height * scale * scale > s.ocr_max_image_pixels:
                scale = math.sqrt(s.ocr_max_image_pixels / (width * height))
            bitmap = page.render(scale=scale)
            try:
                image = bitmap.to_pil().convert("L")
            finally:
                bitmap.close()
            image.info["dpi"] = (scale * 72.0, scale * 72.0)
            return image
        finally:
            page.close()


def _is_blank(image: Any) -> bool:
    from PIL import ImageStat

    return ImageStat.Stat(image).stddev[0] < _BLANK_STDDEV


def _fmt_pages(numbers: list[int]) -> str:
    shown = ", ".join(str(n) for n in numbers[:15])
    return shown + (", ..." if len(numbers) > 15 else "")
