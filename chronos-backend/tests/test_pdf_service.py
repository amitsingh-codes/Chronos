import shutil

import pytest

from app.errors import (
    CorruptPdfError, EmptyFileError, EmptyPdfError, EncryptedPdfError, FileTooLargeError,
    NoReadableTextError, OCRUnavailableError, ProcessingTimeoutError, UnsupportedFileTypeError,
)
from app.services.ocr import TesseractOcrEngine
from app.services.pdf_service import PdfService
from tests import pdf_factory as f
from tests.conftest import make_settings
from tests.helpers import SENTENCES, FakeOcr

PAGE1 = " ".join(SENTENCES[:3])
PAGE2 = " ".join(SENTENCES[3:])


def svc(ocr=None, **settings) -> PdfService:
    return PdfService(make_settings(**settings), ocr or FakeOcr())


def test_text_pdf_extracts_pages_without_ocr():
    ocr = FakeOcr()
    r = svc(ocr).extract(f.text_pdf(PAGE1, PAGE2), "bio.pdf")
    assert r.page_count == 2 and r.pages_processed == 2
    assert [p.method for p in r.pages] == ["text", "text"]
    assert r.ocr_status == "not_needed" and not r.ocr_used and not r.truncated
    assert "mitochondria" in r.pages[0].text and "Ribosomes" in r.pages[1].text
    assert r.is_complete and r.sha256 and ocr.calls == 0
    assert "=== Page 2 ===" in r.marked_text()


def test_scanned_pdf_uses_ocr_and_records_provenance():
    ocr = FakeOcr(texts=["Newton's second law says force equals mass times acceleration in physics."])
    r = svc(ocr).extract(f.scanned_pdf("some scanned page"), "scan.pdf")
    assert r.ocr_status == "used" and r.ocr_used and r.ocr_pages == [1]
    assert r.pages[0].method == "ocr" and r.pages[0].ocr_confidence == 93.0
    assert "force equals mass" in r.text
    assert any("OCR was used" in w for w in r.warnings)


def test_mixed_pdf_only_ocrs_pages_that_need_it():
    ocr = FakeOcr(texts=["Recognised text from the image only page about water boiling at sea level."])
    data = f.mixed_pdf(("text", PAGE1), ("image", "scanned"), ("text", PAGE2))
    r = svc(ocr).extract(data, "mixed.pdf")
    assert [p.method for p in r.pages] == ["text", "ocr", "text"]
    assert r.ocr_pages == [2]
    assert ocr.calls == 1, "pages with selectable text must not be OCR'd"
    assert r.ocr_status == "used"
    assert r.text.index("mitochondria") < r.text.index("Recognised") < r.text.index("Ribosomes")


def test_ocr_unavailable_raises_helpful_error_for_scanned_pdf():
    with pytest.raises(OCRUnavailableError) as exc:
        svc(FakeOcr(available=False)).extract(f.scanned_pdf("x"), "scan.pdf")
    assert "TESSERACT_CMD" in exc.value.message and "tesseract" in exc.value.message.lower()
    assert exc.value.status_code == 503


def test_ocr_unavailable_on_mixed_pdf_is_reported_not_hidden():
    data = f.mixed_pdf(("text", PAGE1), ("image", "scanned"))
    r = svc(FakeOcr(available=False)).extract(data, "mixed.pdf")
    assert r.ocr_status == "unavailable" and not r.ocr_used and not r.is_complete
    assert [p.method for p in r.pages] == ["text", "none"]
    assert any("OCR is unavailable" in w for w in r.warnings)


def test_ocr_disabled_by_configuration():
    with pytest.raises(OCRUnavailableError) as exc:
        svc(ocr_enabled=False).extract(f.scanned_pdf("x"), "scan.pdf")
    assert "OCR_ENABLED" in exc.value.message


def test_ocr_returns_nothing_is_not_reported_as_success():
    with pytest.raises(NoReadableTextError) as exc:
        svc(FakeOcr(texts=[""])).extract(f.scanned_pdf("faint"), "scan.pdf")
    assert "OCR ran but found no readable text" in exc.value.message


def test_ocr_failure_on_one_page_gives_partial_status():
    class Flaky(FakeOcr):
        def recognize(self, image, *, dpi, timeout):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("engine crashed")
            return super().recognize(image, dpi=dpi, timeout=timeout)

    r = svc(Flaky()).extract(f.scanned_pdf("a", "b"), "two.pdf")
    assert r.ocr_status == "partial" and r.ocr_pages == [2]
    assert any("OCR failed" in w for w in r.warnings)
    assert not r.is_complete


def test_blank_pdf_has_no_readable_text():
    with pytest.raises(NoReadableTextError):
        svc().extract(f.blank_pdf(2), "blank.pdf")


def test_zero_page_pdf():
    with pytest.raises(EmptyPdfError):
        svc().extract(f.zero_page_pdf(), "none.pdf")


def test_empty_bytes():
    with pytest.raises(EmptyFileError):
        svc().extract(b"", "empty.pdf")


def test_encrypted_pdf():
    with pytest.raises(EncryptedPdfError):
        svc().extract(f.encrypted_pdf(), "locked.pdf")


@pytest.mark.parametrize("data", [f.corrupt_pdf(), f.header_only_pdf()])
def test_corrupt_pdf(data):
    with pytest.raises(CorruptPdfError):
        svc().extract(data, "bad.pdf")


def test_non_pdf_signature_rejected():
    with pytest.raises(UnsupportedFileTypeError):
        svc().extract(b"MZ\x90\x00 this is an executable", "evil.pdf")


def test_oversized_pdf_rejected():
    data = f.text_pdf(PAGE1)
    with pytest.raises(FileTooLargeError):
        svc(max_pdf_size_mb=0.001).extract(data, "big.pdf")


def test_page_limit_truncates_and_says_so():
    r = svc(max_pdf_pages=2).extract(f.text_pdf(PAGE1, PAGE2, PAGE1, PAGE2), "long.pdf")
    assert r.page_count == 4 and r.pages_processed == 2 and r.truncated and not r.is_complete
    assert any("MAX_PDF_PAGES" in w for w in r.warnings)


def test_ocr_page_limit_truncates():
    ocr = FakeOcr()
    r = svc(ocr, ocr_max_pages=1).extract(f.mixed_pdf(("image", "a"), ("image", "b"), ("text", PAGE1)), "m.pdf")
    assert ocr.calls == 1 and r.truncated and r.ocr_status == "partial"
    assert any("OCR_MAX_PAGES" in w for w in r.warnings)


def test_processing_timeout():
    ticks = iter(range(0, 1000, 100))
    s = PdfService(make_settings(pdf_processing_timeout_seconds=50), FakeOcr(), clock=lambda: next(ticks))
    with pytest.raises(ProcessingTimeoutError):
        s.extract(f.text_pdf(PAGE1, PAGE2), "slow.pdf")


def test_unicode_text_preserved():
    text = "Café résumé naïve façade — Ωmega ≈ 3.14 and the mitochondria produce ATP for the whole cell."
    r = svc().extract(f.text_pdf(text), "u.pdf")
    assert "résumé" in r.text  # Latin-1 range survives standard PDF fonts


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract not installed")
def test_real_tesseract_reads_scanned_pdf():
    """Integration check with the real engine (skipped where Tesseract is absent)."""
    engine = TesseractOcrEngine(make_settings())
    assert engine.status().available
    r = PdfService(make_settings(), engine).extract(
        f.scanned_pdf("Newton's second law states that force equals mass times acceleration."), "s.pdf"
    )
    assert r.ocr_status == "used"
    assert "force" in r.text.lower() and "acceleration" in r.text.lower()
    assert r.pages[0].ocr_confidence and r.pages[0].ocr_confidence > 50


def test_real_tesseract_missing_binary_is_detected():
    engine = TesseractOcrEngine(make_settings(tesseract_cmd="/definitely/not/here/tesseract"))
    st = engine.status()
    assert st.available is False and "TESSERACT_CMD" in st.detail
    assert "/definitely" not in st.detail  # configured path is not echoed
