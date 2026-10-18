"""OCR engine abstraction with a Tesseract implementation.

OCR availability is *verified at runtime* (binary present, language data
installed) rather than assumed from the Python package being importable.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings

logger = logging.getLogger("scorematrix.ocr")

OCR_HELP = (
    "To enable OCR install Tesseract (Windows: the UB-Mannheim installer, "
    "https://github.com/UB-Mannheim/tesseract/wiki; Ubuntu/Debian: `sudo apt install tesseract-ocr`; "
    "macOS: `brew install tesseract`), then either add it to PATH or set TESSERACT_CMD in .env, "
    "and restart the server. Check GET /health -> ocr for the current status."
)


class OcrError(RuntimeError):
    """Raised when OCR of a single image fails."""


@dataclass
class OcrResult:
    text: str
    confidence: float | None = None  # mean word confidence, 0-100


@dataclass
class OcrStatus:
    available: bool
    engine: str
    detail: str
    version: str | None = None
    languages_requested: list[str] = field(default_factory=list)
    languages_available: list[str] = field(default_factory=list)
    languages_used: list[str] = field(default_factory=list)


class OcrEngine(ABC):
    @abstractmethod
    def status(self) -> OcrStatus:
        """Whether OCR can actually run right now."""

    @abstractmethod
    def recognize(self, image: Any, *, dpi: int, timeout: float) -> OcrResult:
        """OCR a PIL image. Raises :class:`OcrError` on failure."""


class TesseractOcrEngine(OcrEngine):
    _STATUS_TTL_SECONDS = 60.0

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._cached: tuple[float, OcrStatus] | None = None

    # ------------------------------------------------------------------ status
    def status(self) -> OcrStatus:
        with self._lock:
            now = time.monotonic()
            if self._cached and now - self._cached[0] < self._STATUS_TTL_SECONDS:
                return self._cached[1]
            status = self._probe()
            self._cached = (now, status)
            return status

    def _probe(self) -> OcrStatus:
        requested = self._settings.ocr_language_list or ["eng"]
        base = OcrStatus(available=False, engine="tesseract", detail="", languages_requested=requested)
        try:
            import pytesseract
        except ImportError:
            base.detail = "The 'pytesseract' Python package is not installed (pip install pytesseract). " + OCR_HELP
            return base
        if self._settings.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self._settings.tesseract_cmd
        try:
            version = str(pytesseract.get_tesseract_version())
        except Exception as exc:  # TesseractNotFoundError, OSError, ...
            logger.info("Tesseract probe failed: %s", type(exc).__name__)
            base.detail = "The Tesseract OCR executable was not found or could not be run. " + OCR_HELP
            return base
        base.version = version
        try:
            installed = sorted(set(pytesseract.get_languages(config="")) - {"osd"})
        except Exception as exc:
            logger.info("Tesseract language probe failed: %s", type(exc).__name__)
            installed = []
        base.languages_available = installed
        used = [lang for lang in requested if lang in installed] if installed else list(requested)
        missing = [lang for lang in requested if installed and lang not in installed]
        if not used:
            base.detail = (
                f"Tesseract {version} is installed but none of the requested languages "
                f"({'+'.join(requested)}) have language data. Install the matching traineddata files "
                "or change OCR_LANGUAGES."
            )
            return base
        base.available = True
        base.languages_used = used
        base.detail = f"Tesseract {version} ready (languages: {'+'.join(used)})."
        if missing:
            base.detail += f" Ignored languages without data: {', '.join(missing)}."
        return base

    # --------------------------------------------------------------- recognise
    def recognize(self, image: Any, *, dpi: int, timeout: float) -> OcrResult:
        status = self.status()
        if not status.available:
            raise OcrError("OCR engine is not available")
        import pytesseract

        try:
            data = pytesseract.image_to_data(
                image,
                lang="+".join(status.languages_used),
                config=f"--dpi {int(dpi)}",
                output_type=pytesseract.Output.DICT,
                timeout=timeout,
            )
        except Exception as exc:
            raise OcrError(f"Tesseract failed: {type(exc).__name__}") from exc
        return _assemble(data)


def _assemble(data: dict[str, list]) -> OcrResult:
    """Rebuild text (lines/paragraphs) and mean confidence from Tesseract TSV data."""
    lines: list[tuple[tuple[int, int], list[str]]] = []
    confs: list[float] = []
    current_key: tuple[int, int, int] | None = None
    for i, raw in enumerate(data.get("text", [])):
        word = (raw or "").strip()
        try:
            conf = float(data["conf"][i])
        except (KeyError, ValueError, IndexError):
            conf = -1.0
        if not word or conf < 0:
            continue
        key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
        confs.append(conf)
        if key != current_key:
            lines.append(((key[0], key[1]), [word]))
            current_key = key
        else:
            lines[-1][1].append(word)
    out: list[str] = []
    prev_par: tuple[int, int] | None = None
    for par, words in lines:
        if prev_par is not None and par != prev_par:
            out.append("")
        out.append(" ".join(words))
        prev_par = par
    mean = round(sum(confs) / len(confs), 1) if confs else None
    return OcrResult(text="\n".join(out).strip(), confidence=mean)
