"""Batch evaluation with bounded concurrency, per-file isolation and deduplication."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.config import Settings
from app.errors import AppError, EmptyFileError, TooManyFilesError, ValidationAppError
from app.schemas.documents import ExtractionResult
from app.schemas.evaluation import (
    BatchEvaluationResponse,
    BatchItemError,
    BatchItemResult,
    BatchSummary,
    RubricCriterion,
)
from app.services.evaluation_service import EvaluationService
from app.utils.concurrency import BoundedConcurrency
from app.utils.text import sanitize_filename, sha256_hex

logger = logging.getLogger("scorematrix.batch")


class BatchSource(ABC):
    """One student submission. Heavy work (read/extract) is deferred until a slot is free."""

    label: str
    filename: str | None

    @abstractmethod
    async def fingerprint(self) -> str:
        """Content hash used to detect identical submissions within one request."""

    @abstractmethod
    async def extract(self) -> ExtractionResult: ...


@dataclass
class _Entry:
    index: int
    future: "asyncio.Future[BatchItemResult]"


def _to_error(exc: Exception) -> BatchItemError:
    if isinstance(exc, AppError):
        return BatchItemError(code=exc.code, message=exc.message, source=exc.source)  # type: ignore[arg-type]
    logger.error("Unexpected batch item failure: %s", type(exc).__name__, exc_info=True)
    return BatchItemError(
        code="internal_error",
        message="An unexpected error occurred while processing this submission.",
        source="server",
    )


class BatchEvaluator:
    def __init__(self, evaluator: EvaluationService, settings: Settings) -> None:
        self.evaluator = evaluator
        self.settings = settings

    async def run(
        self,
        sources: list[BatchSource],
        *,
        question: str,
        rubric: list[RubricCriterion],
        model_answer: str | None = None,
        rubric_source: str = "provided",
        warnings: list[str] | None = None,
    ) -> BatchEvaluationResponse:
        s = self.settings
        if not sources:
            raise ValidationAppError("Provide at least one student answer.")
        if len(sources) > s.batch_max_files:
            raise TooManyFilesError(
                f"A batch may contain at most {s.batch_max_files} submissions.",
                details={"max_files": s.batch_max_files, "received": len(sources)},
            )

        limiter = BoundedConcurrency(s.batch_concurrency)
        seen: dict[str, _Entry] = {}
        loop = asyncio.get_running_loop()

        async def process(index: int, src: BatchSource) -> BatchItemResult:
            base = dict(index=index, filename=src.filename, student_label=src.label)
            entry: _Entry | None = None
            try:
                async with limiter:
                    digest = await src.fingerprint()
                    original = seen.get(digest)
                    if original is None:
                        entry = _Entry(index, loop.create_future())
                        seen[digest] = entry
                    else:
                        reused = await original.future
                        if reused.status == "success":
                            return BatchItemResult(
                                **base, status="skipped", result=reused.result, duplicate_of=original.index
                            )
                        return BatchItemResult(
                            **base, status="failed", error=reused.error, duplicate_of=original.index
                        )
                    document = await src.extract()
                    evaluation = await self.evaluator.evaluate(
                        question=question,
                        answer=document,
                        rubric=rubric,
                        model_answer=model_answer,
                        rubric_source=rubric_source,
                    )
                    result = BatchItemResult(**base, status="success", result=evaluation)
            except Exception as exc:  # noqa: BLE001 - isolate every per-file failure
                result = BatchItemResult(**base, status="failed", error=_to_error(exc))
            if entry is not None and not entry.future.done():
                entry.future.set_result(result)
            return result

        async def guarded(index: int, src: BatchSource) -> BatchItemResult:
            try:
                return await asyncio.wait_for(process(index, src), timeout=s.batch_per_file_timeout_seconds)
            except asyncio.TimeoutError:
                entry = next((e for e in seen.values() if e.index == index), None)
                failed = BatchItemResult(
                    index=index,
                    filename=src.filename,
                    student_label=src.label,
                    status="failed",
                    error=BatchItemError(
                        code="file_timeout",
                        message=f"Processing this submission exceeded {s.batch_per_file_timeout_seconds:g} seconds.",
                        source="server",
                    ),
                )
                if entry is not None and not entry.future.done():
                    entry.future.set_result(failed)
                return failed

        results = await asyncio.gather(*(guarded(i, src) for i, src in enumerate(sources)))
        results = sorted(results, key=lambda r: r.index)  # deterministic order regardless of completion order

        ok = [r for r in results if r.status == "success" and r.result]
        summary = BatchSummary(
            total=len(results),
            succeeded=len(ok),
            failed=sum(1 for r in results if r.status == "failed"),
            skipped=sum(1 for r in results if r.status == "skipped"),
            needs_teacher_review=sum(1 for r in ok if r.result and r.result.needs_teacher_review),
            average_percentage=round(sum(r.result.percentage for r in ok if r.result) / len(ok), 2) if ok else None,
        )
        return BatchEvaluationResponse(
            results=results,
            summary=summary,
            rubric=rubric,
            rubric_source=rubric_source,  # type: ignore[arg-type]
            warnings=warnings or [],
        )


# ------------------------------------------------------------------ source types
class TextSource(BatchSource):
    """A submission supplied as JSON text."""

    def __init__(self, label: str, text: str, loader) -> None:
        self.label = label
        self.filename = None
        self._text = text
        self._loader = loader

    async def fingerprint(self) -> str:
        return sha256_hex(" ".join(self._text.split()))

    async def extract(self) -> ExtractionResult:
        return self._loader.from_text(self._text, name=self.label)


class UploadSource(BatchSource):
    """An uploaded answer file. Bytes are read only when a concurrency slot is free."""

    def __init__(self, upload, loader, label: str) -> None:
        self.upload = upload
        self._loader = loader
        self.label = label
        self.filename = sanitize_filename(upload.filename, "upload")
        self._data: bytes | None = None

    async def fingerprint(self) -> str:
        try:
            self._data = await self._loader.read_upload(self.upload, self._loader.settings.max_pdf_size_bytes)
        finally:
            await self.upload.close()
        if not self._data:
            raise EmptyFileError("The uploaded file is empty (0 bytes).")
        return sha256_hex(self._data)

    async def extract(self) -> ExtractionResult:
        data, self._data = self._data, None  # release memory once handed to extraction
        assert data is not None
        return await self._loader.load_bytes(
            data, filename=self.filename or "upload", content_type=self.upload.content_type
        )
