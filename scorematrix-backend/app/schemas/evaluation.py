"""Schemas for rubric-based answer evaluation (single and batch)."""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.documents import DocumentInfo

Confidence = Literal["high", "medium", "low"]
AssessmentBasis = Literal["evidence_based", "uncertain"]

DISCLAIMER = (
    "This is a provisional AI-generated assessment, not a final grade. AI judgments can be wrong, "
    "especially for OCR-read or handwritten answers. A teacher must review and confirm every mark."
)


class RubricCriterion(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, populate_by_name=True)

    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    max_marks: float = Field(
        gt=0, le=1000, validation_alias=AliasChoices("max_marks", "marks", "max_score", "points")
    )


def check_rubric_total(rubric: list[RubricCriterion], total_marks: float | None) -> None:
    names = [c.name.casefold() for c in rubric]
    if len(set(names)) != len(names):
        raise ValueError("rubric criterion names must be unique")
    if total_marks is not None:
        total = round(sum(c.max_marks for c in rubric), 6)
        if abs(total - total_marks) > 1e-6:
            raise ValueError(
                f"total_marks ({total_marks}) does not equal the sum of rubric criterion marks ({total:g})"
            )


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, populate_by_name=True)

    question: str = Field(min_length=1, max_length=15000)
    student_answer: str = Field(
        min_length=1, max_length=30000, validation_alias=AliasChoices("student_answer", "answer")
    )
    model_answer: str | None = Field(default=None, max_length=15000)
    rubric: list[RubricCriterion] = Field(min_length=1, max_length=30)
    total_marks: float | None = Field(default=None, gt=0, le=10000)
    subject: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def _rubric_consistent(self) -> "EvaluationRequest":
        check_rubric_total(self.rubric, self.total_marks)
        return self


class EvidenceExcerpt(BaseModel):
    text: str
    verified: bool = Field(description="True if this excerpt appears verbatim in the student's answer.")
    pages: list[int] = Field(default_factory=list)


class CriterionResult(BaseModel):
    criterion_id: str
    name: str
    description: str = ""
    max_marks: float
    awarded_marks: float = Field(ge=0)
    feedback: str
    evidence: list[EvidenceExcerpt] = Field(default_factory=list)
    confidence: Confidence
    assessment_basis: AssessmentBasis = Field(
        description="'evidence_based' = marks are backed by at least one verified excerpt (or a clear absence); "
        "'uncertain' = the AI could not back this judgment with verifiable evidence."
    )
    marks_adjusted: bool = Field(
        default=False, description="True if server-side validation had to correct the AI's marks."
    )
    needs_review: bool = False


class EvaluationResponse(BaseModel):
    total_awarded_marks: float
    total_max_marks: float
    percentage: float
    criteria: list[CriterionResult]
    strengths: list[str] = Field(default_factory=list)
    missing_concepts: list[str] = Field(default_factory=list)
    improvement_suggestions: list[str] = Field(default_factory=list)
    overall_feedback: str = ""
    uncertainty_notes: list[str] = Field(default_factory=list)
    needs_teacher_review: bool
    review_reasons: list[str] = Field(default_factory=list)
    final: bool = Field(default=False, description="Always false: results are provisional until a teacher confirms.")
    disclaimer: str = DISCLAIMER
    rubric_source: Literal["provided", "extracted_from_pdf"] = "provided"
    answer_document: DocumentInfo | None = None
    warnings: list[str] = Field(default_factory=list)
    model: str | None = None


# ------------------------------------------------------------------------ batch
class BatchSubmission(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, populate_by_name=True)

    student_id: str | None = Field(default=None, max_length=200)
    student_answer: str = Field(
        max_length=30000, validation_alias=AliasChoices("student_answer", "answer", "answer_text")
    )


class BatchEvaluationRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=15000)
    model_answer: str | None = Field(default=None, max_length=15000)
    rubric: list[RubricCriterion] = Field(min_length=1, max_length=30)
    total_marks: float | None = Field(default=None, gt=0, le=10000)
    submissions: list[BatchSubmission] = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def _rubric_consistent(self) -> "BatchEvaluationRequest":
        check_rubric_total(self.rubric, self.total_marks)
        return self


class BatchItemError(BaseModel):
    code: str
    message: str
    source: Literal["client", "server", "upstream_ai"]


class BatchItemResult(BaseModel):
    index: int
    filename: str | None = None
    student_label: str
    status: Literal["success", "failed", "skipped"]
    result: EvaluationResponse | None = None
    error: BatchItemError | None = None
    duplicate_of: int | None = Field(
        default=None, description="Index of the identical submission whose result is reused (status=skipped)."
    )


class BatchSummary(BaseModel):
    total: int
    succeeded: int
    failed: int
    skipped: int
    needs_teacher_review: int
    average_percentage: float | None = None


class BatchEvaluationResponse(BaseModel):
    results: list[BatchItemResult]
    summary: BatchSummary
    rubric: list[RubricCriterion]
    rubric_source: Literal["provided", "extracted_from_pdf"] = "provided"
    final: bool = False
    disclaimer: str = DISCLAIMER
    warnings: list[str] = Field(default_factory=list)

    @field_validator("results")
    @classmethod
    def _ordered(cls, v: list[BatchItemResult]) -> list[BatchItemResult]:
        return sorted(v, key=lambda r: r.index)
