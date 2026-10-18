"""Schemas for AI question generation."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.documents import DocumentInfo


class Difficulty(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class QuestionType(str, Enum):
    mcq = "mcq"
    short_answer = "short_answer"
    long_answer = "long_answer"
    fill_in_the_blanks = "fill_in_the_blanks"


_TYPE_ALIASES: dict[str, QuestionType] = {}
for _qt, _names in {
    QuestionType.mcq: ["mcq", "mcqs", "multiple_choice", "multiple_choice_question", "multiple_choice_questions"],
    QuestionType.short_answer: ["short_answer", "short_answers", "short", "short_answer_question"],
    QuestionType.long_answer: ["long_answer", "long_answers", "long", "essay", "descriptive"],
    QuestionType.fill_in_the_blanks: [
        "fill_in_the_blanks",
        "fill_in_the_blank",
        "fill_in_blanks",
        "fill_blanks",
        "fill_blank",
        "fill_ups",
        "fib",
    ],
}.items():
    for _n in _names:
        _TYPE_ALIASES[_n] = _qt


def normalize_question_type(value: Any) -> QuestionType:
    """Accept ``"MCQ"``, ``"Short Answer"``, ``"fill-in-the-blanks"`` etc."""
    if isinstance(value, QuestionType):
        return value
    key = re.sub(r"[\s\-]+", "_", str(value).strip().lower())
    try:
        return _TYPE_ALIASES[key]
    except KeyError:
        accepted = ", ".join(t.value for t in QuestionType)
        raise ValueError(f"Unknown question type {value!r}. Accepted: {accepted}") from None


class QuestionOptions(BaseModel):
    """Options shared by the JSON and PDF question-generation endpoints."""

    model_config = ConfigDict(str_strip_whitespace=True)

    num_questions: int = Field(default=10, ge=1, le=50)
    difficulty: Difficulty = Difficulty.medium
    question_types: list[QuestionType] = Field(
        default_factory=lambda: list(QuestionType),
        min_length=1,
        description="mcq, short_answer, long_answer, fill_in_the_blanks (case/spacing-insensitive aliases accepted).",
    )
    total_marks: int | None = Field(default=None, ge=1, le=10000)
    subject: str | None = Field(default=None, max_length=200)
    instructions: str | None = Field(default=None, max_length=2000)

    @field_validator("difficulty", mode="before")
    @classmethod
    def _difficulty_case(cls, v: Any) -> Any:
        return v.strip().lower() if isinstance(v, str) else v

    @field_validator("question_types", mode="before")
    @classmethod
    def _types(cls, v: Any) -> Any:
        if v is None:
            return list(QuestionType)
        if isinstance(v, str):
            v = [part for part in v.split(",") if part.strip()]
        if not isinstance(v, (list, tuple)):
            raise ValueError("question_types must be a list")
        flat: list[Any] = []
        for item in v:
            if isinstance(item, str) and "," in item:
                flat.extend(p for p in item.split(",") if p.strip())
            else:
                flat.append(item)
        seen: list[QuestionType] = []
        for item in flat:
            qt = normalize_question_type(item)
            if qt not in seen:
                seen.append(qt)
        return seen

    @field_validator("subject", "instructions", mode="after")
    @classmethod
    def _blank_to_none(cls, v: str | None) -> str | None:
        return v or None

    @model_validator(mode="after")
    def _marks_cover_questions(self) -> "QuestionOptions":
        if self.total_marks is not None and self.total_marks < self.num_questions:
            raise ValueError(
                "total_marks must be at least num_questions so every question can carry at least 1 mark"
            )
        return self


class QuestionGenerationRequest(QuestionOptions):
    source_text: str = Field(min_length=50, max_length=30000)


class MCQOption(BaseModel):
    label: str
    text: str


class GeneratedQuestion(BaseModel):
    id: int
    type: QuestionType
    question: str
    options: list[MCQOption] | None = None
    correct_option: str | None = Field(default=None, description="Label (A, B, ...) of the correct MCQ option.")
    correct_answer: str
    explanation: str
    marks: int = Field(ge=1)
    difficulty: Difficulty
    source_excerpt: str = Field(description="Passage from the source material that supports the answer.")
    source_pages: list[int] = Field(default_factory=list, description="PDF pages containing the excerpt.")
    evidence_verified: bool = Field(
        description="True only if the excerpt was found verbatim (after whitespace/case normalisation) in the extracted source text."
    )


class QuestionGenerationResponse(BaseModel):
    questions: list[GeneratedQuestion]
    num_requested: int
    num_generated: int
    total_marks: int
    requested_total_marks: int | None = None
    difficulty: Difficulty
    subject: str | None = None
    source: DocumentInfo
    warnings: list[str] = Field(default_factory=list)
    model: str | None = None
