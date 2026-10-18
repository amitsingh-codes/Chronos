"""Schemas for plagiarism / similarity checking."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.documents import DocumentInfo

Assessment = Literal[
    "exact_duplicate",
    "high_lexical_overlap",
    "moderate_lexical_overlap",
    "semantic_similarity_only",
    "low_similarity",
    "insufficient_text",
]


class PlagiarismDocumentInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    id: str | None = Field(default=None, max_length=100)
    name: str | None = Field(default=None, max_length=200)
    text: str = Field(max_length=500_000)


class PlagiarismCheckRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    documents: list[PlagiarismDocumentInput] | None = Field(default=None, max_length=50)
    text1: str | None = Field(default=None, max_length=500_000, description="Shorthand for a two-document check.")
    text2: str | None = Field(default=None, max_length=500_000)
    exclude_text: str | None = Field(
        default=None,
        max_length=100_000,
        description="Text to ignore when scoring (e.g. the assignment prompt or a shared template).",
    )
    use_semantic: bool = True

    @model_validator(mode="after")
    def _normalise(self) -> "PlagiarismCheckRequest":
        if not self.documents:
            if self.text1 is None or self.text2 is None:
                raise ValueError("Provide 'documents' (2 or more) or both 'text1' and 'text2'.")
            self.documents = [
                PlagiarismDocumentInput(id="text1", name="text1", text=self.text1),
                PlagiarismDocumentInput(id="text2", name="text2", text=self.text2),
            ]
        if len(self.documents) < 2:
            raise ValueError("At least two documents are required for comparison.")
        return self


class LexicalSimilarity(BaseModel):
    shingle_jaccard: float
    containment_a_in_b: float = Field(description="Share of A's word n-grams that also occur in B.")
    containment_b_in_a: float
    matched_sentence_ratio_a: float = Field(description="Share of A's sentences with an exact/near match in B.")
    matched_sentence_ratio_b: float
    shingle_size: int


class SemanticSimilarity(BaseModel):
    available: bool
    score: float | None = None
    model: str | None = None
    note: str


class PlagiarismMatch(BaseModel):
    kind: Literal["exact", "near_duplicate", "semantic"]
    score: float
    document_a_excerpt: str
    document_b_excerpt: str
    document_a_pages: list[int] = Field(default_factory=list)
    document_b_pages: list[int] = Field(default_factory=list)


class PairResult(BaseModel):
    document_a: str
    document_b: str
    assessment: Assessment
    exact_duplicate: bool
    reliability: Literal["normal", "low"]
    lexical: LexicalSimilarity | None = None
    semantic: SemanticSimilarity | None = None
    matches: list[PlagiarismMatch] = Field(default_factory=list)
    explanation: str
    warnings: list[str] = Field(default_factory=list)


class DocumentSummary(BaseModel):
    id: str
    name: str
    word_count: int
    status: Literal["ok", "insufficient_text", "empty"]
    duplicate_of: str | None = None
    boilerplate_sentences_ignored: int = 0
    document: DocumentInfo
    warnings: list[str] = Field(default_factory=list)


class PlagiarismSummary(BaseModel):
    pairs_compared: int
    pairs_flagged_for_review: int
    highest_lexical_containment: float


class PlagiarismCheckResponse(BaseModel):
    documents: list[DocumentSummary]
    pairs: list[PairResult]
    summary: PlagiarismSummary
    scope: str
    limitations: list[str]
    warnings: list[str] = Field(default_factory=list)
