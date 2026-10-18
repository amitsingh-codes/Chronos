"""Schemas for retrieval-augmented question answering."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.documents import DocumentInfo


class RagDocumentInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    id: str | None = Field(default=None, max_length=100)
    name: str | None = Field(default=None, max_length=200)
    text: str = Field(min_length=1, max_length=500_000)


class RagOptions(BaseModel):
    """Fields shared by the JSON and multipart forms of ``/api/rag/ask``."""

    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=3, max_length=2000)
    chunk_size: int | None = Field(default=None, ge=100, le=8000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=4000)
    top_k: int | None = Field(default=None, ge=1, le=100)

    @model_validator(mode="after")
    def _overlap_smaller(self) -> "RagOptions":
        if self.chunk_size is not None and self.chunk_overlap is not None and self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class RagAskRequest(RagOptions):
    documents: list[RagDocumentInput] | None = Field(default=None, max_length=50)
    source_text: str | None = Field(default=None, max_length=500_000, description="Shorthand for one document.")

    @model_validator(mode="after")
    def _normalise(self) -> "RagAskRequest":
        if not self.documents:
            if not self.source_text:
                raise ValueError("Provide 'documents' or 'source_text'.")
            self.documents = [RagDocumentInput(id="document-1", name="source_text", text=self.source_text)]
        return self


class Citation(BaseModel):
    source_id: str = Field(description="Identifier such as 'S1' matching an entry in 'retrieved'.")
    document_id: str
    document_name: str
    pages: list[int] = Field(default_factory=list)
    excerpt: str
    quote: str | None = Field(default=None, description="Short passage the model said supports the answer.")
    quote_verified: bool = Field(description="True if 'quote' appears verbatim in the cited source chunk.")


class RetrievedChunk(BaseModel):
    source_id: str
    document_id: str
    document_name: str
    pages: list[int] = Field(default_factory=list)
    text: str
    score: float


class IndexInfo(BaseModel):
    chunks: int
    chunk_size: int
    chunk_overlap: int
    top_k: int
    embedding_backend: str
    embedding_model: str
    scope: str


class RagDocumentInfo(BaseModel):
    id: str
    name: str
    chunks: int
    document: DocumentInfo


class RagAskResponse(BaseModel):
    question: str
    answer: str
    sufficient_evidence: bool
    grounded: bool = Field(description="True if the answer carries at least one verified citation.")
    confidence: Literal["high", "medium", "low", "none"]
    citations: list[Citation]
    retrieved: list[RetrievedChunk]
    documents: list[RagDocumentInfo]
    index: IndexInfo
    warnings: list[str] = Field(default_factory=list)
    model: str | None = None
