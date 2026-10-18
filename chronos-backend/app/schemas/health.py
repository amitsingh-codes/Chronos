"""Schemas for the health endpoint."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AICapability(BaseModel):
    provider: str = "gemini"
    configured: bool
    model: str


class OCRCapability(BaseModel):
    enabled: bool
    available: bool
    engine: str
    version: str | None = None
    languages_requested: list[str] = Field(default_factory=list)
    languages_available: list[str] = Field(default_factory=list)
    detail: str


class EmbeddingCapability(BaseModel):
    backend: str
    model: str
    available: bool
    loaded: bool
    semantic: bool = Field(description="False for the lexical 'hashing' fallback backend.")
    detail: str


class PdfCapability(BaseModel):
    text_extraction: bool
    page_rendering: bool


class HealthResponse(BaseModel):
    status: Literal["ok"] = Field(description="Application health. Optional capabilities are reported separately.")
    app: str
    version: str
    environment: str
    ai: AICapability
    ocr: OCRCapability
    embeddings: EmbeddingCapability
    pdf: PdfCapability
    unavailable_features: list[str] = Field(
        default_factory=list, description="Features that will fail or degrade until the listed capability is fixed."
    )
    limits: dict[str, Any]
