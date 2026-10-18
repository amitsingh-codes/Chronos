"""Application settings, loaded from environment variables / a local ``.env`` file.

Nothing in this module (or anywhere else in the app) hardcodes secrets or
machine-specific paths. Every limit that protects the server is configurable.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ app
    app_name: str = "ScoreMatrix API"
    app_version: str = "2.0.0"
    environment: str = "development"
    log_level: str = "INFO"
    cors_origins: Annotated[list[str], NoDecode] = [
        "http://localhost:3000",
        "http://localhost:5173",
    ]

    # --------------------------------------------------------------- Gemini
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.5-flash"
    # Google recommends leaving temperature at the model default for Gemini 3.x.
    gemini_temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    gemini_max_output_tokens: int = Field(default=16384, ge=256, le=65536)
    # Gemini 3.x uses thinking_level ("minimal" | "low" | "medium" | "high").
    gemini_thinking_level: str | None = None
    gemini_timeout_seconds: float = Field(default=90.0, gt=0)
    gemini_max_retries: int = Field(default=3, ge=0, le=8)
    gemini_max_parse_retries: int = Field(default=1, ge=0, le=3)
    gemini_retry_base_delay_seconds: float = Field(default=1.0, ge=0)
    gemini_retry_max_delay_seconds: float = Field(default=20.0, ge=0)
    gemini_max_concurrency: int = Field(default=4, ge=1, le=64)

    # ------------------------------------------------------ request limits
    max_request_size_mb: float = Field(default=64.0, gt=0)
    max_json_body_size_mb: float = Field(default=5.0, gt=0)
    rate_limit_enabled: bool = True
    rate_limit_requests: int = Field(default=60, ge=1)
    rate_limit_window_seconds: float = Field(default=60.0, gt=0)
    trust_proxy_headers: bool = False

    # ------------------------------------------------------------------ PDF
    max_pdf_size_mb: float = Field(default=20.0, gt=0)
    max_text_file_size_mb: float = Field(default=2.0, gt=0)
    max_pdf_pages: int = Field(default=100, ge=1)
    pdf_processing_timeout_seconds: float = Field(default=180.0, gt=0)
    pdf_max_concurrent_extractions: int = Field(default=2, ge=1, le=32)
    # A page needs OCR when it has fewer than this many letters/digits.
    min_chars_per_page: int = Field(default=25, ge=1)

    # ------------------------------------------------------------------ OCR
    ocr_enabled: bool = True
    ocr_languages: str = "eng"
    ocr_dpi: int = Field(default=200, ge=72, le=600)
    ocr_max_pages: int = Field(default=30, ge=0)
    ocr_timeout_seconds_per_page: float = Field(default=60.0, gt=0)
    ocr_max_image_pixels: int = Field(default=40_000_000, ge=1_000_000)
    tesseract_cmd: str | None = None

    # ----------------------------------------------------------- embeddings
    embedding_backend: Literal["sentence-transformers", "hashing"] = "sentence-transformers"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_batch_size: int = Field(default=32, ge=1)
    embedding_max_texts: int = Field(default=4000, ge=1)

    # ------------------------------------------------------------ questions
    question_max_source_chars: int = Field(default=30000, ge=100)
    question_require_verified_evidence: bool = False

    # ----------------------------------------------------------- evaluation
    eval_max_answer_chars: int = Field(default=30000, ge=100)
    eval_max_question_chars: int = Field(default=15000, ge=100)
    batch_max_files: int = Field(default=20, ge=1)
    batch_concurrency: int = Field(default=3, ge=1, le=32)
    batch_per_file_timeout_seconds: float = Field(default=240.0, gt=0)

    # ----------------------------------------------------------- plagiarism
    plagiarism_max_documents: int = Field(default=10, ge=2, le=50)
    plagiarism_max_chars_per_document: int = Field(default=200_000, ge=1000)
    plagiarism_shingle_size: int = Field(default=5, ge=2, le=12)
    plagiarism_min_words: int = Field(default=30, ge=1)
    plagiarism_semantic_threshold: float = Field(default=0.85, gt=0, le=1)
    plagiarism_near_duplicate_threshold: float = Field(default=0.85, gt=0, le=1)
    plagiarism_max_matches: int = Field(default=25, ge=1)

    # ------------------------------------------------------------------ RAG
    rag_chunk_size: int = Field(default=1000, ge=100, le=8000)
    rag_chunk_overlap: int = Field(default=200, ge=0)
    rag_top_k: int = Field(default=5, ge=1, le=50)
    rag_max_top_k: int = Field(default=20, ge=1, le=100)
    rag_max_documents: int = Field(default=10, ge=1, le=50)
    rag_max_chunks: int = Field(default=2000, ge=1)
    rag_max_chars_per_document: int = Field(default=500_000, ge=1000)
    rag_min_similarity: float = Field(default=0.25, ge=-1.0, le=1.0)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value):
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("["):
                import json

                return json.loads(value)
            return [v.strip() for v in value.split(",") if v.strip()]
        return value

    @field_validator("gemini_api_key", mode="before")
    @classmethod
    def _blank_key_is_none(cls, value):
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("tesseract_cmd", "gemini_thinking_level", "gemini_temperature", mode="before")
    @classmethod
    def _blank_is_none(cls, value):
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> "Settings":
        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")
        if self.rag_top_k > self.rag_max_top_k:
            raise ValueError("RAG_TOP_K must not exceed RAG_MAX_TOP_K")
        return self

    # --------------------------------------------------------- derived values
    @property
    def max_request_size_bytes(self) -> int:
        return int(self.max_request_size_mb * 1024 * 1024)

    @property
    def max_json_body_bytes(self) -> int:
        return int(self.max_json_body_size_mb * 1024 * 1024)

    @property
    def max_pdf_size_bytes(self) -> int:
        return int(self.max_pdf_size_mb * 1024 * 1024)

    @property
    def max_text_file_size_bytes(self) -> int:
        return int(self.max_text_file_size_mb * 1024 * 1024)

    @property
    def ocr_language_list(self) -> list[str]:
        return [lang for lang in self.ocr_languages.replace(",", "+").split("+") if lang.strip()]
