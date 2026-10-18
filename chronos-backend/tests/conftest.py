from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.dependencies import get_ai_service, get_embedding_service, get_ocr_engine
from app.main import create_app
from app.schemas.questions import QuestionGenerationRequest  # noqa: F401  (import check)
from app.services.embeddings import HashingEmbedder
from tests.helpers import FakeAI, FakeOcr


def make_settings(**overrides) -> Settings:
    defaults = dict(
        gemini_api_key=None,
        embedding_backend="hashing",
        rate_limit_enabled=False,
        gemini_retry_base_delay_seconds=0.0,
        rag_min_similarity=0.05,
        log_level="WARNING",
    )
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


class AppHarness:
    """An app with the AI, OCR and embedding layers replaced by test doubles."""

    def __init__(self, settings: Settings | None = None, *, ocr: FakeOcr | None = None, embedder=None):
        self.settings = settings or make_settings()
        self.ai = FakeAI()
        self.ocr = ocr or FakeOcr()
        self.embedder = embedder or HashingEmbedder()
        self.app = create_app(self.settings)
        self.app.dependency_overrides[get_ai_service] = lambda: self.ai
        self.app.dependency_overrides[get_ocr_engine] = lambda: self.ocr
        self.app.dependency_overrides[get_embedding_service] = lambda: self.embedder
        self.client = TestClient(self.app, raise_server_exceptions=False)


@pytest.fixture
def harness() -> AppHarness:
    return AppHarness()


@pytest.fixture
def client(harness: AppHarness) -> TestClient:
    return harness.client


@pytest.fixture
def ai(harness: AppHarness) -> FakeAI:
    return harness.ai


@pytest.fixture
def ocr(harness: AppHarness) -> FakeOcr:
    return harness.ocr


def pdf_upload(data: bytes, name: str = "sample.pdf", mime: str = "application/pdf"):
    return (name, data, mime)
