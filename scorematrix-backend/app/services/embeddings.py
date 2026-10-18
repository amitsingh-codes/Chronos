"""Embedding backends.

* ``sentence-transformers`` - real semantic embeddings (default). Lazily loaded
  so the app starts even if the package/model is missing; capability is
  reported by ``/health`` and failures surface as clear 503 errors.
* ``hashing`` - dependency-free *lexical* bag-of-words hashing. It is NOT
  semantic; it exists as a low-resource fallback and for deterministic tests.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import threading
import time
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.config import Settings
from app.errors import EmbeddingsUnavailableError
from app.utils.text import tokenize

logger = logging.getLogger("scorematrix.embeddings")


@dataclass
class EmbeddingStatus:
    available: bool
    loaded: bool
    detail: str


class EmbeddingService(ABC):
    backend: str = "unknown"
    model_name: str = "unknown"
    semantic: bool = True

    @abstractmethod
    def status(self) -> EmbeddingStatus: ...

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an ``(n, d)`` float32 array of L2-normalised vectors (blocking)."""

    async def embed_async(self, texts: list[str]) -> np.ndarray:
        return await asyncio.to_thread(self.embed, texts)


class SentenceTransformerEmbedder(EmbeddingService):
    backend = "sentence-transformers"
    semantic = True

    def __init__(self, settings: Settings) -> None:
        self.model_name = settings.embedding_model
        self._batch_size = settings.embedding_batch_size
        self._model: Any = None
        self._lock = threading.Lock()
        self._failed_at: float | None = None

    _FAILURE_COOLDOWN_SECONDS = 60.0

    def status(self) -> EmbeddingStatus:
        if importlib.util.find_spec("sentence_transformers") is None:
            return EmbeddingStatus(
                False,
                False,
                "The 'sentence-transformers' package is not installed (pip install sentence-transformers).",
            )
        if self._model is not None:
            return EmbeddingStatus(True, True, f"Model '{self.model_name}' loaded.")
        return EmbeddingStatus(
            True, False, f"Package installed; model '{self.model_name}' loads on first use."
        )

    def _load(self) -> Any:
        with self._lock:
            if self._model is not None:
                return self._model
            if self._failed_at is not None and time.monotonic() - self._failed_at < self._FAILURE_COOLDOWN_SECONDS:
                # Offline hosts make the HuggingFace client retry for a long time; don't repeat that per request.
                raise EmbeddingsUnavailableError(self._failure_message())
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise EmbeddingsUnavailableError(
                    "Semantic embeddings need the 'sentence-transformers' package "
                    "(pip install sentence-transformers). For a lexical-only fallback set "
                    "EMBEDDING_BACKEND=hashing."
                ) from None
            try:
                logger.info("Loading embedding model %s", self.model_name)
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:
                logger.error("Embedding model load failed: %s", type(exc).__name__)
                self._failed_at = time.monotonic()
                raise EmbeddingsUnavailableError(self._failure_message()) from None
            self._failed_at = None
            return self._model

    def _failure_message(self) -> str:
        return (
            f"The embedding model '{self.model_name}' could not be loaded. The first run needs internet "
            "access to download it; afterwards it is cached locally (set HF_HUB_OFFLINE=1 to stop network checks)."
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        model = self._load()
        vectors = model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class HashingEmbedder(EmbeddingService):
    """Signed feature hashing of word unigrams and bigrams (lexical, deterministic)."""

    backend = "hashing"
    model_name = "hashing-bow-1024"
    semantic = False

    def __init__(self, dimensions: int = 1024) -> None:
        self.dimensions = dimensions

    def status(self) -> EmbeddingStatus:
        return EmbeddingStatus(
            True,
            True,
            "Lexical hashing embeddings (not semantic): they match shared words, not paraphrases.",
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
            for feat in features:
                h = zlib.crc32(feat.encode("utf-8"))
                out[row, h % self.dimensions] += 1.0 if (h >> 31) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


def build_embedding_service(settings: Settings) -> EmbeddingService:
    if settings.embedding_backend == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(settings)
