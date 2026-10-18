"""Centralised AI access.

Services depend on the abstract :class:`AIService`; the Gemini implementation
lives here and a fake can be injected in tests. Responsibilities:

* read credentials/model/params from settings (never hardcoded, never logged)
* request schema-constrained JSON and validate it with Pydantic
* bounded retries with exponential backoff + jitter for *transient* failures
  only (429 / 5xx / timeouts / connection errors)
* map every failure to a typed error that distinguishes upstream-AI problems
  from validation problems and user mistakes
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.errors import (
    AIAuthenticationError,
    AIContentBlockedError,
    AIError,
    AIIncompleteResponseError,
    AINotConfiguredError,
    AIRateLimitError,
    AIRequestRejectedError,
    AIResponseError,
    AITimeoutError,
    AIUnavailableError,
)
from app.utils.concurrency import BoundedConcurrency

logger = logging.getLogger("scorematrix.ai")

T = TypeVar("T", bound=BaseModel)

_BLOCKED_FINISH_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "IMAGE_SAFETY"}


@dataclass
class AIStatus:
    provider: str
    configured: bool
    model: str


class AIService(ABC):
    @abstractmethod
    async def generate_structured(
        self,
        *,
        system_instruction: str,
        prompt: str,
        response_model: type[T],
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> T:
        """Return a validated instance of ``response_model`` or raise an :class:`AIError`."""

    @abstractmethod
    def status(self) -> AIStatus: ...


# ----------------------------------------------------------------- classification
@dataclass
class _Failure:
    retryable: bool
    error: AIError


def _status_code(exc: BaseException) -> int | None:
    for attr in ("code", "status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and 100 <= value <= 599:
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def classify_exception(exc: BaseException) -> _Failure:
    """Decide whether a provider exception is transient and which typed error it maps to."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)) or type(exc).__name__ in {
        "ReadTimeout",
        "ConnectTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "TimeoutException",
    }:
        return _Failure(True, AITimeoutError())
    status = _status_code(exc)
    message = str(getattr(exc, "message", "") or exc).lower()
    if status is not None:
        if status == 429:
            return _Failure(True, AIRateLimitError())
        if status == 408:
            return _Failure(True, AITimeoutError())
        if status >= 500:
            return _Failure(True, AIUnavailableError())
        if status in (401, 403) or (status == 400 and "api key" in message and "valid" in message):
            return _Failure(False, AIAuthenticationError())
        if status == 404:
            return _Failure(
                False,
                AIRequestRejectedError(
                    "The configured Gemini model was not found or does not support this request. "
                    "Check GEMINI_MODEL."
                ),
            )
        return _Failure(False, AIRequestRejectedError())
    if isinstance(exc, (ConnectionError, OSError)) or type(exc).__name__ in {
        "ConnectError",
        "TransportError",
        "RemoteProtocolError",
        "NetworkError",
    }:
        return _Failure(True, AIUnavailableError())
    return _Failure(False, AIError("The AI provider call failed unexpectedly."))


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL | re.IGNORECASE)
    return fenced.group(1) if fenced else text


# ------------------------------------------------------------------------ Gemini
class GeminiAIService(AIService):
    def __init__(
        self,
        settings: Settings,
        client: Any | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self.settings = settings
        self._client = client
        self._sleep = sleep
        self._jitter = jitter
        self._limiter = BoundedConcurrency(settings.gemini_max_concurrency)

    # ----------------------------------------------------------------- plumbing
    def status(self) -> AIStatus:
        return AIStatus("gemini", self._configured(), self.settings.gemini_model)

    def _configured(self) -> bool:
        return self._client is not None or self.settings.gemini_api_key is not None

    def _get_client(self) -> Any:
        if self._client is None:
            if self.settings.gemini_api_key is None:
                raise AINotConfiguredError()
            try:
                from google import genai
            except ImportError:
                raise AIError("The 'google-genai' package is not installed on the server.") from None
            self._client = genai.Client(api_key=self.settings.gemini_api_key.get_secret_value())
        return self._client

    def _backoff(self, attempt: int) -> float:
        s = self.settings
        delay = min(s.gemini_retry_max_delay_seconds, s.gemini_retry_base_delay_seconds * (2**attempt))
        return delay * (0.5 + self._jitter() / 2)

    def _build_config(
        self, system_instruction: str, response_model: type[BaseModel], temperature: float | None, max_tokens: int | None
    ) -> Any:
        from google.genai import types

        s = self.settings
        kwargs: dict[str, Any] = {
            "system_instruction": system_instruction,
            "response_mime_type": "application/json",
            "response_schema": response_model,
            "max_output_tokens": max_tokens or s.gemini_max_output_tokens,
        }
        temp = temperature if temperature is not None else s.gemini_temperature
        if temp is not None:
            kwargs["temperature"] = temp
        if s.gemini_thinking_level:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=s.gemini_thinking_level)
        return types.GenerateContentConfig(**kwargs)

    # --------------------------------------------------------------- public API
    async def generate_structured(
        self,
        *,
        system_instruction: str,
        prompt: str,
        response_model: type[T],
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> T:
        client = self._get_client()  # raises AINotConfiguredError before any work
        config = self._build_config(system_instruction, response_model, temperature, max_output_tokens)
        attempts = self.settings.gemini_max_parse_retries + 1
        last_error: AIResponseError | None = None
        for parse_attempt in range(attempts):
            text = await self._call_with_retries(client, prompt, config)
            try:
                return self._parse(text, response_model)
            except AIResponseError as exc:
                last_error = exc
                logger.warning("Malformed AI output (attempt %d/%d)", parse_attempt + 1, attempts)
        assert last_error is not None
        raise last_error

    # ----------------------------------------------------------------- internals
    async def _call_with_retries(self, client: Any, prompt: str, config: Any) -> str:
        s = self.settings
        total = s.gemini_max_retries + 1
        for attempt in range(total):
            try:
                async with self._limiter:
                    response = await asyncio.wait_for(
                        client.aio.models.generate_content(
                            model=s.gemini_model, contents=prompt, config=config
                        ),
                        timeout=s.gemini_timeout_seconds,
                    )
                return self._extract_text(response)
            except AIError:
                raise  # already typed (blocked / incomplete / empty) - never retried
            except Exception as exc:
                failure = classify_exception(exc)
                logger.warning(
                    "Gemini call failed: type=%s status=%s retryable=%s attempt=%d/%d",
                    type(exc).__name__,
                    _status_code(exc),
                    failure.retryable,
                    attempt + 1,
                    total,
                )
                if not failure.retryable or attempt == total - 1:
                    raise failure.error from None
                await self._sleep(self._backoff(attempt))
        raise AIUnavailableError()  # pragma: no cover - loop always returns or raises

    @staticmethod
    def _extract_text(response: Any) -> str:
        candidates = getattr(response, "candidates", None) or []
        finish = ""
        if candidates:
            raw = getattr(candidates[0], "finish_reason", None)
            finish = (getattr(raw, "name", None) or str(raw or "")).split(".")[-1].upper()
        text = None
        try:
            text = getattr(response, "text", None)
        except Exception:  # SDK raises when no text parts exist
            text = None
        block_reason = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
        if not text:
            if block_reason or finish in _BLOCKED_FINISH_REASONS:
                raise AIContentBlockedError()
            if finish == "MAX_TOKENS":
                raise AIIncompleteResponseError(
                    "The AI response hit the output-token limit before producing anything usable. "
                    "Request fewer items or raise GEMINI_MAX_OUTPUT_TOKENS."
                )
            raise AIResponseError("The AI provider returned an empty response.")
        if finish == "MAX_TOKENS":
            raise AIIncompleteResponseError(
                "The AI response was truncated at the output-token limit. Request fewer items "
                "or raise GEMINI_MAX_OUTPUT_TOKENS."
            )
        return text

    @staticmethod
    def _parse(text: str, response_model: type[T]) -> T:
        try:
            payload = json.loads(_strip_code_fences(text))
        except json.JSONDecodeError:
            raise AIResponseError("The AI response was not valid JSON.") from None
        try:
            return response_model.model_validate(payload)
        except ValidationError as exc:
            fields = sorted({".".join(str(p) for p in e["loc"]) for e in exc.errors()})[:10]
            raise AIResponseError(
                "The AI response did not match the expected structure.",
                details={"fields": fields},
            ) from None
