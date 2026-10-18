import asyncio
from types import SimpleNamespace

import pytest
from google.genai import errors
from pydantic import BaseModel

from app.errors import (
    AIAuthenticationError, AIContentBlockedError, AIIncompleteResponseError, AINotConfiguredError,
    AIRateLimitError, AIRequestRejectedError, AIResponseError, AITimeoutError, AIUnavailableError,
)
from app.services.ai_service import GeminiAIService
from tests.conftest import make_settings


class Out(BaseModel):
    answer: str


def response(text='{"answer": "ok"}', finish="STOP", blocked=None):
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=finish))],
        prompt_feedback=SimpleNamespace(block_reason=blocked) if blocked else None,
    )


class Client:
    """Scripted fake of ``genai.Client``: each item is a response or an exception to raise."""

    def __init__(self, *script):
        self.script, self.calls = list(script), 0
        outer = self

        class Models:
            async def generate_content(self, **kwargs):
                outer.calls += 1
                item = outer.script.pop(0)
                if isinstance(item, BaseException):
                    raise item
                return item

        self.aio = SimpleNamespace(models=Models())


def service(client, **settings):
    sleeps = []

    async def fake_sleep(d):
        sleeps.append(d)

    svc = GeminiAIService(make_settings(gemini_api_key="k", **settings), client=client, sleep=fake_sleep, jitter=lambda: 0.0)
    svc.sleeps = sleeps
    return svc


async def run(svc):
    return await svc.generate_structured(system_instruction="s", prompt="p", response_model=Out)


async def test_success_and_code_fence_stripping():
    svc = service(Client(response('```json\n{"answer": "fenced"}\n```')))
    assert (await run(svc)).answer == "fenced"


async def test_transient_errors_retry_with_exponential_backoff_then_succeed():
    server = errors.ServerError(503, {"error": {"message": "overloaded"}})
    client = Client(server, errors.ClientError(429, {}), response())
    svc = service(client, gemini_retry_base_delay_seconds=1.0)
    assert (await run(svc)).answer == "ok"
    assert client.calls == 3
    assert svc.sleeps == [0.5, 1.0]  # base*2**n * (0.5 + jitter/2), jitter=0


async def test_retries_are_bounded():
    client = Client(*[errors.ServerError(500, {})] * 10)
    with pytest.raises(AIUnavailableError):
        await run(service(client, gemini_max_retries=2))
    assert client.calls == 3


async def test_rate_limit_exhaustion_maps_to_rate_limit_error():
    client = Client(*[errors.ClientError(429, {})] * 10)
    with pytest.raises(AIRateLimitError) as exc:
        await run(service(client, gemini_max_retries=1))
    assert exc.value.status_code == 503 and exc.value.source == "upstream_ai"
    assert client.calls == 2


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (errors.ClientError(401, {}), AIAuthenticationError),
        (errors.ClientError(403, {}), AIAuthenticationError),
        (errors.ClientError(400, {"error": {"message": "API key not valid. Please pass a valid API key."}}), AIAuthenticationError),
        (errors.ClientError(400, {"error": {"message": "bad request"}}), AIRequestRejectedError),
        (errors.ClientError(404, {}), AIRequestRejectedError),
    ],
)
async def test_permanent_errors_are_not_retried(exc, expected):
    client = Client(exc, response())
    with pytest.raises(expected):
        await run(service(client))
    assert client.calls == 1


async def test_timeout_is_retried_then_reported():
    client = Client(*[asyncio.TimeoutError()] * 5)
    with pytest.raises(AITimeoutError):
        await run(service(client, gemini_max_retries=1))
    assert client.calls == 2


async def test_malformed_json_gets_one_repair_attempt_then_fails():
    client = Client(response("not json"), response('{"wrong": 1}'))
    with pytest.raises(AIResponseError) as exc:
        await run(service(client, gemini_max_parse_retries=1))
    assert client.calls == 2 and "wrong" not in exc.value.message


async def test_malformed_then_valid_recovers():
    client = Client(response("oops"), response())
    assert (await run(service(client))).answer == "ok"


async def test_truncated_output_is_reported_as_incomplete():
    with pytest.raises(AIIncompleteResponseError):
        await run(service(Client(response('{"answer": "cut', finish="MAX_TOKENS"))))


async def test_blocked_content_is_not_retried():
    client = Client(response(text=None, finish="SAFETY"), response())
    with pytest.raises(AIContentBlockedError):
        await run(service(client))
    assert client.calls == 1


async def test_missing_api_key_fails_before_any_call():
    svc = GeminiAIService(make_settings(gemini_api_key=None))
    with pytest.raises(AINotConfiguredError):
        await run(svc)


async def test_error_messages_never_contain_provider_text_or_key():
    secret = "AIza-super-secret"
    exc = errors.ClientError(401, {"error": {"message": f"bad key {secret}"}})
    with pytest.raises(AIAuthenticationError) as caught:
        await run(service(Client(exc)))
    assert secret not in caught.value.message and secret not in str(caught.value.details)
