import json
import logging
from types import SimpleNamespace

from google.genai import errors

from app.dependencies import get_ai_service
from app.services.ai_service import GeminiAIService
from tests import pdf_factory as f
from tests.conftest import AppHarness, make_settings, pdf_upload
from tests.helpers import SOURCE

GEN = {"source_text": SOURCE, "num_questions": 2}


# ----------------------------------------------------------- rate limiting
def test_rate_limit_blocks_excess_requests_with_retry_after():
    h = AppHarness(make_settings(rate_limit_enabled=True, rate_limit_requests=3, rate_limit_window_seconds=60))
    codes = [h.client.post("/api/questions/generate", json={}).status_code for _ in range(5)]
    assert codes == [422, 422, 422, 429, 429]
    r = h.client.post("/api/questions/generate", json={})
    body = r.json()
    assert r.headers["retry-after"].isdigit() and int(r.headers["retry-after"]) >= 1
    assert body["error"]["code"] == "rate_limit_exceeded" and body["error"]["source"] == "client"
    assert body["error"]["request_id"] == r.headers["x-request-id"]
    # operational endpoints are exempt
    assert h.client.get("/health").status_code == 200 and h.client.get("/openapi.json").status_code == 200


def test_rate_limit_window_expires():
    from app.middleware.core import RateLimitMiddleware

    now = [0.0]

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    mw = RateLimitMiddleware(app, enabled=True, limit=2, window_seconds=10, clock=lambda: now[0])
    import asyncio

    async def hit():
        sent = []

        async def send(m):
            sent.append(m)

        scope = {"type": "http", "method": "POST", "path": "/api/x", "headers": [], "client": ("1.2.3.4", 1)}
        await mw(scope, None, send)
        return sent[0]["status"]

    assert [asyncio.run(hit()) for _ in range(3)] == [200, 200, 429]
    now[0] = 11.0
    assert asyncio.run(hit()) == 200


def test_rate_limit_is_per_client_and_proxy_header_only_when_trusted():
    h = AppHarness(make_settings(rate_limit_enabled=True, rate_limit_requests=1, trust_proxy_headers=True))
    a = h.client.post("/api/questions/generate", json={}, headers={"x-forwarded-for": "10.0.0.1"}).status_code
    b = h.client.post("/api/questions/generate", json={}, headers={"x-forwarded-for": "10.0.0.2"}).status_code
    again = h.client.post("/api/questions/generate", json={}, headers={"x-forwarded-for": "10.0.0.1"}).status_code
    assert (a, b, again) == (422, 422, 429)
    h2 = AppHarness(make_settings(rate_limit_enabled=True, rate_limit_requests=1))  # spoofed header ignored
    codes = [h2.client.post("/api/questions/generate", json={}, headers={"x-forwarded-for": f"9.9.9.{i}"}).status_code for i in range(2)]
    assert codes == [422, 429]


# -------------------------------------------------------- request-size limits
def test_oversized_request_rejected_by_content_length():
    h = AppHarness(make_settings(max_request_size_mb=0.01))
    r = h.client.post("/api/questions/generate-pdf", files={"file": pdf_upload(b"%PDF-1.4 " + b"A" * 50_000)})
    assert r.status_code == 413 and r.json()["error"]["code"] == "request_too_large"
    assert not h.ai.calls


def test_oversized_json_body_rejected():
    h = AppHarness(make_settings(max_json_body_size_mb=0.001))
    r = h.client.post("/api/plagiarism/check", json={"text1": "a" * 3000, "text2": "b" * 3000})
    assert r.status_code == 413 and r.json()["error"]["details"]["max_size_mb"] == 0.001


def test_streaming_body_without_content_length_is_cut_off():
    h = AppHarness(make_settings(max_request_size_mb=0.01))

    def chunks():
        for _ in range(40):
            yield b"x" * 1024

    r = h.client.post("/api/questions/generate-pdf", content=chunks(), headers={"content-type": "multipart/form-data; boundary=zzz"})
    assert r.status_code == 413 and r.json()["error"]["code"] == "request_too_large"


def test_request_within_limits_is_unaffected():
    h = AppHarness(make_settings(max_request_size_mb=1))
    assert h.client.post("/api/questions/generate", json={"source_text": "short"}).status_code == 422


# -------------------------------------------------------------------- CORS
def test_cors_allows_configured_origins_only():
    h = AppHarness(make_settings(cors_origins="https://app.scorematrix.example, http://localhost:5173", max_request_size_mb=0.01))
    ok = h.client.options(
        "/api/questions/generate",
        headers={"Origin": "https://app.scorematrix.example", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"},
    )
    assert ok.status_code == 200 and ok.headers["access-control-allow-origin"] == "https://app.scorematrix.example"
    bad = h.client.options(
        "/api/questions/generate",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in bad.headers
    # browsers can read error responses (e.g. 413) because CORS wraps the limiter
    r = h.client.post(
        "/api/questions/generate-pdf",
        files={"file": pdf_upload(b"%PDF-" + b"A" * 50_000)},
        headers={"Origin": "http://localhost:5173"},
    )
    assert r.status_code == 413 and r.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_origins_parsed_from_env_style_json_and_csv():
    assert make_settings(cors_origins='["https://a.example","https://b.example"]').cors_origins == ["https://a.example", "https://b.example"]
    assert make_settings(cors_origins="https://a.example,https://b.example").cors_origins == ["https://a.example", "https://b.example"]


# --------------------------------------------------------- secrets and errors
def test_gemini_secret_never_appears_in_responses_or_logs(caplog):
    secret = "AIzaSyD-TOP-SECRET-KEY-0123456789"
    settings = make_settings(gemini_api_key=secret, gemini_max_retries=0)
    h = AppHarness(settings)
    leaky = errors.ClientError(401, {"error": {"message": f"API key not valid: {secret}"}})

    class Models:
        async def generate_content(self, **kw):
            raise leaky

    h.app.dependency_overrides[get_ai_service] = lambda: GeminiAIService(settings, client=SimpleNamespace(aio=SimpleNamespace(models=Models())))
    with caplog.at_level(logging.DEBUG):
        r = h.client.post("/api/questions/generate", json=GEN)
        health = h.client.get("/health")
        cfg = h.client.get("/openapi.json")
    assert r.status_code == 502 and r.json()["error"]["code"] == "ai_authentication_failed"
    for text in (r.text, health.text, cfg.text, caplog.text):
        assert secret not in text
    assert "Traceback" not in r.text


def test_settings_repr_does_not_reveal_key():
    s = make_settings(gemini_api_key="AIza-visible-if-bug")
    assert "AIza-visible-if-bug" not in repr(s) and "AIza-visible-if-bug" not in s.model_dump_json()


def test_logs_never_contain_submitted_content(caplog):
    h = AppHarness()
    marker = "CONFIDENTIAL-STUDENT-ESSAY-TEXT"
    with caplog.at_level(logging.DEBUG):
        h.client.post("/api/plagiarism/check", json={"text1": marker, "text2": marker + " more words"})
        h.client.post("/api/evaluation/evaluate", json={"question": marker, "student_answer": marker, "rubric": [{"name": "x", "max_marks": 1}]})
    assert marker not in caplog.text


def test_filenames_are_sanitised_everywhere(client, ai):
    from app.services.evaluation_service import WireEvaluation
    from tests.helpers import evaluation, RUBRIC

    ai.on(WireEvaluation, evaluation())
    pdf = f.text_pdf(SOURCE)
    r = client.post(
        "/api/evaluation/evaluate-batch",
        files=[("files", ("..\\..\\windows\\system32\\evil.pdf", pdf, "application/pdf")), ("files", ("../../../etc/passwd", b"hello", "text/plain"))],
        data={"question": "Q?", "rubric": json.dumps(RUBRIC)},
    )
    res = r.json()["results"]
    assert [x["filename"] for x in res] == ["evil.pdf", "passwd"]
    assert all("/" not in (x["filename"] or "") and ".." not in (x["filename"] or "") for x in res)


def test_method_not_allowed_and_unknown_paths_use_envelope(client):
    r = client.get("/api/questions/generate")
    assert r.status_code == 405 and r.json()["error"]["code"] == "method_not_allowed"


def test_request_id_is_echoed_only_when_safe(client):
    assert client.get("/health", headers={"X-Request-ID": "abc-123"}).headers["x-request-id"] == "abc-123"
    unsafe = client.get("/health", headers={"X-Request-ID": "bad id\twith spaces"}).headers["x-request-id"]
    assert unsafe != "bad id\twith spaces" and len(unsafe) == 32


def test_dependency_override_replaces_ai_service_with_fake(harness, ai):
    """Documents the testing seam: swapping the AI layer needs no monkeypatching."""
    from app.services.question_service import WireQuestionSet
    from tests.helpers import mcq, question_set

    ai.on(WireQuestionSet, question_set(mcq()))
    r = harness.client.post("/api/questions/generate", json={**GEN, "num_questions": 1, "question_types": ["mcq"]})
    assert r.status_code == 200 and r.json()["model"] == "fake-model" and len(ai.calls) == 1


async def test_size_limit_streaming_path_replaces_inner_error_with_413():
    """No Content-Length header: the guard must trip while streaming, even if the app turns
    the read failure into its own 400 (as FastAPI does for body-parse errors)."""
    from app.middleware.core import RequestSizeLimitMiddleware

    async def inner(scope, receive, send):
        try:
            while True:
                message = await receive()
                if not message.get("more_body"):
                    break
            status, body = 200, b"{}"
        except Exception:  # FastAPI maps body-read errors to 400
            status, body = 400, b'{"detail":"There was an error parsing the body"}'
        await send({"type": "http.response.start", "status": status, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})

    chunks = [{"type": "http.request", "body": b"x" * 600, "more_body": True} for _ in range(5)]
    chunks.append({"type": "http.request", "body": b"", "more_body": False})
    sent = []

    async def receive():
        return chunks.pop(0)

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": "/api/x", "headers": [(b"content-type", b"multipart/form-data; boundary=b")], "state": {}}
    await RequestSizeLimitMiddleware(inner, max_bytes=2000, max_json_bytes=2000)(scope, receive, send)
    assert sent[0]["status"] == 413
    assert b"request_too_large" in sent[1]["body"] and len(sent) == 2
