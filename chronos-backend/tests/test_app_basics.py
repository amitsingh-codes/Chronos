import importlib
import json

import pytest

from tests.conftest import AppHarness, make_settings


@pytest.mark.parametrize(
    "module",
    [
        "app.main", "app.config", "app.errors", "app.dependencies", "app.routers.questions",
        "app.routers.evaluation", "app.routers.plagiarism", "app.routers.rag", "app.routers.health",
        "app.services.pdf_service", "app.services.ai_service", "app.services.batch",
    ],
)
def test_modules_import(module):
    importlib.import_module(module)


def test_health_reports_capabilities_without_secrets():
    secret = "AIza-SECRET-KEY-VALUE-123"
    h = AppHarness(make_settings(gemini_api_key=secret))
    r = h.client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["ai"]["configured"] is True
    assert {"ocr", "embeddings", "pdf", "limits", "unavailable_features"} <= body.keys()
    text = r.text
    assert secret not in text and "/home/" not in text and "Traceback" not in text


def test_health_degrades_but_stays_ok_when_ai_missing_and_ocr_unavailable(harness, ocr):
    ocr.available = False
    body = harness.client.get("/health").json()
    assert body["status"] == "ok"
    assert body["ocr"]["available"] is False
    assert "scanned_pdf_ocr" in body["unavailable_features"]


def test_docs_and_openapi_available(client):
    assert client.get("/docs").status_code == 200
    spec = client.get("/openapi.json").json()
    paths = spec["paths"]
    for p in [
        "/health", "/api/questions/generate", "/api/questions/generate-pdf", "/api/evaluation/evaluate",
        "/api/evaluation/evaluate-batch", "/api/plagiarism/check", "/api/rag/ask",
    ]:
        assert p in paths
    # multipart endpoint documents file fields
    props = paths["/api/questions/generate-pdf"]["post"]["requestBody"]["content"]["multipart/form-data"]
    assert props
    dual = paths["/api/evaluation/evaluate"]["post"]["requestBody"]["content"]
    assert set(dual) == {"application/json", "multipart/form-data"}
    assert "EvaluationRequest" in spec["components"]["schemas"]
    assert "RubricCriterion" in spec["components"]["schemas"]


def test_error_envelope_is_consistent_and_hides_internals(client):
    r = client.get("/nope")
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["code"] == "not_found" and body["detail"]
    assert r.headers["x-request-id"]
    r = client.post("/api/questions/generate", json={"source_text": "short"})
    assert r.status_code == 422
    body = r.json()
    assert body["error"]["code"] == "validation_error"
    assert isinstance(body["detail"], list)  # FastAPI-compatible shape preserved
    assert "input" not in json.dumps(body)  # submitted content is never echoed


def test_unhandled_exception_returns_generic_500(harness):
    from app.dependencies import get_ai_service

    def boom():
        raise RuntimeError("database password is hunter2")

    harness.app.dependency_overrides[get_ai_service] = boom
    r = harness.client.post(
        "/api/questions/generate",
        json={"source_text": "x" * 60, "num_questions": 1},
    )
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "internal_error"
    assert "hunter2" not in r.text and "Traceback" not in r.text
