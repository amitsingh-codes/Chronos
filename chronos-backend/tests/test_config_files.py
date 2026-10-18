"""Keeps .env.example and .gitignore honest."""

from pathlib import Path

from app.config import Settings

ROOT = Path(__file__).resolve().parent.parent


def test_env_example_documents_every_setting_and_loads_cleanly():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    keys = {line.split("=", 1)[0].strip() for line in text.splitlines() if "=" in line and not line.lstrip().startswith("#")}
    missing = {name.upper() for name in Settings.model_fields} - keys - {"APP_NAME", "APP_VERSION"}
    assert not missing, f"settings missing from .env.example: {sorted(missing)}"
    s = Settings(_env_file=ROOT / ".env.example")  # must parse and pass validation
    assert s.gemini_model == "gemini-3.5-flash" and s.gemini_temperature is None and s.tesseract_cmd is None
    assert s.cors_origins == ["http://localhost:3000", "http://localhost:5173"]


def test_env_example_contains_only_placeholders():
    key_line = next(l for l in (ROOT / ".env.example").read_text().splitlines() if l.startswith("GEMINI_API_KEY="))
    assert key_line == "GEMINI_API_KEY=your-gemini-api-key-here"


def test_gitignore_protects_secrets():
    lines = (ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in lines and ".env.*" in lines and "!.env.example" in lines
