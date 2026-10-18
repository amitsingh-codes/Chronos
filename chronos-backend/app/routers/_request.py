"""Helpers for endpoints that accept either JSON or multipart/form-data at one path."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, TypeVar

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, TypeAdapter, ValidationError
from starlette.datastructures import FormData, UploadFile

from app.errors import BadRequestError, UnsupportedMediaTypeError, ValidationAppError

M = TypeVar("M", bound=BaseModel)

FILE_SCHEMA = {"type": "string", "format": "binary"}
FILES_SCHEMA = {"type": "array", "items": FILE_SCHEMA}


def request_kind(request: Request) -> str:
    ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if ctype == "application/json":
        return "json"
    if ctype in ("multipart/form-data", "application/x-www-form-urlencoded"):
        return "multipart"  # text-only forms are fine too; files require multipart
    raise UnsupportedMediaTypeError(
        "Send either application/json or multipart/form-data.",
        details={"received": ctype or None},
    )


def raise_validation(exc: ValidationError, where: str) -> None:
    raise RequestValidationError(
        [{"loc": (where, *e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    ) from None


def field_error(where: str, field: str, message: str) -> RequestValidationError:
    return RequestValidationError([{"loc": (where, field), "msg": message, "type": "value_error"}])


async def read_json_model(request: Request, model: type[M]) -> M:
    try:
        payload = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise BadRequestError("The request body is not valid JSON.") from None
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise_validation(exc, "body")
        raise  # pragma: no cover


@asynccontextmanager
async def open_form(request: Request, *, max_files: int = 200) -> AsyncIterator[FormData]:
    form = await request.form(max_files=max_files, max_fields=500)
    try:
        yield form
    finally:
        await form.close()


def form_text(form: FormData, name: str) -> str | None:
    value = form.get(name)
    if value is None:
        return None
    if isinstance(value, UploadFile):
        raise field_error("form", name, "must be sent as a text field, not a file")
    value = str(value).strip()
    return value or None


def form_texts(form: FormData, name: str) -> list[str]:
    out = []
    for value in form.getlist(name):
        if isinstance(value, UploadFile):
            raise field_error("form", name, "must be sent as text, not a file")
        if str(value).strip():
            out.append(str(value))
    return out


def form_file(form: FormData, name: str) -> UploadFile | None:
    value = form.get(name)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, UploadFile):
        raise field_error("form", name, "must be sent as a file upload")
    return value


def form_files(form: FormData, name: str) -> list[UploadFile]:
    files = []
    for value in form.getlist(name):
        if isinstance(value, str) and not value.strip():
            continue
        if not isinstance(value, UploadFile):
            raise field_error("form", name, "must be sent as file uploads")
        files.append(value)
    return files


def form_int(form: FormData, name: str) -> int | None:
    raw = form_text(form, name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        raise field_error("form", name, "must be an integer") from None


def form_float(form: FormData, name: str) -> float | None:
    raw = form_text(form, name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        raise field_error("form", name, "must be a number") from None


def form_bool(form: FormData, name: str, default: bool) -> bool:
    raw = form_text(form, name)
    if raw is None:
        return default
    if raw.lower() in {"1", "true", "yes", "on"}:
        return True
    if raw.lower() in {"0", "false", "no", "off"}:
        return False
    raise field_error("form", name, "must be true or false")


def form_json(form: FormData, name: str, adapter: TypeAdapter[Any]) -> Any:
    raw = form_text(form, name)
    if raw is None:
        return None
    try:
        return adapter.validate_json(raw)
    except ValidationError as exc:
        raise_validation(exc, f"form.{name}")
    except ValueError:
        raise field_error("form", name, "must be valid JSON") from None


def multipart_body(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        **({"required": required} if required else {}),
    }


def dual_request_body(json_model: str, multipart_schema: dict[str, Any]) -> dict[str, Any]:
    """``openapi_extra`` describing both accepted request formats for Swagger/OpenAPI."""
    return {
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {"schema": {"$ref": f"#/components/schemas/{json_model}"}},
                "multipart/form-data": {"schema": multipart_schema},
            },
        }
    }


def require(condition: bool, message: str, field: str) -> None:
    if not condition:
        raise ValidationAppError(message, details={"field": field})


async def load_named(loader, upload: UploadFile, *, allow_text: bool = True):
    """Load an upload; on failure attach the (sanitised) file name to the error details."""
    from app.errors import AppError
    from app.utils.text import sanitize_filename

    try:
        return await loader.load_upload(upload, allow_text=allow_text)
    except AppError as exc:
        exc.details = {**(exc.details or {}), "file": sanitize_filename(upload.filename, "upload")}
        raise
