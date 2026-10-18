"""Builds real PDF files for tests and for the manual sample set.

Needs the dev dependencies (reportlab, Pillow, pypdf). Scanned pages are made
by rendering text to a bitmap and embedding *only the image*, so the PDF has
no selectable text - exactly like a scanner output.
"""

from __future__ import annotations

import io
import textwrap

from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def _draw_text_page(c: canvas.Canvas, text: str) -> None:
    width, height = A4
    y = height - 72
    c.setFont("Helvetica", 12)
    for paragraph in text.split("\n"):
        for line in textwrap.wrap(paragraph, 88, break_on_hyphens=False) or [""]:
            c.drawString(60, y, line)
            y -= 16
    c.showPage()


def _text_image(text: str, size: tuple[int, int] = (1240, 1754), font_px: int = 40) -> Image.Image:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=font_px)
    y = 80
    for paragraph in text.split("\n"):
        for line in textwrap.wrap(paragraph, 42, break_on_hyphens=False) or [""]:
            draw.text((70, y), line, fill="black", font=font)
            y += font_px + 18
    return img


def _draw_image_page(c: canvas.Canvas, text: str) -> None:
    width, height = A4
    img = _text_image(text)
    c.drawImage(ImageReader(img), 0, 0, width=width, height=height)
    c.showPage()


def build_pdf(pages: list[tuple[str, str]]) -> bytes:
    """pages: list of ("text" | "image" | "blank", content)."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for kind, content in pages:
        if kind == "text":
            _draw_text_page(c, content)
        elif kind == "image":
            _draw_image_page(c, content)
        elif kind == "blank":
            c.showPage()
        else:
            raise ValueError(kind)
    c.save()
    return buf.getvalue()


def text_pdf(*pages: str) -> bytes:
    return build_pdf([("text", p) for p in pages])


def scanned_pdf(*pages: str) -> bytes:
    return build_pdf([("image", p) for p in pages])


def mixed_pdf(*pages: tuple[str, str]) -> bytes:
    return build_pdf(list(pages))


def blank_pdf(pages: int = 1) -> bytes:
    return build_pdf([("blank", "")] * pages)


def zero_page_pdf() -> bytes:
    buf = io.BytesIO()
    PdfWriter().write(buf)
    return buf.getvalue()


def encrypted_pdf(text: str = "Secret content that nobody can read without the password.") -> bytes:
    writer = PdfWriter()
    for page in PdfReader(io.BytesIO(text_pdf(text))).pages:
        writer.add_page(page)
    writer.encrypt(user_password="correct-horse", owner_password="owner-secret")
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def corrupt_pdf() -> bytes:
    good = text_pdf("This file will be damaged.")
    return good[: len(good) // 3] + b"\x00\x01garbage\xff" * 20


def header_only_pdf() -> bytes:
    return b"%PDF-1.7\n% nothing else"
