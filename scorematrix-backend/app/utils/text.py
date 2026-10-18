"""Text helpers: normalisation, excerpt verification, sentence splitting, chunking.

Everything here is pure and deterministic so it can be unit-tested directly.
"""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from dataclasses import dataclass

# \w misses combining marks used by Indic/Thai scripts, so extend the word class.
_WORD_RE = re.compile(r"[\w\u0900-\u0DFF\u0E00-\u0E7F]+", re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?\u0964\u0965])\s+|\n{2,}|\n(?=\s*(?:[-\u2022*]\s|\d{1,3}[.)]\s))"
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ELLIPSIS_RE = re.compile(r"\.{3,}|\u2026")

_QUOTE_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2033": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
}  # fmt: skip
_SKIP_CHARS = {"\u00ad", "\u200b", "\u200c", "\u200d", "\ufeff", "\u2060"}

MIN_EXCERPT_CHARS = 8


# ----------------------------------------------------------------- basic helpers
def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def strip_control_chars(text: str) -> str:
    return _CONTROL_RE.sub("", text)


def collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    return [w for w in (m.group(0).casefold().strip("_") for m in _WORD_RE.finditer(text)) if w]


def word_count(text: str) -> int:
    return len(tokenize(text))


def alnum_count(text: str) -> int:
    return sum(1 for ch in text if ch.isalnum())


def garbage_ratio(text: str) -> float:
    """Share of non-space characters that are replacement/private-use/control glyphs."""
    visible = [ch for ch in text if not ch.isspace()]
    if not visible:
        return 0.0
    bad = sum(1 for ch in visible if ch == "\ufffd" or unicodedata.category(ch) in {"Co", "Cc", "Cs"})
    return bad / len(visible)


def is_readable(text: str, min_chars: int) -> bool:
    """Enough letters/digits and not mostly mojibake."""
    return alnum_count(text) >= min_chars and garbage_ratio(text) <= 0.3


def sanitize_filename(name: str | None, default: str = "upload") -> str:
    """Display-safe label derived from an untrusted filename (never used to open files)."""
    base = os.path.basename((name or "").replace("\\", "/")).strip()
    base = _CONTROL_RE.sub("", base)
    base = re.sub(r'[<>:"|?*\x00-\x1f]', "_", base)
    base = base.lstrip(".") or default
    return base[:120]


# ------------------------------------------------------------- normalised matching
def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Case-fold, unify quotes/dashes, collapse whitespace.

    Returns ``(normalised, index_map)`` where ``index_map[i]`` is the index in
    ``text`` of the character that produced ``normalised[i]``.
    """
    out: list[str] = []
    idx: list[int] = []
    prev_space = True  # also strips leading whitespace
    for i, ch in enumerate(text):
        if ch in _SKIP_CHARS:
            continue
        if ch.isspace():
            if not prev_space:
                out.append(" ")
                idx.append(i)
                prev_space = True
            continue
        ch = _QUOTE_MAP.get(ch, ch)
        for piece in unicodedata.normalize("NFKC", ch).casefold():
            out.append(piece)
            idx.append(i)
        prev_space = False
    if out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


def normalize(text: str) -> str:
    return normalize_with_map(text)[0]


@dataclass
class ExcerptMatch:
    pages: list[int]
    text: str | None  # original-case text from the source when the match is within one page


class PageIndex:
    """Locate quoted excerpts inside (page, text) pairs, keeping page provenance."""

    def __init__(self, pairs: list[tuple[int | None, str]]) -> None:
        self._joined = ""
        self._spans: list[tuple[int, int, int | None, str, list[int]]] = []
        for page, text in pairs:
            norm, idx_map = normalize_with_map(text)
            start = len(self._joined)
            self._joined += norm
            self._spans.append((start, len(self._joined), page, text, idx_map))
            self._joined += " "

    def find(self, excerpt: str) -> ExcerptMatch | None:
        if not excerpt or len(normalize(excerpt)) < MIN_EXCERPT_CHARS:
            return None
        fragments = [
            f for f in (normalize(part) for part in _ELLIPSIS_RE.split(excerpt)) if len(f) >= MIN_EXCERPT_CHARS
        ]
        if not fragments:
            return None
        pages: set[int] = set()
        first_hit: tuple[int, int] | None = None
        for frag in fragments:
            pos = self._joined.find(frag)
            if pos < 0:
                return None
            end = pos + len(frag)
            if first_hit is None:
                first_hit = (pos, end)
            for s_start, s_end, page, _text, _map in self._spans:
                if page is not None and pos < s_end and end > s_start:
                    pages.add(page)
        matched_text = None
        if len(fragments) == 1 and first_hit is not None:
            pos, end = first_hit
            for s_start, s_end, _page, text, idx_map in self._spans:
                if s_start <= pos and end <= s_end and idx_map:
                    a, b = pos - s_start, end - s_start
                    matched_text = text[idx_map[a] : idx_map[b - 1] + 1]
                    break
        return ExcerptMatch(pages=sorted(pages), text=matched_text)


# ---------------------------------------------------------------------- sentences
def split_sentences(text: str) -> list[str]:
    return [s for s in (collapse_whitespace(p) for p in _SENTENCE_SPLIT_RE.split(text)) if s]


@dataclass
class Unit:
    text: str
    page: int | None


def sentence_units(pairs: list[tuple[int | None, str]]) -> list[Unit]:
    units: list[Unit] = []
    for page, text in pairs:
        units.extend(Unit(s, page) for s in split_sentences(text))
    return units


# ----------------------------------------------------------------------- chunking
@dataclass
class Chunk:
    text: str
    pages: list[int]


def _split_long(unit: Unit, limit: int) -> list[Unit]:
    if len(unit.text) <= limit:
        return [unit]
    pieces: list[Unit] = []
    words = unit.text.split(" ")
    cur = ""
    for w in words:
        while len(w) > limit:  # a single "word" longer than the limit
            if cur:
                pieces.append(Unit(cur, unit.page))
                cur = ""
            pieces.append(Unit(w[:limit], unit.page))
            w = w[limit:]
        candidate = f"{cur} {w}".strip()
        if len(candidate) > limit:
            pieces.append(Unit(cur, unit.page))
            cur = w
        else:
            cur = candidate
    if cur:
        pieces.append(Unit(cur, unit.page))
    return pieces


def _make_chunk(units: list[Unit]) -> Chunk:
    return Chunk(
        text=" ".join(u.text for u in units),
        pages=sorted({u.page for u in units if u.page is not None}),
    )


def chunk_units(units: list[Unit], chunk_size: int, overlap: int) -> list[Chunk]:
    """Greedy sentence-aligned chunking with trailing-sentence overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap = max(0, min(overlap, chunk_size - 1))
    flat: list[Unit] = []
    for u in units:
        flat.extend(_split_long(u, chunk_size))

    chunks: list[Chunk] = []
    cur: list[Unit] = []
    cur_len = 0
    for u in flat:
        add = len(u.text) + (1 if cur else 0)
        if cur and cur_len + add > chunk_size:
            chunks.append(_make_chunk(cur))
            keep: list[Unit] = []
            kept_len = 0
            for prev in reversed(cur):
                extra = len(prev.text) + (1 if keep else 0)
                if kept_len + extra > overlap:
                    break
                keep.insert(0, prev)
                kept_len += extra
            if keep and kept_len + 1 + len(u.text) > chunk_size:
                keep, kept_len = [], 0
            cur, cur_len = keep, kept_len
            add = len(u.text) + (1 if cur else 0)
        cur.append(u)
        cur_len += add
    if cur:
        chunks.append(_make_chunk(cur))
    return chunks


# ------------------------------------------------------------- prompt-safety helpers
_DELIMITER_RE = re.compile(
    r"<\s*/?\s*(source_material|student_answer|question_paper|rubric_document|document|retrieved_sources?|source)\b",
    re.IGNORECASE,
)
_INJECTION_RE = re.compile(
    r"(ignore|disregard|forget)\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier|system)\s+(instructions?|prompts?|rules?)"
    r"|reveal\s+(your|the)\s+(system\s+prompt|api\s*key|secret|instructions)"
    r"|you\s+are\s+now\s+(a|an|in)\b"
    r"|new\s+instructions?\s*:",
    re.IGNORECASE,
)


def neutralize_delimiters(text: str) -> str:
    """Stop untrusted content from closing/opening our prompt delimiters."""
    return _DELIMITER_RE.sub(lambda m: "< " + m.group(0)[1:].lstrip(), strip_control_chars(text))


def looks_like_prompt_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(text))


def truncate_pairs(
    pairs: list[tuple[int | None, str]], max_chars: int
) -> tuple[list[tuple[int | None, str]], bool]:
    """Keep whole pages while they fit; cut the last page at a sentence boundary."""
    kept: list[tuple[int | None, str]] = []
    used = 0
    for page, text in pairs:
        if used + len(text) <= max_chars:
            kept.append((page, text))
            used += len(text) + 1
            continue
        room = max_chars - used
        if room > 200:
            cut = text[:room]
            m = list(re.finditer(r"[.!?\u0964]\s", cut))
            if m:
                cut = cut[: m[-1].end()]
            kept.append((page, cut.strip()))
        return kept, True
    return kept, False
