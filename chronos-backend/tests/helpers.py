"""Fakes and builders shared by the test-suite. No network, no real Gemini key."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from app.services.ai_service import AIService, AIStatus
from app.services.embeddings import HashingEmbedder
from app.services.evaluation_service import WireCriterion, WireEvaluation
from app.services.ocr import OcrEngine, OcrResult, OcrStatus
from app.services.question_service import WireQuestion, WireQuestionSet

SENTENCES = [
    "The mitochondria is the powerhouse of the cell because it produces most of the cell's supply of ATP.",
    "Photosynthesis takes place in the chloroplasts of plant cells, where light energy is converted into chemical energy.",
    "The nucleus contains the genetic material of the cell and controls its activities.",
    "Osmosis is the movement of water across a semipermeable membrane from low solute concentration to high solute concentration.",
    "Ribosomes are small structures that synthesise proteins by translating messenger RNA.",
    "The cell membrane regulates which substances enter and leave the cell.",
]
SOURCE = " ".join(SENTENCES)


# --------------------------------------------------------------------- fake AI
class FakeAI(AIService):
    """Returns canned wire objects by response model; records calls and concurrency."""

    def __init__(self) -> None:
        self.handlers: dict[type, Any] = {}
        self.calls: list[dict[str, Any]] = []
        self.delay = 0.0
        self.active = 0
        self.max_active = 0

    def on(self, model: type[BaseModel], handler: Any) -> "FakeAI":
        self.handlers[model] = handler
        return self

    async def generate_structured(self, *, system_instruction, prompt, response_model, temperature=None, max_output_tokens=None):
        self.calls.append({"system": system_instruction, "prompt": prompt, "model": response_model})
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            handler = self.handlers.get(response_model)
            if handler is None:
                raise AssertionError(f"No fake handler registered for {response_model.__name__}")
            if isinstance(handler, BaseException) or (isinstance(handler, type) and issubclass(handler, BaseException)):
                raise handler
            return handler(prompt) if callable(handler) and not isinstance(handler, BaseModel) else handler
        finally:
            self.active -= 1

    def status(self) -> AIStatus:
        return AIStatus("fake", True, "fake-model")


# ----------------------------------------------------------------- fake OCR
class FakeOcr(OcrEngine):
    def __init__(self, available: bool = True, texts: list[str] | None = None, confidence: float | None = 93.0):
        self.available = available
        self.texts = texts or ["Recognised scanned text about the structure of plant cells and their organelles."]
        self.confidence = confidence
        self.calls = 0

    def status(self) -> OcrStatus:
        return OcrStatus(
            available=self.available,
            engine="fake",
            detail="fake OCR ready" if self.available else "Tesseract OCR executable was not found. Install it.",
            languages_requested=["eng"],
        )

    def recognize(self, image: Any, *, dpi: int, timeout: float) -> OcrResult:
        text = self.texts[self.calls % len(self.texts)]
        self.calls += 1
        return OcrResult(text=text, confidence=self.confidence)


# --------------------------------------------------------- semantic test double
class ConceptEmbedder(HashingEmbedder):
    """Deterministic stand-in for a semantic model.

    Sentences listed in ``concepts`` map to a shared one-hot "meaning" vector, so differently worded
    paraphrases get identical embeddings. All other text falls back to lexical hashing in a disjoint
    subspace (so unrelated sentences stay orthogonal to the concept vectors).
    """

    backend = "test-concepts"
    model_name = "test-concept-embedder"
    semantic = True
    CONCEPT_DIMS = 64

    def __init__(self, concepts: dict[str, int] | None = None) -> None:
        super().__init__()
        self.concepts = concepts or {}

    def embed(self, texts: list[str]):
        import numpy as np

        base = super().embed(texts)
        out = np.zeros((len(texts), self.CONCEPT_DIMS + base.shape[1]), dtype=np.float32)
        for i, t in enumerate(texts):
            if t in self.concepts:
                out[i, self.concepts[t]] = 1.0
            else:
                out[i, self.CONCEPT_DIMS:] = base[i]
        return out


# ------------------------------------------------------------------- builders
def mcq(excerpt: str = SENTENCES[0], **kw) -> WireQuestion:
    base = dict(
        type="mcq",
        question="Which organelle produces most of the cell's ATP?",
        options=["Nucleus", "Mitochondria", "Ribosome", "Chloroplast"],
        correct_option="B",
        correct_answer="Mitochondria",
        explanation="The source says the mitochondria produces most of the cell's ATP.",
        source_excerpt=excerpt,
    )
    base.update(kw)
    return WireQuestion(**base)


def short(excerpt: str = SENTENCES[1], **kw) -> WireQuestion:
    base = dict(
        type="short_answer",
        question="Where does photosynthesis take place in plant cells?",
        options=None,
        correct_option=None,
        correct_answer="In the chloroplasts.",
        explanation="Photosynthesis occurs in the chloroplasts.",
        source_excerpt=excerpt,
    )
    base.update(kw)
    return WireQuestion(**base)


def long_q(excerpt: str = SENTENCES[3], **kw) -> WireQuestion:
    base = dict(
        type="long_answer",
        question="Explain osmosis and the direction in which water moves.",
        options=None,
        correct_option=None,
        correct_answer="Water moves across a semipermeable membrane from low to high solute concentration.",
        explanation="This is the definition of osmosis in the source.",
        source_excerpt=excerpt,
    )
    base.update(kw)
    return WireQuestion(**base)


def fill(excerpt: str = SENTENCES[2], **kw) -> WireQuestion:
    base = dict(
        type="fill_in_the_blanks",
        question="The _____ contains the genetic material of the cell.",
        options=None,
        correct_option=None,
        correct_answer="nucleus",
        explanation="The nucleus holds the genetic material.",
        source_excerpt=excerpt,
    )
    base.update(kw)
    return WireQuestion(**base)


def question_set(*qs: WireQuestion, notes: list[str] | None = None) -> WireQuestionSet:
    return WireQuestionSet(questions=list(qs), notes=notes or [])


ANSWER = (
    "Photosynthesis happens in the chloroplasts of plant cells. Light energy is converted into chemical energy "
    "stored as glucose. Oxygen is released as a by-product of the process."
)
RUBRIC = [
    {"name": "Location", "description": "Names the organelle", "max_marks": 2},
    {"name": "Energy conversion", "description": "Explains light to chemical energy", "max_marks": 3},
]


def evaluation(awarded: tuple[float, float] = (2, 3), quotes: tuple[list[str], list[str]] | None = None, **kw) -> WireEvaluation:
    q1, q2 = quotes or (
        ["Photosynthesis happens in the chloroplasts of plant cells"],
        ["Light energy is converted into chemical energy"],
    )
    base = dict(
        criteria=[
            WireCriterion(criterion_id="C1", awarded_marks=awarded[0], feedback="Correct organelle named.", evidence_quotes=q1, confidence="high"),
            WireCriterion(criterion_id="C2", awarded_marks=awarded[1], feedback="Energy conversion explained.", evidence_quotes=q2, confidence="high"),
        ],
        strengths=["Clear and accurate"],
        missing_concepts=["No mention of chlorophyll"],
        improvement_suggestions=["Name the pigment involved"],
        overall_feedback="A solid answer.",
        uncertainty_notes=[],
    )
    base.update(kw)
    return WireEvaluation(**base)


def make_handler(fn: Callable[[str], BaseModel]) -> Callable[[str], BaseModel]:
    return fn
