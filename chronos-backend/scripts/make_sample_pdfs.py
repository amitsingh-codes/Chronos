"""Generate sample PDFs + JSON payloads in ./samples for trying the PDF workflows by hand.

    python scripts/make_sample_pdfs.py

Creates a selectable-text PDF, a scanned (image-only) PDF, a mixed PDF, student answers
(one selectable, one scanned, one near-copy for plagiarism checks) and a rubric.
Needs the dev dependencies (reportlab, Pillow).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests import pdf_factory as f  # noqa: E402

OUT = ROOT / "samples"

PAGE1 = (
    "The mitochondria is the powerhouse of the cell because it produces most of the cell's supply of ATP. "
    "The nucleus contains the genetic material of the cell and controls its activities. "
    "Ribosomes are small structures that synthesise proteins by translating messenger RNA."
)
PAGE2 = (
    "Photosynthesis takes place in the chloroplasts of plant cells, where light energy is converted into "
    "chemical energy. Osmosis is the movement of water across a semipermeable membrane from low solute "
    "concentration to high solute concentration. The cell membrane regulates which substances enter and leave the cell."
)
ANSWER = (
    "Photosynthesis happens in the chloroplasts of plant cells. Light energy is converted into chemical energy "
    "stored as glucose. Oxygen is released as a by-product of the process."
)
ANSWER_COPY = (
    "Plants turn sunlight into sugar using chlorophyll. Light energy is converted into chemical energy "
    "stored as glucose. Oxygen is released as a by-product of the process."
)
ESSAY = (
    "Photosynthesis is the process by which green plants convert light energy into chemical energy stored in glucose. "
    "It takes place mainly in the chloroplasts, which contain the pigment chlorophyll. "
    "During the light dependent reactions, water molecules are split and oxygen is released as a by-product. "
    "The Calvin cycle then uses the energy captured to fix carbon dioxide into sugars. "
    "These sugars provide food for the plant and for almost every other organism on Earth."
)
ESSAY_COPY = (
    "Plants turn sunlight into stored sugar using a green pigment, in a process called photosynthesis. "
    "It takes place mainly in the chloroplasts, which contain the pigment chlorophyll. "
    "During the light dependent reactions, water molecules are split and oxygen is released as a by-product. "
    "The Calvin cycle then uses the energy captured to fix carbon dioxide into sugars. "
    "Those sugars feed the plant and nearly every other living thing on our planet."
)
RUBRIC = [
    {"name": "Location", "description": "Names the organelle where photosynthesis occurs", "max_marks": 2},
    {"name": "Energy conversion", "description": "Explains light energy becoming chemical energy", "max_marks": 3},
]
QUESTION = "Explain where photosynthesis occurs and what energy conversion takes place."


def main() -> None:
    OUT.mkdir(exist_ok=True)
    files = {
        "text_biology.pdf": f.text_pdf(PAGE1, PAGE2),
        "scanned_biology.pdf": f.scanned_pdf(PAGE1, PAGE2),
        "mixed_biology.pdf": f.mixed_pdf(("text", PAGE1), ("image", PAGE2)),
        "student_answer.pdf": f.text_pdf(ANSWER),
        "student_answer_scanned.pdf": f.scanned_pdf(ANSWER),
        "student_answer_copy.pdf": f.text_pdf(ANSWER_COPY),
        "essay_original.pdf": f.text_pdf(ESSAY),
        "essay_copy.pdf": f.text_pdf(ESSAY_COPY),
        "essay_original_scanned.pdf": f.scanned_pdf(ESSAY),
        "question_paper.pdf": f.text_pdf(QUESTION),
        "rubric.pdf": f.text_pdf(
            "Marking scheme. Location of photosynthesis: 2 marks. Energy conversion: 3 marks. Total: 5 marks."
        ),
    }
    for name, data in files.items():
        (OUT / name).write_bytes(data)
    (OUT / "rubric.json").write_text(json.dumps(RUBRIC, indent=2), encoding="utf-8")
    (OUT / "generate.json").write_text(
        json.dumps(
            {
                "source_text": f"{PAGE1} {PAGE2}",
                "num_questions": 4,
                "difficulty": "medium",
                "question_types": ["mcq", "short_answer", "long_answer", "fill_in_the_blanks"],
                "total_marks": 20,
                "subject": "Biology",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    payloads = {
        "evaluate.json": {"question": QUESTION, "student_answer": ANSWER, "model_answer": "Photosynthesis occurs in chloroplasts; light energy is converted to chemical energy.", "rubric": RUBRIC},
        "batch.json": {"question": QUESTION, "rubric": RUBRIC, "submissions": [
            {"student_id": "asha", "answer": ANSWER}, {"student_id": "ravi", "answer": ANSWER_COPY}]},
        "plagiarism.json": {"documents": [
            {"id": "asha", "name": "Asha", "text": ESSAY}, {"id": "ravi", "name": "Ravi", "text": ESSAY_COPY}]},
        "rag.json": {"question": "Where does photosynthesis take place?", "documents": [
            {"id": "bio", "name": "biology notes", "text": f"{PAGE1} {PAGE2}"}], "top_k": 3},
    }
    for name, body in payloads.items():
        (OUT / name).write_text(json.dumps(body, indent=2), encoding="utf-8")
    print(f"Wrote {len(files) + 2 + len(payloads)} files to {OUT}")


if __name__ == "__main__":
    main()
