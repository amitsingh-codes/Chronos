import json

import pytest

from app.errors import EmbeddingsUnavailableError
from app.services.embeddings import EmbeddingService, EmbeddingStatus, HashingEmbedder
from tests import pdf_factory as f
from tests.conftest import AppHarness, make_settings, pdf_upload
from tests.helpers import ConceptEmbedder

PHOTO = [
    "Photosynthesis is the process by which green plants convert light energy into chemical energy stored in glucose.",
    "It takes place mainly in the chloroplasts, which contain the pigment chlorophyll.",
    "During the light dependent reactions, water molecules are split and oxygen is released as a by-product.",
    "The Calvin cycle then uses the energy captured to fix carbon dioxide into sugars.",
    "These sugars provide food for the plant and for almost every other organism on Earth.",
]
REVOLUTION = [
    "The French Revolution began in 1789 and transformed the political landscape of Europe.",
    "Citizens stormed the Bastille, abolished feudal privileges and declared the rights of man.",
    "Over the following decade the monarchy fell, a republic was established and radical factions fought for control.",
    "Napoleon Bonaparte eventually rose to power and reshaped the continent through war and legal reform.",
]
A = " ".join(PHOTO)
B = " ".join(["Plants turn sunlight into stored sugar using a green pigment, in a process called photosynthesis."] + PHOTO[1:])
C = " ".join(REVOLUTION)


def check(client, *texts, **extra):
    docs = [{"id": f"d{i}", "name": f"Doc {i}", "text": t} for i, t in enumerate(texts, 1)]
    return client.post("/api/plagiarism/check", json={"documents": docs, **extra})


def pair(res, a, b):
    return next(p for p in res["pairs"] if {p["document_a"], p["document_b"]} == {a, b})


def test_identical_texts_are_exact_duplicates(client):
    res = check(client, A, A.upper().replace(".", " .")).json()
    p = res["pairs"][0]
    assert p["assessment"] == "exact_duplicate" and p["exact_duplicate"] and p["lexical"]["shingle_jaccard"] == 1.0
    assert res["documents"][1]["duplicate_of"] == "d1"
    assert res["summary"]["pairs_flagged_for_review"] == 1


def test_near_copy_reports_lexical_overlap_with_matching_passages(client):
    res = check(client, A, B).json()
    p = res["pairs"][0]
    assert p["assessment"] == "high_lexical_overlap"
    assert p["lexical"]["containment_a_in_b"] > 0.5 and p["lexical"]["matched_sentence_ratio_a"] >= 0.6
    kinds = {m["kind"] for m in p["matches"]}
    assert "exact" in kinds
    assert any(PHOTO[1] == m["document_a_excerpt"] for m in p["matches"])
    assert "plagiari" not in p["explanation"].lower().replace("not proof of plagiarism", "")


def test_unrelated_texts_have_low_similarity(client):
    p = check(client, A, C).json()["pairs"][0]
    assert p["assessment"] == "low_similarity" and p["lexical"]["containment_a_in_b"] == 0 and not p["matches"]


def test_scope_and_limitations_never_claim_external_search(client):
    res = check(client, A, C).json()
    assert "only against each other" in res["scope"] and "No internet" in res["scope"]
    assert any("not proof" in x.lower() for x in res["limitations"])
    assert not any("database" in x.lower() and "searched" in x.lower() and "no " not in x.lower() for x in [res["scope"]])


def test_semantic_similarity_is_reported_separately_and_not_called_plagiarism():
    concepts = {}
    left = [f"Original sentence number {i} discusses the topic of urban planning and city transport systems today." for i in range(5)]
    right = [f"Entirely different wording {i} regarding metropolitan design together with public transit networks nowadays." for i in range(5)]
    for i, (l, r) in enumerate(zip(left, right)):
        concepts[l] = concepts[r] = i
    h = AppHarness(embedder=ConceptEmbedder(concepts))
    p = check(h.client, " ".join(left), " ".join(right)).json()["pairs"][0]
    assert p["lexical"]["containment_a_in_b"] < 0.1
    assert p["semantic"]["available"] and p["semantic"]["score"] > 0.95
    assert p["assessment"] == "semantic_similarity_only"
    assert {m["kind"] for m in p["matches"]} == {"semantic"}
    assert "not evidence by itself" in p["explanation"] and "same topic" in p["semantic"]["note"]


def test_topically_related_but_different_text_is_not_flagged_by_embeddings():
    h = AppHarness(embedder=ConceptEmbedder({}))
    p = check(h.client, A, " ".join(reversed(PHOTO[:2])) + " " + C).json()["pairs"][0]
    assert p["assessment"] in {"low_similarity", "moderate_lexical_overlap"}


def test_semantic_can_be_disabled_and_embedder_failure_degrades_gracefully(harness):
    r = check(harness.client, A, B, use_semantic=False).json()
    assert r["pairs"][0]["semantic"]["available"] is False

    class Broken(HashingEmbedder):
        def embed(self, texts):
            raise EmbeddingsUnavailableError("model could not be loaded")

    h = AppHarness(embedder=Broken())
    res = check(h.client, A, B)
    assert res.status_code == 200
    body = res.json()
    assert body["pairs"][0]["semantic"]["available"] is False and any("Semantic comparison unavailable" in w for w in body["warnings"])
    assert body["pairs"][0]["assessment"] == "high_lexical_overlap"  # lexical result still delivered


def test_lexical_hashing_backend_is_labelled_not_semantic(client):
    res = check(client, A, B).json()
    assert any("lexical" in w.lower() for w in res["warnings"])
    assert res["pairs"][0]["reliability"] == "low"


# ------------------------------------------------------------------ PDFs
def test_pdf_to_pdf_with_page_references(client):
    pdf_a = f.text_pdf(" ".join(PHOTO[:2]), " ".join(PHOTO[2:]))
    pdf_b = f.text_pdf(" ".join(REVOLUTION[:2]), PHOTO[3] + " " + " ".join(REVOLUTION[2:]) + " " + PHOTO[2])
    r = client.post(
        "/api/plagiarism/check",
        files=[("files", pdf_upload(pdf_a, "a.pdf")), ("files", pdf_upload(pdf_b, "b.pdf"))],
    )
    assert r.status_code == 200, r.text
    res = r.json()
    p = res["pairs"][0]
    hit = next(m for m in p["matches"] if m["document_a_excerpt"] == PHOTO[3])
    assert hit["document_a_pages"] == [2] and hit["document_b_pages"] == [2] and hit["kind"] == "exact"
    hit2 = next(m for m in p["matches"] if m["document_a_excerpt"] == PHOTO[2])
    assert hit2["document_a_pages"] == [2]
    assert [d["document"]["source_type"] for d in res["documents"]] == ["pdf", "pdf"]
    assert res["documents"][0]["name"] == "a.pdf" and res["documents"][0]["document"]["page_count"] == 2


def test_text_to_pdf_and_pdf_to_text_comparison(client):
    r = client.post(
        "/api/plagiarism/check",
        files=[("files", pdf_upload(f.text_pdf(A), "essay.pdf"))],
        data={"texts": [B]},
    )
    p = r.json()["pairs"][0]
    assert r.status_code == 200 and p["assessment"] == "high_lexical_overlap"
    assert {p["document_a"], p["document_b"]} == {"doc-1", "doc-2"}


def test_scanned_pdf_compared_via_ocr_and_unavailable_ocr_is_an_error(client, ocr):
    ocr.texts = [A]
    r = client.post("/api/plagiarism/check", files=[("files", pdf_upload(f.scanned_pdf("s"), "s.pdf"))], data={"texts": [A]})
    body = r.json()
    assert r.status_code == 200 and body["pairs"][0]["assessment"] == "exact_duplicate"
    assert body["documents"][0]["document"]["ocr_used"]
    ocr.available = False
    r = client.post("/api/plagiarism/check", files=[("files", pdf_upload(f.scanned_pdf("s"), "s.pdf"))], data={"texts": [A]})
    assert r.status_code == 503 and r.json()["error"]["code"] == "ocr_unavailable"


def test_identical_uploaded_files_are_flagged(client):
    same = f.text_pdf(A)
    r = client.post("/api/plagiarism/check", files=[("files", pdf_upload(same, "x.pdf")), ("files", pdf_upload(same, "y.pdf"))])
    assert r.json()["pairs"][0]["exact_duplicate"] and r.json()["documents"][1]["duplicate_of"] == "doc-1"


def test_blank_pdf_is_reported_as_empty_document_not_a_crash(client):
    r = client.post("/api/plagiarism/check", files=[("files", pdf_upload(f.blank_pdf(), "blank.pdf"))], data={"texts": [A]})
    res = r.json()
    assert r.status_code == 200 and res["documents"][0]["status"] == "empty"
    assert res["pairs"][0]["assessment"] == "insufficient_text"


def test_corrupt_and_wrong_type_uploads_fail_with_file_context(client):
    r = client.post("/api/plagiarism/check", files=[("files", pdf_upload(f.corrupt_pdf(), "bad.pdf"))], data={"texts": [A]})
    assert r.status_code == 422 and r.json()["error"]["details"]["file"] == "bad.pdf"
    r = client.post("/api/plagiarism/check", files=[("files", ("x.bin", b"\x00\x01\x02binary", "application/octet-stream"))], data={"texts": [A]})
    assert r.status_code == 415


# ------------------------------------------------------- edge cases & inputs
def test_multiple_documents_pairs_sorted_by_overlap(client):
    res = check(client, C, A, B, " ".join(REVOLUTION[:2]) + " " + PHOTO[0]).json()
    assert res["summary"]["pairs_compared"] == 6 and len(res["documents"]) == 4
    scores = [max(p["lexical"]["containment_a_in_b"], p["lexical"]["containment_b_in_a"]) for p in res["pairs"]]
    assert scores == sorted(scores, reverse=True)
    top = res["pairs"][0]
    assert {top["document_a"], top["document_b"]} == {"d2", "d3"}


def test_empty_and_very_short_documents(client):
    res = check(client, A, "", "Too short to compare.").json()
    st = {d["id"]: d["status"] for d in res["documents"]}
    assert st == {"d1": "ok", "d2": "empty", "d3": "insufficient_text"}
    assert {p["assessment"] for p in res["pairs"]} == {"insufficient_text"}
    assert all("too short" in p["explanation"] or "empty" in p["explanation"] for p in res["pairs"])


def test_repeated_boilerplate_is_ignored():
    header = "Name of the student and roll number must be written clearly on the first page of the answer sheet before starting."
    tails = [" ".join(PHOTO), " ".join(REVOLUTION) + " " + PHOTO[0], "The Calvin cycle is also called the dark reaction. It happens in the stroma of the chloroplast and builds sugars. " * 2]
    h = AppHarness()
    res = check(h.client, *[f"{header} {t}" for t in tails]).json()
    assert all(d["boilerplate_sentences_ignored"] >= 1 for d in res["documents"])
    for p in res["pairs"]:
        assert not any(m["document_a_excerpt"] == header for m in p["matches"])


def test_exclude_text_removes_assignment_prompt(client):
    prompt = "Describe the process of photosynthesis in detail and explain why it is important for living organisms."
    a, b = f"{prompt} {PHOTO[0]} {PHOTO[1]} {PHOTO[2]}", f"{prompt} {C}"
    with_prompt = check(client, a, b).json()["pairs"][0]["lexical"]["containment_a_in_b"]
    without = check(client, a, b, exclude_text=prompt).json()["pairs"][0]["lexical"]["containment_a_in_b"]
    assert with_prompt > 0 and without == 0


def test_text1_text2_shorthand(client):
    r = client.post("/api/plagiarism/check", json={"text1": A, "text2": B})
    assert r.status_code == 200 and r.json()["pairs"][0]["assessment"] == "high_lexical_overlap"


@pytest.mark.parametrize(
    "payload",
    [{"documents": [{"text": "only one"}]}, {"text1": "x"}, {}, {"documents": [{"id": "a", "text": A}, {"id": "a", "text": B}]}],
)
def test_invalid_requests(client, payload):
    assert client.post("/api/plagiarism/check", json=payload).status_code == 422


def test_document_count_limit():
    h = AppHarness(make_settings(plagiarism_max_documents=3))
    assert check(h.client, A, B, C, A + " x").status_code == 422
    r = h.client.post("/api/plagiarism/check", data={"texts": [A, B, C, "d"]})
    assert r.status_code == 422 and r.json()["error"]["details"]["max_documents"] == 3


def test_several_identical_documents_are_all_flagged_not_erased_as_boilerplate(client):
    """Regression: with 3+ documents, text shared by ALL of them was discarded as 'template'."""
    res = check(client, A, A, A).json()
    assert {p["assessment"] for p in res["pairs"]} == {"exact_duplicate"} and len(res["pairs"]) == 3
    assert [d["status"] for d in res["documents"]] == ["ok"] * 3
    assert [d["duplicate_of"] for d in res["documents"]] == [None, "d1", "d1"]
    assert all(d["boilerplate_sentences_ignored"] == 0 for d in res["documents"])


def test_mostly_shared_documents_stay_scored_and_explain_why(client):
    variant = lambda extra: A + " " + extra  # noqa: E731
    res = check(client, variant("Student one adds a unique closing remark about plant growth and light."),
                variant("Student two writes a different conclusion regarding oxygen production in leaves."),
                variant("Student three ends with an unrelated note about the water cycle and rainfall.")).json()
    assert {p["assessment"] for p in res["pairs"]} == {"high_lexical_overlap"}
    assert any("nothing was removed as boilerplate" in w for w in res["warnings"])
