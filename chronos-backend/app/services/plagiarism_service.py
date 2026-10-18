"""Similarity checking between documents submitted in one request.

Two distinct signals are reported and never conflated:

* **lexical overlap** - shared word n-grams and exact / near-duplicate
  sentences. This is what can indicate copying.
* **semantic similarity** - embedding cosine similarity. It is high for any
  two texts on the same topic, so on its own it is only a hint.

Only the documents in the request are compared. No internet, academic
database or external index is searched, and the output says so.
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from itertools import combinations

from app.config import Settings
from app.errors import EmbeddingsUnavailableError, ValidationAppError
from app.schemas.documents import ExtractionResult
from app.schemas.plagiarism import (
    Assessment,
    DocumentSummary,
    LexicalSimilarity,
    PairResult,
    PlagiarismCheckResponse,
    PlagiarismMatch,
    PlagiarismSummary,
    SemanticSimilarity,
)
from app.services.embeddings import EmbeddingService
from app.utils.text import sentence_units, sha256_hex, tokenize

logger = logging.getLogger("scorematrix.plagiarism")

MIN_SENTENCE_TOKENS = 6
_MAX_BOILERPLATE_SHARE = 0.4
_PAGE_NUMBER_RE = re.compile(r"^(page\s*)?\d+(\s*(of|/)\s*\d+)?$", re.IGNORECASE)

SCOPE = (
    "Documents were compared only against each other (the documents submitted in this request). "
    "No internet, academic database or external plagiarism index was searched."
)
LIMITATIONS = [
    "Similarity is not proof of plagiarism: shared quotations, common phrases, definitions, standard "
    "answers and a shared source text all create overlap.",
    "Lexical overlap (shared wording) and semantic similarity (shared meaning/topic) are reported "
    "separately. Embedding similarity alone is never treated as evidence of copying.",
    "Sentences are matched within pages, so a passage split across a page break may be missed. "
    "OCR-read documents can differ in wording because of recognition errors.",
    "A teacher or integrity officer must review the highlighted passages before drawing any conclusion.",
]


@dataclass
class InputDocument:
    id: str
    name: str
    extraction: ExtractionResult
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Sentence:
    text: str
    page: int | None
    tokens: list[str]

    @property
    def key(self) -> str:
        return " ".join(self.tokens)


@dataclass
class _Doc:
    src: InputDocument
    sentences: list[_Sentence]
    eligible: list[_Sentence]
    tokens: list[str]
    shingles: set[int]
    digest: str
    status: str
    ignored: int = 0
    duplicate_of: str | None = None
    trigram_index: dict[tuple[str, str, str], list[int]] | None = None


def _pages(sentence: _Sentence) -> list[int]:
    return [sentence.page] if sentence.page is not None else []


class PlagiarismService:
    def __init__(self, settings: Settings, embedder: EmbeddingService) -> None:
        self.settings = settings
        self.embedder = embedder

    # ------------------------------------------------------------------- public
    async def check(
        self,
        documents: list[InputDocument],
        *,
        exclude_text: str | None = None,
        use_semantic: bool = True,
    ) -> PlagiarismCheckResponse:
        s = self.settings
        if len(documents) < 2:
            raise ValidationAppError("At least two documents are required for comparison.")
        if len(documents) > s.plagiarism_max_documents:
            raise ValidationAppError(
                f"At most {s.plagiarism_max_documents} documents can be compared per request.",
                details={"max_documents": s.plagiarism_max_documents},
            )
        for d in documents:
            if len(d.extraction.text) > s.plagiarism_max_chars_per_document:
                raise ValidationAppError(
                    f"Document '{d.name}' exceeds {s.plagiarism_max_chars_per_document} characters.",
                    details={"document": d.name},
                )
        warnings: list[str] = []
        docs, notes = self._prepare(documents, exclude_text)
        warnings.extend(notes)

        semantic_vectors, semantic_note = await self._embed(docs, use_semantic, warnings)

        pairs: list[PairResult] = []
        for i, j in combinations(range(len(docs)), 2):
            pairs.append(self._compare(docs[i], docs[j], i, j, semantic_vectors, semantic_note))
        pairs.sort(
            key=lambda p: (
                -(max(p.lexical.containment_a_in_b, p.lexical.containment_b_in_a) if p.lexical else 0.0),
                p.document_a,
                p.document_b,
            )
        )
        flagged = sum(
            1
            for p in pairs
            if p.assessment
            in ("exact_duplicate", "high_lexical_overlap", "moderate_lexical_overlap", "semantic_similarity_only")
        )
        highest = max(
            (max(p.lexical.containment_a_in_b, p.lexical.containment_b_in_a) for p in pairs if p.lexical),
            default=0.0,
        )
        return PlagiarismCheckResponse(
            documents=[self._summary(d) for d in docs],
            pairs=pairs,
            summary=PlagiarismSummary(
                pairs_compared=len(pairs), pairs_flagged_for_review=flagged, highest_lexical_containment=highest
            ),
            scope=SCOPE,
            limitations=LIMITATIONS,
            warnings=warnings,
        )

    # -------------------------------------------------------------- preparation
    def _prepare(self, documents: list[InputDocument], exclude_text: str | None) -> tuple[list[_Doc], list[str]]:
        s = self.settings
        raw: list[list[_Sentence]] = []
        for d in documents:
            sents = []
            for unit in sentence_units(d.extraction.page_pairs()):
                tokens = tokenize(unit.text)
                if not tokens or _PAGE_NUMBER_RE.match(unit.text.strip()):
                    continue
                sents.append(_Sentence(unit.text, unit.page, tokens))
            raw.append(sents)

        excluded: set[str] = set()
        if exclude_text:
            excluded |= {" ".join(tokenize(u.text)) for u in sentence_units([(None, exclude_text)])}
        n = len(documents)
        notes: list[str] = []
        if n >= 3:  # sentences present in (nearly) every document look like a shared template
            threshold = n if n < 5 else math.ceil(0.8 * n)
            counts: dict[str, int] = defaultdict(int)
            for sents in raw:
                for key in {x.key for x in sents if len(x.tokens) >= 3}:
                    counts[key] += 1
            candidates = {k for k, c in counts.items() if c >= threshold}
            # Real boilerplate (headers, instructions) is a small part of a document. If "shared by all"
            # sentences make up most of any document, that sharing is itself the finding (e.g. several
            # students copying one source), so it must NOT be silently discarded.
            share = max(
                (sum(1 for x in sents if x.key in candidates) / len(sents) for sents in raw if sents),
                default=0.0,
            )
            if candidates and share <= _MAX_BOILERPLATE_SHARE:
                excluded |= candidates
            elif candidates:
                notes.append(
                    f"{share:.0%} of at least one document's sentences appear in all documents. That is too much to "
                    "be template text, so nothing was removed as boilerplate; the shared passages are scored as matches."
                )
        excluded.discard("")

        docs: list[_Doc] = []
        for d, sents in zip(documents, raw):
            kept = [x for x in sents if x.key not in excluded]
            ignored = len(sents) - len(kept)
            tokens = [t for x in kept for t in x.tokens]
            k = s.plagiarism_shingle_size
            shingles = {hash(tuple(tokens[i : i + k])) for i in range(max(0, len(tokens) - k + 1))}
            if not sents:
                status = "empty"
            elif len(tokens) < s.plagiarism_min_words:
                status = "insufficient_text"
            else:
                status = "ok"
            docs.append(
                _Doc(
                    src=d,
                    sentences=kept,
                    eligible=[x for x in kept if len(x.tokens) >= MIN_SENTENCE_TOKENS],
                    tokens=tokens,
                    shingles=shingles,
                    digest=sha256_hex(" ".join(tokens)) if tokens else "",
                    status=status,
                    ignored=ignored,
                )
            )
        first_seen: dict[str, str] = {}
        for d in docs:
            if d.digest and d.status == "ok":
                if d.digest in first_seen:
                    d.duplicate_of = first_seen[d.digest]
                else:
                    first_seen[d.digest] = d.src.id
        return docs, notes

    @staticmethod
    def _summary(d: _Doc) -> DocumentSummary:
        warns = list(d.src.warnings) + list(d.src.extraction.warnings)
        if d.status == "empty":
            warns.append("The document contains no usable text.")
        elif d.status == "insufficient_text":
            warns.append("The document is too short for a reliable comparison.")
        return DocumentSummary(
            id=d.src.id,
            name=d.src.name,
            word_count=len(d.tokens),
            status=d.status,  # type: ignore[arg-type]
            duplicate_of=d.duplicate_of,
            boilerplate_sentences_ignored=d.ignored,
            document=d.src.extraction.info(),
            warnings=warns,
        )

    # --------------------------------------------------------------- embeddings
    async def _embed(self, docs: list[_Doc], use_semantic: bool, warnings: list[str]):
        if not use_semantic:
            return None, "Semantic comparison was disabled for this request."
        ok_docs = [d for d in docs if d.status == "ok" and d.eligible]
        total = sum(len(d.eligible) for d in ok_docs)
        if total == 0:
            return None, "No sentences were long enough for semantic comparison."
        if total > self.settings.embedding_max_texts:
            msg = (
                f"Semantic comparison skipped: {total} sentences exceed EMBEDDING_MAX_TEXTS="
                f"{self.settings.embedding_max_texts}."
            )
            warnings.append(msg)
            return None, msg
        try:
            flat = [x.text for d in ok_docs for x in d.eligible]
            matrix = await self.embedder.embed_async(flat)
        except EmbeddingsUnavailableError as exc:
            warnings.append(f"Semantic comparison unavailable: {exc.message}")
            return None, "Semantic comparison was unavailable on the server."
        vectors: dict[str, object] = {}
        offset = 0
        for d in ok_docs:
            vectors[d.src.id] = matrix[offset : offset + len(d.eligible)]
            offset += len(d.eligible)
        if not self.embedder.semantic:
            warnings.append(
                "The configured embedding backend is lexical ('hashing'), so 'semantic' scores only reflect shared words."
            )
        return vectors, ""

    # --------------------------------------------------------------- comparison
    def _compare(self, a: _Doc, b: _Doc, i: int, j: int, vectors, semantic_note: str) -> PairResult:
        s = self.settings
        ida, idb = a.src.id, b.src.id
        if a.status != "ok" or b.status != "ok":
            bad = [d.src.name for d in (a, b) if d.status != "ok"]
            return PairResult(
                document_a=ida,
                document_b=idb,
                assessment="insufficient_text",
                exact_duplicate=False,
                reliability="low",
                explanation=(
                    "Not compared reliably because these document(s) are empty or too short "
                    f"(fewer than {s.plagiarism_min_words} words after removing template/boilerplate text): "
                    + ", ".join(bad)
                    + "."
                ),
            )

        lexical = self._lexical(a, b)
        if a.digest == b.digest:
            return PairResult(
                document_a=ida,
                document_b=idb,
                assessment="exact_duplicate",
                exact_duplicate=True,
                reliability="normal",
                lexical=lexical,
                semantic=SemanticSimilarity(
                    available=False, note="Not computed: the texts are identical after normalisation."
                ),
                explanation=(
                    "The two documents contain identical text once case, punctuation and whitespace are "
                    "normalised. Identical files may be accidental re-uploads; a teacher should decide."
                ),
            )

        matches, ratio_a, ratio_b = self._sentence_matches(a, b)
        lexical.matched_sentence_ratio_a = ratio_a
        lexical.matched_sentence_ratio_b = ratio_b

        semantic = SemanticSimilarity(available=False, note=semantic_note or "Semantic comparison not available.")
        if vectors is not None and a.src.id in vectors and b.src.id in vectors:
            semantic, sem_matches = self._semantic(a, b, vectors, matches)
            matches.extend(sem_matches)

        matches.sort(key=lambda m: ({"exact": 0, "near_duplicate": 1, "semantic": 2}[m.kind], -m.score))
        matches = matches[: s.plagiarism_max_matches]

        containment = max(lexical.containment_a_in_b, lexical.containment_b_in_a)
        sentence_ratio = max(ratio_a, ratio_b)
        semantic_hits = sum(1 for m in matches if m.kind == "semantic")
        assessment: Assessment
        if containment >= 0.5 or sentence_ratio >= 0.5:
            assessment = "high_lexical_overlap"
        elif containment >= 0.2 or sentence_ratio >= 0.25:
            assessment = "moderate_lexical_overlap"
        elif semantic.available and ((semantic.score or 0) >= s.plagiarism_semantic_threshold or semantic_hits >= 3):
            assessment = "semantic_similarity_only"
        else:
            assessment = "low_similarity"

        short = min(len(a.tokens), len(b.tokens)) < 100
        reliability = "low" if short or (vectors is not None and not self.embedder.semantic) else "normal"
        warns = []
        if short:
            warns.append("At least one document is short (under 100 words); scores are less reliable.")
        return PairResult(
            document_a=ida,
            document_b=idb,
            assessment=assessment,
            exact_duplicate=False,
            reliability=reliability,
            lexical=lexical,
            semantic=semantic,
            matches=matches,
            explanation=self._explain(assessment, lexical, semantic),
            warnings=warns,
        )

    def _lexical(self, a: _Doc, b: _Doc) -> LexicalSimilarity:
        inter = len(a.shingles & b.shingles)
        union = len(a.shingles | b.shingles)
        return LexicalSimilarity(
            shingle_jaccard=round(inter / union, 4) if union else 0.0,
            containment_a_in_b=round(inter / len(a.shingles), 4) if a.shingles else 0.0,
            containment_b_in_a=round(inter / len(b.shingles), 4) if b.shingles else 0.0,
            matched_sentence_ratio_a=0.0,
            matched_sentence_ratio_b=0.0,
            shingle_size=self.settings.plagiarism_shingle_size,
        )

    @staticmethod
    def _index(doc: _Doc) -> dict[tuple[str, str, str], list[int]]:
        if doc.trigram_index is None:
            idx: dict[tuple[str, str, str], list[int]] = defaultdict(list)
            for n, sent in enumerate(doc.eligible):
                t = sent.tokens
                for k in {tuple(t[x : x + 3]) for x in range(len(t) - 2)}:
                    idx[k].append(n)
            doc.trigram_index = dict(idx)
        return doc.trigram_index

    def _sentence_matches(self, a: _Doc, b: _Doc) -> tuple[list[PlagiarismMatch], float, float]:
        threshold = self.settings.plagiarism_near_duplicate_threshold
        matches: list[PlagiarismMatch] = []
        matched_a: set[int] = set()
        matched_b: set[int] = set()
        index_b = self._index(b)
        by_key_b = {s_.key: n for n, s_ in reversed(list(enumerate(b.eligible)))}
        for na, sa in enumerate(a.eligible):
            exact = by_key_b.get(sa.key)
            if exact is not None:
                sb = b.eligible[exact]
                matched_a.add(na)
                matched_b.add(exact)
                matches.append(
                    PlagiarismMatch(
                        kind="exact",
                        score=1.0,
                        document_a_excerpt=sa.text,
                        document_b_excerpt=sb.text,
                        document_a_pages=_pages(sa),
                        document_b_pages=_pages(sb),
                    )
                )
                continue
            votes: dict[int, int] = defaultdict(int)
            for k in {tuple(sa.tokens[x : x + 3]) for x in range(len(sa.tokens) - 2)}:
                for nb in index_b.get(k, ()):
                    votes[nb] += 1
            best_n, best_ratio = None, 0.0
            for nb, _ in sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[:5]:
                ratio = SequenceMatcher(None, sa.tokens, b.eligible[nb].tokens, autojunk=False).ratio()
                if ratio > best_ratio:
                    best_n, best_ratio = nb, ratio
            if best_n is not None and best_ratio >= threshold:
                sb = b.eligible[best_n]
                matched_a.add(na)
                matched_b.add(best_n)
                matches.append(
                    PlagiarismMatch(
                        kind="near_duplicate",
                        score=round(best_ratio, 4),
                        document_a_excerpt=sa.text,
                        document_b_excerpt=sb.text,
                        document_a_pages=_pages(sa),
                        document_b_pages=_pages(sb),
                    )
                )
        ratio_a = round(len(matched_a) / len(a.eligible), 4) if a.eligible else 0.0
        ratio_b = round(len(matched_b) / len(b.eligible), 4) if b.eligible else 0.0
        return matches, ratio_a, ratio_b

    def _semantic(self, a: _Doc, b: _Doc, vectors, lexical_matches: list[PlagiarismMatch]):
        va, vb = vectors[a.src.id], vectors[b.src.id]
        sim = va @ vb.T  # normalised vectors -> cosine
        best_a = sim.max(axis=1)
        best_b = sim.max(axis=0)
        score = float((best_a.mean() + best_b.mean()) / 2)
        already = {m.document_a_excerpt for m in lexical_matches}
        threshold = self.settings.plagiarism_semantic_threshold
        found: list[PlagiarismMatch] = []
        for na in range(sim.shape[0]):
            nb = int(sim[na].argmax())
            value = float(sim[na, nb])
            sa = a.eligible[na]
            if value >= threshold and sa.text not in already:
                sb = b.eligible[nb]
                found.append(
                    PlagiarismMatch(
                        kind="semantic",
                        score=round(value, 4),
                        document_a_excerpt=sa.text,
                        document_b_excerpt=sb.text,
                        document_a_pages=_pages(sa),
                        document_b_pages=_pages(sb),
                    )
                )
        found.sort(key=lambda m: -m.score)
        return (
            SemanticSimilarity(
                available=True,
                score=round(score, 4),
                model=self.embedder.model_name,
                note=(
                    "Mean best-match cosine similarity of sentence embeddings. It measures closeness of meaning "
                    "and is high for any two texts on the same topic; it does not show copying."
                    if self.embedder.semantic
                    else "Lexical hashing score (not truly semantic): reflects shared words only."
                ),
            ),
            found,
        )

    @staticmethod
    def _explain(assessment: Assessment, lex: LexicalSimilarity, sem: SemanticSimilarity) -> str:
        shared = f"{max(lex.containment_a_in_b, lex.containment_b_in_a):.0%} of the shorter text's word sequences"
        sentences = f"{max(lex.matched_sentence_ratio_a, lex.matched_sentence_ratio_b):.0%} of sentences"
        base = {
            "high_lexical_overlap": (
                f"Substantial shared wording: {shared} and {sentences} appear in both documents. "
                "Review the matching passages; this may reflect copying, a shared source or quoted material."
            ),
            "moderate_lexical_overlap": (
                f"Some shared wording ({shared}; {sentences}). This can be coincidental or from common "
                "phrases or quotations; review the matching passages."
            ),
            "semantic_similarity_only": (
                "Little shared wording, but the documents are close in meaning. This pattern is common for "
                "texts on the same topic and may indicate paraphrasing, but it is not evidence by itself."
            ),
            "low_similarity": "Little shared wording and no strong semantic match was found between these documents.",
        }[assessment]
        return base
