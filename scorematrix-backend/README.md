# ScoreMatrix Backend

FastAPI backend for **ScoreMatrix**, an AI-assisted assessment platform for teachers and evaluators:
question generation, rubric-based evaluation (single + batch), plagiarism checking and document Q&A (RAG) —
all with first-class **PDF upload** support (selectable-text, scanned/OCR, and mixed PDFs).

> **AI output is advisory.** Evaluation results are always provisional (`"final": false`) and flagged for
> teacher review whenever the evidence is weak. Similarity scores are never a verdict of plagiarism.

## Endpoints

| Method | Path | Input | Notes |
|---|---|---|---|
| GET | `/health` | – | App status + optional capabilities (Gemini configured, OCR, embeddings). No secrets/paths. |
| GET | `/docs`, `/openapi.json` | – | Swagger UI / OpenAPI (documents both JSON and multipart bodies). |
| POST | `/api/questions/generate` | JSON | Existing contract: `source_text` 50–30 000 chars. |
| POST | `/api/questions/generate-pdf` | multipart | PDF upload + the same options as form fields. |
| POST | `/api/evaluation/evaluate` | JSON **or** multipart | Text answer, or answer/question/model-answer/rubric as PDFs. |
| POST | `/api/evaluation/evaluate-batch` | JSON **or** multipart | Many answer files (PDF/.txt) or a JSON list. |
| POST | `/api/plagiarism/check` | JSON **or** multipart | 2+ documents (text and/or PDF), compared with each other. |
| POST | `/api/rag/ask` | JSON **or** multipart | Question + documents (text and/or PDF); page-aware citations. |

## Quick start (Windows PowerShell)

Requires **Python 3.12+**. Run everything from the project folder.

```powershell
# 1. Create a virtual environment
py -3.12 -m venv .venv

# 2. Activate it (if scripts are blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies (sentence-transformers pulls in PyTorch: a large download)
python -m pip install --upgrade pip
pip install -r requirements-dev.txt        # use requirements.txt for a runtime-only install

# 4. Configure
Copy-Item .env.example .env
notepad .env                               # set GEMINI_API_KEY (https://aistudio.google.com/apikey)
```

### 5. Install Tesseract OCR (needed only for scanned/image PDFs)

1. Download the Windows installer from <https://github.com/UB-Mannheim/tesseract/wiki> and run it.
2. In the installer, tick extra **languages** you need (e.g. *Hindi*) — then set `OCR_LANGUAGES=eng+hin` in `.env`.
3. Make it findable — either add `C:\Program Files\Tesseract-OCR` to your `PATH`, or set in `.env`:
   ```
   TESSERACT_CMD='C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```
   > Use **single quotes** (or forward slashes). With *double* quotes, `\t` in `...\tesseract.exe` is read as a TAB and the path breaks.
4. Open a **new** PowerShell window and check: `tesseract --version`, then start the server and open `/health` — `ocr.available` must be `true`.

If OCR is missing the server still works: text PDFs are fine, and scanned PDFs return a clear `ocr_unavailable` error explaining how to enable it. Nothing ever pretends OCR succeeded.

```powershell
# 6. Start the server
uvicorn app.main:app --reload --port 8000

# 7. Open Swagger UI
Start-Process http://localhost:8000/docs

# 8. Run the tests (no Gemini key, internet or Tesseract required for most; real-OCR tests skip if absent)
python -m pytest

# 9. Generate sample PDFs/payloads into .\samples to try every workflow
python scripts\make_sample_pdfs.py
```

> The first RAG / semantic-plagiarism request downloads the embedding model (internet needed once; it is
> cached afterwards — set `HF_HUB_OFFLINE=1` to stop network checks). For a no-download lexical fallback set
> `EMBEDDING_BACKEND=hashing` (it matches shared *words*, not paraphrases, and says so in responses).

## Trying each workflow

In PowerShell use **`curl.exe`** (plain `curl` is an alias for `Invoke-WebRequest`) and the backtick `` ` `` for line continuation.
`-F "rubric=<samples/rubric.json"` reads a *text field* from a file, which avoids PowerShell JSON-quoting problems.

**Question generation – JSON (existing contract)**
```powershell
curl.exe -X POST http://localhost:8000/api/questions/generate `
  -H "Content-Type: application/json" --data "@samples/generate.json"
```

**Question generation – PDF upload** (text, scanned or mixed; fields: `file`, `num_questions`, `difficulty`, repeated or comma-separated `question_types`, `total_marks`, `subject`, `instructions`)
```powershell
curl.exe -X POST http://localhost:8000/api/questions/generate-pdf `
  -F "file=@samples/scanned_biology.pdf" `
  -F "num_questions=5" -F "difficulty=medium" `
  -F "question_types=mcq" -F "question_types=short_answer" -F "question_types=fill_in_the_blanks" `
  -F "total_marks=20" -F "subject=Biology"
```

**Evaluation – JSON, then PDF answer (question and rubric can be separate files too)**
```powershell
curl.exe -X POST http://localhost:8000/api/evaluation/evaluate `
  -H "Content-Type: application/json" --data "@samples/evaluate.json"

curl.exe -X POST http://localhost:8000/api/evaluation/evaluate `
  -F "answer_file=@samples/student_answer_scanned.pdf" `
  -F "question_file=@samples/question_paper.pdf" `
  -F "rubric=<samples/rubric.json"
# or let the AI read the rubric from a PDF (flagged for review): -F "rubric_file=@samples/rubric.pdf"
```

**Batch evaluation** (`files` repeated; optional `student_labels='["Asha","Ravi"]'`; JSON list also accepted)
```powershell
curl.exe -X POST http://localhost:8000/api/evaluation/evaluate-batch `
  -F "files=@samples/student_answer.pdf" -F "files=@samples/student_answer_scanned.pdf" `
  -F "question=Explain where photosynthesis occurs." -F "rubric=<samples/rubric.json"

curl.exe -X POST http://localhost:8000/api/evaluation/evaluate-batch `
  -H "Content-Type: application/json" --data "@samples/batch.json"
```

**Plagiarism** (files and/or `texts`; optional `exclude_text`, `use_semantic`)
```powershell
curl.exe -X POST http://localhost:8000/api/plagiarism/check `
  -F "files=@samples/essay_original.pdf" -F "files=@samples/essay_copy.pdf" -F "files=@samples/essay_original_scanned.pdf"

curl.exe -X POST http://localhost:8000/api/plagiarism/check `
  -H "Content-Type: application/json" --data "@samples/plagiarism.json"
```

**RAG** (files and/or `texts`; optional `chunk_size`, `chunk_overlap`, `top_k`)
```powershell
curl.exe -X POST http://localhost:8000/api/rag/ask `
  -F "files=@samples/text_biology.pdf" -F "files=@samples/scanned_biology.pdf" `
  -F "question=Where does photosynthesis take place?" -F "top_k=3"

curl.exe -X POST http://localhost:8000/api/rag/ask `
  -H "Content-Type: application/json" --data "@samples/rag.json"
```

(On Linux/macOS the same commands work with `\` line continuation and `curl`.)

## How PDF processing works

`app/services/pdf_service.py` is the single, reusable path used by every endpoint (via dependency injection):

1. Validate size and the **`%PDF-` file signature** (file name and MIME type are never trusted), open with `pypdf`;
   reject corrupt, encrypted (password-protected) and zero-page files with specific errors.
2. Extract selectable text **page by page**; a page is "readable" if it has ≥ `MIN_CHARS_PER_PAGE` letters/digits and isn't garbled.
3. **Only pages that aren't readable** are rendered (`pypdfium2`, `OCR_DPI`, pixel-capped) and OCR'd with Tesseract.
   Blank pages are detected and skipped. Good text pages in mixed PDFs are never re-OCR'd.
4. Pages are combined with their numbers; the result reports `page_count`, `pages_processed`, per-page `method`
   (`text`/`ocr`/`none`) and `ocr_confidence`, `ocr_status`, `truncated`, and `warnings`.

`ocr_status`: `not_needed` · `used` · `partial` · `failed` · `unavailable` · `disabled`. If OCR is required and
unavailable you get `503 ocr_unavailable` with install instructions (or, for mixed PDFs where some text was found,
the text that could be read plus an explicit warning). Limits: `MAX_PDF_SIZE_MB`, `MAX_PDF_PAGES`, `OCR_MAX_PAGES`,
`OCR_DPI`, `OCR_TIMEOUT_SECONDS_PER_PAGE`, `PDF_PROCESSING_TIMEOUT_SECONDS`, `PDF_MAX_CONCURRENT_EXTRACTIONS`.
PDFs are processed **in memory**; this app writes no temp files itself (Starlette may spool very large uploads to its own temporary files, and pytesseract passes page images to Tesseract through short-lived temp files that it removes). Multilingual text works to the extent the installed Tesseract language packs and the PDF fonts support it.

## Response conventions

* **Question generation** — every question has `source_excerpt`, `source_pages`, `evidence_verified` (excerpt found verbatim
  in the extracted text). Marks always sum to `total_marks` when given. Invalid AI questions are dropped with a warning.
* **Evaluation** — totals/percentage are computed by the server; marks are clamped to `[0, max]` (flagged `marks_adjusted`);
  evidence quotes are verified against the answer; unsupported judgments are `assessment_basis: "uncertain"`; OCR-read
  answers, truncation, low confidence etc. set `needs_teacher_review` with `review_reasons`.
* **Batch** — per-file `status` (`success`/`failed`/`skipped`), deterministic input order, identical files evaluated once
  (`skipped` + `duplicate_of`), bounded concurrency (`BATCH_CONCURRENCY`), per-file timeout, summary counts.
  It is **synchronous**: keep batches within `BATCH_MAX_FILES`.
* **Plagiarism** — lexical overlap (n-gram containment, exact/near-duplicate sentences) and semantic similarity are reported
  separately, with matching passages and page numbers. Only documents *in the request* are compared; no internet or
  academic database is searched (the response says so).
* **RAG** — in-memory index per request (nothing stored, nothing shared between requests). Answers cite sources (`S1`…) with
  verified quotes and pages; weak evidence returns `sufficient_evidence: false` instead of a guess. Document text is treated as untrusted.

## Errors

Every error has the same envelope (`detail` is kept for FastAPI-style clients):

```json
{ "detail": "…", "error": { "code": "pdf_encrypted", "message": "…", "source": "client", "request_id": "…" } }
```

`source` is `client` (your request), `server` (this service) or `upstream_ai` (Gemini). Response header `X-Request-ID` matches `request_id`.

| HTTP | `error.code` |
|---|---|
| 400 | `bad_request` |
| 413 | `file_too_large`, `request_too_large`, `too_many_files` |
| 415 | `unsupported_file_type` (bad signature/type), `unsupported_media_type` |
| 422 | `validation_error`, `pdf_corrupt`, `pdf_encrypted`, `pdf_empty`, `file_empty`, `pdf_no_readable_text`, `insufficient_text` |
| 429 | `rate_limit_exceeded` (+ `Retry-After`) |
| 502 | `ai_authentication_failed`, `ai_request_rejected`, `ai_content_blocked`, `ai_malformed_response`, `ai_incomplete_response` |
| 503 | `ai_not_configured`, `ai_rate_limited`, `ai_unavailable`, `ocr_unavailable`, `embeddings_unavailable` |
| 504 | `ai_timeout`, `processing_timeout` |

In batch results, failures appear per file with the same `code`/`message`/`source` (plus `file_timeout`).

## Security & reliability

Request-size limits (early via `Content-Length` and while streaming), per-client rate limit (in-memory), CORS from
`CORS_ORIGINS` (also on error responses), upload validation by signature, sanitised file names (used only as labels),
bounded concurrency (PDF extraction, batch, Gemini), per-stage timeouts, no secrets in logs/responses/health
(`SecretStr`; provider error text is never forwarded), no content in logs, prompt-injection hardening (untrusted data is
delimited, delimiters neutralised, instructions to ignore embedded commands), no database, no file execution, no unsafe deserialisation.

## Gemini

Uses the maintained **`google-genai`** SDK with structured JSON output validated by Pydantic. Default model
`gemini-3.5-flash` (set `GEMINI_MODEL`; Gemini 2.5 models were scheduled for shutdown on 2026-10-16). Retries use exponential backoff +
jitter for 429/5xx/timeouts/connection errors only; auth, bad-request, blocked-content and truncation errors are not retried.
All AI calls go through `app/services/ai_service.py` (`AIService`), replaced by a fake in tests.

## Project layout

```
app/
  main.py            app factory, middleware, OpenAPI
  config.py          settings (.env)             errors.py   error types + envelope
  dependencies.py    DI container/providers      middleware/ request-id, size limit, rate limit
  routers/           health, questions, evaluation, plagiarism, rag
  schemas/           Pydantic request/response models
  services/          pdf_service, ocr, documents, ai_service, question_service, evaluation_service,
                     batch, plagiarism_service, rag_service, embeddings
  utils/             text helpers, concurrency
tests/               pytest suite (+ pdf_factory.py builds real PDFs)
scripts/make_sample_pdfs.py
```

## Known limitations

* **No frontend/repo was available when this was written**, so request/response shapes for the JSON endpoints are inferred from the
  specification (see below). Send the frontend API layer if names must be aligned.
* Rate limiting and caches are **per process**; use a shared limiter at the proxy for multi-instance deployments.
* Batch evaluation is synchronous (no job queue); very large batches should be split client-side.
* OCR quality depends on scan quality; handwriting is often unreadable. OCR text is flagged for teacher review.
* Plagiarism compares only documents within the request; sentences are matched within pages (a passage split across a page break can be missed).
* The default `RAG_MIN_SIMILARITY=0.25` is a starting point for the multilingual MiniLM model and should be tuned on real data.
* PDF text extraction uses `pypdf`; complex layouts (multi-column, tables) may extract in an imperfect reading order.

## Compatibility notes for the frontend

Preserved/assumed paths: `/api/questions/generate`, `/api/plagiarism/check`, `/api/evaluation/evaluate`,
`/api/evaluation/evaluate-batch`, `/api/rag/ask`, `/health`. Assumed JSON fields: question generation `source_text`, `num_questions`,
`difficulty`, `question_types`, `total_marks`, `subject`, `instructions` (types accept `MCQ`/`Short Answer`/`fill-in-the-blanks` style
aliases; responses use `mcq`, `short_answer`, `long_answer`, `fill_in_the_blanks`). Evaluation accepts `student_answer` (alias `answer`)
and rubric items `name`/`description`/`max_marks` (aliases `marks`, `points`). Plagiarism accepts `documents[]` or `text1`+`text2`;
RAG accepts `documents[]` or `source_text`. FastAPI-style `detail` is retained on all error responses.
