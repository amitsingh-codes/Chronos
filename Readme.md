# Chronos — Teacher Workspace

Chronos is an AI-powered workspace designed to help teachers spend less time on repetitive academic tasks and more time teaching and connecting with students. It brings document-based question answering, question generation, answer evaluation, and similarity checking into one interface.

> **Project status:** Academic / prototype project. AI-generated assessments and grading are provisional and should be reviewed by a teacher before being used for official decisions.

## Demo video

Add the path or URL to your demo video below.

<!-- Replace the placeholder with a relative path (recommended) or a hosted video URL. -->
**Video:** `PATH_TO_YOUR_VIDEO`

For a video stored in this repository, you could use a path such as `assets/chronos-demo.mp4`. Create the `assets/` folder and put your video there, or replace the placeholder with a shareable URL.

GitHub Markdown does not reliably play an MP4 inline. To link to a video in the repository, use:

```markdown
[Watch the Chronos demo](assets/chronos-demo.mp4)
```

If the video is hosted online, replace the link with its URL.

## Features

- **Ask your material (RAG):** Upload study material and ask questions grounded in the document, with retrieved passages and page-aware citations.
- **Question Studio:** Generate questions from supplied text or documents with configurable question count, difficulty, question types, total marks, subject, and instructions.
- **Answer evaluation:** Assess student answers against question criteria or rubrics, with per-question marks, evidence, strengths, and suggested next steps.
- **PDF workflows:** Extract text from supported PDFs and render pages for OCR when required.
- **Evidence-aware responses:** Show supporting excerpts and citations so teachers can review how an answer was produced.
- **Similarity checking:** Compare submitted documents for potential text similarity.
- **Teacher-focused workspace:** Bring common document and assessment tasks into a single interface.

Feature availability depends on backend configuration and the services installed on the machine.

## Screenshots

Screenshots of the current interface can be added here. Store them in a folder such as `assets/screenshots/` and link them with relative paths:

```markdown
![Ask your material](assets/screenshots/ask-your-material.png)
![Question Studio](assets/screenshots/question-studio.png)
![Answer evaluation](assets/screenshots/answer-evaluation.png)
```

## Tech stack

Based on the current project setup:

- **Frontend:** Web UI running locally on port `5174` during development. Confirm the framework and dependencies in the frontend's `package.json`.
- **Backend:** Python with FastAPI and Uvicorn.
- **AI generation:** Google Gemini API.
- **Retrieval / embeddings:** Sentence Transformers (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` as shown in the health response).
- **PDF processing:** PDF text extraction and page rendering; OCR via Tesseract when installed and configured.
- **API documentation:** FastAPI's interactive Swagger UI.

> Update this section if your repository uses different libraries or services. Do not add technologies that are not actually used by the project.

## Architecture overview

1. A teacher enters a question, uploads learning material, or submits student work in the frontend.
2. The frontend sends a request to the corresponding FastAPI endpoint.
3. The backend extracts text from supported documents and can run OCR when needed.
4. Retrieval workflows select relevant passages from the provided material.
5. Gemini generates questions, answers, or assessment feedback using the request context.
6. The frontend displays results, supporting evidence, citations, and any available PDF output.
7. A teacher reviews AI-generated output before relying on it.

## Getting started

### Prerequisites

- Python installed.
- Node.js and npm installed for the frontend.
- A Google Gemini API key with access to a supported text-generation model.
- Git (optional, for cloning the repository).
- Tesseract OCR installed and configured if scanned or image-based PDFs need OCR.

### 1. Start the backend

Open a terminal in the backend project directory. In the current local setup, the backend folder was:

```powershell
cd "C:\Users\amitw\Downloads\scorematrix-backend-everthing\scorematrix-backend"
```

Create a virtual environment if one does not already exist:

```powershell
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project's dependencies using its provided requirements file (use the actual filename present in the repository):

```powershell
pip install -r requirements.txt
```

Create or update the `.env` file using the project's environment-variable template. At minimum, configure a valid Gemini API key and model. **Never commit your real API key to Git.**

Example configuration (adjust to the backend's actual settings):

```env
ENVIRONMENT=development
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:3000,http://localhost:5173,http://localhost:5174
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
GEMINI_MODEL=YOUR_SUPPORTED_GEMINI_MODEL
OCR_ENABLED=true
OCR_LANGUAGES=eng
TESSERACT_CMD=
```

Start the API:

```powershell
python -m uvicorn app.main:app --reload
```

The backend should be available at `http://127.0.0.1:8000`.

- Health check: <http://127.0.0.1:8000/health>
- API documentation: <http://127.0.0.1:8000/docs>

Keep the backend terminal open while using the frontend. If your backend folder, entry point, or environment variables differ, follow the actual repository configuration.

### 2. Start the frontend

Open a second terminal in the frontend project directory (the directory containing `package.json`), then install dependencies and start the development server:

```powershell
npm install
npm run dev
```

Open the local URL printed by the frontend development server. In the current setup, it was:

<http://localhost:5174>

If `npm run dev` uses another port, use the URL printed in the terminal and add that exact origin to `CORS_ORIGINS` in the backend `.env`, then restart the backend.

## Main workflows

### Ask your material

1. Upload a supported PDF or text document.
2. Enter a question about the material.
3. Choose the number of passages to retrieve.
4. Run the workflow.
5. Review the answer and citations against the original document.

### Question Studio

1. Provide source text or upload a supported source document, depending on the form's available input.
2. Choose the number of questions and difficulty.
3. Select one or more question types.
4. Optionally set total marks, subject, and additional instructions.
5. Generate the questions and review each answer and its supporting evidence.

### Answer evaluation

1. Upload student work and provide the relevant question, rubric, or reference material as required by the interface.
2. Run the evaluation.
3. Review per-question scores, evidence, strengths, and next steps.
4. Verify the extracted rubric and OCR output, especially for handwriting or low-quality scans.
5. Confirm marks manually before using them for formal grading.

### Similarity checking

1. Provide the documents supported by the similarity-checking interface.
2. Run the comparison.
3. Review the similarity results in context. Similarity is a signal for review, not proof of plagiarism by itself.

## Configuration notes

- **CORS:** The frontend's exact origin must be included in `CORS_ORIGINS`. For local development at port `5174`, include `http://localhost:5174`.
- **Gemini quota:** API requests may fail when the selected model's requests-per-day, requests-per-minute, token limits, or billing quota is exhausted. Use a model available to your Google AI Studio project and check the project's current quota.
- **OCR:** If the backend health endpoint reports that OCR is unavailable, install Tesseract OCR and configure `TESSERACT_CMD` to point to the executable if it is not on `PATH`. For Windows, the common install location is `C:\Program Files\Tesseract-OCR\tesseract.exe`; verify the actual location on your computer.
- **PDFs:** Text-based PDFs may work without OCR. Scanned or image-based PDFs generally need working OCR support.
- **Secrets:** Keep `.env` out of source control. Use a `.env.example` file with placeholder values for sharing configuration.

## API documentation

When the backend is running, open:

- Swagger UI: <http://127.0.0.1:8000/docs>
- Health endpoint: <http://127.0.0.1:8000/health>

Use Swagger UI to inspect each endpoint's required request fields and test the backend independently from the frontend.

## Troubleshooting

| Problem | What to check |
|---|---|
| Frontend cannot reach the backend | Confirm Uvicorn is running, the API base URL is correct, and the frontend origin is in `CORS_ORIGINS`. |
| HTTP 422 | Compare the request body or form fields with the endpoint schema shown in Swagger UI. |
| HTTP 429 / AI rate limit | Check Gemini model quota and retry only after the relevant limit resets. |
| HTTP 502 from the AI provider | Check the backend terminal for the underlying provider error, API key validity, and model availability. |
| OCR unavailable | Install Tesseract and configure its executable path. |
| Scanned PDF cannot be processed | Confirm OCR is installed and enabled; try a text-based PDF to isolate the issue. |
| PDF is rejected | Check file type, file size, page limits, and whether the PDF can be opened normally. |

## Responsible use

Chronos is intended to support teachers, not replace their judgment. AI outputs can be incomplete or incorrect. Teachers should review generated questions, answer keys, citations, extracted text, and marks before sharing them with students or recording official grades. Similarity results should also be reviewed rather than treated as a standalone plagiarism verdict.

## Repository structure

The exact structure depends on how the frontend and backend repositories are organized. A typical layout may look like:

```text
chronos/
├── frontend/              # Web application
├── backend/               # FastAPI service
│   ├── app/
│   │   └── main.py
│   ├── .env.example
│   └── requirements.txt
├── assets/
│   └── chronos-demo.mp4   # Optional demo video
└── README.md
```

Adjust this example to match the actual repository layout.

## Contributing

1. Create a branch for your change.
2. Keep API request formats aligned with the backend schema.
3. Test relevant workflows through the frontend and Swagger UI.
4. Never commit API keys, credentials, or private student documents.

## License

Add the license used by your project here. If no license has been chosen, do not imply that the project is open source under a particular license.
