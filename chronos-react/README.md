# Chronos frontend (React + Vite)

```powershell
npm install
npm run dev        # http://localhost:5173
```

Start the backend first (`uvicorn app.main:app --port 8000`). Port 5173 is already in the backend's default
`CORS_ORIGINS`. The backend URL can be changed in the UI, or via `VITE_API_URL` in a `.env` file.

Production build: `npm run build` (output in `dist/`; add its origin to `CORS_ORIGINS`).
