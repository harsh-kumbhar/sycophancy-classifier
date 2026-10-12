# Sycophancy Classifier — React frontend

A minimal React + Vite frontend for the existing FastAPI API. Includes Markdown/GFM and LaTeX rendering for model responses, plus evidence-based reasoning panels in Demo Mode.

## Prerequisites
- Node.js 20+ recommended
- FastAPI backend running at `http://127.0.0.1:8000`

## Start
From this directory:

```powershell
npm install
npm run dev
```

Open the local URL printed by Vite (normally http://localhost:5173).

## Math and Markdown rendering
Assistant responses render Markdown and LaTeX. Literal model delimiters such as `\[` and `\]` are normalized before rendering.

## API URL
By default, the frontend calls `http://127.0.0.1:8000/api`.
To override it, create `.env` in this frontend directory:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000/api
```

## Pages
1. Introduction
2. Workspace hub
3. Live chat with a persistent analysis panel
4. Demo mode with random validated example and live simulation
5. Dedicated demo results page

The frontend never computes a behavioral label. Classification remains in the existing Python backend.
