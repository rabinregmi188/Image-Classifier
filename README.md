# Image Classifier

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/rabinregmi188/Image-Classifier)

Image Classifier is a full-stack computer vision dashboard built with FastAPI, PyTorch, SQLite, and a colorful frontend. Users can upload an image, run visual classification, inspect confidence scores, and review stored prediction history.

## Features

- Image upload with drag-and-drop UI
- Torch-powered visual classification pipeline
- Top-3 confidence scores and extracted visual features
- Prediction history persisted with SQLite
- Responsive dashboard with bright gradient styling
- FastAPI backend serving both API routes and frontend assets

## Tech Stack

- FastAPI
- PyTorch
- SQLite
- JavaScript
- HTML
- CSS

## Run locally

```bash
python3 -m uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000`.

## Deploy on Render

This repo includes a `render.yaml` blueprint for a one-click Render setup.

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/api/health`

If you want uploads and prediction history to persist across redeploys, attach a persistent disk in Render and point these environment variables to that mount path:

- `DATABASE_PATH`
- `UPLOADS_DIR`
