# Image Classifier

Image Classifier is a full-stack computer vision dashboard built with FastAPI, NumPy, SQLite, and a colorful frontend. Users can upload an image, run visual classification, inspect confidence scores, and review stored prediction history.

## Features

- Image upload with drag-and-drop UI
- NumPy-based visual classification pipeline
- Top-3 confidence scores and extracted visual features
- Prediction history persisted with SQLite
- Responsive dashboard with bright gradient styling
- FastAPI backend serving both API routes and frontend assets

## Tech Stack

- FastAPI
- NumPy
- SQLite
- JavaScript
- HTML
- CSS

## Vercel Deployment

This repo is structured for Vercel's Python runtime:

- root `app.py` exports the FastAPI `app`
- frontend assets live in `public/`
- uploads and SQLite history default to `/tmp` on Vercel

Important: Vercel's filesystem is ephemeral, so uploaded files and prediction history can reset between deployments or cold starts unless you move them to external storage.

## Run locally

```bash
python3 -m uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000`.
