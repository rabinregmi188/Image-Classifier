from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .classifier import VisualClassifier
from .database import fetch_history, fetch_stats, init_db, insert_prediction

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = Path(os.environ.get("UPLOADS_DIR", str(BASE_DIR / "uploads")))
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Image Classifier")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = VisualClassifier()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/history")
def history(limit: int = 12) -> dict[str, object]:
    safe_limit = min(max(limit, 1), 30)
    return {"items": fetch_history(safe_limit)}


@app.get("/api/stats")
def stats() -> dict[str, object]:
    return fetch_stats()


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Received an empty file.")

    suffix = Path(file.filename or "upload.png").suffix or ".png"
    stored_name = f"{uuid.uuid4().hex}{suffix.lower()}"
    destination = UPLOADS_DIR / stored_name
    destination.write_bytes(image_bytes)

    try:
        result = classifier.predict(image_bytes)
    except Exception as exc:  # pragma: no cover - defensive for bad images
        if destination.exists():
            destination.unlink()
        raise HTTPException(status_code=400, detail="Could not process that image.") from exc

    record_id = insert_prediction(
        {
            "original_name": file.filename or "uploaded-image",
            "stored_name": stored_name,
            "label": result["label"],
            "confidence": result["confidence"],
            "width": result["width"],
            "height": result["height"],
            "features": result["features"],
            "predictions": result["predictions"],
        }
    )

    return {
        "id": record_id,
        "image_url": f"/uploads/{stored_name}",
        **result,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/{full_path:path}")
def spa_fallback(full_path: str) -> FileResponse:
    candidate = STATIC_DIR / full_path
    if candidate.exists() and candidate.is_file():
        return FileResponse(candidate)
    return FileResponse(STATIC_DIR / "index.html")
