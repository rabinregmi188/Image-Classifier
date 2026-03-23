from __future__ import annotations

from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile

from app.classifier import VisualClassifier
from app.database import fetch_history, fetch_stats, init_db, insert_prediction

app = FastAPI(title="Image Classifier API")
classifier = VisualClassifier()


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/")
async def handle_get(action: str = Query("health")) -> dict[str, Any]:
    if action == "health":
        return {"status": "ok"}

    if action == "stats":
        return fetch_stats()

    if action == "history":
        items = fetch_history(12)
        return {
            "items": [
                {
                    "id": item["id"],
                    "original_name": item["original_name"],
                    "label": item["label"],
                    "confidence": item["confidence"],
                    "created_at": item["created_at"],
                }
                for item in items
            ]
        }

    raise HTTPException(status_code=400, detail="Unsupported action.")


@app.post("/")
async def handle_post(
    request: Request,
    action: str = Query(...),
    file: UploadFile | None = File(default=None),
) -> dict[str, Any]:
    if action != "predict":
        raise HTTPException(status_code=400, detail="Unsupported action.")

    if file is None:
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Received an empty file.")

    try:
        result = classifier.predict(image_bytes)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail="Could not process that image.") from exc

    record_id = insert_prediction(
        {
            "original_name": file.filename or "uploaded-image",
            "stored_name": "vercel-function",
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
        **result,
        "request_path": str(request.url.path),
    }
