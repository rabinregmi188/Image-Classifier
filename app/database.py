from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "predictions.db"


def get_connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_name TEXT NOT NULL,
                stored_name TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                features_json TEXT NOT NULL,
                predictions_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def insert_prediction(payload: dict[str, Any]) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions (
                original_name,
                stored_name,
                label,
                confidence,
                width,
                height,
                features_json,
                predictions_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["original_name"],
                payload["stored_name"],
                payload["label"],
                payload["confidence"],
                payload["width"],
                payload["height"],
                json.dumps(payload["features"]),
                json.dumps(payload["predictions"]),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def fetch_history(limit: int = 12) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def fetch_stats() -> dict[str, Any]:
    with get_connection() as conn:
        total_predictions = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        average_confidence = conn.execute("SELECT AVG(confidence) FROM predictions").fetchone()[0]
        top_labels = conn.execute(
            """
            SELECT label, COUNT(*) AS count
            FROM predictions
            GROUP BY label
            ORDER BY count DESC, label ASC
            LIMIT 4
            """
        ).fetchall()

    return {
        "total_predictions": int(total_predictions or 0),
        "average_confidence": round(float(average_confidence or 0.0), 3),
        "top_labels": [{"label": row["label"], "count": row["count"]} for row in top_labels],
    }


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "original_name": row["original_name"],
        "stored_name": row["stored_name"],
        "label": row["label"],
        "confidence": row["confidence"],
        "width": row["width"],
        "height": row["height"],
        "features": json.loads(row["features_json"]),
        "predictions": json.loads(row["predictions_json"]),
        "created_at": row["created_at"],
    }
