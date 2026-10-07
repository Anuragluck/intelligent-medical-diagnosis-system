"""Optional PostgreSQL access for prediction history."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()


def database_url() -> str | None:
    return os.getenv("DATABASE_URL")


def save_prediction(model_name: str, diagnosis: str, confidence: float, inputs: dict) -> bool:
    url = database_url()
    if not url:
        return False
    import psycopg

    with psycopg.connect(url) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id BIGSERIAL PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL,
                model_name TEXT NOT NULL,
                predicted_diagnosis TEXT NOT NULL,
                confidence DOUBLE PRECISION NOT NULL,
                input_summary JSONB NOT NULL
            )
        """)
        connection.execute(
            "INSERT INTO prediction_history "
            "(created_at, model_name, predicted_diagnosis, confidence, input_summary) "
            "VALUES (%s, %s, %s, %s, %s)",
            (datetime.now(timezone.utc), model_name, diagnosis, confidence,
             psycopg.types.json.Jsonb(inputs)),
        )
    return True


def recent_predictions(limit: int = 20) -> list[dict]:
    url = database_url()
    if not url:
        return []
    import psycopg

    with psycopg.connect(url, row_factory=psycopg.rows.dict_row) as connection:
        return connection.execute(
            "SELECT created_at, model_name, predicted_diagnosis, confidence "
            "FROM prediction_history ORDER BY created_at DESC LIMIT %s", (limit,)
        ).fetchall()
