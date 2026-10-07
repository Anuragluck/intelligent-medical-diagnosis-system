"""Create clearly synthetic symptom records for reproducible model demos."""

from __future__ import annotations

import numpy as np
import pandas as pd

DIAGNOSES = [
    "Common cold",
    "Seasonal allergy",
    "Influenza-like illness",
    "Migraine-like headache",
    "Gastroenteritis-like illness",
    "No major symptoms",
]

SYMPTOMS = [
    "cough", "sore_throat", "runny_nose", "sneezing", "itchy_eyes",
    "fever", "fatigue", "headache", "nausea", "diarrhea", "light_sensitivity",
]

FEATURES = [
    "age", "temperature_c", "heart_rate", "symptom_duration_days",
    *SYMPTOMS,
    "symptom_count", "fever_flag", "age_band",
]
TARGET = "diagnosis"

# Class-conditional symptom probabilities make labels learnable while retaining overlap.
_PROBABILITIES = np.array([
    [.72, .68, .76, .20, .10, .12, .38, .30, .08, .04, .04],
    [.08, .12, .45, .82, .78, .02, .12, .15, .03, .02, .02],
    [.35, .34, .24, .08, .04, .86, .78, .52, .20, .09, .05],
    [.02, .04, .03, .04, .02, .03, .26, .90, .30, .02, .78],
    [.02, .03, .03, .02, .01, .20, .55, .27, .81, .74, .02],
    [.03, .03, .04, .05, .03, .01, .06, .07, .02, .01, .01],
])


def make_dataset(records: int = 12_000, seed: int = 42) -> pd.DataFrame:
    """Return a deterministic, balanced synthetic cohort. No real patient data is used."""
    if records < len(DIAGNOSES):
        raise ValueError(f"records must be at least {len(DIAGNOSES)}")

    rng = np.random.default_rng(seed)
    labels = np.arange(records) % len(DIAGNOSES)
    rng.shuffle(labels)
    symptoms = np.vstack([
        rng.binomial(1, _PROBABILITIES[label]) for label in labels
    ]).astype("int8")

    # Clinical-like measurements are generated from class distributions for demo purposes.
    temp_means = np.array([37.1, 36.8, 38.2, 36.9, 37.4, 36.7])
    hr_means = np.array([82, 74, 98, 78, 91, 72])
    ages = rng.integers(1, 91, size=records)
    temperatures = np.clip(rng.normal(temp_means[labels], .45), 35.0, 41.5).round(1)
    heart_rates = np.clip(rng.normal(hr_means[labels], 11), 45, 155).round().astype(int)
    duration = np.clip(rng.gamma(shape=2.0, scale=1.7, size=records), 1, 21).round().astype(int)

    # Add modest label noise so the task is not perfectly separable.
    flip = rng.random(records) < .04
    labels[flip] = rng.integers(0, len(DIAGNOSES), size=flip.sum())

    frame = pd.DataFrame(symptoms, columns=SYMPTOMS)
    frame.insert(0, "symptom_duration_days", duration)
    frame.insert(0, "heart_rate", heart_rates)
    frame.insert(0, "temperature_c", temperatures)
    frame.insert(0, "age", ages)
    frame[TARGET] = [DIAGNOSES[index] for index in labels]
    return engineer_features(frame)


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive stable features from raw age and symptom columns."""
    result = frame.copy()
    result["symptom_count"] = result[SYMPTOMS].sum(axis=1).astype(int)
    result["fever_flag"] = (result["temperature_c"] >= 38.0).astype(int)
    result["age_band"] = pd.cut(
        result["age"], bins=[0, 12, 18, 40, 65, 120],
        labels=False, include_lowest=True,
    ).astype(int)
    return result


def prepare_input(values: dict[str, int | float | bool]) -> pd.DataFrame:
    """Build one model-ready row from dashboard values."""
    row = {name: int(values.get(name, 0)) for name in SYMPTOMS}
    row.update({
        "age": int(values["age"]),
        "temperature_c": float(values["temperature_c"]),
        "heart_rate": int(values["heart_rate"]),
        "symptom_duration_days": int(values["symptom_duration_days"]),
    })
    frame = pd.DataFrame([row])
    return engineer_features(frame)[FEATURES]
