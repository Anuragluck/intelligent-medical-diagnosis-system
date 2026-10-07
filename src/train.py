"""Train and compare the four classifiers used by the portfolio project."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data import DIAGNOSES, FEATURES, TARGET, make_dataset

try:
    import tensorflow as tf
except ImportError:
    tf = None


def train(records: int = 12_000, seed: int = 42) -> dict:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    np.random.seed(seed)
    if tf is not None:
        tf.random.set_seed(seed)
    root = Path(__file__).resolve().parents[1]
    artifact_dir = root / os.getenv("MODEL_DIR", "artifacts")
    data_dir = root / "data" / "generated"
    reports_dir = root / "reports"
    for directory in (artifact_dir, data_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    frame = make_dataset(records=records, seed=seed)
    frame.to_csv(data_dir / "synthetic_patients.csv", index=False)
    x = frame[FEATURES].astype("float32")
    y = frame[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=.2, random_state=seed, stratify=y
    )
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1500, class_weight="balanced")
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
            random_state=seed, n_jobs=-1,
        ),
        "SVM": make_pipeline(
            StandardScaler(), SVC(C=3, kernel="rbf", probability=True,
                                 class_weight="balanced", random_state=seed)
        ),
    }
    outcomes: dict[str, dict] = {}
    confusion: dict[str, list] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, prediction, average="macro", zero_division=0
        )
        outcomes[name] = {
            "accuracy": float(accuracy_score(y_test, prediction)),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
        }
        confusion[name] = confusion_matrix(y_test, prediction, labels=DIAGNOSES).tolist()
        joblib.dump(model, artifact_dir / f"{name.lower().replace(' ', '_')}.joblib")

    if tf is not None:
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train).astype("float32")
        x_test_scaled = scaler.transform(x_test).astype("float32")
        class_to_index = {label: idx for idx, label in enumerate(DIAGNOSES)}
        y_train_idx = y_train.map(class_to_index).to_numpy()
        y_test_idx = y_test.map(class_to_index).to_numpy()
        network = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(len(FEATURES),)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(len(DIAGNOSES), activation="softmax"),
        ])
        network.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        network.fit(x_train_scaled, y_train_idx, validation_split=.1, epochs=35,
                    batch_size=64, verbose=0,
                    callbacks=[tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=5, restore_best_weights=True
                    )])
        nn_probabilities = network.predict(x_test_scaled, verbose=0)
        network.save(artifact_dir / "neural_network.keras")
        joblib.dump(scaler, artifact_dir / "neural_network_scaler.joblib")
        nn_backend = "TensorFlow/Keras"
        nn_labels = DIAGNOSES
    else:
        # Portable fallback so the project can still be reproduced without TensorFlow.
        network = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=500,
                          early_stopping=True, random_state=seed),
        )
        network.fit(x_train, y_train)
        nn_probabilities = network.predict_proba(x_test)
        joblib.dump(network, artifact_dir / "neural_network.joblib")
        nn_backend = "scikit-learn MLP fallback (install TensorFlow for Keras model)"
        nn_labels = list(network.classes_)
        y_test_idx = y_test

    nn_prediction = np.asarray(nn_labels)[np.argmax(nn_probabilities, axis=1)]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, nn_prediction, average="macro", zero_division=0
    )
    outcomes["Neural Network"] = {
        "accuracy": float(accuracy_score(y_test, nn_prediction)),
        "macro_precision": float(precision), "macro_recall": float(recall), "macro_f1": float(f1),
    }
    confusion["Neural Network"] = confusion_matrix(
        y_test, nn_prediction, labels=DIAGNOSES
    ).tolist()

    report = {
        "dataset": "synthetic only", "records": int(records), "seed": int(seed),
        "test_records": int(len(y_test)), "diagnoses": DIAGNOSES,
        "features": FEATURES, "metrics": outcomes, "neural_network_backend": nn_backend,
        "confusion_matrices": confusion,
    }
    (reports_dir / "model_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(outcomes).T.sort_values("accuracy", ascending=False).to_csv(
        reports_dir / "model_metrics.csv", index_label="model"
    )
    print(pd.DataFrame(outcomes).T.sort_values("accuracy", ascending=False).to_string(
        float_format=lambda value: f"{value:.3f}"
    ))
    print(f"\nSynthetic records: {records}; artifacts: {artifact_dir}; reports: {reports_dir}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=12_000)
    parser.add_argument("--seed", type=int, default=42)
    options = parser.parse_args()
    train(records=options.records, seed=options.seed)
