"""Streamlit interface for the synthetic diagnosis demo."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.data import DIAGNOSES, SYMPTOMS, prepare_input
from src.database import database_url, recent_predictions, save_prediction

try:
    import tensorflow as tf
except ImportError:
    tf = None

load_dotenv()
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / os.getenv("MODEL_DIR", "artifacts")

st.set_page_config(page_title="Medical Diagnosis Demo", page_icon="🩺", layout="wide")
st.title("Intelligent Medical Diagnosis System")
st.caption("A machine-learning portfolio demo using synthetic data")
st.warning(
    "Educational demonstration only. This is not medical advice and must not be used "
    "to diagnose, treat, or make decisions about anyone's health. Do not enter real patient data."
)


@st.cache_resource
def load_models():
    paths = [
        ARTIFACTS / "logistic_regression.joblib", ARTIFACTS / "random_forest.joblib",
        ARTIFACTS / "svm.joblib",
    ]
    if not all(path.exists() for path in paths):
        return None
    loaded = {
        "Logistic Regression": joblib.load(paths[0]),
        "Random Forest": joblib.load(paths[1]),
        "SVM": joblib.load(paths[2]),
    }
    keras_path = ARTIFACTS / "neural_network.keras"
    fallback_path = ARTIFACTS / "neural_network.joblib"
    if tf is not None and keras_path.exists():
        loaded["Neural Network"] = tf.keras.models.load_model(keras_path)
        loaded["neural_scaler"] = joblib.load(ARTIFACTS / "neural_network_scaler.joblib")
        loaded["neural_backend"] = "tensorflow"
    elif fallback_path.exists():
        loaded["Neural Network"] = joblib.load(fallback_path)
        loaded["neural_backend"] = "sklearn"
    else:
        return None
    return loaded


models = load_models()
if models is None:
    st.info("Models are not trained yet. Run `python -m src.train --records 12000 --seed 42` from the project folder.")
    st.stop()

with st.sidebar:
    st.header("Input features")
    age = st.slider("Age", min_value=1, max_value=100, value=35)
    temperature = st.slider("Temperature (°C)", min_value=35.0, max_value=41.5, value=37.0, step=.1)
    heart_rate = st.slider("Heart rate (bpm)", min_value=45, max_value=155, value=75)
    duration = st.slider("Symptom duration (days)", min_value=1, max_value=21, value=3)
    st.subheader("Symptoms")
    symptoms = {name: st.checkbox(name.replace("_", " ").title()) for name in SYMPTOMS}
    model_name = st.selectbox("Model", ["Logistic Regression", "Random Forest", "SVM", "Neural Network"])
    predict = st.button("Estimate", type="primary", width="stretch")

col_a, col_b = st.columns([1.2, 1])
with col_a:
    st.subheader("Prediction")
    if predict:
        raw_values = {
            **symptoms, "age": age, "temperature_c": temperature,
            "heart_rate": heart_rate, "symptom_duration_days": duration,
        }
        features = prepare_input(raw_values)
        if model_name == "Neural Network" and models["neural_backend"] == "tensorflow":
            model_features = models["neural_scaler"].transform(features).astype("float32")
            probabilities = models[model_name].predict(model_features, verbose=0)[0]
            labels = DIAGNOSES
        else:
            model = models[model_name]
            probabilities = model.predict_proba(features)[0]
            labels = list(model.classes_)
        ranked = sorted(zip(labels, probabilities), key=lambda item: item[1], reverse=True)
        diagnosis, confidence = ranked[0]
        st.metric("Highest model estimate", diagnosis, f"{confidence:.1%} model confidence")
        st.caption("Model confidence is not a probability that this is the true diagnosis.")
        display = pd.DataFrame(ranked, columns=["Outcome class", "Model score"])
        display["Model score"] = display["Model score"].map(lambda value: f"{value:.1%}")
        st.dataframe(display, width="stretch", hide_index=True)
        try:
            if save_prediction(model_name, diagnosis, float(confidence), raw_values):
                st.caption("Prediction saved to configured PostgreSQL history.")
        except Exception as exc:
            st.warning(f"Could not save prediction history: {exc}")
    else:
        st.write("Enter demo inputs in the sidebar and select **Estimate** to see model scores.")

with col_b:
    st.subheader("About this demo")
    st.write("Four classifiers were trained on a reproducible synthetic cohort: Logistic Regression, Random Forest, SVM, and a TensorFlow neural network.")
    st.write("Metrics describe performance on a synthetic holdout set only.")
    metrics_path = ROOT / "reports" / "model_metrics.csv"
    if metrics_path.exists():
        st.caption("Current synthetic holdout comparison")
        metric_df = pd.read_csv(metrics_path)
        st.dataframe(metric_df, width="stretch", hide_index=True)

if database_url():
    with st.expander("Recent demo prediction history"):
        try:
            rows = recent_predictions()
            if rows:
                history = pd.DataFrame(rows)
                history["created_at"] = pd.to_datetime(history["created_at"]).dt.strftime("%Y-%m-%d %H:%M UTC")
                history["confidence"] = history["confidence"].map(lambda value: f"{value:.1%}")
                st.dataframe(history, width="stretch", hide_index=True)
            else:
                st.caption("No predictions have been saved yet.")
        except Exception as exc:
            st.caption(f"PostgreSQL is configured but unavailable: {exc}")
