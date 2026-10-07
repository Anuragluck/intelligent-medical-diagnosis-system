# Intelligent Medical Diagnosis System

A portfolio project that demonstrates a complete symptom-based machine-learning workflow: synthetic data generation, feature engineering, comparison of four classifiers, an interactive Streamlit dashboard, and optional PostgreSQL prediction history.

> **Safety and data note:** this repository uses generated synthetic records only. It is an educational demonstration, not a medical device, diagnosis, or treatment recommendation. Do not enter real patient information. The generated benchmark does not establish clinical accuracy or suitability.

## What is included

- Reproducible generator for 12,000 synthetic records (configurable), with six simulated outcome classes.
- Feature engineering for symptom count, fever flag, and age band.
- Evaluation of Logistic Regression, Random Forest, SVM, and a TensorFlow/Keras neural network on the same stratified holdout set.
- Saved model artifacts, a metrics report, and confusion matrix data from the training run.
- Streamlit interface for entering symptoms and viewing a model's ranked estimates.
- Optional PostgreSQL storage for prediction history. The app remains usable when PostgreSQL is not configured.

The reported scores are calculated when you train the models; they depend on the generated data and random seed. They should not be presented as validated clinical performance.

## Run locally

1. Create and activate a virtual environment, then install dependencies:

   ```bash
   python -m venv .venv
   # Windows: .venv\\Scripts\\activate
   # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Generate the data, train all four models, and write evaluation outputs:

   ```bash
   python -m src.train --records 12000 --seed 42
   ```

3. Start the dashboard:

   ```bash
   streamlit run app.py
   ```

4. (Optional) Start PostgreSQL with Docker Compose, copy `.env.example` to `.env`, and run the app again:

   ```bash
   docker compose up -d db
   ```

Generated data and trained artifacts are intentionally git-ignored. The training command recreates them locally.

## Project layout

```text
app.py                     Streamlit dashboard
src/data.py                Synthetic cohort and feature engineering
src/database.py            Optional PostgreSQL persistence
src/train.py               Training and evaluation entry point
artifacts/                 Local model files (generated)
data/generated/            Generated synthetic CSV (generated)
reports/                   Metrics and confusion matrices (generated)
docker-compose.yml         Local PostgreSQL service
```

## Model evaluation

The training pipeline uses a stratified 80/20 split and reports accuracy, macro precision/recall/F1, and a confusion matrix for each model. These are internal synthetic-data metrics only. For a real clinical study, the system would need an appropriately sourced, de-identified dataset, external validation, bias and safety assessment, and clinical oversight.

### Reproduced benchmark (seed 42)

The 12,000-record run in this repository produced the following holdout accuracies:

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Logistic Regression | 86.5% | 86.5% |
| Neural Network (scikit-learn fallback in this run) | 86.0% | 86.0% |
| SVM | 86.0% | 85.9% |
| Random Forest | 85.5% | 85.5% |

The full precision, recall, and confusion matrices are in `reports/model_metrics.json`. This run does not reproduce an 89% result. The resume claim should only use 89% if it comes from a separate, documented experiment; these generated records are not a substitute for clinical data.

## PostgreSQL configuration

Set `DATABASE_URL` in the environment (or a local `.env` file) to a PostgreSQL connection string. On a successful prediction, the app stores the selected model, predicted class, confidence, and symptom inputs. No database credentials are committed.
