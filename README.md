# Intelligent Medical Diagnosis System

I built this project to practise a full machine-learning workflow, from preparing data and comparing models to using a trained model in a web app.

The dashboard takes symptoms and basic measurements, then shows estimates from the selected model. It can also save prediction history to PostgreSQL when a database is configured.

## Project features

- Creates a reproducible dataset of 12,000+ **synthetic records** by default.
- Uses Pandas and NumPy for data preparation and feature engineering.
- Adds symptom count, fever flag, and age band features.
- Trains and compares Logistic Regression, Random Forest, SVM, and a neural network.
- Reports accuracy, macro precision, macro recall, macro F1, and confusion matrices.
- Runs a Streamlit dashboard for entering symptoms and viewing ranked model estimates.
- Supports optional PostgreSQL storage for prediction history.

## Tools used

Python, TensorFlow/Keras, scikit-learn, Pandas, NumPy, Streamlit, PostgreSQL, and Docker Compose.

The neural network uses TensorFlow/Keras when TensorFlow is installed. The checked-in benchmark was generated in an environment without TensorFlow, so it used the scikit-learn MLP neural-network fallback.

## Model results

I generated 12,000 records with random seed 42 and used a stratified 80/20 train/test split. The best accuracy in this run was **86.5%**.

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Logistic Regression | 86.5% | 86.5% |
| Neural Network | 86.0% | 86.0% |
| SVM | 86.0% | 85.9% |
| Random Forest | 85.5% | 85.5% |

Full results and confusion matrices are in [`reports/model_metrics.json`](reports/model_metrics.json). These scores are from synthetic data and do not show how the models would perform on real patients.

## How to run

From the project folder, create a virtual environment, install the packages, train the models, and start the dashboard:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.train --records 12000 --seed 42
streamlit run app.py
```

The training command generates the CSV, saves model files locally, and writes the evaluation reports. Generated data and model artifacts are ignored by Git and can be recreated with the command above.

### PostgreSQL (optional)

With Docker Desktop running, start the local database and create a local environment file:

```powershell
docker compose up -d db
Copy-Item .env.example .env
```

Then start the dashboard with `streamlit run app.py`. The example database credentials are for local development only. Without a configured database, the dashboard still runs but does not save prediction history.

## Project flow

1. `src/data.py` generates the synthetic records and creates the engineered features.
2. `src/train.py` splits the data, trains four models, and writes their evaluation results.
3. `app.py` loads the trained models and shows ranked estimates for the entered demo symptoms.
4. `src/database.py` saves and reads prediction history when PostgreSQL is configured.

## Main files

```text
app.py                     Streamlit dashboard
src/data.py                Synthetic data and feature engineering
src/train.py               Training and model comparison
src/database.py            PostgreSQL prediction history
reports/model_metrics.csv  Model score summary
reports/model_metrics.json Full metrics and confusion matrices
docker-compose.yml         Local PostgreSQL setup
```

## Note about the data

All records in this project are generated for learning and demonstration. They are not real patient records. This app is not a medical device and must not be used to diagnose or treat anyone. Please do not enter real patient information. Model scores are not medical advice or clinical validation.
