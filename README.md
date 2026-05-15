# NPS Preditive Case

A data science project following the **CRISP-DM** framework for the **Predictive NPS** case: anticipating customer satisfaction (identifying **detractors** via `nps_detrator` and complementary analysis of the NPS score) using only operational data available before the survey.

## Overview

This repository covers everything from business understanding to a **deployment demonstration** (model serialization and inference simulation). The base dataset for the challenge is `desafio_nps_fase_1.csv`.

Main deliverables:

- EDA and hypotheses about dissatisfaction drivers (e.g., complaints, delivery delays).
- Training and test sets ready for modeling.
- Classification models (baseline and Random Forest) and regression models (Linear and Random Forest Regressor) for NPS score prediction.
- Executive summary, metric targets, and a usage example with the persisted model.

## Project Structure

| File | Description |
| ---- | ----------- |
| `01_business_understanding.ipynb` | Business context, problem definition, and analytical objectives (Phase 1). |
| `02_data_understanding.ipynb` | EDA, descriptive statistics, distributions, and the relationship between variables and NPS (Phase 2). |
| `03_data_preparation.ipynb` | Data cleaning, feature engineering, target definition, and export of train/test sets (Phase 3). |
| `04_data_modeling.ipynb` | Training and evaluation: logistic regression, Random Forest (classification), linear regression, and Random Forest Regressor for NPS score (Phase 4). |
| `05_evaluation.ipynb` | Evaluation summary, business/metric targets, model saving with `joblib`, and a simulated "production" prediction (Phase 5). |
| `desafio_nps_fase_1.csv` | Original challenge dataset. |
| `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` | Sets generated during data preparation (input for modeling and the deployment notebook). |
| `model_rf_classifier.pkl` | Random Forest trained for detractor classification (generated in notebook 05; can be recreated by re-running the cells). |
| `requirements.txt` | Python environment dependencies. |

## Requirements

- Python **3.12+** (the project was developed with a compatible Jupyter environment; newer versions typically work with the same dependencies).
- `pip`
- Jupyter Notebook or JupyterLab

## Environment Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Start Jupyter:

```bash
jupyter lab
```

## Recommended Execution Order

Run the notebooks in this order to reproduce the full CRISP-DM flow:

1. `01_business_understanding.ipynb`
2. `02_data_understanding.ipynb`
3. `03_data_preparation.ipynb`
4. `04_data_modeling.ipynb`
5. `05_evaluation.ipynb`

**Notes:**

- `03_data_preparation.ipynb` generates `X_train.csv`, `X_test.csv`, `y_train.csv`, and `y_test.csv`.
- `04_data_modeling.ipynb` depends on those files to train and evaluate the models.
- `05_evaluation.ipynb` retrains the Random Forest with the same hyperparameters used in Phase 4, saves `model_rf_classifier.pkl`, and demonstrates `joblib.load` + prediction on a single row from `X_test.csv` (useful if the model object in notebook 04 was overwritten by later experiments).

## Modeling

**Classification (`nps_detrator`):**

- **Logistic regression** — baseline, with `class_weight='balanced'` and scaled features where applicable.
- **Random Forest** — main model (`RandomForestClassifier` with `class_weight='balanced'`, `random_state=42`), focused on recall, AUC-ROC, F1, and precision as documented in the evaluation phase.

**Regression:**

- **Linear regression** (`LinearRegression`) for NPS score prediction.
- **Random Forest Regressor** (`RandomForestRegressor`) as a non-linear alternative to capture more complex relationships between operational variables and NPS score.

Metrics and business interpretation (targets, expected impact, next steps) are consolidated in `05_evaluation.ipynb`.

## Expected Outcome

By working through the notebooks, you will get:

- Clear business alignment and well-defined analytical questions;
- Exploratory understanding of the data and key operational drivers;
- Reproducible preparation pipelines and train/test sets;
- Evaluated models and a **serialized artifact** ready for conceptual integration (API or batch);
- Executive narrative with recommendations for monitoring, A/B testing, and retraining.
