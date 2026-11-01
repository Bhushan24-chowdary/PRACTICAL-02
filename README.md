# Student Result Prediction ML CI

[![Student Result ML CI](https://github.com/Bhushan24-chowdary/student-result-ml-ci/actions/workflows/ml-ci.yml/badge.svg?branch=main)](https://github.com/Bhushan24-chowdary/student-result-ml-ci/actions/workflows/ml-ci.yml)

A reproducible machine-learning demo that predicts a student's result (`PASS` or `FAIL`) from attendance and assessment scores. GitHub Actions installs dependencies, generates synthetic data, trains and evaluates a logistic-regression model, and runs automated tests on pushes and pull requests to `main`.

> Demonstration only: the data is synthetic, and this model is not intended for consequential academic decisions.

## Model pipeline

- Generate 300 synthetic records with a fixed random seed.
- Use attendance, internal marks, assignment marks, and previous score as features.
- Assign `PASS` for weighted scores of 60 or higher (25% attendance, 35% internal marks, 20% assignments, 20% previous score).
- Split into 240 training and 60 test records, scale the features, and train logistic regression.
- Report accuracy and a confusion matrix; save the model and metrics.

## Run locally

Python 3.11 or newer is recommended.

```powershell
python -m pip install -r requirements.txt
python train_model.py
python -m unittest discover -s . -p "test_*.py" -v
```

Training generates `student_results.csv`, `student_result_model.pkl`, and `metrics.json` in the project directory. They are ignored by Git and are not committed by CI.

## GitHub Actions

The **Student Result ML CI** workflow (`.github/workflows/ml-ci.yml`) runs on pushes and pull requests targeting `main`; it can also be started manually from the **Actions** tab. It uses Python 3.11, installs `requirements.txt`, trains the model, and runs seven automated tests. A failed training or test step fails the workflow.

Generated data, model, and metrics exist only on the temporary runner during CI.

## Repository layout

```text
.github/workflows/ml-ci.yml  CI workflow
README.md                    Project overview and usage
requirements.txt             Python dependencies
train_model.py               Data generation and model training
test_ml_pipeline.py          Automated model and artifact tests
.gitignore                   Generated files and Python caches
```
