# Student Result Prediction ML CI

[![Student Result ML CI](https://github.com/Bhushan24-chowdary/student-result-ml-ci/actions/workflows/ml-ci.yml/badge.svg?branch=main)](https://github.com/Bhushan24-chowdary/student-result-ml-ci/actions/workflows/ml-ci.yml)

A small, reproducible machine-learning demonstration that predicts a student's result (`PASS` or `FAIL`) from attendance and assessment scores. GitHub Actions installs the dependencies, generates synthetic data, trains and evaluates a logistic-regression model, then runs the automated tests on every push and pull request to `main`.

> This project uses randomly generated demonstration records only. It is not trained on real student data and is not intended for consequential academic decisions.

## Model pipeline

- Generate 300 synthetic student records with a fixed random seed.
- Use attendance, internal marks, assignment marks, and previous score as features.
- Assign the target from a weighted score: 25% attendance, 35% internal marks, 20% assignments, and 20% previous score; scores of 60 or higher are `PASS`.
- Split the data into 240 training and 60 test records, scale the features, and train a logistic-regression classifier.
- Report accuracy and a confusion matrix, then save the model and metrics.

## Run locally

Python 3.11 or newer is recommended.

```powershell
python -m pip install -r requirements.txt
python train_model.py
python -m unittest discover -s . -p "test_*.py" -v
```

Training creates `student_results.csv`, `student_result_model.pkl`, and `metrics.json` in the project directory. These generated files are ignored by Git and are not committed by the workflow.

## GitHub Actions

The **Student Result ML CI** workflow is at `.github/workflows/ml-ci.yml`. It runs on pushes and pull requests targeting `main`, and can also be started manually from the repository's **Actions** tab using **Run workflow**. The job checks out the code, sets up Python 3.11, installs `requirements.txt`, runs `train_model.py`, and discovers the seven `unittest` checks. A failed training step or test makes the workflow fail.

The dataset, trained model, and metrics are generated on the temporary GitHub-hosted runner; this workflow does not commit them back to the repository.

## Repository layout

```text
.github/workflows/ml-ci.yml  GitHub Actions training and test workflow
README.md                    Project overview and usage
requirements.txt             Python dependencies
train_model.py               Synthetic-data generation and model training
test_ml_pipeline.py          Automated model and artifact tests
.gitignore                   Local generated files and Python caches
```
# student-result-ml-ci
Student Result Prediction ML model with GitHub Actions CI
