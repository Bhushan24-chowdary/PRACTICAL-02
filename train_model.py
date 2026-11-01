import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = (
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score",
)
OUTPUT_DIR = Path(__file__).resolve().parent


def create_dataset() -> pd.DataFrame:
    """Create a deterministic synthetic dataset for the CI demonstration."""
    rng = np.random.default_rng(42)
    number_of_students = 300
    data = pd.DataFrame(
        {
            "attendance": rng.integers(50, 101, number_of_students),
            "internal_marks": rng.integers(20, 101, number_of_students),
            "assignment_marks": rng.integers(30, 101, number_of_students),
            "previous_score": rng.integers(30, 101, number_of_students),
        }
    )
    data["weighted_score"] = (
        0.25 * data["attendance"]
        + 0.35 * data["internal_marks"]
        + 0.20 * data["assignment_marks"]
        + 0.20 * data["previous_score"]
    )
    data["result"] = (data["weighted_score"] >= 60).astype(int)
    return data


def train_model(output_dir: Path = OUTPUT_DIR) -> float:
    """Train, evaluate, and save the model, dataset, and metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Creating dataset...")
    data = create_dataset()
    data.to_csv(output_dir / "student_results.csv", index=False)
    print("Dataset created successfully.")
    print(f"Number of records: {len(data)}")

    X = data.loc[:, FEATURES]
    y = data["result"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training records: {len(X_train)}")
    print(f"Testing records : {len(X_test)}")

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=42),
            ),
        ]
    )
    print("Training model...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    print("\nModel Evaluation")
    print("----------------")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(matrix)

    model_path = output_dir / "student_result_model.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel saved as {model_path.name}")

    metrics = {
        "accuracy": float(accuracy),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "confusion_matrix": matrix.tolist(),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Metrics saved as {metrics_path.name}")
    return float(accuracy)


if __name__ == "__main__":
    train_model()
