import json
from pathlib import Path
import unittest

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_result_model.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"
DATASET_PATH = BASE_DIR / "student_results.csv"


class TestMLPipeline(unittest.TestCase):
    def test_dataset_created(self):
        self.assertTrue(DATASET_PATH.is_file())

    def test_model_created(self):
        self.assertTrue(MODEL_PATH.is_file())

    def test_metrics_created(self):
        self.assertTrue(METRICS_PATH.is_file())

    def test_accuracy_is_valid(self):
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["training_records"], 240)
        self.assertEqual(metrics["testing_records"], 60)

    def test_model_prediction(self):
        model = joblib.load(MODEL_PATH)
        sample = pd.DataFrame(
            [
                {
                    "attendance": 85,
                    "internal_marks": 75,
                    "assignment_marks": 80,
                    "previous_score": 78,
                }
            ]
        )
        prediction = model.predict(sample)[0]
        self.assertIn(int(prediction), [0, 1])

    def test_high_performance_student(self):
        model = joblib.load(MODEL_PATH)
        sample = pd.DataFrame(
            [
                {
                    "attendance": 90,
                    "internal_marks": 85,
                    "assignment_marks": 88,
                    "previous_score": 80,
                }
            ]
        )
        prediction = model.predict(sample)[0]
        self.assertEqual(int(prediction), 1)

    def test_low_performance_student(self):
        model = joblib.load(MODEL_PATH)
        sample = pd.DataFrame(
            [
                {
                    "attendance": 55,
                    "internal_marks": 30,
                    "assignment_marks": 40,
                    "previous_score": 35,
                }
            ]
        )
        prediction = model.predict(sample)[0]
        self.assertEqual(int(prediction), 0)


if __name__ == "__main__":
    unittest.main()
