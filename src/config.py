from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_TABLE_PATH = ROOT / "model_table.csv"
MODEL_ARTIFACT_PATH = ROOT / "models" / "longitudinal_random_forest.joblib"
TIME_WINDOWS = [7, 30, 90, 180]
DEFAULT_TARGET_HORIZON_DAYS = 30
TARGET_COLUMNS = [
    "medical_exam",
    "lab_test",
    "imaging",
    "instrument_test",
]
RANDOM_STATE = 42

# Keep model comparison simple and clinically interpretable at first.
BASELINE_MODELS = [
    "logistic_regression",
    "random_forest",
    "linear_svm",
]
