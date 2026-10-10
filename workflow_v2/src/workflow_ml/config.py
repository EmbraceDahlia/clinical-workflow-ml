from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR.parent / "data"
ARTIFACT_DIR = PROJECT_DIR / "artifacts"

LABELS = ("medical_exam", "lab_test", "imaging", "instrument_test")
HORIZON_DAYS = 30

SEED = 42
CV_FOLDS = 5
