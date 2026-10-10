from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MODEL_ARTIFACT_PATH, MODEL_TABLE_PATH
from src.model_artifact import train_and_export_model


if __name__ == "__main__":
    metrics = train_and_export_model(MODEL_TABLE_PATH, MODEL_ARTIFACT_PATH)
    print(f"Saved model artifact to: {MODEL_ARTIFACT_PATH}")
    print("Held-out test metrics:")
    print(metrics.round(3).to_string(index=False))