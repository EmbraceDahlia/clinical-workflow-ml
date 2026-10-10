from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MODEL_ARTIFACT_PATH, MODEL_TABLE_PATH
from src.model_artifact import train_and_export_model


def ensemble_demo_patients() -> list[str] | None:
    """Patients the workflow_v2 bundle keeps out of training, when that bundle exists.

    Holding out the same patients here lets the app compare both models on patients
    that neither has seen.
    """
    try:
        from workflow_ml.serving import BUNDLE_PATH, load_bundle
    except ImportError:
        return None
    return load_bundle().demo_patients if BUNDLE_PATH.exists() else None


if __name__ == "__main__":
    metrics = train_and_export_model(
        MODEL_TABLE_PATH, MODEL_ARTIFACT_PATH, holdout_patients=ensemble_demo_patients()
    )
    print(f"Saved model artifact to: {MODEL_ARTIFACT_PATH}")
    print("Held-out test metrics:")
    print(metrics.round(3).to_string(index=False))