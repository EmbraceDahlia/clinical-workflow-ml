from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_ARTIFACT_PATH, TARGET_COLUMNS
from src.evaluate_models import (
    default_score_threshold,
    evaluate_model,
    extract_per_label_scores,
    tune_thresholds,
)
from src.preprocess import build_preprocessor, select_feature_columns
from src.target_definition import define_targets
from src.train_baselines import train_model


def train_and_export_model(
    table_path: Path,
    output_path: Path = MODEL_ARTIFACT_PATH,
    holdout_patients: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Train, evaluate, and export the longitudinal Random Forest model.

    ``holdout_patients`` fixes the test patients, so another model that kept the same
    patients out of training can be compared on them. The bundle records its held-out
    patients either way.
    """
    from src.data_split import patient_group_split
    from src.config import RANDOM_STATE

    df = pd.read_csv(table_path)
    if holdout_patients is None:
        train_val_df, test_df = patient_group_split(
            df,
            target_col="medical_exam",
            random_state=RANDOM_STATE,
        )
    else:
        held_out = df["patient_id"].isin(set(holdout_patients))
        train_val_df, test_df = df[~held_out].copy(), df[held_out].copy()
    train_df, val_df = patient_group_split(
        train_val_df,
        target_col="medical_exam",
        random_state=RANDOM_STATE,
    )

    feature_columns = select_feature_columns(
        train_df.drop(columns=TARGET_COLUMNS),
        experiment="longitudinal",
    )
    X_train = train_df[feature_columns]
    X_val = val_df[feature_columns]
    y_train = define_targets(train_df)
    y_val = define_targets(val_df)

    threshold_preprocessor = build_preprocessor(X_train, feature_columns)
    X_train_processed = threshold_preprocessor.fit_transform(X_train)
    X_val_processed = threshold_preprocessor.transform(X_val)
    threshold_model = train_model(
        "random_forest",
        X_train_processed,
        y_train,
    )
    thresholds = tune_thresholds(
        y_val,
        extract_per_label_scores(threshold_model, X_val_processed),
        default_score_threshold(threshold_model),
    )

    X_train_val = train_val_df[feature_columns]
    final_preprocessor = build_preprocessor(X_train_val, feature_columns)
    X_train_val_processed = final_preprocessor.fit_transform(X_train_val)
    final_model = train_model(
        "random_forest",
        X_train_val_processed,
        define_targets(train_val_df),
    )
    X_test_processed = final_preprocessor.transform(test_df[feature_columns])
    test_metrics = evaluate_model(
        final_model,
        X_test_processed,
        define_targets(test_df),
        thresholds=thresholds,
    )

    artifact = {
        "schema_version": 1,
        "experiment": "longitudinal",
        "model_name": "random_forest",
        "feature_columns": feature_columns,
        "target_columns": TARGET_COLUMNS,
        "thresholds": thresholds,
        "holdout_patients": sorted(test_df["patient_id"].unique()),
        "preprocessor": final_preprocessor,
        "model": final_model,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    return test_metrics


def load_model_artifact(
    artifact_path: Path = MODEL_ARTIFACT_PATH,
) -> dict[str, object]:
    """Load a model bundle written by ``train_and_export_model``."""
    artifact = joblib.load(artifact_path)
    required_keys = {
        "schema_version",
        "feature_columns",
        "target_columns",
        "thresholds",
        "preprocessor",
        "model",
    }
    missing_keys = required_keys - artifact.keys()
    if missing_keys:
        raise ValueError(f"Model artifact is missing keys: {sorted(missing_keys)}")
    if artifact["schema_version"] != 1:
        raise ValueError("Unsupported model artifact schema version")
    return artifact


def predict_workflows(
    features: pd.DataFrame | Mapping[str, object],
    artifact: dict[str, object],
) -> pd.DataFrame:
    """Return per-label scores and thresholded predictions for input rows."""
    if isinstance(features, Mapping):
        input_frame = pd.DataFrame([features])
    else:
        input_frame = features.copy()

    feature_columns = artifact["feature_columns"]
    missing_columns = [
        column for column in feature_columns if column not in input_frame.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required model features: {missing_columns}")

    processed = artifact["preprocessor"].transform(input_frame[feature_columns])
    score_list = extract_per_label_scores(artifact["model"], processed)
    result = pd.DataFrame(index=input_frame.index)
    for index, target in enumerate(artifact["target_columns"]):
        scores = np.asarray(
            score_list[index] if isinstance(score_list, list) else score_list
        ).reshape(-1)
        threshold = float(artifact["thresholds"][target])
        result[f"{target}_score"] = scores
        result[f"{target}_predicted"] = scores >= threshold
    return result