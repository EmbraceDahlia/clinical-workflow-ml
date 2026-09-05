from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def _coerce_binary_scores(raw_scores):
    if isinstance(raw_scores, list):
        return [arr[:, 1] if getattr(arr, "ndim", 1) > 1 and arr.shape[1] > 1 else arr for arr in raw_scores]
    if hasattr(raw_scores, "ndim") and raw_scores.ndim > 1 and raw_scores.shape[1] > 1:
        return raw_scores[:, 1]
    return raw_scores


def extract_per_label_scores(model, X_test):
    if hasattr(model, "estimators_"):
        estimators = list(model.estimators_)
        if all(hasattr(est, "predict_proba") for est in estimators):
            return [est.predict_proba(X_test)[:, 1] for est in estimators]
        if all(hasattr(est, "decision_function") for est in estimators):
            return [est.decision_function(X_test) for est in estimators]
        raise AttributeError("Underlying label estimators must support predict_proba or decision_function")

    if hasattr(model, "predict_proba"):
        raw_scores = model.predict_proba(X_test)
        return _coerce_binary_scores(raw_scores)

    if hasattr(model, "decision_function"):
        raw_scores = model.decision_function(X_test)
        return _coerce_binary_scores(raw_scores)

    raise AttributeError("Model must expose either predict_proba or decision_function for evaluation")


def default_score_threshold(model) -> float:
    has_proba = hasattr(model, "predict_proba")
    if hasattr(model, "estimators_"):
        has_proba = has_proba or any(hasattr(est, "predict_proba") for est in model.estimators_)
    return 0.5 if has_proba else 0.0


def _as_score_array(score_list, index: int) -> np.ndarray:
    scores = score_list[index] if isinstance(score_list, list) else score_list
    return np.asarray(scores)


def _best_f1_threshold(y_true: np.ndarray, scores: np.ndarray, default_threshold: float) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if y_true.size == 0 or y_true.max() == y_true.min():
        return float(default_threshold)

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    candidates = [float(default_threshold)]
    if len(thresholds):
        candidates.extend(float(value) for value in thresholds)

    best_threshold = float(default_threshold)
    best_f1 = -1.0
    for threshold in dict.fromkeys(candidates):
        predicted = (scores >= threshold).astype(int)
        score = f1_score(y_true, predicted, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def tune_thresholds(
    y_true: pd.DataFrame,
    score_list,
    default_threshold: float,
) -> dict[str, float]:
    return {
        target: _best_f1_threshold(
            y_true[target].astype(int).to_numpy(),
            _as_score_array(score_list, index),
            default_threshold,
        )
        for index, target in enumerate(y_true.columns)
    }


def evaluate_scores(
    y_true: pd.DataFrame,
    score_list,
    thresholds: dict[str, float] | None = None,
    default_threshold: float = 0.5,
) -> pd.DataFrame:
    rows = []
    for index, target in enumerate(y_true.columns):
        labels = y_true[target].astype(int)
        scores = _as_score_array(score_list, index)
        threshold = float((thresholds or {}).get(target, default_threshold))
        predicted = (scores >= threshold).astype(int)
        if labels.nunique() < 2:
            pr_auc = 0.0
        else:
            pr_auc = float(average_precision_score(labels, scores))

        rows.append(
            {
                "target": target,
                "threshold": threshold,
                "accuracy": accuracy_score(labels, predicted),
                "precision": precision_score(labels, predicted, zero_division=0),
                "recall": recall_score(labels, predicted, zero_division=0),
                "f1": f1_score(labels, predicted, zero_division=0),
                "pr_auc": pr_auc,
                "confusion_matrix": confusion_matrix(labels, predicted).tolist(),
            }
        )

    return pd.DataFrame(rows)


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    score_list = extract_per_label_scores(model, X_test)
    return evaluate_scores(
        y_test,
        score_list,
        thresholds=thresholds,
        default_threshold=default_score_threshold(model),
    )


def summarize_per_label(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model_name, metrics in results.items():
        for _, row in metrics.iterrows():
            record = {
                "model": model_name,
                "target": row["target"],
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
            }
            if "pr_auc" in row:
                record["pr_auc"] = float(row["pr_auc"])
            if "threshold" in row:
                record["threshold"] = float(row["threshold"])
            rows.append(record)

    return pd.DataFrame(rows)
