"""Threshold selection and per-label metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from workflow_ml.config import LABELS


def best_f1_thresholds(y_true: np.ndarray, scores: np.ndarray) -> np.ndarray:
    thresholds = []
    for column in range(y_true.shape[1]):
        precision, recall, cutoffs = precision_recall_curve(y_true[:, column], scores[:, column])
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        thresholds.append(cutoffs[np.argmax(f1[:-1])])
    return np.asarray(thresholds)


def f1_per_label(y_true: np.ndarray, decisions: np.ndarray) -> np.ndarray:
    true_positive = (y_true * decisions).sum(axis=0)
    return 2 * true_positive / np.maximum(y_true.sum(axis=0) + decisions.sum(axis=0), 1)


def score_table(y_true: np.ndarray, scores: np.ndarray, decisions: np.ndarray) -> pd.DataFrame:
    """`scores` are probabilities (for the AUCs); `decisions` are the 0/1 predictions."""
    rows = []
    for column, label in enumerate(LABELS):
        truth, guess = y_true[:, column], decisions[:, column]
        tp = int((truth * guess).sum())
        fp = int(guess.sum() - tp)
        fn = int(truth.sum() - tp)
        tn = int(len(truth) - tp - fp - fn)
        rows.append(
            {
                "label": label,
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1),
                "f1": 2 * tp / max(2 * tp + fp + fn, 1),
                "accuracy": (tp + tn) / len(truth),
                "pr_auc": average_precision_score(truth, scores[:, column]),
                "roc_auc": roc_auc_score(truth, scores[:, column]),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            }
        )
    return pd.DataFrame(rows)
