"""Stacking: base models are scored out-of-fold, then a meta-learner combines their scores."""

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from workflow_ml.config import LABELS
from workflow_ml.models.base import WorkflowModel
from workflow_ml.split import Fold


def cross_fit(
    make_model: Callable[[], WorkflowModel], table: pd.DataFrame, folds: list[Fold]
) -> np.ndarray:
    """Out-of-fold scores: each row is scored by a model that never saw its patient."""
    scores = np.zeros((len(table), len(LABELS)))
    for train, held_out in folds:
        scores[held_out] = make_model().fit(table, train).predict_proba(table, held_out)
    return scores


def _logit(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


class StackingModel:
    """One logistic regression per label over the logits of every base model and label."""

    def __init__(self, strength: float = 0.1):
        self.strength = strength

    def fit(self, base_scores: np.ndarray, y_true: np.ndarray) -> "StackingModel":
        inputs = _logit(base_scores)
        self.learners = [
            LogisticRegression(C=self.strength, max_iter=2000).fit(inputs, y_true[:, column])
            for column in range(y_true.shape[1])
        ]
        return self

    def predict_proba(self, base_scores: np.ndarray) -> np.ndarray:
        inputs = _logit(base_scores)
        return np.column_stack([learner.predict_proba(inputs)[:, 1] for learner in self.learners])
