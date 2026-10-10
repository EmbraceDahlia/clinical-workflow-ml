"""Gradient boosting: trees are added one by one, each correcting the errors of the previous ones.

Uses scikit-learn's histogram-based implementation (the LightGBM algorithm).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from workflow_ml.config import LABELS, SEED
from workflow_ml.features import feature_columns
from workflow_ml.phenotypes import PhenotypeEncoder
from workflow_ml.split import early_stopping_split


class BoostingModel:
    def __init__(self, learning_rate: float = 0.04, leaves: int = 63,
                 phenotypes: bool = False, seed: int = SEED):
        self.params = dict(
            learning_rate=learning_rate, max_leaf_nodes=leaves, min_samples_leaf=20,
            max_features=0.4, l2_regularization=2.0, max_iter=2000,
            early_stopping=True, scoring="average_precision", n_iter_no_change=60,
            random_state=seed,
        )
        self.encoder = PhenotypeEncoder(seed=seed) if phenotypes else None
        self.seed = seed

    def _design(self, table: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
        frame = table.iloc[rows][self.columns]
        if self.encoder is None:
            return frame
        return pd.concat([frame, self.encoder.transform(table, rows)], axis=1)

    def fit(self, table: pd.DataFrame, rows: np.ndarray) -> "BoostingModel":
        # Early stopping watches patients the trees are not fitted on.
        fit_rows, stop_rows = early_stopping_split(table, rows, self.seed)
        # A column with no observed value in the fit rows cannot be binned.
        observed = table.iloc[fit_rows][feature_columns(table)].notna().any()
        self.columns = list(observed[observed].index)
        if self.encoder is not None:
            self.encoder.fit(table, fit_rows)
        x_fit, x_stop = self._design(table, fit_rows), self._design(table, stop_rows)

        self.boosters = {}
        for label in LABELS:
            target = table[label].to_numpy()
            booster = HistGradientBoostingClassifier(**self.params)
            self.boosters[label] = booster.fit(
                x_fit, target[fit_rows], X_val=x_stop, y_val=target[stop_rows]
            )
        return self

    def predict_proba(self, table: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
        frame = self._design(table, rows)
        return np.column_stack([self.boosters[label].predict_proba(frame)[:, 1] for label in LABELS])
