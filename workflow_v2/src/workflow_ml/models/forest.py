"""Random Forest (bagging of decision trees), one multi-output forest for all labels."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from workflow_ml.config import LABELS, SEED
from workflow_ml.features import feature_columns


class ForestModel:
    def __init__(self, n_trees: int = 300, min_leaf: int = 2, seed: int = SEED):
        self.forest = RandomForestClassifier(
            n_estimators=n_trees, min_samples_leaf=min_leaf, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
        )

    def fit(self, table: pd.DataFrame, rows: np.ndarray) -> "ForestModel":
        self.columns = feature_columns(table)
        frame = table.iloc[rows]
        self.forest.fit(frame[self.columns], frame[list(LABELS)])
        return self

    def predict_proba(self, table: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
        per_label = self.forest.predict_proba(table.iloc[rows][self.columns])
        return np.column_stack([scores[:, 1] for scores in per_label])
