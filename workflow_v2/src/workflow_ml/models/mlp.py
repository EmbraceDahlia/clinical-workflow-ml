"""Multi-layer perceptron trained with backpropagation; four sigmoid outputs, one per label."""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline

from workflow_ml.config import LABELS, SEED
from workflow_ml.features import feature_columns
from workflow_ml.preprocess import dense_preprocessor


class MLPModel:
    def __init__(self, hidden: tuple[int, ...] = (256, 128), alpha: float = 1e-3, seed: int = SEED):
        self.hidden, self.alpha, self.seed = hidden, alpha, seed

    def fit(self, table: pd.DataFrame, rows: np.ndarray) -> "MLPModel":
        self.columns = feature_columns(table)
        network = MLPClassifier(
            hidden_layer_sizes=self.hidden, alpha=self.alpha, batch_size=256,
            learning_rate_init=1e-3, max_iter=60, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=6, random_state=self.seed,
        )
        self.pipeline = make_pipeline(dense_preprocessor(self.columns), network)
        frame = table.iloc[rows]
        self.pipeline.fit(frame[self.columns], frame[list(LABELS)].to_numpy())
        return self

    def predict_proba(self, table: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(table.iloc[rows][self.columns])
