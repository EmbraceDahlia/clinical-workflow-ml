from typing import Protocol

import numpy as np
import pandas as pd


class WorkflowModel(Protocol):
    """All models train on rows of the shared table and return an (n_rows, n_labels) score matrix."""

    def fit(self, table: pd.DataFrame, rows: np.ndarray) -> "WorkflowModel": ...

    def predict_proba(self, table: pd.DataFrame, rows: np.ndarray) -> np.ndarray: ...
