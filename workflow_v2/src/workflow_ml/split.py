"""Patient-grouped splits: a patient's encounters never appear on both sides."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from workflow_ml.config import CV_FOLDS, LABELS, SEED

Fold = tuple[np.ndarray, np.ndarray]


def patient_folds(table: pd.DataFrame, seed: int = SEED) -> list[Fold]:
    """(train rows, held-out rows) pairs; every patient is held out exactly once."""
    # The four labels are packed into one class so folds keep similar label mixes.
    pattern = table[list(LABELS)].to_numpy() @ (2 ** np.arange(len(LABELS)))
    splitter = StratifiedGroupKFold(CV_FOLDS, shuffle=True, random_state=seed)
    return list(splitter.split(table, pattern, table["patient_id"]))


def early_stopping_split(
    table: pd.DataFrame, rows: np.ndarray, seed: int = SEED, fraction: float = 0.15
) -> Fold:
    splitter = GroupShuffleSplit(1, test_size=fraction, random_state=seed)
    fit, stop = next(splitter.split(rows, groups=table["patient_id"].to_numpy()[rows]))
    return rows[fit], rows[stop]
