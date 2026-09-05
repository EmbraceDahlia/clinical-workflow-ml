from __future__ import annotations

import pandas as pd

from src.config import TARGET_COLUMNS


def define_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Return the binary workflow labels in a consistent column order."""
    missing = [column for column in TARGET_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing target columns: {missing}")

    y = df[TARGET_COLUMNS].copy()
    for column in TARGET_COLUMNS:
        y[column] = y[column].fillna(0).astype(int)
    return y
