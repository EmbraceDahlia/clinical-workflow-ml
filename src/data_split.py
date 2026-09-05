from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TARGET_COLUMNS


def patient_group_split(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on patient ID so no patient appears in both train and test."""
    if "patient_id" not in df.columns:
        raise ValueError("DataFrame must contain a 'patient_id' column")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    unique_patients = df["patient_id"].dropna().unique()
    if len(unique_patients) < 2:
        raise ValueError("Need at least two unique patients to create a train/test split")

    target_columns = [column for column in TARGET_COLUMNS if column in df.columns]
    if target_col not in target_columns:
        target_columns.append(target_col)

    patient_groups = df.groupby("patient_id", sort=False)
    patient_rows = patient_groups.size().reindex(unique_patients).to_numpy(dtype=float)
    patient_positive_counts = (
        patient_groups[target_columns].sum().reindex(unique_patients).fillna(0).to_numpy(dtype=float)
    )
    n_test = max(1, min(len(unique_patients) - 1, round(len(unique_patients) * test_size)))
    desired_rows = patient_rows.sum() * test_size
    desired_positive_counts = patient_positive_counts.sum(axis=0) * test_size

    def score(test_rows: float, test_positive_counts: np.ndarray, test_patient_count: int) -> float:
        row_error = ((test_rows - desired_rows) / max(desired_rows, 1.0)) ** 2
        positive_denominator = np.maximum(desired_positive_counts, 1.0)
        positive_error = np.mean(
            ((test_positive_counts - desired_positive_counts) / positive_denominator) ** 2
        )
        patient_error = ((test_patient_count - n_test) / max(n_test, 1)) ** 2
        return float(row_error + positive_error + patient_error)

    best_test_indices: np.ndarray | None = None
    best_score = float("inf")
    for iteration in range(32):
        rng = np.random.default_rng(random_state + iteration)
        order = np.argsort(-patient_positive_counts.sum(axis=1) + rng.random(len(unique_patients)) * 1e-6)
        test_indices: list[int] = []
        test_rows = 0.0
        test_positive_counts = np.zeros(len(target_columns), dtype=float)

        for position, patient_index in enumerate(order):
            remaining = len(unique_patients) - position - 1
            slots_remaining = n_test - len(test_indices)
            must_test = slots_remaining == remaining + 1
            can_test = slots_remaining > 0
            if must_test:
                choose_test = True
            elif not can_test:
                choose_test = False
            else:
                candidate_score = score(
                    test_rows + patient_rows[patient_index],
                    test_positive_counts + patient_positive_counts[patient_index],
                    len(test_indices) + 1,
                )
                current_score = score(test_rows, test_positive_counts, len(test_indices))
                choose_test = candidate_score <= current_score

            if choose_test:
                test_indices.append(int(patient_index))
                test_rows += patient_rows[patient_index]
                test_positive_counts += patient_positive_counts[patient_index]

        test_index_set = set(test_indices)
        current_score = score(test_rows, test_positive_counts, len(test_indices))
        for test_position, test_index in enumerate(test_indices):
            for candidate_index in range(len(unique_patients)):
                if candidate_index in test_index_set:
                    continue
                candidate_rows = test_rows - patient_rows[test_index] + patient_rows[candidate_index]
                candidate_positive_counts = (
                    test_positive_counts
                    - patient_positive_counts[test_index]
                    + patient_positive_counts[candidate_index]
                )
                candidate_score = score(candidate_rows, candidate_positive_counts, len(test_indices))
                if candidate_score < current_score:
                    test_index_set.remove(test_index)
                    test_index_set.add(candidate_index)
                    test_indices[test_position] = candidate_index
                    test_rows = candidate_rows
                    test_positive_counts = candidate_positive_counts
                    current_score = candidate_score
                    test_index = candidate_index

        final_score = score(test_rows, test_positive_counts, len(test_indices))
        if final_score < best_score:
            best_score = final_score
            best_test_indices = np.asarray(test_indices, dtype=int)

    assert best_test_indices is not None
    test_patients = set(unique_patients[best_test_indices])

    train_df = df[~df["patient_id"].isin(test_patients)].copy()
    test_df = df[df["patient_id"].isin(test_patients)].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split produced an empty set; reduce test_size or add more patients")

    return train_df, test_df
