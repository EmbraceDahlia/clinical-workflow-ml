from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime

import pandas as pd

from src.config import TIME_WINDOWS


RECORD_TYPE_TO_FEATURE = {
    "Encounter": "encounters",
    "Allergy": "allergies",
    "Condition": "conditions",
    "Immunization": "immunizations",
    "Procedure": "procedures",
    "Imaging study": "imaging",
    "Medication": "medications",
    "Observation": "observations",
}
RECORD_TYPES = list(RECORD_TYPE_TO_FEATURE)


def build_model_features(
    current_features: Mapping[str, object],
    history_records: Sequence[Mapping[str, object]] | pd.DataFrame,
    evaluation_date: date | datetime | str,
    feature_columns: list[str],
) -> dict[str, object]:
    """Derive the model's longitudinal counts and recency from dated records."""
    evaluation_day = pd.Timestamp(evaluation_date).normalize()
    history = pd.DataFrame(history_records).copy()
    if history.empty:
        history = pd.DataFrame(columns=["Record type", "Record date"])

    history["feature_type"] = history["Record type"].map(RECORD_TYPE_TO_FEATURE)
    history["event_date"] = (
        pd.to_datetime(history["Record date"], errors="coerce", utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )
    history = history.dropna(subset=["feature_type", "event_date"])
    history = history.loc[history["event_date"] < evaluation_day]

    features = dict(current_features)
    for feature_type in RECORD_TYPE_TO_FEATURE.values():
        event_dates = history.loc[
            history["feature_type"].eq(feature_type), "event_date"
        ]
        for window_days in TIME_WINDOWS:
            lower_bound = evaluation_day - pd.Timedelta(days=window_days)
            features[f"prior_{feature_type}_count_{window_days}d"] = float(
                ((event_dates >= lower_bound) & (event_dates < evaluation_day)).sum()
            )
        features[f"days_since_{feature_type}_last"] = (
            int((evaluation_day - event_dates.max()).days)
            if not event_dates.empty
            else -1
        )

    encounter_dates = history.loc[
        history["feature_type"].eq("encounters"), "event_date"
    ]
    features["patient_total_encounters"] = int(len(encounter_dates))
    features["days_since_last_encounter"] = (
        int((evaluation_day - encounter_dates.max()).days)
        if not encounter_dates.empty
        else -1
    )

    missing = [column for column in feature_columns if column not in features]
    if missing:
        raise ValueError(f"Could not derive required model features: {missing}")
    return {column: features[column] for column in feature_columns}