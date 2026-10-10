"""Features must not change when every record after a cutoff date is deleted."""

import pandas as pd

from workflow_ml.data import load_raw
from workflow_ml.features import build_features, feature_columns, fit_vocabulary


def _truncate(raw: dict[str, pd.DataFrame], cutoff: pd.Timestamp) -> dict[str, pd.DataFrame]:
    cut = {"patients": raw["patients"]}
    for name, frame in raw.items():
        if name != "patients":
            cut[name] = frame[frame["DATE"] <= cutoff].reset_index(drop=True)
    conditions = cut["conditions"].copy()
    conditions["END"] = conditions["END"].where(conditions["END"] <= cutoff)
    cut["conditions"] = conditions
    return cut


def test_features_use_no_future_information():
    raw = load_raw()
    vocabulary = fit_vocabulary(raw)
    cutoff = pd.Timestamp("2012-06-30", tz="UTC")

    full = build_features(raw, vocabulary)
    past = build_features(_truncate(raw, cutoff), vocabulary)

    full = full[full["t"] <= cutoff].set_index("encounter_id").sort_index()
    past = past.set_index("encounter_id").sort_index()
    columns = feature_columns(past.reset_index())
    pd.testing.assert_frame_equal(full[columns], past[columns], check_dtype=False)
