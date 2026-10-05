"""Encounter-level features.

Leakage rule: every feature is computed from records dated at or before the encounter
start `t`. `tests/test_features.py` checks this by deleting all later records and
verifying that nothing changes.
"""

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from workflow_ml.config import ARTIFACT_DIR, HORIZON_DAYS, LABELS
from workflow_ml.data import encounter_frame, load_raw
from workflow_ml.labels import build_labels, workflow_events

META_COLUMNS = ("encounter_id", "patient_id", "t")
BASE_CATEGORICALS = ("enc_class", "enc_type", "reason")
LAGS = (1, 2, 3)
CATEGORICAL_COLUMNS = BASE_CATEGORICALS + tuple(
    f"prev{lag}_{column}" for lag in LAGS for column in BASE_CATEGORICALS
)
TRAILING_WINDOWS = (30, 90, 365)
HISTORY_SOURCES = ("conditions", "procedures", "medications", "careplans", "immunizations")
VISIT_CODE_SOURCES = ("conditions", "procedures", "medications", "careplans")
TABLE_PATH = ARTIFACT_DIR / "table.parquet"


@dataclass(frozen=True)
class Vocabulary:
    """Fixed code lists, so feature columns do not depend on which rows are present."""

    categories: dict[str, tuple[str, ...]]
    encounter_types: tuple[str, ...]
    conditions: tuple[str, ...]
    procedures: tuple[str, ...]
    visit_codes: dict[str, tuple[str, ...]]


def fit_vocabulary(
    raw: dict[str, pd.DataFrame],
    n_encounter_types: int = 20,
    n_conditions: int = 45,
    n_procedures: int = 30,
    n_visit_codes: int = 40,
) -> Vocabulary:
    index = encounter_frame(raw)

    def top(frame: pd.DataFrame, n: int) -> tuple[str, ...]:
        return tuple(frame["DESCRIPTION"].value_counts().head(n).index)

    return Vocabulary(
        categories={c: tuple(sorted(index[c].unique())) for c in BASE_CATEGORICALS},
        encounter_types=tuple(index["enc_type"].value_counts().head(n_encounter_types).index),
        conditions=top(raw["conditions"], n_conditions),
        procedures=top(raw["procedures"], n_procedures),
        visit_codes={source: top(raw[source], n_visit_codes) for source in VISIT_CODE_SOURCES},
    )


def _slug(text: str, limit: int = 32) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:limit]


def _unique_names(prefix: str, texts: tuple[str, ...]) -> list[str]:
    names: list[str] = []
    for position, text in enumerate(texts):
        name = f"{prefix}_{_slug(text)}"
        names.append(name if name not in names else f"{name}_{position}")
    return names


def _latest(index: pd.DataFrame, events: pd.DataFrame, inclusive: bool) -> pd.DataFrame:
    """For each encounter, the patient's most recent event before `t` (or at `t` if inclusive)."""
    left = index[["patient_id", "t"]].reset_index().sort_values("t", kind="stable")
    merged = pd.merge_asof(
        left, events.sort_values("DATE", kind="stable"), left_on="t", right_on="DATE",
        left_by="patient_id", right_by="PATIENT", allow_exact_matches=inclusive,
    )
    return merged.set_index("index").reindex(index.index)


def _recency(index: pd.DataFrame, events: pd.DataFrame, name: str) -> dict[str, pd.Series]:
    """Days since the last event, the gaps between earlier events, and the running count.

    `due` is the time left until the next event if the last gap repeats, which lets tree
    models see schedules (yearly wellness visits, four-weekly prenatal visits) directly.
    """
    events = events[["PATIENT", "DATE"]].drop_duplicates().sort_values(["PATIENT", "DATE"])
    by_patient = events.groupby("PATIENT")["DATE"]
    events = events.assign(gap=by_patient.diff().dt.days, count=by_patient.cumcount() + 1)
    events["previous_gap"] = events.groupby("PATIENT")["gap"].shift(1)

    last = _latest(index, events, inclusive=False)
    since = (index["t"] - last["DATE"]).dt.days
    return {
        f"since_{name}": since,
        f"due_{name}": last["gap"] - since,
        f"gap_{name}": last["gap"],
        f"gap2_{name}": last["previous_gap"],
        f"count_{name}": last["count"].fillna(0),
    }


def _events_before(index: pd.DataFrame, events: pd.DataFrame, days_back: int = 0) -> pd.Series:
    """How many of the patient's events are dated before `t - days_back`."""
    events = events[["PATIENT", "DATE"]].drop_duplicates().sort_values(["PATIENT", "DATE"])
    events = events.assign(count=events.groupby("PATIENT").cumcount() + 1)
    shifted = index.assign(t=index["t"] - pd.Timedelta(days=days_back))
    return _latest(shifted, events, inclusive=False)["count"].fillna(0)


def _trailing_counts(index: pd.DataFrame, days: int) -> np.ndarray:
    """Number of earlier encounters of the same patient within the last `days` days."""
    window = days * 86_400
    seconds = (index["t"] - index["t"].min()).dt.total_seconds().to_numpy().astype(np.int64)
    patient = index["patient_id"].astype("category").cat.codes.to_numpy().astype(np.int64)
    # One sortable key per row; the stride keeps a patient's window out of the previous patient.
    key = patient * (seconds.max() + window + 1) + seconds
    return np.arange(len(index)) - np.searchsorted(key, key - window, side="left")


def _encounter_features(index: pd.DataFrame, vocabulary: Vocabulary) -> dict[str, pd.Series]:
    features: dict[str, pd.Series] = {
        "age": index["age"],
        "male": index["male"],
        "year": index["t"].dt.year,
    }
    by_patient = index.groupby("patient_id", sort=False)
    features["n_previous"] = by_patient.cumcount()

    codes = {
        column: pd.Series(
            pd.Categorical(index[column], categories=vocabulary.categories[column]).codes,
            index=index.index,
        )
        for column in BASE_CATEGORICALS
    }
    for column, values in codes.items():
        features[column] = values
        for lag in LAGS:
            features[f"prev{lag}_{column}"] = (
                values.groupby(index["patient_id"], sort=False).shift(lag).fillna(-1).astype(int)
            )

    for lag in range(1, 6):
        gap = by_patient["t"].shift(lag - 1) - by_patient["t"].shift(lag)
        features[f"encounter_gap{lag}"] = gap.dt.days
    for days in TRAILING_WINDOWS:
        features[f"encounters_last_{days}d"] = _trailing_counts(index, days)

    # Chains of visits that share the same reason or type (follow-ups, prenatal schedule).
    for column in ("reason", "enc_type"):
        chain = index.groupby(["patient_id", column], sort=False)["t"]
        features[f"same_{column}_since"] = (index["t"] - chain.shift(1)).dt.days
        features[f"same_{column}_gap"] = (chain.shift(1) - chain.shift(2)).dt.days
        features[f"same_{column}_count"] = chain.cumcount()

    as_events = index.rename(columns={"patient_id": "PATIENT", "t": "DATE"})
    for enc_class in vocabulary.categories["enc_class"]:
        events = as_events[as_events["enc_class"] == enc_class]
        features.update(_recency(index, events, f"class_{enc_class}"))
    for name, enc_type in zip(_unique_names("type", vocabulary.encounter_types),
                              vocabulary.encounter_types):
        features.update(_recency(index, as_events[as_events["enc_type"] == enc_type], name))
    return features


def _workflow_history_features(
    index: pd.DataFrame, events: dict[str, pd.DataFrame]
) -> dict[str, pd.Series]:
    features: dict[str, pd.Series] = {}
    for label in LABELS:
        features.update(_recency(index, events[label], label))
        # Recent event rates describe how intensively this patient is being followed.
        total = _events_before(index, events[label])
        for days in TRAILING_WINDOWS:
            features[f"{label}_last_{days}d"] = total - _events_before(index, events[label], days)
        at_visit = index[["patient_id", "t"]].merge(
            events[label].assign(hit=1), how="left",
            left_on=["patient_id", "t"], right_on=["PATIENT", "DATE"],
        )
        features[f"now_{label}"] = at_visit["hit"].fillna(0).to_numpy()
    return features


def _outcome_history_features(index: pd.DataFrame, labels: pd.DataFrame) -> dict[str, pd.Series]:
    """Share of the patient's earlier comparable encounters that were followed by each workflow.

    Only encounters at least HORIZON_DAYS old count: their 30-day window has closed, so
    their outcome is already known when the current encounter starts.
    """
    scopes = {
        "patient": ["patient_id"],
        "type": ["patient_id", "enc_type"],
        "reason": ["patient_id", "reason"],
    }
    frame = pd.concat([index[["patient_id", "t", "enc_type", "reason"]], labels], axis=1)
    frame = frame.sort_values("t", kind="stable")
    lookup = frame.assign(t=frame["t"] - pd.Timedelta(days=HORIZON_DAYS)).reset_index()

    features: dict[str, pd.Series] = {}
    for scope, keys in scopes.items():
        grouped = frame.groupby(keys, sort=False)
        running = grouped[list(LABELS)].cumsum().assign(n=grouped.cumcount() + 1)
        resolved = pd.merge_asof(
            lookup[["index", "t", *keys]], pd.concat([frame[["t", *keys]], running], axis=1),
            on="t", by=keys,
        ).set_index("index").reindex(index.index)
        features[f"resolved_{scope}_count"] = resolved["n"].fillna(0)
        for label in LABELS:
            features[f"rate_{scope}_{label}"] = resolved[label] / resolved["n"]
    return features


def _condition_features(
    index: pd.DataFrame, conditions: pd.DataFrame, vocabulary: Vocabulary
) -> dict[str, pd.Series]:
    """Days each common condition has been active; missing when absent or resolved."""
    features: dict[str, pd.Series] = {}
    by_description = dict(tuple(conditions.groupby("DESCRIPTION")))
    empty = conditions.iloc[:0]
    for name, description in zip(_unique_names("active", vocabulary.conditions),
                                 vocabulary.conditions):
        onset = _latest(index, by_description.get(description, empty)[["PATIENT", "DATE", "END"]],
                        inclusive=True)
        active = onset["DATE"].notna() & (onset["END"].isna() | (onset["END"] >= index["t"]))
        features[name] = (index["t"] - onset["DATE"]).dt.days.where(active)
    return features


def _visit_content_features(
    index: pd.DataFrame, raw: dict[str, pd.DataFrame], vocabulary: Vocabulary
) -> dict[str, pd.Series]:
    """What was recorded at the encounter itself, limited to records stamped no later than `t`."""
    start = index.set_index("encounter_id")["t"]
    features: dict[str, pd.Series] = {}
    for source in (*VISIT_CODE_SOURCES, "observations", "immunizations"):
        records = raw[source]
        records = records[records["DATE"] <= records["ENCOUNTER"].map(start)]
        counts = records.groupby("ENCOUNTER").size()
        features[f"now_n_{source}"] = index["encounter_id"].map(counts).fillna(0)

        codes = vocabulary.visit_codes.get(source, ())
        encounters_with = records.groupby("DESCRIPTION")["ENCOUNTER"].agg(set).to_dict()
        for name, description in zip(_unique_names(f"now_{source}", codes), codes):
            hit = index["encounter_id"].isin(encounters_with.get(description, set()))
            features[name] = hit.astype("int8")
    return features


def build_features(raw: dict[str, pd.DataFrame], vocabulary: Vocabulary) -> pd.DataFrame:
    index = encounter_frame(raw)
    events = workflow_events(raw)

    features = _encounter_features(index, vocabulary)
    features.update(_workflow_history_features(index, events))
    features.update(_outcome_history_features(index, build_labels(index, events)))
    features.update(_condition_features(index, raw["conditions"], vocabulary))
    for source in HISTORY_SOURCES:
        features.update(_recency(index, raw[source], source))
    by_procedure = dict(tuple(raw["procedures"].groupby("DESCRIPTION")))
    for name, procedure in zip(_unique_names("proc", vocabulary.procedures), vocabulary.procedures):
        features.update(_recency(index, by_procedure.get(procedure, raw["procedures"].iloc[:0]), name))
    features.update(_visit_content_features(index, raw, vocabulary))

    matrix = pd.DataFrame({name: np.asarray(values) for name, values in features.items()})
    numeric = [column for column in matrix.columns if column not in CATEGORICAL_COLUMNS]
    matrix[numeric] = matrix[numeric].astype("float32")
    return pd.concat([index[list(META_COLUMNS)], matrix], axis=1)


def build_table(raw: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Meta columns + features + the four labels, one row per encounter."""
    raw = raw or load_raw()
    table = build_features(raw, fit_vocabulary(raw))
    labels = build_labels(table, workflow_events(raw))
    return pd.concat([table, labels], axis=1)


def load_table(refresh: bool = False) -> pd.DataFrame:
    if refresh or not TABLE_PATH.exists():
        TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        build_table().to_parquet(TABLE_PATH, index=False)
    return pd.read_parquet(TABLE_PATH)


def feature_columns(table: pd.DataFrame) -> list[str]:
    excluded = set(META_COLUMNS) | set(LABELS)
    return [column for column in table.columns if column not in excluded]
