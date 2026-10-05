"""The new vectorised labels must equal the original pipeline's labels."""

import sys
from datetime import timedelta

import pandas as pd
import pytest

from workflow_ml.config import HORIZON_DAYS, LABELS, PROJECT_DIR
from workflow_ml.data import encounter_frame, load_raw
from workflow_ml.labels import build_labels, workflow_events

sys.path.insert(0, str(PROJECT_DIR.parent))
legacy = pytest.importorskip("src.build_model_table")


def test_labels_match_original_pipeline():
    raw = load_raw()
    index = encounter_frame(raw)
    sample = index[index["patient_id"].isin(index["patient_id"].drop_duplicates().sample(80, random_state=0))]
    new = build_labels(index, workflow_events(raw)).loc[sample.index]

    horizon = timedelta(days=HORIZON_DAYS)
    for label, events in legacy.make_workflow_events().items():
        events = events.dropna(subset=["PATIENT", "EVENT_DATE"])
        by_patient = events.groupby("PATIENT")["EVENT_DATE"].apply(list).to_dict()
        old = [
            int(any(t < event <= t + horizon for event in by_patient.get(patient, [])))
            for patient, t in zip(sample["patient_id"], sample["t"])
        ]
        assert new[label].tolist() == old, label


def test_labels_are_binary_and_complete():
    raw = load_raw()
    labels = build_labels(encounter_frame(raw), workflow_events(raw))
    assert list(labels.columns) == list(LABELS)
    assert set(pd.unique(labels.to_numpy().ravel())) <= {0, 1}
