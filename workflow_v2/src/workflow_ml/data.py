"""Load the Synthea CSV export into tidy frames with one UTC `DATE` column each."""

from pathlib import Path

import pandas as pd

from workflow_ml.config import DATA_DIR

_EVENT_DATE_COLUMN = {
    "encounters": "START",
    "conditions": "START",
    "medications": "START",
    "careplans": "START",
    "procedures": "DATE",
    "observations": "DATE",
    "imaging_studies": "DATE",
    "immunizations": "DATE",
}


def _to_utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce").dt.as_unit("ns")


def load_raw(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    raw = {"patients": pd.read_csv(data_dir / "patients.csv", encoding="utf-8-sig")}
    raw["patients"]["BIRTHDATE"] = _to_utc(raw["patients"]["BIRTHDATE"])

    for name, date_column in _EVENT_DATE_COLUMN.items():
        frame = pd.read_csv(data_dir / f"{name}.csv", encoding="utf-8-sig", low_memory=False)
        frame["DATE"] = _to_utc(frame[date_column])
        raw[name] = frame.dropna(subset=["PATIENT", "DATE"]).reset_index(drop=True)

    raw["conditions"]["END"] = _to_utc(raw["conditions"]["STOP"])
    return raw


def encounter_frame(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per encounter, ordered by patient then time. This is the modelling index."""
    patients = raw["patients"].rename(columns={"Id": "PATIENT"})[["PATIENT", "BIRTHDATE", "GENDER"]]
    encounters = raw["encounters"].merge(patients, on="PATIENT", how="left")

    frame = pd.DataFrame(
        {
            "encounter_id": encounters["Id"],
            "patient_id": encounters["PATIENT"],
            "t": encounters["DATE"],
            "enc_class": encounters["ENCOUNTERCLASS"].astype(str),
            "enc_type": encounters["DESCRIPTION"].astype(str),
            "reason": encounters["REASONCODE"].fillna(0).astype("int64").astype(str),
            "age": ((encounters["DATE"] - encounters["BIRTHDATE"]).dt.days / 365.25).clip(0, 120),
            "male": (encounters["GENDER"] == "M").astype("int8"),
        }
    )
    return frame.sort_values(["patient_id", "t"], kind="stable").reset_index(drop=True)
