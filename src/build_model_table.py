"""Build an encounter-level, leakage-controlled workflow dataset."""

from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MODEL_TABLE_PATH, TARGET_COLUMNS, TIME_WINDOWS
from src.procedure_mapping import classify_procedure


DATA = ROOT / "data"
OUTPUT = MODEL_TABLE_PATH
HORIZON_DAYS = 30


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name, encoding="utf-8-sig", low_memory=False)


LAB_OBSERVATION_TERMS = (
    "glucose", "cholesterol", "creatinine", "hemoglobin", "hematocrit",
    "platelet", "leukocyte", "erythrocyte", "triglyceride", "bilirubin",
    "albumin", "sodium", "potassium", "urine", "microalbumin",
    "urea nitrogen", "calcium", "chloride", "carbon dioxide",
    "mcv", "mch", "glomerular filtration", "globulin",
    "alkaline phosphatase", "alanine aminotransferase", "aspartate aminotransferase",
    "low density lipoprotein", "high density lipoprotein", "hemoglobin a1c",
    "prostate specific", "nt-probnp", "thyroxine", "thyrotropin",
    "ige ab", "wbc auto", "rbc auto", "leukocyte esterase",
    "in serum or plasma", "by automated count", "by test strip",
    "hemoglobin.gastrointestinal", "her2", "immune stain",
)
EXAM_ENCOUNTER_TERMS = (
    "general examination", "physical examination", "medical examination",
    "encounter for check up", "check-up", "check up",
    "well child visit", "well-child",
)


def text_contains(value: pd.Series, terms: tuple[str, ...]) -> pd.Series:
    pattern = "|".join(re.escape(term) for term in terms)
    return value.fillna("").str.lower().str.contains(pattern, regex=True)


def _label_event_frame(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["PATIENT", "EVENT_DATE"]
    if "DESCRIPTION" in df.columns:
        columns.append("DESCRIPTION")
    return df.loc[:, [column for column in columns if column in df.columns]].copy()


def make_workflow_events() -> dict[str, pd.DataFrame]:
    observations = load_csv("observations.csv")
    observations["EVENT_DATE"] = pd.to_datetime(observations["DATE"], utc=True)
    procedures = load_csv("procedures.csv")
    procedures["EVENT_DATE"] = pd.to_datetime(procedures["DATE"], utc=True)
    imaging = load_csv("imaging_studies.csv")
    imaging["EVENT_DATE"] = pd.to_datetime(imaging["DATE"], utc=True)
    encounters = load_csv("encounters.csv")
    encounters["EVENT_DATE"] = pd.to_datetime(encounters["START"], utc=True)

    observation_text = observations["DESCRIPTION"].fillna("")
    encounter_text = encounters["DESCRIPTION"].fillna("")

    lab_observations = observations[text_contains(observation_text, LAB_OBSERVATION_TERMS)]
    procedure_categories = procedures["DESCRIPTION"].apply(classify_procedure).str[0]
    lab_procedures = procedures[procedure_categories.eq("lab_test")]
    instrument_procedures = procedures[procedure_categories.eq("instrument_test")]
    imaging_procedures = procedures[procedure_categories.eq("imaging")]
    exam_procedures = procedures[procedure_categories.eq("medical_exam")]
    examinations = encounters[text_contains(encounter_text, EXAM_ENCOUNTER_TERMS)]

    return {
        "medical_exam": pd.concat(
            [_label_event_frame(examinations), _label_event_frame(exam_procedures)],
            ignore_index=True,
        ),
        "lab_test": pd.concat(
            [_label_event_frame(lab_observations), _label_event_frame(lab_procedures)],
            ignore_index=True,
        ),
        "imaging": pd.concat(
            [_label_event_frame(imaging), _label_event_frame(imaging_procedures)],
            ignore_index=True,
        ),
        "instrument_test": _label_event_frame(instrument_procedures),
    }


def _safe_days_since_reference(index_date: pd.Timestamp, event_dates: np.ndarray) -> int:
    prior_mask = event_dates < index_date
    if not np.any(prior_mask):
        return -1
    return int((index_date - event_dates[prior_mask][-1]).days)


def _compute_window_counts(event_dates: np.ndarray, encounter_dates: pd.Series, window_days: int) -> np.ndarray:
    if len(event_dates) == 0 or len(encounter_dates) == 0:
        return np.zeros(len(encounter_dates), dtype=float)

    event_array = pd.DatetimeIndex(event_dates).tz_localize(None).to_numpy("datetime64[ns]")
    encounter_array = pd.DatetimeIndex(encounter_dates).tz_localize(None).to_numpy("datetime64[ns]")
    window_delta = np.timedelta64(window_days, "D")
    counts = []
    for encounter_time in encounter_array:
        lower_bound = encounter_time - window_delta
        counts.append(np.count_nonzero((event_array >= lower_bound) & (event_array < encounter_time)))
    return np.asarray(counts, dtype=float)


def _compute_days_since_last(event_dates: np.ndarray, encounter_dates: pd.Series) -> np.ndarray:
    if len(event_dates) == 0 or len(encounter_dates) == 0:
        return np.full(len(encounter_dates), -1, dtype=int)

    event_array = np.sort(pd.DatetimeIndex(event_dates).tz_localize(None).to_numpy("datetime64[ns]"))
    encounter_array = pd.DatetimeIndex(encounter_dates).tz_localize(None).to_numpy("datetime64[ns]")
    positions = np.searchsorted(event_array, encounter_array, side="left")
    last_dates = np.full(len(encounter_array), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    valid = positions > 0
    last_dates[valid] = event_array[positions[valid] - 1]
    deltas = (encounter_array - last_dates).astype("timedelta64[D]").astype(np.int64)
    deltas = np.where(valid, deltas, -1)
    return deltas.astype(int)


# def _compute_prior_description_features(
#     event_dates: np.ndarray,
#     descriptions: np.ndarray,
#     encounter_dates: pd.Series,
# ) -> tuple[np.ndarray, np.ndarray]:
#     event_array = pd.DatetimeIndex(event_dates).tz_localize(None).to_numpy("datetime64[ns]")
#     encounter_array = pd.DatetimeIndex(encounter_dates).tz_localize(None).to_numpy("datetime64[ns]")
#     order = np.argsort(event_array)
#     event_array = event_array[order]
#     descriptions = descriptions[order]
#     positions = np.searchsorted(event_array, encounter_array, side="left")
#     last_descriptions = np.full(len(encounter_array), "", dtype=object)
#     distinct_counts = np.zeros(len(encounter_array), dtype=float)
#     for index, position in enumerate(positions):
#         if position:
#             prior_descriptions = descriptions[:position]
#             last_descriptions[index] = prior_descriptions[-1]
#             distinct_counts[index] = pd.unique(prior_descriptions).size
#     return last_descriptions, distinct_counts


def build_table(time_windows: list[int] | None = None) -> pd.DataFrame:
    patients = load_csv("patients.csv")
    patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"], errors="coerce")
    encounters = load_csv("encounters.csv")
    encounters["INDEX_DATE"] = pd.to_datetime(encounters["START"], utc=True)
    encounters = encounters.sort_values(["PATIENT", "INDEX_DATE"]).reset_index(drop=True)

    table = encounters[[
        "Id",
        "PATIENT",
        "INDEX_DATE",
        "ENCOUNTERCLASS",
        "CODE",
        "REASONCODE",
    ]].rename(columns={
        "Id": "encounter_id",
        "PATIENT": "patient_id",
        "CODE": "encounter_code",
        "REASONCODE": "reason_code",
    })
    table["encounter_code"] = table["encounter_code"].astype(str)
    table["reason_code"] = table["reason_code"].fillna("NO_REASON_RECORDED").astype(str)

    demographic_columns = [
        column for column in ["Id", "BIRTHDATE", "GENDER"]
        if column in patients.columns
    ]
    table = table.merge(
        patients[demographic_columns].rename(columns={"Id": "patient_id"}),
        on="patient_id", how="left",
    )
    table["age_at_encounter"] = (
        (table["INDEX_DATE"].dt.tz_localize(None) - table["BIRTHDATE"]).dt.days / 365.25
    ).clip(lower=0, upper=120)
    table = table.drop(columns="BIRTHDATE")

    patient_groups = table.groupby("patient_id", sort=False)
    table["patient_total_encounters"] = patient_groups.cumcount().astype(int)
    previous_encounter = patient_groups["INDEX_DATE"].shift(1)
    table["days_since_last_encounter"] = (
        (table["INDEX_DATE"] - previous_encounter).dt.days
        .fillna(-1)
        .astype(int)
    )

    history_sources = {
        "allergies.csv": ("START", "allergies"),
        "conditions.csv": ("START", "conditions"),
        "immunizations.csv": ("DATE", "immunizations"),
        "procedures.csv": ("DATE", "procedures"),
        "imaging_studies.csv": ("DATE", "imaging"),
        "medications.csv": ("START", "medications"),
        "observations.csv": ("DATE", "observations"),
    }
    time_windows = time_windows or TIME_WINDOWS

    for filename, (date_column, feature_base) in history_sources.items():
        source = load_csv(filename)
        source["EVENT_DATE"] = pd.to_datetime(source[date_column], utc=True, errors="coerce")
        source = source.dropna(subset=["PATIENT", "EVENT_DATE"])
        # has_description = "DESCRIPTION" in source.columns
        # if has_description:
        #     source["DESCRIPTION"] = source["DESCRIPTION"].fillna("").astype(str)

        grouped = source.groupby("PATIENT")
        for patient_id, group in grouped:
            patient_mask = table["patient_id"].eq(patient_id)
            if not patient_mask.any():
                continue
            encounter_dates = table.loc[patient_mask, "INDEX_DATE"]
            group = group.sort_values("EVENT_DATE")
            event_dates = group["EVENT_DATE"].to_numpy()

            for window_days in time_windows:
                table.loc[patient_mask, f"prior_{feature_base}_count_{window_days}d"] = _compute_window_counts(
                    event_dates,
                    encounter_dates,
                    window_days,
                )

            # if has_description:
            #     last_descriptions, distinct_counts = _compute_prior_description_features(
            #         event_dates,
            #         group["DESCRIPTION"].to_numpy(),
            #         encounter_dates,
            #     )
            #     table.loc[patient_mask, f"last_{feature_base}_description"] = last_descriptions
            #     table.loc[patient_mask, f"prior_{feature_base}_distinct_description_count"] = distinct_counts

            table.loc[patient_mask, f"days_since_{feature_base}_last"] = _compute_days_since_last(event_dates, encounter_dates)

    for column in [column for column in table.columns if column.startswith("prior_")]:
        table[column] = table[column].fillna(0).astype(float)
    for column in [column for column in table.columns if column.startswith("days_since_")]:
        table[column] = table[column].fillna(-1).astype(int)

    _add_prior_screening_measurements(table)

    workflow_events = make_workflow_events()
    for label, events in workflow_events.items():
        events = events.dropna(subset=["PATIENT", "EVENT_DATE"])
        by_patient = events.groupby("PATIENT")["EVENT_DATE"].apply(list).to_dict()
        table[label] = [
            int(any(index_date < event_date <= index_date + timedelta(days=HORIZON_DAYS)
                   for event_date in by_patient.get(patient_id, [])))
            for patient_id, index_date in zip(table["patient_id"], table["INDEX_DATE"])
        ]

    non_target_columns = [column for column in table.columns if column not in TARGET_COLUMNS]
    table = table[non_target_columns + TARGET_COLUMNS]
    return table


def _add_prior_screening_measurements(table: pd.DataFrame) -> None:
    observations = load_csv("observations.csv")
    observations["EVENT_DATE"] = pd.to_datetime(observations["DATE"], utc=True, errors="coerce")
    observations["DESCRIPTION"] = observations["DESCRIPTION"].fillna("").astype(str)
    observations["VALUE_NUMERIC"] = pd.to_numeric(observations["VALUE"], errors="coerce")
    observations = observations.dropna(subset=["PATIENT", "EVENT_DATE", "VALUE_NUMERIC"])

    measurement_terms = {
        "height": "body height",
        "weight": "body weight",
        "bmi": "body mass index",
        "bmi_percentile": "body mass index (bmi) [percentile] per age and gender",
        "systolic_bp": "systolic blood pressure",
        "diastolic_bp": "diastolic blood pressure",
        "heart_rate": "heart rate",
    }
    for feature_name, term in measurement_terms.items():
        source = observations[observations["DESCRIPTION"].str.lower().eq(term)]
        grouped = source.groupby("PATIENT")
        for patient_id, group in grouped:
            patient_mask = table["patient_id"].eq(patient_id)
            if not patient_mask.any():
                continue
            group = group.sort_values("EVENT_DATE")
            event_dates = group["EVENT_DATE"].to_numpy()
            encounter_dates = table.loc[patient_mask, "INDEX_DATE"]
            positions = np.searchsorted(
                pd.DatetimeIndex(event_dates).tz_localize(None).to_numpy("datetime64[ns]"),
                pd.DatetimeIndex(encounter_dates).tz_localize(None).to_numpy("datetime64[ns]"),
                side="left",
            )
            values = group["VALUE_NUMERIC"].to_numpy(dtype=float)
            latest = np.full(len(positions), np.nan)
            valid = positions > 0
            latest[valid] = values[positions[valid] - 1]
            table.loc[patient_mask, f"screening_prior_{feature_name}"] = latest

    screening_columns = [
        column for column in table.columns if column.startswith("screening_prior_")
    ]
    table["screening_prior_measurement_count"] = table[screening_columns].notna().sum(axis=1)


def main() -> None:
    result = build_table()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        OUTPUT.unlink()
    temp_path = OUTPUT.with_suffix(".tmp")
    result.to_csv(temp_path, index=False)
    temp_path.replace(OUTPUT)
    print(f"Wrote {len(result):,} rows and {len(result.columns):,} columns to {OUTPUT.name}")
    print(result[TARGET_COLUMNS].sum().to_string())


if __name__ == "__main__":
    main()