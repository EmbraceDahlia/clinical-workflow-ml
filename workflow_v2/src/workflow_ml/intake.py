"""Turn a hand-entered application (profile + dated history records) into model input.

The model was trained on Synthea records, so an application is rewritten as a one-patient
Synthea-style record set and passed through the same feature builder as the training data.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import get_close_matches

import pandas as pd

from workflow_ml.features import Vocabulary, build_features

PATIENT_ID = "applicant"
CURRENT_ENCOUNTER_ID = "current-evaluation"
# History record types the model can use (allergies are not model inputs).
RECORD_SOURCES = {
    "Condition": "conditions",
    "Procedure": "procedures",
    "Medication": "medications",
    "Observation": "observations",
    "Immunization": "immunizations",
    "Imaging study": "imaging_studies",
}
EVENT_COLUMNS = ["PATIENT", "ENCOUNTER", "DATE", "DESCRIPTION"]
DEFAULT_CLASS = "ambulatory"


@dataclass(frozen=True)
class Catalog:
    """Names the model saw in training, used to map free-text details onto its vocabulary."""

    descriptions: dict[str, tuple[str, ...]]          # record source -> known descriptions
    encounter_by_code: dict[str, tuple[str, str]]     # encounter code -> (description, class)
    class_of_encounter: dict[str, str]                # encounter description -> usual class
    usual_encounter: dict[str, str]                   # class -> its most common description


def build_catalog(raw: dict[str, pd.DataFrame]) -> Catalog:
    encounters = raw["encounters"]

    def most_common(values: pd.Series) -> str:
        return values.value_counts().index[0]

    by_code = encounters.groupby("CODE")[["DESCRIPTION", "ENCOUNTERCLASS"]].agg(most_common)
    sources = [source for source in RECORD_SOURCES.values() if "DESCRIPTION" in raw[source]]
    return Catalog(
        descriptions={
            "encounters": tuple(encounters["DESCRIPTION"].unique()),
            **{source: tuple(raw[source]["DESCRIPTION"].dropna().unique()) for source in sources},
        },
        encounter_by_code={str(code): tuple(row) for code, row in by_code.iterrows()},
        class_of_encounter=encounters.groupby("DESCRIPTION")["ENCOUNTERCLASS"].agg(most_common).to_dict(),
        usual_encounter=encounters.groupby("ENCOUNTERCLASS")["DESCRIPTION"].agg(most_common).to_dict(),
    )


def _known_name(text: object, known: Sequence[str]) -> str:
    """The training-data description closest to `text`, or `text` itself when nothing is close."""
    text = str(text or "").strip()
    by_lower = {name.lower(): name for name in known}
    match = get_close_matches(text.lower(), list(by_lower), n=1, cutoff=0.6)
    return by_lower[match[0]] if match else text


def _events(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    frame["DATE"] = pd.to_datetime(frame["DATE"], utc=True).dt.as_unit("ns")
    for column in ("PATIENT", "ENCOUNTER", "DESCRIPTION"):
        frame[column] = frame[column].astype("str")
    return frame


def application_to_raw(
    profile: Mapping[str, object],
    history_records: Sequence[Mapping[str, object]],
    evaluation_date: object,
    catalog: Catalog,
) -> dict[str, pd.DataFrame]:
    """One-patient record set: the dated history plus the evaluation as the latest encounter.

    A dated record without an encounter of its own is attached to an implied visit on
    that day, because in the training data every record belongs to an encounter.
    """
    evaluation = pd.Timestamp(evaluation_date).normalize()
    history = pd.DataFrame(list(history_records), columns=["Record type", "Record date", "Details or code"])
    history["date"] = pd.to_datetime(history["Record date"], errors="coerce").dt.normalize()
    history = history[history["date"] < evaluation]

    code = str(profile.get("encounter_code", ""))
    description, usual_class = catalog.encounter_by_code.get(
        code, (catalog.usual_encounter[DEFAULT_CLASS], DEFAULT_CLASS)
    )
    encounters = [
        {
            "Id": CURRENT_ENCOUNTER_ID, "PATIENT": PATIENT_ID, "DATE": evaluation,
            "ENCOUNTERCLASS": str(profile.get("ENCOUNTERCLASS") or usual_class),
            "DESCRIPTION": description,
            "REASONCODE": pd.to_numeric(profile.get("reason_code"), errors="coerce"),
        }
    ]
    visit_ids: dict[pd.Timestamp, str] = {}
    for day, records in history.groupby("date"):
        visit_ids[day] = f"visit-{day:%Y%m%d}"
        stated = records.loc[records["Record type"] == "Encounter", "Details or code"]
        name = (
            _known_name(stated.iloc[0], catalog.descriptions["encounters"])
            if len(stated) else catalog.usual_encounter[DEFAULT_CLASS]
        )
        encounters.append(
            {
                "Id": visit_ids[day], "PATIENT": PATIENT_ID, "DATE": day,
                "ENCOUNTERCLASS": catalog.class_of_encounter.get(name, DEFAULT_CLASS),
                "DESCRIPTION": name, "REASONCODE": float("nan"),
            }
        )

    raw = {
        "patients": pd.DataFrame(
            {
                "Id": [PATIENT_ID],
                "BIRTHDATE": pd.to_datetime([profile["date_of_birth"]], utc=True).as_unit("ns"),
                "GENDER": [profile.get("GENDER", "F")],
            }
        ),
        "encounters": pd.DataFrame(encounters),
        "careplans": _events([]),
    }
    raw["encounters"]["DATE"] = pd.to_datetime(raw["encounters"]["DATE"], utc=True).dt.as_unit("ns")

    for record_type, source in RECORD_SOURCES.items():
        records = history[history["Record type"] == record_type]
        known = catalog.descriptions.get(source, ())
        raw[source] = _events(
            [
                {
                    "PATIENT": PATIENT_ID, "ENCOUNTER": visit_ids[row["date"]], "DATE": row["date"],
                    "DESCRIPTION": _known_name(row["Details or code"], known),
                }
                for _, row in records.iterrows()
            ]
        )
    raw["conditions"]["END"] = pd.Series(pd.NaT, index=raw["conditions"].index, dtype="datetime64[ns, UTC]")
    return raw


def application_table(
    profile: Mapping[str, object],
    history_records: Sequence[Mapping[str, object]],
    evaluation_date: object,
    catalog: Catalog,
    vocabulary: Vocabulary,
) -> tuple[pd.DataFrame, int]:
    """Feature table for the applicant and the row number of the evaluation encounter."""
    raw = application_to_raw(profile, history_records, evaluation_date, catalog)
    table = build_features(raw, vocabulary)
    return table, int((table["encounter_id"] == CURRENT_ENCOUNTER_ID).to_numpy().argmax())
