"""A hand-entered history must become the same kind of records the model was trained on."""

from datetime import date

from workflow_ml.data import load_raw
from workflow_ml.intake import CURRENT_ENCOUNTER_ID, application_to_raw, build_catalog

PROFILE = {"GENDER": "M", "date_of_birth": date(1960, 3, 1), "ENCOUNTERCLASS": "ambulatory",
           "encounter_code": 185345009, "reason_code": "444814009"}


def _record(kind: str, day: date, details: str) -> dict[str, object]:
    return {"Record type": kind, "Record date": day, "Details or code": details}


def test_examination_filed_as_procedure_still_makes_a_wellness_visit():
    catalog = build_catalog(load_raw())
    records = [
        _record("Procedure", date(2025, 10, 25), "annual general examination"),
        _record("Observation", date(2025, 10, 25), "blood glucose test"),
        _record("Condition", date(2026, 12, 1), "dated after the evaluation, must be ignored"),
    ]
    raw = application_to_raw(PROFILE, records, date(2026, 10, 10), catalog)

    visits = raw["encounters"].set_index("Id")
    assert list(visits.index) == [CURRENT_ENCOUNTER_ID, "visit-20251025"]
    assert visits.loc["visit-20251025", "ENCOUNTERCLASS"] == "wellness"
    assert raw["observations"]["DESCRIPTION"].tolist() == ["Glucose"]
    assert raw["conditions"].empty
