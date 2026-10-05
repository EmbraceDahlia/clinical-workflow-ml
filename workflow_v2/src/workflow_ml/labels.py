"""Workflow labels: was an event of each category recorded in the 30 days after the encounter?

The keyword rules are the ones used by the original pipeline (`src/build_model_table.py`
and `src/procedure_mapping.py`), so both versions predict the same target.
"""

import re

import pandas as pd

from workflow_ml.config import HORIZON_DAYS, LABELS

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

# First matching rule wins: echocardiography must stay an instrument test, not imaging.
PROCEDURE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("instrument_test", (
        "spirometry", "respiratory function", "echocardiography", "electrocardiogram",
        "electroencephalogram", "(eeg)", "cardioversion", "endoscopy", "bronchoscopy",
        "colonoscopy",
    )),
    ("imaging", (
        "computed tomography", "magnetic resonance", "x-ray", "mammograph", "mammogram",
        "ultrasound", "ultrasonography", "bone density scan", "fetal anatomy study",
    )),
    ("medical_exam", (
        "physical examination", "physical exam", "general examination",
        "medical examination", "brief general examination", "review of systems",
        "hearing examination", "digital examination", "pelvic examination",
    )),
    ("lab_test", (
        "hemoglobin", "hematocrit", "platelet", "laboratory", "blood test", "blood typing",
        "urine", "glucose", "cholesterol", "culture", "antigen", "antibody",
        "cytopathology", "smear", "biopsy", "pregnancy test", "sputum", "protein test",
        "gonorrhea", "syphilis", "rubella", "tuberculosis", "aneuploidy",
        "alpha-fetoprotein", "occult blood", "allergy screening",
    )),
)


def classify_procedure(description: object) -> str:
    text = "" if pd.isna(description) else str(description).strip().lower()
    for category, terms in PROCEDURE_RULES:
        if any(term in text for term in terms):
            return category
    return "other"


def _contains_any(text: pd.Series, terms: tuple[str, ...]) -> pd.Series:
    pattern = "|".join(re.escape(term) for term in terms)
    return text.fillna("").str.lower().str.contains(pattern, regex=True)


def workflow_events(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Distinct (PATIENT, DATE) pairs on which each workflow category was recorded."""
    procedures = raw["procedures"]
    category = procedures["DESCRIPTION"].map(
        {text: classify_procedure(text) for text in procedures["DESCRIPTION"].unique()}
    )
    observations = raw["observations"]
    encounters = raw["encounters"]

    sources = {
        "medical_exam": [
            encounters[_contains_any(encounters["DESCRIPTION"], EXAM_ENCOUNTER_TERMS)],
            procedures[category == "medical_exam"],
        ],
        "lab_test": [
            observations[_contains_any(observations["DESCRIPTION"], LAB_OBSERVATION_TERMS)],
            procedures[category == "lab_test"],
        ],
        "imaging": [raw["imaging_studies"], procedures[category == "imaging"]],
        "instrument_test": [procedures[category == "instrument_test"]],
    }
    return {
        label: pd.concat([frame[["PATIENT", "DATE"]] for frame in frames])
        .drop_duplicates()
        .sort_values("DATE", kind="stable")
        .reset_index(drop=True)
        for label, frames in sources.items()
    }


def build_labels(index: pd.DataFrame, events: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Label is 1 when an event falls in (t, t + HORIZON_DAYS]."""
    horizon = pd.Timedelta(days=HORIZON_DAYS)
    left = index[["patient_id", "t"]].reset_index().sort_values("t", kind="stable")
    labels = {}
    for label in LABELS:
        nxt = pd.merge_asof(
            left, events[label], left_on="t", right_on="DATE",
            left_by="patient_id", right_by="PATIENT",
            direction="forward", allow_exact_matches=False,
        ).set_index("index")["DATE"].reindex(index.index)
        labels[label] = ((nxt - index["t"]) <= horizon).astype("int8")
    return pd.DataFrame(labels, index=index.index)
