from __future__ import annotations

import pandas as pd


# First matching rule wins. Imaging is after specific instrument terms so
# echocardiography stays an instrument test rather than a generic ultrasound.
RULES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "instrument_test",
        (
            "spirometry",
            "respiratory function",
            "echocardiography",
            "electrocardiogram",
            "electroencephalogram",
            "(eeg)",
            "cardioversion",
            "endoscopy",
            "bronchoscopy",
            "colonoscopy",
        ),
        "instrument procedure keyword",
    ),
    (
        "imaging",
        (
            "computed tomography",
            "magnetic resonance",
            "x-ray",
            "mammograph",
            "mammogram",
            "ultrasound",
            "ultrasonography",
            "bone density scan",
            "fetal anatomy study",
        ),
        "imaging procedure keyword",
    ),
    (
        "medical_exam",
        (
            "physical examination",
            "physical exam",
            "general examination",
            "medical examination",
            "brief general examination",
            "review of systems",
            "hearing examination",
            "digital examination",
            "pelvic examination",
        ),
        "examination procedure keyword",
    ),
    (
        "lab_test",
        (
            "hemoglobin",
            "hematocrit",
            "platelet",
            "laboratory",
            "blood test",
            "blood typing",
            "urine",
            "glucose",
            "cholesterol",
            "culture",
            "antigen",
            "antibody",
            "cytopathology",
            "smear",
            "biopsy",
            "pregnancy test",
            "sputum",
            "protein test",
            "gonorrhea",
            "syphilis",
            "rubella",
            "tuberculosis",
            "aneuploidy",
            "alpha-fetoprotein",
            "occult blood",
            "allergy screening",
        ),
        "laboratory procedure keyword",
    ),
    # (
    #     "specialist_review",
    #     ("referral", "consultation", "specialist", "clinic"),
    #     "specialist procedure keyword",
    # ),
)


def classify_procedure(description: object) -> tuple[str, str, str]:
    text = "" if pd.isna(description) else str(description).strip().lower()
    for category, terms, rule in RULES:
        if any(term in text for term in terms):
            return category, "medium", rule
    return "other", "low", "no reviewed category rule matched"


def build_procedure_mapping(procedures: pd.DataFrame) -> pd.DataFrame:
    pairs = procedures[["CODE", "DESCRIPTION"]].drop_duplicates().copy()
    classifications = pairs["DESCRIPTION"].apply(classify_procedure)
    pairs[["CATEGORY", "CONFIDENCE", "MATCH_RULE"]] = pd.DataFrame(
        classifications.tolist(), index=pairs.index
    )
    pairs["REVIEWED"] = False
    return pairs.sort_values(["CATEGORY", "CODE", "DESCRIPTION"], na_position="last").reset_index(drop=True)
