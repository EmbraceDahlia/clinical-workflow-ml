"""Ready-made synthetic applicants for the showcase app's manual-entry form."""

from __future__ import annotations

from datetime import date

EVALUATION_DATE = date(2026, 10, 1)

EXAMPLES: dict[str, dict[str, object]] = {
    "Yearly check-up is due": {
        "summary": "Four yearly examinations with lab tests; the last one was 350 days ago.",
        "expected": "Expected: both models suggest a medical examination and laboratory testing.",
        "date_of_birth": date(1960, 3, 1),
        "gender": "M",
        "visit": ("Encounter for symptom", "ambulatory"),
        "reason": "Viral sinusitis (disorder)",
        "prior_total_encounters": 4,
        "history": (
            "On 2022-09-29 the applicant had an annual general examination encounter with a blood glucose test "
            "and a cholesterol panel, received a flu vaccine, and was diagnosed with hypertension.\n"
            "On 2023-10-05 he had an annual general examination encounter with a blood glucose test "
            "and a cholesterol panel, received a flu vaccine, and was diagnosed with prediabetes.\n"
            "On 2024-10-10 he had an annual general examination encounter with a blood glucose test "
            "and a cholesterol panel, and received a flu vaccine.\n"
            "On 2025-10-16 he had an annual general examination encounter with a blood glucose test "
            "and a cholesterol panel, and received a flu vaccine."
        ),
    },
    "Early pregnancy": {
        "summary": "First prenatal visit four weeks ago with an ultrasound scan.",
        "expected": "Expected: the ensemble suggests imaging (just above its threshold); the original model suggests nothing.",
        "date_of_birth": date(1994, 5, 16),
        "gender": "F",
        "visit": ("Prenatal visit", "ambulatory"),
        "reason": "Normal pregnancy",
        "prior_total_encounters": 4,
        "history": (
            "On 2025-06-12 the applicant had an outpatient procedure visit with removal of a subcutaneous "
            "contraceptive and insertion of a subcutaneous contraceptive.\n"
            "On 2026-07-09 she had an outpatient procedure visit with removal of a subcutaneous contraceptive.\n"
            "On 2026-09-03 she had a prenatal initial visit with a diagnosis of normal pregnancy, a standard "
            "pregnancy test, an ultrasound scan for fetal viability, an evaluation of uterine fundal height "
            "and auscultation of the fetal heart.\n"
            "On 2026-09-17 she had a general examination encounter where body weight, body height and "
            "body mass index were recorded, and she received a flu vaccine."
        ),
    },
    "Little medical history": {
        "summary": "One visit for sinusitis three years ago and nothing since.",
        "expected": "Expected: neither model suggests a workflow.",
        "date_of_birth": date(1994, 2, 11),
        "gender": "M",
        "visit": ("Encounter for symptom", "ambulatory"),
        "reason": "Viral sinusitis (disorder)",
        "prior_total_encounters": 1,
        "history": (
            "On 2023-09-27 the applicant had a visit for symptoms and was diagnosed with viral sinusitis."
        ),
    },
}
