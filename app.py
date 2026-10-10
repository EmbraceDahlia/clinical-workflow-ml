from __future__ import annotations

import os
import re
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, MODEL_ARTIFACT_PATH, MODEL_TABLE_PATH, TARGET_COLUMNS
from src.demo_examples import EVALUATION_DATE, EXAMPLES
from src.history_agent import extract_history_records, history_model_name
from src.longitudinal_agent import extract_longitudinal_features
from src.model_artifact import load_model_artifact, predict_workflows

try:
    from workflow_ml.config import LABELS
    from workflow_ml.features import load_table
    from workflow_ml.serving import BUNDLE_PATH, evidence, load_bundle
except ImportError:  # without the workflow_v2 package the app still runs on the original model
    BUNDLE_PATH = None

ENSEMBLE = "Ensemble v2 · mean F1 0.789"
BASELINE = "Original Random Forest · mean F1 0.536"
SHORT_NAME = {ENSEMBLE: "Ensemble v2", BASELINE: "Original"}
TARGETS = list(TARGET_COLUMNS)


st.set_page_config(
    page_title="Clinical Workflow Desk",
    page_icon=":material/clinical_notes:",
    layout="wide",
)

WORKFLOW_INFO = {
    "medical_exam": {
        "title": "Medical examination",
        "request": "Clinical examination review",
        "reviewer": "Primary care physician",
    },
    "lab_test": {
        "title": "Laboratory testing",
        "request": "Laboratory workflow review",
        "reviewer": "Laboratory reviewer",
    },
    "imaging": {
        "title": "Imaging",
        "request": "Imaging workflow review",
        "reviewer": "Radiology reviewer",
    },
    "instrument_test": {
        "title": "Instrument-based testing",
        "request": "Instrument-based testing review",
        "reviewer": "Specialist reviewer",
    },
}
REVIEWER_ROLES = [
    "Primary care physician",
    "Specialist reviewer",
    "Laboratory reviewer",
    "Radiology reviewer",
    "Other clinician",
]
TASK_STATUSES = [
    "Awaiting clinician",
    "In review",
    "Review complete",
    "Not indicated",
]


@st.cache_resource
def get_model() -> dict[str, object]:
    return load_model_artifact(MODEL_ARTIFACT_PATH)


@st.cache_resource
def get_ensemble():
    return load_bundle(BUNDLE_PATH)


def _with_suggestion_flag(demo: pd.DataFrame) -> pd.DataFrame:
    demo["suggested"] = demo[[f"{target}_predicted" for target in TARGETS]].any(axis=1)
    return demo.set_index("encounter_id")


@st.cache_resource(show_spinner="Scoring the held-out synthetic patients with the ensemble...")
def ensemble_demo() -> pd.DataFrame:
    """Held-out patients' encounters with features, live ensemble predictions and outcomes."""
    bundle = get_ensemble()
    table = load_table()
    rows = table.index[table["patient_id"].isin(bundle.demo_patients)].to_numpy()
    demo = pd.concat([table.iloc[rows], bundle.predict(table, rows)], axis=1)
    encounter_types = bundle.vocabulary.categories["enc_type"]
    demo["encounter_type"] = [encounter_types[code] for code in demo["enc_type"]]
    return _with_suggestion_flag(demo)


@st.cache_resource(show_spinner="Scoring the held-out synthetic patients with the original model...")
def baseline_demo() -> pd.DataFrame:
    """The same view for the original model, on the patients it kept out of training."""
    artifact = get_model()
    table = pd.read_csv(MODEL_TABLE_PATH)
    table = table[table["patient_id"].isin(artifact.get("holdout_patients", []))].reset_index(drop=True)
    demo = pd.concat([table, predict_workflows(table, artifact)], axis=1)
    descriptions = pd.read_csv(DATA_DIR / "encounters.csv", usecols=["Id", "DESCRIPTION"])
    demo["encounter_type"] = demo["encounter_id"].map(descriptions.set_index("Id")["DESCRIPTION"])
    demo["t"] = pd.to_datetime(demo["INDEX_DATE"], utc=True)
    demo = demo.rename(columns={"age_at_encounter": "age", "patient_total_encounters": "n_previous"})
    return _with_suggestion_flag(demo.sort_values(["patient_id", "t"]))


def demo_encounters(model_name: str) -> pd.DataFrame | None:
    """Held-out encounters scored by `model_name`, or None when that is not possible here."""
    if model_name not in available_models():
        return None
    if model_name == ENSEMBLE:
        return ensemble_demo()
    if not MODEL_TABLE_PATH.exists() or not get_model().get("holdout_patients"):
        return None
    return baseline_demo()


def demo_predictions(model_name: str, encounter: pd.Series) -> dict[str, dict[str, object]]:
    if model_name == ENSEMBLE:
        return ensemble_predictions(encounter, encounter, get_ensemble())
    artifact = get_model()
    return {
        target: {
            "score": float(encounter[f"{target}_score"]),
            "threshold": float(artifact["thresholds"][target]),
            "proposed": bool(encounter[f"{target}_predicted"]),
            "signals": global_signals(artifact, target),
        }
        for target in TARGETS
    }


def available_models() -> list[str]:
    models = []
    if BUNDLE_PATH is not None and BUNDLE_PATH.exists():
        models.append(ENSEMBLE)
    if MODEL_ARTIFACT_PATH.exists():
        models.append(BASELINE)
    return models



def print_model_inputs(model_name: str, inputs: dict[str, object]) -> None:
    width = max(len(str(name)) for name in inputs)
    print(f"\n=== Parameters passed to {model_name} ({len(inputs)}) ===", flush=True)
    for name, value in inputs.items():
        print(f"{str(name):<{width}} : {value}", flush=True)
    print(flush=True)


def safe_extraction_error(error: Exception) -> str:
    message = str(error).replace("\r", " ").replace("\n", " ").strip()
    message = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[redacted]", message, flags=re.IGNORECASE)
    message = re.sub(
        r"(?i)(bearer\s+)\S+",
        r"\1[redacted]",
        message,
    )
    if len(message) > 400:
        message = message[:397] + "..."
    return f"{type(error).__name__}: {message or 'No error detail was provided.'}"


def get_top_signals(artifact: dict[str, object], target: str) -> list[tuple[str, float]]:
    model = artifact["model"]
    preprocessor = artifact["preprocessor"]
    target_index = artifact["target_columns"].index(target)
    estimator = model.estimators_[target_index]
    feature_names = preprocessor.get_feature_names_out()
    importances = estimator.feature_importances_
    top_indices = importances.argsort()[::-1][:3]
    return [
        (
            str(feature_names[index]).replace("numeric__", "").replace("categorical__", ""),
            float(importances[index]),
        )
        for index in top_indices
        if importances[index] > 0
    ]


BLANK_FORM = "Blank form"
NO_REASON = "No reason recorded"
FORM_DEFAULTS = {
    "reference": "",
    "date_of_birth": date(1985, 1, 1),
    "gender": "F",
    "evaluation_date": date.today(),
    "visit_type": "General examination of patient (procedure) · wellness",
    "visit_reason": NO_REASON,
    "prior_total_encounters": 0,
    "medical_history_text": "",
}


@st.cache_data
def visit_options() -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    """Visit types and reasons found in the training data, most common first."""
    encounters = pd.read_csv(DATA_DIR / "encounters.csv")
    by_type = encounters.groupby(["DESCRIPTION", "ENCOUNTERCLASS"])["CODE"].agg(["size", "first"])
    visits = {
        f"{description} · {encounter_class}": {
            "encounter_code": int(row["first"]),
            "encounter_description": description,
            "ENCOUNTERCLASS": encounter_class,
        }
        for (description, encounter_class), row in by_type.sort_values("size", ascending=False).iterrows()
    }
    by_reason = encounters.groupby("REASONDESCRIPTION")["REASONCODE"].agg(["size", "first"])
    # The original model stores reason codes the way pandas printed them, e.g. "72892002.0".
    reasons = {NO_REASON: "NO_REASON_RECORDED"} | {
        description: str(float(row["first"]))
        for description, row in by_reason.sort_values("size", ascending=False).iterrows()
    }
    return visits, reasons


def apply_example() -> None:
    """Fill the form from the chosen example, or clear it."""
    example = EXAMPLES.get(st.session_state["example"])
    if example is None:
        st.session_state.update(FORM_DEFAULTS)
        return
    description, encounter_class = example["visit"]
    st.session_state.update(
        {
            "reference": f"DEMO-{list(EXAMPLES).index(st.session_state['example']) + 1:03d}",
            "date_of_birth": example["date_of_birth"],
            "gender": example["gender"],
            "evaluation_date": EVALUATION_DATE,
            "visit_type": f"{description} · {encounter_class}",
            "visit_reason": example["reason"],
            "prior_total_encounters": example["prior_total_encounters"],
            "medical_history_text": example["history"],
        }
    )


def render_manual_inputs() -> tuple[dict[str, object], str]:
    visits, reasons = visit_options()
    st.subheader("Applicant and evaluation")
    first_row = st.columns([1.4, 1])
    applicant_columns = first_row[0].columns(2)
    with applicant_columns[0]:
        date_of_birth = st.date_input(
            "Date of birth",
            key="date_of_birth",
            min_value=date(1900, 1, 1),
            max_value=date.today(),
        )
        gender = st.selectbox(
            "Gender",
            ["F", "M", "Other"],
            key="gender",
            format_func=lambda value: {"F": "Female", "M": "Male", "Other": "Other / not recorded"}[value],
        )
        prior_total_encounters = st.number_input(
            "Total prior encounters",
            key="prior_total_encounters",
            min_value=0,
            max_value=100_000,
            step=1,
            help="Lifetime encounter count; the original model relies on it heavily. The larger of this and the extracted encounters is used.",
        )
    with applicant_columns[1]:
        evaluation_date = st.date_input("Evaluation date", key="evaluation_date")
        visit_type = st.selectbox(
            "Visit type",
            list(visits),
            key="visit_type",
            help="The kind of encounter being evaluated. Type to search.",
        )
        visit_reason = st.selectbox(
            "Reason for visit",
            list(reasons),
            key="visit_reason",
            help="Type to search.",
        )
    with first_row[1]:
        st.text_area(
            "Medical history (free text)",
            key="medical_history_text",
            max_chars=12000,
            height=220,
            placeholder="Enter a synthetic medical history. The dated records are extracted automatically.",
            help=f"Extraction model: {history_model_name()}. Text is sent only when you generate suggestions.",
        )

    if date_of_birth > evaluation_date:
        st.error("Date of birth must be on or before the evaluation date.")
    age_at_evaluation = max(
        0.0,
        min(120.0, (evaluation_date - date_of_birth).days / 365.25),
    )
    current_features = {
        "GENDER": gender,
        "age_at_encounter": age_at_evaluation,
        **visits[visit_type],
        "reason_code": reasons[visit_reason],
        "date_of_birth": date_of_birth,
        "prior_total_encounters": prior_total_encounters,
    }

    return (
        current_features | {"evaluation_date": evaluation_date},
        st.session_state.get("medical_history_text", ""),
    )


def global_signals(artifact: dict[str, object], target: str) -> list[str]:
    return [
        f"{name.replace('_', ' ')} ({importance:.0%})"
        for name, importance in get_top_signals(artifact, target)
    ]


def baseline_predictions(
    features: dict[str, object], artifact: dict[str, object]
) -> dict[str, dict[str, object]]:
    prediction = predict_workflows(features, artifact).iloc[0]
    return {
        target: {
            "score": float(prediction[f"{target}_score"]),
            "threshold": float(artifact["thresholds"][target]),
            "proposed": bool(prediction[f"{target}_predicted"]),
            "signals": global_signals(artifact, target),
        }
        for target in artifact["target_columns"]
    }


def ensemble_predictions(
    prediction: pd.Series, features: pd.Series, bundle
) -> dict[str, dict[str, object]]:
    return {
        label: {
            "score": float(prediction[f"{label}_score"]),
            "threshold": bundle.thresholds[label],
            "proposed": bool(prediction[f"{label}_predicted"]),
            "signals": evidence(features, label),
        }
        for label in LABELS
    }


def make_case(
    reference: str,
    model_name: str,
    predictions: dict[str, dict[str, object]],
    features: dict[str, object],
    history_records: list[dict[str, object]],
    actual: dict[str, bool] | None = None,
    comparison: dict[str, dict[str, dict[str, object]]] | None = None,
) -> dict[str, object]:
    return {
        "id": uuid.uuid4().hex,
        "reference": reference,
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "model": model_name,
        "features": features,
        "history_records": history_records,
        "predictions": predictions,
        "actual": actual,
        "comparison": comparison or {},
        "status": "Awaiting underwriter review",
        "decision_note": "",
    }


def cases() -> list[dict[str, object]]:
    return st.session_state.setdefault("applications", [])


def tasks() -> list[dict[str, object]]:
    return st.session_state.setdefault("workflow_tasks", [])


def render_overview() -> None:
    applications = cases()
    all_tasks = tasks()
    pending_review = sum(case["status"] == "Awaiting underwriter review" for case in applications)
    active_tasks = sum(task["status"] in TASK_STATUSES[:2] for task in all_tasks)

    st.title("Workflow overview")
    st.caption("A review queue for synthetic policy applications and clinician referrals.")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Applications", len(applications))
    metric_columns[1].metric("Awaiting underwriting", pending_review)
    metric_columns[2].metric("Clinical review requests", len(all_tasks))
    metric_columns[3].metric("Open requests", active_tasks)

    if not applications:
        st.info("No applications in this session yet. Start with New application.")
        return

    st.subheader("Recent applications")
    for case in applications[:8]:
        proposed = [
            WORKFLOW_INFO[label]["title"]
            for label, result in case["predictions"].items()
            if result["proposed"]
        ]
        with st.container(border=True):
            left, middle, right = st.columns([2, 3, 2])
            left.markdown(f"**{case['reference']}**")
            left.caption(case["created"])
            middle.write(", ".join(proposed) if proposed else "No categories passed the model thresholds")
            right.caption(case["status"])


def render_demo_intake(model_name: str) -> None:
    """Pick a synthetic patient the selected model never trained on and score one encounter."""
    demo = demo_encounters(model_name)
    if demo is None:
        st.info(
            "The original model has no record of which patients it kept out of training. "
            "Run `python src/build_model_table.py` and `python src/export_model.py` again to enable this tab."
        )
        return
    other_name = next((name for name in available_models() if name != model_name), None)
    other = demo_encounters(other_name) if other_name else None

    st.caption(
        f"{demo['patient_id'].nunique()} synthetic patients were held out of training. "
        "The model scores them live, and the review page shows what was actually recorded afterwards."
    )
    visits = demo.groupby("patient_id").size()
    patient = st.selectbox(
        "Synthetic patient",
        visits.index,
        format_func=lambda patient_id: f"Patient {patient_id[:8]} · {visits[patient_id]} encounters",
    )
    encounters = demo[demo["patient_id"] == patient]
    if st.toggle("Only encounters where the model suggests a workflow", value=True):
        encounters = encounters[encounters["suggested"]]
    if encounters.empty:
        st.info("The model suggests no workflow for this patient. Turn the filter off or pick another patient.")
        return

    encounter_id = st.selectbox(
        "Evaluation encounter",
        encounters.index,
        index=len(encounters) - 1,
        format_func=lambda row: f"{demo.at[row, 't']:%Y-%m-%d} · {demo.at[row, 'encounter_type']}",
    )
    encounter = demo.loc[encounter_id]
    earlier = demo[(demo["patient_id"] == patient) & (demo["t"] < encounter["t"])].tail(6)
    left, right = st.columns([1, 2])
    left.metric("Age at evaluation", f"{encounter['age']:.0f}")
    left.metric("Earlier encounters", f"{encounter['n_previous']:.0f}")
    right.markdown("**Most recent earlier encounters**")
    right.dataframe(
        pd.DataFrame({"Date": earlier["t"].dt.strftime("%Y-%m-%d"), "Encounter": earlier["encounter_type"]}),
        hide_index=True,
        width="stretch",
    )

    if st.button("Generate workflow suggestions", type="primary", width="stretch", key="demo_generate"):
        comparison = {}
        if other is not None and encounter_id in other.index:
            comparison[other_name] = demo_predictions(other_name, other.loc[encounter_id])
        case = make_case(
            f"DEMO-{patient[:6]}-{encounter['t']:%Y%m%d}",
            model_name,
            demo_predictions(model_name, encounter),
            {
                "patient": patient[:8],
                "evaluation date": f"{encounter['t']:%Y-%m-%d}",
                "encounter": encounter["encounter_type"],
                "age": f"{encounter['age']:.0f}",
                "earlier encounters": f"{encounter['n_previous']:.0f}",
            },
            [],
            actual={target: bool(encounter[target]) for target in TARGETS},
            comparison=comparison,
        )
        cases().insert(0, case)
        st.success(f"Application {case['reference']} is ready for underwriter review.")


def render_intake(model_name: str) -> None:
    st.title("New application")
    st.caption(f"Model: {model_name}")
    st.warning("Showcase only: use synthetic data. Do not enter real policyholder identifiers or health information.")

    demo_tab, manual_tab = st.tabs(["Synthetic patient", "Manual entry"])
    with demo_tab:
        render_demo_intake(model_name)
    with manual_tab:
        if model_name == ENSEMBLE:
            st.caption(
                "Hand-entered histories are much shorter than the records the model was trained on, "
                "so treat these scores as illustrative."
            )
        render_manual_intake(model_name)


def render_manual_intake(model_name: str) -> None:
    for name, value in FORM_DEFAULTS.items():
        st.session_state.setdefault(name, value)
    example = st.selectbox(
        "Start from an example",
        [BLANK_FORM, *EXAMPLES],
        key="example",
        on_change=apply_example,
        help="Fills every field with a synthetic applicant. You can still edit anything before generating.",
    )
    if example in EXAMPLES:
        st.caption(f"{EXAMPLES[example]['summary']} {EXAMPLES[example]['expected']}")

    with st.form("application_intake"):
        reference = st.text_input("Application reference", key="reference", placeholder="DEMO-1001")
        current_features, history_text = render_manual_inputs()
        submitted = st.form_submit_button(
            "Generate workflow suggestions",
            type="primary",
            width="stretch",
        )

    if submitted:
        if not reference.strip():
            st.error("Enter a demo application reference.")
        elif current_features["date_of_birth"] > current_features["evaluation_date"]:
            st.error("Date of birth must be on or before the evaluation date.")
        elif current_features["evaluation_date"] > date.today():
            st.error("Evaluation date cannot be in the future.")
        else:
            evaluation_date = current_features.pop("evaluation_date")
            profile = dict(current_features)
            current_features.pop("date_of_birth")
            prior_total_encounters = int(current_features.pop("prior_total_encounters"))
            step = "Starting"
            with st.status("Generating workflow suggestions...", expanded=True) as status:
                try:
                    history_records = []
                    longitudinal = {}
                    if history_text.strip():
                        step = "Step 1/4: Checking OpenAI configuration"
                        st.write(step)
                        if not os.environ.get("OPENAI_API_KEY"):
                            raise ValueError("OPENAI_API_KEY is not set in your environment.")

                        step = "Step 2/4: Extracting history with the OpenAI agent"
                        st.write(f"{step} ({history_model_name()})")
                        if model_name == ENSEMBLE:
                            extracted = extract_history_records(
                                history_text,
                                model=history_model_name(),
                                evaluation_date=evaluation_date,
                            )["records"]
                            history_records = [r for r in extracted if r["Record date"] is not None]
                            st.write(
                                f"Extracted {len(extracted)} records; {len(history_records)} have usable dates."
                            )
                        else:
                            extraction = extract_longitudinal_features(
                                history_text,
                                evaluation_date,
                                model=history_model_name(),
                            )
                            longitudinal = extraction["features"]
                            st.write(
                                f"Extracted {len(extraction['events'])} events and {len(longitudinal)} longitudinal fields."
                            )
                            if extraction["mismatches"]:
                                st.write(
                                    f"Corrected {len(extraction['mismatches'])} agent arithmetic errors: "
                                    + ", ".join(extraction["mismatches"])
                                )
                    else:
                        st.write("No medical history text entered; skipping extraction.")

                    step = "Step 3/4: Computing model features from the dated history"
                    st.write(step)
                    if model_name == ENSEMBLE:
                        bundle = get_ensemble()
                        prediction, model_features = bundle.predict_application(
                            profile, history_records, evaluation_date
                        )
                        step = "Step 4/4: Running the prediction model"
                        st.write(step)
                        print_model_inputs(model_name, model_features.to_dict())
                        predictions = ensemble_predictions(prediction, model_features, bundle)
                        feature_values = {
                            **{name: str(value) for name, value in profile.items()},
                            "dated history records": len(history_records),
                            "earlier encounters derived": f"{model_features['n_previous']:.0f}",
                        }
                    else:
                        artifact = get_model()
                        merged = current_features | longitudinal
                        feature_values = {
                            column: merged.get(column, -1 if column.startswith("days_since") else 0)
                            for column in artifact["feature_columns"]
                        }
                        feature_values["patient_total_encounters"] = max(
                            int(feature_values["patient_total_encounters"]), prior_total_encounters
                        )
                        step = "Step 4/4: Running the prediction model"
                        st.write(step)
                        print_model_inputs(model_name, feature_values)
                        predictions = baseline_predictions(feature_values, artifact)

                    case = make_case(
                        reference.strip(),
                        model_name,
                        predictions,
                        feature_values,
                        history_records,
                    )
                except Exception as error:
                    status.update(label=f"Failed at: {step}", state="error", expanded=True)
                    st.error(safe_extraction_error(error))
                else:
                    status.update(label="Workflow suggestions ready", state="complete", expanded=False)
                    cases().insert(0, case)
                    st.success(f"Application {case['reference']} is ready for underwriter review.")


def _outcome(suggested: bool, happened: bool) -> str:
    if suggested == happened:
        return "Correct"
    return "Missed" if happened else "False alarm"


def render_scores(case: dict[str, object]) -> None:
    """All four categories at a glance; synthetic patients also show the recorded outcome
    and, when both models held the patient out, the other model's answer."""
    actual = case.get("actual")
    answers = {case["model"]: case["predictions"], **case.get("comparison", {})}
    several = len(answers) > 1
    rows = []
    for label in case["predictions"]:
        row = {"Workflow": WORKFLOW_INFO[label]["title"]}
        for model_name, predictions in answers.items():
            prefix = f"{SHORT_NAME[model_name]}: " if several else ""
            result = predictions[label]
            row[f"{prefix}score / threshold"] = f"{result['score']:.2f} / {result['threshold']:.2f}"
            row[f"{prefix}suggests"] = "Yes" if result["proposed"] else "No"
        if actual is not None:
            row["Recorded within 30 days"] = "Yes" if actual[label] else "No"
            for model_name, predictions in answers.items():
                prefix = f"{SHORT_NAME[model_name]}: " if several else ""
                row[f"{prefix}outcome"] = _outcome(predictions[label]["proposed"], actual[label])
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    if actual is not None:
        totals = [
            f"{SHORT_NAME[model_name]} {sum(predictions[label]['proposed'] == actual[label] for label in actual)} of {len(actual)}"
            for model_name, predictions in answers.items()
        ]
        st.caption("Categories matching what was recorded for this synthetic patient: " + ", ".join(totals) + ".")


def render_review() -> None:
    st.title("Underwriter review")
    pending = [case for case in cases() if case["status"] == "Awaiting underwriter review"]
    if not pending:
        st.info("There are no applications awaiting review.")
        return

    pending_by_id = {case["id"]: case for case in pending}
    selected_case_id = st.selectbox(
        "Application",
        list(pending_by_id),
        format_func=lambda case_id: (
            f"{pending_by_id[case_id]['reference']} · {pending_by_id[case_id]['created']}"
        ),
        key="review_case",
    )
    case = pending_by_id[selected_case_id]
    st.subheader(case["reference"])
    st.caption(f"Model: {case['model']}")
    st.caption("Model outputs are workflow-category signals derived from synthetic future-event labels, not procedure orders.")
    render_scores(case)

    proposed_labels = [
        label for label, result in case["predictions"].items() if result["proposed"]
    ]
    if not proposed_labels:
        st.info("No workflow category exceeded its validation-tuned threshold.")
        if st.button("Close without referral", type="secondary"):
            case["status"] = "Closed without referral"
            st.rerun()
        return

    for label in proposed_labels:
        result = case["predictions"][label]
        title = WORKFLOW_INFO[label]["title"]
        with st.container(border=True):
            title_column, score_column = st.columns([3, 1])
            title_column.markdown(f"**{title}**")
            score_column.metric("Model score", f"{result['score']:.3f}", help="A model score, not a calibrated probability.")
            st.caption(f"Threshold: {result['threshold']:.3f}. The category is surfaced because its score met or exceeded this threshold.")
            signal_text = ", ".join(result["signals"]) or "No signal available"
            if case["model"] == ENSEMBLE:
                st.write(f"**In this patient's history:** {signal_text}")
                st.caption("These are feature values for this case, not evidence of medical necessity.")
            else:
                st.write(f"**Most influential model features overall:** {signal_text}")
                st.caption("These are global training-set signals, not a case-specific explanation or evidence of medical necessity.")

    with st.expander("Submitted model inputs"):
        st.dataframe(
            pd.DataFrame(
                [
                    {"Input": name.replace("_", " "), "Value": str(value)}
                    for name, value in case["features"].items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )

    with st.expander("Extracted history records (audit)"):
        if case["history_records"]:
            st.dataframe(
                pd.DataFrame(case["history_records"]).astype(str),
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("No history records were extracted.")

    with st.form(f"decision_{case['id']}"):
        decision = st.radio("Decision", ["Approve selected referrals", "Decline referrals"], horizontal=True)
        selected = st.multiselect(
            "Referrals to send for clinician review",
            proposed_labels,
            default=proposed_labels,
            format_func=lambda label: WORKFLOW_INFO[label]["request"],
        )
        note = st.text_area("Underwriter note", placeholder="Record the review decision for this demo case.")
        decided = st.form_submit_button("Record decision", type="primary")

    if decided:
        if decision == "Approve selected referrals" and not selected:
            st.error("Select at least one referral, or choose Decline referrals.")
            return
        if decision == "Decline referrals" and not note.strip():
            st.error("Add a short note when declining referrals.")
            return

        case["decision_note"] = note.strip()
        if decision == "Decline referrals":
            case["status"] = "Referrals declined"
        else:
            case["status"] = "Approved for clinician review"
            for label in selected:
                tasks().append(
                    {
                        "id": uuid.uuid4().hex,
                        "case_id": case["id"],
                        "reference": case["reference"],
                        "workflow": WORKFLOW_INFO[label]["request"],
                        "reviewer": WORKFLOW_INFO[label]["reviewer"],
                        "status": "Awaiting clinician",
                        "note": "",
                    }
                )
        st.rerun()


def render_workflow_queue() -> None:
    st.title("Clinician workflow queue")
    st.caption("These are review requests. A qualified clinician determines whether any procedure is appropriate.")
    all_tasks = tasks()
    if not all_tasks:
        st.info("No clinician review requests have been created yet.")
        return

    for task in all_tasks:
        with st.expander(f"{task['reference']} · {task['workflow']} · {task['status']}"):
            st.write(f"**Assigned reviewer:** {task['reviewer']}")
            with st.form(f"task_{task['id']}"):
                role_index = REVIEWER_ROLES.index(task["reviewer"]) if task["reviewer"] in REVIEWER_ROLES else 0
                reviewer = st.selectbox("Reviewer role", REVIEWER_ROLES, index=role_index)
                status = st.selectbox("Review status", TASK_STATUSES, index=TASK_STATUSES.index(task["status"]))
                note = st.text_area("Clinician review note", value=task["note"])
                updated = st.form_submit_button("Save workflow update")
            if updated:
                task["reviewer"] = reviewer
                task["status"] = status
                task["note"] = note.strip()
                st.rerun()


def main() -> None:
    st.markdown(
        """
        <style>
        :root { --ink: #1E244E; --muted: #4b5188; --blue: #8DAAF6; --lav: #ABADDA; --gold: #E5B46E; }
        .stApp { background: linear-gradient(118deg, #f4f5fc 0%, #eceefa 56%, #e3e6f6 100%); color: var(--ink); }
        [data-testid="stSidebar"] { background: var(--ink); }
        [data-testid="stSidebar"] * { color: #f4f5fc !important; }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color: var(--lav) !important; }
        h1, h2, h3 { color: var(--ink); }
        h1 { border-bottom: 3px solid var(--gold); padding-bottom: 6px; }
        .stApp p, .stApp label, .stApp li, .stApp span, [data-testid="stWidgetLabel"] *, [data-testid="stMarkdownContainer"] * { color: var(--ink); }
        [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: var(--muted) !important; }
        [data-testid="stForm"], [data-testid="stExpander"], div[data-testid="stVerticalBlockBorderWrapper"] { background: rgba(255,255,255,.75); border: 1px solid var(--lav); border-radius: 8px; }
        [data-testid="stStatusWidget"], [data-testid="stStatus"] { background: #ffffff; }
        textarea, input, [data-baseweb="select"] * { color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important; }
        textarea, input { background-color: #ffffff !important; }
        div[data-testid="stMetric"] { background: rgba(255,255,255,.85); border-top: 3px solid var(--gold); padding: 12px 14px; }
        div[data-testid="stMetric"] label { color: var(--muted); }
        div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] { background: var(--blue); border-color: var(--blue); }
        div.stButton > button[kind="primary"] *, div.stFormSubmitButton > button[kind="primary"] * { color: var(--ink) !important; font-weight: 600; }
        div.stButton > button[kind="primary"]:hover, div.stFormSubmitButton > button[kind="primary"]:hover { background: var(--gold); border-color: var(--gold); }        div[data-testid="stVerticalBlock"] > div { animation: rise-in .24s ease-out both; }
        @keyframes rise-in { from { opacity: .65; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("## Clinical Workflow Desk")
    st.sidebar.caption("Synthetic underwriting workflow showcase")
    page = st.sidebar.radio(
        "Workspace",
        ["Overview", "New application", "Underwriter review", "Clinician workflows"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    models = available_models()
    if not models:
        st.error("No trained model found.")
        st.code(
            "python -m workflow_ml.serving      # ensemble v2\n"
            "python src\\build_model_table.py\npython src\\export_model.py   # original model\n"
            "streamlit run app.py",
            language="powershell",
        )
        st.stop()
    st.sidebar.caption("Prediction model")
    model_name = st.sidebar.radio("Prediction model", models, label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption("Synthetic data only · No clinical or underwriting decisions")

    if page == "Overview":
        render_overview()
    elif page == "New application":
        render_intake(model_name)
    elif page == "Underwriter review":
        render_review()
    else:
        render_workflow_queue()


if __name__ == "__main__":
    main()
