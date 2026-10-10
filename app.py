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

from src.config import MODEL_ARTIFACT_PATH
from src.history_agent import extract_history_records, history_model_name
from src.manual_features import build_model_features
from src.model_artifact import load_model_artifact, predict_workflows


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


def render_manual_inputs() -> tuple[dict[str, object], str]:
    st.subheader("Applicant and evaluation")
    first_row = st.columns([1.4, 1])
    applicant_columns = first_row[0].columns(2)
    with applicant_columns[0]:
        date_of_birth = st.date_input(
            "Date of birth",
            value=date(1985, 1, 1),
            min_value=date(1900, 1, 1),
            max_value=date.today(),
        )
        gender = st.selectbox(
            "Gender",
            ["F", "M", "Other"],
            format_func=lambda value: {"F": "Female", "M": "Male", "Other": "Other / not recorded"}[value],
        )
        encounter_class = st.text_input("Evaluation type", "outpatient")
    with applicant_columns[1]:
        evaluation_date = st.date_input("Evaluation date", value=date.today())
        encounter_code = st.number_input(
            "Encounter code (numeric)",
            min_value=0,
            max_value=9_999_999_999,
            value=410620009,
            step=1,
            format="%d",
        )
        reason_code = st.text_input("Reason code", "NO_REASON_RECORDED")
        prior_total_encounters = st.number_input(
            "Total prior encounters",
            min_value=0,
            max_value=100_000,
            value=0,
            step=1,
            help="Lifetime encounter count; the model relies on it heavily. The larger of this and the extracted encounters is used.",
        )
    with first_row[1]:
        st.text_area(
            "Medical history (free text)",
            key="medical_history_text",
            max_chars=12000,
            height=220,
            placeholder="Enter a synthetic medical history. The 7, 30 and 90 day model inputs are extracted automatically.",
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
        "ENCOUNTERCLASS": encounter_class,
        "encounter_code": encounter_code,
        "reason_code": reason_code,
        "date_of_birth": date_of_birth,
        "prior_total_encounters": prior_total_encounters,
    }

    return (
        current_features | {"evaluation_date": evaluation_date},
        st.session_state.get("medical_history_text", ""),
    )


def make_prediction_case(
    reference: str,
    features: dict[str, object],
    artifact: dict[str, object],
    history_records: list[dict[str, object]],
) -> dict[str, object]:
    prediction = predict_workflows(features, artifact).iloc[0]
    predictions = {}
    for target in artifact["target_columns"]:
        predictions[target] = {
            "score": float(prediction[f"{target}_score"]),
            "threshold": float(artifact["thresholds"][target]),
            "proposed": bool(prediction[f"{target}_predicted"]),
            "signals": get_top_signals(artifact, target),
        }
    return {
        "id": uuid.uuid4().hex,
        "reference": reference,
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "features": features,
        "history_records": history_records,
        "predictions": predictions,
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


def render_intake(artifact: dict[str, object]) -> None:
    st.title("New application")
    st.caption("Enter an applicant profile and a free-text medical history.")
    st.warning("Showcase only: use synthetic data. Do not enter real policyholder identifiers or health information.")

    with st.form("application_intake"):
        reference = st.text_input("Application reference", placeholder="DEMO-1001")
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
            current_features.pop("date_of_birth")
            prior_total_encounters = int(current_features.pop("prior_total_encounters"))
            step = "Starting"
            with st.status("Generating workflow suggestions...", expanded=True) as status:
                try:
                    history_records = []
                    if history_text.strip():
                        step = "Step 1/4: Checking OpenAI configuration"
                        st.write(step)
                        if not os.environ.get("OPENAI_API_KEY"):
                            raise ValueError("OPENAI_API_KEY is not set in your environment.")

                        step = "Step 2/4: Extracting history with the OpenAI agent"
                        st.write(f"{step} ({history_model_name()})")
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
                        st.write("No medical history text entered; skipping extraction.")

                    step = "Step 3/4: Computing 7/30/90 day model features"
                    st.write(step)
                    feature_values = build_model_features(
                        current_features,
                        history_records,
                        evaluation_date,
                        artifact["feature_columns"],
                    )
                    feature_values["patient_total_encounters"] = max(
                        int(feature_values["patient_total_encounters"]), prior_total_encounters
                    )

                    step = "Step 4/4: Running the prediction model"
                    st.write(step)
                    case = make_prediction_case(
                        reference.strip(),
                        feature_values,
                        artifact,
                        history_records,
                    )
                except Exception as error:
                    status.update(label=f"Failed at: {step}", state="error", expanded=True)
                    st.error(safe_extraction_error(error))
                else:
                    status.update(label="Workflow suggestions ready", state="complete", expanded=False)
                    cases().insert(0, case)
                    st.success(f"Application {case['reference']} is ready for underwriter review.")


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
    st.caption("Model outputs are workflow-category signals derived from synthetic future-event labels, not procedure orders.")

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
            signals = result["signals"]
            signal_text = ", ".join(
                f"{name.replace('_', ' ')} ({importance:.0%})"
                for name, importance in signals
            ) or "No feature importance available"
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
    st.sidebar.caption("Synthetic data only · No clinical or underwriting decisions")

    try:
        artifact = get_model()
    except FileNotFoundError:
        st.error(f"Model artifact not found: {MODEL_ARTIFACT_PATH}")
        st.code("python src\\build_model_table.py\npython src\\export_model.py\nstreamlit run app.py", language="powershell")
        st.stop()

    if page == "Overview":
        render_overview()
    elif page == "New application":
        render_intake(artifact)
    elif page == "Underwriter review":
        render_review()
    else:
        render_workflow_queue()


if __name__ == "__main__":
    main()
