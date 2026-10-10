# Machine Learning-Based Clinical Workflow Recommendation Using Longitudinal EHR Data

A research project investigating whether machine learning can recommend
upcoming clinical workflows from longitudinal electronic health record (EHR)
information. The project is inspired by rule-based healthcare workflow
automation in insurance underwriting, where medical information is reviewed to
determine whether laboratory testing, medical examination, imaging, or other
procedures are required.

This project does not predict medical necessity, make underwriting decisions,
or provide clinical advice. Its workflow labels are proxies derived from
observed synthetic records.

## Dataset

The project uses the publicly available [Synthea synthetic EHR dataset](https://synthetichealth.github.io/synthea/). The repository currently uses linked patient, encounter, condition, observation, procedure, medication, and imaging records stored as Synthea-style CSV files in `data/`.

Public datasets containing actual insurance underwriting requirements and
workflow decisions are difficult to obtain. Synthea provides a related,
publicly available alternative with longitudinal patient histories while
avoiding the use of real patient data.

## Overview

Workflow recommendation is formulated as a **multi-label classification**
problem. A patient encounter may correspond to one or more workflow categories
at the same time. The four categories investigated in the current experiment
are:

- Medical examination
- Laboratory testing
- Imaging
- Instrument-based testing

The main comparison evaluates models using basic current-encounter information
against models that additionally use longitudinal history, including previous
conditions, medications, procedures, observations, imaging, and encounters.
Logistic Regression, Linear SVM, and Random Forest are evaluated using
precision, recall, and F1-score.

## Current Scope

The feature builder creates one row per encounter. Features are restricted to
information available before the encounter, while workflow labels represent
events recorded within the following 30 days:

- `medical_exam`
- `lab_test`
- `imaging`
- `instrument_test`

A label value of `1` means that a corresponding workflow event was observed in
the future window; `0` means that it was not observed. These labels are proxies
for workflow requirements, not confirmed requirements or clinical outcomes.

The current experiments use three feature sets:

- **Basic**: demographic and current-encounter information.
- **Longitudinal**: Basic features plus prior encounter, event-count, and
  recency features.
- **Full**: Longitudinal features plus prior screening measurements.

Direct identifiers, location fields, payer data, provider data, and
organization data are excluded from the first experiment.

## Experimental Results

The reported Random Forest results show the following mean F1-scores across
the workflow labels:

| Feature set | Mean F1-score |
| --- | ---: |
| Basic | 0.261 |
| Longitudinal | 0.536 |
| Full | 0.493 |

The Longitudinal feature set improved the reported mean F1-score substantially
over Basic features, suggesting that prior patient history provides useful
predictive information. Adding prior screening measurements in the Full
feature set reduced the reported mean F1-score relative to Longitudinal
features.

The experiment also highlights the challenge of **class imbalance**. Some
workflow labels, particularly instrument-based testing, occur much less often
than others. Generated comparison figures are stored under `reports/` after
the pipeline is run.

## Setup

Use Python 3.13 or a compatible Python 3 version:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On this Windows workspace, the configured interpreter can also be run as:

```powershell
C:/Python313/python.exe -m pip install -r requirements.txt
```

## Build the Modeling Table

From the project root, run:

```powershell
python src\build_model_table.py
```

The builder computes leakage-controlled history features, derives the four
workflow labels from events in the following 30 days, and writes the generated
encounter-level table to `model_table.csv`.

## Run the Pipeline

Using the project-local virtual environment:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_pipeline -q
.\.venv\Scripts\python.exe src\build_model_table.py
.\.venv\Scripts\python.exe src\run_pipeline.py
```

`run_pipeline.py` runs the `basic`, `longitudinal`, and `full` experiments. For
each experiment it:

1. Splits data by patient into training, validation, and test sets.
2. Preprocesses numeric and categorical features.
3. Trains Logistic Regression, Random Forest, and One-vs-Rest Linear SVM
   baselines.
4. Tunes a decision threshold for each label on the validation set.
5. Evaluates the final predictions on the held-out test set.
6. Writes per-label reports and comparison charts.

Patient-grouped splitting prevents encounters from the same patient from
appearing on both sides of a split. The pipeline reports per-label precision,
recall, F1, average precision (`pr_auc`), and confusion matrices, together
with mean precision, recall, and F1 by model. Thresholds are selected by
maximizing validation-set F1; the test split is reserved for final metrics.

The generated artifacts are:

- `model_table.csv`
- `reports/<experiment>/per_label/`
- Summary comparison charts under `reports/`

These outputs are intentionally ignored by Git. `train_baselines.py` provides
the model factory used by the pipeline and does not need to be run separately.

## Export a Model for an Application

To create a reusable Longitudinal Random Forest model bundle, build the feature
table and run:

```powershell
python src\build_model_table.py
python src\export_model.py
```

This writes `models/longitudinal_random_forest.joblib`. The bundle contains the
fitted preprocessing steps, classifier, expected feature columns, and
validation-tuned threshold for each workflow label. It is intentionally ignored
by Git; regenerate it from the project data and installed dependencies when
setting up another environment.

An application can load the bundle once at startup and predict from a row or
DataFrame containing the same longitudinal input features:

```python
from src.model_artifact import load_model_artifact, predict_workflows

model = load_model_artifact()
prediction = predict_workflows(feature_row, model)
```

The result contains one score and one thresholded prediction column per label.
The input must include the model's longitudinal history features; the model
does not derive those history features from raw EHR records. These predictions
are for a synthetic-data demonstration, not clinical decision-making.

## Run the Showcase App

Install project requirements, create the synthetic feature table and model
artifact if they are not already present, then launch the local app:

```powershell
python -m pip install -r requirements.txt
python src\build_model_table.py
python src\export_model.py
streamlit run app.py
```

### Use the ensemble model (workflow_v2) in the app

The app offers two prediction models in the sidebar. The original Random Forest
needs only the steps above. To add the ensemble (mean F1 0.789), install the
`workflow_v2` package into the same environment and export its bundle once:

```powershell
python -m pip install -e workflow_v2
python -m workflow_ml.serving
streamlit run app.py
```

With the ensemble selected, **New application** has a **Synthetic patient** tab.
It lists the synthetic patients that were held out of training, scores the chosen
encounter live, and the review page shows what was actually recorded in the
following 30 days. **Manual entry** also works with the ensemble: the dated
history is converted to Synthea-style records and passed through the same
feature builder, but such short histories are outside what the model was
trained on, so treat those scores as illustrative.

To enable background free-text extraction, configure the API key and model in
the same PowerShell session before starting the app:

```powershell
$env:OPENAI_API_KEY = "<your API key>"
$env:HISTORY_EXTRACTION_MODEL = "gpt-4.1-mini"
streamlit run app.py
```

The app always uses manual intake: enter applicant/evaluation details, then add
dated encounters, allergies, conditions, immunizations, procedures, imaging,
medications, and observations in the history grid. It derives the model's
7/30/90/180-day counts and recency features from those records. Optional record
details are shown to the reviewer but are not analyzed by the model.
When free text is provided, the main **Generate workflow suggestions** action
starts extraction in the background. Review and add or discard the returned
draft records, then submit **Generate workflow suggestions** again to run the
classifier. Drafts include source quotes and model-reported extraction
confidence. Relative or ambiguous dates are left blank for the user to resolve.
The free text is sent to the configured OpenAI model only after that action is
started; use synthetic text for this showcase.

The remaining steps are model-score review, an underwriter decision, and
clinician review requests for approved categories. Cases are held only in
Streamlit session state, so they are not persistent or shared across users.
Feature importances shown during review are global training-set signals, not
case-specific explanations. Use synthetic records only; the app does not create
medical orders or make insurance eligibility decisions.

## Future Work

Future experiments will investigate imbalance-aware methods to improve the
prediction of rare workflow requirements and determine whether better rare-label
performance improves overall results. The project will also investigate
recurrent neural networks (RNNs) and other sequence-aware approaches to model
the order of longitudinal EHR events explicitly, comparing them with the
current feature-based baselines.

Any future clinical or operational use would require validated workflow
decisions, clinical review, privacy controls, fairness analysis, and an
appropriate human-in-the-loop process.

## References

- [Synthea Synthetic Patient Generator](https://synthetichealth.github.io/synthea/)
- [Analysis of Dimensionality Reduction Techniques on Big Data](https://ieeexplore.ieee.org/document/9036908)
- [Linear Dimensionality Reduction: Survey, Insights, and Generalizations](https://jmlr.org/papers/volume16/cunningham15a/cunningham15a.pdf)
- [Recurrent Neural Networks: A Gentle Introduction and Overview](https://arxiv.org/pdf/1912.05911)
- [A Review of Methods for Imbalanced Multi-label Classification](https://www.sciencedirect.com/science/article/abs/pii/S0031320321001527)

