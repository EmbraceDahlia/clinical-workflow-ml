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

