# Clinical workflow recommendation — version 2

Predicts, at each patient encounter, which workflows will be recorded in the **next 30 days**:
`medical_exam`, `lab_test`, `imaging`, `instrument_test` (multi-label classification on the
Synthea EHR data in `../data`). The target is identical to the original pipeline in `../src`;
`tests/test_labels.py` checks this.

## Setup

```bash
cd workflow_v2
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m ipykernel install --user --name workflow-v2 --display-name "Python (workflow-v2)"
```

## Run

```bash
.venv/bin/pytest -q                      # label parity + leakage tests
.venv/bin/python -m workflow_ml.train    # train everything, write artifacts/reports/
.venv/bin/jupyter lab notebooks/         # step-by-step notebooks, kernel "Python (workflow-v2)"
```

The first `train` run takes about 15 minutes on an Apple M1; later runs reuse cached scores.
Add `--refresh` to rebuild the features and retrain.

## Method

| Step | Algorithm | Covered in class |
| --- | --- | --- |
| Base model | Random Forest (bagging) | yes |
| Base model | Multi-layer perceptron (backpropagation) | yes |
| Extra features (ablation) | PCA + K-Means patient phenotypes | yes |
| Blend weights and thresholds | Genetic algorithm, fitness = F1 | yes |
| Base model | Gradient boosting (histogram-based) | **new** |
| Base model | GRU recurrent network over the last 24 encounters | **new** |
| Combiner | Stacking on out-of-fold scores | **new** |

### Features

Every feature uses only records dated at or before the encounter start
(`tests/test_features.py` deletes all later records and checks nothing changes).

| Prefix | Meaning |
| --- | --- |
| `since_`, `gap_`, `count_` | recency, rhythm and volume of each kind of visit, workflow, procedure |
| `due_` | days until the next event if the last gap repeats (captures visit schedules) |
| `*_last_30d/90d/365d` | recent event rates |
| `rate_` | share of the patient's earlier encounters that were followed by each workflow |
| `active_` | days each common condition has been active |
| `now_` | diagnoses, procedures, medications and care plans recorded at this encounter |

### Evaluation

5-fold cross-validation grouped by patient. Base models, the stacking meta-learner, the GA
blend and the decision thresholds are all fitted on four folds and applied to the fifth,
so every reported number comes from patients the model never saw.

## Layout

```
src/workflow_ml/
  config.py      paths, labels, horizon, seed
  data.py        load the Synthea CSVs
  labels.py      30-day workflow labels
  features.py    feature table (cached as artifacts/table.parquet)
  preprocess.py  scaling and one-hot encoding for the neural models
  phenotypes.py  PCA + K-Means features
  split.py       patient-grouped folds
  models/        forest.py, mlp.py, boosting.py, gru.py
  stacking.py    out-of-fold scoring and the meta-learner
  genetic.py     genetic algorithm and blend
  evaluate.py    thresholds and metrics
  train.py       full pipeline and report
notebooks/       01 data → 07 final evaluation
tests/           label parity, leakage
```

## Results

Mean F1 across the four labels, out-of-fold over all 1,171 patients (53,346 encounters).

| Model | All patients | Without the 3 largest patients | Mean ± std over folds |
| --- | ---: | ---: | ---: |
| Original pipeline (reported) | 0.536 | – | – |
| MLP | 0.606 | 0.525 | 0.587 ± 0.064 |
| GRU | 0.684 | 0.561 | 0.634 ± 0.071 |
| Random Forest | 0.745 | 0.581 | 0.671 ± 0.101 |
| Gradient boosting | 0.783 | 0.680 | 0.737 ± 0.078 |
| Stacking | 0.779 | 0.679 | 0.738 ± 0.075 |
| **GA blend (final)** | **0.789** | **0.681** | 0.742 ± 0.076 |

Final model per label:

| Label | Precision | Recall | F1 | PR-AUC | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| medical_exam | 0.630 | 0.784 | 0.699 | 0.753 | 0.933 |
| lab_test | 0.865 | 0.855 | 0.860 | 0.943 | 0.980 |
| imaging | 0.761 | 0.729 | 0.745 | 0.728 | 0.938 |
| instrument_test | 0.840 | 0.864 | 0.851 | 0.761 | 0.970 |

PCA + K-Means phenotype features did not help gradient boosting (0.782 vs 0.783), so the
final model does not use them; the ablation is in notebook 04.

## Limitations

- Three patients have 1,500–2,000 encounters each and hold most `instrument_test`
  positives. Without them that label drops to about 0.58 F1 and the mean to 0.68.
- Scores differ by fold (0.64 to 0.82) for the same reason, so a single train/test split
  would be unreliable on this dataset.
- Many future events in Synthea start at random (a new illness, an accident), which caps
  how well any model can predict the next 30 days.
- `medical_exam` is capped by coin-flip follow-ups: after an urgent-care visit a check-up
  follows in 51% of cases (57% after acute bronchitis), and nothing recorded at the visit
  separates the two outcomes.
- Labels are proxies from synthetic records, not confirmed clinical requirements.
