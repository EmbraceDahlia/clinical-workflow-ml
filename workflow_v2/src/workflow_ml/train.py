"""End-to-end run: features -> base models -> stacking -> GA blend -> report.

    python -m workflow_ml.train            # reuse cached base-model scores where present
    python -m workflow_ml.train --refresh  # rebuild features and retrain everything

Everything reported is out-of-fold: the base models, the stacking meta-learner, the GA
blend and the decision thresholds are all fitted on four patient folds and applied to
the fifth, so no patient is ever scored by something trained on their own records.
"""

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from workflow_ml.config import ARTIFACT_DIR, LABELS
from workflow_ml.evaluate import best_f1_thresholds, f1_per_label, score_table
from workflow_ml.features import load_table
from workflow_ml.genetic import Blend, evolve_blend
from workflow_ml.models.base import WorkflowModel
from workflow_ml.models.boosting import BoostingModel
from workflow_ml.models.forest import ForestModel
from workflow_ml.models.gru import GRUModel
from workflow_ml.models.mlp import MLPModel
from workflow_ml.split import Fold, patient_folds
from workflow_ml.stacking import StackingModel, cross_fit

BASE_MODELS = {
    "random_forest": ForestModel,
    "mlp": MLPModel,
    "gradient_boosting": BoostingModel,
    "gru": GRUModel,
}
SCORE_DIR = ARTIFACT_DIR / "scores"
REPORT_DIR = ARTIFACT_DIR / "reports"
LARGE_PATIENT_ENCOUNTERS = 800


@dataclass
class Experiment:
    table: pd.DataFrame
    folds: list[Fold]
    scores: dict[str, np.ndarray]     # model -> (n_rows, n_labels) out-of-fold probabilities
    decisions: dict[str, np.ndarray]  # model -> (n_rows, n_labels) out-of-fold 0/1 predictions
    final_blend: Blend                # GA blend refitted on all rows, for deployment

    @property
    def y_true(self) -> np.ndarray:
        return self.table[list(LABELS)].to_numpy()


def cached_scores(
    name: str, make_model: Callable[[], WorkflowModel], table: pd.DataFrame,
    folds: list[Fold], refresh: bool = False,
) -> np.ndarray:
    """Out-of-fold scores of one model, stored on disk because training takes minutes."""
    path = SCORE_DIR / f"{name}.npy"
    if refresh or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, cross_fit(make_model, table, folds))
    return np.load(path)


def out_of_fold_decisions(scores: np.ndarray, y_true: np.ndarray, folds: list[Fold]) -> np.ndarray:
    """0/1 predictions where each fold's thresholds are tuned on the other folds."""
    decisions = np.zeros(y_true.shape, dtype=int)
    for train, held_out in folds:
        decisions[held_out] = scores[held_out] >= best_f1_thresholds(y_true[train], scores[train])
    return decisions


def combine(
    scores: dict[str, np.ndarray], y_true: np.ndarray, folds: list[Fold]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], Blend]:
    """Add the stacking and GA-blend models and turn every model's scores into decisions."""
    scores = dict(scores)
    base = np.hstack(list(scores.values()))
    scores["stacking"] = np.zeros(y_true.shape)
    for train, held_out in folds:
        meta = StackingModel().fit(base[train], y_true[train])
        scores["stacking"][held_out] = meta.predict_proba(base[held_out])

    candidates = np.stack(list(scores.values()), axis=1)
    decisions = {name: out_of_fold_decisions(values, y_true, folds) for name, values in scores.items()}
    scores["ga_blend"] = np.zeros(y_true.shape)
    decisions["ga_blend"] = np.zeros(y_true.shape, dtype=int)
    for train, held_out in folds:
        blend = evolve_blend(candidates[train], y_true[train])
        scores["ga_blend"][held_out] = blend.scores(candidates[held_out])
        decisions["ga_blend"][held_out] = scores["ga_blend"][held_out] >= blend.thresholds

    return scores, decisions, evolve_blend(candidates, y_true)


def run_experiment(refresh: bool = False) -> Experiment:
    table = load_table(refresh)
    folds = patient_folds(table)
    y_true = table[list(LABELS)].to_numpy()

    base_scores = {}
    for name in BASE_MODELS:
        print(f"scoring {name} ...", flush=True)
        base_scores[name] = cached_scores(name, BASE_MODELS[name], table, folds, refresh)

    print("stacking and evolving the blend ...", flush=True)
    scores, decisions, final_blend = combine(base_scores, y_true, folds)
    return Experiment(table, folds, scores, decisions, final_blend)


def report(experiment: Experiment) -> pd.DataFrame:
    """Per-label metrics for every model, on all patients and without the few very large ones."""
    y_true = experiment.y_true
    visits = experiment.table.groupby("patient_id")["patient_id"].transform("size").to_numpy()
    scopes = {
        "all_patients": np.arange(len(y_true)),
        "without_largest_patients": np.flatnonzero(visits <= LARGE_PATIENT_ENCOUNTERS),
    }
    frames = []
    for model, scores in experiment.scores.items():
        for scope, rows in scopes.items():
            metrics = score_table(y_true[rows], scores[rows], experiment.decisions[model][rows])
            frames.append(metrics.assign(model=model, scope=scope))
    return pd.concat(frames, ignore_index=True)


def fold_f1(experiment: Experiment) -> pd.DataFrame:
    """Mean F1 of each model inside each held-out fold, to show how much folds differ."""
    y_true = experiment.y_true
    return pd.DataFrame(
        {
            model: [f1_per_label(y_true[held_out], decisions[held_out]).mean()
                    for _, held_out in experiment.folds]
            for model, decisions in experiment.decisions.items()
        }
    ).rename_axis("fold")


def summary(experiment: Experiment, results: pd.DataFrame) -> pd.DataFrame:
    """Mean F1 across the four labels: one row per model."""
    table = results.groupby(["model", "scope"], sort=False)["f1"].mean().unstack("scope")
    per_fold = fold_f1(experiment)
    table["fold_mean"], table["fold_std"] = per_fold.mean(), per_fold.std()
    return table.round(3)


def save(experiment: Experiment, results: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORT_DIR / "results.csv", index=False)
    summary(experiment, results).to_csv(REPORT_DIR / "summary.csv")
    fold_f1(experiment).round(4).to_csv(REPORT_DIR / "fold_f1.csv")
    blend = experiment.final_blend
    models = [name for name in experiment.scores if name != "ga_blend"]
    (REPORT_DIR / "ga_blend.json").write_text(json.dumps(
        {
            "models": models,
            "weights": dict(zip(LABELS, blend.weights.round(4).tolist())),
            "thresholds": dict(zip(LABELS, blend.thresholds.round(4).tolist())),
        },
        indent=2,
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the workflow models.")
    parser.add_argument("--refresh", action="store_true", help="rebuild features and retrain all models")
    arguments = parser.parse_args()

    experiment = run_experiment(arguments.refresh)
    results = report(experiment)
    save(experiment, results)

    print("\nMean F1 across the four labels (out-of-fold)")
    print(summary(experiment, results).to_string())
    final = results[(results["model"] == "ga_blend") & (results["scope"] == "all_patients")]
    print("\nFinal model (GA blend), all patients")
    print(final[["label", "precision", "recall", "f1", "pr_auc", "roc_auc"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
