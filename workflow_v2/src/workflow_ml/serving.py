"""Export the trained pipeline as one bundle and serve predictions from it.

    python -m workflow_ml.serving     # train and write artifacts/model/workflow_model.joblib

The bundle is trained on four patient folds. The fifth fold (DEMO_FOLD) is never used
for training, so an application can show live predictions for patients the model has
not seen and compare them with what actually happened.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from workflow_ml.config import ARTIFACT_DIR, LABELS
from workflow_ml.data import load_raw
from workflow_ml.features import Vocabulary, fit_vocabulary, load_table
from workflow_ml.genetic import Blend, evolve_blend
from workflow_ml.intake import Catalog, application_table, build_catalog
from workflow_ml.models.base import WorkflowModel
from workflow_ml.split import patient_folds
from workflow_ml.stacking import StackingModel
from workflow_ml.train import BASE_MODELS, cached_scores, combine

BUNDLE_PATH = ARTIFACT_DIR / "model" / "workflow_model.joblib"
# The fold with the median score in artifacts/reports/fold_f1.csv: a typical one, not the best.
DEMO_FOLD = 1


@dataclass
class ModelBundle:
    models: dict[str, WorkflowModel]
    stacker: StackingModel
    blend: Blend
    vocabulary: Vocabulary
    catalog: Catalog
    demo_patients: list[str]

    @property
    def thresholds(self) -> dict[str, float]:
        return dict(zip(LABELS, self.blend.thresholds.tolist()))

    def predict(self, table: pd.DataFrame, rows: Sequence[int]) -> pd.DataFrame:
        """One `<label>_score` and one `<label>_predicted` column per workflow label."""
        rows = np.asarray(rows)
        base = [model.predict_proba(table, rows) for model in self.models.values()]
        stacked = self.stacker.predict_proba(np.hstack(base))
        scores = self.blend.scores(np.stack([*base, stacked], axis=1))

        result = pd.DataFrame(index=table.index[rows])
        for column, label in enumerate(LABELS):
            result[f"{label}_score"] = scores[:, column]
            result[f"{label}_predicted"] = scores[:, column] >= self.blend.thresholds[column]
        return result

    def predict_application(
        self,
        profile: Mapping[str, object],
        history_records: Sequence[Mapping[str, object]],
        evaluation_date: object,
    ) -> tuple[pd.Series, pd.Series]:
        """Prediction and feature values for a hand-entered applicant (see `intake`)."""
        table, row = application_table(
            profile, history_records, evaluation_date, self.catalog, self.vocabulary
        )
        return self.predict(table, [row]).iloc[0], table.iloc[row]


def evidence(features: pd.Series, label: str) -> list[str]:
    """Case-specific facts behind a score, read straight from the encounter's features."""
    since = features[f"since_{label}"]
    lines = [
        "Never recorded before for this patient" if pd.isna(since)
        else f"Last recorded {since:.0f} days ago",
        f"{features[f'{label}_last_365d']:.0f} recorded in the last 12 months",
    ]
    similar_visits = features["resolved_type_count"]
    if similar_visits > 0:
        rate = features[f"rate_type_{label}"]
        lines.append(f"Followed {rate:.0%} of {similar_visits:.0f} earlier visits of this type")
    return lines


def export_bundle(path: Path = BUNDLE_PATH) -> ModelBundle:
    table = load_table()
    folds = patient_folds(table)
    train, held_out = folds[DEMO_FOLD]
    y_true = table[list(LABELS)].to_numpy()

    models = {}
    for name, make_model in BASE_MODELS.items():
        print(f"training {name} ...", flush=True)
        models[name] = make_model().fit(table, train)
    models["gru"].to("cpu")

    # The meta-learners are fitted on out-of-fold scores of the training patients only.
    base_scores = {name: cached_scores(name, make, table, folds) for name, make in BASE_MODELS.items()}
    scores, _, _ = combine(base_scores, y_true, folds)
    base = np.hstack(list(base_scores.values()))
    candidates = np.stack([scores[name] for name in (*BASE_MODELS, "stacking")], axis=1)

    raw = load_raw()
    bundle = ModelBundle(
        models=models,
        stacker=StackingModel().fit(base[train], y_true[train]),
        blend=evolve_blend(candidates[train], y_true[train]),
        vocabulary=fit_vocabulary(raw),
        catalog=build_catalog(raw),
        demo_patients=sorted(table["patient_id"].iloc[held_out].unique()),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return bundle


def load_bundle(path: Path = BUNDLE_PATH) -> ModelBundle:
    return joblib.load(path)


if __name__ == "__main__":
    # Import through the package so the saved classes are not pickled as `__main__.*`.
    from workflow_ml.serving import BUNDLE_PATH as saved_to
    from workflow_ml.serving import export_bundle as export

    export()
    print(f"Saved model bundle to {saved_to}")
