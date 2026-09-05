from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compare_models import compare_models
from src.config import BASELINE_MODELS, MODEL_TABLE_PATH, RANDOM_STATE, TARGET_COLUMNS
from src.data_split import patient_group_split
from src.evaluate_models import (
    default_score_threshold,
    evaluate_model,
    extract_per_label_scores,
    summarize_per_label,
    tune_thresholds,
)
from src.plot_results import (
    plot_per_label_results,
    plot_experiment_metric,
    plot_overall_metric,
)
from src.preprocess import build_preprocessor, select_feature_columns
from src.target_definition import define_targets
from src.train_baselines import train_model


def to_dense_if_sparse(matrix):
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return matrix

def run_pipeline(
    table_path: Path = MODEL_TABLE_PATH,
    experiment: str = "longitudinal",
) -> dict[str, object]:

    df = pd.read_csv(table_path)

    train_val_df, test_df = patient_group_split(
        df,
        target_col="medical_exam",
        random_state=RANDOM_STATE,
    )

    train_df, val_df = patient_group_split(
        train_val_df,
        target_col="medical_exam",
        random_state=RANDOM_STATE,
    )

    X_train = train_df.drop(columns=TARGET_COLUMNS)
    X_val = val_df.drop(columns=TARGET_COLUMNS)
    X_test = test_df.drop(columns=TARGET_COLUMNS)

    y_train = define_targets(train_df)
    y_val = define_targets(val_df)
    y_test = define_targets(test_df)

    # Select features for this experiment
    feature_columns = select_feature_columns(
        X_train,
        experiment=experiment,
    )

    # Build preprocessing pipeline
    preprocessor = build_preprocessor(
        X_train,
        feature_columns,
    )

    X_train_processed = to_dense_if_sparse(
        preprocessor.fit_transform(
            X_train[feature_columns]
        )
    )

    X_val_processed = to_dense_if_sparse(
        preprocessor.transform(
            X_val[feature_columns]
        )
    )

    X_test_processed = to_dense_if_sparse(
        preprocessor.transform(
            X_test[feature_columns]
        )
    )

    results = {}
    thresholds_by_model = {}

    for model_name in BASELINE_MODELS:

        model = train_model(
            model_name,
            X_train_processed,
            y_train,
        )

        val_scores = extract_per_label_scores(
            model,
            X_val_processed,
        )

        thresholds = tune_thresholds(
            y_val,
            val_scores,
            default_score_threshold(model),
        )

        thresholds_by_model[model_name] = thresholds

        metrics = evaluate_model(
            model,
            X_test_processed,
            y_test,
            thresholds=thresholds,
        )

        results[model_name] = metrics

    summary = compare_models(results)
    per_label = summarize_per_label(results)

    print(f"\n{'=' * 70}")
    print(f"{experiment.upper()} RESULTS")
    print(f"{'=' * 70}")

    print(summary.round(3).to_string(index=False))

    print("\nPer-label metrics:")
    print(per_label.round(3).to_string(index=False))

    print("\nValidation-tuned thresholds:")
    for model_name, thresholds in thresholds_by_model.items():
        print(
            model_name,
            {
                target: round(value, 3)
                for target, value in thresholds.items()
            },
        )

    plot_per_label_results(
        per_label,
        output_dir=Path("reports") / experiment / "per_label",
    )
    return {
        "results": results,
        "summary": summary,
        "thresholds": thresholds_by_model,
    }

def run_pipeline_summary(
    table_path: Path = MODEL_TABLE_PATH,
) -> dict[str, object]:

    experiments = [
        "basic",
        "longitudinal",
        "full",
    ]

    all_results = {}
    all_summaries = []
    all_per_label = []

    for experiment in experiments:

        output = run_pipeline(
            table_path=table_path,
            experiment=experiment,
        )

        all_results[experiment] = output

        # Overall model results
        summary = output["summary"].copy()
        summary["experiment"] = experiment
        all_summaries.append(summary)

        # Per-label results
        per_label = summarize_per_label(
            output["results"]
        )
        per_label["experiment"] = experiment
        all_per_label.append(per_label)

        print(f"\n{'=' * 70}")
        print(f"{experiment.upper()} RESULTS")
        print(f"{'=' * 70}")
        print(summary.round(3).to_string(index=False))
        print("\nPer-label metrics:")
        print(per_label.round(3).to_string(index=False))

    combined_summary = pd.concat(
        all_summaries,
        ignore_index=True,
    )

    combined_per_label = pd.concat(
        all_per_label,
        ignore_index=True,
    )

    # Overall comparison
    plot_overall_metric(
        combined_summary,
        model="random_forest",
        metric="f1",
    )

    # Per-workflow comparisons
    plot_experiment_metric(
        combined_per_label,
        metric="f1",
    )

    plot_experiment_metric(
        combined_per_label,
        metric="precision",
    )

    plot_experiment_metric(
        combined_per_label,
        metric="recall",
    )

    return {
        "results": all_results,
        "summary": combined_summary,
        "per_label": combined_per_label,
    }

if __name__ == "__main__":
    run_pipeline()   
    # run_pipeline_summary()
