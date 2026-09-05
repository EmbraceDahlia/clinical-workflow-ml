from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import TARGET_COLUMNS


def plot_per_label_results(per_label: pd.DataFrame, output_dir: str | Path | None = None) -> None:
    if per_label.empty:
        raise ValueError("Cannot plot empty per-label results")

    target_dir = Path(output_dir) if output_dir else Path("reports") / "per_label"
    target_dir.mkdir(parents=True, exist_ok=True)

    models = sorted(per_label["model"].unique())
    available_targets = set(per_label["target"].unique())
    targets = [target for target in TARGET_COLUMNS if target in available_targets]
    model_colors = ["#023E8A", "#0077B6", "#00B4D8"]
    # model_colors = ["#0077B6", "#E9C46A", "#E76F51"]
    bar_width = 0.8 / len(models)
    x_positions = list(range(len(targets)))
    fig, axis = plt.subplots(figsize=(14, 7))

    for model_index, model in enumerate(models):
        subset = per_label[per_label["model"] == model].set_index("target").reindex(targets)
        values = subset["f1"].astype(float).to_numpy()
        positions = [position - 0.4 + (model_index + 0.5) * bar_width for position in x_positions]
        bars = axis.bar(
            positions,
            values,
            bar_width,
            label=model,
            color=model_colors[model_index % len(model_colors)],
        )
        for bar, value in zip(bars, values):
            if pd.notna(value):
                axis.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}",
                          ha="center", va="bottom", fontsize=8)

    axis.set_title("F1 Score by Target and Model")
    axis.set_xlabel("Target")
    axis.set_ylabel("F1 Score")
    axis.set_xticks(x_positions, targets, rotation=20, ha="right")
    axis.set_ylim(0, 1.1)
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    axis.legend(title="Model")
    fig.tight_layout()

    output_path = target_dir / "metrics_by_target.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved combined per-label graph to {output_path}")

def plot_experiment_metric(
    per_label: pd.DataFrame,
    metric: str,
    output_dir: str | Path | None = None,
) -> None:

    if per_label.empty:
        raise ValueError("Cannot plot empty results")

    if metric not in {"precision", "recall", "f1"}:
        raise ValueError(
            "Metric must be one of: precision, recall, f1"
        )

    target_dir = (
        Path(output_dir)
        if output_dir
        else Path("reports") / "experiments"
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    # We only want Random Forest for these comparison charts
    data = per_label[
        per_label["model"] == "random_forest"
    ].copy()

    experiments = [
        "basic",
        "longitudinal",
        "full",
    ]

    available_targets = set(data["target"].unique())

    targets = [
        target
        for target in TARGET_COLUMNS
        if target in available_targets
    ]

    bar_width = 0.8 / len(experiments)
    x_positions = list(range(len(targets)))

    fig, axis = plt.subplots(figsize=(14, 7))

    for experiment_index, experiment in enumerate(experiments):

        subset = (
            data[data["experiment"] == experiment]
            .set_index("target")
            .reindex(targets)
        )

        values = (
            subset[metric]
            .astype(float)
            .to_numpy()
        )

        positions = [
            position - 0.4
            + (experiment_index + 0.5) * bar_width
            for position in x_positions
        ]

        bars = axis.bar(
            positions,
            values,
            bar_width,
            label=experiment.capitalize(),
        )

        for bar, value in zip(bars, values):
            if pd.notna(value):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.02,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    metric_title = metric.replace("_", " ").title()

    axis.set_title(
        f"Random Forest {metric_title} by Workflow Category"
    )
    axis.set_xlabel("Workflow Category")
    axis.set_ylabel(metric_title)

    axis.set_xticks(
        x_positions,
        targets,
        rotation=20,
        ha="right",
    )

    axis.set_ylim(0, 1.1)
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.3,
    )

    axis.legend(title="Feature Set")

    fig.tight_layout()

    output_path = target_dir / f"random_forest_{metric}.png"

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)

    print(
        f"Saved {metric} comparison graph to {output_path}"
    )

def plot_overall_metric(
    summary: pd.DataFrame,
    model: str,
    metric: str,
    output_dir: str | Path | None = None,
) -> None:

    if summary.empty:
        raise ValueError("Cannot plot empty summary")

    target_dir = (
        Path(output_dir)
        if output_dir
        else Path("reports") / "experiments"
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    data = summary[
        summary["model"] == model
    ].copy()

    experiments = [
        "basic",
        "longitudinal",
        "full",
    ]

    data = (
        data.set_index("experiment")
        .reindex(experiments)
        .reset_index()
    )

    metric_column = f"mean_{metric}"
    values = data[metric_column].astype(float).to_numpy()

    labels = [
        experiment.capitalize()
        for experiment in experiments
    ]

    fig, axis = plt.subplots(figsize=(9, 6))

    bars = axis.bar(
        labels,
        values,
    )

    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    metric_title = metric.replace("_", " ").title()

    axis.set_title(
        f"Random Forest Mean {metric_title} Across Feature Sets"
    )
    axis.set_xlabel("Feature Set")
    axis.set_ylabel(f"Mean {metric_title}")
    axis.set_ylim(0, 0.65)

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.3,
    )

    fig.tight_layout()

    output_path = (
        target_dir /
        f"random_forest_overall_{metric}.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)

    print(
        f"Saved overall {metric} graph to {output_path}"
    )