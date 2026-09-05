from __future__ import annotations

import pandas as pd


def compare_models(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    score_rows = []
    for model_name, metrics in results.items():
        if metrics.empty:
            continue
        row = {
            "model": model_name,
            "mean_f1": metrics["f1"].mean(),
            "mean_precision": metrics["precision"].mean(),
            "mean_recall": metrics["recall"].mean(),
        }
        if "pr_auc" in metrics.columns:
            row["mean_pr_auc"] = metrics["pr_auc"].mean()
        score_rows.append(row)

    if not score_rows:
        return pd.DataFrame(columns=["model", "mean_f1", "mean_precision", "mean_recall", "mean_pr_auc"])

    summary = pd.DataFrame(score_rows)
    summary = summary.sort_values("mean_f1", ascending=False).reset_index(drop=True)
    return summary
