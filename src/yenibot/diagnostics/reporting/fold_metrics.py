from __future__ import annotations

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from yenibot.diagnostics.metrics import classification_metrics, rank_ic


def fold_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, part in predictions.groupby("fold"):
        metrics = classification_metrics(part["label"], part["prob_long"])
        rows.append(
            {
                "fold": int(fold),
                "count": int(len(part)),
                "start": str(part["timestamp"].min()) if "timestamp" in part.columns else "",
                "end": str(part["timestamp"].max()) if "timestamp" in part.columns else "",
                "rank_ic": rank_ic(part["prob_long"], part["forward_return"]),
                "long_f1": metrics["long_f1"],
                "prauc": metrics["prauc"],
                "label_long_rate": float(part["label"].mean()),
                "pred_long_rate_050": float((part["prob_long"] >= 0.5).mean()),
                "prob_long_mean": float(part["prob_long"].mean()),
                "prob_long_std": float(part["prob_long"].std(ddof=0)),
                "prob_long_p10": float(part["prob_long"].quantile(0.10)),
                "prob_long_p50": float(part["prob_long"].quantile(0.50)),
                "prob_long_p90": float(part["prob_long"].quantile(0.90)),
                "forward_return_mean": float(part["forward_return"].mean()),
                "forward_return_std": float(part["forward_return"].std(ddof=0)),
            }
        )
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)

def recent_fold_diagnostics(fold_metrics: pd.DataFrame, *, recent_folds: int = 5) -> pd.DataFrame:
    if fold_metrics.empty:
        return pd.DataFrame()
    ordered = fold_metrics.sort_values("fold")
    recent = ordered.tail(recent_folds)
    metrics = [
        "rank_ic",
        "long_f1",
        "prauc",
        "label_long_rate",
        "pred_long_rate_050",
        "prob_long_mean",
        "prob_long_std",
        "forward_return_mean",
    ]
    rows = []
    for metric in metrics:
        if metric not in ordered.columns:
            continue
        all_values = ordered[metric].replace([np.inf, -np.inf], np.nan).dropna()
        recent_values = recent[metric].replace([np.inf, -np.inf], np.nan).dropna()
        if all_values.empty or recent_values.empty:
            continue
        rows.append(
            {
                "metric": metric,
                "all_mean": float(all_values.mean()),
                "recent_mean": float(recent_values.mean()),
                "recent_minus_all": float(recent_values.mean() - all_values.mean()),
                "recent_min": float(recent_values.min()),
                "recent_max": float(recent_values.max()),
                "recent_folds": ",".join(map(str, recent["fold"].astype(int).tolist())),
            }
        )
    return pd.DataFrame(rows)

def good_bad_fold_summary(fold_metrics: pd.DataFrame, *, good_ic: float = 0.10, bad_ic: float = -0.08) -> dict[str, Any]:
    good = fold_metrics.loc[fold_metrics["rank_ic"] >= good_ic, "fold"].astype(int).tolist()
    bad = fold_metrics.loc[fold_metrics["rank_ic"] <= bad_ic, "fold"].astype(int).tolist()
    return {
        "good_ic_threshold": good_ic,
        "bad_ic_threshold": bad_ic,
        "good_folds": good,
        "bad_folds": bad,
        "good_fold_count": len(good),
        "bad_fold_count": len(bad),
    }

