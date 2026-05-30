from __future__ import annotations

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from .utils import _metrics_at_threshold

def best_f1_threshold(labels: pd.Series, scores: pd.Series) -> dict[str, float]:
    y_true = labels.astype(int).to_numpy()
    y_score = scores.astype(float).to_numpy()
    if len(np.unique(y_true)) < 2 or len(np.unique(y_score)) < 2:
        threshold = float(np.median(y_score)) if len(y_score) else 0.5
        return {"threshold": threshold, **_metrics_at_threshold(labels, scores, threshold)}

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_idx = int(np.nanargmax(f1))
    threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 1.0
    return {"threshold": threshold, "f1": float(f1[best_idx]), "precision": float(precision[best_idx]), "recall": float(recall[best_idx]), "pred_long_rate": float((y_score >= threshold).mean())}

def constrained_f1_threshold(
    labels: pd.Series,
    scores: pd.Series,
    *,
    max_pred_long_rate: float = 0.70,
    min_precision: float = 0.30,
) -> dict[str, float | bool | str]:
    y_score = scores.astype(float).to_numpy()
    if len(y_score) == 0:
        return {
            "threshold": 0.5,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "pred_long_rate": 0.0,
            "threshold_source": "empty_fallback",
            "constraints_satisfied": False,
            "reject_reason": "empty_source",
        }

    fallback = best_f1_threshold(labels, scores)
    candidates = sorted(set(float(value) for value in y_score if np.isfinite(value)))
    if 0.5 not in candidates:
        candidates.append(0.5)
    rows = []
    for threshold in candidates:
        metrics = _metrics_at_threshold(labels, scores, threshold)
        rows.append({"threshold": threshold, **metrics})
    feasible = [
        row
        for row in rows
        if row["pred_long_rate"] <= max_pred_long_rate
        and row["precision"] >= min_precision
        and row["pred_long_rate"] > 0.0
    ]
    if not feasible:
        return {
            **fallback,
            "threshold_source": "unconstrained_fallback",
            "constraints_satisfied": False,
            "reject_reason": "no_threshold_satisfies_pred_rate_precision",
        }
    best = max(
        feasible,
        key=lambda row: (
            row["f1"],
            row["precision"],
            -row["pred_long_rate"],
            row["threshold"],
        ),
    )
    return {
        **best,
        "threshold_source": "constrained_f1",
        "constraints_satisfied": True,
        "reject_reason": "",
    }

def threshold_diagnostics(
    predictions: pd.DataFrame,
    *,
    score_column: str = "prob_long",
    threshold_source: str = "val",
    max_pred_long_rate: float | None = None,
    min_precision: float | None = None,
) -> pd.DataFrame:
    """Report whether bad F1 is mostly a threshold/calibration problem."""

    max_pred_long_rate = 0.70 if max_pred_long_rate is None else float(max_pred_long_rate)
    min_precision = 0.30 if min_precision is None else float(min_precision)
    rows = []
    has_splits = "split" in predictions.columns
    for fold, fold_part in predictions.groupby("fold"):
        if has_splits and threshold_source == "val":
            source = fold_part[fold_part["split"] == "val"]
            target = fold_part[fold_part["split"] == "test"]
            source_name = "val"
            if source.empty and not target.empty:
                source = target
                source_name = "same_split_oracle"
        else:
            source = fold_part
            target = fold_part
            source_name = "same_split_oracle"
        if source.empty or target.empty:
            continue

        source_best = best_f1_threshold(source["label"], source[score_column])
        target_at_source = _metrics_at_threshold(target["label"], target[score_column], source_best["threshold"])
        source_constrained = constrained_f1_threshold(
            source["label"],
            source[score_column],
            max_pred_long_rate=max_pred_long_rate,
            min_precision=min_precision,
        )
        target_at_constrained = _metrics_at_threshold(
            target["label"],
            target[score_column],
            source_constrained["threshold"],
        )
        target_oracle = best_f1_threshold(target["label"], target[score_column])
        rows.append(
            {
                "fold": int(fold),
                "threshold_source": source_name,
                "selected_threshold": source_best["threshold"],
                "source_best_f1": source_best["f1"],
                "test_f1_at_selected_threshold": target_at_source["f1"],
                "test_precision_at_selected_threshold": target_at_source["precision"],
                "test_recall_at_selected_threshold": target_at_source["recall"],
                "test_pred_long_rate_at_selected_threshold": target_at_source["pred_long_rate"],
                "constrained_threshold": source_constrained["threshold"],
                "constrained_threshold_source": source_constrained["threshold_source"],
                "constrained_threshold_constraints_satisfied": source_constrained["constraints_satisfied"],
                "constrained_threshold_reject_reason": source_constrained["reject_reason"],
                "source_constrained_f1": source_constrained["f1"],
                "source_constrained_precision": source_constrained["precision"],
                "source_constrained_recall": source_constrained["recall"],
                "source_constrained_pred_long_rate": source_constrained["pred_long_rate"],
                "test_f1_at_constrained_threshold": target_at_constrained["f1"],
                "test_precision_at_constrained_threshold": target_at_constrained["precision"],
                "test_recall_at_constrained_threshold": target_at_constrained["recall"],
                "test_pred_long_rate_at_constrained_threshold": target_at_constrained["pred_long_rate"],
                "test_oracle_best_threshold": target_oracle["threshold"],
                "test_oracle_best_f1": target_oracle["f1"],
                "test_f1_at_050": _metrics_at_threshold(target["label"], target[score_column], 0.5)["f1"],
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)

def threshold_grid_diagnostics(
    predictions: pd.DataFrame,
    *,
    score_column: str = "prob_long",
    threshold_source: str = "val",
    max_pred_long_rates: list[float] | None = None,
    min_precision: float = 0.30,
) -> pd.DataFrame:
    max_pred_long_rates = max_pred_long_rates or [0.30, 0.40, 0.50, 0.60, 0.70]
    rows = []
    has_splits = "split" in predictions.columns
    for fold, fold_part in predictions.groupby("fold"):
        if has_splits and threshold_source == "val":
            source = fold_part[fold_part["split"] == "val"]
            target = fold_part[fold_part["split"] == "test"]
            source_name = "val"
            if source.empty and not target.empty:
                source = target
                source_name = "same_split_oracle"
        else:
            source = fold_part
            target = fold_part
            source_name = "same_split_oracle"
        if source.empty or target.empty:
            continue

        base_long_rate = float(target["label"].mean())
        for cap in max_pred_long_rates:
            threshold = constrained_f1_threshold(
                source["label"],
                source[score_column],
                max_pred_long_rate=float(cap),
                min_precision=float(min_precision),
            )
            target_metrics = _metrics_at_threshold(target["label"], target[score_column], float(threshold["threshold"]))
            selected = target[target[score_column].astype(float) >= float(threshold["threshold"])]
            row = {
                "fold": int(fold),
                "threshold_source": source_name,
                "max_pred_long_rate": float(cap),
                "threshold": float(threshold["threshold"]),
                "source_f1": float(threshold["f1"]),
                "source_precision": float(threshold["precision"]),
                "source_recall": float(threshold["recall"]),
                "source_pred_long_rate": float(threshold["pred_long_rate"]),
                "constraints_satisfied": bool(threshold["constraints_satisfied"]),
                "reject_reason": str(threshold["reject_reason"]),
                "test_f1": target_metrics["f1"],
                "test_precision": target_metrics["precision"],
                "test_recall": target_metrics["recall"],
                "test_pred_long_rate": target_metrics["pred_long_rate"],
                "selected_count": int(len(selected)),
                "base_long_rate": base_long_rate,
                "selected_long_rate": float(selected["label"].mean()) if not selected.empty else np.nan,
                "lift_vs_base": float(selected["label"].mean() / base_long_rate)
                if not selected.empty and base_long_rate > 0
                else np.nan,
                "mean_forward_return": float(selected["forward_return"].mean())
                if not selected.empty and "forward_return" in selected.columns
                else np.nan,
            }
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["max_pred_long_rate", "fold"]).reset_index(drop=True)

def threshold_grid_summary_diagnostics(threshold_grid: pd.DataFrame) -> pd.DataFrame:
    if threshold_grid is None or threshold_grid.empty:
        return pd.DataFrame()
    frame = threshold_grid.copy()
    aggregations: dict[str, Any] = {
        "folds": ("fold", "nunique"),
        "threshold_mean": ("threshold", "mean"),
        "constraints_satisfied_fold_rate": ("constraints_satisfied", lambda values: float(pd.Series(values).astype(bool).mean())),
        "mean_source_f1": ("source_f1", "mean"),
        "mean_source_precision": ("source_precision", "mean"),
        "mean_source_recall": ("source_recall", "mean"),
        "mean_source_pred_long_rate": ("source_pred_long_rate", "mean"),
        "mean_f1": ("test_f1", "mean"),
        "mean_precision": ("test_precision", "mean"),
        "mean_recall": ("test_recall", "mean"),
        "mean_selection_rate": ("test_pred_long_rate", "mean"),
        "mean_lift_vs_base": ("lift_vs_base", "mean"),
        "positive_lift_fold_rate": ("lift_vs_base", lambda values: float((pd.to_numeric(values, errors="coerce") > 1.0).mean())),
    }
    if "mean_forward_return" in frame.columns:
        aggregations["mean_forward_return"] = ("mean_forward_return", "mean")
        aggregations["positive_forward_return_fold_rate"] = (
            "mean_forward_return",
            lambda values: float((pd.to_numeric(values, errors="coerce") > 0.0).mean()),
        )
    return frame.groupby("max_pred_long_rate", as_index=False).agg(**aggregations).reset_index(drop=True)

def threshold_summary_diagnostics(threshold_metrics: pd.DataFrame) -> pd.DataFrame:
    if threshold_metrics.empty:
        return pd.DataFrame()
    columns = [
        "selected_threshold",
        "source_best_f1",
        "test_f1_at_selected_threshold",
        "test_precision_at_selected_threshold",
        "test_recall_at_selected_threshold",
        "test_pred_long_rate_at_selected_threshold",
        "constrained_threshold",
        "source_constrained_f1",
        "source_constrained_precision",
        "source_constrained_recall",
        "source_constrained_pred_long_rate",
        "test_f1_at_constrained_threshold",
        "test_precision_at_constrained_threshold",
        "test_recall_at_constrained_threshold",
        "test_pred_long_rate_at_constrained_threshold",
        "test_oracle_best_f1",
        "test_f1_at_050",
    ]
    rows = []
    for column in columns:
        if column not in threshold_metrics.columns:
            continue
        values = threshold_metrics[column].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "metric": column,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "p25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "p75": float(values.quantile(0.75)),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)

