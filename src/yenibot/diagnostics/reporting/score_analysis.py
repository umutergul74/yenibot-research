from __future__ import annotations

from __future__ import annotations
from fnmatch import fnmatch
from typing import Any
import numpy as np
import pandas as pd
from yenibot.diagnostics.metrics import rank_ic
from yenibot.features.builder import (
    raw_order_flow_v2_model_exclusions,
)

from .utils import _assign_score_bins, _resolve_score_bands, _config_get
from .threshold_analysis import threshold_grid_diagnostics, threshold_grid_summary_diagnostics

def score_lift_diagnostics(
    predictions: pd.DataFrame,
    *,
    score_column: str = "prob_long",
    bins: int = 10,
) -> pd.DataFrame:
    required = {"label", score_column}
    if not required.issubset(predictions.columns):
        return pd.DataFrame()

    frame = _assign_score_bins(predictions, score_column=score_column, bins=bins)
    if frame.empty:
        return pd.DataFrame()
    base_long_rate = float(frame["label"].mean())
    grouped = frame.groupby("score_bin", observed=True).agg(
        count=("label", "size"),
        mean_prob_long=(score_column, "mean"),
        actual_long_rate=("label", "mean"),
    )
    if "forward_return" in frame.columns:
        grouped["mean_forward_return"] = frame.groupby("score_bin", observed=True)["forward_return"].mean()
    grouped = grouped.reset_index()
    grouped["base_long_rate"] = base_long_rate
    grouped["lift_vs_base"] = grouped["actual_long_rate"] / base_long_rate if base_long_rate > 0 else np.nan
    grouped["is_top_bin"] = grouped["score_bin"] == grouped["score_bin"].max()
    return grouped

def score_band_diagnostics(
    predictions: pd.DataFrame,
    *,
    score_column: str = "prob_long",
    bins: int = 10,
    bands: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    required = {"label", score_column}
    if not required.issubset(predictions.columns):
        return pd.DataFrame()

    frame = _assign_score_bins(predictions, score_column=score_column, bins=bins)
    if frame.empty:
        return pd.DataFrame()

    resolved_bands = _resolve_score_bands(bands, bins=int(frame["score_bin"].max()) + 1)
    base_long_rate = float(frame["label"].mean())
    rows = []
    for band in resolved_bands:
        part = frame[(frame["score_bin"] >= band["min_bin"]) & (frame["score_bin"] <= band["max_bin"])]
        if part.empty:
            continue
        selected_positive = int(part["label"].astype(int).sum())
        total_positive = int(frame["label"].astype(int).sum())
        precision = float(part["label"].mean())
        recall = float(selected_positive / total_positive) if total_positive else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        row = {
            "band": band["name"],
            "min_bin": int(band["min_bin"]),
            "max_bin": int(band["max_bin"]),
            "count": int(len(part)),
            "selection_rate": float(len(part) / len(frame)),
            "mean_prob_long": float(part[score_column].mean()),
            "actual_long_rate": precision,
            "base_long_rate": base_long_rate,
            "lift_vs_base": float(precision / base_long_rate) if base_long_rate > 0 else np.nan,
            "recall": recall,
            "f1": float(f1),
        }
        if "forward_return" in part.columns:
            row["mean_forward_return"] = float(part["forward_return"].mean())
            row["rank_ic_within_band"] = rank_ic(part[score_column], part["forward_return"])
        rows.append(row)
    return pd.DataFrame(rows)

def score_band_by_fold_diagnostics(
    predictions: pd.DataFrame,
    *,
    score_column: str = "prob_long",
    bins: int = 10,
    bands: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    rows = []
    if "fold" not in predictions.columns:
        return pd.DataFrame()
    for fold, part in predictions.groupby("fold"):
        band_lift = score_band_diagnostics(part, score_column=score_column, bins=bins, bands=bands)
        if band_lift.empty:
            continue
        band_lift = band_lift.copy()
        band_lift.insert(0, "fold", int(fold))
        rows.append(band_lift)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["fold", "min_bin", "max_bin"]).reset_index(drop=True)

def score_band_summary_diagnostics(score_band_by_fold: pd.DataFrame) -> pd.DataFrame:
    if score_band_by_fold is None or score_band_by_fold.empty:
        return pd.DataFrame()
    frame = score_band_by_fold.copy()
    aggregations: dict[str, Any] = {
        "folds": ("fold", "nunique"),
        "mean_selection_rate": ("selection_rate", "mean"),
        "mean_actual_long_rate": ("actual_long_rate", "mean"),
        "mean_lift_vs_base": ("lift_vs_base", "mean"),
        "mean_f1": ("f1", "mean"),
        "mean_recall": ("recall", "mean"),
        "median_lift_vs_base": ("lift_vs_base", "median"),
        "positive_lift_fold_rate": ("lift_vs_base", lambda values: float((values > 1.0).mean())),
    }
    if "mean_forward_return" in frame.columns:
        aggregations["mean_forward_return"] = ("mean_forward_return", "mean")
        aggregations["positive_forward_return_fold_rate"] = (
            "mean_forward_return",
            lambda values: float((values > 0).mean()),
        )
    if "rank_ic_within_band" in frame.columns:
        aggregations["mean_rank_ic_within_band"] = ("rank_ic_within_band", "mean")
    summary = frame.groupby(["band", "min_bin", "max_bin"], as_index=False).agg(**aggregations)
    sort_columns = ["mean_lift_vs_base"]
    if "mean_forward_return" in summary.columns:
        sort_columns.append("mean_forward_return")
    return summary.sort_values(sort_columns, ascending=False).reset_index(drop=True)

def score_policy_grid_diagnostics(
    predictions: pd.DataFrame,
    *,
    score_column: str = "prob_long",
    bins: int = 10,
    bands: list[dict[str, Any]] | None = None,
    threshold_caps: list[float] | None = None,
    min_precision: float = 0.30,
) -> pd.DataFrame:
    test_predictions = predictions[predictions["split"] == "test"].copy() if "split" in predictions.columns else predictions.copy()
    rows: list[dict[str, Any]] = []

    band_global = score_band_diagnostics(test_predictions, score_column=score_column, bins=bins, bands=bands)
    band_by_fold = score_band_by_fold_diagnostics(test_predictions, score_column=score_column, bins=bins, bands=bands)
    band_summary = score_band_summary_diagnostics(band_by_fold)
    if not band_summary.empty:
        global_by_band = {str(row["band"]): row for _, row in band_global.iterrows()} if not band_global.empty else {}
        for _, item in band_summary.iterrows():
            band = str(item["band"])
            global_row = global_by_band.get(band, {})
            rows.append(
                {
                    "policy_type": "score_band",
                    "policy_name": band,
                    "band": band,
                    "threshold_cap": np.nan,
                    "threshold_mean": np.nan,
                    "folds": int(item.get("folds", 0)),
                    "selection_rate": float(global_row.get("selection_rate", item.get("mean_selection_rate", np.nan))),
                    "mean_selection_rate": float(item.get("mean_selection_rate", np.nan)),
                    "precision": float(global_row.get("actual_long_rate", item.get("mean_actual_long_rate", np.nan))),
                    "mean_precision": float(item.get("mean_actual_long_rate", np.nan)),
                    "recall": float(global_row.get("recall", item.get("mean_recall", np.nan))),
                    "mean_recall": float(item.get("mean_recall", np.nan)),
                    "f1": float(global_row.get("f1", item.get("mean_f1", np.nan))),
                    "mean_f1": float(item.get("mean_f1", np.nan)),
                    "lift_vs_base": float(global_row.get("lift_vs_base", item.get("mean_lift_vs_base", np.nan))),
                    "mean_lift_vs_base": float(item.get("mean_lift_vs_base", np.nan)),
                    "positive_lift_fold_rate": float(item.get("positive_lift_fold_rate", np.nan)),
                    "mean_forward_return": float(item.get("mean_forward_return", np.nan)),
                    "forward_return": float(global_row.get("mean_forward_return", item.get("mean_forward_return", np.nan))),
                    "positive_forward_return_fold_rate": float(item.get("positive_forward_return_fold_rate", np.nan)),
                }
            )

    threshold_grid = threshold_grid_diagnostics(
        predictions,
        score_column=score_column,
        max_pred_long_rates=threshold_caps,
        min_precision=min_precision,
    )
    threshold_summary = threshold_grid_summary_diagnostics(threshold_grid)
    for _, item in threshold_summary.iterrows():
        cap = float(item["max_pred_long_rate"])
        rows.append(
            {
                "policy_type": "threshold_cap",
                "policy_name": f"threshold_cap_{cap:.2f}",
                "band": "",
                "threshold_cap": cap,
                "threshold_mean": float(item.get("threshold_mean", np.nan)),
                "folds": int(item.get("folds", 0)),
                "selection_rate": float(item.get("mean_selection_rate", np.nan)),
                "mean_selection_rate": float(item.get("mean_selection_rate", np.nan)),
                "precision": float(item.get("mean_precision", np.nan)),
                "mean_precision": float(item.get("mean_precision", np.nan)),
                "recall": float(item.get("mean_recall", np.nan)),
                "mean_recall": float(item.get("mean_recall", np.nan)),
                "f1": float(item.get("mean_f1", np.nan)),
                "mean_f1": float(item.get("mean_f1", np.nan)),
                "lift_vs_base": float(item.get("mean_lift_vs_base", np.nan)),
                "mean_lift_vs_base": float(item.get("mean_lift_vs_base", np.nan)),
                "positive_lift_fold_rate": float(item.get("positive_lift_fold_rate", np.nan)),
                "mean_forward_return": float(item.get("mean_forward_return", np.nan)),
                "forward_return": float(item.get("mean_forward_return", np.nan)),
                "positive_forward_return_fold_rate": float(item.get("positive_forward_return_fold_rate", np.nan)),
                "constraints_satisfied_fold_rate": float(item.get("constraints_satisfied_fold_rate", np.nan)),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

def select_score_policy(policy_grid: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    if policy_grid is None or policy_grid.empty:
        return pd.DataFrame()
    threshold_cfg = _config_get(config or {}, ["validation", "threshold_checks"], {}) or {}
    policy_cfg = _config_get(config or {}, ["validation", "policy_selection"], {}) or {}
    max_selection_rate = float(_config_get(policy_cfg, ["max_selection_rate"], _config_get(threshold_cfg, ["max_pred_long_rate"], 0.70)))
    min_precision = float(_config_get(policy_cfg, ["min_precision"], _config_get(threshold_cfg, ["min_precision"], 0.30)))
    min_lift = float(_config_get(policy_cfg, ["min_lift_vs_base"], 1.0))
    min_positive_lift_rate = float(_config_get(policy_cfg, ["min_positive_lift_fold_rate"], 0.55))
    min_positive_return_rate = float(_config_get(policy_cfg, ["min_positive_forward_return_fold_rate"], 0.55))
    min_forward_return = float(_config_get(policy_cfg, ["min_forward_return"], 0.0))

    frame = policy_grid.copy()
    for column in (
        "selection_rate",
        "precision",
        "lift_vs_base",
        "positive_lift_fold_rate",
        "positive_forward_return_fold_rate",
        "mean_forward_return",
        "f1",
    ):
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    reasons = []
    passes = []
    for _, row in frame.iterrows():
        row_reasons = []
        if not pd.notna(row.get("selection_rate")) or float(row["selection_rate"]) > max_selection_rate:
            row_reasons.append("selection_rate")
        if not pd.notna(row.get("precision")) or float(row["precision"]) < min_precision:
            row_reasons.append("precision")
        if not pd.notna(row.get("lift_vs_base")) or float(row["lift_vs_base"]) <= min_lift:
            row_reasons.append("lift_vs_base")
        if not pd.notna(row.get("mean_forward_return")) or float(row["mean_forward_return"]) <= min_forward_return:
            row_reasons.append("mean_forward_return")
        if (
            pd.notna(row.get("positive_lift_fold_rate"))
            and float(row["positive_lift_fold_rate"]) < min_positive_lift_rate
        ):
            row_reasons.append("positive_lift_fold_rate")
        if (
            pd.notna(row.get("positive_forward_return_fold_rate"))
            and float(row["positive_forward_return_fold_rate"]) < min_positive_return_rate
        ):
            row_reasons.append("positive_forward_return_fold_rate")
        reasons.append(";".join(row_reasons))
        passes.append(len(row_reasons) == 0)
    frame["policy_pass"] = passes
    frame["policy_reject_reason"] = reasons
    frame = frame.sort_values(
        [
            "policy_pass",
            "mean_forward_return",
            "positive_forward_return_fold_rate",
            "lift_vs_base",
            "f1",
            "selection_rate",
        ],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)
    selected = frame.head(1).copy()
    selected["selected_policy"] = True
    return selected

def score_lift_by_fold_diagnostics(
    predictions: pd.DataFrame,
    *,
    score_column: str = "prob_long",
    bins: int = 10,
) -> pd.DataFrame:
    rows = []
    for fold, part in predictions.groupby("fold"):
        lift = score_lift_diagnostics(part, score_column=score_column, bins=bins)
        if lift.empty:
            continue
        bottom = lift.sort_values("score_bin").iloc[0]
        top = lift.sort_values("score_bin").iloc[-1]
        long_rate_spearman = lift["score_bin"].corr(lift["actual_long_rate"], method="spearman")
        row = {
            "fold": int(fold),
            "count": int(part["label"].notna().sum()),
            "base_long_rate": float(part["label"].mean()),
            "bottom_bin_long_rate": float(bottom["actual_long_rate"]),
            "top_bin_long_rate": float(top["actual_long_rate"]),
            "top_lift_vs_base": float(top["lift_vs_base"]),
            "top_minus_bottom_long_rate": float(top["actual_long_rate"] - bottom["actual_long_rate"]),
            "bin_long_rate_spearman": float(long_rate_spearman) if pd.notna(long_rate_spearman) else np.nan,
            "fold_rank_ic": rank_ic(part[score_column], part["forward_return"]) if "forward_return" in part.columns else np.nan,
        }
        if "mean_forward_return" in lift.columns:
            row["bottom_bin_forward_return"] = float(bottom["mean_forward_return"])
            row["top_bin_forward_return"] = float(top["mean_forward_return"])
            row["top_minus_bottom_forward_return"] = float(top["mean_forward_return"] - bottom["mean_forward_return"])
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)

def stationarity_policy_diagnostics(feature_columns: list[str], config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Verify that configured nonstationary feature patterns are absent from model inputs."""

    patterns = []
    if config is not None:
        patterns = list(_config_get(config, ["features", "stationarity", "exclude_patterns"], []) or [])
    rows = []
    for pattern in patterns:
        matches = sorted(column for column in feature_columns if fnmatch(column, pattern))
        rows.append(
            {
                "check": "nonstationary_feature_excluded",
                "pattern": pattern,
                "passed": len(matches) == 0,
                "matched_count": len(matches),
                "matched_features": ",".join(matches),
            }
        )
    raw_v2_matches = sorted(set(feature_columns) & raw_order_flow_v2_model_exclusions(config or {}))
    if raw_v2_matches:
        rows.append(
            {
                "check": "order_flow_v2_stable_only",
                "pattern": "<raw_order_flow_v2_exact_columns>",
                "passed": False,
                "matched_count": len(raw_v2_matches),
                "matched_features": ",".join(raw_v2_matches),
            }
        )
    elif config is not None and _config_get(config, ["features", "order_flow_v2", "stable_only"], False):
        rows.append(
            {
                "check": "order_flow_v2_stable_only",
                "pattern": "<raw_order_flow_v2_exact_columns>",
                "passed": True,
                "matched_count": 0,
                "matched_features": "",
            }
        )
    if rows:
        total_matches = sum(row["matched_count"] for row in rows)
        rows.append(
            {
                "check": "stationarity_policy_overall",
                "pattern": "<all>",
                "passed": total_matches == 0,
                "matched_count": total_matches,
                "matched_features": ",".join(
                    sorted(
                        {
                            feature
                            for row in rows
                            for feature in str(row["matched_features"]).split(",")
                            if feature
                        }
                    )
                ),
            }
        )
    return pd.DataFrame(rows, columns=["check", "pattern", "passed", "matched_count", "matched_features"])

