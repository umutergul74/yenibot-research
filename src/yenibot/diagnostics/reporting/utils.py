from __future__ import annotations

from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from yenibot.diagnostics.metrics import rank_ic
from yenibot.features.builder import (
    LABEL_COLUMNS,
    METADATA_COLUMNS,
    RAW_COLUMNS,
)

def classify_feature_column(feature: str) -> tuple[str, str]:
    timeframe = "4h" if feature.startswith("4h_") else "1h"
    name = feature[3:] if timeframe == "4h" else feature
    if name.startswith("fut_"):
        if "funding" in name:
            return "futures", "futures_funding_context"
        if any(token in name for token in ("toptrader", "long_short", "taker_long_short")):
            return "futures", "futures_positioning_context"
        if "_oi_" in name or name.startswith("fut_oi"):
            return "futures", "futures_open_interest_context"
        return "futures", "futures_context"
    if name.startswith("ih15_"):
        return "intrahour", "order_flow_intrahour"
    if "_x_" in name and any(token in name for token in ("rv14_rank", "gk14_rank", "atr14_rank")):
        return timeframe, "flow_volatility_interaction"
    if "_stable_" in name:
        base_name = name.split("_stable_", 1)[0]
        if any(token in base_name for token in ("log_return", "realized_vol", "gk_vol", "atr", "adx", "vwap", "denoised", "volume_log_zscore")):
            return timeframe, "volatility_structure_stable"
        return timeframe, "order_flow_v2_stable"
    if any(token in name for token in ("orderflow_efficiency", "absorption_pressure", "cvd_price_divergence", "large_trade_pressure", "cvd_pressure")):
        return timeframe, "order_flow_v2_raw"
    if any(token in name for token in ("taker_imbalance", "taker_buy_ratio_delta", "taker_buy_ratio_zscore")):
        return timeframe, "order_flow_v2_bounded"
    if any(token in name for token in ("true_cvd", "cvd_cumulative", "taker_buy_ratio", "taker_sell_ratio", "buy_sell_imbalance")):
        return timeframe, "order_flow_tier1"
    if any(token in name for token in ("vpt", "whale", "vol_per_trade", "large_trade_ratio")):
        return timeframe, "whale"
    if any(token in name for token in ("log_return", "realized_vol", "gk_vol", "atr", "adx", "vwap", "denoised")):
        return timeframe, "volatility_structure"
    return timeframe, "other"

def model_feature_columns_frame(feature_columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"position": range(len(feature_columns)), "feature": list(feature_columns)})

def _write_parquet_with_csv_fallback(frame: pd.DataFrame, path: Path) -> None:
    try:
        frame.to_parquet(path, index=False)
    except ImportError:
        frame.to_csv(path.with_suffix(".csv"), index=False)

def _metrics_at_threshold(labels: pd.Series, scores: pd.Series, threshold: float) -> dict[str, float]:
    y_true = labels.astype(int).to_numpy()
    y_score = scores.astype(float).to_numpy()
    y_pred = y_score >= threshold
    tp = int(((y_true == 1) & y_pred).sum())
    fp = int(((y_true == 0) & y_pred).sum())
    fn = int(((y_true == 1) & ~y_pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "pred_long_rate": float(y_pred.mean()),
    }

def _top_joined_tokens(values: pd.Series, *, limit: int = 8) -> str:
    seen: list[str] = []
    for value in values.dropna().astype(str):
        for token in value.split(","):
            token = token.strip()
            if token and token not in seen:
                seen.append(token)
            if len(seen) >= limit:
                return ",".join(seen)
    return ",".join(seen)

def _feature_target_rank_ic(frame: pd.DataFrame, feature: str, target_column: str) -> float:
    if feature not in frame.columns or target_column not in frame.columns:
        return np.nan
    pair = frame[[feature, target_column]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3:
        return np.nan
    return rank_ic(pair[feature], pair[target_column])

def _resolve_score_bands(bands: list[dict[str, Any]] | None, bins: int) -> list[dict[str, Any]]:
    max_bin = max(0, bins - 1)
    if bands is None:
        bands = [
            {"name": "top_10", "min_bin": max_bin, "max_bin": max_bin},
            {"name": "top_20", "min_bin": max(0, int(np.floor(bins * 0.80))), "max_bin": max_bin},
            {"name": "top_30", "min_bin": max(0, int(np.floor(bins * 0.70))), "max_bin": max_bin},
            {"name": "upper_half", "min_bin": max(0, int(np.floor(bins * 0.50))), "max_bin": max_bin},
            {
                "name": "mid_upper_40_90",
                "min_bin": max(0, int(np.floor(bins * 0.40))),
                "max_bin": max(0, max_bin - 1),
            },
        ]
    resolved = []
    for item in bands:
        name = str(item.get("name", f"bins_{item.get('min_bin')}_{item.get('max_bin')}"))
        min_bin = int(item.get("min_bin", max_bin))
        max_item_bin = int(item.get("max_bin", max_bin))
        min_bin = min(max(min_bin, 0), max_bin)
        max_item_bin = min(max(max_item_bin, 0), max_bin)
        if min_bin > max_item_bin:
            continue
        resolved.append({"name": name, "min_bin": min_bin, "max_bin": max_item_bin})
    return resolved

def _threshold_summary_mean(threshold_summary: pd.DataFrame | None, metric: str) -> float:
    if threshold_summary is None or threshold_summary.empty:
        return np.nan
    if "metric" not in threshold_summary.columns or "mean" not in threshold_summary.columns:
        return np.nan
    row = threshold_summary.loc[threshold_summary["metric"] == metric]
    if row.empty:
        return np.nan
    return float(row["mean"].iloc[0])

def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value

def _config_get(config: object, path: list[str], default: object) -> object:
    current = config
    for key in path:
        if isinstance(current, dict):
            if key not in current:
                return default
            current = current[key]
        else:
            if not hasattr(current, key):
                return default
            current = getattr(current, key)
    return current

def _summary_markdown(report: dict[str, Any], fold_metrics: pd.DataFrame) -> str:
    top_good = fold_metrics.sort_values("rank_ic", ascending=False).head(5)[["fold", "rank_ic"]]
    top_bad = fold_metrics.sort_values("rank_ic", ascending=True).head(5)[["fold", "rank_ic"]]
    threshold_selected = report.get("threshold_selected", {}) or {}
    threshold_constrained = report.get("threshold_constrained", {}) or {}

    def fmt(value: object) -> str:
        try:
            return f"{float(value):.6f}"
        except (TypeError, ValueError):
            return "nan"

    lines = [
        "# Phase 1 Diagnostics",
        "",
        f"Decision: {'PASS' if report.get('passed') else 'FAIL'}",
        f"Mean Rank IC: {fmt(report.get('mean_rank_ic'))}",
        f"Rank IC Std: {fmt(report.get('std_rank_ic'))}",
        f"Positive IC Fraction: {fmt(report.get('positive_ic_fraction'))}",
        f"Mean Long F1: {fmt(report.get('mean_long_f1'))}",
        f"Mean PRAUC: {fmt(report.get('mean_prauc'))}",
        f"Calibration Separation: {fmt(report.get('calibration_separation'))}",
    ]
    if threshold_selected:
        lines.extend(
            [
                f"Selected Threshold Mean: {fmt(threshold_selected.get('selected_threshold_mean'))}",
                f"Selected-Threshold Long F1: {fmt(threshold_selected.get('test_f1_at_selected_threshold'))}",
                f"Selected-Threshold Precision: {fmt(threshold_selected.get('test_precision_at_selected_threshold'))}",
                f"Selected-Threshold Recall: {fmt(threshold_selected.get('test_recall_at_selected_threshold'))}",
                f"Selected-Threshold Pred Long Rate: {fmt(threshold_selected.get('test_pred_long_rate_at_selected_threshold'))}",
            ]
        )
    if threshold_constrained:
        lines.extend(
            [
                f"Constrained Threshold Mean: {fmt(threshold_constrained.get('constrained_threshold_mean'))}",
                f"Constrained-Threshold Long F1: {fmt(threshold_constrained.get('test_f1_at_constrained_threshold'))}",
                f"Constrained-Threshold Precision: {fmt(threshold_constrained.get('test_precision_at_constrained_threshold'))}",
                f"Constrained-Threshold Recall: {fmt(threshold_constrained.get('test_recall_at_constrained_threshold'))}",
                f"Constrained-Threshold Pred Long Rate: {fmt(threshold_constrained.get('test_pred_long_rate_at_constrained_threshold'))}",
            ]
        )
    lines.extend(["", "## Checks"])
    checks = report.get("checks", {})
    lines.extend(f"- {name}: {value}" for name, value in checks.items())
    threshold_checks = report.get("checks_threshold_selected", {})
    if threshold_checks:
        lines.append("")
        lines.append("## Threshold-Selected Checks")
        lines.extend(f"- {name}: {value}" for name, value in threshold_checks.items())
    constrained_checks = report.get("checks_threshold_constrained", {})
    if constrained_checks:
        lines.append("")
        lines.append("## Threshold-Constrained Checks")
        lines.extend(f"- {name}: {value}" for name, value in constrained_checks.items())
    alerts = report.get("alerts", [])
    if alerts:
        lines.append("")
        lines.append("## Alerts")
        lines.extend(f"- {alert}" for alert in alerts)
    lines.append("")
    lines.append("## Best Folds")
    lines.extend(f"- fold {int(row.fold)}: {row.rank_ic:.6f}" for row in top_good.itertuples())
    lines.append("")
    lines.append("## Worst Folds")
    lines.extend(f"- fold {int(row.fold)}: {row.rank_ic:.6f}" for row in top_bad.itertuples())
    lines.append("")
    return "\n".join(lines)

def _diagnostic_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = RAW_COLUMNS | METADATA_COLUMNS | LABEL_COLUMNS
    excluded |= {
        "fold",
        "source_row_position",
        "forward_return",
        "prob_long",
        "prob_long_raw",
        "prob_long_calibrated",
    }
    excluded_prefixes = ("regime_prob_", "pred_")
    columns = []
    for column in frame.columns:
        if column in excluded or any(column.startswith(prefix) for prefix in excluded_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    return sorted(columns)

def _assign_score_bins(predictions: pd.DataFrame, *, score_column: str, bins: int) -> pd.DataFrame:
    frame = predictions.copy().replace([np.inf, -np.inf], np.nan).dropna(subset=["label", score_column])
    if frame.empty:
        return frame
    frame["score_bin"] = pd.qcut(
        frame[score_column].rank(method="first"),
        q=min(bins, len(frame)),
        labels=False,
        duplicates="drop",
    )
    return frame

