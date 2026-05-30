from __future__ import annotations

from __future__ import annotations
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from yenibot.diagnostics import (
    attach_threshold_summary_to_phase1_report,
    calibration_table,
    calibrate_split_probabilities_from_val,
    bad_fold_regime_diagnostics,
    experiment_ledger_diagnostics,
    feature_group_diagnostics,
    feature_profile_diagnostics,
    fold_diagnostics,
    mtf_leakage_diagnostics,
    phase1_report,
    recent_fold_diagnostics,
    regime_by_fold_diagnostics,
    regime_diagnostics,
    score_band_by_fold_diagnostics,
    score_band_diagnostics,
    score_band_summary_diagnostics,
    score_policy_grid_diagnostics,
    select_score_policy,
    score_lift_by_fold_diagnostics,
    score_lift_diagnostics,
    stationarity_policy_diagnostics,
    threshold_diagnostics,
    threshold_grid_diagnostics,
    threshold_grid_summary_diagnostics,
    threshold_summary_diagnostics,
)

_TRAINING_EXECUTION_KEYS = (
    "run_id_source",
    "training_executed_count",
    "training_skipped_count",
    "all_training_scopes_reused",
    "reused_training_scopes",
)


def _cfg(config: Any, path: list[str], default: Any = None) -> Any:
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

def _set_cfg(config: dict[str, Any], path: list[str], value: Any) -> None:
    current: dict[str, Any] = config
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value

def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value

def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")

def _table_markdown(title: str, frame: pd.DataFrame) -> str:
    lines = [f"# {title}", ""]
    if frame.empty:
        lines.append("No rows were produced.")
        return "\n".join(lines)
    lines.append("| " + " | ".join(frame.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(frame.columns)) + " |")
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in frame.columns) + " |")
    return "\n".join(lines)

def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base

def _numeric_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns or frame.empty:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.mean()) if not values.empty else np.nan

def _prediction_key_columns(frame: pd.DataFrame) -> list[str]:
    keys = []
    if "split" in frame.columns:
        keys.append("split")
    for column in ("fold", "timestamp", "source_row_position"):
        if column in frame.columns:
            keys.append(column)
    return keys

def _float(row: dict[str, Any], key: str, default: float = np.nan) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None

def _optional_gate_float(gates: dict[str, Any], key: str, default: float | None = None) -> float | None:
    value = gates.get(key, default)
    if value is None:
        return None
    return float(value)

def _metric_or(row: dict[str, Any], key: str, fallback: float) -> float:
    value = _float(row, key, np.nan)
    if np.isnan(value):
        return fallback
    return value

def _is_stability_scope(fold_scope: str) -> bool:
    fold_scope = str(fold_scope)
    return fold_scope == "full" or fold_scope.startswith("blend_")

def _diagnostic_candidate_type(fold_scope: str) -> str:
    return "blend" if str(fold_scope).startswith("blend_") else "profile"

def _entry_threshold_policy_frame(entry: dict[str, Any]) -> pd.DataFrame:
    diagnostics = entry.get("diagnostics", {}) or {}
    threshold_metrics = diagnostics.get("threshold_metrics")
    if threshold_metrics is None or threshold_metrics.empty:
        return pd.DataFrame()

    frame = threshold_metrics.copy()
    calibrated = diagnostics.get("calibrated_threshold_metrics")
    if calibrated is not None and not calibrated.empty and "fold" in calibrated.columns:
        calibrated_keep = [
            column
            for column in (
                "fold",
                "selected_threshold",
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
            )
            if column in calibrated.columns
        ]
        calibrated_frame = calibrated[calibrated_keep].rename(
            columns={column: f"calibrated_{column}" for column in calibrated_keep if column != "fold"}
        )
        frame = frame.merge(calibrated_frame, on="fold", how="left")

    source = _entry_official_threshold_source(entry)
    use_calibrated = source.startswith("calibrated_")
    source_base = source.replace("calibrated_", "", 1) if use_calibrated else source
    if "selected" in source_base:
        family = "selected"
    elif "constrained" in source_base:
        family = "constrained"
    else:
        family = "constrained"
        if not source:
            source = "validation_constrained_threshold"
    prefix = "calibrated_" if use_calibrated else ""

    metric_map = {
        "threshold": f"{prefix}{family}_threshold",
        "f1": f"{prefix}test_f1_at_{family}_threshold",
        "precision": f"{prefix}test_precision_at_{family}_threshold",
        "recall": f"{prefix}test_recall_at_{family}_threshold",
        "pred_rate": f"{prefix}test_pred_long_rate_at_{family}_threshold",
    }

    def metric_series(column: str) -> pd.Series:
        if column not in frame.columns:
            return pd.Series(np.nan, index=frame.index)
        return pd.to_numeric(frame[column], errors="coerce")

    frame["official_threshold_source"] = source
    frame["official_threshold_uses_calibration"] = bool(use_calibrated)
    frame["official_threshold"] = metric_series(metric_map["threshold"])
    frame["test_f1_at_official_threshold"] = metric_series(metric_map["f1"])
    frame["test_precision_at_official_threshold"] = metric_series(metric_map["precision"])
    frame["test_recall_at_official_threshold"] = metric_series(metric_map["recall"])
    frame["test_pred_long_rate_at_official_threshold"] = metric_series(metric_map["pred_rate"])
    return frame

def _rank_ic_for_frame(predictions: pd.DataFrame) -> float:
    if predictions.empty or "prob_long" not in predictions.columns or "forward_return" not in predictions.columns:
        return np.nan
    frame = predictions[["prob_long", "forward_return"]].copy()
    frame["prob_long"] = pd.to_numeric(frame["prob_long"], errors="coerce")
    frame["forward_return"] = pd.to_numeric(frame["forward_return"], errors="coerce")
    frame = frame.dropna()
    if len(frame) < 3 or frame["prob_long"].nunique() < 2 or frame["forward_return"].nunique() < 2:
        return np.nan
    return float(frame["prob_long"].corr(frame["forward_return"], method="spearman"))

def _score_ks_statistic(positive_scores: pd.Series, negative_scores: pd.Series) -> float:
    pos = pd.to_numeric(positive_scores, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    neg = pd.to_numeric(negative_scores, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    values = np.sort(np.unique(np.concatenate([pos, neg])))
    pos_sorted = np.sort(pos)
    neg_sorted = np.sort(neg)
    pos_cdf = np.searchsorted(pos_sorted, values, side="right") / len(pos_sorted)
    neg_cdf = np.searchsorted(neg_sorted, values, side="right") / len(neg_sorted)
    return float(np.max(np.abs(pos_cdf - neg_cdf)))

def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")

def _test_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if "split" in predictions.columns:
        return predictions[predictions["split"] == "test"].copy()
    return predictions.copy()

def _entry_official_threshold_source(entry: dict[str, Any]) -> str:
    row = (entry.get("diagnostics", {}) or {}).get("row", {}) or {}
    return str(row.get("official_threshold_source") or row.get("guarded_threshold_source") or "")

def _future_oos_monitor_state(config: dict[str, Any], latest_data_end: Any) -> dict[str, Any]:
    policy_review = _cfg(config, ["experiments", "policy_review"], {}) or {}
    monitor = policy_review.get("future_oos_monitor", {}) or {}
    status = str(policy_review.get("status", "")).lower()
    anchor_data_end = str(monitor.get("anchor_data_end", "") or "")
    latest_text = str(latest_data_end or "")
    new_bars_since_anchor = 0
    if anchor_data_end and latest_text:
        try:
            anchor_ts = pd.to_datetime(anchor_data_end, utc=True)
            latest_ts = pd.to_datetime(latest_text, utc=True)
            if pd.notna(anchor_ts) and pd.notna(latest_ts) and latest_ts > anchor_ts:
                new_bars_since_anchor = int((latest_ts - anchor_ts).total_seconds() // 3600)
        except (TypeError, ValueError):
            new_bars_since_anchor = 0

    min_new_bars = int(monitor.get("min_new_bars", 0) or 0)
    preferred_new_bars = int(monitor.get("preferred_new_bars", 0) or 0)
    future_oos_ready = bool(min_new_bars > 0 and new_bars_since_anchor >= min_new_bars)
    future_oos_preferred_ready = bool(preferred_new_bars > 0 and new_bars_since_anchor >= preferred_new_bars)
    allow_roll_forward = bool(monitor.get("allow_holdout_roll_forward", False))
    retired_or_failed = any(token in status for token in ("failed", "invalidated", "retired"))
    lock_active = bool(monitor.get("enabled", False)) and bool(anchor_data_end) and retired_or_failed and not allow_roll_forward
    if not bool(monitor.get("enabled", False)):
        next_action = "monitor_disabled"
    elif future_oos_ready:
        next_action = "future_oos_window_available"
    else:
        next_action = "wait_for_new_unseen_bars"

    return {
        "monitor_enabled": bool(monitor.get("enabled", False)),
        "anchor_run_id": str(monitor.get("anchor_run_id", "")),
        "anchor_data_end": anchor_data_end,
        "latest_available_data_end": latest_text,
        "new_bars_since_anchor": new_bars_since_anchor,
        "min_new_bars": min_new_bars,
        "preferred_new_bars": preferred_new_bars,
        "min_new_bars_remaining": max(0, min_new_bars - new_bars_since_anchor),
        "preferred_new_bars_remaining": max(0, preferred_new_bars - new_bars_since_anchor),
        "future_oos_ready": future_oos_ready,
        "future_oos_preferred_ready": future_oos_preferred_ready,
        "allow_holdout_roll_forward": allow_roll_forward,
        "holdout_roll_forward_locked": lock_active,
        "next_action": next_action,
    }

def _holdout_signal_pass_reasons(row: dict[str, Any], config: dict[str, Any]) -> list[str]:
    target_rank_ic = float(_cfg(config, ["validation", "target_rank_ic"], 0.03))
    reasons = []
    if float(row.get("mean_rank_ic", 0.0)) <= target_rank_ic:
        reasons.append("mean_rank_ic")
    if float(row.get("top_10_lift_global", 0.0)) <= 1.0:
        reasons.append("top_10_lift_global")
    if float(row.get("top_10_forward_return_global", 0.0)) <= 0.0:
        reasons.append("top_10_forward_return_global")
    if not bool(row.get("holdout_policy_pass", False)):
        reasons.append("holdout_policy")
    if float(row.get("calibration_separation", 0.0)) <= 0.0:
        reasons.append("calibration_separation")
    if not bool(row.get("mtf_leakage_passed", False)):
        reasons.append("mtf_leakage")
    if not bool(row.get("stationarity_policy_passed", True)):
        reasons.append("stationarity_policy")
    return reasons

def _holdout_threshold_pass_reasons(row: dict[str, Any], config: dict[str, Any]) -> list[str]:
    max_pred_long_rate = float(_cfg(config, ["validation", "threshold_checks", "max_pred_long_rate"], 0.70))
    min_long_f1 = float(_cfg(config, ["validation", "min_long_f1"], 0.45))
    reasons = []
    if float(row.get("holdout_cv_threshold_f1", 0.0)) <= min_long_f1:
        reasons.append("holdout_cv_threshold_f1")
    if float(row.get("holdout_cv_threshold_pred_long_rate", 1.0)) > max_pred_long_rate:
        reasons.append("holdout_cv_threshold_pred_long_rate")
    return reasons

def summarize_profile_predictions(
    predictions: pd.DataFrame,
    config: dict[str, Any],
    *,
    profile: str,
    feature_columns: list[str],
    fold_scope: str,
    promotable: bool | None = None,
    reject_reason: str = "",
) -> dict[str, Any]:
    from .profile_resolver import profile_config
    profile_cfg = profile_config(config, profile)
    test_predictions = _test_predictions(predictions)
    report = phase1_report(test_predictions, profile_cfg)
    calibration = calibration_table(
        test_predictions["label"],
        test_predictions["prob_long"],
        bins=int(_cfg(profile_cfg, ["validation", "calibration_bins"], 10)),
    )
    fold_metrics = fold_diagnostics(test_predictions)
    regime_metrics = regime_diagnostics(test_predictions)
    regime_by_fold = regime_by_fold_diagnostics(
        test_predictions,
        fold_metrics,
        bad_ic=float(_cfg(profile_cfg, ["validation", "bad_fold_ic_threshold"], -0.08)),
    )
    bad_fold_regime = bad_fold_regime_diagnostics(regime_by_fold)
    threshold_cfg = _cfg(profile_cfg, ["validation", "threshold_checks"], {}) or {}
    threshold_metrics = threshold_diagnostics(
        predictions,
        max_pred_long_rate=float(threshold_cfg.get("max_pred_long_rate", 0.70)),
        min_precision=float(threshold_cfg.get("min_precision", 0.30)),
    )
    threshold_summary = threshold_summary_diagnostics(threshold_metrics)
    report = attach_threshold_summary_to_phase1_report(report, threshold_summary, profile_cfg)
    calibrated_report = None
    calibrated_calibration = pd.DataFrame()
    calibrated_predictions = pd.DataFrame()
    calibrated_threshold_report = None
    calibrated_threshold_metrics = pd.DataFrame()
    calibrated_threshold_summary = pd.DataFrame()
    calibration_cfg = _cfg(profile_cfg, ["validation", "calibration"], {}) or {}
    if bool(calibration_cfg.get("enabled", False)):
        try:
            calibration_method = str(calibration_cfg.get("method", "isotonic"))
            calibrated_splits = calibrate_split_probabilities_from_val(
                predictions,
                method=calibration_method,
            )
            calibrated_predictions = calibrated_splits[calibrated_splits["split"] == "test"].copy()
            report_frame = calibrated_predictions.copy()
            report_frame["prob_long"] = report_frame["prob_long_calibrated"]
            calibrated_report = phase1_report(report_frame, profile_cfg)
            calibrated_calibration = calibration_table(
                report_frame["label"],
                report_frame["prob_long"],
                bins=int(_cfg(profile_cfg, ["validation", "calibration_bins"], 10)),
            )
            calibrated_threshold_metrics = threshold_diagnostics(
                calibrated_splits,
                score_column="prob_long_calibrated",
                max_pred_long_rate=float(threshold_cfg.get("max_pred_long_rate", 0.70)),
                min_precision=float(threshold_cfg.get("min_precision", 0.30)),
            )
            calibrated_threshold_summary = threshold_summary_diagnostics(calibrated_threshold_metrics)
            calibrated_threshold_report = attach_threshold_summary_to_phase1_report(
                dict(calibrated_report),
                calibrated_threshold_summary,
                profile_cfg,
            )
        except ValueError:
            calibrated_report = None
            calibrated_calibration = pd.DataFrame()
            calibrated_predictions = pd.DataFrame()
            calibrated_threshold_report = None
            calibrated_threshold_metrics = pd.DataFrame()
            calibrated_threshold_summary = pd.DataFrame()
    score_bins = int(_cfg(profile_cfg, ["validation", "score_lift_bins"], _cfg(profile_cfg, ["validation", "calibration_bins"], 10)))
    score_bands = _cfg(profile_cfg, ["validation", "score_bands"], None)
    policy_cfg = _cfg(profile_cfg, ["validation", "policy_selection"], {}) or {}
    threshold_caps = [float(value) for value in policy_cfg.get("threshold_caps", [0.30, 0.40, 0.50, 0.60, 0.70])]
    score_lift = score_lift_diagnostics(test_predictions, bins=score_bins)
    score_lift_by_fold = score_lift_by_fold_diagnostics(test_predictions, bins=score_bins)
    score_band_lift = score_band_diagnostics(test_predictions, bins=score_bins, bands=score_bands)
    score_band_by_fold = score_band_by_fold_diagnostics(test_predictions, bins=score_bins, bands=score_bands)
    score_band_summary = score_band_summary_diagnostics(score_band_by_fold)
    threshold_grid = threshold_grid_diagnostics(
        predictions,
        max_pred_long_rates=threshold_caps,
        min_precision=float(threshold_cfg.get("min_precision", 0.30)),
    )
    threshold_grid_summary = threshold_grid_summary_diagnostics(threshold_grid)
    score_policy_grid = score_policy_grid_diagnostics(
        predictions,
        bins=score_bins,
        bands=score_bands,
        threshold_caps=threshold_caps,
        min_precision=float(threshold_cfg.get("min_precision", 0.30)),
    )
    score_policy_selection = select_score_policy(score_policy_grid, profile_cfg)
    recent = recent_fold_diagnostics(
        fold_metrics,
        recent_folds=int(_cfg(profile_cfg, ["validation", "recent_folds"], 5)),
    )
    mtf = mtf_leakage_diagnostics(test_predictions)
    stationarity = stationarity_policy_diagnostics(feature_columns, profile_cfg)
    data_window = _frame_window(test_predictions)
    ledger = experiment_ledger_diagnostics(
        report=report,
        config=profile_cfg,
        feature_columns=feature_columns,
        fold_metrics=fold_metrics,
        recent_fold_summary=recent,
        threshold_summary=threshold_summary,
        score_band_lift=score_band_lift,
        score_lift_by_fold=score_lift_by_fold,
        score_band_summary=score_band_summary,
        fold_scope=fold_scope,
        data_start=data_window["data_start"],
        data_end=data_window["data_end"],
        promotable=promotable,
        reject_reason=reject_reason,
    )
    row = ledger.iloc[0].to_dict()
    calibrated_guard = (
        _threshold_guard_from_report(calibrated_threshold_report, prefix="calibrated")
        if calibrated_threshold_report
        else {}
    )
    official_threshold = _select_official_threshold_candidate(
        raw_report=report,
        raw_threshold_summary=threshold_summary,
        calibrated_threshold_report=calibrated_threshold_report,
        calibrated_threshold_summary=calibrated_threshold_summary,
        config=profile_cfg,
    )
    _apply_official_threshold_fields(
        row,
        ledger,
        official=official_threshold,
        calibrated=calibrated_guard,
    )
    min_long_f1 = float(_cfg(profile_cfg, ["validation", "min_long_f1"], 0.45))
    threshold_cfg = _cfg(profile_cfg, ["validation", "threshold_checks"], {}) or {}
    max_pred_long_rate = float(threshold_cfg.get("max_pred_long_rate", 0.70))
    official_f1 = _optional_float(row.get("test_f1_at_official_threshold"))
    official_pred_rate = _optional_float(row.get("test_pred_long_rate_at_official_threshold"))
    official_checks = dict(report.get("checks", {}) or {})
    official_checks["long_f1"] = bool(official_f1 is not None and official_f1 > min_long_f1)
    official_checks["threshold_pred_long_rate"] = bool(
        official_pred_rate is not None and official_pred_rate <= max_pred_long_rate
    )
    row["passed_phase1_official_threshold"] = all(bool(value) for value in official_checks.values())
    ledger.loc[:, "passed_phase1_official_threshold"] = row["passed_phase1_official_threshold"]
    row["mtf_leakage_passed"] = bool(mtf.empty or mtf["passed"].all())
    row["stationarity_policy_passed"] = bool(stationarity.empty or stationarity["passed"].all())
    row["fold_count"] = int(fold_metrics["fold"].nunique()) if not fold_metrics.empty else 0
    return {
        "report": report,
        "calibration": calibration,
        "calibrated_report": calibrated_report,
        "calibrated_calibration": calibrated_calibration,
        "calibrated_predictions": calibrated_predictions,
        "calibrated_threshold_report": calibrated_threshold_report,
        "calibrated_threshold_metrics": calibrated_threshold_metrics,
        "calibrated_threshold_summary": calibrated_threshold_summary,
        "fold_metrics": fold_metrics,
        "regime_metrics": regime_metrics,
        "regime_by_fold": regime_by_fold,
        "bad_fold_regime": bad_fold_regime,
        "threshold_metrics": threshold_metrics,
        "threshold_summary": threshold_summary,
        "threshold_grid": threshold_grid,
        "threshold_grid_summary": threshold_grid_summary,
        "score_lift": score_lift,
        "score_lift_by_fold": score_lift_by_fold,
        "score_band_lift": score_band_lift,
        "score_band_by_fold": score_band_by_fold,
        "score_band_summary": score_band_summary,
        "score_policy_grid": score_policy_grid,
        "score_policy_selection": score_policy_selection,
        "recent_fold_summary": recent,
        "mtf_leakage": mtf,
        "stationarity_policy": stationarity,
        "feature_groups": feature_group_diagnostics(feature_columns),
        "feature_profile": feature_profile_diagnostics(feature_columns, profile_cfg),
        "ledger": ledger,
        "row": row,
    }

def _frame_window(frame: pd.DataFrame) -> dict[str, str]:
    if "timestamp" not in frame.columns or frame.empty:
        return {"data_start": "", "data_end": ""}
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    return {"data_start": str(timestamps.min()), "data_end": str(timestamps.max())}

def _threshold_guard_from_report(report: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    guarded = report.get("threshold_guarded", {}) or {}
    source = str(guarded.get("threshold_source", ""))
    if prefix and source:
        source = f"{prefix}_{source}"
    return {
        "threshold_source": source,
        "reject_reason": str(guarded.get("reject_reason", "")),
        "threshold_mean": guarded.get("threshold_mean", np.nan),
        "f1": guarded.get("test_f1_at_guarded_threshold", np.nan),
        "precision": guarded.get("test_precision_at_guarded_threshold", np.nan),
        "recall": guarded.get("test_recall_at_guarded_threshold", np.nan),
        "pred_long_rate": guarded.get("test_pred_long_rate_at_guarded_threshold", np.nan),
        "passed": bool(report.get("passed_threshold_guarded", False)),
    }

def _select_official_threshold_candidate(
    *,
    raw_report: dict[str, Any],
    raw_threshold_summary: pd.DataFrame | None,
    calibrated_threshold_report: dict[str, Any] | None,
    calibrated_threshold_summary: pd.DataFrame | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    threshold_cfg = _cfg(config, ["validation", "threshold_checks"], {}) or {}
    max_pred_long_rate = float(threshold_cfg.get("max_pred_long_rate", 0.70))
    min_precision = float(threshold_cfg.get("min_precision", 0.30))
    raw_candidate = _threshold_guard_from_report(raw_report)
    raw_candidate["candidate_order"] = 0
    raw_candidate["selection_score"] = _threshold_selection_score(
        raw_threshold_summary,
        str(raw_candidate.get("threshold_source", "")),
    )
    candidates = [raw_candidate]
    if calibrated_threshold_report:
        calibrated_candidate = _threshold_guard_from_report(calibrated_threshold_report, prefix="calibrated")
        calibrated_candidate["candidate_order"] = 1
        calibrated_candidate["selection_score"] = _threshold_selection_score(
            calibrated_threshold_summary,
            str(calibrated_candidate.get("threshold_source", "")),
        )
        candidates.append(calibrated_candidate)
    guarded_candidates = [
        candidate
        for candidate in candidates
        if _threshold_candidate_is_guarded(
            candidate,
            max_pred_long_rate=max_pred_long_rate,
            min_precision=min_precision,
        )
    ]
    if guarded_candidates:
        selected = max(
            guarded_candidates,
            key=lambda item: (
                _optional_float(item.get("selection_score")) or -np.inf,
                -int(item.get("candidate_order", 999) or 999),
            ),
        )
    else:
        selected = candidates[0]
    selected = dict(selected)
    selected["uses_calibration"] = str(selected.get("threshold_source", "")).startswith("calibrated_")
    selected["candidate_count"] = len(candidates)
    selected["guarded_candidate_count"] = len(guarded_candidates)
    return selected

def _apply_official_threshold_fields(
    row: dict[str, Any],
    ledger: pd.DataFrame,
    *,
    official: dict[str, Any],
    calibrated: dict[str, Any] | None = None,
) -> None:
    calibrated = calibrated or {}
    additions = {
        "official_threshold_source": str(official.get("threshold_source", "")),
        "official_threshold_reason": str(official.get("reject_reason", "")),
        "official_threshold_mean": official.get("threshold_mean", np.nan),
        "test_f1_at_official_threshold": official.get("f1", np.nan),
        "test_precision_at_official_threshold": official.get("precision", np.nan),
        "test_recall_at_official_threshold": official.get("recall", np.nan),
        "test_pred_long_rate_at_official_threshold": official.get("pred_long_rate", np.nan),
        "official_threshold_uses_calibration": bool(official.get("uses_calibration", False)),
        "official_threshold_candidate_count": int(official.get("candidate_count", 1) or 1),
        "official_threshold_guarded_candidate_count": int(official.get("guarded_candidate_count", 0) or 0),
        "official_threshold_selection_score": official.get("selection_score", np.nan),
        "calibrated_guarded_threshold_source": str(calibrated.get("threshold_source", "")),
        "calibrated_guarded_threshold_reason": str(calibrated.get("reject_reason", "")),
        "calibrated_guarded_threshold_mean": calibrated.get("threshold_mean", np.nan),
        "test_f1_at_calibrated_guarded_threshold": calibrated.get("f1", np.nan),
        "test_precision_at_calibrated_guarded_threshold": calibrated.get("precision", np.nan),
        "test_recall_at_calibrated_guarded_threshold": calibrated.get("recall", np.nan),
        "test_pred_long_rate_at_calibrated_guarded_threshold": calibrated.get("pred_long_rate", np.nan),
    }
    row.update(additions)
    for column, value in additions.items():
        ledger.loc[:, column] = value

def _threshold_selection_score(
    threshold_summary: pd.DataFrame | None,
    source: str,
) -> float:
    if "constrained" in str(source):
        score = _threshold_summary_metric(threshold_summary, "source_constrained_f1")
        if np.isfinite(score):
            return score
    score = _threshold_summary_metric(threshold_summary, "source_best_f1")
    if np.isfinite(score):
        return score
    return np.nan

def _threshold_candidate_is_guarded(
    candidate: dict[str, Any],
    *,
    max_pred_long_rate: float,
    min_precision: float,
) -> bool:
    f1 = _optional_float(candidate.get("f1"))
    precision = _optional_float(candidate.get("precision"))
    pred_rate = _optional_float(candidate.get("pred_long_rate"))
    return bool(
        f1 is not None
        and precision is not None
        and pred_rate is not None
        and pred_rate <= max_pred_long_rate
        and precision >= min_precision
    )

def _threshold_summary_metric(threshold_summary: pd.DataFrame | None, metric: str) -> float:
    if threshold_summary is None or threshold_summary.empty:
        return np.nan
    if "metric" not in threshold_summary.columns or "mean" not in threshold_summary.columns:
        return np.nan
    row = threshold_summary.loc[threshold_summary["metric"].astype(str) == str(metric)]
    if row.empty:
        return np.nan
    return _float(row.iloc[0].to_dict(), "mean")

