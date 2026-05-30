from __future__ import annotations

from __future__ import annotations
import json
import zipfile
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from yenibot.features.builder import (
    resolve_feature_profile,
)

from .utils import (
    classify_feature_column, model_feature_columns_frame, _config_get, _threshold_summary_mean,
    _json_safe, _summary_markdown, _write_parquet_with_csv_fallback
)
from .fold_metrics import recent_fold_diagnostics, good_bad_fold_summary
from .threshold_analysis import (
    threshold_summary_diagnostics
)
from .score_analysis import (
    score_lift_diagnostics, score_band_diagnostics, score_band_by_fold_diagnostics,
    score_band_summary_diagnostics, score_lift_by_fold_diagnostics, stationarity_policy_diagnostics
)
from .fold_analysis import (
    regime_by_fold_diagnostics, bad_fold_regime_diagnostics,
    bad_fold_feature_forensics, bad_fold_feature_forensics_summary,
    bad_fold_group_forensics, bad_fold_group_forensics_summary
)

def feature_group_diagnostics(feature_columns: list[str]) -> pd.DataFrame:
    rows = []
    for position, feature in enumerate(feature_columns):
        timeframe, family = classify_feature_column(feature)
        rows.append({"position": position, "feature": feature, "timeframe": timeframe, "family": family})
    if not rows:
        return pd.DataFrame(columns=["position", "feature", "timeframe", "family", "count"])
    frame = pd.DataFrame(rows)
    counts = frame.groupby(["timeframe", "family"], as_index=False).agg(count=("feature", "size"))
    return frame.merge(counts, on=["timeframe", "family"], how="left")

def feature_group_importance_summary(importance: pd.DataFrame) -> pd.DataFrame:
    if importance is None or importance.empty or "feature" not in importance.columns:
        return pd.DataFrame()
    frame = importance.copy()
    groups = frame["feature"].map(classify_feature_column)
    frame["timeframe"] = [item[0] for item in groups]
    frame["family"] = [item[1] for item in groups]
    value_column = "rank_ic_drop"
    grouped = frame.groupby(["timeframe", "family"], as_index=False).agg(
        features=("feature", "nunique"),
        rows=("feature", "size"),
        mean_rank_ic_drop=(value_column, "mean"),
        median_rank_ic_drop=(value_column, "median"),
        min_rank_ic_drop=(value_column, "min"),
        max_rank_ic_drop=(value_column, "max"),
        positive_drop_rate=(value_column, lambda values: float((values > 0).mean())),
        total_positive_drop=(value_column, lambda values: float(values[values > 0].sum())),
    )
    return grouped.sort_values(["total_positive_drop", "mean_rank_ic_drop"], ascending=False).reset_index(drop=True)

def feature_profile_diagnostics(feature_columns: list[str], config: dict[str, Any] | None = None) -> pd.DataFrame:
    if config is None:
        return pd.DataFrame()
    profile = resolve_feature_profile(config)
    rows = [
        {
            "check": "active_feature_profile",
            "pattern": str(profile.get("name")),
            "matched_count": len(feature_columns),
            "matched_features": "",
        }
    ]
    for pattern in list(profile.get("include_patterns", []) or []):
        matches = sorted(column for column in feature_columns if fnmatch(column, pattern))
        rows.append(
            {
                "check": "profile_include_pattern",
                "pattern": pattern,
                "matched_count": len(matches),
                "matched_features": ",".join(matches),
            }
        )
    for pattern in list(profile.get("exclude_patterns", []) or []):
        matches = sorted(column for column in feature_columns if fnmatch(column, pattern))
        rows.append(
            {
                "check": "profile_exclude_pattern_absent",
                "pattern": pattern,
                "matched_count": len(matches),
                "matched_features": ",".join(matches),
            }
        )
    return pd.DataFrame(rows)

def experiment_ledger_diagnostics(
    *,
    report: dict[str, Any],
    config: dict[str, Any] | None = None,
    feature_columns: list[str] | None = None,
    fold_metrics: pd.DataFrame | None = None,
    recent_fold_summary: pd.DataFrame | None = None,
    threshold_summary: pd.DataFrame | None = None,
    score_band_lift: pd.DataFrame | None = None,
    score_lift_by_fold: pd.DataFrame | None = None,
    score_band_summary: pd.DataFrame | None = None,
    fold_scope: str = "",
    data_start: str = "",
    data_end: str = "",
    promotable: bool | None = None,
    reject_reason: str = "",
    timestamp: str | None = None,
) -> pd.DataFrame:
    def safe_float(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return np.nan
        return number if np.isfinite(number) else np.nan

    profile_name = ""
    if config is not None:
        profile_name = str(resolve_feature_profile(config).get("name") or "")
    recent_rank_ic_mean = np.nan
    if recent_fold_summary is not None and not recent_fold_summary.empty:
        recent_row = recent_fold_summary.loc[recent_fold_summary["metric"] == "rank_ic"]
        if not recent_row.empty:
            recent_rank_ic_mean = float(recent_row["recent_mean"].iloc[0])
    recent_rank_ic_min = np.nan
    if recent_fold_summary is not None and not recent_fold_summary.empty:
        recent_row = recent_fold_summary.loc[recent_fold_summary["metric"] == "rank_ic"]
        if not recent_row.empty and "recent_min" in recent_row:
            recent_rank_ic_min = float(recent_row["recent_min"].iloc[0])

    negative_ic_count = np.nan
    negative_ic_fraction = np.nan
    worst_5_rank_ic_mean = np.nan
    rank_ic_cvar_20 = np.nan
    bad_fold_rank_ic_mean = np.nan
    top_10_bad_fold_lift_mean = np.nan
    if fold_metrics is not None and not fold_metrics.empty and "rank_ic" in fold_metrics.columns:
        fold_frame = fold_metrics.copy()
        rank_ic = pd.to_numeric(fold_frame["rank_ic"], errors="coerce").dropna().sort_values()
        if not rank_ic.empty:
            negative_ic_count = int((rank_ic < 0).sum())
            negative_ic_fraction = float((rank_ic < 0).mean())
            worst_5_rank_ic_mean = float(rank_ic.head(min(5, len(rank_ic))).mean())
            cvar_count = max(1, int(np.ceil(len(rank_ic) * 0.20)))
            rank_ic_cvar_20 = float(rank_ic.head(cvar_count).mean())
            bad_ic_threshold = float(_config_get(config or {}, ["validation", "bad_fold_ic_threshold"], -0.08))
            bad_fold_rows = fold_frame[pd.to_numeric(fold_frame["rank_ic"], errors="coerce") <= bad_ic_threshold]
            if not bad_fold_rows.empty:
                bad_fold_rank_ic_mean = float(pd.to_numeric(bad_fold_rows["rank_ic"], errors="coerce").mean())
                if score_lift_by_fold is not None and not score_lift_by_fold.empty:
                    lift_frame = score_lift_by_fold.copy()
                    bad_lift = lift_frame[lift_frame["fold"].isin(bad_fold_rows["fold"])]
                    if "top_lift_vs_base" in bad_lift.columns and not bad_lift.empty:
                        top_10_bad_fold_lift_mean = float(
                            pd.to_numeric(bad_lift["top_lift_vs_base"], errors="coerce").mean()
                        )
    top_10_lift = np.nan
    top_10_lift_fold_mean = np.nan
    top_10_lift_global = np.nan
    top_10_positive_lift_fold_rate = np.nan
    top_10_forward_return_fold_mean = np.nan
    top_10_forward_return_global = np.nan
    if score_band_summary is not None and not score_band_summary.empty:
        top_row = score_band_summary.loc[score_band_summary["band"] == "top_10"]
        if not top_row.empty:
            top_10_lift_fold_mean = float(top_row["mean_lift_vs_base"].iloc[0])
            top_10_lift = top_10_lift_fold_mean
            if "positive_lift_fold_rate" in top_row:
                top_10_positive_lift_fold_rate = float(top_row["positive_lift_fold_rate"].iloc[0])
            if "mean_forward_return" in top_row:
                top_10_forward_return_fold_mean = float(top_row["mean_forward_return"].iloc[0])
    if score_band_lift is not None and not score_band_lift.empty:
        top_row = score_band_lift.loc[score_band_lift["band"] == "top_10"]
        if not top_row.empty:
            if "lift_vs_base" in top_row:
                top_10_lift_global = float(top_row["lift_vs_base"].iloc[0])
            if "mean_forward_return" in top_row:
                top_10_forward_return_global = float(top_row["mean_forward_return"].iloc[0])
    selected_threshold_mean = _threshold_summary_mean(threshold_summary, "selected_threshold")
    test_f1_at_selected_threshold = _threshold_summary_mean(threshold_summary, "test_f1_at_selected_threshold")
    test_precision_at_selected_threshold = _threshold_summary_mean(
        threshold_summary,
        "test_precision_at_selected_threshold",
    )
    test_recall_at_selected_threshold = _threshold_summary_mean(threshold_summary, "test_recall_at_selected_threshold")
    test_pred_long_rate_at_selected_threshold = _threshold_summary_mean(
        threshold_summary,
        "test_pred_long_rate_at_selected_threshold",
    )
    constrained_threshold_mean = _threshold_summary_mean(threshold_summary, "constrained_threshold")
    test_f1_at_constrained_threshold = _threshold_summary_mean(threshold_summary, "test_f1_at_constrained_threshold")
    test_precision_at_constrained_threshold = _threshold_summary_mean(
        threshold_summary,
        "test_precision_at_constrained_threshold",
    )
    test_recall_at_constrained_threshold = _threshold_summary_mean(
        threshold_summary,
        "test_recall_at_constrained_threshold",
    )
    test_pred_long_rate_at_constrained_threshold = _threshold_summary_mean(
        threshold_summary,
        "test_pred_long_rate_at_constrained_threshold",
    )
    test_oracle_best_f1 = _threshold_summary_mean(threshold_summary, "test_oracle_best_f1")
    test_f1_at_050 = _threshold_summary_mean(threshold_summary, "test_f1_at_050")
    selected_threshold_passed = bool(report.get("passed_threshold_selected", False))
    constrained_threshold_passed = bool(report.get("passed_threshold_constrained", False))
    guarded = report.get("threshold_guarded", {}) or {}
    guarded_threshold_passed = bool(report.get("passed_threshold_guarded", False))
    return pd.DataFrame(
        [
            {
                "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
                "profile": profile_name,
                "feature_count": int(len(feature_columns or [])),
                "fold_scope": fold_scope,
                "data_start": data_start,
                "data_end": data_end,
                "mean_rank_ic": float(report.get("mean_rank_ic", np.nan)),
                "std_rank_ic": float(report.get("std_rank_ic", np.nan)),
                "positive_ic_fraction": float(report.get("positive_ic_fraction", np.nan)),
                "mean_long_f1": float(report.get("mean_long_f1", np.nan)),
                "selected_threshold_mean": selected_threshold_mean,
                "test_f1_at_selected_threshold": test_f1_at_selected_threshold,
                "test_precision_at_selected_threshold": test_precision_at_selected_threshold,
                "test_recall_at_selected_threshold": test_recall_at_selected_threshold,
                "test_pred_long_rate_at_selected_threshold": test_pred_long_rate_at_selected_threshold,
                "constrained_threshold_mean": constrained_threshold_mean,
                "test_f1_at_constrained_threshold": test_f1_at_constrained_threshold,
                "test_precision_at_constrained_threshold": test_precision_at_constrained_threshold,
                "test_recall_at_constrained_threshold": test_recall_at_constrained_threshold,
                "test_pred_long_rate_at_constrained_threshold": test_pred_long_rate_at_constrained_threshold,
                "guarded_threshold_source": str(guarded.get("threshold_source", "")),
                "guarded_threshold_reason": str(guarded.get("reject_reason", "")),
                "guarded_threshold_mean": safe_float(guarded.get("threshold_mean", np.nan)),
                "test_f1_at_guarded_threshold": safe_float(guarded.get("test_f1_at_guarded_threshold", np.nan)),
                "test_precision_at_guarded_threshold": safe_float(
                    guarded.get("test_precision_at_guarded_threshold", np.nan)
                ),
                "test_recall_at_guarded_threshold": safe_float(
                    guarded.get("test_recall_at_guarded_threshold", np.nan)
                ),
                "test_pred_long_rate_at_guarded_threshold": safe_float(
                    guarded.get("test_pred_long_rate_at_guarded_threshold", np.nan)
                ),
                "test_oracle_best_f1": test_oracle_best_f1,
                "test_f1_at_050": test_f1_at_050,
                "mean_prauc": float(report.get("mean_prauc", np.nan)),
                "calibration_separation": float(report.get("calibration_separation", np.nan)),
                "recent_rank_ic_mean": recent_rank_ic_mean,
                "recent_rank_ic_min": recent_rank_ic_min,
                "negative_ic_count": negative_ic_count,
                "negative_ic_fraction": negative_ic_fraction,
                "worst_5_rank_ic_mean": worst_5_rank_ic_mean,
                "rank_ic_cvar_20": rank_ic_cvar_20,
                "bad_fold_rank_ic_mean": bad_fold_rank_ic_mean,
                "top_10_lift": top_10_lift,
                "top_10_lift_fold_mean": top_10_lift_fold_mean,
                "top_10_lift_global": top_10_lift_global,
                "top_10_positive_lift_fold_rate": top_10_positive_lift_fold_rate,
                "top_10_bad_fold_lift_mean": top_10_bad_fold_lift_mean,
                "top_10_forward_return_fold_mean": top_10_forward_return_fold_mean,
                "top_10_forward_return_global": top_10_forward_return_global,
                "passed_phase1": bool(report.get("passed", False)),
                "passed_phase1_selected_threshold": selected_threshold_passed,
                "passed_phase1_constrained_threshold": constrained_threshold_passed,
                "passed_phase1_guarded_threshold": guarded_threshold_passed,
                "promotable": bool(promotable) if promotable is not None else bool(report.get("passed", False)),
                "reject_reason": reject_reason,
            }
        ]
    )

def mtf_leakage_diagnostics(predictions: pd.DataFrame, *, htf_hours: int = 4) -> pd.DataFrame:
    required = {"timestamp", "4h_source_timestamp", "4h_available_timestamp"}
    if not required.issubset(predictions.columns):
        missing = sorted(required - set(predictions.columns))
        return pd.DataFrame([{"check": "mtf_alignment", "passed": False, "detail": f"missing columns: {missing}"}])

    timestamp = pd.to_datetime(predictions["timestamp"], utc=True)
    source = pd.to_datetime(predictions["4h_source_timestamp"], utc=True)
    available = pd.to_datetime(predictions["4h_available_timestamp"], utc=True)
    expected_available = source + pd.Timedelta(hours=htf_hours)
    availability_violations = int((available > timestamp).sum())
    shift_violations = int((available != expected_available).sum())
    rows = [
        {
            "check": "4h_available_lte_primary_timestamp",
            "passed": availability_violations == 0,
            "violations": availability_violations,
            "detail": "4H feature availability must never be after the 1H row timestamp.",
        },
        {
            "check": "4h_source_plus_period_equals_available",
            "passed": shift_violations == 0,
            "violations": shift_violations,
            "detail": f"4H source timestamps must be shifted forward by {htf_hours} hours.",
        },
    ]
    return pd.DataFrame(rows)

def write_phase1_diagnostic_bundle(
    *,
    output_dir: str | Path,
    report: dict[str, Any],
    predictions: pd.DataFrame,
    calibration: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    regime_metrics: pd.DataFrame | None = None,
    importance: pd.DataFrame | None = None,
    tsne: pd.DataFrame | None = None,
    calibrated_report: dict[str, Any] | None = None,
    calibrated_calibration: pd.DataFrame | None = None,
    calibrated_predictions: pd.DataFrame | None = None,
    threshold_metrics: pd.DataFrame | None = None,
    threshold_summary: pd.DataFrame | None = None,
    mtf_leakage: pd.DataFrame | None = None,
    regime_by_fold: pd.DataFrame | None = None,
    bad_fold_regime: pd.DataFrame | None = None,
    feature_audit: pd.DataFrame | None = None,
    stationarity_policy: pd.DataFrame | None = None,
    score_lift: pd.DataFrame | None = None,
    score_lift_by_fold: pd.DataFrame | None = None,
    score_band_lift: pd.DataFrame | None = None,
    score_band_by_fold: pd.DataFrame | None = None,
    score_band_summary: pd.DataFrame | None = None,
    recent_fold_summary: pd.DataFrame | None = None,
    feature_groups: pd.DataFrame | None = None,
    feature_profile: pd.DataFrame | None = None,
    feature_group_importance: pd.DataFrame | None = None,
    group_permutation_importance: pd.DataFrame | None = None,
    bad_fold_feature_forensics_table: pd.DataFrame | None = None,
    bad_fold_group_forensics_table: pd.DataFrame | None = None,
    experiment_ledger: pd.DataFrame | None = None,
    model_feature_columns: list[str] | None = None,
    config: dict[str, Any] | None = None,
    prefix: str = "phase1_diagnostics",
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    bundle_dir = output_path / f"{prefix}_{stamp}"
    bundle_dir.mkdir(parents=True, exist_ok=False)

    serializable_report = _json_safe(report)
    (bundle_dir / "phase1_report.json").write_text(
        json.dumps(serializable_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if config is not None:
        (bundle_dir / "config.json").write_text(
            json.dumps(_json_safe(config), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    (bundle_dir / "summary.md").write_text(
        _summary_markdown(serializable_report, fold_metrics),
        encoding="utf-8",
    )

    _write_parquet_with_csv_fallback(predictions, bundle_dir / "test_predictions.parquet")
    calibration.to_csv(bundle_dir / "calibration.csv", index=False)
    fold_metrics.to_csv(bundle_dir / "fold_metrics.csv", index=False)
    if model_feature_columns is not None:
        model_feature_columns_frame(model_feature_columns).to_csv(bundle_dir / "model_feature_columns.csv", index=False)
        if feature_groups is None:
            feature_groups = feature_group_diagnostics(model_feature_columns)
        if feature_profile is None:
            feature_profile = feature_profile_diagnostics(model_feature_columns, config)
        if stationarity_policy is None:
            stationarity_policy = stationarity_policy_diagnostics(model_feature_columns, config)
    if regime_metrics is not None and not regime_metrics.empty:
        regime_metrics.to_csv(bundle_dir / "regime_metrics.csv", index=False)
    if regime_by_fold is None and regime_metrics is not None:
        regime_by_fold = regime_by_fold_diagnostics(
            predictions,
            fold_metrics,
            bad_ic=float(_config_get(config or {}, ["validation", "bad_fold_ic_threshold"], -0.08)),
        )
    if regime_by_fold is not None and not regime_by_fold.empty:
        regime_by_fold.to_csv(bundle_dir / "regime_by_fold.csv", index=False)
        if bad_fold_regime is None:
            bad_fold_regime = bad_fold_regime_diagnostics(regime_by_fold)
    if bad_fold_regime is not None and not bad_fold_regime.empty:
        bad_fold_regime.to_csv(bundle_dir / "bad_fold_regime_diagnostics.csv", index=False)
    if importance is not None and not importance.empty:
        importance.to_csv(bundle_dir / "permutation_importance.csv", index=False)
        if feature_group_importance is None:
            feature_group_importance = feature_group_importance_summary(importance)
    if tsne is not None and not tsne.empty:
        _write_parquet_with_csv_fallback(tsne, bundle_dir / "tsne_embeddings.parquet")
    if calibrated_report is not None:
        (bundle_dir / "calibrated_phase1_report.json").write_text(
            json.dumps(_json_safe(calibrated_report), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if calibrated_calibration is not None and not calibrated_calibration.empty:
        calibrated_calibration.to_csv(bundle_dir / "calibrated_calibration.csv", index=False)
    if calibrated_predictions is not None and not calibrated_predictions.empty:
        _write_parquet_with_csv_fallback(calibrated_predictions, bundle_dir / "calibrated_test_predictions.parquet")
    if threshold_metrics is not None and not threshold_metrics.empty:
        threshold_metrics.to_csv(bundle_dir / "threshold_metrics.csv", index=False)
        if threshold_summary is None:
            threshold_summary = threshold_summary_diagnostics(threshold_metrics)
    if threshold_summary is not None and not threshold_summary.empty:
        threshold_summary.to_csv(bundle_dir / "threshold_summary.csv", index=False)
    if mtf_leakage is not None and not mtf_leakage.empty:
        mtf_leakage.to_csv(bundle_dir / "mtf_leakage.csv", index=False)
    if feature_audit is not None and not feature_audit.empty:
        feature_audit.to_csv(bundle_dir / "good_bad_feature_audit.csv", index=False)
    if stationarity_policy is not None and not stationarity_policy.empty:
        stationarity_policy.to_csv(bundle_dir / "stationarity_policy.csv", index=False)
    if score_lift is None and config is not None:
        bins = int(_config_get(config, ["validation", "calibration_bins"], 10))
        score_lift = score_lift_diagnostics(predictions, bins=bins)
    if score_lift is not None and not score_lift.empty:
        score_lift.to_csv(bundle_dir / "score_lift.csv", index=False)
    if score_lift_by_fold is None and config is not None:
        bins = int(_config_get(config, ["validation", "score_lift_bins"], _config_get(config, ["validation", "calibration_bins"], 10)))
        score_lift_by_fold = score_lift_by_fold_diagnostics(predictions, bins=bins)
    if score_lift_by_fold is not None and not score_lift_by_fold.empty:
        score_lift_by_fold.to_csv(bundle_dir / "score_lift_by_fold.csv", index=False)
    score_bands = _config_get(config, ["validation", "score_bands"], None) if config is not None else None
    if score_band_lift is None and config is not None:
        bins = int(_config_get(config, ["validation", "score_lift_bins"], _config_get(config, ["validation", "calibration_bins"], 10)))
        score_band_lift = score_band_diagnostics(predictions, bins=bins, bands=score_bands)
    if score_band_lift is not None and not score_band_lift.empty:
        score_band_lift.to_csv(bundle_dir / "score_band_lift.csv", index=False)
    if score_band_by_fold is None and config is not None:
        bins = int(_config_get(config, ["validation", "score_lift_bins"], _config_get(config, ["validation", "calibration_bins"], 10)))
        score_band_by_fold = score_band_by_fold_diagnostics(predictions, bins=bins, bands=score_bands)
    if score_band_by_fold is not None and not score_band_by_fold.empty:
        score_band_by_fold.to_csv(bundle_dir / "score_band_by_fold.csv", index=False)
        if score_band_summary is None:
            score_band_summary = score_band_summary_diagnostics(score_band_by_fold)
    if score_band_summary is not None and not score_band_summary.empty:
        score_band_summary.to_csv(bundle_dir / "score_band_summary.csv", index=False)
    if recent_fold_summary is None and config is not None:
        recent_fold_count = int(_config_get(config, ["validation", "recent_folds"], 5))
        recent_fold_summary = recent_fold_diagnostics(fold_metrics, recent_folds=recent_fold_count)
    if recent_fold_summary is not None and not recent_fold_summary.empty:
        recent_fold_summary.to_csv(bundle_dir / "recent_fold_summary.csv", index=False)
    if feature_groups is not None and not feature_groups.empty:
        feature_groups.to_csv(bundle_dir / "feature_groups.csv", index=False)
    if feature_profile is not None and not feature_profile.empty:
        feature_profile.to_csv(bundle_dir / "feature_profile.csv", index=False)
    if feature_group_importance is not None and not feature_group_importance.empty:
        feature_group_importance.to_csv(bundle_dir / "feature_group_importance.csv", index=False)
    if group_permutation_importance is not None and not group_permutation_importance.empty:
        group_permutation_importance.to_csv(bundle_dir / "group_permutation_importance.csv", index=False)
    if bad_fold_feature_forensics_table is None:
        bad_fold_feature_forensics_table = bad_fold_feature_forensics(
            predictions,
            fold_metrics,
            feature_columns=model_feature_columns,
        )
    if bad_fold_feature_forensics_table is not None and not bad_fold_feature_forensics_table.empty:
        bad_fold_feature_forensics_table.to_csv(bundle_dir / "bad_fold_feature_forensics.csv", index=False)
        bad_fold_feature_forensics_summary(bad_fold_feature_forensics_table).to_csv(
            bundle_dir / "bad_fold_feature_forensics_summary.csv",
            index=False,
        )
    if bad_fold_group_forensics_table is None:
        bad_fold_group_forensics_table = bad_fold_group_forensics(
            predictions,
            fold_metrics,
            feature_columns=model_feature_columns,
        )
    if bad_fold_group_forensics_table is not None and not bad_fold_group_forensics_table.empty:
        bad_fold_group_forensics_table.to_csv(bundle_dir / "bad_fold_group_forensics.csv", index=False)
        bad_fold_group_forensics_summary(bad_fold_group_forensics_table).to_csv(
            bundle_dir / "bad_fold_group_forensics_summary.csv",
            index=False,
        )
    if experiment_ledger is None:
        experiment_ledger = experiment_ledger_diagnostics(
            report=serializable_report,
            config=config,
            feature_columns=model_feature_columns,
            fold_metrics=fold_metrics,
            recent_fold_summary=recent_fold_summary,
            threshold_summary=threshold_summary,
            score_band_lift=score_band_lift,
            score_lift_by_fold=score_lift_by_fold,
            score_band_summary=score_band_summary,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    if experiment_ledger is not None and not experiment_ledger.empty:
        experiment_ledger.to_csv(bundle_dir / "experiment_ledger.csv", index=False)
        (bundle_dir / "experiment_ledger.json").write_text(
            json.dumps(_json_safe(experiment_ledger.iloc[0].to_dict()), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    fold_summary = good_bad_fold_summary(fold_metrics)
    (bundle_dir / "good_bad_folds.json").write_text(
        json.dumps(_json_safe(fold_summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    zip_path = output_path / f"{bundle_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle_dir.rglob("*")):
            archive.write(path, path.relative_to(bundle_dir))
    return zip_path

def attach_threshold_summary_to_phase1_report(
    report: dict[str, Any],
    threshold_summary: pd.DataFrame | None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Add val-selected threshold F1 context without replacing the core Phase 1 checks."""

    updated = dict(report)
    if threshold_summary is None or threshold_summary.empty:
        return updated

    selected_threshold = _threshold_summary_mean(threshold_summary, "selected_threshold")
    selected_f1 = _threshold_summary_mean(threshold_summary, "test_f1_at_selected_threshold")
    selected_precision = _threshold_summary_mean(threshold_summary, "test_precision_at_selected_threshold")
    selected_recall = _threshold_summary_mean(threshold_summary, "test_recall_at_selected_threshold")
    selected_pred_rate = _threshold_summary_mean(threshold_summary, "test_pred_long_rate_at_selected_threshold")
    constrained_threshold = _threshold_summary_mean(threshold_summary, "constrained_threshold")
    constrained_f1 = _threshold_summary_mean(threshold_summary, "test_f1_at_constrained_threshold")
    constrained_precision = _threshold_summary_mean(threshold_summary, "test_precision_at_constrained_threshold")
    constrained_recall = _threshold_summary_mean(threshold_summary, "test_recall_at_constrained_threshold")
    constrained_pred_rate = _threshold_summary_mean(threshold_summary, "test_pred_long_rate_at_constrained_threshold")
    oracle_f1 = _threshold_summary_mean(threshold_summary, "test_oracle_best_f1")
    f1_at_050 = _threshold_summary_mean(threshold_summary, "test_f1_at_050")
    min_long_f1 = float(_config_get(config or {}, ["validation", "min_long_f1"], 0.45))
    threshold_cfg = _config_get(config or {}, ["validation", "threshold_checks"], {}) or {}
    max_pred_long_rate = float(_config_get(threshold_cfg, ["max_pred_long_rate"], 0.70))
    min_precision = float(_config_get(threshold_cfg, ["min_precision"], 0.30))

    threshold_checks = {
        "long_f1_selected_threshold": bool(pd.notna(selected_f1) and selected_f1 > min_long_f1),
        "selected_precision": bool(pd.notna(selected_precision) and selected_precision >= min_precision),
        "selected_pred_long_rate": bool(pd.notna(selected_pred_rate) and selected_pred_rate <= max_pred_long_rate),
    }
    constrained_checks = {
        "long_f1_constrained_threshold": bool(pd.notna(constrained_f1) and constrained_f1 > min_long_f1),
        "constrained_precision": bool(pd.notna(constrained_precision) and constrained_precision >= min_precision),
        "constrained_pred_long_rate": bool(
            pd.notna(constrained_pred_rate) and constrained_pred_rate <= max_pred_long_rate
        ),
    }
    selected_guard_ok = all(bool(value) for value in threshold_checks.values())
    constrained_guard_ok = all(bool(value) for value in constrained_checks.values())
    if selected_guard_ok:
        guarded_source = "validation_selected_threshold"
        guarded_reason = ""
        guarded_threshold = selected_threshold
        guarded_f1 = selected_f1
        guarded_precision = selected_precision
        guarded_recall = selected_recall
        guarded_pred_rate = selected_pred_rate
    else:
        guarded_source = "validation_constrained_threshold"
        guarded_reasons = []
        if not threshold_checks["selected_pred_long_rate"]:
            guarded_reasons.append("selected_threshold_pred_long_rate_above_guardrail")
        if not threshold_checks["selected_precision"]:
            guarded_reasons.append("selected_threshold_precision_below_minimum")
        if not threshold_checks["long_f1_selected_threshold"]:
            guarded_reasons.append("selected_threshold_f1_below_target")
        guarded_reason = ";".join(guarded_reasons)
        guarded_threshold = constrained_threshold
        guarded_f1 = constrained_f1
        guarded_precision = constrained_precision
        guarded_recall = constrained_recall
        guarded_pred_rate = constrained_pred_rate

    guarded_checks = {
        "long_f1_guarded_threshold": bool(pd.notna(guarded_f1) and guarded_f1 > min_long_f1),
        "guarded_precision": bool(pd.notna(guarded_precision) and guarded_precision >= min_precision),
        "guarded_pred_long_rate": bool(pd.notna(guarded_pred_rate) and guarded_pred_rate <= max_pred_long_rate),
    }
    core_checks = dict(updated.get("checks", {}) or {})
    threshold_phase_checks = {
        key: value
        for key, value in core_checks.items()
        if key != "long_f1"
    }
    threshold_phase_checks.update(threshold_checks)
    constrained_phase_checks = {
        key: value
        for key, value in core_checks.items()
        if key != "long_f1"
    }
    constrained_phase_checks.update(constrained_checks)
    guarded_phase_checks = {
        key: value
        for key, value in core_checks.items()
        if key != "long_f1"
    }
    guarded_phase_checks.update(guarded_checks)

    updated["threshold_selected"] = {
        "selected_threshold_mean": selected_threshold,
        "test_f1_at_selected_threshold": selected_f1,
        "test_precision_at_selected_threshold": selected_precision,
        "test_recall_at_selected_threshold": selected_recall,
        "test_pred_long_rate_at_selected_threshold": selected_pred_rate,
        "test_oracle_best_f1": oracle_f1,
        "test_f1_at_050": f1_at_050,
    }
    updated["threshold_constrained"] = {
        "constrained_threshold_mean": constrained_threshold,
        "test_f1_at_constrained_threshold": constrained_f1,
        "test_precision_at_constrained_threshold": constrained_precision,
        "test_recall_at_constrained_threshold": constrained_recall,
        "test_pred_long_rate_at_constrained_threshold": constrained_pred_rate,
    }
    updated["threshold_guarded"] = {
        "threshold_source": guarded_source,
        "reject_reason": guarded_reason,
        "selected_threshold_constraints_satisfied": selected_guard_ok,
        "constrained_threshold_constraints_satisfied": constrained_guard_ok,
        "threshold_mean": guarded_threshold,
        "test_f1_at_guarded_threshold": guarded_f1,
        "test_precision_at_guarded_threshold": guarded_precision,
        "test_recall_at_guarded_threshold": guarded_recall,
        "test_pred_long_rate_at_guarded_threshold": guarded_pred_rate,
    }
    updated["checks_threshold_selected"] = threshold_phase_checks
    updated["checks_threshold_constrained"] = constrained_phase_checks
    updated["checks_threshold_guarded"] = guarded_phase_checks
    updated["passed_threshold_selected"] = all(bool(value) for value in threshold_phase_checks.values())
    updated["passed_threshold_constrained"] = all(bool(value) for value in constrained_phase_checks.values())
    updated["passed_threshold_guarded"] = all(bool(value) for value in guarded_phase_checks.values())
    return updated

