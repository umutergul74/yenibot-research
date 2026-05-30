from __future__ import annotations

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .utils import (
    _cfg, _float, _optional_gate_float, _metric_or, _optional_float,
    _is_stability_scope, _diagnostic_candidate_type, _entry_threshold_policy_frame,
    _rank_ic_for_frame, _score_ks_statistic, _numeric_mean
)

def _passes_triage(row: dict[str, Any], control: dict[str, Any], config: dict[str, Any]) -> tuple[bool, str]:
    gates = _cfg(config, ["experiments", "promotion_gates", "triage"], {}) or {}
    reasons = []
    if _float(row, "mean_rank_ic") < _float(control, "mean_rank_ic") + float(gates.get("min_mean_rank_ic_delta", 0.005)):
        reasons.append("mean_rank_ic_delta")
    if _float(row, "std_rank_ic") > _float(control, "std_rank_ic") + float(gates.get("max_std_rank_ic_delta", 0.005)):
        reasons.append("std_rank_ic")
    if _float(row, "positive_ic_fraction") < _float(control, "positive_ic_fraction"):
        reasons.append("positive_ic_fraction")
    if _float(row, "top_10_lift_global") < float(gates.get("min_top_10_lift_global", 1.05)):
        reasons.append("top_10_lift_global")
    if _float(row, "top_10_positive_lift_fold_rate") < float(gates.get("min_top_10_positive_lift_fold_rate", 0.55)):
        reasons.append("top_10_positive_lift_fold_rate")
    worst_5_delta = _optional_gate_float(gates, "min_worst_5_rank_ic_delta", None)
    if worst_5_delta is not None and _float(row, "worst_5_rank_ic_mean") < _float(control, "worst_5_rank_ic_mean") + worst_5_delta:
        reasons.append("worst_5_rank_ic_delta")
    negative_delta = _optional_gate_float(gates, "max_negative_ic_fraction_delta", None)
    if negative_delta is not None and _float(row, "negative_ic_fraction") > _float(control, "negative_ic_fraction") + negative_delta:
        reasons.append("negative_ic_fraction")
    bad_fold_lift_floor = _optional_gate_float(gates, "min_top_10_bad_fold_lift_mean", None)
    if bad_fold_lift_floor is not None and _float(row, "top_10_bad_fold_lift_mean") < bad_fold_lift_floor:
        reasons.append("top_10_bad_fold_lift_mean")
    if not bool(row.get("mtf_leakage_passed", False)):
        reasons.append("mtf_leakage")
    if not bool(row.get("stationarity_policy_passed", False)):
        reasons.append("stationarity_policy")
    return not reasons, ";".join(reasons)

def _passes_full(row: dict[str, Any], control: dict[str, Any], config: dict[str, Any]) -> tuple[bool, str]:
    if bool(row.get("passed_phase1", False)):
        return True, ""
    gates = _cfg(config, ["experiments", "promotion_gates", "full"], {}) or {}
    reasons = []
    if _float(row, "mean_rank_ic") < _float(control, "mean_rank_ic") + float(gates.get("min_mean_rank_ic_delta", 0.005)):
        reasons.append("mean_rank_ic_delta")
    min_positive = max(_float(control, "positive_ic_fraction"), float(gates.get("min_positive_ic_fraction_floor", 0.75)))
    if _float(row, "positive_ic_fraction") < min_positive:
        reasons.append("positive_ic_fraction")
    if _float(row, "std_rank_ic") > _float(control, "std_rank_ic") + float(gates.get("max_std_rank_ic_delta", 0.0)):
        reasons.append("std_rank_ic")
    selected_f1 = _metric_or(
        row,
        "test_f1_at_official_threshold",
        _metric_or(
            row,
            "test_f1_at_guarded_threshold",
            _metric_or(
                row,
                "test_f1_at_constrained_threshold",
                _metric_or(row, "test_f1_at_selected_threshold", _float(row, "mean_long_f1")),
            ),
        ),
    )
    control_selected_f1 = _metric_or(
        control,
        "test_f1_at_official_threshold",
        _metric_or(
            control,
            "test_f1_at_guarded_threshold",
            _metric_or(
                control,
                "test_f1_at_constrained_threshold",
                _metric_or(control, "test_f1_at_selected_threshold", _float(control, "mean_long_f1")),
            ),
        ),
    )
    selected_f1_floor = _optional_gate_float(gates, "min_selected_threshold_f1", None)
    if selected_f1_floor is not None and selected_f1 < selected_f1_floor:
        reasons.append("official_threshold_f1")
    selected_f1_delta = _optional_gate_float(gates, "min_selected_threshold_f1_delta", None)
    if selected_f1_delta is not None and selected_f1 < control_selected_f1 + selected_f1_delta:
        reasons.append("official_threshold_f1_delta")
    threshold_checks = _cfg(config, ["validation", "threshold_checks"], {}) or {}
    max_pred_long_rate = float(threshold_checks.get("max_pred_long_rate", 0.70))
    official_pred_rate = _float(
        row,
        "test_pred_long_rate_at_official_threshold",
        _float(row, "test_pred_long_rate_at_constrained_threshold", np.nan),
    )
    if np.isfinite(official_pred_rate) and official_pred_rate > max_pred_long_rate:
        reasons.append("official_pred_long_rate")
    mean_long_f1_delta = _optional_gate_float(gates, "min_long_f1_delta", None)
    if mean_long_f1_delta is not None and _float(row, "mean_long_f1") < _float(control, "mean_long_f1") + mean_long_f1_delta:
        reasons.append("mean_long_f1_delta")
    if _float(row, "top_10_lift_global") < _float(control, "top_10_lift_global") + float(gates.get("min_top_10_lift_global_delta", 0.05)):
        reasons.append("top_10_lift_global_delta")
    top_lift_floor = _optional_gate_float(gates, "min_top_10_lift_global", None)
    if top_lift_floor is not None and _float(row, "top_10_lift_global") < top_lift_floor:
        reasons.append("top_10_lift_global")
    worst_5_delta = _optional_gate_float(gates, "min_worst_5_rank_ic_delta", None)
    if worst_5_delta is not None and _float(row, "worst_5_rank_ic_mean") < _float(control, "worst_5_rank_ic_mean") + worst_5_delta:
        reasons.append("worst_5_rank_ic_delta")
    negative_delta = _optional_gate_float(gates, "max_negative_ic_fraction_delta", None)
    if negative_delta is not None and _float(row, "negative_ic_fraction") > _float(control, "negative_ic_fraction") + negative_delta:
        reasons.append("negative_ic_fraction")
    bad_fold_lift_delta = _optional_gate_float(gates, "min_top_10_bad_fold_lift_mean_delta", None)
    if bad_fold_lift_delta is not None and _float(row, "top_10_bad_fold_lift_mean") < _float(control, "top_10_bad_fold_lift_mean") + bad_fold_lift_delta:
        reasons.append("top_10_bad_fold_lift_mean_delta")
    if not bool(row.get("mtf_leakage_passed", False)):
        reasons.append("mtf_leakage")
    if not bool(row.get("stationarity_policy_passed", False)):
        reasons.append("stationarity_policy")
    return not reasons, ";".join(reasons)

def _auto_full_profiles(settings: dict[str, Any], triage_rows: list[dict[str, Any]]) -> list[str]:
    control_profile = str(settings["control_profile"])
    profiles = [control_profile]
    for profile in settings.get("always_full_profiles", []) or []:
        profile = str(profile)
        if profile not in profiles:
            profiles.append(profile)

    passed_candidates = [
        row
        for row in triage_rows
        if row["profile"] != control_profile and bool(row.get("promotable"))
    ]
    passed_candidates = sorted(
        passed_candidates,
        key=lambda row: (
            _float(row, "mean_rank_ic", -np.inf),
            _float(row, "top_10_lift_global", -np.inf),
            _float(row, "worst_5_rank_ic_mean", -np.inf),
            _float(row, "top_10_positive_lift_fold_rate", -np.inf),
        ),
        reverse=True,
    )
    max_auto = settings.get("max_auto_full_candidates", None)
    if max_auto is not None:
        passed_candidates = passed_candidates[: max(0, int(max_auto))]

    for row in passed_candidates:
        profile = str(row["profile"])
        if profile not in profiles:
            profiles.append(profile)
    return profiles

def _validation_reliability_metrics(part: pd.DataFrame) -> dict[str, float]:
    if part.empty:
        return {
            "val_rank_ic": np.nan,
            "val_score_gap": np.nan,
            "val_score_ks": np.nan,
            "val_average_precision": np.nan,
            "val_label_long_rate": np.nan,
            "val_prob_long_std": np.nan,
            "val_pred_long_rate_050": np.nan,
        }
    frame = part.replace([np.inf, -np.inf], np.nan).dropna(subset=["label", "prob_long"]).copy()
    if frame.empty:
        return _validation_reliability_metrics(pd.DataFrame())
    labels = frame["label"].astype(int)
    scores = pd.to_numeric(frame["prob_long"], errors="coerce")
    pos_scores = scores.loc[labels == 1]
    neg_scores = scores.loc[labels == 0]
    score_gap = (
        float(pos_scores.mean() - neg_scores.mean())
        if not pos_scores.empty and not neg_scores.empty
        else np.nan
    )
    if len(labels.unique()) > 1 and scores.nunique(dropna=True) > 1:
        average_precision = float(average_precision_score(labels, scores))
    else:
        average_precision = np.nan
    return {
        "val_rank_ic": _rank_ic_for_frame(frame),
        "val_score_gap": score_gap,
        "val_score_ks": _score_ks_statistic(pos_scores, neg_scores),
        "val_average_precision": average_precision,
        "val_label_long_rate": float(labels.mean()) if len(labels) else np.nan,
        "val_prob_long_std": float(scores.std(ddof=0)) if scores.notna().any() else np.nan,
        "val_pred_long_rate_050": float((scores >= 0.5).mean()) if scores.notna().any() else np.nan,
    }

def _reliability_gate_definitions(config: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = _cfg(config, ["validation", "fold_reliability_gates"], {}) or {}
    gates = cfg.get("gates", []) or []
    if gates:
        return [dict(item) for item in gates if isinstance(item, dict) and str(item.get("name", "")).strip()]
    return [
        {"name": "val_rank_ic_positive", "min_val_rank_ic": 0.0},
        {"name": "val_score_gap_positive", "min_val_score_gap": 0.0},
        {"name": "val_rank_ic_and_score_gap_positive", "min_val_rank_ic": 0.0, "min_val_score_gap": 0.0},
        {
            "name": "val_rank_ic_score_gap_and_ks",
            "min_val_rank_ic": 0.0,
            "min_val_score_gap": 0.0,
            "min_val_score_ks": 0.08,
        },
    ]

def _gate_threshold_check(metrics: dict[str, float], gate: dict[str, Any], key: str, metric: str) -> bool:
    if key not in gate:
        return True
    value = _float(metrics, metric)
    threshold = _optional_float(gate.get(key))
    if threshold is None or not np.isfinite(threshold):
        return True
    if not np.isfinite(value):
        return False
    if key.startswith("min_"):
        return value >= threshold
    if key.startswith("max_"):
        return value <= threshold
    return True

def _fold_reliability_gate_passed(metrics: dict[str, float], gate: dict[str, Any]) -> bool:
    checks = [
        ("min_val_rank_ic", "val_rank_ic"),
        ("max_val_rank_ic", "val_rank_ic"),
        ("min_val_score_gap", "val_score_gap"),
        ("max_val_score_gap", "val_score_gap"),
        ("min_val_score_ks", "val_score_ks"),
        ("max_val_score_ks", "val_score_ks"),
        ("min_val_average_precision", "val_average_precision"),
        ("max_val_average_precision", "val_average_precision"),
        ("min_val_prob_long_std", "val_prob_long_std"),
        ("max_val_prob_long_std", "val_prob_long_std"),
        ("max_val_pred_long_rate_050", "val_pred_long_rate_050"),
    ]
    if "min_val_average_precision_lift_vs_base" in gate:
        ap = _float(metrics, "val_average_precision")
        base = _float(metrics, "val_label_long_rate")
        lift = ap - base if np.isfinite(ap) and np.isfinite(base) else np.nan
        metrics = {**metrics, "val_average_precision_lift_vs_base": lift}
        checks.append(("min_val_average_precision_lift_vs_base", "val_average_precision_lift_vs_base"))
    return all(_gate_threshold_check(metrics, gate, key, metric) for key, metric in checks)

def _fold_reliability_gate_frame(entries: list[dict[str, Any]], config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "gate_name",
        "fold",
        "gate_passed",
        "val_rank_ic",
        "val_score_gap",
        "val_score_ks",
        "val_average_precision",
        "val_label_long_rate",
        "val_prob_long_std",
        "val_pred_long_rate_050",
        "test_rank_ic",
        "test_official_f1",
        "test_official_pred_long_rate",
        "test_top_10_lift_vs_base",
        "test_top_10_forward_return",
    ]
    cfg = _cfg(config, ["validation", "fold_reliability_gates"], {}) or {}
    if not bool(cfg.get("enabled", False)):
        return pd.DataFrame(columns=columns)
    gates = _reliability_gate_definitions(config)
    rows: list[dict[str, Any]] = []
    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if not _is_stability_scope(fold_scope):
            continue
        predictions = entry.get("predictions")
        if not isinstance(predictions, pd.DataFrame) or predictions.empty:
            continue
        if not {"fold", "split", "label", "prob_long", "forward_return"}.issubset(predictions.columns):
            continue
        diagnostics = entry.get("diagnostics", {}) or {}
        fold_metrics = diagnostics.get("fold_metrics")
        fold_by_id = (
            {int(row["fold"]): row.to_dict() for _, row in fold_metrics.dropna(subset=["fold"]).iterrows()}
            if isinstance(fold_metrics, pd.DataFrame) and not fold_metrics.empty and "fold" in fold_metrics.columns
            else {}
        )
        threshold_metrics = _entry_threshold_policy_frame(entry)
        threshold_by_id = (
            {int(row["fold"]): row.to_dict() for _, row in threshold_metrics.dropna(subset=["fold"]).iterrows()}
            if threshold_metrics is not None and not threshold_metrics.empty and "fold" in threshold_metrics.columns
            else {}
        )
        score_bands = diagnostics.get("score_band_by_fold")
        top10_by_id: dict[int, dict[str, Any]] = {}
        if isinstance(score_bands, pd.DataFrame) and not score_bands.empty and {"fold", "band"}.issubset(score_bands.columns):
            top10 = score_bands.loc[score_bands["band"].astype(str) == "top_10"].copy()
            top10_by_id = {int(row["fold"]): row.to_dict() for _, row in top10.dropna(subset=["fold"]).iterrows()}

        candidate = str(entry.get("profile", ""))
        candidate_type = _diagnostic_candidate_type(fold_scope)
        for fold, fold_part in predictions.groupby("fold"):
            fold_id = int(fold)
            validation = fold_part.loc[fold_part["split"].astype(str) == "val"].copy()
            test = fold_part.loc[fold_part["split"].astype(str) == "test"].copy()
            if validation.empty or test.empty:
                continue
            val_metrics = _validation_reliability_metrics(validation)
            fold_row = fold_by_id.get(fold_id, {})
            threshold_row = threshold_by_id.get(fold_id, {})
            top10_row = top10_by_id.get(fold_id, {})
            for gate in gates:
                row = {
                    "candidate": candidate,
                    "candidate_type": candidate_type,
                    "fold_scope": fold_scope,
                    "gate_name": str(gate.get("name", "")),
                    "fold": fold_id,
                    "gate_passed": _fold_reliability_gate_passed(dict(val_metrics), gate),
                    **val_metrics,
                    "test_rank_ic": _float(fold_row, "rank_ic", _rank_ic_for_frame(test)),
                    "test_official_f1": _float(
                        threshold_row,
                        "test_f1_at_official_threshold",
                        _float(threshold_row, "test_f1_at_constrained_threshold"),
                    ),
                    "test_official_pred_long_rate": _float(
                        threshold_row,
                        "test_pred_long_rate_at_official_threshold",
                        _float(threshold_row, "test_pred_long_rate_at_constrained_threshold"),
                    ),
                    "test_top_10_lift_vs_base": _float(top10_row, "lift_vs_base"),
                    "test_top_10_forward_return": _float(top10_row, "mean_forward_return"),
                }
                rows.append(row)
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["candidate_type", "candidate", "fold_scope", "gate_name", "fold"])
        .reset_index(drop=True)
    )

def _fold_reliability_gate_summary_frame(detail: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "gate_name",
        "fold_count",
        "accepted_fold_count",
        "rejected_fold_count",
        "accepted_fraction",
        "all_rank_ic_mean",
        "all_rank_ic_std",
        "accepted_rank_ic_mean",
        "accepted_rank_ic_std",
        "accepted_positive_ic_fraction",
        "accepted_official_f1_mean",
        "accepted_top_10_forward_return_mean",
        "rejected_negative_fold_capture_rate",
        "false_reject_positive_fold_rate",
        "accepted_rank_ic_mean_delta",
        "accepted_rank_ic_std_delta",
        "accepted_official_f1_delta",
        "gate_passed_cv",
        "reject_reason",
        "next_action",
    ]
    if detail.empty:
        return pd.DataFrame(columns=columns)
    cfg = _cfg(config, ["validation", "fold_reliability_gates"], {}) or {}
    min_fraction = float(cfg.get("min_accepted_fraction", 0.50))
    min_folds = int(cfg.get("min_accepted_folds", 12))
    min_positive = float(cfg.get("min_positive_ic_fraction", 0.75))
    max_std = float(cfg.get("max_rank_ic_std", 0.06))
    min_f1_delta = float(cfg.get("min_official_f1_delta", 0.0))
    rows: list[dict[str, Any]] = []
    for (candidate, candidate_type, fold_scope, gate_name), part in detail.groupby(
        ["candidate", "candidate_type", "fold_scope", "gate_name"],
        dropna=False,
    ):
        rank = pd.to_numeric(part["test_rank_ic"], errors="coerce")
        passed = part["gate_passed"].astype(bool)
        accepted = part.loc[passed]
        rejected = part.loc[~passed]
        accepted_rank = pd.to_numeric(accepted["test_rank_ic"], errors="coerce")
        rejected_rank = pd.to_numeric(rejected["test_rank_ic"], errors="coerce")
        total_negative = int((rank < 0.0).sum())
        rejected_negative = int((rejected_rank < 0.0).sum())
        total_positive = int((rank > 0.0).sum())
        rejected_positive = int((rejected_rank > 0.0).sum())
        all_f1 = pd.to_numeric(part["test_official_f1"], errors="coerce")
        accepted_f1 = pd.to_numeric(accepted["test_official_f1"], errors="coerce")
        all_mean = float(rank.mean()) if rank.notna().any() else np.nan
        all_std = float(rank.std(ddof=1)) if rank.notna().sum() > 1 else np.nan
        accepted_mean = float(accepted_rank.mean()) if accepted_rank.notna().any() else np.nan
        accepted_std = float(accepted_rank.std(ddof=1)) if accepted_rank.notna().sum() > 1 else np.nan
        f1_delta = (
            float(accepted_f1.mean() - all_f1.mean())
            if accepted_f1.notna().any() and all_f1.notna().any()
            else np.nan
        )
        reasons: list[str] = []
        accepted_count = int(len(accepted))
        fold_count = int(part["fold"].nunique())
        accepted_fraction = float(accepted_count / fold_count) if fold_count else 0.0
        positive_fraction = float((accepted_rank > 0.0).mean()) if accepted_count else np.nan
        if accepted_count < min_folds:
            reasons.append("accepted_fold_count")
        if accepted_fraction < min_fraction:
            reasons.append("accepted_fraction")
        if not np.isfinite(positive_fraction) or positive_fraction < min_positive:
            reasons.append("accepted_positive_ic_fraction")
        if not np.isfinite(accepted_std) or accepted_std > max_std:
            reasons.append("accepted_rank_ic_std")
        if not np.isfinite(f1_delta) or f1_delta < min_f1_delta:
            reasons.append("accepted_official_f1_delta")
        if np.isfinite(accepted_std) and np.isfinite(all_std) and accepted_std >= all_std:
            reasons.append("does_not_reduce_rank_ic_std")
        reject_reason = ";".join(dict.fromkeys(reasons))
        if not reject_reason:
            next_action = "pre_register_reliability_gate_for_future_oos_review"
        elif "accepted_fold_count" in reject_reason or "accepted_fraction" in reject_reason:
            next_action = "gate_too_sparse_for_phase1_decision"
        elif "accepted_rank_ic_std" in reject_reason or "does_not_reduce_rank_ic_std" in reject_reason:
            next_action = "gate_does_not_reduce_fold_std"
        else:
            next_action = "gate_diagnostic_only_do_not_promote"
        rows.append(
            {
                "candidate": candidate,
                "candidate_type": candidate_type,
                "fold_scope": fold_scope,
                "gate_name": gate_name,
                "fold_count": fold_count,
                "accepted_fold_count": accepted_count,
                "rejected_fold_count": int(len(rejected)),
                "accepted_fraction": accepted_fraction,
                "all_rank_ic_mean": all_mean,
                "all_rank_ic_std": all_std,
                "accepted_rank_ic_mean": accepted_mean,
                "accepted_rank_ic_std": accepted_std,
                "accepted_positive_ic_fraction": positive_fraction,
                "accepted_official_f1_mean": float(accepted_f1.mean()) if accepted_f1.notna().any() else np.nan,
                "accepted_top_10_forward_return_mean": _numeric_mean(accepted, "test_top_10_forward_return"),
                "rejected_negative_fold_capture_rate": (
                    float(rejected_negative / total_negative) if total_negative else np.nan
                ),
                "false_reject_positive_fold_rate": (
                    float(rejected_positive / total_positive) if total_positive else np.nan
                ),
                "accepted_rank_ic_mean_delta": (
                    accepted_mean - all_mean if np.isfinite(accepted_mean) and np.isfinite(all_mean) else np.nan
                ),
                "accepted_rank_ic_std_delta": (
                    accepted_std - all_std if np.isfinite(accepted_std) and np.isfinite(all_std) else np.nan
                ),
                "accepted_official_f1_delta": f1_delta,
                "gate_passed_cv": not bool(reject_reason),
                "reject_reason": reject_reason,
                "next_action": next_action,
            }
        )
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(
            ["gate_passed_cv", "accepted_rank_ic_std_delta", "accepted_rank_ic_mean_delta"],
            ascending=[False, True, False],
        )
        .reset_index(drop=True)
    )

