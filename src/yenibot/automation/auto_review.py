from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, UnicodeDecodeError, OSError):
        return pd.DataFrame()


def _to_bool(value: Any, default: bool = False) -> bool:
    if pd.isna(value):
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return default


def _to_float(value: Any, default: float | None = None) -> float | None:
    if value is None or pd.isna(value):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _metric(row: dict[str, Any], key: str, default: float | None = None) -> float | None:
    return _to_float(row.get(key), default)


def _official_long_f1(control: dict[str, Any]) -> tuple[float | None, str, dict[str, Any]]:
    """Return the Phase 1 F1 source after enforcing the pred-long-rate guardrail."""

    official_f1 = _metric(control, "test_f1_at_official_threshold")
    official_rate = _metric(control, "test_pred_long_rate_at_official_threshold")
    official_source = str(control.get("official_threshold_source") or "")
    if official_f1 is not None:
        return official_f1, official_source or "validation_official_threshold", {
            "official_threshold_value": official_f1,
            "official_threshold_pred_long_rate": official_rate,
            "official_threshold_source": official_source,
            "official_threshold_uses_calibration": _to_bool(control.get("official_threshold_uses_calibration", False)),
            "official_threshold_selection_score": _metric(control, "official_threshold_selection_score"),
            "calibrated_guarded_threshold_value": _metric(control, "test_f1_at_calibrated_guarded_threshold"),
            "calibrated_guarded_threshold_pred_long_rate": _metric(
                control,
                "test_pred_long_rate_at_calibrated_guarded_threshold",
            ),
        }

    guarded_f1 = _metric(control, "test_f1_at_guarded_threshold")
    guarded_rate = _metric(control, "test_pred_long_rate_at_guarded_threshold")
    guarded_source = str(control.get("guarded_threshold_source") or "")
    if guarded_f1 is not None:
        return guarded_f1, guarded_source or "validation_guarded_threshold", {
            "guarded_threshold_value": guarded_f1,
            "guarded_threshold_pred_long_rate": guarded_rate,
            "guarded_threshold_source": guarded_source,
        }

    selected_f1 = _metric(control, "test_f1_at_selected_threshold")
    selected_pred_rate = _metric(control, "test_pred_long_rate_at_selected_threshold")
    constrained_f1 = _metric(control, "test_f1_at_constrained_threshold")
    constrained_pred_rate = _metric(control, "test_pred_long_rate_at_constrained_threshold")
    fixed_050_f1 = _metric(control, "mean_long_f1")
    details = {
        "fixed_0_50_value": fixed_050_f1,
        "selected_threshold_value": selected_f1,
        "selected_threshold_pred_long_rate": selected_pred_rate,
        "constrained_threshold_value": constrained_f1,
        "constrained_threshold_pred_long_rate": constrained_pred_rate,
    }
    if selected_f1 is not None and (selected_pred_rate is None or selected_pred_rate <= 0.70):
        return selected_f1, "validation_selected_threshold", details
    if constrained_f1 is not None:
        return constrained_f1, "validation_constrained_threshold", details
    if selected_f1 is not None:
        return selected_f1, "validation_selected_threshold_above_pred_rate_guardrail", details
    return fixed_050_f1, "fixed_0_50_threshold", details


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return [_json_ready(row) for row in frame.to_dict(orient="records")]


def _best_row(frame: pd.DataFrame, metric: str) -> dict[str, Any]:
    if frame.empty or metric not in frame.columns:
        return {}
    scored = frame.copy()
    scored[metric] = pd.to_numeric(scored[metric], errors="coerce")
    scored = scored.dropna(subset=[metric])
    if scored.empty:
        return {}
    return _json_ready(scored.sort_values(metric, ascending=False).iloc[0].to_dict())


def _profile_row(frame: pd.DataFrame, profile: str, fold_scope: str = "full") -> dict[str, Any]:
    if frame.empty or "profile" not in frame.columns:
        return {}
    scoped = frame.copy()
    if "fold_scope" in scoped.columns:
        scoped = scoped[scoped["fold_scope"].astype(str) == fold_scope]
    matched = scoped[scoped["profile"].astype(str) == str(profile)]
    if matched.empty:
        return {}
    return _json_ready(matched.iloc[0].to_dict())


def _full_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "fold_scope" not in frame.columns:
        return frame
    return frame[frame["fold_scope"].astype(str) == "full"].copy()


def _missing_required_files(report_dir: Path) -> list[str]:
    required = [
        "profile_comparison.csv",
        "decision_report.json",
        "experiment_selection.csv",
        "missing_selected_profiles.csv",
        "training_execution_summary.json",
    ]
    return [name for name in required if not (report_dir / name).exists()]


def _selected_missing_profiles(report_dir: Path) -> list[dict[str, Any]]:
    missing = _read_csv(report_dir / "missing_selected_profiles.csv")
    if missing.empty:
        return []
    return _records(missing)


def _holdout_policy(report_dir: Path, decision: dict[str, Any]) -> dict[str, Any]:
    guard = _read_csv(report_dir / "experiment_policy_guard.csv")
    reservation = _read_csv(report_dir / "holdout_reservation.csv")
    guard_row = _json_ready(guard.iloc[0].to_dict()) if not guard.empty else {}
    reservation_row = _json_ready(reservation.iloc[0].to_dict()) if not reservation.empty else {}
    decision_holdout = decision.get("holdout", {}) if isinstance(decision.get("holdout"), dict) else {}
    return {
        "status": guard_row.get("status"),
        "action": guard_row.get("action"),
        "next_action": guard_row.get("next_action") or decision_holdout.get("next_action"),
        "future_oos_ready": _to_bool(
            guard_row.get("future_oos_ready", reservation_row.get("future_oos_ready", False))
        ),
        "future_oos_preferred_ready": _to_bool(
            guard_row.get("future_oos_preferred_ready", reservation_row.get("future_oos_preferred_ready", False))
        ),
        "new_bars_since_anchor": _to_float(
            guard_row.get("new_bars_since_anchor", reservation_row.get("new_bars_since_anchor"))
        ),
        "min_new_bars_remaining": _to_float(
            guard_row.get("min_new_bars_remaining", reservation_row.get("min_new_bars_remaining"))
        ),
        "preferred_new_bars_remaining": _to_float(
            guard_row.get(
                "preferred_new_bars_remaining",
                reservation_row.get("preferred_new_bars_remaining"),
            )
        ),
        "min_ready_at": guard_row.get("min_ready_at"),
        "preferred_ready_at": guard_row.get("preferred_ready_at"),
        "holdout_roll_forward_locked": _to_bool(
            guard_row.get(
                "holdout_roll_forward_locked",
                reservation_row.get("holdout_roll_forward_locked", True),
            ),
            default=True,
        ),
    }


def _cv_promotable_candidates(full_frame: pd.DataFrame, control_row: dict[str, Any]) -> list[dict[str, Any]]:
    if full_frame.empty or not control_row:
        return []
    control_profile = str(control_row.get("profile", ""))
    control_ic = _metric(control_row, "mean_rank_ic")
    control_std = _metric(control_row, "std_rank_ic")
    control_positive = _metric(control_row, "positive_ic_fraction")
    if control_ic is None or control_std is None or control_positive is None:
        return []
    candidates = []
    for _, row in full_frame.iterrows():
        item = _json_ready(row.to_dict())
        if str(item.get("profile")) == control_profile:
            continue
        mean_ic = _metric(item, "mean_rank_ic")
        std_ic = _metric(item, "std_rank_ic")
        positive = _metric(item, "positive_ic_fraction")
        if mean_ic is None or std_ic is None or positive is None:
            continue
        if mean_ic >= control_ic + 0.005 and std_ic <= control_std + 0.005 and positive >= control_positive:
            candidates.append(item)
    return candidates


def _best_holdout_rows(holdout: pd.DataFrame) -> dict[str, Any]:
    if holdout.empty:
        return {}
    out = {
        "best_mean_rank_ic": _best_row(holdout, "mean_rank_ic"),
        "best_top_10_lift": _best_row(holdout, "top_10_lift_global"),
        "best_top_10_forward_return": _best_row(holdout, "top_10_forward_return_global"),
    }
    pass_col = "holdout_signal_pass"
    if pass_col in holdout.columns:
        signal_passed = holdout[holdout[pass_col].map(_to_bool)]
        out["best_signal_pass_mean_rank_ic"] = _best_row(signal_passed, "mean_rank_ic")
    return out


def _best_future_candidate(plan: pd.DataFrame) -> dict[str, Any]:
    if plan.empty:
        return {}
    scored = plan.copy()
    if "stage" in scored.columns:
        preferred = scored[
            scored["stage"].astype(str).isin(
                ["future_oos_candidate", "future_oos_score_band_policy"]
            )
        ].copy()
        if not preferred.empty:
            scored = preferred
    for column in ["cv_mean_rank_ic", "holdout_mean_rank_ic", "future_oos_priority_score"]:
        if column in scored.columns:
            scored[column] = pd.to_numeric(scored[column], errors="coerce")
    metric = "future_oos_priority_score" if "future_oos_priority_score" in scored.columns else None
    if metric and scored[metric].notna().any():
        return _json_ready(scored.sort_values(metric, ascending=False).iloc[0].to_dict())
    if "holdout_mean_rank_ic" in scored.columns and scored["holdout_mean_rank_ic"].notna().any():
        return _json_ready(scored.sort_values("holdout_mean_rank_ic", ascending=False).iloc[0].to_dict())
    if "cv_mean_rank_ic" in scored.columns and scored["cv_mean_rank_ic"].notna().any():
        return _json_ready(scored.sort_values("cv_mean_rank_ic", ascending=False).iloc[0].to_dict())
    if "plan_rank" in scored.columns:
        scored["plan_rank"] = pd.to_numeric(scored["plan_rank"], errors="coerce")
        if scored["plan_rank"].notna().any():
            return _json_ready(scored.sort_values("plan_rank", ascending=True).iloc[0].to_dict())
    return _json_ready(scored.iloc[0].to_dict())


def _forensics_summary(
    *,
    fold_stability_summary: pd.DataFrame,
    fold_stability_forensics: pd.DataFrame,
    threshold_forensics: pd.DataFrame,
    control_profile: str,
) -> dict[str, Any]:
    control_summary = {}
    if not fold_stability_summary.empty and "candidate" in fold_stability_summary.columns:
        matched = fold_stability_summary[
            fold_stability_summary["candidate"].astype(str) == str(control_profile)
        ]
        if not matched.empty:
            control_summary = _json_ready(matched.iloc[0].to_dict())

    worst_control_fold = {}
    top_std_driver = {}
    if not fold_stability_forensics.empty and "candidate" in fold_stability_forensics.columns:
        control_folds = fold_stability_forensics[
            fold_stability_forensics["candidate"].astype(str) == str(control_profile)
        ].copy()
        if not control_folds.empty:
            if "rank_ic" in control_folds.columns:
                control_folds["rank_ic"] = pd.to_numeric(control_folds["rank_ic"], errors="coerce")
                ranked = control_folds.dropna(subset=["rank_ic"]).sort_values("rank_ic", ascending=True)
                if not ranked.empty:
                    worst_control_fold = _json_ready(ranked.iloc[0].to_dict())
            if "rank_ic_variance_contribution" in control_folds.columns:
                control_folds["rank_ic_variance_contribution"] = pd.to_numeric(
                    control_folds["rank_ic_variance_contribution"],
                    errors="coerce",
                )
                ranked = control_folds.dropna(subset=["rank_ic_variance_contribution"]).sort_values(
                    "rank_ic_variance_contribution",
                    ascending=False,
                )
                if not ranked.empty:
                    top_std_driver = _json_ready(ranked.iloc[0].to_dict())

    issue_counts: dict[str, int] = {}
    if not threshold_forensics.empty and "primary_issue" in threshold_forensics.columns:
        issue_counts = {
            str(key): int(value)
            for key, value in threshold_forensics["primary_issue"].astype(str).value_counts().to_dict().items()
        }

    return {
        "control_fold_stability_summary": control_summary,
        "worst_control_fold": worst_control_fold,
        "top_control_std_driver_fold": top_std_driver,
        "threshold_issue_counts": issue_counts,
    }


def _next_action(
    *,
    missing_files: list[str],
    missing_profiles: list[dict[str, Any]],
    policy: dict[str, Any],
    cv_promotable: list[dict[str, Any]],
    decision: dict[str, Any],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if missing_files:
        reasons.append("required_report_files_missing")
        return "fix_report_generation", reasons
    if missing_profiles:
        reasons.append("selected_profiles_missing_from_comparison")
        return "fix_missing_selected_profiles", reasons
    if not bool(decision.get("holdout_boundary_passed", True)):
        reasons.append("holdout_boundary_audit_failed")
        return "rerun_training_with_holdout_split", reasons
    if cv_promotable:
        reasons.append("candidate_passed_cv_gates")
        if not bool(policy.get("future_oos_ready", False)):
            reasons.append("future_oos_not_ready_do_not_promote_from_current_holdout")
            return "wait_for_new_unseen_bars_keep_control", reasons
        return "review_cv_promotable_candidate_on_future_oos", reasons
    if not bool(policy.get("future_oos_ready", False)):
        reasons.append("future_oos_not_ready")
        return "wait_for_new_unseen_bars_keep_control", reasons
    reasons.append("no_candidate_beats_control_cv_gates")
    return "keep_control_profile", reasons


def _phase2_readiness(review: dict[str, Any]) -> dict[str, Any]:
    control = review.get("cv", {}).get("control", {}) or {}
    policy = review.get("holdout", {}).get("policy", {}) or {}
    completeness = review.get("report_completeness", {}) or {}
    mean_rank_ic = _metric(control, "mean_rank_ic")
    std_rank_ic = _metric(control, "std_rank_ic")
    positive_ic_fraction = _metric(control, "positive_ic_fraction")
    fixed_050_f1 = _metric(control, "mean_long_f1")
    selected_f1 = _metric(control, "test_f1_at_selected_threshold")
    constrained_f1 = _metric(control, "test_f1_at_constrained_threshold")
    selected_pred_rate = _metric(control, "test_pred_long_rate_at_selected_threshold")
    constrained_pred_rate = _metric(control, "test_pred_long_rate_at_constrained_threshold")
    calibration_separation = _metric(control, "calibration_separation")
    official_long_f1, official_long_f1_source, official_long_f1_details = _official_long_f1(control)
    advisories: list[str] = []
    if fixed_050_f1 is not None and fixed_050_f1 <= 0.45 and official_long_f1_source != "fixed_0_50_threshold":
        advisories.append("fixed_0_50_f1_below_target_calibration_issue")
    if constrained_f1 is not None and constrained_f1 <= 0.45:
        advisories.append("constrained_threshold_f1_below_target")
    if selected_pred_rate is not None and selected_pred_rate > 0.70:
        advisories.append("selected_threshold_pred_long_rate_above_guardrail")
    if constrained_pred_rate is not None and constrained_pred_rate > 0.70:
        advisories.append("constrained_threshold_pred_long_rate_above_guardrail")
    checks = [
        {
            "check": "report_complete",
            "passed": bool(completeness.get("complete", False)),
            "value": completeness.get("complete", False),
            "target": True,
            "blocker": "report_missing_required_outputs",
        },
        {
            "check": "mean_rank_ic",
            "passed": mean_rank_ic is not None and mean_rank_ic > 0.03,
            "value": mean_rank_ic,
            "target": "> 0.03",
            "blocker": "mean_rank_ic_below_phase1_target",
        },
        {
            "check": "rank_ic_std",
            "passed": std_rank_ic is not None and std_rank_ic < 0.03,
            "value": std_rank_ic,
            "target": "< 0.03",
            "blocker": "rank_ic_std_above_phase1_target",
        },
        {
            "check": "positive_ic_fraction",
            "passed": positive_ic_fraction is not None and positive_ic_fraction > 0.75,
            "value": positive_ic_fraction,
            "target": "> 0.75",
            "blocker": "positive_ic_fraction_below_phase1_target",
        },
        {
            "check": "long_f1",
            "passed": official_long_f1 is not None and official_long_f1 > 0.45,
            "value": official_long_f1,
            "target": "> 0.45",
            "blocker": "long_f1_below_phase1_target",
            "source": official_long_f1_source,
            **official_long_f1_details,
        },
        {
            "check": "calibration_separation",
            "passed": calibration_separation is not None and calibration_separation > 0.0,
            "value": calibration_separation,
            "target": "> 0",
            "blocker": "calibration_separation_missing_or_nonpositive",
        },
        {
            "check": "mtf_leakage",
            "passed": _to_bool(control.get("mtf_leakage_passed"), default=False),
            "value": control.get("mtf_leakage_passed"),
            "target": True,
            "blocker": "mtf_leakage_check_failed",
        },
        {
            "check": "stationarity_policy",
            "passed": _to_bool(control.get("stationarity_policy_passed"), default=False),
            "value": control.get("stationarity_policy_passed"),
            "target": True,
            "blocker": "stationarity_policy_check_failed",
        },
        {
            "check": "future_unseen_oos_ready",
            "passed": bool(policy.get("future_oos_ready", False)),
            "value": policy.get("future_oos_ready", False),
            "target": True,
            "blocker": "future_unseen_oos_not_ready",
        },
    ]
    blockers = [str(item["blocker"]) for item in checks if not bool(item["passed"])]
    ready = not blockers
    return {
        "ready_for_phase2": ready,
        "decision": "READY_FOR_PHASE2" if ready else "DO_NOT_PROCEED_TO_PHASE2",
        "blockers": blockers,
        "advisories": advisories,
        "long_f1_source": official_long_f1_source,
        "checks": checks,
        "next_action": (
            "freeze_phase1_candidate_and_prepare_phase2_design"
            if ready
            else "continue_phase1_validation_until_all_blockers_clear"
        ),
    }


def _phase1_transition_plan(review: dict[str, Any]) -> dict[str, Any]:
    control = review.get("cv", {}).get("control", {}) or {}
    phase2 = review.get("phase2_readiness", {}) or {}
    policy = review.get("holdout", {}).get("policy", {}) or {}
    report_complete = bool(review.get("report_completeness", {}).get("complete", False))
    leakage_ok = _to_bool(control.get("mtf_leakage_passed"), default=False)
    stationarity_ok = _to_bool(control.get("stationarity_policy_passed"), default=False)
    mean_rank_ic = _metric(control, "mean_rank_ic")
    std_rank_ic = _metric(control, "std_rank_ic")
    positive_ic_fraction = _metric(control, "positive_ic_fraction")
    mean_long_f1 = _metric(control, "mean_long_f1")
    selected_f1 = _metric(control, "test_f1_at_selected_threshold")
    constrained_f1 = _metric(control, "test_f1_at_constrained_threshold")
    selected_pred_rate = _metric(control, "test_pred_long_rate_at_selected_threshold")
    constrained_pred_rate = _metric(control, "test_pred_long_rate_at_constrained_threshold")
    calibration_separation = _metric(control, "calibration_separation")
    phase2_long_f1_source = str(phase2.get("long_f1_source") or "")
    official_f1, fallback_source, _ = _official_long_f1(control)
    if not phase2_long_f1_source:
        phase2_long_f1_source = fallback_source
    signal_present = (
        report_complete
        and leakage_ok
        and stationarity_ok
        and mean_rank_ic is not None
        and mean_rank_ic > 0.03
        and positive_ic_fraction is not None
        and positive_ic_fraction > 0.75
        and calibration_separation is not None
        and calibration_separation > 0.0
    )
    selected_threshold_near_phase2 = selected_f1 is not None and selected_f1 > 0.45
    constrained_threshold_near_phase2 = constrained_f1 is not None and constrained_f1 > 0.42
    research_ready = bool(signal_present and (selected_threshold_near_phase2 or constrained_threshold_near_phase2))
    if bool(phase2.get("ready_for_phase2", False)):
        decision = "READY_FOR_PHASE2_DESIGN"
    elif research_ready:
        decision = "PHASE1_RESEARCH_READY_PHASE2_BLOCKED"
    else:
        decision = "CONTINUE_PHASE1_SIGNAL_DEVELOPMENT"
    metric_gaps = {
        "mean_rank_ic_margin_vs_0_03": None if mean_rank_ic is None else mean_rank_ic - 0.03,
        "rank_ic_std_excess_vs_0_03": None if std_rank_ic is None else std_rank_ic - 0.03,
        "positive_ic_fraction_margin_vs_0_75": (
            None if positive_ic_fraction is None else positive_ic_fraction - 0.75
        ),
        "mean_long_f1_gap_vs_0_45": None if mean_long_f1 is None else 0.45 - mean_long_f1,
        "official_long_f1_gap_vs_0_45": None if official_f1 is None else 0.45 - official_f1,
        "selected_threshold_f1_gap_vs_0_45": None if selected_f1 is None else 0.45 - selected_f1,
        "constrained_threshold_f1_gap_vs_0_45": None if constrained_f1 is None else 0.45 - constrained_f1,
        "selected_threshold_pred_long_rate_excess_vs_0_70": (
            None if selected_pred_rate is None else selected_pred_rate - 0.70
        ),
        "constrained_threshold_pred_long_rate_excess_vs_0_70": (
            None if constrained_pred_rate is None else constrained_pred_rate - 0.70
        ),
        "future_oos_min_bars_remaining": policy.get("min_new_bars_remaining"),
        "future_oos_preferred_bars_remaining": policy.get("preferred_new_bars_remaining"),
    }
    allowed_actions = [
        "run_05_cpu_slim_only_to_monitor_reports",
        "wait_for_future_unseen_oos_before_promotion",
        "use_phase1_predictions_for_score_band_diagnostics_only",
        "work_on_rank_ic_std_and_f1_blockers_inside_phase1",
    ]
    if research_ready:
        allowed_actions.append("prepare_phase2_design_document_without_backtest_or_execution_code")
    blocked_actions = [
        "do_not_start_phase2_backtest",
        "do_not_build_execution_or_live_bot",
        "do_not_promote_profile_or_blend_from_current_holdout",
        "do_not_tune_weights_against_current_holdout",
        "do_not_relax_phase1_success_criteria_silently",
    ]
    recommended_focus = []
    blockers = set(str(item) for item in phase2.get("blockers", []) or [])
    if "rank_ic_std_above_phase1_target" in blockers:
        recommended_focus.append("reduce_fold_to_fold_rank_ic_volatility")
    if "long_f1_below_phase1_target" in blockers:
        recommended_focus.append("improve_long_class_decision_quality_without_changing_labels_to_3class")
    advisories = set(str(item) for item in phase2.get("advisories", []) or [])
    if advisories.intersection(
        {
            "constrained_threshold_f1_below_target",
            "selected_threshold_pred_long_rate_above_guardrail",
            "constrained_threshold_pred_long_rate_above_guardrail",
        }
    ):
        recommended_focus.append("improve_threshold_constrained_f1_without_reusing_holdout")
    if "future_unseen_oos_not_ready" in blockers:
        recommended_focus.append("wait_for_new_unseen_bars_before_any_promotion")
    return {
        "decision": decision,
        "research_ready_without_phase2": research_ready,
        "ready_for_phase2": bool(phase2.get("ready_for_phase2", False)),
        "phase2_blockers": list(phase2.get("blockers", []) or []),
        "phase2_advisories": list(phase2.get("advisories", []) or []),
        "long_f1_source": phase2_long_f1_source,
        "metric_gaps": metric_gaps,
        "allowed_actions": allowed_actions,
        "blocked_actions": blocked_actions,
        "recommended_focus": recommended_focus,
        "watch_candidates": {
            "control_profile": review.get("control_profile"),
            "best_future_oos_candidate_plan_row": review.get("future_oos", {}).get("best_candidate_plan_row", {}),
            "best_cv_blend_by_mean_ic": review.get("blends", {}).get("best_mean_rank_ic_blend", {}),
            "best_cv_blend_by_top_10_lift": review.get("blends", {}).get("best_top_10_lift_blend", {}),
            "best_holdout_diagnostic_by_mean_ic": review.get("holdout", {}).get("best", {}).get("best_mean_rank_ic", {}),
            "best_holdout_diagnostic_by_top_10_lift": review.get("holdout", {}).get("best", {}).get("best_top_10_lift", {}),
            "control_fold_stability_summary": review.get("forensics", {}).get("control_fold_stability_summary", {}),
            "top_control_std_driver_fold": review.get("forensics", {}).get("top_control_std_driver_fold", {}),
            "threshold_issue_counts": review.get("forensics", {}).get("threshold_issue_counts", {}),
        },
        "future_oos": {
            "ready": bool(policy.get("future_oos_ready", False)),
            "min_ready_at": policy.get("min_ready_at"),
            "preferred_ready_at": policy.get("preferred_ready_at"),
            "new_bars_since_anchor": policy.get("new_bars_since_anchor"),
            "min_new_bars_remaining": policy.get("min_new_bars_remaining"),
            "preferred_new_bars_remaining": policy.get("preferred_new_bars_remaining"),
        },
    }


def review_experiment_report(report_dir: str | Path) -> dict[str, Any]:
    """Create a deterministic review summary from a Phase 1 experiment report directory."""
    report_path = Path(report_dir)
    decision = _read_json(report_path / "decision_report.json")
    comparison = _read_csv(report_path / "profile_comparison.csv")
    blends = _read_csv(report_path / "profile_blend.csv")
    holdout = _read_csv(report_path / "holdout_evaluation.csv")
    policy_plan = _read_csv(report_path / "future_oos_candidate_plan.csv")
    fold_stability_summary = _read_csv(report_path / "fold_stability_summary.csv")
    fold_stability_forensics = _read_csv(report_path / "fold_stability_forensics.csv")
    threshold_forensics = _read_csv(report_path / "threshold_forensics.csv")
    training = _read_json(report_path / "training_execution_summary.json")
    missing_files = _missing_required_files(report_path)
    missing_profiles = _selected_missing_profiles(report_path)
    control_profile = str(decision.get("control_profile") or "")
    full_frame = _full_profiles(comparison)
    control_row = _profile_row(comparison, control_profile, "full")
    best_cv_profile = _best_row(full_frame, "mean_rank_ic")
    best_cv_lift_profile = _best_row(full_frame, "top_10_lift_global")
    best_blend = _best_row(blends, "mean_rank_ic")
    best_blend_lift = _best_row(blends, "top_10_lift_global")
    holdout_best = _best_holdout_rows(holdout)
    policy = _holdout_policy(report_path, decision)
    cv_promotable = _cv_promotable_candidates(full_frame, control_row)
    action, reasons = _next_action(
        missing_files=missing_files,
        missing_profiles=missing_profiles,
        policy=policy,
        cv_promotable=cv_promotable,
        decision=decision,
    )
    review = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_dir": str(report_path),
        "run_id": str(decision.get("run_id") or report_path.name),
        "control_profile": control_profile,
        "training": {
            "executed_training_scopes": training.get("executed_training_scopes", []),
            "training_executed_count": training.get("training_executed_count"),
            "training_skipped_count": training.get("training_skipped_count"),
            "all_training_scopes_reused": training.get("all_training_scopes_reused"),
            "metadata_available": bool(training),
        },
        "report_completeness": {
            "missing_required_files": missing_files,
            "missing_selected_profiles": missing_profiles,
            "complete": not missing_files and not missing_profiles,
        },
        "cv": {
            "control": control_row,
            "best_mean_rank_ic_profile": best_cv_profile,
            "best_top_10_lift_profile": best_cv_lift_profile,
            "cv_promotable_candidates": cv_promotable,
        },
        "blends": {
            "best_mean_rank_ic_blend": best_blend,
            "best_top_10_lift_blend": best_blend_lift,
            "count": int(len(blends)) if not blends.empty else 0,
        },
        "holdout": {
            "policy": policy,
            "best": holdout_best,
            "diagnostic_only": not bool(policy.get("future_oos_ready", False)),
        },
        "future_oos": {
            "candidate_count": int(len(policy_plan)) if not policy_plan.empty else 0,
            "best_candidate_plan_row": _best_future_candidate(policy_plan),
        },
        "forensics": _forensics_summary(
            fold_stability_summary=fold_stability_summary,
            fold_stability_forensics=fold_stability_forensics,
            threshold_forensics=threshold_forensics,
            control_profile=control_profile,
        ),
        "next_action": {
            "action": action,
            "reasons": reasons,
            "do_not_promote_from_current_holdout": not bool(policy.get("future_oos_ready", False)),
        },
        "decision_recommendation": decision.get("recommendation"),
    }
    review["phase2_readiness"] = _phase2_readiness(review)
    review["phase1_transition_plan"] = _phase1_transition_plan(review)
    return review


def _fmt(value: Any, digits: int = 4) -> str:
    number = _to_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def _profile_label(row: dict[str, Any]) -> str:
    if not row:
        return "None"
    profile = row.get("profile") or row.get("candidate") or "unknown"
    return str(profile)


def _row_metric_line(row: dict[str, Any]) -> str:
    if not row:
        return "- No row available."
    return (
        f"- `{_profile_label(row)}`: mean IC `{_fmt(row.get('mean_rank_ic'))}`, "
        f"std `{_fmt(row.get('std_rank_ic'))}`, positive folds `{_fmt(row.get('positive_ic_fraction'))}`, "
        f"top-10 lift `{_fmt(row.get('top_10_lift_global'))}`, "
        f"official F1 `{_fmt(row.get('test_f1_at_official_threshold', row.get('test_f1_at_guarded_threshold')))}`, "
        f"official source `{row.get('official_threshold_source', row.get('guarded_threshold_source', ''))}`"
    )


def auto_review_markdown(review: dict[str, Any]) -> str:
    policy = review["holdout"]["policy"]
    phase2 = review.get("phase2_readiness", {})
    transition = review.get("phase1_transition_plan", {})
    forensics = review.get("forensics", {}) or {}
    fold_summary = forensics.get("control_fold_stability_summary", {}) or {}
    std_driver = forensics.get("top_control_std_driver_fold", {}) or {}
    lines = [
        f"# Phase 1 Auto Review - {review['run_id']}",
        "",
        "## Verdict",
        f"- Next action: `{review['next_action']['action']}`",
        f"- Reasons: `{';'.join(review['next_action']['reasons']) or 'none'}`",
        f"- Do not promote from current holdout: `{review['next_action']['do_not_promote_from_current_holdout']}`",
        f"- Decision recommendation from diagnostics: `{review.get('decision_recommendation')}`",
        f"- Phase 2 readiness: `{phase2.get('decision')}`",
        f"- Transition plan: `{transition.get('decision')}`",
        "",
        "## Report Completeness",
        f"- Complete: `{review['report_completeness']['complete']}`",
        f"- Missing required files: `{review['report_completeness']['missing_required_files']}`",
        f"- Missing selected profiles: `{len(review['report_completeness']['missing_selected_profiles'])}`",
        "",
        "## Training",
        f"- Executed scopes: `{review['training'].get('training_executed_count')}`",
        f"- Reused scopes: `{review['training'].get('training_skipped_count')}`",
        f"- All scopes reused: `{review['training'].get('all_training_scopes_reused')}`",
        "",
        "## CV Read",
        f"- Control profile: `{review['control_profile']}`",
        _row_metric_line(review["cv"]["control"]),
        "- Best full profile by mean IC:",
        _row_metric_line(review["cv"]["best_mean_rank_ic_profile"]),
        "- Best full profile by top-10 lift:",
        _row_metric_line(review["cv"]["best_top_10_lift_profile"]),
        f"- CV promotable candidates: `{len(review['cv']['cv_promotable_candidates'])}`",
        "",
        "## Blend Read",
        f"- Blend rows: `{review['blends']['count']}`",
        "- Best blend by mean IC:",
        _row_metric_line(review["blends"]["best_mean_rank_ic_blend"]),
        "- Best blend by top-10 lift:",
        _row_metric_line(review["blends"]["best_top_10_lift_blend"]),
        "",
        "## Holdout Policy",
        f"- Future OOS ready: `{policy.get('future_oos_ready')}`",
        f"- Future OOS preferred ready: `{policy.get('future_oos_preferred_ready')}`",
        f"- New bars since anchor: `{policy.get('new_bars_since_anchor')}`",
        f"- Min bars remaining: `{policy.get('min_new_bars_remaining')}`",
        f"- Preferred bars remaining: `{policy.get('preferred_new_bars_remaining')}`",
        f"- Min ready at: `{policy.get('min_ready_at')}`",
        f"- Preferred ready at: `{policy.get('preferred_ready_at')}`",
        "",
        "## Holdout Diagnostic Read",
        "- Best holdout row by mean IC:",
        _row_metric_line(review["holdout"]["best"].get("best_mean_rank_ic", {})),
        "- Best holdout row by top-10 lift:",
        _row_metric_line(review["holdout"]["best"].get("best_top_10_lift", {})),
        "",
        "## Fold And Threshold Forensics",
        f"- Control rank IC std: `{_fmt(fold_summary.get('rank_ic_std'))}`",
        f"- Control bad folds: `{fold_summary.get('bad_fold_count', '')}`",
        f"- Control negative folds: `{fold_summary.get('negative_fold_count', '')}`",
        f"- Top std-driver fold: `{std_driver.get('fold', '')}` with rank IC `{_fmt(std_driver.get('rank_ic'))}`",
        f"- Top std-driver issue: `{std_driver.get('primary_issue', '')}`",
        f"- Threshold issue counts: `{forensics.get('threshold_issue_counts', {})}`",
        "",
        "## Interpretation",
        "- Treat current holdout results as diagnostics unless `future_oos_ready` is true.",
        "- Promote only candidates that beat the control on CV gates and then survive future unseen bars.",
        "- If IC is high but fold stability is worse than control, keep it as a watch candidate rather than a new base.",
    ]
    return "\n".join(lines) + "\n"


def phase2_readiness_markdown(readiness: dict[str, Any]) -> str:
    lines = [
        "# Phase 2 Readiness",
        "",
        f"- Decision: `{readiness.get('decision')}`",
        f"- Ready: `{readiness.get('ready_for_phase2')}`",
        f"- Next action: `{readiness.get('next_action')}`",
        f"- Blockers: `{';'.join(readiness.get('blockers', [])) or 'none'}`",
        f"- Advisories: `{';'.join(readiness.get('advisories', [])) or 'none'}`",
        f"- Long F1 source: `{readiness.get('long_f1_source')}`",
        "",
        "## Checks",
        "",
        "| check | passed | value | target | blocker | source |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in readiness.get("checks", []) or []:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("check", "")),
                    str(item.get("passed", "")),
                    str(item.get("value", "")),
                    str(item.get("target", "")),
                    str(item.get("blocker", "")),
                    str(item.get("source", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def phase1_transition_plan_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# Phase 1 Transition Plan",
        "",
        f"- Decision: `{plan.get('decision')}`",
        f"- Research ready without Phase 2: `{plan.get('research_ready_without_phase2')}`",
        f"- Ready for Phase 2: `{plan.get('ready_for_phase2')}`",
        f"- Phase 2 blockers: `{';'.join(plan.get('phase2_blockers', [])) or 'none'}`",
        f"- Phase 2 advisories: `{';'.join(plan.get('phase2_advisories', [])) or 'none'}`",
        f"- Long F1 source: `{plan.get('long_f1_source')}`",
        "",
        "## Metric Gaps",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key, value in (plan.get("metric_gaps") or {}).items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Allowed Actions", ""])
    for item in plan.get("allowed_actions", []) or []:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Blocked Actions", ""])
    for item in plan.get("blocked_actions", []) or []:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Recommended Focus", ""])
    for item in plan.get("recommended_focus", []) or []:
        lines.append(f"- `{item}`")
    future = plan.get("future_oos", {}) or {}
    lines.extend(
        [
            "",
            "## Future OOS",
            "",
            f"- Ready: `{future.get('ready')}`",
            f"- New bars since anchor: `{future.get('new_bars_since_anchor')}`",
            f"- Min bars remaining: `{future.get('min_new_bars_remaining')}`",
            f"- Preferred bars remaining: `{future.get('preferred_new_bars_remaining')}`",
            f"- Min ready at: `{future.get('min_ready_at')}`",
            f"- Preferred ready at: `{future.get('preferred_ready_at')}`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_auto_review(report_dir: str | Path) -> dict[str, Any]:
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)
    review = review_experiment_report(report_path)
    auto_review_path = report_path / "auto_review.md"
    next_actions_path = report_path / "next_actions.json"
    auto_review_path.write_text(auto_review_markdown(review), encoding="utf-8")
    phase2_readiness_path = report_path / "phase2_readiness.json"
    phase2_readiness_md_path = report_path / "phase2_readiness.md"
    transition_plan_path = report_path / "phase1_transition_plan.json"
    transition_plan_md_path = report_path / "phase1_transition_plan.md"
    phase2_readiness_path.write_text(
        json.dumps(_json_ready(review["phase2_readiness"]), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    phase2_readiness_md_path.write_text(phase2_readiness_markdown(review["phase2_readiness"]), encoding="utf-8")
    transition_plan_path.write_text(
        json.dumps(_json_ready(review["phase1_transition_plan"]), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    transition_plan_md_path.write_text(
        phase1_transition_plan_markdown(review["phase1_transition_plan"]),
        encoding="utf-8",
    )
    next_actions_path.write_text(
        json.dumps(_json_ready(review["next_action"]), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    review_path = report_path / "auto_review.json"
    review_path.write_text(json.dumps(_json_ready(review), indent=2, sort_keys=True), encoding="utf-8")
    return {
        "review": review,
        "auto_review_path": str(auto_review_path),
        "auto_review_json_path": str(review_path),
        "next_actions_path": str(next_actions_path),
        "phase2_readiness_path": str(phase2_readiness_path),
        "phase2_readiness_md_path": str(phase2_readiness_md_path),
        "phase1_transition_plan_path": str(transition_plan_path),
        "phase1_transition_plan_md_path": str(transition_plan_md_path),
    }


def _resolve_input(path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if path.is_dir():
        return path, None
    temp = tempfile.TemporaryDirectory()
    with zipfile.ZipFile(path) as archive:
        archive.extractall(temp.name)
    roots = [item for item in Path(temp.name).iterdir() if item.is_dir()]
    if len(roots) == 1:
        return roots[0], temp
    return Path(temp.name), temp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a Phase 1 auto-review for an experiment report directory or zip.")
    parser.add_argument("path", help="Experiment report directory or phase1 experiment bundle zip.")
    args = parser.parse_args(argv)
    input_path = Path(args.path)
    report_dir, temp = _resolve_input(input_path)
    try:
        result = write_auto_review(report_dir)
        if temp is not None:
            auto_review_copy = input_path.with_name(f"{input_path.stem}_auto_review.md")
            next_actions_copy = input_path.with_name(f"{input_path.stem}_next_actions.json")
            shutil.copyfile(result["auto_review_path"], auto_review_copy)
            shutil.copyfile(result["next_actions_path"], next_actions_copy)
            result["auto_review_path"] = str(auto_review_copy)
            result["next_actions_path"] = str(next_actions_copy)
        print(result["auto_review_path"])
        print(result["next_actions_path"])
    finally:
        if temp is not None:
            temp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
