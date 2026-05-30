from __future__ import annotations

from __future__ import annotations
import copy
from itertools import combinations
from typing import Any
import numpy as np
import pandas as pd

from .utils import _cfg, _set_cfg, _slug, _prediction_key_columns, _hash_payload, _float, _metric_or, summarize_profile_predictions

def _rank_score_by_fold(frame: pd.DataFrame) -> pd.Series:
    group_keys = [column for column in ("split", "fold") if column in frame.columns]
    if not group_keys:
        return frame["prob_long"].rank(method="average", pct=True)
    return frame.groupby(group_keys, dropna=False)["prob_long"].rank(method="average", pct=True)

def _profile_blend_predictions(
    entries: list[dict[str, Any]],
    *,
    method: str,
    weights: list[float] | None = None,
) -> pd.DataFrame:
    if len(entries) < 2:
        return pd.DataFrame()

    normalized_weights: list[float] | None = None
    if weights is not None:
        if len(weights) != len(entries):
            raise ValueError("Blend weights must match the number of profile entries")
        raw_weights = np.asarray(weights, dtype=float)
        if not np.isfinite(raw_weights).all() or (raw_weights < 0).any() or raw_weights.sum() <= 0:
            raise ValueError("Blend weights must be finite non-negative values with a positive sum")
        normalized_weights = (raw_weights / raw_weights.sum()).tolist()

    frames = []
    profiles = []
    for idx, entry in enumerate(entries):
        prediction = entry["predictions"].copy()
        profile = str(entry["profile"])
        prediction["_blend_profile"] = profile
        prediction["_blend_weight"] = 1.0 if normalized_weights is None else float(normalized_weights[idx])
        if method in {"rank_mean", "rank_weighted"}:
            prediction["_blend_score"] = _rank_score_by_fold(prediction)
        elif method in {"prob_mean", "prob_weighted"}:
            prediction["_blend_score"] = prediction["prob_long"].astype(float)
        else:
            raise ValueError(f"Unknown profile blend method: {method}")
        prediction["_blend_weighted_score"] = prediction["_blend_score"] * prediction["_blend_weight"]
        frames.append(prediction)
        profiles.append(profile)
    if len(frames) < 2:
        return pd.DataFrame()

    stacked = pd.concat(frames, ignore_index=True)
    key_columns = _prediction_key_columns(stacked)
    if not {"fold", "timestamp"}.issubset(key_columns):
        return pd.DataFrame()

    profile_count = len(set(profiles))
    grouped = stacked.groupby(key_columns, dropna=False)
    if normalized_weights is None:
        stats = grouped["_blend_score"].agg(
            prob_long_blend="mean",
            prob_long_profile_std="std",
            prob_long_profile_min="min",
            prob_long_profile_max="max",
            blend_profile_count="count",
        ).reset_index()
    else:
        stats = grouped.agg(
            prob_long_blend=("_blend_weighted_score", "sum"),
            prob_long_profile_std=("_blend_score", "std"),
            prob_long_profile_min=("_blend_score", "min"),
            prob_long_profile_max=("_blend_score", "max"),
            blend_profile_count=("_blend_score", "count"),
        ).reset_index()
    stats = stats.loc[stats["blend_profile_count"] == profile_count].copy()
    if stats.empty:
        return pd.DataFrame()

    base = grouped.first().reset_index()
    drop_columns = ["_blend_profile", "_blend_score", "_blend_weight", "_blend_weighted_score", "prob_long_blend"]
    base = base.drop(columns=[column for column in drop_columns if column in base.columns], errors="ignore")
    base = base.merge(stats, on=key_columns, how="inner")
    base["prob_long"] = base["prob_long_blend"]
    regime_columns = [column for column in stacked.columns if column.startswith("regime_prob_")]
    if regime_columns:
        regime_avg = grouped[regime_columns].mean().reset_index()
        base = base.drop(columns=[column for column in regime_columns if column in base.columns]).merge(
            regime_avg,
            on=key_columns,
            how="left",
        )
    base = base.drop(columns=["prob_long_blend"], errors="ignore")
    base["blend_method"] = method
    base["blend_profiles"] = ",".join(profiles)
    if normalized_weights is not None:
        base["blend_weights"] = ",".join(f"{weight:.6g}" for weight in normalized_weights)
    return base.sort_values(key_columns).reset_index(drop=True)

def _blend_entry_config(config: dict[str, Any], profile: str, feature_columns: list[str], *, description: str) -> dict[str, Any]:
    blend_cfg = copy.deepcopy(config)
    profiles = copy.deepcopy(_cfg(blend_cfg, ["features", "profiles"], {}) or {})
    profiles[profile] = {
        "description": description,
        "include_patterns": list(feature_columns),
        "exclude_patterns": [],
    }
    _set_cfg(blend_cfg, ["features", "profiles"], profiles)
    return blend_cfg

def _profile_blend_entries(entries: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    full_entries = [
        entry
        for entry in entries
        if str(entry.get("fold_scope", "")) == "full"
        and not str(entry.get("profile", "")).startswith("blend_")
    ]
    if len(full_entries) < 2:
        return []

    blend_settings = _cfg(config, ["experiments", "profile_blends"], {}) or {}
    include_auto_equal = bool(blend_settings.get("include_auto_equal_weight", True))
    include_auto_rank = bool(blend_settings.get("include_auto_rank_mean", True))
    blend_entries = []
    entry_by_profile = {str(entry["profile"]): entry for entry in full_entries}

    def append_blend(
        combo: list[dict[str, Any]],
        *,
        method: str,
        weights: list[float] | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        profiles = [str(entry["profile"]) for entry in combo]
        feature_columns = sorted({column for entry in combo for column in entry.get("feature_columns", [])})
        combo_hash = _hash_payload({"profiles": profiles, "method": method, "weights": weights})[:10]
        predictions = _profile_blend_predictions(combo, method=method, weights=weights)
        if predictions.empty:
            return
        profile = _slug(name) if name else f"blend_{method}_{combo_hash}"
        if not profile.startswith("blend_"):
            profile = f"blend_{profile}"
        blend_cfg = _blend_entry_config(
            config,
            profile,
            feature_columns,
            description=description or f"Diagnostic {method} blend of: {', '.join(profiles)}",
        )
        diagnostics = summarize_profile_predictions(
            predictions,
            blend_cfg,
            profile=profile,
            feature_columns=feature_columns,
            fold_scope=f"blend_{method}",
        )
        diagnostics["row"]["blend_profiles"] = ",".join(profiles)
        diagnostics["row"]["blend_method"] = method
        diagnostics["row"]["profile_count"] = len(profiles)
        if weights is not None:
            raw_weights = np.asarray(weights, dtype=float)
            normalized = raw_weights / raw_weights.sum()
            diagnostics["row"]["blend_weights"] = ",".join(f"{weight:.6g}" for weight in normalized)
        diagnostics["row"]["prob_long_profile_std_mean"] = float(
            pd.to_numeric(predictions["prob_long_profile_std"], errors="coerce").mean()
        )
        diagnostics["row"]["prob_long_profile_std_p90"] = float(
            pd.to_numeric(predictions["prob_long_profile_std"], errors="coerce").quantile(0.90)
        )
        blend_entries.append(
            {
                "profile": profile,
                "fold_scope": f"blend_{method}",
                "feature_columns": feature_columns,
                "predictions": predictions,
                "diagnostics": diagnostics,
                "summary": diagnostics["row"],
                "config": blend_cfg,
            }
        )

    if include_auto_equal:
        combos = list(combinations(full_entries, 2))
        if len(full_entries) > 2:
            combos.append(tuple(full_entries))
        for combo in combos:
            methods = ["prob_mean"]
            if include_auto_rank:
                methods.append("rank_mean")
            for method in methods:
                append_blend(list(combo), method=method)

    for spec in blend_settings.get("weighted", []) or []:
        if not isinstance(spec, dict) or not bool(spec.get("enabled", True)):
            continue
        profiles = [str(profile) for profile in spec.get("profiles", []) or []]
        if len(profiles) < 2:
            continue
        if any(profile not in entry_by_profile for profile in profiles):
            continue
        method = str(spec.get("method", "prob_weighted"))
        weights = [float(weight) for weight in spec.get("weights", []) or []]
        append_blend(
            [entry_by_profile[profile] for profile in profiles],
            method=method,
            weights=weights,
            name=str(spec.get("name", "")) or None,
            description=str(spec.get("description", "")) or None,
        )
    return blend_entries

def _profile_blend_frame(entries: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for entry in entries:
        if not str(entry.get("fold_scope", "")).startswith("blend_"):
            continue
        row = dict(entry["diagnostics"]["row"])
        predictions = entry.get("predictions", pd.DataFrame())
        if isinstance(predictions, pd.DataFrame) and not predictions.empty:
            row["blend_profiles"] = str(predictions["blend_profiles"].iloc[0]) if "blend_profiles" in predictions.columns else row.get("blend_profiles", "")
            row["blend_method"] = str(predictions["blend_method"].iloc[0]) if "blend_method" in predictions.columns else row.get("blend_method", "")
            row["blend_weights"] = str(predictions["blend_weights"].iloc[0]) if "blend_weights" in predictions.columns else row.get("blend_weights", "")
            row["profile_count"] = int(predictions["blend_profile_count"].max()) if "blend_profile_count" in predictions.columns else row.get("profile_count", 0)
            row["prob_long_profile_std_mean"] = float(pd.to_numeric(predictions["prob_long_profile_std"], errors="coerce").mean())
            row["prob_long_profile_std_p90"] = float(pd.to_numeric(predictions["prob_long_profile_std"], errors="coerce").quantile(0.90))
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()

def _profile_blend_review_frame(
    profile_blend: pd.DataFrame,
    comparison: pd.DataFrame,
    config: dict[str, Any],
    control_profile: str,
) -> pd.DataFrame:
    if profile_blend.empty:
        return profile_blend
    reviewed = profile_blend.copy()
    control_rows = comparison[
        (comparison["profile"] == control_profile)
        & (comparison["fold_scope"] == "full")
    ]
    if control_rows.empty:
        reviewed["control_profile"] = control_profile
        reviewed["reviewable"] = False
        reviewed["review_reason"] = "missing_full_control"
        return reviewed

    control = control_rows.iloc[0].to_dict()
    gates = _cfg(config, ["experiments", "profile_blend_review_gates"], {}) or {}
    min_mean_delta = float(gates.get("min_mean_rank_ic_delta", 0.005))
    max_std_delta = float(gates.get("max_std_rank_ic_delta", 0.0))
    min_positive = float(gates.get("min_positive_ic_fraction", 0.70))
    min_top_delta = float(gates.get("min_top_10_lift_global_delta", 0.02))
    leader_gates = _cfg(config, ["experiments", "profile_blend_leader_gates"], {}) or {}
    tail_lift_gates = leader_gates.get("tail_lift", gates) or gates
    stability_gates = leader_gates.get("stability", {}) or {}
    balanced_gates = leader_gates.get("balanced")

    rows = []
    for _, item in reviewed.iterrows():
        row = item.to_dict()
        row["control_profile"] = control_profile
        row["mean_rank_ic_delta_vs_control"] = _float(row, "mean_rank_ic") - _float(control, "mean_rank_ic")
        row["std_rank_ic_delta_vs_control"] = _float(row, "std_rank_ic") - _float(control, "std_rank_ic")
        row["positive_ic_fraction_delta_vs_control"] = _float(row, "positive_ic_fraction") - _float(control, "positive_ic_fraction")
        row["top_10_lift_global_delta_vs_control"] = _float(row, "top_10_lift_global") - _float(control, "top_10_lift_global")
        row["selected_threshold_f1_delta_vs_control"] = _metric_or(
            row,
            "test_f1_at_official_threshold",
            _metric_or(
                row,
                "test_f1_at_constrained_threshold",
                _metric_or(row, "test_f1_at_selected_threshold", _float(row, "mean_long_f1")),
            ),
        ) - _metric_or(
            control,
            "test_f1_at_official_threshold",
            _metric_or(
                control,
                "test_f1_at_constrained_threshold",
                _metric_or(control, "test_f1_at_selected_threshold", _float(control, "mean_long_f1")),
            ),
        )
        row["worst_5_rank_ic_delta_vs_control"] = _float(row, "worst_5_rank_ic_mean") - _float(control, "worst_5_rank_ic_mean")
        reasons = []
        if row["mean_rank_ic_delta_vs_control"] < min_mean_delta:
            reasons.append("mean_rank_ic_delta")
        if row["std_rank_ic_delta_vs_control"] > max_std_delta:
            reasons.append("std_rank_ic_delta")
        if _float(row, "positive_ic_fraction") < min_positive:
            reasons.append("positive_ic_fraction")
        if row["top_10_lift_global_delta_vs_control"] < min_top_delta:
            reasons.append("top_10_lift_global_delta")
        if not bool(row.get("mtf_leakage_passed", False)):
            reasons.append("mtf_leakage")
        if not bool(row.get("stationarity_policy_passed", False)):
            reasons.append("stationarity_policy")
        row["reviewable"] = not reasons
        row["review_reason"] = ";".join(reasons)
        tail_lift_reasons = _profile_blend_gate_reasons(row, tail_lift_gates)
        stability_reasons = _profile_blend_gate_reasons(row, stability_gates)
        balanced_reasons = _profile_blend_gate_reasons(row, balanced_gates or {})
        row["tail_lift_eligible"] = not tail_lift_reasons
        row["tail_lift_reason"] = ";".join(tail_lift_reasons)
        row["stability_eligible"] = not stability_reasons
        row["stability_reason"] = ";".join(stability_reasons)
        row["balanced_eligible"] = bool(balanced_gates) and not balanced_reasons
        row["balanced_reason"] = ";".join(balanced_reasons if balanced_gates else ["not_configured"])
        rows.append(row)

    if not rows:
        return reviewed
    frame = (
        pd.DataFrame(rows)
        .sort_values(
            ["reviewable", "mean_rank_ic", "top_10_lift_global", "worst_5_rank_ic_mean"],
            ascending=[False, False, False, False],
        )
        .reset_index(drop=True)
    )
    return _mark_profile_blend_leaders(frame)

def _profile_blend_gate_reasons(row: dict[str, Any], gates: dict[str, Any]) -> list[str]:
    reasons = []
    if not gates:
        return reasons
    checks = [
        ("min_mean_rank_ic_delta", "mean_rank_ic_delta_vs_control", "mean_rank_ic_delta", "min"),
        ("max_std_rank_ic_delta", "std_rank_ic_delta_vs_control", "std_rank_ic_delta", "max"),
        ("min_positive_ic_fraction", "positive_ic_fraction", "positive_ic_fraction", "min"),
        ("min_top_10_lift_global", "top_10_lift_global", "top_10_lift_global", "min"),
        ("min_top_10_lift_global_delta", "top_10_lift_global_delta_vs_control", "top_10_lift_global_delta", "min"),
        ("min_worst_5_rank_ic_delta", "worst_5_rank_ic_delta_vs_control", "worst_5_rank_ic_delta", "min"),
    ]
    for gate_key, metric_key, reason, direction in checks:
        if gate_key not in gates:
            continue
        value = _float(row, metric_key)
        gate = float(gates[gate_key])
        if direction == "min" and value < gate:
            reasons.append(reason)
        if direction == "max" and value > gate:
            reasons.append(reason)
    if not bool(row.get("mtf_leakage_passed", False)):
        reasons.append("mtf_leakage")
    if not bool(row.get("stationarity_policy_passed", False)):
        reasons.append("stationarity_policy")
    return reasons

def _select_profile_blend_leader(profile_blend: pd.DataFrame, role: str) -> dict[str, Any]:
    if profile_blend.empty:
        return {}
    if role == "tail_lift":
        eligible_column = "tail_lift_eligible"
        sort_columns = ["top_10_lift_global", "mean_rank_ic", "worst_5_rank_ic_mean"]
        ascending = [False, False, False]
    elif role == "stability":
        eligible_column = "stability_eligible"
        sort_columns = ["mean_rank_ic", "worst_5_rank_ic_mean", "std_rank_ic", "positive_ic_fraction"]
        ascending = [False, False, True, False]
    elif role == "balanced":
        eligible_column = "balanced_eligible"
        sort_columns = [
            "mean_rank_ic",
            "std_rank_ic",
            "positive_ic_fraction",
            "top_10_lift_global",
            "worst_5_rank_ic_mean",
        ]
        ascending = [False, True, False, False, False]
    else:
        raise ValueError(f"Unknown profile blend leader role: {role}")
    if eligible_column not in profile_blend.columns:
        return {}
    candidates = profile_blend[profile_blend[eligible_column].astype(bool)].copy()
    if candidates.empty:
        return {}
    candidates = candidates.sort_values(sort_columns, ascending=ascending)
    return candidates.iloc[0].to_dict()

def _profile_blend_leaders(profile_blend: pd.DataFrame) -> dict[str, Any]:
    leaders = {
        "balanced_leader": _select_profile_blend_leader(profile_blend, "balanced"),
        "tail_lift_leader": _select_profile_blend_leader(profile_blend, "tail_lift"),
        "stability_leader": _select_profile_blend_leader(profile_blend, "stability"),
    }
    return {key: value for key, value in leaders.items() if value}

def _mark_profile_blend_leaders(profile_blend: pd.DataFrame) -> pd.DataFrame:
    if profile_blend.empty:
        return profile_blend
    marked = profile_blend.copy()
    marked["balanced_leader"] = False
    marked["tail_lift_leader"] = False
    marked["stability_leader"] = False
    leaders = _profile_blend_leaders(marked)
    for role, leader in leaders.items():
        profile = str(leader.get("profile", ""))
        if not profile:
            continue
        marked.loc[marked["profile"] == profile, role] = True
    roles = []
    for _, row in marked.iterrows():
        item_roles = []
        if bool(row.get("balanced_leader", False)):
            item_roles.append("balanced")
        if bool(row.get("tail_lift_leader", False)):
            item_roles.append("tail_lift")
        if bool(row.get("stability_leader", False)):
            item_roles.append("stability")
        roles.append(",".join(item_roles))
    marked["leader_roles"] = roles
    return marked

def _best_profile_blend(profile_blend: pd.DataFrame) -> dict[str, Any]:
    leaders = _profile_blend_leaders(profile_blend)
    return leaders.get("balanced_leader") or leaders.get("tail_lift_leader") or leaders.get("stability_leader") or {}

