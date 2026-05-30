from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from fnmatch import fnmatch
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

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
    write_phase1_diagnostic_bundle,
)
from yenibot.features import filter_feature_columns, resolve_feature_profile, select_feature_columns
from yenibot.training import run_walk_forward_training
from yenibot.training.trainer import _add_regime_probs, _build_model, _device, _make_dataset, _predict_dataset


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


def _holdout_policy_action(
    *,
    frozen: dict[str, Any],
    observed_policy: dict[str, Any],
    frozen_selection: str,
    config: dict[str, Any] | None = None,
    holdout_boundary_passed: bool = True,
) -> str:
    if not holdout_boundary_passed:
        return "invalid_holdout_training_boundary_rerun_04"
    policy_status = str(_cfg(config or {}, ["experiments", "policy_review", "status"], "")).lower()
    if any(token in policy_status for token in ("failed", "invalidated", "retired")):
        return "retired_frozen_policy_keep_control_profile"
    frozen_consistent = bool(frozen.get("holdout_policy_consistency_pass", False))
    frozen_signal = bool(frozen.get("holdout_signal_pass", False))
    frozen_threshold = bool(frozen.get("holdout_threshold_pass", False))
    observed_consistent = bool(observed_policy.get("holdout_policy_consistency_pass", False))
    observed_name = str(observed_policy.get("candidate", ""))
    threshold_allowed = bool(_cfg(config or {}, ["experiments", "policy_review", "threshold_deployment_allowed"], False))
    if frozen_consistent and frozen_signal and frozen_threshold and threshold_allowed:
        return "review_frozen_threshold_and_score_policy"
    if frozen_consistent and frozen_signal:
        return "review_frozen_score_band_policy_only_no_threshold_deployment"
    if observed_consistent and observed_name and observed_name != frozen_selection:
        return "holdout_only_candidate_do_not_promote_without_future_oos"
    return "keep_control_profile"


def profile_config(config: dict[str, Any], profile: str) -> dict[str, Any]:
    """Return an in-memory config copy with the requested active feature profile."""

    updated = copy.deepcopy(config)
    _set_cfg(updated, ["features", "active_profile"], profile)
    resolve_feature_profile(updated)
    overrides = _profile_config_overrides(updated, profile)
    if overrides:
        _deep_update(updated, overrides)
    return updated


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _profile_config_overrides(config: dict[str, Any], profile: str) -> dict[str, Any]:
    profiles = _cfg(config, ["features", "profiles"], {}) or {}
    if not isinstance(profiles, dict):
        return {}

    def load(name: str, seen: set[str] | None = None) -> dict[str, Any]:
        seen = set() if seen is None else seen
        if name in seen:
            raise ValueError(f"Cyclic feature profile inheritance detected at {name}")
        seen.add(name)
        current = profiles.get(name)
        if not isinstance(current, dict):
            return {}
        parent_name = current.get("inherit")
        overrides = load(str(parent_name), seen) if parent_name else {}
        current_overrides = current.get("config_overrides", current.get("training_overrides", {})) or {}
        if not isinstance(current_overrides, dict):
            raise ValueError(f"Feature profile config_overrides must be a mapping: {name}")
        return _deep_update(overrides, current_overrides)

    return load(str(profile))


def _profile_rejection_reason(profile: str, experiments: dict[str, Any]) -> str:
    memory = experiments.get("experiment_memory", {}) or {}
    if not bool(memory.get("enabled", False)) or not bool(memory.get("reject_retests", True)):
        return ""
    if profile in {str(item) for item in memory.get("allow_retest_profiles", []) or []}:
        return ""

    rejected_profiles = memory.get("rejected_profiles", {}) or {}
    if isinstance(rejected_profiles, dict) and profile in rejected_profiles:
        value = rejected_profiles[profile]
        if isinstance(value, dict):
            return str(value.get("reason") or "historically_rejected_profile")
        return str(value or "historically_rejected_profile")

    for item in memory.get("rejected_profile_patterns", []) or []:
        if isinstance(item, str):
            pattern = item
            reason = "historically_rejected_profile_pattern"
        elif isinstance(item, dict):
            pattern = str(item.get("pattern", ""))
            reason = str(item.get("reason") or "historically_rejected_profile_pattern")
        else:
            continue
        if pattern and fnmatch(profile, pattern):
            return reason
    return ""


def _filter_memory_rejected_profiles(
    profiles: list[str],
    experiments: dict[str, Any],
    *,
    role: str,
    protected_profiles: set[str] | None = None,
) -> tuple[list[str], list[dict[str, str]]]:
    protected = set() if protected_profiles is None else {str(profile) for profile in protected_profiles}
    selected: list[str] = []
    skipped: list[dict[str, str]] = []
    for profile in profiles:
        profile = str(profile)
        if profile in selected:
            continue
        reason = "" if profile in protected else _profile_rejection_reason(profile, experiments)
        if reason:
            skipped.append({"profile": profile, "role": role, "skip_reason": reason})
            continue
        selected.append(profile)
    return selected, skipped


def _policy_status_is_retired_or_failed(status: str) -> bool:
    return any(token in str(status).lower() for token in ("failed", "invalidated", "retired"))


def _future_oos_allowed_benchmark_profiles(config: dict[str, Any], control_profile: str) -> list[str]:
    policy_review = _cfg(config, ["experiments", "policy_review"], {}) or {}
    future_items = {str(item) for item in policy_review.get("future_oos_candidates", []) or []}
    profiles_cfg = _cfg(config, ["features", "profiles"], {}) or {}
    allowed = [str(control_profile)]

    for item in future_items:
        if item in profiles_cfg and item not in allowed:
            allowed.append(item)

    for blend in (_cfg(config, ["experiments", "profile_blends", "weighted"], []) or []):
        if not isinstance(blend, dict):
            continue
        name = str(blend.get("name", ""))
        candidate_names = {name, f"blend_{name}"}
        if not candidate_names.intersection(future_items):
            continue
        for profile in blend.get("profiles", []) or []:
            profile = str(profile)
            if profile and profile not in allowed:
                allowed.append(profile)
    return allowed


def _experiment_policy_guard(settings: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    policy_review = _cfg(config, ["experiments", "policy_review"], {}) or {}
    holdout = settings.get("holdout", {}) or _cfg(config, ["experiments", "holdout"], {}) or {}
    latest_data_end = _holdout_latest_available_data_end(holdout)
    monitor_state = _future_oos_monitor_state(config, latest_data_end)
    control = str(settings.get("control_profile") or _cfg(config, ["experiments", "control_profile"], ""))
    status = str(policy_review.get("status", ""))
    enabled = bool(policy_review.get("enabled", False))
    locked = bool(
        enabled
        and _policy_status_is_retired_or_failed(status)
        and monitor_state["holdout_roll_forward_locked"]
        and not monitor_state["future_oos_ready"]
    )
    allowed = _future_oos_allowed_benchmark_profiles(config, control)
    if locked:
        action = "wait_for_new_unseen_bars_keep_control_profile"
        reason = (
            "clean_holdout_policy_failed_and_future_oos_not_ready; "
            "profile search is locked to control/future-OOS benchmark profiles"
        )
    elif enabled and _policy_status_is_retired_or_failed(status) and monitor_state["future_oos_ready"]:
        action = "future_oos_window_available_review_predefined_candidates"
        reason = "future_oos_minimum_window_available"
    else:
        action = "normal_experiment_flow"
        reason = ""
    return {
        "enabled": enabled,
        "status": status,
        "profile_search_locked": locked,
        "action": action,
        "reason": reason,
        "allowed_benchmark_profiles": allowed,
        **monitor_state,
    }


def _apply_experiment_policy_guard(settings: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(settings)
    guard = _experiment_policy_guard(updated, config)
    blocked_candidates: list[str] = []
    blocked_full: list[str] = []
    blocked_seed: list[str] = []

    if guard["profile_search_locked"]:
        control = str(updated["control_profile"])
        allowed = set(str(profile) for profile in guard["allowed_benchmark_profiles"])
        candidates = [str(profile) for profile in updated.get("candidate_profiles", []) or []]
        blocked_candidates = [profile for profile in candidates if profile != control]
        if blocked_candidates:
            updated["candidate_profiles"] = []
            updated["profiles"] = [control]

        filtered_full = []
        for profile in [str(item) for item in updated.get("always_full_profiles", []) or []]:
            if profile in allowed:
                filtered_full.append(profile)
            else:
                blocked_full.append(profile)
        if control not in filtered_full:
            filtered_full.insert(0, control)
        updated["always_full_profiles"] = list(dict.fromkeys(filtered_full))

        seed_audit = copy.deepcopy(updated.get("seed_audit", {}) or {})
        if seed_audit:
            filtered_seed = []
            for profile in [str(item) for item in seed_audit.get("profiles", []) or []]:
                if profile in allowed:
                    filtered_seed.append(profile)
                else:
                    blocked_seed.append(profile)
            if control not in filtered_seed:
                filtered_seed.insert(0, control)
            seed_audit["profiles"] = list(dict.fromkeys(filtered_seed))
            updated["seed_audit"] = seed_audit

        skipped = list(updated.get("skipped_profiles", []) or [])
        for role, profiles in (
            ("candidate_profile", blocked_candidates),
            ("always_full_profile", blocked_full),
            ("seed_audit_profile", blocked_seed),
        ):
            for profile in profiles:
                skipped.append(
                    {
                        "profile": profile,
                        "role": role,
                        "skip_reason": "future_oos_not_ready_profile_search_locked",
                    }
                )
        updated["skipped_profiles"] = skipped

    guard["blocked_candidate_profiles"] = blocked_candidates
    guard["blocked_full_profiles"] = blocked_full
    guard["blocked_seed_profiles"] = blocked_seed
    updated["experiment_policy_guard"] = guard
    return updated


def experiment_settings(config: dict[str, Any]) -> dict[str, Any]:
    experiments = copy.deepcopy(_cfg(config, ["experiments"], {}) or {})
    control = str(experiments.get("control_profile") or _cfg(config, ["features", "active_profile"]))
    raw_candidates = [str(profile) for profile in experiments.get("candidate_profiles", [])]
    candidates, skipped_candidates = _filter_memory_rejected_profiles(
        raw_candidates,
        experiments,
        role="candidate_profile",
        protected_profiles={control},
    )
    profiles = []
    for profile in [control, *candidates]:
        if profile not in profiles:
            profiles.append(profile)
    experiments.setdefault("mode", "staged")
    experiments["control_profile"] = control
    experiments["candidate_profiles"] = [profile for profile in profiles if profile != control]
    experiments["profiles"] = profiles
    experiments.setdefault("triage_fold_ids", [])
    experiments.setdefault("full_cv_profiles", "auto")
    experiments.setdefault("always_full_profiles", [control])
    always_full, skipped_always_full = _filter_memory_rejected_profiles(
        [str(profile) for profile in experiments.get("always_full_profiles", []) or []],
        experiments,
        role="always_full_profile",
        protected_profiles={control},
    )
    if control not in always_full:
        always_full.insert(0, control)
    experiments["always_full_profiles"] = always_full
    experiments.setdefault("max_auto_full_candidates", None)
    experiments.setdefault("resume_existing", True)
    experiments.setdefault("force_retrain", False)
    seed_audit = copy.deepcopy(experiments.get("seed_audit", {}) or {})
    seed_audit.setdefault("enabled", False)
    seed_audit.setdefault("profiles", [control])
    seed_profiles, skipped_seed_profiles = _filter_memory_rejected_profiles(
        [str(profile) for profile in seed_audit.get("profiles", []) or [control]],
        experiments,
        role="seed_audit_profile",
        protected_profiles={control},
    )
    seed_audit["profiles"] = seed_profiles or [control]
    seed_audit.setdefault("seeds", [])
    seed_audit.setdefault("fold_ids", experiments.get("triage_fold_ids", []))
    experiments["seed_audit"] = seed_audit
    experiments["skipped_profiles"] = [*skipped_candidates, *skipped_always_full, *skipped_seed_profiles]
    experiments = _apply_experiment_policy_guard(experiments, config)
    return experiments


def _profile_requires_intrahour_features(config: dict[str, Any], profile: str) -> bool:
    profile_cfg = profile_config(config, profile)
    resolved = resolve_feature_profile(profile_cfg)
    return any("ih15" in str(pattern) for pattern in resolved.get("include_patterns", []) or [])


def _profile_requires_futures_context_features(config: dict[str, Any], profile: str) -> bool:
    profile_cfg = profile_config(config, profile)
    resolved = resolve_feature_profile(profile_cfg)
    return any("fut_" in str(pattern) for pattern in resolved.get("include_patterns", []) or [])


def _missing_intrahour_include_patterns(config: dict[str, Any], profile: str, feature_columns: tuple[str, ...]) -> list[str]:
    profile_cfg = profile_config(config, profile)
    resolved = resolve_feature_profile(profile_cfg)
    patterns = [str(pattern) for pattern in resolved.get("include_patterns", []) or [] if "ih15_" in str(pattern)]
    return [
        pattern
        for pattern in patterns
        if not any(fnmatch(column, pattern) for column in feature_columns)
    ]


def _preflight_experiment_profiles(
    settings: dict[str, Any],
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Skip candidate profiles that cannot change the current feature matrix."""

    updated = copy.deepcopy(settings)
    base_columns = select_feature_columns(frame)
    control = str(updated["control_profile"])
    selected_profiles: list[str] = []
    skipped_profiles = list(updated.get("skipped_profiles", []) or [])
    seen_signatures: dict[tuple[str, ...], str] = {}
    profile_feature_columns: dict[str, tuple[str, ...]] = {}

    for profile in [str(item) for item in updated.get("profiles", [])]:
        cfg = profile_config(config, profile)
        feature_columns = tuple(filter_feature_columns(base_columns, cfg))
        profile_feature_columns[profile] = feature_columns
        has_intrahour = any(column.startswith("ih15_") for column in feature_columns)
        has_futures_context = any(column.startswith("fut_") for column in feature_columns)
        missing_intrahour_patterns = _missing_intrahour_include_patterns(config, profile, feature_columns)
        if profile != control and _profile_requires_intrahour_features(config, profile) and (
            not has_intrahour or missing_intrahour_patterns
        ):
            reason = "missing_intrahour_features_rerun_01_02_03"
            if missing_intrahour_patterns:
                reason = f"{reason}:{','.join(missing_intrahour_patterns[:6])}"
            skipped_profiles.append(
                {
                    "profile": profile,
                    "role": "candidate_profile",
                    "skip_reason": reason,
                }
            )
            continue
        if profile != control and _profile_requires_futures_context_features(config, profile) and not has_futures_context:
            skipped_profiles.append(
                {
                    "profile": profile,
                    "role": "candidate_profile",
                    "skip_reason": "missing_futures_context_features_rerun_01_02_03",
                }
            )
            continue
        duplicate_of = seen_signatures.get(feature_columns)
        has_config_overrides = bool(_profile_config_overrides(config, profile))
        if profile != control and duplicate_of and not has_config_overrides:
            skipped_profiles.append(
                {
                    "profile": profile,
                    "role": "candidate_profile",
                    "skip_reason": f"duplicate_feature_signature:{duplicate_of}",
                }
            )
            continue
        selected_profiles.append(profile)
        seen_signatures[feature_columns] = profile

    if control not in selected_profiles:
        raise ValueError(f"Control profile was removed during experiment preflight: {control}")

    selected_set = set(selected_profiles)
    updated["profiles"] = selected_profiles
    updated["candidate_profiles"] = [profile for profile in selected_profiles if profile != control]

    def profile_is_runnable(profile: str, role: str) -> bool:
        if profile == control or profile in selected_set:
            return True
        if profile not in profile_feature_columns:
            cfg = profile_config(config, profile)
            profile_feature_columns[profile] = tuple(filter_feature_columns(base_columns, cfg))
        feature_columns = profile_feature_columns[profile]
        if _profile_requires_intrahour_features(config, profile) and not any(
            column.startswith("ih15_") for column in feature_columns
        ):
            skipped_profiles.append(
                {
                    "profile": profile,
                    "role": role,
                    "skip_reason": "missing_intrahour_features_rerun_01_02_03",
                }
            )
            return False
        if _profile_requires_futures_context_features(config, profile) and not any(
            column.startswith("fut_") for column in feature_columns
        ):
            skipped_profiles.append(
                {
                    "profile": profile,
                    "role": role,
                    "skip_reason": "missing_futures_context_features_rerun_01_02_03",
                }
            )
            return False
        return True

    updated["always_full_profiles"] = [
        str(profile)
        for profile in updated.get("always_full_profiles", []) or []
        if profile_is_runnable(str(profile), "always_full_profile")
    ]
    seed_audit = copy.deepcopy(updated.get("seed_audit", {}) or {})
    if seed_audit:
        seed_audit["profiles"] = [
            str(profile)
            for profile in seed_audit.get("profiles", []) or []
            if profile_is_runnable(str(profile), "seed_audit_profile")
        ]
        updated["seed_audit"] = seed_audit
    updated["skipped_profiles"] = skipped_profiles
    return updated


def experiment_root(checkpoint_dir: str | Path) -> Path:
    return Path(checkpoint_dir) / "experiments"


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _experiment_signature(config: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    comparable_settings = copy.deepcopy(settings)
    comparable_settings.pop("run_id", None)
    comparable_settings.pop("experiment_policy_guard", None)
    return {
        "settings": comparable_settings,
        "feature_profiles": _cfg(config, ["features", "profiles"], {}),
        "model": _cfg(config, ["model"], {}),
        "training": _cfg(config, ["training"], {}),
        "walk_forward": _cfg(config, ["walk_forward"], {}),
        "validation": _cfg(config, ["validation"], {}),
    }


def _matching_latest_run(checkpoint_dir: str | Path, signature_hash: str) -> Path | None:
    root = experiment_root(checkpoint_dir)
    if not root.exists():
        return None
    runs = sorted([path for path in root.glob("*") if path.is_dir()], key=lambda path: path.name, reverse=True)
    for run in runs:
        manifest_path = run / "experiment_manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = _read_json(manifest_path)
        except json.JSONDecodeError:
            continue
        if manifest.get("signature_hash") == signature_hash:
            return run
    return None


def resolve_experiment_run_id(
    checkpoint_dir: str | Path,
    config: dict[str, Any],
    settings: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> tuple[str, str]:
    settings = experiment_settings(config) if settings is None else settings
    if run_id:
        return str(run_id), "explicit_argument"
    if settings.get("run_id"):
        return str(settings["run_id"]), "config"
    signature_hash = _hash_payload(_experiment_signature(config, settings))
    if bool(settings.get("resume_existing", True)) and not bool(settings.get("force_retrain", False)):
        existing = _matching_latest_run(checkpoint_dir, signature_hash)
        if existing is not None:
            return existing.name, "matching_existing"
    return new_run_id(), "new"


def latest_experiment_run(checkpoint_dir: str | Path) -> Path:
    root = experiment_root(checkpoint_dir)
    runs = sorted([path for path in root.glob("*") if path.is_dir()], key=lambda path: path.name)
    if not runs:
        raise FileNotFoundError(f"No experiment runs found under {root}")
    return runs[-1]


def profile_run_dir(checkpoint_dir: str | Path, run_id: str, profile: str) -> Path:
    return experiment_root(checkpoint_dir) / run_id / _slug(profile)


def _frame_window(frame: pd.DataFrame) -> dict[str, str]:
    if "timestamp" not in frame.columns or frame.empty:
        return {"data_start": "", "data_end": ""}
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    return {"data_start": str(timestamps.min()), "data_end": str(timestamps.max())}


def _training_signature(
    *,
    frame: pd.DataFrame,
    config: dict[str, Any],
    profile: str,
    feature_columns: list[str],
    fold_ids: list[int] | None,
    fold_scope: str,
) -> dict[str, Any]:
    return {
        "profile": profile,
        "fold_scope": fold_scope,
        "fold_ids": fold_ids,
        "feature_columns": feature_columns,
        "feature_columns_hash": _hash_payload(feature_columns),
        "config_hash": _hash_payload(config),
        "frame_rows": int(len(frame)),
        **_frame_window(frame),
    }


def _manifest_path(output_dir: Path) -> Path:
    return output_dir / "training_manifest.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


_TRAINING_EXECUTION_KEYS = (
    "run_id_source",
    "training_executed_count",
    "training_skipped_count",
    "all_training_scopes_reused",
    "reused_training_scopes",
)


def _training_execution_summary_path(run_dir: Path) -> Path:
    return run_dir / "training_execution_summary.json"


def _training_execution_summary(
    *,
    run_id: str,
    run_id_source: str | None,
    executed_results: list[dict[str, Any]],
    skipped_results: list[dict[str, Any]],
    profile_results: list[dict[str, Any]],
    seed_results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "run_id_source": run_id_source,
        "training_executed_count": int(len(executed_results)),
        "training_skipped_count": int(len(skipped_results)),
        "all_training_scopes_reused": bool(profile_results or seed_results) and len(executed_results) == 0,
        "reused_training_scopes": [
            {"profile": str(result["profile"]), "fold_scope": str(result["fold_scope"])}
            for result in skipped_results
        ],
        "executed_training_scopes": [
            {"profile": str(result["profile"]), "fold_scope": str(result["fold_scope"])}
            for result in executed_results
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _load_training_execution_summary(run_dir: Path, run_manifest: dict[str, Any]) -> dict[str, Any]:
    summary_path = _training_execution_summary_path(run_dir)
    if summary_path.exists():
        summary = _read_json(summary_path)
        summary["training_execution_metadata_source"] = "training_execution_summary"
        summary["training_execution_metadata_available"] = True
        return summary

    decision_path = run_dir / "decision_report.json"
    if decision_path.exists():
        prior_decision = _read_json(decision_path)
        if any(key in prior_decision for key in _TRAINING_EXECUTION_KEYS):
            summary = {key: prior_decision.get(key) for key in _TRAINING_EXECUTION_KEYS if key in prior_decision}
            summary["run_id"] = str(prior_decision.get("run_id") or run_dir.name)
            summary["training_execution_metadata_source"] = "prior_decision_report"
            summary["training_execution_metadata_available"] = any(
                key in summary for key in ("training_executed_count", "training_skipped_count")
            )
            return summary

    summary = {
        "run_id": str(run_manifest.get("run_id") or run_dir.name),
        "run_id_source": run_manifest.get("run_id_source"),
        "training_executed_count": None,
        "training_skipped_count": None,
        "all_training_scopes_reused": None,
        "reused_training_scopes": [],
        "executed_training_scopes": [],
        "training_execution_metadata_source": "run_manifest_only",
        "training_execution_metadata_available": False,
    }
    return summary


def _is_complete(output_dir: Path, expected_signature_hash: str) -> bool:
    manifest_path = _manifest_path(output_dir)
    predictions_path = output_dir / "predictions_all.parquet"
    if not manifest_path.exists() or not predictions_path.exists():
        return False
    manifest = _read_json(manifest_path)
    return bool(manifest.get("completed")) and manifest.get("signature_hash") == expected_signature_hash


def _test_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if "split" in predictions.columns:
        return predictions[predictions["split"] == "test"].copy()
    return predictions.copy()


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


def _threshold_summary_metric(threshold_summary: pd.DataFrame | None, metric: str) -> float:
    if threshold_summary is None or threshold_summary.empty:
        return np.nan
    if "metric" not in threshold_summary.columns or "mean" not in threshold_summary.columns:
        return np.nan
    row = threshold_summary.loc[threshold_summary["metric"].astype(str) == str(metric)]
    if row.empty:
        return np.nan
    return _float(row.iloc[0].to_dict(), "mean")


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


def run_profile_experiment(
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    profile: str,
    checkpoint_dir: str | Path,
    run_id: str,
    fold_scope: str,
    fold_ids: list[int] | None = None,
    resume_existing: bool = True,
    force_retrain: bool = False,
    device: str | None = None,
) -> dict[str, Any]:
    cfg = profile_config(config, profile)
    feature_columns = filter_feature_columns(select_feature_columns(frame), cfg)
    output_dir = profile_run_dir(checkpoint_dir, run_id, profile) / fold_scope
    signature = _training_signature(
        frame=frame,
        config=cfg,
        profile=profile,
        feature_columns=feature_columns,
        fold_ids=fold_ids,
        fold_scope=fold_scope,
    )
    signature_hash = _hash_payload(signature)
    skipped = False
    if resume_existing and not force_retrain and _is_complete(output_dir, signature_hash):
        predictions = pd.read_parquet(output_dir / "predictions_all.parquet")
        skipped = True
    else:
        result = run_walk_forward_training(
            frame,
            cfg,
            feature_columns=feature_columns,
            checkpoint_dir=output_dir,
            fold_ids=fold_ids,
            device=device,
        )
        predictions = result["predictions"]
        manifest = {
            **signature,
            "signature_hash": signature_hash,
            "completed": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "prediction_rows": int(len(predictions)),
        }
        _write_json(_manifest_path(output_dir), manifest)

    diagnostics = summarize_profile_predictions(
        predictions,
        config,
        profile=profile,
        feature_columns=feature_columns,
        fold_scope=fold_scope,
    )
    return {
        "profile": profile,
        "fold_scope": fold_scope,
        "output_dir": output_dir,
        "skipped": skipped,
        "feature_columns": feature_columns,
        "predictions": predictions,
        "diagnostics": diagnostics,
        "summary": diagnostics["row"],
    }


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


def _decision_rows(rows: list[dict[str, Any]], config: dict[str, Any], *, scope: str) -> list[dict[str, Any]]:
    if not rows:
        return []
    settings = experiment_settings(config)
    control_profile = settings["control_profile"]
    control = next(row for row in rows if row["profile"] == control_profile)
    decided = []
    for row in rows:
        updated = dict(row)
        if row["profile"] == control_profile:
            updated["promotable"] = False
            updated["reject_reason"] = "control_profile"
        elif scope == "triage":
            passed, reason = _passes_triage(row, control, config)
            updated["promotable"] = passed
            updated["reject_reason"] = reason
        else:
            passed, reason = _passes_full(row, control, config)
            updated["promotable"] = passed
            updated["reject_reason"] = reason
        decided.append(updated)
    return decided


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


def _comparison_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "profile",
        "fold_scope",
        "feature_count",
        "fold_count",
        "mean_rank_ic",
        "std_rank_ic",
        "positive_ic_fraction",
        "mean_long_f1",
        "test_f1_at_selected_threshold",
        "test_precision_at_selected_threshold",
        "test_recall_at_selected_threshold",
        "test_pred_long_rate_at_selected_threshold",
        "selected_threshold_mean",
        "test_f1_at_constrained_threshold",
        "test_precision_at_constrained_threshold",
        "test_recall_at_constrained_threshold",
        "test_pred_long_rate_at_constrained_threshold",
        "constrained_threshold_mean",
        "guarded_threshold_source",
        "guarded_threshold_reason",
        "test_f1_at_guarded_threshold",
        "test_precision_at_guarded_threshold",
        "test_recall_at_guarded_threshold",
        "test_pred_long_rate_at_guarded_threshold",
        "guarded_threshold_mean",
        "official_threshold_source",
        "official_threshold_reason",
        "test_f1_at_official_threshold",
        "test_precision_at_official_threshold",
        "test_recall_at_official_threshold",
        "test_pred_long_rate_at_official_threshold",
        "official_threshold_mean",
        "official_threshold_uses_calibration",
        "official_threshold_selection_score",
        "calibrated_guarded_threshold_source",
        "test_f1_at_calibrated_guarded_threshold",
        "test_pred_long_rate_at_calibrated_guarded_threshold",
        "mean_prauc",
        "calibration_separation",
        "recent_rank_ic_mean",
        "recent_rank_ic_min",
        "negative_ic_count",
        "negative_ic_fraction",
        "worst_5_rank_ic_mean",
        "rank_ic_cvar_20",
        "bad_fold_rank_ic_mean",
        "top_10_lift_fold_mean",
        "top_10_lift_global",
        "top_10_positive_lift_fold_rate",
        "top_10_bad_fold_lift_mean",
        "mtf_leakage_passed",
        "stationarity_policy_passed",
        "passed_phase1",
        "passed_phase1_selected_threshold",
        "passed_phase1_constrained_threshold",
        "passed_phase1_guarded_threshold",
        "passed_phase1_official_threshold",
        "promotable",
        "reject_reason",
        "data_start",
        "data_end",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[columns].sort_values(["fold_scope", "mean_rank_ic"], ascending=[True, False]).reset_index(drop=True)


def _best_candidate(comparison: pd.DataFrame, control_profile: str) -> dict[str, Any]:
    candidates = comparison[
        (comparison["profile"] != control_profile)
        & (comparison["fold_scope"] == "full")
        & (comparison["promotable"].astype(bool))
    ].copy()
    if candidates.empty:
        return {}
    candidates = candidates.sort_values(
        ["passed_phase1", "mean_rank_ic", "top_10_lift_global", "worst_5_rank_ic_mean"],
        ascending=[False, False, False, False],
    )
    return candidates.iloc[0].to_dict()


def _comparison_markdown(comparison: pd.DataFrame, decision: dict[str, Any]) -> str:
    lines = ["# Experiment Profile Comparison", ""]
    if comparison.empty:
        lines.append("No profile runs were found.")
    else:
        display_cols = [
            "profile",
            "fold_scope",
            "feature_count",
            "mean_rank_ic",
            "std_rank_ic",
            "positive_ic_fraction",
            "worst_5_rank_ic_mean",
            "mean_long_f1",
            "test_f1_at_selected_threshold",
            "test_f1_at_constrained_threshold",
            "test_f1_at_guarded_threshold",
            "guarded_threshold_source",
            "test_pred_long_rate_at_guarded_threshold",
            "test_f1_at_official_threshold",
            "official_threshold_source",
            "test_pred_long_rate_at_official_threshold",
            "test_f1_at_calibrated_guarded_threshold",
            "test_pred_long_rate_at_constrained_threshold",
            "top_10_lift_global",
            "top_10_bad_fold_lift_mean",
            "passed_phase1_selected_threshold",
            "passed_phase1_constrained_threshold",
            "passed_phase1_guarded_threshold",
            "passed_phase1_official_threshold",
            "promotable",
            "reject_reason",
        ]
        visible = comparison[display_cols].copy()
        lines.append("| " + " | ".join(display_cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(display_cols)) + " |")
        for _, row in visible.iterrows():
            values = [str(row[column]) for column in display_cols]
            lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "## Decision", "", json.dumps(_json_ready(decision), indent=2, sort_keys=True)])
    return "\n".join(lines)


def _write_decision_files(run_dir: Path, comparison: pd.DataFrame, decision: dict[str, Any]) -> None:
    comparison.to_csv(run_dir / "profile_comparison.csv", index=False)
    (run_dir / "profile_comparison.md").write_text(_comparison_markdown(comparison, decision), encoding="utf-8")
    _write_json(run_dir / "decision_report.json", decision)
    _write_json(run_dir / "best_candidate.json", decision.get("best_candidate") or {})


def _experiment_selection_frame(settings: dict[str, Any]) -> pd.DataFrame:
    columns = ["profile", "role", "selected", "expected_fold_scope", "skip_reason"]
    rows: list[dict[str, Any]] = [
        {
            "profile": str(settings["control_profile"]),
            "role": "control_profile",
            "selected": True,
            "expected_fold_scope": "triage",
            "skip_reason": "",
        }
    ]
    for role, key in (
        ("candidate_profile", "candidate_profiles"),
        ("always_full_profile", "always_full_profiles"),
        ("seed_audit_profile", "seed_audit_profiles"),
    ):
        values = settings.get(key, [])
        if key == "seed_audit_profiles":
            values = (settings.get("seed_audit", {}) or {}).get("profiles", [])
        expected_scope = {
            "candidate_profile": "triage",
            "always_full_profile": "full",
            "seed_audit_profile": "seed_audit",
        }[role]
        for profile in values or []:
            rows.append(
                {
                    "profile": str(profile),
                    "role": role,
                    "selected": True,
                    "expected_fold_scope": expected_scope,
                    "skip_reason": "",
                }
            )
    for skipped in settings.get("skipped_profiles", []) or []:
        rows.append(
            {
                "profile": str(skipped.get("profile", "")),
                "role": str(skipped.get("role", "skipped_profile")),
                "selected": False,
                "expected_fold_scope": "",
                "skip_reason": str(skipped.get("skip_reason", "")),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).drop_duplicates().reset_index(drop=True)


def _experiment_selection_markdown(selection: pd.DataFrame) -> str:
    lines = ["# Experiment Selection", ""]
    if selection.empty:
        lines.append("No profile selection metadata was produced.")
        return "\n".join(lines)
    lines.append("| profile | role | selected | expected_fold_scope | skip_reason |")
    lines.append("| --- | --- | --- | --- | --- |")
    for _, row in selection.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["profile"]),
                    str(row["role"]),
                    str(bool(row["selected"])),
                    str(row.get("expected_fold_scope", "")),
                    str(row.get("skip_reason", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _write_experiment_selection(path: Path, settings: dict[str, Any]) -> pd.DataFrame:
    path.mkdir(parents=True, exist_ok=True)
    selection = _experiment_selection_frame(settings)
    selection.to_csv(path / "experiment_selection.csv", index=False)
    (path / "experiment_selection.md").write_text(_experiment_selection_markdown(selection), encoding="utf-8")
    _write_json(
        path / "experiment_selection.json",
        {
            "control_profile": settings.get("control_profile", ""),
            "selected_profiles": selection.loc[selection["selected"].astype(bool), "profile"].drop_duplicates().tolist(),
            "skipped_profiles": settings.get("skipped_profiles", []) or [],
            "rows": selection.to_dict(orient="records"),
        },
    )
    return selection


def _missing_selected_profiles(selection: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    columns = ["profile", "role", "expected_fold_scope", "reason"]
    if selection.empty:
        return pd.DataFrame(columns=columns)
    completed = {
        (str(row["profile"]), str(row["fold_scope"]))
        for _, row in comparison.iterrows()
        if str(row.get("fold_scope", "")) in {"triage", "full"}
    }
    rows = []
    comparable_scopes = {"triage", "full"}
    for _, row in selection.iterrows():
        if not bool(row.get("selected", False)):
            continue
        scope = str(row.get("expected_fold_scope", ""))
        if scope not in comparable_scopes:
            continue
        profile = str(row.get("profile", ""))
        if (profile, scope) in completed:
            continue
        rows.append(
            {
                "profile": profile,
                "role": str(row.get("role", "")),
                "expected_fold_scope": scope,
                "reason": "missing_selected_profile_output",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _missing_selected_markdown(missing: pd.DataFrame) -> str:
    lines = ["# Missing Selected Profiles", ""]
    if missing.empty:
        lines.append("All selected comparison profiles have completed outputs.")
        return "\n".join(lines)
    lines.append("| profile | role | expected_fold_scope | reason |")
    lines.append("| --- | --- | --- | --- |")
    for _, row in missing.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["profile"]),
                    str(row["role"]),
                    str(row["expected_fold_scope"]),
                    str(row["reason"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _write_missing_selected_profiles(path: Path, missing: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    missing.to_csv(path / "missing_selected_profiles.csv", index=False)
    (path / "missing_selected_profiles.md").write_text(_missing_selected_markdown(missing), encoding="utf-8")
    _write_json(path / "missing_selected_profiles.json", {"rows": missing.to_dict(orient="records")})


def _parquet_timestamps(path: Path) -> pd.Series:
    try:
        frame = pd.read_parquet(path, columns=["timestamp"])
    except (TypeError, ValueError):
        frame = pd.read_parquet(path)
    if "timestamp" not in frame.columns:
        return pd.Series(dtype="datetime64[ns, UTC]")
    return pd.to_datetime(frame["timestamp"], utc=True).dropna()


def _default_holdout_path(config: dict[str, Any], holdout: dict[str, Any]) -> Path | None:
    explicit = str(holdout.get("holdout_path") or "").strip()
    if explicit:
        return Path(explicit)
    data_dir = _cfg(config, ["paths", "data_dir"], None) or _cfg(config, ["paths", "local_data_dir"], None)
    if not data_dir:
        return None
    filename = str(holdout.get("holdout_filename") or "holdout_1h.parquet")
    return Path(str(data_dir)) / "processed" / filename


def _resolve_holdout_settings(settings: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Attach durable holdout metadata from config/default Drive paths.

    Notebook 04 injects holdout metadata into its in-memory config before training,
    but notebook 05 may be run in a fresh session. This resolver lets diagnostics
    recover the reserved holdout from config and the standard parquet location.
    """

    updated = copy.deepcopy(settings)
    holdout = copy.deepcopy(updated.get("holdout") or _cfg(config, ["experiments", "holdout"], {}) or {})
    if not holdout:
        return updated

    holdout.setdefault("enabled", True)
    holdout.setdefault("policy", "profile_selection_only_before_holdout; holdout is reserved for one-shot final validation")
    holdout_path = _default_holdout_path(config, holdout)
    if holdout_path is not None:
        holdout["holdout_path"] = str(holdout_path)
        if holdout_path.exists():
            timestamps = _parquet_timestamps(holdout_path)
            if not timestamps.empty:
                holdout.setdefault("holdout_rows", int(len(timestamps)))
                holdout.setdefault("holdout_bars", int(holdout.get("holdout_rows", len(timestamps))))
                holdout.setdefault("holdout_data_start", str(timestamps.min()))
                holdout.setdefault("holdout_data_end", str(timestamps.max()))

                data_dir = _cfg(config, ["paths", "data_dir"], None) or _cfg(config, ["paths", "local_data_dir"], None)
                labeled_path = Path(str(data_dir)) / "processed" / "labeled_1h.parquet" if data_dir else None
                if labeled_path is not None and labeled_path.exists():
                    labeled_timestamps = _parquet_timestamps(labeled_path)
                    if not labeled_timestamps.empty:
                        holdout.setdefault("latest_available_data_end", str(labeled_timestamps.max()))
                        holdout_start = pd.to_datetime(holdout["holdout_data_start"], utc=True)
                        selection_timestamps = labeled_timestamps.loc[labeled_timestamps < holdout_start]
                        if not selection_timestamps.empty:
                            holdout.setdefault("selection_rows", int(len(selection_timestamps)))
                            holdout.setdefault("selection_data_start", str(selection_timestamps.min()))
                            holdout.setdefault("selection_data_end", str(selection_timestamps.max()))
    latest_data_end = _holdout_latest_available_data_end(holdout)
    if latest_data_end:
        monitor_state = _future_oos_monitor_state(config, latest_data_end)
        for key, value in monitor_state.items():
            holdout.setdefault(key, value)

    updated["holdout"] = holdout
    return updated


def _holdout_latest_available_data_end(holdout: dict[str, Any]) -> str:
    """Return the latest labeled-data timestamp, not the frozen holdout end.

    A failed clean holdout can freeze `holdout_data_end` at the anchor while
    fresher rows accumulate outside the frozen window. Future-OOS monitoring
    must count those fresher rows without allowing the holdout to roll forward.
    """

    for key in ("latest_available_data_end", "latest_data_end", "data_end", "holdout_data_end"):
        value = str(holdout.get(key, "") or "")
        if value:
            return value
    return ""


def _selection_frame_before_holdout(frame: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    holdout = settings.get("holdout", {}) or {}
    if not bool(holdout.get("enabled", False)) or "timestamp" not in frame.columns:
        return frame
    holdout_start = holdout.get("holdout_data_start")
    if not holdout_start:
        return frame
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    start = pd.to_datetime(holdout_start, utc=True)
    if not (timestamps >= start).any():
        return frame
    return frame.loc[timestamps < start].copy().reset_index(drop=True)


def _holdout_reservation_frame(settings: dict[str, Any]) -> pd.DataFrame:
    holdout = settings.get("holdout", {}) or {}
    columns = [
        "enabled",
        "holdout_bars",
        "selection_rows",
        "holdout_rows",
        "selection_data_start",
        "selection_data_end",
        "holdout_data_start",
        "holdout_data_end",
        "holdout_path",
        "policy",
        "split_mode",
        "unused_rows_after_anchor",
        "anchor_run_id",
        "anchor_data_end",
        "latest_available_data_end",
        "new_bars_since_anchor",
        "min_new_bars_remaining",
        "preferred_new_bars_remaining",
        "future_oos_ready",
        "future_oos_preferred_ready",
        "holdout_roll_forward_locked",
    ]
    if not holdout:
        return pd.DataFrame(columns=columns)
    row = {column: holdout.get(column, "") for column in columns}
    row["enabled"] = bool(holdout.get("enabled", False))
    return pd.DataFrame([row], columns=columns)


def _holdout_reservation_markdown(frame: pd.DataFrame) -> str:
    lines = ["# Holdout Reservation", ""]
    if frame.empty:
        lines.append("No holdout reservation metadata was attached to this experiment run.")
        return "\n".join(lines)
    lines.append("| field | value |")
    lines.append("| --- | --- |")
    row = frame.iloc[0].to_dict()
    for key, value in row.items():
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def _write_holdout_reservation(path: Path, settings: dict[str, Any]) -> pd.DataFrame:
    path.mkdir(parents=True, exist_ok=True)
    frame = _holdout_reservation_frame(settings)
    frame.to_csv(path / "holdout_reservation.csv", index=False)
    (path / "holdout_reservation.md").write_text(_holdout_reservation_markdown(frame), encoding="utf-8")
    _write_json(path / "holdout_reservation.json", {"rows": frame.to_dict(orient="records")})
    return frame


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


def _future_oos_ready_at_fields(monitor_state: dict[str, Any]) -> dict[str, str]:
    anchor = str(monitor_state.get("anchor_data_end", "") or "")
    fields = {"min_ready_at": "", "preferred_ready_at": ""}
    if not anchor:
        return fields
    try:
        anchor_ts = pd.to_datetime(anchor, utc=True)
    except (TypeError, ValueError):
        return fields
    if pd.isna(anchor_ts):
        return fields
    min_new_bars = int(monitor_state.get("min_new_bars", 0) or 0)
    preferred_new_bars = int(monitor_state.get("preferred_new_bars", 0) or 0)
    if min_new_bars > 0:
        fields["min_ready_at"] = str(anchor_ts + pd.Timedelta(hours=min_new_bars))
    if preferred_new_bars > 0:
        fields["preferred_ready_at"] = str(anchor_ts + pd.Timedelta(hours=preferred_new_bars))
    return fields


def prepare_training_holdout_split(
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    holdout_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create the training/holdout split without contaminating a failed clean holdout.

    After a clean holdout invalidates a frozen policy, the anchor holdout window
    must not silently roll forward just because fresher rows exist. Unless the
    config explicitly allows holdout roll-forward, rows after the anchor end stay
    unused for training and are counted only as future OOS monitoring bars.
    """

    if frame.empty or "timestamp" not in frame.columns:
        raise ValueError("Holdout split requires a non-empty frame with a timestamp column")

    data = frame.copy().reset_index(drop=True)
    timestamps = pd.to_datetime(data["timestamp"], utc=True)
    order = np.argsort(timestamps.to_numpy())
    data = data.iloc[order].reset_index(drop=True)
    timestamps = pd.to_datetime(data["timestamp"], utc=True)

    holdout_cfg = _cfg(config, ["experiments", "holdout"], {}) or {}
    holdout_bars = int(holdout_cfg.get("holdout_bars", 4320) or 4320)
    if len(data) <= holdout_bars:
        raise ValueError(f"Not enough rows for a {holdout_bars}-bar holdout: {len(data)} rows")

    latest_data_end = str(timestamps.max())
    monitor_state = _future_oos_monitor_state(config, latest_data_end)
    split_mode = "rolling_latest_holdout"
    unused_rows_after_anchor = 0
    split_data = data

    if monitor_state["holdout_roll_forward_locked"] and monitor_state["anchor_data_end"]:
        anchor_ts = pd.to_datetime(monitor_state["anchor_data_end"], utc=True)
        before_or_at_anchor = timestamps <= anchor_ts
        if before_or_at_anchor.any():
            split_data = data.loc[before_or_at_anchor].copy().reset_index(drop=True)
            unused_rows_after_anchor = int((timestamps > anchor_ts).sum())
            split_mode = "frozen_anchor_holdout"

    if len(split_data) <= holdout_bars:
        raise ValueError(
            f"Not enough rows for a {holdout_bars}-bar holdout after applying {split_mode}: {len(split_data)} rows"
        )

    holdout = split_data.tail(holdout_bars).copy().reset_index(drop=True)
    selection = split_data.iloc[:-holdout_bars].copy().reset_index(drop=True)
    if holdout_path is not None:
        Path(holdout_path).parent.mkdir(parents=True, exist_ok=True)
        holdout.to_parquet(holdout_path, index=False)

    holdout_ts = pd.to_datetime(holdout["timestamp"], utc=True)
    selection_ts = pd.to_datetime(selection["timestamp"], utc=True)
    meta = {
        "enabled": True,
        "holdout_bars": holdout_bars,
        "selection_rows": int(len(selection)),
        "holdout_rows": int(len(holdout)),
        "selection_data_start": str(selection_ts.min()),
        "selection_data_end": str(selection_ts.max()),
        "holdout_data_start": str(holdout_ts.min()),
        "holdout_data_end": str(holdout_ts.max()),
        "holdout_path": str(holdout_path or ""),
        "policy": str(
            holdout_cfg.get(
                "policy",
                "profile_selection_only_before_holdout; holdout is reserved for one-shot final validation",
            )
        ),
        "split_mode": split_mode,
        "unused_rows_after_anchor": unused_rows_after_anchor,
        **monitor_state,
    }
    return selection, holdout, meta


def _holdout_boundary_audit_frame(entries: list[dict[str, Any]], settings: dict[str, Any]) -> pd.DataFrame:
    """Verify experiment outputs stop before the reserved holdout window.

    This guards against accidentally diagnosing an old run that was trained before
    the holdout split existed. If any CV/blend/seed entry reaches into the reserved
    holdout period, holdout policy decisions must be treated as invalid.
    """

    columns = [
        "profile",
        "fold_scope",
        "data_start",
        "data_end",
        "holdout_data_start",
        "passed",
        "reason",
    ]
    holdout = settings.get("holdout", {}) or {}
    if not bool(holdout.get("enabled", False)):
        return pd.DataFrame(columns=columns)

    holdout_start_raw = holdout.get("holdout_data_start")
    if not holdout_start_raw:
        return pd.DataFrame(
            [
                {
                    "profile": "",
                    "fold_scope": "",
                    "data_start": "",
                    "data_end": "",
                    "holdout_data_start": "",
                    "passed": False,
                    "reason": "missing_holdout_data_start",
                }
            ],
            columns=columns,
        )

    holdout_start = pd.to_datetime(holdout_start_raw, utc=True)
    rows = []
    for entry in entries:
        row = entry.get("diagnostics", {}).get("row", {}) or {}
        data_start = str(row.get("data_start", ""))
        data_end = str(row.get("data_end", ""))
        reason = ""
        passed = False
        if not data_end:
            reason = "missing_entry_data_end"
        else:
            try:
                end_ts = pd.to_datetime(data_end, utc=True)
                passed = bool(end_ts < holdout_start)
                if not passed:
                    reason = "entry_data_end_reaches_reserved_holdout"
            except (TypeError, ValueError):
                reason = "invalid_entry_data_end"
        rows.append(
            {
                "profile": str(entry.get("profile", "")),
                "fold_scope": str(entry.get("fold_scope", "")),
                "data_start": data_start,
                "data_end": data_end,
                "holdout_data_start": str(holdout_start),
                "passed": passed,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _write_holdout_boundary_audit(path: Path, audit: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    audit.to_csv(path / "holdout_boundary_audit.csv", index=False)
    (path / "holdout_boundary_audit.md").write_text(
        _table_markdown("Holdout Boundary Audit", audit),
        encoding="utf-8",
    )
    _write_json(path / "holdout_boundary_audit.json", {"rows": audit.to_dict(orient="records")})


def _read_holdout_context(settings: dict[str, Any], config: dict[str, Any]) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    holdout = settings.get("holdout", {}) or {}
    if not bool(holdout.get("enabled", False)):
        return pd.DataFrame(), None
    holdout_path = Path(str(holdout.get("holdout_path", "")))
    if not holdout_path.exists():
        return pd.DataFrame(), None

    holdout_frame = pd.read_parquet(holdout_path).copy()
    if holdout_frame.empty or "timestamp" not in holdout_frame.columns:
        return pd.DataFrame(), None
    holdout_frame["timestamp"] = pd.to_datetime(holdout_frame["timestamp"], utc=True)
    holdout_start = pd.to_datetime(holdout.get("holdout_data_start", holdout_frame["timestamp"].min()), utc=True)

    seq_len = int(_cfg(config, ["model", "seq_len"], 64))
    context_rows = max(seq_len - 1, 0)
    context = pd.DataFrame()
    data_dir = _cfg(config, ["paths", "data_dir"], None)
    if data_dir:
        labeled_path = Path(str(data_dir)) / "processed" / "labeled_1h.parquet"
        if labeled_path.exists() and context_rows:
            full = pd.read_parquet(labeled_path)
            if "timestamp" in full.columns:
                full = full.copy()
                full["timestamp"] = pd.to_datetime(full["timestamp"], utc=True)
                selection_end = pd.to_datetime(holdout.get("selection_data_end", holdout_start), utc=True)
                context = full.loc[full["timestamp"] <= selection_end].tail(context_rows).copy()

    if not context.empty:
        frame = pd.concat([context, holdout_frame], ignore_index=True)
        frame = frame.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
    else:
        frame = holdout_frame.sort_values("timestamp").reset_index(drop=True)
    return frame, holdout_start


def _load_torch_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _predict_holdout_for_profile(
    *,
    scope_dir: Path,
    manifest: dict[str, Any],
    holdout_context: pd.DataFrame,
    holdout_start: pd.Timestamp,
    config: dict[str, Any],
) -> pd.DataFrame:
    profile = str(manifest["profile"])
    cfg = profile_config(config, profile)
    feature_columns = list(manifest["feature_columns"])
    required = [*feature_columns, *list(_cfg(cfg, ["hmm", "features"], []) or []), "label"]
    forward_column = f"fwd_return_{int(_cfg(cfg, ['labeling', 'max_holding_bars'], 10))}h"
    if forward_column not in holdout_context.columns:
        forward_column = "fwd_return_10h"
    required.append(forward_column)
    missing = [column for column in dict.fromkeys(required) if column not in holdout_context.columns]
    if missing:
        raise ValueError(f"Holdout frame is missing columns for {profile}: {missing}")

    torch_device = _device(None)
    batch_size = int(_cfg(cfg, ["training", "batch_size"], 256))
    rows = []
    model_paths = sorted(scope_dir.glob("model_fold_*.pt"))
    for model_path in model_paths:
        fold = int(model_path.stem.rsplit("_", 1)[-1])
        scaler_path = scope_dir / f"scaler_fold_{fold:03d}.pkl"
        hmm_path = scope_dir / f"hmm_fold_{fold:03d}.pkl"
        if not scaler_path.exists() or not hmm_path.exists():
            continue

        part = holdout_context.copy().reset_index(drop=True)
        scaler = joblib.load(scaler_path)
        part.loc[:, feature_columns] = scaler.transform(part[feature_columns])
        hmm = joblib.load(hmm_path)
        part = _add_regime_probs(part, hmm, cfg)
        dataset = _make_dataset(part, feature_columns, cfg)

        checkpoint = _load_torch_checkpoint(model_path, torch_device)
        model = _build_model(len(feature_columns), cfg).to(torch_device)
        model.load_state_dict(checkpoint["model_state_dict"])
        prediction = _predict_dataset(model, dataset, part, batch_size=batch_size, device=torch_device)
        prediction = prediction.loc[pd.to_datetime(prediction["timestamp"], utc=True) >= holdout_start].copy()
        if prediction.empty:
            continue
        prediction["split"] = "test"
        prediction["fold"] = fold
        prediction["model_fold"] = fold
        prediction["profile"] = profile
        rows.append(prediction)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _aggregate_holdout_predictions(predictions: pd.DataFrame, *, profile: str) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    frame = predictions.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    group_keys = ["timestamp"]
    first_columns = [
        column
        for column in (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "label",
            "forward_return",
            "tb_return",
            "hit_type",
            "4h_source_timestamp",
            "4h_available_timestamp",
        )
        if column in frame.columns
    ]
    aggregations: dict[str, Any] = {column: (column, "first") for column in first_columns}
    aggregations["prob_long"] = ("prob_long", "mean")
    aggregations["model_fold_count"] = ("model_fold", "nunique")
    for column in [column for column in frame.columns if column.startswith("regime_prob_")]:
        aggregations[column] = (column, "mean")
    out = frame.groupby(group_keys, as_index=False).agg(**aggregations)
    out["split"] = "test"
    out["fold"] = 0
    out["source_row_position"] = np.arange(len(out))
    out["profile"] = profile
    return out.sort_values("timestamp").reset_index(drop=True)


def _holdout_markdown(holdout_evaluation: pd.DataFrame, holdout_decision: dict[str, Any]) -> str:
    lines = ["# Holdout Evaluation", ""]
    if holdout_evaluation.empty:
        lines.append("No holdout evaluation was produced.")
        if holdout_decision:
            lines.extend(["", "## Decision", "", json.dumps(_json_ready(holdout_decision), indent=2, sort_keys=True)])
        return "\n".join(lines)
    display_cols = [
        "candidate",
        "candidate_type",
        "mean_rank_ic",
        "mean_long_f1",
        "mean_prauc",
        "calibration_separation",
        "top_10_lift_global",
        "top_10_forward_return_global",
        "cv_policy_name",
        "cv_policy_lift_vs_base",
        "cv_policy_forward_return",
        "holdout_cv_threshold_f1",
        "holdout_cv_threshold_pred_long_rate",
        "holdout_cv_threshold_source",
        "holdout_policy_name",
        "holdout_policy_selection_rate",
        "holdout_policy_lift_vs_base",
        "holdout_policy_forward_return",
        "holdout_policy_pass",
        "holdout_policy_consistency_pass",
        "holdout_policy_consistency_reject_reason",
        "mtf_leakage_passed",
        "holdout_signal_pass",
        "holdout_signal_reject_reason",
        "holdout_threshold_pass",
        "holdout_threshold_reject_reason",
        "holdout_soft_pass",
        "holdout_reject_reason",
        "frozen_selection",
    ]
    visible = holdout_evaluation[[column for column in display_cols if column in holdout_evaluation.columns]].copy()
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    lines.extend(["", "## Decision", "", json.dumps(_json_ready(holdout_decision), indent=2, sort_keys=True)])
    return "\n".join(lines)


def _write_holdout_files(
    path: Path,
    *,
    holdout_evaluation: pd.DataFrame,
    holdout_score_bands: pd.DataFrame,
    holdout_thresholds: pd.DataFrame,
    holdout_decision: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    holdout_evaluation.to_csv(path / "holdout_evaluation.csv", index=False)
    holdout_score_bands.to_csv(path / "holdout_score_band_summary.csv", index=False)
    holdout_thresholds.to_csv(path / "holdout_threshold_summary.csv", index=False)
    policy_columns = [
        column
        for column in (
            "candidate",
            "candidate_type",
            "cv_policy_name",
            "cv_policy_type",
            "cv_policy_selection_rate",
            "cv_policy_precision",
            "cv_policy_f1",
            "cv_policy_lift_vs_base",
            "cv_policy_forward_return",
            "cv_policy_positive_lift_fold_rate",
            "cv_policy_positive_forward_return_fold_rate",
            "cv_policy_pass",
            "cv_policy_reject_reason",
            "holdout_policy_name",
            "holdout_policy_type",
            "holdout_policy_source",
            "holdout_policy_selection_rate",
            "holdout_policy_precision",
            "holdout_policy_recall",
            "holdout_policy_f1",
            "holdout_policy_lift_vs_base",
            "holdout_policy_forward_return",
            "holdout_policy_selection_rate_delta_vs_cv",
            "holdout_policy_precision_delta_vs_cv",
            "holdout_policy_lift_delta_vs_cv",
            "holdout_policy_forward_return_delta_vs_cv",
            "holdout_policy_pass",
            "holdout_policy_reject_reason",
            "holdout_policy_consistency_pass",
            "holdout_policy_consistency_reject_reason",
            "holdout_signal_pass",
            "holdout_signal_reject_reason",
            "holdout_threshold_pass",
            "holdout_threshold_reject_reason",
            "holdout_soft_pass",
            "holdout_reject_reason",
            "frozen_selection",
        )
        if column in holdout_evaluation.columns
    ]
    holdout_policy_evaluation = (
        holdout_evaluation[policy_columns].copy()
        if policy_columns
        else pd.DataFrame()
    )
    holdout_policy_evaluation.to_csv(path / "holdout_policy_evaluation.csv", index=False)
    consistency_columns = [
        column
        for column in (
            "candidate",
            "candidate_type",
            "frozen_selection",
            "cv_policy_name",
            "cv_policy_type",
            "cv_policy_lift_vs_base",
            "cv_policy_forward_return",
            "cv_policy_positive_lift_fold_rate",
            "cv_policy_pass",
            "holdout_policy_name",
            "holdout_policy_type",
            "holdout_policy_lift_vs_base",
            "holdout_policy_forward_return",
            "holdout_policy_lift_delta_vs_cv",
            "holdout_policy_forward_return_delta_vs_cv",
            "holdout_policy_pass",
            "holdout_signal_pass",
            "holdout_threshold_pass",
            "holdout_policy_consistency_pass",
            "holdout_policy_consistency_reject_reason",
        )
        if column in holdout_evaluation.columns
    ]
    holdout_policy_consistency = (
        holdout_evaluation[consistency_columns].copy()
        if consistency_columns
        else pd.DataFrame()
    )
    holdout_policy_consistency.to_csv(path / "holdout_policy_consistency.csv", index=False)
    (path / "holdout_policy_consistency.md").write_text(
        _table_markdown("Holdout Policy Consistency", holdout_policy_consistency),
        encoding="utf-8",
    )
    _write_json(
        path / "holdout_policy_consistency.json",
        {"rows": holdout_policy_consistency.to_dict(orient="records")},
    )
    holdout_policy_decision = _holdout_policy_decision_frame(holdout_decision, config)
    holdout_policy_decision.to_csv(path / "holdout_policy_decision.csv", index=False)
    (path / "holdout_policy_decision.md").write_text(
        _table_markdown("Holdout Policy Decision", holdout_policy_decision),
        encoding="utf-8",
    )
    _write_json(
        path / "holdout_policy_decision.json",
        {"rows": holdout_policy_decision.to_dict(orient="records")},
    )
    (path / "holdout_evaluation.md").write_text(
        _holdout_markdown(holdout_evaluation, holdout_decision),
        encoding="utf-8",
    )
    _write_json(
        path / "holdout_evaluation.json",
        {
            "decision": holdout_decision,
            "rows": holdout_evaluation.to_dict(orient="records"),
            "score_bands": holdout_score_bands.to_dict(orient="records"),
            "thresholds": holdout_thresholds.to_dict(orient="records"),
            "policy_evaluation": holdout_policy_evaluation.to_dict(orient="records"),
            "policy_consistency": holdout_policy_consistency.to_dict(orient="records"),
            "policy_decision": holdout_policy_decision.to_dict(orient="records"),
        },
    )


def _holdout_policy_decision_frame(
    holdout_decision: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    columns = [
        "available",
        "frozen_selection",
        "frozen_selection_source",
        "score_policy_recommendation",
        "policy_action",
        "holdout_boundary_passed",
        "configured_frozen_candidate",
        "configured_policy_type",
        "configured_policy_name",
        "configured_status",
        "configured_threshold_deployment_allowed",
        "configured_future_oos_candidates",
        "configured_frozen_candidate_available",
        "configured_policy_match",
        "threshold_deployment_blocked_by_policy",
        "frozen_candidate",
        "frozen_cv_policy_name",
        "frozen_holdout_policy_name",
        "frozen_policy_consistency_pass",
        "frozen_signal_pass",
        "frozen_threshold_pass",
        "frozen_soft_pass",
        "frozen_holdout_policy_lift_vs_base",
        "frozen_holdout_policy_forward_return",
        "observed_best_policy_candidate",
        "observed_best_policy_is_frozen",
        "observed_best_policy_lift_vs_base",
        "observed_best_policy_forward_return",
        "do_not_promote_observed_best_from_same_holdout",
        "warning",
    ]
    if not bool(holdout_decision.get("available", False)):
        return pd.DataFrame(columns=columns)

    frozen = holdout_decision.get("frozen_policy_validation") or {}
    observed = holdout_decision.get("observed_best_policy_candidate") or {}
    frozen_selection = str(holdout_decision.get("frozen_selection", ""))
    frozen_selection_source = str(holdout_decision.get("frozen_selection_source", ""))
    observed_name = str(observed.get("candidate", ""))
    holdout_boundary_passed = bool(holdout_decision.get("holdout_boundary_passed", True))
    policy_review = _cfg(config or {}, ["experiments", "policy_review"], {}) or {}
    configured_candidate = str(policy_review.get("frozen_candidate", ""))
    configured_policy_type = str(policy_review.get("policy_type", ""))
    configured_policy_name = str(policy_review.get("policy_name", ""))
    configured_status = str(policy_review.get("status", ""))
    configured_threshold_allowed = bool(policy_review.get("threshold_deployment_allowed", False))
    future_candidates = ",".join(str(item) for item in policy_review.get("future_oos_candidates", []) or [])
    configured_candidate_available = bool(holdout_decision.get("configured_frozen_candidate_available", False))
    configured_policy_match = bool(
        configured_candidate
        and configured_candidate_available
        and configured_policy_name
        and frozen_selection == configured_candidate
        and str(frozen.get("cv_policy_name", "")) == configured_policy_name
        and str(frozen.get("holdout_policy_name", "")) == configured_policy_name
        and (not configured_policy_type or str(frozen.get("cv_policy_type", "")) == configured_policy_type)
        and (not configured_policy_type or str(frozen.get("holdout_policy_type", "")) == configured_policy_type)
    )
    action = _holdout_policy_action(
        frozen=frozen,
        observed_policy=observed,
        frozen_selection=frozen_selection,
        config=config,
        holdout_boundary_passed=holdout_boundary_passed,
    )
    row = {
        "available": True,
        "frozen_selection": frozen_selection,
        "frozen_selection_source": frozen_selection_source,
        "score_policy_recommendation": str(holdout_decision.get("score_policy_recommendation", "")),
        "policy_action": action,
        "holdout_boundary_passed": holdout_boundary_passed,
        "configured_frozen_candidate": configured_candidate,
        "configured_policy_type": configured_policy_type,
        "configured_policy_name": configured_policy_name,
        "configured_status": configured_status,
        "configured_threshold_deployment_allowed": configured_threshold_allowed,
        "configured_future_oos_candidates": future_candidates,
        "configured_frozen_candidate_available": configured_candidate_available,
        "configured_policy_match": configured_policy_match,
        "threshold_deployment_blocked_by_policy": not configured_threshold_allowed,
        "frozen_candidate": str(frozen.get("candidate", "")),
        "frozen_cv_policy_name": str(frozen.get("cv_policy_name", "")),
        "frozen_holdout_policy_name": str(frozen.get("holdout_policy_name", "")),
        "frozen_policy_consistency_pass": bool(frozen.get("holdout_policy_consistency_pass", False)),
        "frozen_signal_pass": bool(frozen.get("holdout_signal_pass", False)),
        "frozen_threshold_pass": bool(frozen.get("holdout_threshold_pass", False)),
        "frozen_soft_pass": bool(frozen.get("holdout_soft_pass", False)),
        "frozen_holdout_policy_lift_vs_base": _float(frozen, "holdout_policy_lift_vs_base"),
        "frozen_holdout_policy_forward_return": _float(frozen, "holdout_policy_forward_return"),
        "observed_best_policy_candidate": observed_name,
        "observed_best_policy_is_frozen": bool(observed_name and observed_name == frozen_selection),
        "observed_best_policy_lift_vs_base": _float(observed, "holdout_policy_lift_vs_base"),
        "observed_best_policy_forward_return": _float(observed, "holdout_policy_forward_return"),
        "do_not_promote_observed_best_from_same_holdout": bool(observed_name and observed_name != frozen_selection),
        "warning": str(holdout_decision.get("observed_best_policy_warning", "")),
    }
    return pd.DataFrame([row], columns=columns)


def _threshold_summary_value(threshold_summary: pd.DataFrame | None, metric: str) -> float:
    if threshold_summary is None or threshold_summary.empty:
        return np.nan
    if "metric" not in threshold_summary.columns or "mean" not in threshold_summary.columns:
        return np.nan
    matched = threshold_summary.loc[threshold_summary["metric"].astype(str) == metric, "mean"]
    if matched.empty:
        return np.nan
    try:
        return float(matched.iloc[0])
    except (TypeError, ValueError):
        return np.nan


def _cv_selected_threshold(entry: dict[str, Any] | None) -> tuple[float, str]:
    if not entry:
        return 0.5, "fallback_0.50_missing_cv_entry"
    diagnostics = entry.get("diagnostics", {}) or {}
    row = diagnostics.get("row", {}) or {}
    threshold = _float(row, "constrained_threshold_mean", np.nan)
    if np.isfinite(threshold):
        return threshold, "cv_constrained_threshold"
    threshold = _threshold_summary_value(diagnostics.get("threshold_summary"), "constrained_threshold")
    if np.isfinite(threshold):
        return threshold, "cv_constrained_threshold"
    threshold = _float(row, "selected_threshold_mean", np.nan)
    if np.isfinite(threshold):
        return threshold, "cv_selected_threshold"
    threshold = _threshold_summary_value(diagnostics.get("threshold_summary"), "selected_threshold")
    if np.isfinite(threshold):
        return threshold, "cv_selected_threshold"
    return 0.5, "fallback_0.50_missing_cv_threshold"


def _binary_metrics_at_threshold(labels: pd.Series, scores: pd.Series, threshold: float) -> dict[str, float]:
    y_true = labels.astype(int).to_numpy()
    y_score = pd.to_numeric(scores, errors="coerce").fillna(-np.inf).to_numpy(dtype=float)
    y_pred = (y_score >= float(threshold)).astype(int)
    true_positive = float(((y_true == 1) & (y_pred == 1)).sum())
    false_positive = float(((y_true == 0) & (y_pred == 1)).sum())
    false_negative = float(((y_true == 1) & (y_pred == 0)).sum())
    precision = true_positive / max(true_positive + false_positive, 1.0)
    recall = true_positive / max(true_positive + false_negative, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "pred_long_rate": float(y_pred.mean()) if len(y_pred) else np.nan,
    }


def _attach_holdout_cv_threshold_metrics(
    row: dict[str, Any],
    predictions: pd.DataFrame,
    cv_entry: dict[str, Any] | None,
) -> dict[str, Any]:
    threshold, source = _cv_selected_threshold(cv_entry)
    metrics = _binary_metrics_at_threshold(predictions["label"], predictions["prob_long"], threshold)
    row["holdout_cv_threshold"] = float(threshold)
    row["holdout_cv_threshold_source"] = source
    row["holdout_cv_threshold_f1"] = metrics["f1"]
    row["holdout_cv_threshold_precision"] = metrics["precision"]
    row["holdout_cv_threshold_recall"] = metrics["recall"]
    row["holdout_cv_threshold_pred_long_rate"] = metrics["pred_long_rate"]
    return row


def _cv_score_policy(entry: dict[str, Any] | None) -> dict[str, Any]:
    if not entry:
        return {}
    diagnostics = entry.get("diagnostics", {}) or {}
    selection = diagnostics.get("score_policy_selection")
    if isinstance(selection, pd.DataFrame) and not selection.empty:
        return selection.iloc[0].to_dict()
    grid = diagnostics.get("score_policy_grid")
    if isinstance(grid, pd.DataFrame) and not grid.empty:
        chosen = select_score_policy(grid, entry.get("config", {}) or {})
        if not chosen.empty:
            return chosen.iloc[0].to_dict()
    return {}


def _policy_metrics_from_mask(predictions: pd.DataFrame, mask: pd.Series) -> dict[str, float]:
    selected = predictions.loc[mask].copy()
    base_long_rate = float(predictions["label"].mean()) if len(predictions) else np.nan
    if selected.empty:
        return {
            "selection_rate": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "lift_vs_base": np.nan,
            "forward_return": np.nan,
        }
    true_positive = float(selected["label"].astype(int).sum())
    total_positive = float(predictions["label"].astype(int).sum())
    precision = float(selected["label"].mean())
    recall = true_positive / max(total_positive, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "selection_rate": float(len(selected) / len(predictions)) if len(predictions) else np.nan,
        "precision": precision,
        "recall": float(recall),
        "f1": float(f1),
        "lift_vs_base": float(precision / base_long_rate) if base_long_rate and base_long_rate > 0 else np.nan,
        "forward_return": float(selected["forward_return"].mean()) if "forward_return" in selected.columns else np.nan,
    }


def _evaluate_score_policy_on_holdout(
    predictions: pd.DataFrame,
    policy: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    if not policy:
        return {"source": "missing_cv_policy", "reject_reason": "missing_cv_policy"}
    policy_type = str(policy.get("policy_type", ""))
    policy_name = str(policy.get("policy_name", ""))
    if policy_type == "score_band":
        score_bins = int(_cfg(config, ["validation", "score_lift_bins"], _cfg(config, ["validation", "calibration_bins"], 10)))
        score_bands = _cfg(config, ["validation", "score_bands"], None)
        band_rows = score_band_diagnostics(predictions, bins=score_bins, bands=score_bands)
        matched = band_rows.loc[band_rows["band"].astype(str) == policy_name]
        if matched.empty:
            return {
                "name": policy_name,
                "type": policy_type,
                "source": "cv_score_policy_selection",
                "reject_reason": "missing_holdout_band",
            }
        item = matched.iloc[0].to_dict()
        metrics = {
            "selection_rate": _float(item, "selection_rate"),
            "precision": _float(item, "actual_long_rate"),
            "recall": _float(item, "recall"),
            "f1": _float(item, "f1"),
            "lift_vs_base": _float(item, "lift_vs_base"),
            "forward_return": _float(item, "mean_forward_return"),
        }
    elif policy_type == "threshold_cap":
        threshold = _float(policy, "threshold_mean", np.nan)
        if not np.isfinite(threshold):
            return {
                "name": policy_name,
                "type": policy_type,
                "source": "cv_score_policy_selection",
                "reject_reason": "missing_cv_threshold_mean",
            }
        mask = pd.to_numeric(predictions["prob_long"], errors="coerce") >= threshold
        metrics = _policy_metrics_from_mask(predictions, mask)
    else:
        return {
            "name": policy_name,
            "type": policy_type,
            "source": "cv_score_policy_selection",
            "reject_reason": "unknown_policy_type",
        }

    threshold_cfg = _cfg(config, ["validation", "threshold_checks"], {}) or {}
    policy_cfg = _cfg(config, ["validation", "policy_selection"], {}) or {}
    max_selection_rate = float(policy_cfg.get("max_selection_rate", threshold_cfg.get("max_pred_long_rate", 0.70)))
    min_precision = float(policy_cfg.get("min_precision", threshold_cfg.get("min_precision", 0.30)))
    min_lift = float(policy_cfg.get("min_lift_vs_base", 1.0))
    min_forward_return = float(policy_cfg.get("min_forward_return", 0.0))
    reasons = []
    if metrics["selection_rate"] > max_selection_rate:
        reasons.append("selection_rate")
    if metrics["precision"] < min_precision:
        reasons.append("precision")
    if not np.isfinite(metrics["lift_vs_base"]) or metrics["lift_vs_base"] <= min_lift:
        reasons.append("lift_vs_base")
    if not np.isfinite(metrics["forward_return"]) or metrics["forward_return"] <= min_forward_return:
        reasons.append("forward_return")
    return {
        "name": policy_name,
        "type": policy_type,
        "source": "cv_score_policy_selection",
        **metrics,
        "pass": len(reasons) == 0,
        "reject_reason": ";".join(reasons),
    }


def _attach_holdout_policy_metrics(
    row: dict[str, Any],
    predictions: pd.DataFrame,
    cv_entry: dict[str, Any] | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    policy = _cv_score_policy(cv_entry)
    row["cv_policy_name"] = str(policy.get("policy_name", ""))
    row["cv_policy_type"] = str(policy.get("policy_type", ""))
    row["cv_policy_selection_rate"] = _float(policy, "selection_rate")
    row["cv_policy_precision"] = _float(policy, "precision")
    row["cv_policy_recall"] = _float(policy, "recall")
    row["cv_policy_f1"] = _float(policy, "f1")
    row["cv_policy_lift_vs_base"] = _float(policy, "lift_vs_base")
    row["cv_policy_forward_return"] = _float(policy, "forward_return")
    row["cv_policy_positive_lift_fold_rate"] = _float(policy, "positive_lift_fold_rate")
    row["cv_policy_positive_forward_return_fold_rate"] = _float(policy, "positive_forward_return_fold_rate")
    row["cv_policy_pass"] = bool(policy.get("policy_pass", False))
    row["cv_policy_reject_reason"] = str(policy.get("policy_reject_reason", ""))
    metrics = _evaluate_score_policy_on_holdout(predictions, policy, config)
    row["holdout_policy_name"] = metrics.get("name", "")
    row["holdout_policy_type"] = metrics.get("type", "")
    row["holdout_policy_source"] = metrics.get("source", "")
    row["holdout_policy_selection_rate"] = metrics.get("selection_rate", np.nan)
    row["holdout_policy_precision"] = metrics.get("precision", np.nan)
    row["holdout_policy_recall"] = metrics.get("recall", np.nan)
    row["holdout_policy_f1"] = metrics.get("f1", np.nan)
    row["holdout_policy_lift_vs_base"] = metrics.get("lift_vs_base", np.nan)
    row["holdout_policy_forward_return"] = metrics.get("forward_return", np.nan)
    row["holdout_policy_pass"] = bool(metrics.get("pass", False))
    row["holdout_policy_reject_reason"] = metrics.get("reject_reason", "")
    return row


def _attach_holdout_policy_consistency(row: dict[str, Any]) -> dict[str, Any]:
    row["holdout_policy_selection_rate_delta_vs_cv"] = (
        _float(row, "holdout_policy_selection_rate") - _float(row, "cv_policy_selection_rate")
    )
    row["holdout_policy_precision_delta_vs_cv"] = (
        _float(row, "holdout_policy_precision") - _float(row, "cv_policy_precision")
    )
    row["holdout_policy_lift_delta_vs_cv"] = (
        _float(row, "holdout_policy_lift_vs_base") - _float(row, "cv_policy_lift_vs_base")
    )
    row["holdout_policy_forward_return_delta_vs_cv"] = (
        _float(row, "holdout_policy_forward_return") - _float(row, "cv_policy_forward_return")
    )

    reasons = []
    if not str(row.get("cv_policy_name", "")).strip():
        reasons.append("missing_cv_policy")
    if not bool(row.get("cv_policy_pass", False)):
        reasons.append("cv_policy")
    if str(row.get("cv_policy_name", "")) != str(row.get("holdout_policy_name", "")):
        reasons.append("policy_name_mismatch")
    if str(row.get("cv_policy_type", "")) != str(row.get("holdout_policy_type", "")):
        reasons.append("policy_type_mismatch")
    if not bool(row.get("holdout_policy_pass", False)):
        reasons.append("holdout_policy")
    if not bool(row.get("holdout_signal_pass", False)):
        reasons.append("holdout_signal")
    row["holdout_policy_consistency_pass"] = len(reasons) == 0
    row["holdout_policy_consistency_reject_reason"] = ";".join(reasons)
    return row


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


def _attach_holdout_soft_pass(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    signal_reasons = _holdout_signal_pass_reasons(row, config)
    threshold_reasons = _holdout_threshold_pass_reasons(row, config)
    reasons = [*signal_reasons, *threshold_reasons]
    row["holdout_signal_pass"] = len(signal_reasons) == 0
    row["holdout_signal_reject_reason"] = ";".join(signal_reasons)
    row["holdout_threshold_pass"] = len(threshold_reasons) == 0
    row["holdout_threshold_reject_reason"] = ";".join(threshold_reasons)
    row["holdout_soft_pass"] = len(reasons) == 0
    row["holdout_reject_reason"] = ";".join(reasons)
    return row


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


def _frozen_policy_monitoring_plan_frame(config: dict[str, Any], settings: dict[str, Any]) -> pd.DataFrame:
    policy_review = _cfg(config, ["experiments", "policy_review"], {}) or {}
    monitor = policy_review.get("future_oos_monitor", {}) or {}
    holdout = settings.get("holdout", {}) or {}
    latest_data_end = _holdout_latest_available_data_end(holdout)
    monitor_state = _future_oos_monitor_state(config, latest_data_end)
    ready_at = _future_oos_ready_at_fields(monitor_state)
    row = {
        "enabled": bool(monitor.get("enabled", False)),
        "frozen_candidate": str(policy_review.get("frozen_candidate", "")),
        "policy_type": str(policy_review.get("policy_type", "")),
        "policy_name": str(policy_review.get("policy_name", "")),
        "status": str(policy_review.get("status", "")),
        "threshold_deployment_allowed": bool(policy_review.get("threshold_deployment_allowed", False)),
        "future_oos_candidates": ",".join(str(item) for item in policy_review.get("future_oos_candidates", []) or []),
        "anchor_run_id": monitor_state["anchor_run_id"],
        "anchor_data_end": monitor_state["anchor_data_end"],
        "latest_available_data_end": monitor_state["latest_available_data_end"],
        "new_bars_since_anchor": monitor_state["new_bars_since_anchor"],
        "min_new_bars": monitor_state["min_new_bars"],
        "preferred_new_bars": monitor_state["preferred_new_bars"],
        "min_new_bars_remaining": monitor_state["min_new_bars_remaining"],
        "preferred_new_bars_remaining": monitor_state["preferred_new_bars_remaining"],
        "min_ready_at": ready_at["min_ready_at"],
        "preferred_ready_at": ready_at["preferred_ready_at"],
        "future_oos_ready": monitor_state["future_oos_ready"],
        "future_oos_preferred_ready": monitor_state["future_oos_preferred_ready"],
        "allow_holdout_roll_forward": monitor_state["allow_holdout_roll_forward"],
        "holdout_roll_forward_locked": monitor_state["holdout_roll_forward_locked"],
        "current_holdout_data_end": str(holdout.get("holdout_data_end", "") or ""),
        "frozen_holdout_data_end": str(holdout.get("holdout_data_end", "") or ""),
        "next_action": monitor_state["next_action"],
        "policy": str(monitor.get("policy", "")),
    }
    return pd.DataFrame([row])


def _write_frozen_policy_monitoring_plan(path: Path, frame: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path / "frozen_policy_monitoring_plan.csv", index=False)
    (path / "frozen_policy_monitoring_plan.md").write_text(
        _table_markdown("Frozen Policy Monitoring Plan", frame),
        encoding="utf-8",
    )
    _write_json(path / "frozen_policy_monitoring_plan.json", {"rows": frame.to_dict(orient="records")})


def _experiment_policy_guard_frame(settings: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    guard = copy.deepcopy(settings.get("experiment_policy_guard") or _experiment_policy_guard(settings, config))
    ready_at = _future_oos_ready_at_fields(guard)
    row = {
        "enabled": bool(guard.get("enabled", False)),
        "status": str(guard.get("status", "")),
        "profile_search_locked": bool(guard.get("profile_search_locked", False)),
        "action": str(guard.get("action", "")),
        "reason": str(guard.get("reason", "")),
        "allowed_benchmark_profiles": ",".join(str(item) for item in guard.get("allowed_benchmark_profiles", []) or []),
        "blocked_candidate_profiles": ",".join(str(item) for item in guard.get("blocked_candidate_profiles", []) or []),
        "blocked_full_profiles": ",".join(str(item) for item in guard.get("blocked_full_profiles", []) or []),
        "blocked_seed_profiles": ",".join(str(item) for item in guard.get("blocked_seed_profiles", []) or []),
        "future_oos_ready": bool(guard.get("future_oos_ready", False)),
        "future_oos_preferred_ready": bool(guard.get("future_oos_preferred_ready", False)),
        "new_bars_since_anchor": int(guard.get("new_bars_since_anchor", 0) or 0),
        "min_new_bars_remaining": int(guard.get("min_new_bars_remaining", 0) or 0),
        "preferred_new_bars_remaining": int(guard.get("preferred_new_bars_remaining", 0) or 0),
        "min_ready_at": ready_at["min_ready_at"],
        "preferred_ready_at": ready_at["preferred_ready_at"],
        "holdout_roll_forward_locked": bool(guard.get("holdout_roll_forward_locked", False)),
        "next_action": str(guard.get("next_action", "")),
        "anchor_run_id": str(guard.get("anchor_run_id", "")),
        "anchor_data_end": str(guard.get("anchor_data_end", "")),
        "latest_available_data_end": str(guard.get("latest_available_data_end", "")),
    }
    return pd.DataFrame([row])


def _write_experiment_policy_guard(path: Path, frame: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path / "experiment_policy_guard.csv", index=False)
    (path / "experiment_policy_guard.md").write_text(
        _table_markdown("Experiment Policy Guard", frame),
        encoding="utf-8",
    )
    _write_json(path / "experiment_policy_guard.json", {"rows": frame.to_dict(orient="records")})


def _recommendation_with_policy_guard(recommendation: str, settings: dict[str, Any]) -> str:
    guard = settings.get("experiment_policy_guard", {}) or {}
    if bool(guard.get("profile_search_locked", False)) and recommendation not in {
        "fix_missing_selected_profiles",
        "rerun_training_with_holdout_split",
    }:
        return str(guard.get("action") or "wait_for_new_unseen_bars_keep_control_profile")
    return recommendation


def _future_oos_candidate_plan_frame(
    settings: dict[str, Any],
    config: dict[str, Any],
    payoff_policy_robustness_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    policy_review = _cfg(config, ["experiments", "policy_review"], {}) or {}
    guard = settings.get("experiment_policy_guard", {}) or _experiment_policy_guard(settings, config)
    ready_at = _future_oos_ready_at_fields(guard)
    profiles_cfg = _cfg(config, ["features", "profiles"], {}) or {}
    weighted_blends = _cfg(config, ["experiments", "profile_blends", "weighted"], []) or []
    rows: list[dict[str, Any]] = []

    def weighted_blend_profiles(candidate: str) -> list[str]:
        for blend in weighted_blends:
            if not isinstance(blend, dict):
                continue
            name = str(blend.get("name", ""))
            if candidate not in {name, f"blend_{name}"}:
                continue
            return [str(profile) for profile in blend.get("profiles", []) or [] if str(profile)]
        return []

    def add_row(
        *,
        candidate: str,
        candidate_type: str,
        stage: str,
        required_profiles: list[str] | None = None,
        note: str = "",
        policy_name: str = "",
        policy_type: str = "",
        selection_source: str = "",
        cv_mean_label_lift_vs_base: float | None = None,
        cv_mean_forward_return: float | None = None,
        cv_payoff_alignment_fold_rate: float | None = None,
        current_holdout_mean_label_lift_vs_base: float | None = None,
        current_holdout_mean_forward_return: float | None = None,
        current_holdout_mean_tb_return: float | None = None,
        current_holdout_payoff_alignment_fold_rate: float | None = None,
        current_holdout_reject_reason: str = "",
    ) -> None:
        required = [str(profile) for profile in required_profiles or [] if str(profile)]
        missing_profiles = [profile for profile in required if profile not in profiles_cfg]
        allowed = set(str(item) for item in guard.get("allowed_benchmark_profiles", []) or [])
        all_required_allowed = all(profile in allowed for profile in required) if required else candidate in allowed
        is_retired = stage == "retired_frozen_policy"
        candidate_status = {
            "control_profile": "active_control",
            "future_oos_candidate": "pre_registered_future_oos_candidate",
            "future_oos_score_band_policy": "pre_registered_future_oos_policy",
            "retired_frozen_policy": "historical_retired_policy_do_not_promote",
        }.get(stage, "diagnostic_candidate")
        candidate_id = candidate if not policy_name else f"{candidate}::{policy_name}"
        candidate_label = candidate if not policy_name else f"{candidate} [{policy_name}]"
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_label": candidate_label,
                "candidate": candidate,
                "candidate_type": candidate_type,
                "stage": stage,
                "required_profiles": ",".join(required),
                "missing_required_profiles": ",".join(missing_profiles),
                "all_required_profiles_allowed": all_required_allowed,
                "candidate_status": candidate_status,
                "profile_search_locked": bool(guard.get("profile_search_locked", False)),
                "future_oos_ready": bool(guard.get("future_oos_ready", False)),
                "min_new_bars_remaining": int(guard.get("min_new_bars_remaining", 0) or 0),
                "min_ready_at": ready_at["min_ready_at"],
                "preferred_ready_at": ready_at["preferred_ready_at"],
                "action": str(guard.get("action", "")),
                "evaluation_status": "wait_for_future_oos" if not bool(guard.get("future_oos_ready", False)) else "ready_for_future_oos_review",
                "promotion_allowed_now": (
                    bool(guard.get("future_oos_ready", False))
                    and bool(all_required_allowed)
                    and not is_retired
                ),
                "note": note,
                "policy_name": policy_name,
                "policy_type": policy_type,
                "selection_source": selection_source,
                "cv_mean_label_lift_vs_base": cv_mean_label_lift_vs_base,
                "cv_mean_forward_return": cv_mean_forward_return,
                "cv_payoff_alignment_fold_rate": cv_payoff_alignment_fold_rate,
                "current_holdout_diagnostic_only": bool(policy_name),
                "current_holdout_mean_label_lift_vs_base": current_holdout_mean_label_lift_vs_base,
                "current_holdout_mean_forward_return": current_holdout_mean_forward_return,
                "current_holdout_mean_tb_return": current_holdout_mean_tb_return,
                "current_holdout_payoff_alignment_fold_rate": current_holdout_payoff_alignment_fold_rate,
                "current_holdout_reject_reason": current_holdout_reject_reason,
            }
        )

    control = str(settings.get("control_profile", ""))
    if control:
        add_row(
            candidate=control,
            candidate_type="profile",
            stage="control_profile",
            required_profiles=[control],
            note="Current control profile remains the safe baseline.",
        )

    frozen = str(policy_review.get("frozen_candidate", "")).strip()
    if frozen:
        add_row(
            candidate=frozen,
            candidate_type=str(policy_review.get("policy_type", "score_policy")),
            stage="retired_frozen_policy",
            note=str(policy_review.get("note", "")),
        )

    future_items = [str(item) for item in policy_review.get("future_oos_candidates", []) or []]
    for item in future_items:
        matched = False
        for blend in weighted_blends:
            if not isinstance(blend, dict):
                continue
            name = str(blend.get("name", ""))
            if item not in {name, f"blend_{name}"}:
                continue
            add_row(
                candidate=item,
                candidate_type="weighted_blend",
                stage="future_oos_candidate",
                required_profiles=[str(profile) for profile in blend.get("profiles", []) or []],
                note=str(blend.get("description", "")),
            )
            matched = True
            break
        if matched:
            continue
        add_row(
            candidate=item,
            candidate_type="profile" if item in profiles_cfg else "unknown",
            stage="future_oos_candidate",
            required_profiles=[item] if item in profiles_cfg else [],
            note="" if item in profiles_cfg else "Candidate is not a known feature profile or configured weighted blend.",
        )

    if payoff_policy_robustness_summary is not None and not payoff_policy_robustness_summary.empty:
        policy_rows = payoff_policy_robustness_summary.copy()
        holdout_lookup: dict[tuple[str, str], dict[str, Any]] = {}
        if {"candidate", "band", "evaluation_scope"}.issubset(policy_rows.columns):
            holdout_rows = policy_rows[policy_rows["evaluation_scope"].astype(str) == "holdout"]
            holdout_lookup = {
                (str(row.get("candidate", "")).strip(), str(row.get("band", "")).strip()): row.to_dict()
                for _, row in holdout_rows.iterrows()
            }
        if "future_oos_policy_candidate" in policy_rows.columns:
            candidate_mask = policy_rows["future_oos_policy_candidate"].map(
                lambda value: bool(value) if isinstance(value, (bool, np.bool_)) else str(value).strip().lower() in {"1", "true", "yes"}
            )
            policy_rows = policy_rows[
                (policy_rows.get("evaluation_scope", "").astype(str) == "cv_test")
                & candidate_mask
            ]
        else:
            policy_rows = policy_rows.iloc[0:0]
        existing_policy_keys = {
            (str(row.get("candidate", "")), str(row.get("stage", "")), str(row.get("policy_name", "")))
            for row in rows
        }
        for _, policy_row in policy_rows.iterrows():
            candidate = str(policy_row.get("candidate", "")).strip()
            band = str(policy_row.get("band", "")).strip()
            if not candidate or not band:
                continue
            key = (candidate, "future_oos_score_band_policy", band)
            if key in existing_policy_keys:
                continue
            if candidate in profiles_cfg:
                required_profiles = [candidate]
                candidate_type = "profile_score_band"
            else:
                required_profiles = weighted_blend_profiles(candidate)
                candidate_type = "weighted_blend_score_band" if required_profiles else "score_band_policy"
            current_holdout = holdout_lookup.get((candidate, band), {})
            add_row(
                candidate=candidate,
                candidate_type=candidate_type,
                stage="future_oos_score_band_policy",
                required_profiles=required_profiles,
                note=(
                    "CV payoff-policy robustness pre-registered this score band for future unseen OOS review. "
                    "Current holdout remains diagnostic-only and must not be used for promotion."
                ),
                policy_name=band,
                policy_type="score_band",
                selection_source="cv_payoff_policy_robustness",
                cv_mean_label_lift_vs_base=_optional_float(policy_row.get("mean_label_lift_vs_base")),
                cv_mean_forward_return=_optional_float(policy_row.get("mean_forward_return")),
                cv_payoff_alignment_fold_rate=_optional_float(policy_row.get("payoff_alignment_fold_rate")),
                current_holdout_mean_label_lift_vs_base=_optional_float(current_holdout.get("mean_label_lift_vs_base")),
                current_holdout_mean_forward_return=_optional_float(current_holdout.get("mean_forward_return")),
                current_holdout_mean_tb_return=_optional_float(current_holdout.get("mean_tb_return")),
                current_holdout_payoff_alignment_fold_rate=_optional_float(current_holdout.get("payoff_alignment_fold_rate")),
                current_holdout_reject_reason=str(current_holdout.get("reject_reason", "")),
            )
            existing_policy_keys.add(key)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    stage_order = {
        "control_profile": 0,
        "future_oos_candidate": 20,
        "future_oos_score_band_policy": 30,
        "retired_frozen_policy": 80,
    }
    policy_order = {
        "top_10": 10,
        "top_20": 20,
        "top_30": 30,
        "upper_half": 50,
        "mid_upper_40_90": 60,
    }
    out = frame.copy()
    out["_stage_order"] = out["stage"].map(stage_order).fillna(99).astype(int)
    out["_policy_order"] = out["policy_name"].map(policy_order).fillna(999).astype(int)
    out = out.sort_values(
        ["_stage_order", "candidate_type", "candidate_label", "_policy_order", "candidate_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    out.insert(0, "plan_rank", np.arange(1, len(out) + 1, dtype=int))
    return out.drop(columns=["_stage_order", "_policy_order"])


def _write_future_oos_candidate_plan(path: Path, frame: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path / "future_oos_candidate_plan.csv", index=False)
    (path / "future_oos_candidate_plan.md").write_text(
        _table_markdown("Future OOS Candidate Plan", frame),
        encoding="utf-8",
    )
    _write_json(path / "future_oos_candidate_plan.json", {"rows": frame.to_dict(orient="records")})


def _performance_gap_reasons(row: dict[str, Any], config: dict[str, Any]) -> str:
    target_rank_ic = float(_cfg(config, ["validation", "target_rank_ic"], 0.03))
    max_rank_ic_std = float(_cfg(config, ["validation", "max_rank_ic_std"], 0.03))
    min_positive_ic_fraction = float(_cfg(config, ["validation", "min_positive_ic_fraction"], 0.75))
    min_long_f1 = float(_cfg(config, ["validation", "min_long_f1"], 0.45))
    threshold_cfg = _cfg(config, ["validation", "threshold_checks"], {}) or {}
    max_pred_long_rate = float(threshold_cfg.get("max_pred_long_rate", 0.70))
    reasons = []
    if _float(row, "mean_rank_ic") < target_rank_ic:
        reasons.append("cv_rank_ic_below_target")
    if _float(row, "std_rank_ic") > max_rank_ic_std:
        reasons.append("cv_rank_ic_std_above_phase1_target")
    if _float(row, "positive_ic_fraction") < min_positive_ic_fraction:
        reasons.append("cv_positive_ic_fraction_below_target")
    selected_f1 = _float(row, "test_f1_at_selected_threshold", np.nan)
    constrained_f1 = _float(row, "test_f1_at_constrained_threshold", np.nan)
    guarded_f1 = _float(row, "test_f1_at_guarded_threshold", np.nan)
    official_f1 = _float(row, "test_f1_at_official_threshold", guarded_f1)
    fixed_f1 = _float(row, "mean_long_f1", np.nan)
    if selected_f1 < min_long_f1:
        reasons.append("cv_selected_threshold_f1_below_target")
    if constrained_f1 < min_long_f1:
        reasons.append("cv_constrained_threshold_f1_below_target")
    if np.isfinite(guarded_f1) and guarded_f1 < min_long_f1:
        reasons.append("cv_guarded_threshold_f1_below_target")
    if np.isfinite(official_f1) and official_f1 < min_long_f1:
        reasons.append("cv_official_threshold_f1_below_target")
    if not np.isfinite(selected_f1) and not np.isfinite(constrained_f1) and not np.isfinite(guarded_f1) and not np.isfinite(official_f1) and fixed_f1 < min_long_f1:
        reasons.append("cv_fixed_0_50_f1_below_target")
    if _float(row, "test_pred_long_rate_at_selected_threshold", np.nan) > max_pred_long_rate:
        reasons.append("cv_selected_threshold_pred_long_rate_above_guardrail")
    if _float(row, "test_pred_long_rate_at_constrained_threshold", np.nan) > max_pred_long_rate:
        reasons.append("cv_constrained_threshold_pred_long_rate_above_guardrail")
    if _float(row, "test_pred_long_rate_at_official_threshold", np.nan) > max_pred_long_rate:
        reasons.append("cv_official_threshold_pred_long_rate_above_guardrail")
    if _float(row, "top_10_lift_global") < 1.0:
        reasons.append("cv_top_10_lift_below_base")
    if not bool(row.get("mtf_leakage_passed", False)):
        reasons.append("mtf_leakage")
    if not bool(row.get("stationarity_policy_passed", False)):
        reasons.append("stationarity_policy")
    return ";".join(reasons)


def _holdout_gap_reasons(holdout_row: dict[str, Any], config: dict[str, Any]) -> str:
    if not holdout_row:
        return "missing_holdout_evaluation"
    target_rank_ic = float(_cfg(config, ["validation", "target_rank_ic"], 0.03))
    min_long_f1 = float(_cfg(config, ["validation", "min_long_f1"], 0.45))
    reasons = []
    if _float(holdout_row, "mean_rank_ic") < target_rank_ic:
        reasons.append("holdout_rank_ic_below_target")
    if _float(holdout_row, "top_10_lift_global") <= 1.0:
        reasons.append("holdout_top_10_lift_not_above_base")
    if _float(holdout_row, "top_10_forward_return_global") <= 0.0:
        reasons.append("holdout_top_10_forward_return_not_positive")
    if _float(holdout_row, "holdout_cv_threshold_f1") < min_long_f1:
        reasons.append("holdout_cv_threshold_f1_below_target")
    if not bool(holdout_row.get("holdout_policy_pass", False)):
        reasons.append("holdout_policy")
    if not bool(holdout_row.get("holdout_signal_pass", False)):
        signal_reason = str(holdout_row.get("holdout_signal_reject_reason", "holdout_signal")).strip(";")
        reasons.append(signal_reason or "holdout_signal")
    if not bool(holdout_row.get("holdout_threshold_pass", False)):
        threshold_reason = str(holdout_row.get("holdout_threshold_reject_reason", "holdout_threshold")).strip(";")
        reasons.append(threshold_reason or "holdout_threshold")
    if not bool(holdout_row.get("mtf_leakage_passed", False)):
        reasons.append("holdout_mtf_leakage")
    return ";".join(dict.fromkeys(reason for reason in reasons if reason))


def _performance_gap_action(
    *,
    cv_reasons: str,
    holdout_reasons: str,
    guard: dict[str, Any],
    candidate_type: str,
) -> str:
    if bool(guard.get("profile_search_locked", False)):
        return "wait_for_future_oos_do_not_tune_current_holdout"
    if holdout_reasons and holdout_reasons != "missing_holdout_evaluation":
        return "do_not_promote_investigate_holdout_failure"
    if cv_reasons:
        return "improve_cv_stability_before_promotion"
    if candidate_type == "blend":
        return "candidate_blend_ready_for_predefined_future_oos_review"
    return "candidate_profile_ready_for_predefined_future_oos_review"


def _performance_gap_analysis_frame(
    entries: list[dict[str, Any]],
    holdout_evaluation: pd.DataFrame,
    config: dict[str, Any],
    settings: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "feature_count",
        "fold_count",
        "cv_mean_rank_ic",
        "cv_std_rank_ic",
        "cv_positive_ic_fraction",
        "cv_worst_5_rank_ic_mean",
        "cv_top_10_lift_global",
        "cv_top_10_forward_return_global",
        "cv_selected_threshold_f1",
        "cv_constrained_threshold_f1",
        "cv_guarded_threshold_f1",
        "cv_guarded_threshold_source",
        "cv_guarded_threshold_pred_long_rate",
        "cv_official_threshold_f1",
        "cv_official_threshold_source",
        "cv_official_threshold_pred_long_rate",
        "cv_calibrated_guarded_threshold_f1",
        "cv_calibrated_guarded_threshold_pred_long_rate",
        "holdout_available",
        "holdout_mean_rank_ic",
        "holdout_top_10_lift_global",
        "holdout_top_10_forward_return_global",
        "holdout_cv_threshold_f1",
        "holdout_policy_lift_vs_base",
        "holdout_policy_forward_return",
        "holdout_soft_pass",
        "cv_to_holdout_rank_ic_delta",
        "cv_to_holdout_top_10_lift_delta",
        "cv_to_holdout_top_10_forward_return_delta",
        "cv_phase1_blockers",
        "holdout_blockers",
        "profile_search_locked",
        "future_oos_ready",
        "next_action",
        "research_track",
        "note",
    ]
    rows: list[dict[str, Any]] = []
    holdout_by_candidate = {}
    if not holdout_evaluation.empty and "candidate" in holdout_evaluation.columns:
        holdout_by_candidate = {
            str(row["candidate"]): row.to_dict()
            for _, row in holdout_evaluation.iterrows()
        }
    guard = settings.get("experiment_policy_guard", {}) or _experiment_policy_guard(settings, config)
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if fold_scope != "full" and not fold_scope.startswith("blend_"):
            continue
        row = dict(entry["diagnostics"]["row"])
        candidate = str(row.get("profile", entry.get("profile", "")))
        key = (candidate, fold_scope)
        if key in seen:
            continue
        seen.add(key)
        candidate_type = "blend" if fold_scope.startswith("blend_") else "profile"
        holdout_row = holdout_by_candidate.get(candidate, {})
        cv_reasons = _performance_gap_reasons(row, config)
        holdout_reasons = _holdout_gap_reasons(holdout_row, config) if holdout_by_candidate else "missing_holdout_evaluation"
        tracks = []
        if "std" in cv_reasons or "positive_ic_fraction" in cv_reasons:
            tracks.append("fold_stability")
        if "f1" in cv_reasons or "threshold" in holdout_reasons:
            tracks.append("threshold_calibration")
        if "top_10" in holdout_reasons or "policy" in holdout_reasons:
            tracks.append("score_band_policy")
        if "forward_return" in holdout_reasons:
            tracks.append("feature_regime_mismatch")
        if not tracks:
            tracks.append("future_oos_validation")
        rows.append(
            {
                "candidate": candidate,
                "candidate_type": candidate_type,
                "fold_scope": fold_scope,
                "feature_count": int(row.get("feature_count", 0) or 0),
                "fold_count": int(row.get("fold_count", 0) or 0),
                "cv_mean_rank_ic": _float(row, "mean_rank_ic"),
                "cv_std_rank_ic": _float(row, "std_rank_ic"),
                "cv_positive_ic_fraction": _float(row, "positive_ic_fraction"),
                "cv_worst_5_rank_ic_mean": _float(row, "worst_5_rank_ic_mean"),
                "cv_top_10_lift_global": _float(row, "top_10_lift_global"),
                "cv_top_10_forward_return_global": _float(row, "top_10_forward_return_global"),
                "cv_selected_threshold_f1": _float(row, "test_f1_at_selected_threshold"),
                "cv_constrained_threshold_f1": _float(row, "test_f1_at_constrained_threshold"),
                "cv_guarded_threshold_f1": _float(row, "test_f1_at_guarded_threshold"),
                "cv_guarded_threshold_source": str(row.get("guarded_threshold_source", "")),
                "cv_guarded_threshold_pred_long_rate": _float(row, "test_pred_long_rate_at_guarded_threshold"),
                "cv_official_threshold_f1": _float(row, "test_f1_at_official_threshold"),
                "cv_official_threshold_source": str(row.get("official_threshold_source", "")),
                "cv_official_threshold_pred_long_rate": _float(row, "test_pred_long_rate_at_official_threshold"),
                "cv_calibrated_guarded_threshold_f1": _float(row, "test_f1_at_calibrated_guarded_threshold"),
                "cv_calibrated_guarded_threshold_pred_long_rate": _float(row, "test_pred_long_rate_at_calibrated_guarded_threshold"),
                "holdout_available": bool(holdout_row),
                "holdout_mean_rank_ic": _float(holdout_row, "mean_rank_ic") if holdout_row else np.nan,
                "holdout_top_10_lift_global": _float(holdout_row, "top_10_lift_global") if holdout_row else np.nan,
                "holdout_top_10_forward_return_global": _float(holdout_row, "top_10_forward_return_global") if holdout_row else np.nan,
                "holdout_cv_threshold_f1": _float(holdout_row, "holdout_cv_threshold_f1") if holdout_row else np.nan,
                "holdout_policy_lift_vs_base": _float(holdout_row, "holdout_policy_lift_vs_base") if holdout_row else np.nan,
                "holdout_policy_forward_return": _float(holdout_row, "holdout_policy_forward_return") if holdout_row else np.nan,
                "holdout_soft_pass": bool(holdout_row.get("holdout_soft_pass", False)) if holdout_row else False,
                "cv_to_holdout_rank_ic_delta": (
                    _float(holdout_row, "mean_rank_ic") - _float(row, "mean_rank_ic") if holdout_row else np.nan
                ),
                "cv_to_holdout_top_10_lift_delta": (
                    _float(holdout_row, "top_10_lift_global") - _float(row, "top_10_lift_global") if holdout_row else np.nan
                ),
                "cv_to_holdout_top_10_forward_return_delta": (
                    _float(holdout_row, "top_10_forward_return_global") - _float(row, "top_10_forward_return_global")
                    if holdout_row
                    else np.nan
                ),
                "cv_phase1_blockers": cv_reasons,
                "holdout_blockers": holdout_reasons,
                "profile_search_locked": bool(guard.get("profile_search_locked", False)),
                "future_oos_ready": bool(guard.get("future_oos_ready", False)),
                "next_action": _performance_gap_action(
                    cv_reasons=cv_reasons,
                    holdout_reasons=holdout_reasons,
                    guard=guard,
                    candidate_type=candidate_type,
                ),
                "research_track": ";".join(dict.fromkeys(tracks)),
                "note": "Diagnostics only; do not tune profiles or weights against the current frozen holdout.",
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows)
        .reindex(columns=columns)
        .sort_values(
            ["candidate_type", "cv_mean_rank_ic", "cv_top_10_lift_global"],
            ascending=[True, False, False],
        )
        .reset_index(drop=True)
    )


def _performance_gap_markdown(frame: pd.DataFrame) -> str:
    lines = ["# Performance Gap Analysis", ""]
    if frame.empty:
        lines.append("No full-profile or blend candidates were available for performance gap analysis.")
        return "\n".join(lines)
    display_cols = [
        "candidate",
        "candidate_type",
        "cv_mean_rank_ic",
        "cv_std_rank_ic",
        "cv_positive_ic_fraction",
        "cv_top_10_lift_global",
        "cv_guarded_threshold_f1",
        "cv_guarded_threshold_source",
        "cv_official_threshold_f1",
        "cv_official_threshold_source",
        "holdout_mean_rank_ic",
        "holdout_top_10_lift_global",
        "holdout_top_10_forward_return_global",
        "cv_phase1_blockers",
        "holdout_blockers",
        "next_action",
        "research_track",
    ]
    visible = frame[[column for column in display_cols if column in frame.columns]].copy()
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    return "\n".join(lines)


def _write_performance_gap_analysis(path: Path, frame: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path / "performance_gap_analysis.csv", index=False)
    (path / "performance_gap_analysis.md").write_text(_performance_gap_markdown(frame), encoding="utf-8")
    _write_json(path / "performance_gap_analysis.json", {"rows": frame.to_dict(orient="records")})


def _fmt_metric(value: Any, digits: int = 4) -> str:
    number = _optional_float(value)
    if number is None:
        return "NA"
    return f"{number:.{digits}f}"


def _first_frame_row(frame: pd.DataFrame, mask: pd.Series | None = None) -> dict[str, Any]:
    if frame.empty:
        return {}
    selected = frame.loc[mask] if mask is not None else frame
    if selected.empty:
        return {}
    return selected.iloc[0].to_dict()


def _control_comparison_row(comparison: pd.DataFrame, control_profile: str) -> dict[str, Any]:
    if comparison.empty:
        return {}
    full_mask = (comparison["profile"].astype(str) == str(control_profile)) & (
        comparison["fold_scope"].astype(str) == "full"
    )
    row = _first_frame_row(comparison, full_mask)
    if row:
        return row
    control_mask = comparison["profile"].astype(str) == str(control_profile)
    return _first_frame_row(comparison, control_mask)


def _control_gap_row(performance_gap_analysis: pd.DataFrame, control_profile: str) -> dict[str, Any]:
    if performance_gap_analysis.empty:
        return {}
    mask = (performance_gap_analysis["candidate"].astype(str) == str(control_profile)) & (
        performance_gap_analysis["fold_scope"].astype(str) == "full"
    )
    row = _first_frame_row(performance_gap_analysis, mask)
    if row:
        return row
    return _first_frame_row(performance_gap_analysis, performance_gap_analysis["candidate"].astype(str) == str(control_profile))


def _phase1_blocker_action_plan_frame(
    *,
    comparison: pd.DataFrame,
    profile_blend: pd.DataFrame,
    performance_gap_analysis: pd.DataFrame,
    fold_stability_forensics: pd.DataFrame,
    fold_stability_summary: pd.DataFrame,
    threshold_forensics: pd.DataFrame,
    payoff_policy_robustness_summary: pd.DataFrame,
    future_oos_candidate_plan: pd.DataFrame,
    phase2_readiness: dict[str, Any] | None,
    config: dict[str, Any],
    settings: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "priority",
        "blocker",
        "severity",
        "control_profile",
        "metric_value",
        "target",
        "passed",
        "evidence",
        "recommended_action",
        "allowed_now",
        "requires_02_03",
        "requires_04",
        "next_notebook",
        "promotion_allowed_now",
        "source_files",
        "notes",
    ]
    control_profile = str(settings.get("control_profile", ""))
    guard = settings.get("experiment_policy_guard", {}) or _experiment_policy_guard(settings, config)
    readiness = phase2_readiness or {}
    checks = readiness.get("checks", {}) or {}
    blockers = {str(item) for item in readiness.get("blockers", []) or []}
    rows: list[dict[str, Any]] = []

    target_rank_ic = float(_cfg(config, ["validation", "target_rank_ic"], 0.03))
    max_rank_ic_std = float(_cfg(config, ["validation", "max_rank_ic_std"], 0.03))
    min_positive_ic = float(_cfg(config, ["validation", "min_positive_ic_fraction"], 0.75))
    min_long_f1 = float(_cfg(config, ["validation", "min_long_f1"], 0.45))
    max_pred_rate = float(_cfg(config, ["validation", "threshold_checks", "max_pred_long_rate"], 0.70))

    control = _control_comparison_row(comparison, control_profile)
    control_gap = _control_gap_row(performance_gap_analysis, control_profile)
    control_stability = _first_frame_row(
        fold_stability_summary,
        (fold_stability_summary["candidate"].astype(str) == control_profile)
        & (fold_stability_summary["fold_scope"].astype(str) == "full")
        if not fold_stability_summary.empty
        else None,
    )
    control_thresholds = (
        threshold_forensics.loc[
            (threshold_forensics["candidate"].astype(str) == control_profile)
            & (threshold_forensics["fold_scope"].astype(str) == "full")
        ].copy()
        if not threshold_forensics.empty
        else pd.DataFrame()
    )
    control_payoff = (
        payoff_policy_robustness_summary.loc[
            (payoff_policy_robustness_summary["candidate"].astype(str) == control_profile)
            & (payoff_policy_robustness_summary["evaluation_scope"].astype(str) == "cv_test")
        ].copy()
        if not payoff_policy_robustness_summary.empty
        and {"candidate", "evaluation_scope"}.issubset(payoff_policy_robustness_summary.columns)
        else pd.DataFrame()
    )

    def add_row(
        *,
        priority: int,
        blocker: str,
        severity: str,
        metric_value: Any = np.nan,
        target: str = "",
        passed: bool = False,
        evidence: str,
        recommended_action: str,
        allowed_now: bool,
        requires_02_03: bool,
        requires_04: bool,
        next_notebook: str,
        promotion_allowed_now: bool,
        source_files: str,
        notes: str = "",
    ) -> None:
        rows.append(
            {
                "priority": int(priority),
                "blocker": blocker,
                "severity": severity,
                "control_profile": control_profile,
                "metric_value": metric_value,
                "target": target,
                "passed": bool(passed),
                "evidence": evidence,
                "recommended_action": recommended_action,
                "allowed_now": bool(allowed_now),
                "requires_02_03": bool(requires_02_03),
                "requires_04": bool(requires_04),
                "next_notebook": next_notebook,
                "promotion_allowed_now": bool(promotion_allowed_now),
                "source_files": source_files,
                "notes": notes,
            }
        )

    missing_selected = performance_gap_analysis.empty and comparison.empty
    add_row(
        priority=1,
        blocker="experiment_integrity",
        severity="critical" if missing_selected else "ok",
        metric_value=0 if missing_selected else 1,
        target="all selected profiles present in comparison and diagnostics",
        passed=not missing_selected,
        evidence=(
            "No comparison rows were available; rerun 04 before trusting diagnostics."
            if missing_selected
            else f"{len(comparison)} comparison rows and {len(performance_gap_analysis)} performance-gap rows are available."
        ),
        recommended_action=(
            "Rerun 04_training_walk_forward.ipynb to produce completed profile predictions."
            if missing_selected
            else "Continue using 05 diagnostics; experiment integrity is sufficient for review."
        ),
        allowed_now=True,
        requires_02_03=False,
        requires_04=missing_selected,
        next_notebook="04" if missing_selected else "05",
        promotion_allowed_now=False,
        source_files="profile_comparison.csv;performance_gap_analysis.csv;experiment_selection.csv",
        notes="This check prevents empty or stale diagnostics from being interpreted as model performance.",
    )

    mean_rank_ic = _float(control, "mean_rank_ic")
    positive_fraction = _float(control, "positive_ic_fraction")
    mean_passed = bool(np.isfinite(mean_rank_ic) and mean_rank_ic > target_rank_ic)
    positive_passed = bool(np.isfinite(positive_fraction) and positive_fraction >= min_positive_ic)
    add_row(
        priority=2,
        blocker="signal_strength",
        severity="ok" if mean_passed and positive_passed else "high",
        metric_value=mean_rank_ic,
        target=f"mean_rank_ic>{target_rank_ic:.3f}; positive_ic_fraction>={min_positive_ic:.2f}",
        passed=mean_passed and positive_passed,
        evidence=(
            f"Control mean Rank IC={_fmt_metric(mean_rank_ic)}, "
            f"positive IC fraction={_fmt_metric(positive_fraction)}."
        ),
        recommended_action=(
            "Keep the current control as the safe benchmark; do not weaken it with holdout-derived promotions."
            if mean_passed and positive_passed
            else "Stop promotion review and return to feature quality only if this remains weak on future OOS."
        ),
        allowed_now=False,
        requires_02_03=False,
        requires_04=False,
        next_notebook="none",
        promotion_allowed_now=False,
        source_files="profile_comparison.csv;phase2_readiness.json",
        notes="Mean signal is not the current bottleneck; stability and deployment-quality thresholding are.",
    )

    rank_ic_std = _float(control, "std_rank_ic")
    std_passed = bool(np.isfinite(rank_ic_std) and rank_ic_std <= max_rank_ic_std)
    worst_fold = control_stability.get("worst_fold", "NA") if control_stability else "NA"
    worst_ic = control_stability.get("worst_fold_rank_ic", np.nan) if control_stability else np.nan
    top5_var = control_stability.get("top_5_variance_contribution", np.nan) if control_stability else np.nan
    add_row(
        priority=3,
        blocker="fold_stability",
        severity="critical" if not std_passed else "ok",
        metric_value=rank_ic_std,
        target=f"std_rank_ic<={max_rank_ic_std:.3f}",
        passed=std_passed,
        evidence=(
            f"Control Rank IC std={_fmt_metric(rank_ic_std)}; worst fold={worst_fold} "
            f"Rank IC={_fmt_metric(worst_ic)}; top-5 variance contribution={_fmt_metric(top5_var)}."
        ),
        recommended_action=(
            "Use fold_stability_forensics to isolate recurring bad-fold regimes. Do not create new broad profiles or tune "
            "against holdout; any new hypothesis must be pre-registered and checked on CV/future OOS."
        ),
        allowed_now=True,
        requires_02_03=False,
        requires_04=False,
        next_notebook="05",
        promotion_allowed_now=False,
        source_files=(
            "fold_stability_summary.csv;fold_stability_forensics.csv;bad_fold_signature.csv;"
            "score_separation_forensics.csv;feature_drift_forensics.csv;feature_family_drift_summary.csv;"
            "score_distribution_shift.csv;score_distribution_shift_summary.csv;"
            "fold_reliability_gate.csv;fold_reliability_gate_summary.csv;"
            "regime_stability_forensics.csv;regime_stability_summary.csv"
        ),
        notes="This is the main remaining statistical blocker after mean IC improved.",
    )

    official_f1 = _float(control, "test_f1_at_official_threshold", _float(control, "test_f1_at_guarded_threshold"))
    official_rate = _float(
        control,
        "test_pred_long_rate_at_official_threshold",
        _float(control, "test_pred_long_rate_at_guarded_threshold"),
    )
    official_source = str(control.get("official_threshold_source") or control.get("guarded_threshold_source") or "")
    calibrated_f1 = _float(control, "test_f1_at_calibrated_guarded_threshold")
    calibrated_rate = _float(control, "test_pred_long_rate_at_calibrated_guarded_threshold")
    guarded_f1 = _float(control, "test_f1_at_guarded_threshold")
    guarded_rate = _float(control, "test_pred_long_rate_at_guarded_threshold")
    selected_rate = _float(control, "test_pred_long_rate_at_selected_threshold")
    guarded_source = str(control.get("guarded_threshold_source", ""))
    threshold_passed = bool(np.isfinite(official_f1) and official_f1 > min_long_f1 and official_rate <= max_pred_rate)
    issue_counts = (
        control_thresholds["primary_issue"].value_counts().to_dict()
        if not control_thresholds.empty and "primary_issue" in control_thresholds.columns
        else {}
    )
    add_row(
        priority=4,
        blocker="official_threshold_f1",
        severity="critical" if not threshold_passed else "ok",
        metric_value=official_f1,
        target=f"official_f1>{min_long_f1:.2f}; pred_long_rate<={max_pred_rate:.2f}",
        passed=threshold_passed,
        evidence=(
            f"Official F1={_fmt_metric(official_f1)} from {official_source or 'NA'}; "
            f"official pred-long rate={_fmt_metric(official_rate)}; "
            f"raw guarded F1={_fmt_metric(guarded_f1)} from {guarded_source or 'NA'}; "
            f"guarded pred-long rate={_fmt_metric(guarded_rate)}; selected pred-long rate={_fmt_metric(selected_rate)}; "
            f"calibrated guarded F1={_fmt_metric(calibrated_f1)}; calibrated pred-long rate={_fmt_metric(calibrated_rate)}; "
            f"threshold issue counts={issue_counts}."
        ),
        recommended_action=(
            "Optimize score separation/calibration on CV only. Selected-threshold F1 is not official when it exceeds "
            "the pred-long-rate guardrail."
        ),
        allowed_now=True,
        requires_02_03=False,
        requires_04=False,
        next_notebook="05",
        promotion_allowed_now=False,
        source_files=(
            "threshold_forensics.csv;threshold_policy_review.csv;threshold_transfer_review.csv;"
            "regime_threshold_policy_by_fold.csv;regime_threshold_policy_summary.csv;"
            "score_separation_forensics.csv;probability_quality_forensics.csv;probability_quality_summary.csv;"
            "feature_drift_forensics.csv;profile_comparison.csv;phase2_readiness.json"
        ),
        notes="This prevents an unrealistically broad long gate from masking deployment risk.",
    )

    top10_lift = _float(control, "top_10_lift_global")
    holdout_top10_return = _float(control_gap, "holdout_top_10_forward_return_global")
    top10_forward = _float(control_gap, "cv_top_10_forward_return_global")
    payoff_pass = bool(np.isfinite(top10_lift) and top10_lift > 1.0)
    if np.isfinite(holdout_top10_return) and holdout_top10_return <= 0:
        payoff_pass = False
    top_payoff_rows = (
        control_payoff.loc[control_payoff["band"].astype(str) == "top_10"]
        if not control_payoff.empty and "band" in control_payoff.columns
        else pd.DataFrame()
    )
    top_payoff = top_payoff_rows.iloc[0].to_dict() if not top_payoff_rows.empty else {}
    add_row(
        priority=5,
        blocker="score_band_payoff",
        severity="high" if not payoff_pass else "medium",
        metric_value=top10_lift,
        target="top_10 lift>1.0 and forward-return alignment positive",
        passed=payoff_pass,
        evidence=(
            f"CV top-10 lift={_fmt_metric(top10_lift)}; CV top-10 forward return={_fmt_metric(top10_forward, 6)}; "
            f"holdout top-10 forward return={_fmt_metric(holdout_top10_return, 6)}; "
            f"CV payoff alignment fold rate={_fmt_metric(top_payoff.get('payoff_alignment_fold_rate'))}."
        ),
        recommended_action=(
            "Treat score-band rows as policy diagnostics only. If holdout payoff is weak, wait for future OOS instead of "
            "changing the score band on the seen holdout."
        ),
        allowed_now=True,
        requires_02_03=False,
        requires_04=False,
        next_notebook="05",
        promotion_allowed_now=False,
        source_files="payoff_policy_robustness_summary.csv;bad_fold_signature.csv;performance_gap_analysis.csv;holdout_evaluation.csv",
        notes="Label lift alone is insufficient; forward-return payoff must survive unseen data.",
    )

    best_blend = {}
    if not profile_blend.empty:
        sortable = profile_blend.copy()
        if "reviewable" in sortable.columns:
            sortable = sortable.sort_values(
                ["reviewable", "mean_rank_ic", "top_10_lift_global"],
                ascending=[False, False, False],
            )
        else:
            sortable = sortable.sort_values(["mean_rank_ic", "top_10_lift_global"], ascending=[False, False])
        best_blend = sortable.iloc[0].to_dict() if not sortable.empty else {}
    best_blend_name = str(best_blend.get("blend_name") or best_blend.get("profile") or "none")
    best_blend_ic = _float(best_blend, "mean_rank_ic")
    best_blend_std = _float(best_blend, "std_rank_ic")
    best_blend_reviewable = bool(best_blend.get("reviewable", False)) if best_blend else False
    add_row(
        priority=6,
        blocker="candidate_promotion",
        severity="medium",
        metric_value=best_blend_ic,
        target="candidate/blend must beat control gates without worse stability",
        passed=False,
        evidence=(
            f"Best blend by review ordering={best_blend_name}; mean IC={_fmt_metric(best_blend_ic)}; "
            f"std={_fmt_metric(best_blend_std)}; reviewable={best_blend_reviewable}; "
            f"profile search locked={bool(guard.get('profile_search_locked', False))}."
        ),
        recommended_action=(
            "Keep the control profile as the working baseline. Use pre-registered future-OOS candidates only; do not "
            "promote current-holdout winners."
        ),
        allowed_now=False,
        requires_02_03=False,
        requires_04=False,
        next_notebook="none",
        promotion_allowed_now=False,
        source_files="profile_blend.csv;profile_comparison.csv;future_oos_candidate_plan.csv",
        notes="This row intentionally blocks opportunistic promotion from already-seen diagnostics.",
    )

    future_ready = bool(guard.get("future_oos_ready", False))
    future_plan = _first_frame_row(future_oos_candidate_plan)
    min_remaining = guard.get("min_new_bars_remaining", future_plan.get("min_new_bars_remaining", "NA"))
    min_ready_at = future_plan.get("min_ready_at") or guard.get("min_ready_at") or ""
    preferred_ready_at = future_plan.get("preferred_ready_at") or guard.get("preferred_ready_at") or ""
    future_blocked = "future_unseen_oos_not_ready" in blockers or not future_ready
    add_row(
        priority=7,
        blocker="future_unseen_oos",
        severity="critical" if future_blocked else "ok",
        metric_value=0 if future_blocked else 1,
        target="future_oos_ready=True before promotion",
        passed=not future_blocked,
        evidence=(
            f"future_oos_ready={future_ready}; min bars remaining={min_remaining}; "
            f"min_ready_at={min_ready_at}; preferred_ready_at={preferred_ready_at}."
        ),
        recommended_action=(
            "Wait for fresh unseen bars after the anchor, then evaluate only pre-registered candidates. Do not roll "
            "or retune the frozen holdout."
        ),
        allowed_now=False,
        requires_02_03=False,
        requires_04=False,
        next_notebook="none_until_future_oos_ready",
        promotion_allowed_now=future_ready,
        source_files="future_oos_candidate_plan.csv;experiment_policy_guard.csv;phase2_readiness.json",
        notes="This is a governance gate, not a model-performance tweak.",
    )

    phase2_passed = bool(readiness.get("ready_for_phase2", False) or readiness.get("passed", False))
    add_row(
        priority=8,
        blocker="phase2_decision",
        severity="critical" if not phase2_passed else "ok",
        metric_value=1 if phase2_passed else 0,
        target="all Phase 1 readiness checks pass",
        passed=phase2_passed,
        evidence=(
            f"Phase 2 status={readiness.get('status', 'NA')}; blockers={sorted(blockers)}; "
            f"checks={checks}."
        ),
        recommended_action=(
            "Do not build Phase 2 execution/backtest code. Continue diagnostics and future-OOS monitoring until the "
            "official gates pass."
        ),
        allowed_now=True,
        requires_02_03=False,
        requires_04=False,
        next_notebook="05",
        promotion_allowed_now=phase2_passed,
        source_files="phase2_readiness.json;phase1_transition_plan.json;auto_review.json",
        notes="This final row keeps the project aligned with SKILLS.md and the Phase 1 boundary.",
    )

    return pd.DataFrame(rows, columns=columns).sort_values("priority").reset_index(drop=True)


def _phase1_blocker_action_plan_markdown(frame: pd.DataFrame) -> str:
    lines = ["# Phase 1 Blocker Action Plan", ""]
    if frame.empty:
        lines.append("No blocker action plan rows were produced.")
        return "\n".join(lines)
    lines.append(
        "This file translates the current diagnostics into operational actions. "
        "Rows marked as not promotion-allowed must not be used to justify Phase 2."
    )
    lines.append("")
    display_cols = [
        "priority",
        "blocker",
        "severity",
        "metric_value",
        "target",
        "passed",
        "recommended_action",
        "next_notebook",
        "promotion_allowed_now",
        "source_files",
    ]
    visible = frame[[column for column in display_cols if column in frame.columns]].copy()
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]).replace("\n", " ") for column in visible.columns) + " |")
    lines.append("")
    lines.append("## Evidence")
    for _, row in frame.iterrows():
        lines.append("")
        lines.append(f"### {int(row['priority'])}. {row['blocker']}")
        lines.append(str(row["evidence"]))
        if str(row.get("notes", "")):
            lines.append("")
            lines.append(f"Notes: {row['notes']}")
    return "\n".join(lines)


def _write_phase1_blocker_action_plan(path: Path, frame: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path / "phase1_blocker_action_plan.csv", index=False)
    (path / "phase1_blocker_action_plan.md").write_text(
        _phase1_blocker_action_plan_markdown(frame),
        encoding="utf-8",
    )
    _write_json(path / "phase1_blocker_action_plan.json", {"rows": frame.to_dict(orient="records")})


def _diagnostic_candidate_type(fold_scope: str) -> str:
    return "blend" if str(fold_scope).startswith("blend_") else "profile"


def _is_stability_scope(fold_scope: str) -> bool:
    fold_scope = str(fold_scope)
    return fold_scope == "full" or fold_scope.startswith("blend_")


def _entry_official_threshold_source(entry: dict[str, Any]) -> str:
    row = (entry.get("diagnostics", {}) or {}).get("row", {}) or {}
    return str(row.get("official_threshold_source") or row.get("guarded_threshold_source") or "")


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


def _fold_stability_forensics_frame(
    entries: list[dict[str, Any]],
    config: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "fold",
        "start",
        "end",
        "rank_ic",
        "rank_ic_mean",
        "rank_ic_std",
        "rank_ic_zscore",
        "rank_ic_abs_zscore",
        "rank_ic_variance_contribution",
        "rank_ic_std_driver_rank",
        "rank_ic_bucket",
        "long_f1_050",
        "test_f1_at_selected_threshold",
        "test_pred_long_rate_at_selected_threshold",
        "test_f1_at_constrained_threshold",
        "test_pred_long_rate_at_constrained_threshold",
        "official_threshold_source",
        "official_threshold_uses_calibration",
        "test_f1_at_official_threshold",
        "test_pred_long_rate_at_official_threshold",
        "top_10_lift_vs_base",
        "top_10_forward_return",
        "primary_issue",
        "recommended_track",
    ]
    rows: list[dict[str, Any]] = []
    target_rank_ic = float(_cfg(config, ["validation", "target_rank_ic"], 0.03))
    bad_ic = float(_cfg(config, ["validation", "bad_fold_ic_threshold"], -0.08))
    min_long_f1 = float(_cfg(config, ["validation", "min_long_f1"], 0.45))
    max_pred_long_rate = float(_cfg(config, ["validation", "threshold_checks", "max_pred_long_rate"], 0.70))

    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if not _is_stability_scope(fold_scope):
            continue
        diagnostics = entry.get("diagnostics", {}) or {}
        fold_metrics = diagnostics.get("fold_metrics")
        if fold_metrics is None or fold_metrics.empty:
            continue
        frame = fold_metrics.copy()
        threshold_metrics = _entry_threshold_policy_frame(entry)
        if threshold_metrics is not None and not threshold_metrics.empty:
            merge_columns = [
                column
                for column in (
                    "fold",
                    "test_f1_at_selected_threshold",
                    "test_pred_long_rate_at_selected_threshold",
                    "test_f1_at_constrained_threshold",
                    "test_pred_long_rate_at_constrained_threshold",
                    "official_threshold_source",
                    "official_threshold_uses_calibration",
                    "test_f1_at_official_threshold",
                    "test_pred_long_rate_at_official_threshold",
                )
                if column in threshold_metrics.columns
            ]
            if "fold" in merge_columns:
                frame = frame.merge(threshold_metrics[merge_columns], on="fold", how="left")
        score_bands = diagnostics.get("score_band_by_fold")
        if score_bands is not None and not score_bands.empty and {"fold", "band"}.issubset(score_bands.columns):
            top_band = score_bands.loc[score_bands["band"].astype(str) == "top_10"].copy()
            if not top_band.empty:
                rename = {
                    "lift_vs_base": "top_10_lift_vs_base",
                    "mean_forward_return": "top_10_forward_return",
                }
                keep = [column for column in ["fold", *rename.keys()] if column in top_band.columns]
                top_band = top_band[keep].rename(columns=rename)
                frame = frame.merge(top_band, on="fold", how="left")

        rank_values = pd.to_numeric(frame["rank_ic"], errors="coerce")
        mean_rank = float(rank_values.mean()) if rank_values.notna().any() else np.nan
        std_rank = float(rank_values.std(ddof=1)) if rank_values.notna().sum() > 1 else 0.0
        deviations = rank_values - mean_rank
        variance_total = float(np.square(deviations.dropna()).sum())
        frame["_rank_ic_mean"] = mean_rank
        frame["_rank_ic_std"] = std_rank
        frame["_rank_ic_zscore"] = deviations / std_rank if std_rank > 0 else np.nan
        frame["_rank_ic_variance_contribution"] = (
            np.square(deviations) / variance_total if variance_total > 0 else np.nan
        )
        frame["_rank_ic_std_driver_rank"] = (
            frame["_rank_ic_variance_contribution"].rank(method="first", ascending=False)
            if variance_total > 0
            else np.nan
        )

        candidate = str(entry.get("profile", ""))
        for _, item in frame.iterrows():
            row = item.to_dict()
            rank_ic = _float(row, "rank_ic")
            constrained_f1 = _float(row, "test_f1_at_constrained_threshold")
            constrained_rate = _float(row, "test_pred_long_rate_at_constrained_threshold")
            official_f1 = _float(row, "test_f1_at_official_threshold", constrained_f1)
            official_rate = _float(row, "test_pred_long_rate_at_official_threshold", constrained_rate)
            top_return = _float(row, "top_10_forward_return")
            zscore = _float(row, "_rank_ic_zscore")
            abs_zscore = abs(zscore) if np.isfinite(zscore) else np.nan
            if np.isfinite(rank_ic) and rank_ic <= bad_ic:
                bucket = "bad_rank_ic"
                issue = "bad_rank_ic"
                track = "fold_stability"
            elif np.isfinite(rank_ic) and rank_ic < 0:
                bucket = "negative_rank_ic"
                issue = "negative_rank_ic"
                track = "fold_stability"
            elif np.isfinite(rank_ic) and rank_ic < target_rank_ic:
                bucket = "below_target_rank_ic"
                issue = "below_target_rank_ic"
                track = "fold_stability"
            elif np.isfinite(abs_zscore) and abs_zscore >= 1.0:
                bucket = "variance_driver_high_side" if rank_ic >= mean_rank else "variance_driver_low_side"
                issue = bucket
                track = "fold_stability"
            elif np.isfinite(official_f1) and official_f1 < min_long_f1:
                bucket = "rank_ic_ok"
                issue = "official_threshold_f1"
                track = "threshold_calibration"
            elif np.isfinite(official_rate) and official_rate > max_pred_long_rate:
                bucket = "rank_ic_ok"
                issue = "official_threshold_pred_long_rate"
                track = "threshold_calibration"
            elif np.isfinite(top_return) and top_return <= 0:
                bucket = "rank_ic_ok"
                issue = "top_10_payoff"
                track = "score_band_policy"
            else:
                bucket = "rank_ic_ok"
                issue = "ok"
                track = "monitor"
            rows.append(
                {
                    "candidate": candidate,
                    "candidate_type": _diagnostic_candidate_type(fold_scope),
                    "fold_scope": fold_scope,
                    "fold": int(row.get("fold")),
                    "start": str(row.get("start", "")),
                    "end": str(row.get("end", "")),
                    "rank_ic": rank_ic,
                    "rank_ic_mean": mean_rank,
                    "rank_ic_std": std_rank,
                    "rank_ic_zscore": zscore,
                    "rank_ic_abs_zscore": abs_zscore,
                    "rank_ic_variance_contribution": _float(row, "_rank_ic_variance_contribution"),
                    "rank_ic_std_driver_rank": _float(row, "_rank_ic_std_driver_rank"),
                    "rank_ic_bucket": bucket,
                    "long_f1_050": _float(row, "long_f1"),
                    "test_f1_at_selected_threshold": _float(row, "test_f1_at_selected_threshold"),
                    "test_pred_long_rate_at_selected_threshold": _float(row, "test_pred_long_rate_at_selected_threshold"),
                    "test_f1_at_constrained_threshold": constrained_f1,
                    "test_pred_long_rate_at_constrained_threshold": constrained_rate,
                    "official_threshold_source": str(row.get("official_threshold_source", "")),
                    "official_threshold_uses_calibration": bool(row.get("official_threshold_uses_calibration", False)),
                    "test_f1_at_official_threshold": official_f1,
                    "test_pred_long_rate_at_official_threshold": official_rate,
                    "top_10_lift_vs_base": _float(row, "top_10_lift_vs_base"),
                    "top_10_forward_return": top_return,
                    "primary_issue": issue,
                    "recommended_track": track,
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["candidate_type", "candidate", "rank_ic_variance_contribution"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def _fold_stability_summary_frame(forensics: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "fold_count",
        "rank_ic_mean",
        "rank_ic_std",
        "negative_fold_count",
        "bad_fold_count",
        "top_5_variance_contribution",
        "worst_fold",
        "worst_fold_rank_ic",
        "worst_fold_top_10_forward_return",
        "constrained_f1_fail_fold_rate",
        "constrained_pred_rate_fail_fold_rate",
        "official_f1_fail_fold_rate",
        "official_pred_rate_fail_fold_rate",
        "top_10_payoff_fail_fold_rate",
        "main_blocker",
    ]
    if forensics.empty:
        return pd.DataFrame(columns=columns)
    min_long_f1 = float(_cfg(config or {}, ["validation", "min_long_f1"], 0.45))
    max_pred_long_rate = float(_cfg(config or {}, ["validation", "threshold_checks", "max_pred_long_rate"], 0.70))
    rows = []
    for (candidate, candidate_type, fold_scope), part in forensics.groupby(["candidate", "candidate_type", "fold_scope"]):
        sorted_var = part.sort_values("rank_ic_variance_contribution", ascending=False)
        worst = part.sort_values("rank_ic", ascending=True).iloc[0].to_dict()
        constrained_f1_fail = pd.to_numeric(part["test_f1_at_constrained_threshold"], errors="coerce") < min_long_f1
        constrained_rate_fail = pd.to_numeric(part["test_pred_long_rate_at_constrained_threshold"], errors="coerce") > max_pred_long_rate
        official_f1_fail = pd.to_numeric(part["test_f1_at_official_threshold"], errors="coerce") < min_long_f1
        official_rate_fail = pd.to_numeric(part["test_pred_long_rate_at_official_threshold"], errors="coerce") > max_pred_long_rate
        top_payoff_fail = pd.to_numeric(part["top_10_forward_return"], errors="coerce") <= 0.0
        issue_counts = part["primary_issue"].value_counts()
        main_blocker = str(issue_counts.index[0]) if not issue_counts.empty else ""
        rows.append(
            {
                "candidate": candidate,
                "candidate_type": candidate_type,
                "fold_scope": fold_scope,
                "fold_count": int(part["fold"].nunique()),
                "rank_ic_mean": float(pd.to_numeric(part["rank_ic"], errors="coerce").mean()),
                "rank_ic_std": float(pd.to_numeric(part["rank_ic"], errors="coerce").std(ddof=1)),
                "negative_fold_count": int((pd.to_numeric(part["rank_ic"], errors="coerce") < 0.0).sum()),
                "bad_fold_count": int((part["rank_ic_bucket"].astype(str) == "bad_rank_ic").sum()),
                "top_5_variance_contribution": float(
                    pd.to_numeric(sorted_var["rank_ic_variance_contribution"], errors="coerce").head(5).sum()
                ),
                "worst_fold": int(worst.get("fold")),
                "worst_fold_rank_ic": _float(worst, "rank_ic"),
                "worst_fold_top_10_forward_return": _float(worst, "top_10_forward_return"),
                "constrained_f1_fail_fold_rate": float(constrained_f1_fail.mean()),
                "constrained_pred_rate_fail_fold_rate": float(constrained_rate_fail.mean()),
                "official_f1_fail_fold_rate": float(official_f1_fail.mean()),
                "official_pred_rate_fail_fold_rate": float(official_rate_fail.mean()),
                "top_10_payoff_fail_fold_rate": float(top_payoff_fail.mean()),
                "main_blocker": main_blocker,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["candidate_type", "rank_ic_std"], ascending=[True, True]).reset_index(drop=True)


def _threshold_forensics_frame(entries: list[dict[str, Any]], config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "fold",
        "selected_threshold",
        "test_f1_at_selected_threshold",
        "test_pred_long_rate_at_selected_threshold",
        "constrained_threshold",
        "test_f1_at_constrained_threshold",
        "test_pred_long_rate_at_constrained_threshold",
        "calibrated_constrained_threshold",
        "test_f1_at_calibrated_constrained_threshold",
        "test_pred_long_rate_at_calibrated_constrained_threshold",
        "official_threshold_source",
        "official_threshold_uses_calibration",
        "official_threshold",
        "test_f1_at_official_threshold",
        "test_precision_at_official_threshold",
        "test_recall_at_official_threshold",
        "test_pred_long_rate_at_official_threshold",
        "test_oracle_best_f1",
        "selected_f1_gap_vs_target",
        "constrained_f1_gap_vs_target",
        "official_f1_gap_vs_target",
        "selected_pred_rate_excess_vs_guardrail",
        "constrained_pred_rate_excess_vs_guardrail",
        "official_pred_rate_excess_vs_guardrail",
        "primary_issue",
        "recommended_action",
    ]
    min_long_f1 = float(_cfg(config, ["validation", "min_long_f1"], 0.45))
    max_pred_long_rate = float(_cfg(config, ["validation", "threshold_checks", "max_pred_long_rate"], 0.70))
    rows: list[dict[str, Any]] = []
    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if not _is_stability_scope(fold_scope):
            continue
        threshold_metrics = _entry_threshold_policy_frame(entry)
        if threshold_metrics is None or threshold_metrics.empty:
            continue
        candidate = str(entry.get("profile", ""))
        for _, item in threshold_metrics.iterrows():
            row = item.to_dict()
            selected_f1 = _float(row, "test_f1_at_selected_threshold")
            constrained_f1 = _float(row, "test_f1_at_constrained_threshold")
            selected_rate = _float(row, "test_pred_long_rate_at_selected_threshold")
            constrained_rate = _float(row, "test_pred_long_rate_at_constrained_threshold")
            official_f1 = _float(row, "test_f1_at_official_threshold", constrained_f1)
            official_rate = _float(row, "test_pred_long_rate_at_official_threshold", constrained_rate)
            selected_gap = min_long_f1 - selected_f1
            constrained_gap = min_long_f1 - constrained_f1
            official_gap = min_long_f1 - official_f1
            selected_rate_excess = selected_rate - max_pred_long_rate
            constrained_rate_excess = constrained_rate - max_pred_long_rate
            official_rate_excess = official_rate - max_pred_long_rate
            official_source = str(row.get("official_threshold_source", ""))
            if np.isfinite(official_rate_excess) and official_rate_excess > 0:
                issue = "official_pred_long_rate"
                action = "raise_cv_threshold_or_reduce_score_compression"
            elif np.isfinite(official_gap) and official_gap > 0:
                issue = "official_f1"
                action = "improve_score_ranking_or_calibration_before_phase2"
            elif np.isfinite(selected_rate_excess) and selected_rate_excess > 0:
                issue = "selected_threshold_too_broad"
                action = "prefer_constrained_threshold_for_review"
            elif np.isfinite(selected_gap) and selected_gap > 0:
                issue = "selected_f1"
                action = "improve_validation_threshold_transfer"
            elif np.isfinite(constrained_rate_excess) and constrained_rate_excess > 0:
                issue = "constrained_pred_long_rate"
                action = "raise_cv_threshold_or_reduce_score_compression"
            elif np.isfinite(constrained_gap) and constrained_gap > 0:
                issue = "constrained_f1"
                action = "improve_score_ranking_or_calibration_before_phase2"
            else:
                issue = "ok"
                action = "monitor"
            rows.append(
                {
                    "candidate": candidate,
                    "candidate_type": _diagnostic_candidate_type(fold_scope),
                    "fold_scope": fold_scope,
                    "fold": int(row.get("fold")),
                    "selected_threshold": _float(row, "selected_threshold"),
                    "test_f1_at_selected_threshold": selected_f1,
                    "test_pred_long_rate_at_selected_threshold": selected_rate,
                    "constrained_threshold": _float(row, "constrained_threshold"),
                    "test_f1_at_constrained_threshold": constrained_f1,
                    "test_pred_long_rate_at_constrained_threshold": constrained_rate,
                    "calibrated_constrained_threshold": _float(row, "calibrated_constrained_threshold"),
                    "test_f1_at_calibrated_constrained_threshold": _float(
                        row,
                        "calibrated_test_f1_at_constrained_threshold",
                    ),
                    "test_pred_long_rate_at_calibrated_constrained_threshold": _float(
                        row,
                        "calibrated_test_pred_long_rate_at_constrained_threshold",
                    ),
                    "official_threshold_source": official_source,
                    "official_threshold_uses_calibration": bool(row.get("official_threshold_uses_calibration", False)),
                    "official_threshold": _float(row, "official_threshold"),
                    "test_f1_at_official_threshold": official_f1,
                    "test_precision_at_official_threshold": _float(row, "test_precision_at_official_threshold"),
                    "test_recall_at_official_threshold": _float(row, "test_recall_at_official_threshold"),
                    "test_pred_long_rate_at_official_threshold": official_rate,
                    "test_oracle_best_f1": _float(row, "test_oracle_best_f1"),
                    "selected_f1_gap_vs_target": selected_gap,
                    "constrained_f1_gap_vs_target": constrained_gap,
                    "official_f1_gap_vs_target": official_gap,
                    "selected_pred_rate_excess_vs_guardrail": selected_rate_excess,
                    "constrained_pred_rate_excess_vs_guardrail": constrained_rate_excess,
                    "official_pred_rate_excess_vs_guardrail": official_rate_excess,
                    "primary_issue": issue,
                    "recommended_action": action,
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["candidate_type", "candidate", "constrained_f1_gap_vs_target"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def _threshold_policy_review_frame(entries: list[dict[str, Any]], config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "policy_name",
        "policy_type",
        "threshold_source",
        "threshold_cap",
        "threshold_mean",
        "source_selection_metric",
        "source_f1",
        "source_precision",
        "source_pred_long_rate",
        "test_f1",
        "test_precision",
        "test_recall",
        "test_pred_long_rate",
        "mean_lift_vs_base",
        "mean_forward_return",
        "positive_lift_fold_rate",
        "positive_forward_return_fold_rate",
        "constraints_satisfied_fold_rate",
        "f1_passed",
        "precision_passed",
        "pred_long_rate_passed",
        "policy_passed_cv_test",
        "selection_guard",
        "recommended_action",
    ]
    min_long_f1 = float(_cfg(config, ["validation", "min_long_f1"], 0.45))
    threshold_cfg = _cfg(config, ["validation", "threshold_checks"], {}) or {}
    max_pred_long_rate = float(threshold_cfg.get("max_pred_long_rate", 0.70))
    min_precision = float(threshold_cfg.get("min_precision", 0.30))
    rows: list[dict[str, Any]] = []

    def summary_metric(summary: pd.DataFrame | None, metric: str) -> float:
        return _threshold_summary_metric(summary, metric)

    def add_row(
        *,
        entry: dict[str, Any],
        policy_name: str,
        policy_type: str,
        threshold_source: str,
        threshold_cap: float = np.nan,
        threshold_mean: float = np.nan,
        source_selection_metric: str = "",
        source_f1: float = np.nan,
        source_precision: float = np.nan,
        source_pred_long_rate: float = np.nan,
        test_f1: float = np.nan,
        test_precision: float = np.nan,
        test_recall: float = np.nan,
        test_pred_long_rate: float = np.nan,
        mean_lift_vs_base: float = np.nan,
        mean_forward_return: float = np.nan,
        positive_lift_fold_rate: float = np.nan,
        positive_forward_return_fold_rate: float = np.nan,
        constraints_satisfied_fold_rate: float = np.nan,
    ) -> None:
        f1_passed = bool(np.isfinite(test_f1) and test_f1 > min_long_f1)
        precision_passed = bool(np.isfinite(test_precision) and test_precision >= min_precision)
        rate_passed = bool(np.isfinite(test_pred_long_rate) and test_pred_long_rate <= max_pred_long_rate)
        policy_passed = bool(f1_passed and precision_passed and rate_passed)
        if policy_passed:
            action = "monitor_on_future_oos_only"
        elif not f1_passed:
            action = "score_separation_gap_do_not_promote"
        elif not rate_passed:
            action = "threshold_too_broad_do_not_promote"
        else:
            action = "precision_gap_do_not_promote"
        rows.append(
            {
                "candidate": str(entry.get("profile", "")),
                "candidate_type": _diagnostic_candidate_type(str(entry.get("fold_scope", ""))),
                "fold_scope": str(entry.get("fold_scope", "")),
                "policy_name": policy_name,
                "policy_type": policy_type,
                "threshold_source": threshold_source,
                "threshold_cap": threshold_cap,
                "threshold_mean": threshold_mean,
                "source_selection_metric": source_selection_metric,
                "source_f1": source_f1,
                "source_precision": source_precision,
                "source_pred_long_rate": source_pred_long_rate,
                "test_f1": test_f1,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_pred_long_rate": test_pred_long_rate,
                "mean_lift_vs_base": mean_lift_vs_base,
                "mean_forward_return": mean_forward_return,
                "positive_lift_fold_rate": positive_lift_fold_rate,
                "positive_forward_return_fold_rate": positive_forward_return_fold_rate,
                "constraints_satisfied_fold_rate": constraints_satisfied_fold_rate,
                "f1_passed": f1_passed,
                "precision_passed": precision_passed,
                "pred_long_rate_passed": rate_passed,
                "policy_passed_cv_test": policy_passed,
                "selection_guard": "source_threshold_selected_on_validation_not_test",
                "recommended_action": action,
            }
        )

    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if not _is_stability_scope(fold_scope):
            continue
        diagnostics = entry.get("diagnostics", {}) or {}
        summary = diagnostics.get("threshold_summary")
        if summary is not None and not summary.empty:
            add_row(
                entry=entry,
                policy_name="validation_selected_threshold",
                policy_type="threshold",
                threshold_source="validation_selected_threshold",
                threshold_mean=summary_metric(summary, "selected_threshold"),
                source_selection_metric="source_best_f1",
                source_f1=summary_metric(summary, "source_best_f1"),
                test_f1=summary_metric(summary, "test_f1_at_selected_threshold"),
                test_precision=summary_metric(summary, "test_precision_at_selected_threshold"),
                test_recall=summary_metric(summary, "test_recall_at_selected_threshold"),
                test_pred_long_rate=summary_metric(summary, "test_pred_long_rate_at_selected_threshold"),
            )
            add_row(
                entry=entry,
                policy_name="validation_constrained_threshold",
                policy_type="threshold",
                threshold_source="validation_constrained_threshold",
                threshold_mean=summary_metric(summary, "constrained_threshold"),
                source_selection_metric="source_constrained_f1",
                source_f1=summary_metric(summary, "source_constrained_f1"),
                source_precision=summary_metric(summary, "source_constrained_precision"),
                source_pred_long_rate=summary_metric(summary, "source_constrained_pred_long_rate"),
                test_f1=summary_metric(summary, "test_f1_at_constrained_threshold"),
                test_precision=summary_metric(summary, "test_precision_at_constrained_threshold"),
                test_recall=summary_metric(summary, "test_recall_at_constrained_threshold"),
                test_pred_long_rate=summary_metric(summary, "test_pred_long_rate_at_constrained_threshold"),
            )
        calibrated_summary = diagnostics.get("calibrated_threshold_summary")
        if calibrated_summary is not None and not calibrated_summary.empty:
            add_row(
                entry=entry,
                policy_name="calibrated_validation_constrained_threshold",
                policy_type="threshold",
                threshold_source="calibrated_validation_constrained_threshold",
                threshold_mean=summary_metric(calibrated_summary, "constrained_threshold"),
                source_selection_metric="source_constrained_f1",
                source_f1=summary_metric(calibrated_summary, "source_constrained_f1"),
                source_precision=summary_metric(calibrated_summary, "source_constrained_precision"),
                source_pred_long_rate=summary_metric(calibrated_summary, "source_constrained_pred_long_rate"),
                test_f1=summary_metric(calibrated_summary, "test_f1_at_constrained_threshold"),
                test_precision=summary_metric(calibrated_summary, "test_precision_at_constrained_threshold"),
                test_recall=summary_metric(calibrated_summary, "test_recall_at_constrained_threshold"),
                test_pred_long_rate=summary_metric(calibrated_summary, "test_pred_long_rate_at_constrained_threshold"),
            )
        grid_summary = diagnostics.get("threshold_grid_summary")
        if grid_summary is not None and not grid_summary.empty:
            for _, item in grid_summary.iterrows():
                cap = _float(item.to_dict(), "max_pred_long_rate")
                add_row(
                    entry=entry,
                    policy_name=f"validation_threshold_cap_{cap:.2f}",
                    policy_type="threshold_cap",
                    threshold_source="validation_threshold_cap_sweep",
                    threshold_cap=cap,
                    threshold_mean=_float(item.to_dict(), "threshold_mean"),
                    source_selection_metric="mean_source_f1",
                    source_f1=_float(item.to_dict(), "mean_source_f1"),
                    source_precision=_float(item.to_dict(), "mean_source_precision"),
                    source_pred_long_rate=_float(item.to_dict(), "mean_source_pred_long_rate"),
                    test_f1=_float(item.to_dict(), "mean_f1"),
                    test_precision=_float(item.to_dict(), "mean_precision"),
                    test_recall=_float(item.to_dict(), "mean_recall"),
                    test_pred_long_rate=_float(item.to_dict(), "mean_selection_rate"),
                    mean_lift_vs_base=_float(item.to_dict(), "mean_lift_vs_base"),
                    mean_forward_return=_float(item.to_dict(), "mean_forward_return"),
                    positive_lift_fold_rate=_float(item.to_dict(), "positive_lift_fold_rate"),
                    positive_forward_return_fold_rate=_float(item.to_dict(), "positive_forward_return_fold_rate"),
                    constraints_satisfied_fold_rate=_float(item.to_dict(), "constraints_satisfied_fold_rate"),
                )
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(
            ["candidate_type", "candidate", "fold_scope", "policy_passed_cv_test", "test_f1", "test_pred_long_rate"],
            ascending=[True, True, True, False, False, True],
        )
        .reset_index(drop=True)
    )


def _threshold_policy_review_markdown(frame: pd.DataFrame) -> str:
    lines = ["# Threshold Policy Review", ""]
    if frame.empty:
        lines.append("No threshold policy rows were produced.")
        return "\n".join(lines)
    lines.append(
        "Policies are evaluated on CV test folds, but threshold values are selected from validation/source folds. "
        "Rows are diagnostics only and must not be promoted from the seen holdout."
    )
    lines.append("")
    display_cols = [
        "candidate",
        "fold_scope",
        "policy_name",
        "test_f1",
        "test_precision",
        "test_pred_long_rate",
        "source_f1",
        "mean_lift_vs_base",
        "mean_forward_return",
        "policy_passed_cv_test",
        "recommended_action",
    ]
    visible = frame[[column for column in display_cols if column in frame.columns]].copy()
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    return "\n".join(lines)


def _write_threshold_policy_review(path: Path, frame: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path / "threshold_policy_review.csv", index=False)
    (path / "threshold_policy_review.md").write_text(_threshold_policy_review_markdown(frame), encoding="utf-8")
    _write_json(path / "threshold_policy_review.json", {"rows": frame.to_dict(orient="records")})


def _threshold_metrics_at_value(labels: pd.Series, scores: pd.Series, threshold: float) -> dict[str, float]:
    labels_array = labels.astype(int).to_numpy()
    scores_array = scores.astype(float).to_numpy()
    predictions = scores_array >= float(threshold)
    tp = int(((labels_array == 1) & predictions).sum())
    fp = int(((labels_array == 0) & predictions).sum())
    fn = int(((labels_array == 1) & ~predictions).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "pred_long_rate": float(predictions.mean()) if len(predictions) else 0.0,
    }


def _threshold_selection_stats(frame: pd.DataFrame, threshold: float) -> dict[str, float]:
    if frame.empty or "prob_long" not in frame.columns or "label" not in frame.columns:
        return {
            "actual_long_rate": np.nan,
            "label_lift_vs_base": np.nan,
            "mean_forward_return": np.nan,
            "mean_tb_return": np.nan,
        }
    scores = pd.to_numeric(frame["prob_long"], errors="coerce")
    labels = pd.to_numeric(frame["label"], errors="coerce")
    base_rate = float(labels.mean()) if labels.notna().any() else np.nan
    selected = frame.loc[scores >= float(threshold)].copy()
    if selected.empty:
        return {
            "actual_long_rate": 0.0,
            "label_lift_vs_base": 0.0 if np.isfinite(base_rate) and base_rate > 0 else np.nan,
            "mean_forward_return": np.nan,
            "mean_tb_return": np.nan,
        }
    actual_rate = float(pd.to_numeric(selected["label"], errors="coerce").mean())
    return {
        "actual_long_rate": actual_rate,
        "label_lift_vs_base": float(actual_rate / base_rate) if np.isfinite(base_rate) and base_rate > 0 else np.nan,
        "mean_forward_return": (
            float(pd.to_numeric(selected["forward_return"], errors="coerce").mean())
            if "forward_return" in selected.columns
            else np.nan
        ),
        "mean_tb_return": (
            float(pd.to_numeric(selected["tb_return"], errors="coerce").mean())
            if "tb_return" in selected.columns
            else np.nan
        ),
    }


def _regime_columns(frame: pd.DataFrame) -> list[str]:
    return sorted([column for column in frame.columns if str(column).startswith("regime_prob_")])


def _with_dominant_regime(frame: pd.DataFrame) -> pd.DataFrame:
    regime_columns = _regime_columns(frame)
    if not regime_columns:
        return pd.DataFrame()
    out = frame.copy()
    out["dominant_regime"] = out[regime_columns].idxmax(axis=1).str.rsplit("_", n=1).str[-1].astype(int)
    return out


def _select_validation_threshold(
    frame: pd.DataFrame,
    *,
    max_pred_long_rate: float,
    min_precision: float,
) -> dict[str, float]:
    if frame.empty or not {"label", "prob_long"}.issubset(frame.columns):
        return {
            "threshold": np.nan,
            "source_f1": np.nan,
            "source_precision": np.nan,
            "source_recall": np.nan,
            "source_pred_long_rate": np.nan,
            "constraint_satisfied": False,
        }
    clean = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["label", "prob_long"]).copy()
    if clean.empty:
        return _select_validation_threshold(
            pd.DataFrame(),
            max_pred_long_rate=max_pred_long_rate,
            min_precision=min_precision,
        )
    labels = clean["label"].astype(int)
    scores = pd.to_numeric(clean["prob_long"], errors="coerce")
    candidates = set(float(value) for value in scores.dropna().unique())
    candidates.update(float(value) for value in scores.quantile([0.50, 0.60, 0.70, 0.80, 0.90]).dropna().to_list())
    candidates.add(0.5)
    if scores.notna().any():
        candidates.add(float(scores.max()) + 1e-9)
        candidates.add(float(scores.min()) - 1e-9)
    rows: list[dict[str, float]] = []
    for threshold in sorted(candidates):
        metrics = _threshold_metrics_at_value(labels, scores, threshold)
        constraint = (
            metrics["pred_long_rate"] <= max_pred_long_rate
            and metrics["precision"] >= min_precision
        )
        rows.append(
            {
                "threshold": float(threshold),
                "source_f1": metrics["f1"],
                "source_precision": metrics["precision"],
                "source_recall": metrics["recall"],
                "source_pred_long_rate": metrics["pred_long_rate"],
                "constraint_satisfied": bool(constraint),
            }
        )
    if not rows:
        return _select_validation_threshold(
            pd.DataFrame(),
            max_pred_long_rate=max_pred_long_rate,
            min_precision=min_precision,
        )
    frame_rows = pd.DataFrame(rows)
    constrained = frame_rows.loc[frame_rows["constraint_satisfied"].astype(bool)].copy()
    if constrained.empty:
        constrained = frame_rows.loc[frame_rows["source_pred_long_rate"] <= max_pred_long_rate].copy()
    if constrained.empty:
        constrained = frame_rows.copy()
    selected = constrained.sort_values(
        ["source_f1", "source_precision", "source_pred_long_rate", "threshold"],
        ascending=[False, False, True, True],
    ).iloc[0]
    return {
        "threshold": float(selected["threshold"]),
        "source_f1": float(selected["source_f1"]),
        "source_precision": float(selected["source_precision"]),
        "source_recall": float(selected["source_recall"]),
        "source_pred_long_rate": float(selected["source_pred_long_rate"]),
        "constraint_satisfied": bool(selected["constraint_satisfied"]),
    }


def _metrics_for_masked_predictions(frame: pd.DataFrame, predictions: pd.Series) -> dict[str, float]:
    labels = frame["label"].astype(int).to_numpy()
    pred = predictions.astype(bool).to_numpy()
    tp = int(((labels == 1) & pred).sum())
    fp = int(((labels == 0) & pred).sum())
    fn = int(((labels == 1) & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    selected = frame.loc[pred].copy()
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "pred_long_rate": float(pred.mean()) if len(pred) else 0.0,
        "selected_count": int(pred.sum()),
        "label_lift_vs_base": (
            float(selected["label"].mean() / frame["label"].mean())
            if not selected.empty and float(frame["label"].mean()) > 0
            else np.nan
        ),
        "mean_forward_return": (
            float(pd.to_numeric(selected["forward_return"], errors="coerce").mean())
            if not selected.empty and "forward_return" in selected.columns
            else np.nan
        ),
        "mean_tb_return": (
            float(pd.to_numeric(selected["tb_return"], errors="coerce").mean())
            if not selected.empty and "tb_return" in selected.columns
            else np.nan
        ),
    }


def _regime_threshold_policy_frames(
    entries: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_fold_columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "fold",
        "policy_name",
        "regime_count",
        "regime_threshold_count",
        "fallback_threshold",
        "regime_thresholds_json",
        "validation_f1",
        "validation_precision",
        "validation_recall",
        "validation_pred_long_rate",
        "test_f1",
        "test_precision",
        "test_recall",
        "test_pred_long_rate",
        "test_label_lift_vs_base",
        "test_mean_forward_return",
        "test_mean_tb_return",
        "official_f1",
        "official_precision",
        "official_recall",
        "official_pred_long_rate",
        "f1_delta_vs_official",
        "precision_delta_vs_official",
        "pred_long_rate_delta_vs_official",
        "policy_passed_fold",
        "selection_guard",
        "reject_reason",
    ]
    summary_columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "policy_name",
        "fold_count",
        "test_f1_mean",
        "test_precision_mean",
        "test_pred_long_rate_mean",
        "test_mean_forward_return_mean",
        "official_f1_mean",
        "official_pred_long_rate_mean",
        "f1_delta_vs_official_mean",
        "pred_long_rate_delta_vs_official_mean",
        "policy_passed_fold_rate",
        "positive_forward_return_fold_rate",
        "regime_threshold_count_mean",
        "reviewable",
        "reject_reason",
        "next_action",
    ]
    cfg = _cfg(config, ["validation", "regime_threshold_policy"], {}) or {}
    if not bool(cfg.get("enabled", False)):
        return pd.DataFrame(columns=by_fold_columns), pd.DataFrame(columns=summary_columns)
    min_val_rows = int(cfg.get("min_regime_val_rows", 80))
    min_test_rows = int(cfg.get("min_regime_test_rows", 40))
    min_val_longs = int(cfg.get("min_regime_val_longs", 5))
    max_pred_long_rate = float(cfg.get("max_pred_long_rate", _cfg(config, ["validation", "threshold_checks", "max_pred_long_rate"], 0.70)))
    min_precision = float(cfg.get("min_precision", _cfg(config, ["validation", "threshold_checks", "min_precision"], 0.30)))
    min_long_f1 = float(_cfg(config, ["validation", "min_long_f1"], 0.45))
    rows: list[dict[str, Any]] = []
    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if not _is_stability_scope(fold_scope):
            continue
        predictions = entry.get("predictions")
        if not isinstance(predictions, pd.DataFrame) or predictions.empty:
            continue
        frame = _with_dominant_regime(predictions)
        if frame.empty or not {"fold", "split", "label", "prob_long"}.issubset(frame.columns):
            continue
        threshold_metrics = _entry_threshold_policy_frame(entry)
        threshold_by_fold = (
            {int(row["fold"]): row.to_dict() for _, row in threshold_metrics.dropna(subset=["fold"]).iterrows()}
            if threshold_metrics is not None and not threshold_metrics.empty and "fold" in threshold_metrics.columns
            else {}
        )
        candidate = str(entry.get("profile", ""))
        candidate_type = _diagnostic_candidate_type(fold_scope)
        for fold, fold_part in frame.groupby("fold"):
            fold_id = int(fold)
            validation = fold_part.loc[fold_part["split"].astype(str) == "val"].copy()
            test = fold_part.loc[fold_part["split"].astype(str) == "test"].copy()
            if validation.empty or test.empty:
                continue
            threshold_row = threshold_by_fold.get(fold_id, {})
            fallback = _float(
                threshold_row,
                "official_threshold",
                _float(threshold_row, "constrained_threshold", _float(threshold_row, "selected_threshold", 0.5)),
            )
            if not np.isfinite(fallback):
                fallback = 0.5
            regime_thresholds: dict[int, dict[str, float]] = {}
            for regime, val_regime in validation.groupby("dominant_regime"):
                test_regime = test.loc[test["dominant_regime"].astype(int) == int(regime)]
                if len(val_regime) < min_val_rows or len(test_regime) < min_test_rows:
                    continue
                if int(pd.to_numeric(val_regime["label"], errors="coerce").sum()) < min_val_longs:
                    continue
                selected = _select_validation_threshold(
                    val_regime,
                    max_pred_long_rate=max_pred_long_rate,
                    min_precision=min_precision,
                )
                if np.isfinite(selected["threshold"]):
                    regime_thresholds[int(regime)] = selected

            def thresholds_for(part: pd.DataFrame) -> pd.Series:
                return part["dominant_regime"].astype(int).map(
                    {regime: values["threshold"] for regime, values in regime_thresholds.items()}
                ).fillna(fallback)

            validation_thresholds = thresholds_for(validation)
            test_thresholds = thresholds_for(test)
            validation_predictions = pd.to_numeric(validation["prob_long"], errors="coerce") >= validation_thresholds
            test_predictions = pd.to_numeric(test["prob_long"], errors="coerce") >= test_thresholds
            validation_metrics = _metrics_for_masked_predictions(validation, validation_predictions)
            test_metrics = _metrics_for_masked_predictions(test, test_predictions)
            official_f1 = _float(
                threshold_row,
                "test_f1_at_official_threshold",
                _float(threshold_row, "test_f1_at_constrained_threshold"),
            )
            official_precision = _float(
                threshold_row,
                "test_precision_at_official_threshold",
                _float(threshold_row, "test_precision_at_constrained_threshold"),
            )
            official_recall = _float(
                threshold_row,
                "test_recall_at_official_threshold",
                _float(threshold_row, "test_recall_at_constrained_threshold"),
            )
            official_rate = _float(
                threshold_row,
                "test_pred_long_rate_at_official_threshold",
                _float(threshold_row, "test_pred_long_rate_at_constrained_threshold"),
            )
            reasons: list[str] = []
            if test_metrics["f1"] < min_long_f1:
                reasons.append("test_f1")
            if test_metrics["precision"] < min_precision:
                reasons.append("test_precision")
            if test_metrics["pred_long_rate"] > max_pred_long_rate:
                reasons.append("test_pred_long_rate")
            if np.isfinite(official_f1) and test_metrics["f1"] <= official_f1:
                reasons.append("f1_not_above_official")
            if np.isfinite(test_metrics["mean_forward_return"]) and test_metrics["mean_forward_return"] <= 0:
                reasons.append("selected_forward_return")
            rows.append(
                {
                    "candidate": candidate,
                    "candidate_type": candidate_type,
                    "fold_scope": fold_scope,
                    "fold": fold_id,
                    "policy_name": "validation_regime_specific_threshold",
                    "regime_count": int(test["dominant_regime"].nunique()),
                    "regime_threshold_count": int(len(regime_thresholds)),
                    "fallback_threshold": float(fallback),
                    "regime_thresholds_json": json.dumps(regime_thresholds, sort_keys=True),
                    "validation_f1": validation_metrics["f1"],
                    "validation_precision": validation_metrics["precision"],
                    "validation_recall": validation_metrics["recall"],
                    "validation_pred_long_rate": validation_metrics["pred_long_rate"],
                    "test_f1": test_metrics["f1"],
                    "test_precision": test_metrics["precision"],
                    "test_recall": test_metrics["recall"],
                    "test_pred_long_rate": test_metrics["pred_long_rate"],
                    "test_label_lift_vs_base": test_metrics["label_lift_vs_base"],
                    "test_mean_forward_return": test_metrics["mean_forward_return"],
                    "test_mean_tb_return": test_metrics["mean_tb_return"],
                    "official_f1": official_f1,
                    "official_precision": official_precision,
                    "official_recall": official_recall,
                    "official_pred_long_rate": official_rate,
                    "f1_delta_vs_official": (
                        test_metrics["f1"] - official_f1 if np.isfinite(official_f1) else np.nan
                    ),
                    "precision_delta_vs_official": (
                        test_metrics["precision"] - official_precision if np.isfinite(official_precision) else np.nan
                    ),
                    "pred_long_rate_delta_vs_official": (
                        test_metrics["pred_long_rate"] - official_rate if np.isfinite(official_rate) else np.nan
                    ),
                    "policy_passed_fold": not bool(reasons),
                    "selection_guard": "per_regime_thresholds_selected_on_validation_only",
                    "reject_reason": ";".join(dict.fromkeys(reasons)),
                }
            )
    by_fold = pd.DataFrame(rows, columns=by_fold_columns) if rows else pd.DataFrame(columns=by_fold_columns)
    if by_fold.empty:
        return by_fold, pd.DataFrame(columns=summary_columns)
    summary_rows: list[dict[str, Any]] = []
    min_delta = float(cfg.get("min_f1_delta_vs_official", 0.01))
    min_pass_rate = float(cfg.get("min_policy_pass_fold_rate", 0.55))
    min_positive_return_rate = float(cfg.get("min_positive_forward_return_fold_rate", 0.55))
    for (candidate, candidate_type, fold_scope, policy_name), part in by_fold.groupby(
        ["candidate", "candidate_type", "fold_scope", "policy_name"],
        dropna=False,
    ):
        f1_delta = _numeric_mean(part, "f1_delta_vs_official")
        pass_rate = float(part["policy_passed_fold"].astype(bool).mean())
        positive_return_rate = float((pd.to_numeric(part["test_mean_forward_return"], errors="coerce") > 0).mean())
        reasons: list[str] = []
        if _numeric_mean(part, "test_f1") < min_long_f1:
            reasons.append("test_f1")
        if _numeric_mean(part, "test_pred_long_rate") > max_pred_long_rate:
            reasons.append("test_pred_long_rate")
        if not np.isfinite(f1_delta) or f1_delta < min_delta:
            reasons.append("f1_delta_vs_official")
        if pass_rate < min_pass_rate:
            reasons.append("policy_pass_fold_rate")
        if positive_return_rate < min_positive_return_rate:
            reasons.append("positive_forward_return_fold_rate")
        reject_reason = ";".join(dict.fromkeys(reasons))
        summary_rows.append(
            {
                "candidate": candidate,
                "candidate_type": candidate_type,
                "fold_scope": fold_scope,
                "policy_name": policy_name,
                "fold_count": int(part["fold"].nunique()),
                "test_f1_mean": _numeric_mean(part, "test_f1"),
                "test_precision_mean": _numeric_mean(part, "test_precision"),
                "test_pred_long_rate_mean": _numeric_mean(part, "test_pred_long_rate"),
                "test_mean_forward_return_mean": _numeric_mean(part, "test_mean_forward_return"),
                "official_f1_mean": _numeric_mean(part, "official_f1"),
                "official_pred_long_rate_mean": _numeric_mean(part, "official_pred_long_rate"),
                "f1_delta_vs_official_mean": f1_delta,
                "pred_long_rate_delta_vs_official_mean": _numeric_mean(part, "pred_long_rate_delta_vs_official"),
                "policy_passed_fold_rate": pass_rate,
                "positive_forward_return_fold_rate": positive_return_rate,
                "regime_threshold_count_mean": _numeric_mean(part, "regime_threshold_count"),
                "reviewable": not bool(reject_reason),
                "reject_reason": reject_reason,
                "next_action": (
                    "pre_register_regime_threshold_policy_for_future_oos_review"
                    if not reject_reason
                    else "diagnostic_only_do_not_promote"
                ),
            }
        )
    summary = (
        pd.DataFrame(summary_rows, columns=summary_columns)
        .sort_values(["reviewable", "f1_delta_vs_official_mean", "test_f1_mean"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    return by_fold, summary


def _regime_threshold_policy_markdown(summary: pd.DataFrame) -> str:
    lines = ["# Regime Threshold Policy Review", ""]
    lines.append(
        "Per-regime thresholds are selected on each fold's validation split and evaluated on that fold's test split. "
        "This is a CV-only diagnostic and must not be promoted from the current holdout."
    )
    if summary.empty:
        lines.extend(["", "No regime-threshold policy rows were produced."])
        return "\n".join(lines)
    display_cols = [
        "candidate",
        "fold_scope",
        "test_f1_mean",
        "test_pred_long_rate_mean",
        "official_f1_mean",
        "f1_delta_vs_official_mean",
        "policy_passed_fold_rate",
        "positive_forward_return_fold_rate",
        "reviewable",
        "reject_reason",
        "next_action",
    ]
    visible = summary[[column for column in display_cols if column in summary.columns]].copy()
    lines.append("")
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    return "\n".join(lines)


def _write_regime_threshold_policy(path: Path, by_fold: pd.DataFrame, summary: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    by_fold.to_csv(path / "regime_threshold_policy_by_fold.csv", index=False)
    summary.to_csv(path / "regime_threshold_policy_summary.csv", index=False)
    (path / "regime_threshold_policy.md").write_text(_regime_threshold_policy_markdown(summary), encoding="utf-8")
    _write_json(
        path / "regime_threshold_policy.json",
        {
            "regime_threshold_policy_by_fold": by_fold.to_dict(orient="records"),
            "regime_threshold_policy_summary": summary.to_dict(orient="records"),
        },
    )


def _regime_stability_frames(
    entries: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    forensics_columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "fold",
        "regime",
        "count",
        "row_share",
        "fold_rank_ic",
        "fold_rank_ic_bucket",
        "regime_rank_ic",
        "regime_label_long_rate",
        "regime_prob_long_mean",
        "regime_score_gap",
        "regime_forward_return_mean",
        "official_f1_in_regime",
        "official_pred_long_rate_in_regime",
    ]
    summary_columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "regime",
        "fold_count",
        "row_share_mean",
        "row_share_negative_fold_mean",
        "row_share_positive_fold_mean",
        "row_share_gap_negative_minus_positive",
        "regime_rank_ic_mean",
        "regime_rank_ic_std",
        "regime_negative_ic_fraction",
        "fold_rank_ic_when_regime_present_mean",
        "official_f1_in_regime_mean",
        "official_pred_long_rate_in_regime_mean",
        "suspect_score",
        "likely_issue",
        "recommended_action",
    ]
    rows: list[dict[str, Any]] = []
    bad_ic = float(_cfg(config, ["validation", "bad_fold_ic_threshold"], -0.08))
    target_ic = float(_cfg(config, ["validation", "target_rank_ic"], 0.03))
    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if not _is_stability_scope(fold_scope):
            continue
        predictions = entry.get("predictions")
        if not isinstance(predictions, pd.DataFrame) or predictions.empty:
            continue
        frame = _with_dominant_regime(predictions)
        if frame.empty or not {"fold", "split", "label", "prob_long", "forward_return"}.issubset(frame.columns):
            continue
        test_frame = frame.loc[frame["split"].astype(str) == "test"].copy()
        if test_frame.empty:
            continue
        diagnostics = entry.get("diagnostics", {}) or {}
        fold_metrics = diagnostics.get("fold_metrics")
        fold_rank = (
            {int(row["fold"]): _float(row.to_dict(), "rank_ic") for _, row in fold_metrics.dropna(subset=["fold"]).iterrows()}
            if isinstance(fold_metrics, pd.DataFrame) and not fold_metrics.empty and "fold" in fold_metrics.columns
            else {}
        )
        threshold_metrics = _entry_threshold_policy_frame(entry)
        threshold_by_fold = (
            {int(row["fold"]): row.to_dict() for _, row in threshold_metrics.dropna(subset=["fold"]).iterrows()}
            if threshold_metrics is not None and not threshold_metrics.empty and "fold" in threshold_metrics.columns
            else {}
        )
        candidate = str(entry.get("profile", ""))
        candidate_type = _diagnostic_candidate_type(fold_scope)
        for fold, fold_part in test_frame.groupby("fold"):
            fold_id = int(fold)
            fold_total = int(len(fold_part))
            fold_ic = float(fold_rank.get(fold_id, _rank_ic_for_frame(fold_part)))
            if np.isfinite(fold_ic) and fold_ic <= bad_ic:
                bucket = "bad_fold"
            elif np.isfinite(fold_ic) and fold_ic < 0:
                bucket = "negative_fold"
            elif np.isfinite(fold_ic) and fold_ic < target_ic:
                bucket = "below_target_fold"
            else:
                bucket = "positive_fold"
            threshold_row = threshold_by_fold.get(fold_id, {})
            official_threshold = _float(
                threshold_row,
                "official_threshold",
                _float(threshold_row, "constrained_threshold", _float(threshold_row, "selected_threshold", 0.5)),
            )
            if not np.isfinite(official_threshold):
                official_threshold = 0.5
            for regime, part in fold_part.groupby("dominant_regime"):
                labels = part["label"].astype(int)
                scores = pd.to_numeric(part["prob_long"], errors="coerce")
                pred = scores >= official_threshold
                metrics = _metrics_for_masked_predictions(part, pred)
                pos_scores = scores.loc[labels == 1]
                neg_scores = scores.loc[labels == 0]
                rows.append(
                    {
                        "candidate": candidate,
                        "candidate_type": candidate_type,
                        "fold_scope": fold_scope,
                        "fold": fold_id,
                        "regime": int(regime),
                        "count": int(len(part)),
                        "row_share": float(len(part) / fold_total) if fold_total else np.nan,
                        "fold_rank_ic": fold_ic,
                        "fold_rank_ic_bucket": bucket,
                        "regime_rank_ic": _rank_ic_for_frame(part),
                        "regime_label_long_rate": float(labels.mean()) if len(labels) else np.nan,
                        "regime_prob_long_mean": float(scores.mean()) if scores.notna().any() else np.nan,
                        "regime_score_gap": (
                            float(pos_scores.mean() - neg_scores.mean())
                            if not pos_scores.empty and not neg_scores.empty
                            else np.nan
                        ),
                        "regime_forward_return_mean": _numeric_mean(part, "forward_return"),
                        "official_f1_in_regime": metrics["f1"],
                        "official_pred_long_rate_in_regime": metrics["pred_long_rate"],
                    }
                )
    forensics = pd.DataFrame(rows, columns=forensics_columns) if rows else pd.DataFrame(columns=forensics_columns)
    if forensics.empty:
        return forensics, pd.DataFrame(columns=summary_columns)
    summary_rows: list[dict[str, Any]] = []
    for (candidate, candidate_type, fold_scope, regime), part in forensics.groupby(
        ["candidate", "candidate_type", "fold_scope", "regime"],
        dropna=False,
    ):
        rank = pd.to_numeric(part["regime_rank_ic"], errors="coerce")
        negative_part = part.loc[pd.to_numeric(part["fold_rank_ic"], errors="coerce") < 0.0]
        positive_part = part.loc[pd.to_numeric(part["fold_rank_ic"], errors="coerce") >= 0.0]
        row_share_negative = _numeric_mean(negative_part, "row_share")
        row_share_positive = _numeric_mean(positive_part, "row_share")
        share_gap = (
            row_share_negative - row_share_positive
            if np.isfinite(row_share_negative) and np.isfinite(row_share_positive)
            else np.nan
        )
        negative_ic_fraction = float((rank < 0.0).mean()) if rank.notna().any() else np.nan
        suspect = 0.0
        if np.isfinite(share_gap) and share_gap > 0:
            suspect += min(share_gap * 2.0, 1.0)
        if np.isfinite(negative_ic_fraction):
            suspect += negative_ic_fraction
        regime_ic_mean = float(rank.mean()) if rank.notna().any() else np.nan
        if np.isfinite(regime_ic_mean) and regime_ic_mean < 0:
            suspect += 0.5
        if np.isfinite(share_gap) and share_gap > 0.08 and np.isfinite(regime_ic_mean) and regime_ic_mean < target_ic:
            issue = "regime_overrepresented_in_negative_folds"
            action = "inspect_regime_specific_score_distribution_before_new_features"
        elif np.isfinite(regime_ic_mean) and regime_ic_mean < 0:
            issue = "regime_signal_reversal"
            action = "inspect_regime_features_and_threshold_transfer"
        elif np.isfinite(negative_ic_fraction) and negative_ic_fraction >= 0.50:
            issue = "regime_unstable_rank_ic"
            action = "monitor_regime_before_policy_changes"
        else:
            issue = "monitor"
            action = "no_regime_specific_change"
        summary_rows.append(
            {
                "candidate": candidate,
                "candidate_type": candidate_type,
                "fold_scope": fold_scope,
                "regime": int(regime),
                "fold_count": int(part["fold"].nunique()),
                "row_share_mean": _numeric_mean(part, "row_share"),
                "row_share_negative_fold_mean": row_share_negative,
                "row_share_positive_fold_mean": row_share_positive,
                "row_share_gap_negative_minus_positive": share_gap,
                "regime_rank_ic_mean": regime_ic_mean,
                "regime_rank_ic_std": float(rank.std(ddof=1)) if rank.notna().sum() > 1 else np.nan,
                "regime_negative_ic_fraction": negative_ic_fraction,
                "fold_rank_ic_when_regime_present_mean": _numeric_mean(part, "fold_rank_ic"),
                "official_f1_in_regime_mean": _numeric_mean(part, "official_f1_in_regime"),
                "official_pred_long_rate_in_regime_mean": _numeric_mean(part, "official_pred_long_rate_in_regime"),
                "suspect_score": float(suspect),
                "likely_issue": issue,
                "recommended_action": action,
            }
        )
    summary = (
        pd.DataFrame(summary_rows, columns=summary_columns)
        .sort_values(["candidate_type", "candidate", "suspect_score"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    return forensics, summary


def _regime_stability_markdown(summary: pd.DataFrame) -> str:
    lines = ["# Regime Stability Forensics", ""]
    lines.append(
        "This report checks whether HMM regimes are overrepresented in negative or unstable folds. "
        "It is diagnostic only; do not use it to change holdout policy without future-OOS confirmation."
    )
    if summary.empty:
        lines.extend(["", "No regime-stability rows were produced."])
        return "\n".join(lines)
    display_cols = [
        "candidate",
        "fold_scope",
        "regime",
        "row_share_gap_negative_minus_positive",
        "regime_rank_ic_mean",
        "regime_negative_ic_fraction",
        "official_f1_in_regime_mean",
        "suspect_score",
        "likely_issue",
        "recommended_action",
    ]
    visible = summary[[column for column in display_cols if column in summary.columns]].copy()
    lines.append("")
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    return "\n".join(lines)


def _write_regime_stability(path: Path, forensics: pd.DataFrame, summary: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    forensics.to_csv(path / "regime_stability_forensics.csv", index=False)
    summary.to_csv(path / "regime_stability_summary.csv", index=False)
    (path / "regime_stability.md").write_text(_regime_stability_markdown(summary), encoding="utf-8")
    _write_json(
        path / "regime_stability.json",
        {
            "regime_stability_forensics": forensics.to_dict(orient="records"),
            "regime_stability_summary": summary.to_dict(orient="records"),
        },
    )


def _ewma_last(values: list[float], *, alpha: float = 0.35) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return np.nan
    state = finite[0]
    for value in finite[1:]:
        state = alpha * value + (1.0 - alpha) * state
    return float(state)


def _threshold_transfer_review_frames(
    entries: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_fold_columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "policy_name",
        "policy_type",
        "fold",
        "threshold",
        "threshold_source",
        "threshold_history_count",
        "selection_guard",
        "validation_f1_at_policy_threshold",
        "validation_precision_at_policy_threshold",
        "validation_pred_long_rate_at_policy_threshold",
        "test_f1",
        "test_precision",
        "test_recall",
        "test_pred_long_rate",
        "test_actual_long_rate",
        "test_label_lift_vs_base",
        "test_mean_forward_return",
        "test_mean_tb_return",
    ]
    summary_columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "policy_name",
        "policy_type",
        "fold_count",
        "threshold_mean",
        "threshold_std",
        "threshold_history_mean",
        "validation_f1",
        "validation_precision",
        "validation_pred_long_rate",
        "test_f1",
        "test_precision",
        "test_recall",
        "test_pred_long_rate",
        "test_label_lift_vs_base",
        "test_mean_forward_return",
        "test_mean_tb_return",
        "positive_lift_fold_rate",
        "positive_forward_return_fold_rate",
        "f1_delta_vs_official",
        "pred_long_rate_delta_vs_official",
        "precision_delta_vs_official",
        "f1_passed",
        "precision_passed",
        "pred_long_rate_passed",
        "policy_passed_cv_test",
        "selection_guard",
        "recommended_action",
    ]
    min_long_f1 = float(_cfg(config, ["validation", "min_long_f1"], 0.45))
    threshold_cfg = _cfg(config, ["validation", "threshold_checks"], {}) or {}
    max_pred_long_rate = float(threshold_cfg.get("max_pred_long_rate", 0.70))
    min_precision = float(threshold_cfg.get("min_precision", 0.30))
    rows: list[dict[str, Any]] = []

    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if not _is_stability_scope(fold_scope):
            continue
        predictions = entry.get("predictions")
        if not isinstance(predictions, pd.DataFrame) or predictions.empty:
            continue
        if not {"fold", "label", "prob_long"}.issubset(predictions.columns):
            continue
        threshold_metrics = _entry_threshold_policy_frame(entry)
        if threshold_metrics is None or threshold_metrics.empty or "fold" not in threshold_metrics.columns:
            continue
        metric_by_fold = {
            int(row["fold"]): row.to_dict()
            for _, row in threshold_metrics.dropna(subset=["fold"]).iterrows()
        }
        sorted_folds = sorted(metric_by_fold)
        prior_constrained: list[float] = []
        prior_selected: list[float] = []
        candidate = str(entry.get("profile", ""))
        candidate_type = _diagnostic_candidate_type(fold_scope)

        for fold in sorted_folds:
            fold_part = predictions.loc[predictions["fold"].astype(int) == int(fold)].copy()
            if fold_part.empty:
                continue
            if "split" in fold_part.columns:
                validation = fold_part.loc[fold_part["split"].astype(str) == "val"].copy()
                test = fold_part.loc[fold_part["split"].astype(str) == "test"].copy()
            else:
                validation = fold_part.copy()
                test = fold_part.copy()
            if test.empty:
                continue
            metrics_row = metric_by_fold[fold]
            current_constrained = _float(metrics_row, "constrained_threshold")
            current_selected = _float(metrics_row, "selected_threshold")
            current_official = _float(metrics_row, "official_threshold", current_constrained)
            policy_thresholds: list[dict[str, Any]] = [
                {
                    "policy_name": "official_threshold_policy",
                    "policy_type": "current_fold_validation",
                    "threshold": current_official,
                    "threshold_source": str(metrics_row.get("official_threshold_source", "validation_constrained_threshold")),
                    "threshold_history_count": 0,
                    "selection_guard": "current_fold_validation_threshold_not_test",
                },
                {
                    "policy_name": "validation_constrained_threshold",
                    "policy_type": "current_fold_validation",
                    "threshold": current_constrained,
                    "threshold_source": "validation_constrained_threshold",
                    "threshold_history_count": 0,
                    "selection_guard": "current_fold_validation_threshold_not_test",
                },
                {
                    "policy_name": "validation_selected_threshold",
                    "policy_type": "current_fold_validation",
                    "threshold": current_selected,
                    "threshold_source": "validation_selected_threshold",
                    "threshold_history_count": 0,
                    "selection_guard": "current_fold_validation_threshold_not_test",
                },
            ]
            if prior_constrained:
                policy_thresholds.extend(
                    [
                        {
                            "policy_name": "past_median_constrained_threshold",
                            "policy_type": "causal_threshold_transfer",
                            "threshold": float(np.median(prior_constrained)),
                            "threshold_source": "historical_validation_constrained_thresholds",
                            "threshold_history_count": len(prior_constrained),
                            "selection_guard": "past_validation_thresholds_only_first_fold_skipped",
                        },
                        {
                            "policy_name": "past_mean_constrained_threshold",
                            "policy_type": "causal_threshold_transfer",
                            "threshold": float(np.mean(prior_constrained)),
                            "threshold_source": "historical_validation_constrained_thresholds",
                            "threshold_history_count": len(prior_constrained),
                            "selection_guard": "past_validation_thresholds_only_first_fold_skipped",
                        },
                        {
                            "policy_name": "past_ewma_constrained_threshold",
                            "policy_type": "causal_threshold_transfer",
                            "threshold": _ewma_last(prior_constrained),
                            "threshold_source": "historical_validation_constrained_thresholds",
                            "threshold_history_count": len(prior_constrained),
                            "selection_guard": "past_validation_thresholds_only_first_fold_skipped",
                        },
                        {
                            "policy_name": "previous_fold_constrained_threshold",
                            "policy_type": "causal_threshold_transfer",
                            "threshold": float(prior_constrained[-1]),
                            "threshold_source": "previous_validation_constrained_threshold",
                            "threshold_history_count": 1,
                            "selection_guard": "previous_fold_validation_threshold_only",
                        },
                    ]
                )
            if prior_selected:
                policy_thresholds.append(
                    {
                        "policy_name": "past_median_selected_threshold",
                        "policy_type": "causal_threshold_transfer",
                        "threshold": float(np.median(prior_selected)),
                        "threshold_source": "historical_validation_selected_thresholds",
                        "threshold_history_count": len(prior_selected),
                        "selection_guard": "past_validation_thresholds_only_first_fold_skipped",
                    }
                )

            for policy in policy_thresholds:
                threshold = float(policy["threshold"])
                if not np.isfinite(threshold):
                    continue
                threshold = float(np.clip(threshold, 0.0, 1.0))
                test_metrics = _threshold_metrics_at_value(test["label"], test["prob_long"], threshold)
                selection_stats = _threshold_selection_stats(test, threshold)
                if validation.empty:
                    validation_metrics = {"f1": np.nan, "precision": np.nan, "pred_long_rate": np.nan}
                else:
                    validation_metrics = _threshold_metrics_at_value(
                        validation["label"],
                        validation["prob_long"],
                        threshold,
                    )
                rows.append(
                    {
                        "candidate": candidate,
                        "candidate_type": candidate_type,
                        "fold_scope": fold_scope,
                        "policy_name": policy["policy_name"],
                        "policy_type": policy["policy_type"],
                        "fold": int(fold),
                        "threshold": threshold,
                        "threshold_source": policy["threshold_source"],
                        "threshold_history_count": int(policy["threshold_history_count"]),
                        "selection_guard": policy["selection_guard"],
                        "validation_f1_at_policy_threshold": validation_metrics["f1"],
                        "validation_precision_at_policy_threshold": validation_metrics["precision"],
                        "validation_pred_long_rate_at_policy_threshold": validation_metrics["pred_long_rate"],
                        "test_f1": test_metrics["f1"],
                        "test_precision": test_metrics["precision"],
                        "test_recall": test_metrics["recall"],
                        "test_pred_long_rate": test_metrics["pred_long_rate"],
                        "test_actual_long_rate": selection_stats["actual_long_rate"],
                        "test_label_lift_vs_base": selection_stats["label_lift_vs_base"],
                        "test_mean_forward_return": selection_stats["mean_forward_return"],
                        "test_mean_tb_return": selection_stats["mean_tb_return"],
                    }
                )
            if np.isfinite(current_constrained):
                prior_constrained.append(float(current_constrained))
            if np.isfinite(current_selected):
                prior_selected.append(float(current_selected))

    if not rows:
        return pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=by_fold_columns)

    by_fold = (
        pd.DataFrame(rows, columns=by_fold_columns)
        .sort_values(["candidate_type", "candidate", "fold_scope", "policy_name", "fold"])
        .reset_index(drop=True)
    )
    summaries: list[dict[str, Any]] = []
    group_cols = ["candidate", "candidate_type", "fold_scope", "policy_name", "policy_type"]
    for keys, part in by_fold.groupby(group_cols, dropna=False):
        candidate, candidate_type, fold_scope, policy_name, policy_type = keys
        official = by_fold.loc[
            (by_fold["candidate"].astype(str) == str(candidate))
            & (by_fold["fold_scope"].astype(str) == str(fold_scope))
            & (by_fold["policy_name"].astype(str) == "official_threshold_policy")
        ]
        official_f1 = float(pd.to_numeric(official["test_f1"], errors="coerce").mean()) if not official.empty else np.nan
        official_rate = (
            float(pd.to_numeric(official["test_pred_long_rate"], errors="coerce").mean())
            if not official.empty
            else np.nan
        )
        official_precision = (
            float(pd.to_numeric(official["test_precision"], errors="coerce").mean())
            if not official.empty
            else np.nan
        )
        test_f1 = float(pd.to_numeric(part["test_f1"], errors="coerce").mean())
        test_precision = float(pd.to_numeric(part["test_precision"], errors="coerce").mean())
        test_pred_rate = float(pd.to_numeric(part["test_pred_long_rate"], errors="coerce").mean())
        f1_passed = bool(np.isfinite(test_f1) and test_f1 > min_long_f1)
        precision_passed = bool(np.isfinite(test_precision) and test_precision >= min_precision)
        rate_passed = bool(np.isfinite(test_pred_rate) and test_pred_rate <= max_pred_long_rate)
        policy_passed = bool(f1_passed and precision_passed and rate_passed)
        f1_delta = test_f1 - official_f1 if np.isfinite(official_f1) else np.nan
        rate_delta = test_pred_rate - official_rate if np.isfinite(official_rate) else np.nan
        precision_delta = test_precision - official_precision if np.isfinite(official_precision) else np.nan
        if policy_passed and np.isfinite(f1_delta) and f1_delta >= 0.005:
            action = "pre_register_for_future_oos_threshold_policy"
        elif not rate_passed:
            action = "reject_threshold_too_broad"
        elif not f1_passed:
            action = "score_separation_gap_not_threshold_only"
        elif np.isfinite(f1_delta) and f1_delta < -0.005:
            action = "reject_weaker_than_official"
        else:
            action = "monitor_no_clear_advantage"
        summaries.append(
            {
                "candidate": candidate,
                "candidate_type": candidate_type,
                "fold_scope": fold_scope,
                "policy_name": policy_name,
                "policy_type": policy_type,
                "fold_count": int(part["fold"].nunique()),
                "threshold_mean": float(pd.to_numeric(part["threshold"], errors="coerce").mean()),
                "threshold_std": float(pd.to_numeric(part["threshold"], errors="coerce").std(ddof=0)),
                "threshold_history_mean": float(pd.to_numeric(part["threshold_history_count"], errors="coerce").mean()),
                "validation_f1": float(pd.to_numeric(part["validation_f1_at_policy_threshold"], errors="coerce").mean()),
                "validation_precision": float(pd.to_numeric(part["validation_precision_at_policy_threshold"], errors="coerce").mean()),
                "validation_pred_long_rate": float(
                    pd.to_numeric(part["validation_pred_long_rate_at_policy_threshold"], errors="coerce").mean()
                ),
                "test_f1": test_f1,
                "test_precision": test_precision,
                "test_recall": float(pd.to_numeric(part["test_recall"], errors="coerce").mean()),
                "test_pred_long_rate": test_pred_rate,
                "test_label_lift_vs_base": float(pd.to_numeric(part["test_label_lift_vs_base"], errors="coerce").mean()),
                "test_mean_forward_return": float(pd.to_numeric(part["test_mean_forward_return"], errors="coerce").mean()),
                "test_mean_tb_return": float(pd.to_numeric(part["test_mean_tb_return"], errors="coerce").mean()),
                "positive_lift_fold_rate": float((pd.to_numeric(part["test_label_lift_vs_base"], errors="coerce") > 1.0).mean()),
                "positive_forward_return_fold_rate": float(
                    (pd.to_numeric(part["test_mean_forward_return"], errors="coerce") > 0.0).mean()
                ),
                "f1_delta_vs_official": f1_delta,
                "pred_long_rate_delta_vs_official": rate_delta,
                "precision_delta_vs_official": precision_delta,
                "f1_passed": f1_passed,
                "precision_passed": precision_passed,
                "pred_long_rate_passed": rate_passed,
                "policy_passed_cv_test": policy_passed,
                "selection_guard": ";".join(sorted({str(value) for value in part["selection_guard"].dropna()})),
                "recommended_action": action,
            }
        )
    summary = (
        pd.DataFrame(summaries, columns=summary_columns)
        .sort_values(
            ["candidate_type", "candidate", "fold_scope", "policy_passed_cv_test", "test_f1", "test_pred_long_rate"],
            ascending=[True, True, True, False, False, True],
        )
        .reset_index(drop=True)
    )
    return summary, by_fold


def _threshold_transfer_review_markdown(summary: pd.DataFrame, by_fold: pd.DataFrame) -> str:
    lines = ["# Threshold Transfer Review", ""]
    if summary.empty:
        lines.append("No threshold transfer rows were produced.")
        return "\n".join(lines)
    lines.append(
        "This report compares current validation thresholds with causal threshold-transfer policies "
        "built only from prior validation folds. It is diagnostic only and must not override Phase 1 gates."
    )
    lines.append("")
    display_cols = [
        "candidate",
        "fold_scope",
        "policy_name",
        "fold_count",
        "test_f1",
        "test_precision",
        "test_pred_long_rate",
        "f1_delta_vs_official",
        "test_label_lift_vs_base",
        "positive_lift_fold_rate",
        "policy_passed_cv_test",
        "recommended_action",
    ]
    visible = summary[[column for column in display_cols if column in summary.columns]].copy()
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    if not by_fold.empty:
        lines.append("")
        lines.append(f"By-fold rows: {len(by_fold)}. See `threshold_transfer_by_fold.csv` for fold-level evidence.")
    return "\n".join(lines)


def _write_threshold_transfer_review(path: Path, summary: pd.DataFrame, by_fold: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path / "threshold_transfer_review.csv", index=False)
    by_fold.to_csv(path / "threshold_transfer_by_fold.csv", index=False)
    (path / "threshold_transfer_review.md").write_text(
        _threshold_transfer_review_markdown(summary, by_fold),
        encoding="utf-8",
    )
    _write_json(
        path / "threshold_transfer_review.json",
        {
            "summary": summary.to_dict(orient="records"),
            "by_fold": by_fold.to_dict(orient="records"),
        },
    )


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


def _score_quantile(part: pd.DataFrame, label_value: int, quantile: float) -> float:
    values = pd.to_numeric(part.loc[part["label"].astype(int) == int(label_value), "prob_long"], errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.quantile(float(quantile))) if not values.empty else np.nan


def _score_separation_forensics_frame(entries: list[dict[str, Any]], config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "fold",
        "start",
        "end",
        "count",
        "label_long_rate",
        "rank_ic",
        "rank_ic_bucket",
        "score_gap_pos_minus_neg",
        "score_gap_z",
        "score_ks",
        "prob_long_pos_mean",
        "prob_long_neg_mean",
        "prob_long_pos_p25",
        "prob_long_pos_p50",
        "prob_long_pos_p75",
        "prob_long_neg_p25",
        "prob_long_neg_p50",
        "prob_long_neg_p75",
        "prob_long_std",
        "prob_long_iqr",
        "official_threshold",
        "official_f1",
        "official_precision",
        "official_recall",
        "official_pred_long_rate",
        "selected_threshold",
        "selected_f1",
        "selected_pred_long_rate",
        "constrained_threshold",
        "constrained_f1",
        "constrained_pred_long_rate",
        "top_10_lift_vs_base",
        "top_10_forward_return",
        "primary_issue",
    ]
    target_rank_ic = float(_cfg(config, ["validation", "target_rank_ic"], 0.03))
    min_long_f1 = float(_cfg(config, ["validation", "min_long_f1"], 0.45))
    rows: list[dict[str, Any]] = []
    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if not _is_stability_scope(fold_scope):
            continue
        predictions = entry.get("predictions")
        if not isinstance(predictions, pd.DataFrame) or predictions.empty:
            continue
        if not {"fold", "label", "prob_long"}.issubset(predictions.columns):
            continue
        test_predictions = _test_predictions(predictions)
        if test_predictions.empty:
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
        if (
            isinstance(score_bands, pd.DataFrame)
            and not score_bands.empty
            and {"fold", "band"}.issubset(score_bands.columns)
        ):
            top10 = score_bands.loc[score_bands["band"].astype(str) == "top_10"].copy()
            top10_by_id = {int(row["fold"]): row.to_dict() for _, row in top10.dropna(subset=["fold"]).iterrows()}

        candidate = str(entry.get("profile", ""))
        candidate_type = _diagnostic_candidate_type(fold_scope)
        for fold, part in test_predictions.groupby("fold"):
            fold_id = int(fold)
            part = part.replace([np.inf, -np.inf], np.nan).dropna(subset=["label", "prob_long"]).copy()
            if part.empty:
                continue
            labels = part["label"].astype(int)
            scores = pd.to_numeric(part["prob_long"], errors="coerce")
            pos_scores = scores.loc[labels == 1]
            neg_scores = scores.loc[labels == 0]
            pos_mean = float(pos_scores.mean()) if not pos_scores.empty else np.nan
            neg_mean = float(neg_scores.mean()) if not neg_scores.empty else np.nan
            score_std = float(scores.std(ddof=0)) if scores.notna().any() else np.nan
            score_gap = pos_mean - neg_mean if np.isfinite(pos_mean) and np.isfinite(neg_mean) else np.nan
            rank_row = fold_by_id.get(fold_id, {})
            threshold_row = threshold_by_id.get(fold_id, {})
            top10_row = top10_by_id.get(fold_id, {})
            rank_ic_value = _float(rank_row, "rank_ic")
            official_f1 = _float(threshold_row, "test_f1_at_official_threshold", _float(threshold_row, "test_f1_at_constrained_threshold"))
            top10_lift = _float(top10_row, "lift_vs_base")
            top10_return = _float(top10_row, "mean_forward_return")
            if np.isfinite(rank_ic_value) and rank_ic_value < 0:
                issue = "negative_rank_ic"
            elif np.isfinite(score_gap) and score_gap <= 0:
                issue = "score_reversal"
            elif np.isfinite(official_f1) and official_f1 < min_long_f1:
                issue = "official_f1_gap"
            elif np.isfinite(top10_lift) and top10_lift < 1.0:
                issue = "top_10_label_lift_gap"
            elif np.isfinite(top10_return) and top10_return <= 0.0:
                issue = "top_10_payoff_gap"
            else:
                issue = "ok"
            if np.isfinite(rank_ic_value) and rank_ic_value < 0:
                bucket = "negative_rank_ic"
            elif np.isfinite(rank_ic_value) and rank_ic_value < target_rank_ic:
                bucket = "below_target_rank_ic"
            else:
                bucket = "rank_ic_ok"
            rows.append(
                {
                    "candidate": candidate,
                    "candidate_type": candidate_type,
                    "fold_scope": fold_scope,
                    "fold": fold_id,
                    "start": str(part["timestamp"].min()) if "timestamp" in part.columns else str(rank_row.get("start", "")),
                    "end": str(part["timestamp"].max()) if "timestamp" in part.columns else str(rank_row.get("end", "")),
                    "count": int(len(part)),
                    "label_long_rate": float(labels.mean()) if len(labels) else np.nan,
                    "rank_ic": rank_ic_value,
                    "rank_ic_bucket": bucket,
                    "score_gap_pos_minus_neg": score_gap,
                    "score_gap_z": float(score_gap / score_std) if np.isfinite(score_gap) and np.isfinite(score_std) and score_std > 0 else np.nan,
                    "score_ks": _score_ks_statistic(pos_scores, neg_scores),
                    "prob_long_pos_mean": pos_mean,
                    "prob_long_neg_mean": neg_mean,
                    "prob_long_pos_p25": _score_quantile(part, 1, 0.25),
                    "prob_long_pos_p50": _score_quantile(part, 1, 0.50),
                    "prob_long_pos_p75": _score_quantile(part, 1, 0.75),
                    "prob_long_neg_p25": _score_quantile(part, 0, 0.25),
                    "prob_long_neg_p50": _score_quantile(part, 0, 0.50),
                    "prob_long_neg_p75": _score_quantile(part, 0, 0.75),
                    "prob_long_std": score_std,
                    "prob_long_iqr": float(scores.quantile(0.75) - scores.quantile(0.25)) if scores.notna().any() else np.nan,
                    "official_threshold": _float(threshold_row, "official_threshold"),
                    "official_f1": official_f1,
                    "official_precision": _float(threshold_row, "test_precision_at_official_threshold"),
                    "official_recall": _float(threshold_row, "test_recall_at_official_threshold"),
                    "official_pred_long_rate": _float(threshold_row, "test_pred_long_rate_at_official_threshold"),
                    "selected_threshold": _float(threshold_row, "selected_threshold"),
                    "selected_f1": _float(threshold_row, "test_f1_at_selected_threshold"),
                    "selected_pred_long_rate": _float(threshold_row, "test_pred_long_rate_at_selected_threshold"),
                    "constrained_threshold": _float(threshold_row, "constrained_threshold"),
                    "constrained_f1": _float(threshold_row, "test_f1_at_constrained_threshold"),
                    "constrained_pred_long_rate": _float(threshold_row, "test_pred_long_rate_at_constrained_threshold"),
                    "top_10_lift_vs_base": top10_lift,
                    "top_10_forward_return": top10_return,
                    "primary_issue": issue,
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["candidate_type", "candidate", "fold_scope", "rank_ic"], ascending=[True, True, True, True])
        .reset_index(drop=True)
    )


def _mean_for_mask(frame: pd.DataFrame, mask: pd.Series, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return np.nan
    values = pd.to_numeric(frame.loc[mask, column], errors="coerce")
    return float(values.mean()) if values.notna().any() else np.nan


def _bad_fold_signature_frame(score_forensics: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "bad_definition",
        "fold_count",
        "bad_fold_count",
        "good_fold_count",
        "bad_rank_ic_mean",
        "good_rank_ic_mean",
        "bad_score_gap_mean",
        "good_score_gap_mean",
        "score_gap_delta_bad_minus_good",
        "bad_score_ks_mean",
        "good_score_ks_mean",
        "bad_label_long_rate_mean",
        "good_label_long_rate_mean",
        "label_long_rate_delta_bad_minus_good",
        "bad_official_f1_mean",
        "good_official_f1_mean",
        "official_f1_delta_bad_minus_good",
        "bad_official_pred_long_rate_mean",
        "good_official_pred_long_rate_mean",
        "bad_top_10_lift_mean",
        "good_top_10_lift_mean",
        "bad_top_10_forward_return_mean",
        "good_top_10_forward_return_mean",
        "bad_primary_issue_counts",
        "likely_signature",
        "recommended_next_action",
    ]
    if score_forensics.empty:
        return pd.DataFrame(columns=columns)
    target_rank_ic = float(_cfg(config, ["validation", "target_rank_ic"], 0.03))
    rows: list[dict[str, Any]] = []
    for (candidate, candidate_type, fold_scope), part in score_forensics.groupby(
        ["candidate", "candidate_type", "fold_scope"],
        dropna=False,
    ):
        rank_values = pd.to_numeric(part["rank_ic"], errors="coerce")
        bad_mask = rank_values < 0.0
        if int(bad_mask.sum()) < 3:
            bad_mask = rank_values < target_rank_ic
            definition = f"rank_ic_below_target_{target_rank_ic:.3f}"
        else:
            definition = "negative_rank_ic"
        good_mask = rank_values >= target_rank_ic
        bad_count = int(bad_mask.sum())
        good_count = int(good_mask.sum())
        if bad_count == 0:
            continue
        bad_score_gap = _mean_for_mask(part, bad_mask, "score_gap_pos_minus_neg")
        good_score_gap = _mean_for_mask(part, good_mask, "score_gap_pos_minus_neg")
        score_gap_delta = bad_score_gap - good_score_gap if np.isfinite(bad_score_gap) and np.isfinite(good_score_gap) else np.nan
        bad_label_rate = _mean_for_mask(part, bad_mask, "label_long_rate")
        good_label_rate = _mean_for_mask(part, good_mask, "label_long_rate")
        label_delta = bad_label_rate - good_label_rate if np.isfinite(bad_label_rate) and np.isfinite(good_label_rate) else np.nan
        bad_f1 = _mean_for_mask(part, bad_mask, "official_f1")
        good_f1 = _mean_for_mask(part, good_mask, "official_f1")
        f1_delta = bad_f1 - good_f1 if np.isfinite(bad_f1) and np.isfinite(good_f1) else np.nan
        bad_top_lift = _mean_for_mask(part, bad_mask, "top_10_lift_vs_base")
        good_top_lift = _mean_for_mask(part, good_mask, "top_10_lift_vs_base")
        bad_top_return = _mean_for_mask(part, bad_mask, "top_10_forward_return")
        good_top_return = _mean_for_mask(part, good_mask, "top_10_forward_return")
        issue_counts = part.loc[bad_mask, "primary_issue"].astype(str).value_counts().to_dict()
        signatures: list[str] = []
        if np.isfinite(score_gap_delta) and score_gap_delta < -0.01:
            signatures.append("score_separation_compresses_or_reverses")
        if np.isfinite(bad_top_lift) and bad_top_lift < 1.0:
            signatures.append("top_score_label_lift_fails")
        if np.isfinite(bad_top_return) and bad_top_return <= 0.0:
            signatures.append("top_score_payoff_reverses")
        if np.isfinite(label_delta) and abs(label_delta) >= 0.05:
            signatures.append("label_distribution_shift")
        if np.isfinite(f1_delta) and f1_delta < -0.05:
            signatures.append("official_threshold_f1_collapses")
        if not signatures:
            signatures.append("rank_ic_variance_without_single_score_signature")
        if "score_separation_compresses_or_reverses" in signatures:
            action = "inspect_bad_fold_feature_drift_and_add_only_pre_registered_score_separation_features"
        elif "top_score_payoff_reverses" in signatures:
            action = "do_not_promote_score_band_policy_until_future_oos_confirms_payoff"
        elif "label_distribution_shift" in signatures:
            action = "review_label_regime_balance_before_feature_changes"
        elif "official_threshold_f1_collapses" in signatures:
            action = "focus_on_fold_specific_score_separation_not_threshold_smoothing"
        else:
            action = "use_fold_level_feature_importance_or_future_oos_before_new_profile_search"
        rows.append(
            {
                "candidate": candidate,
                "candidate_type": candidate_type,
                "fold_scope": fold_scope,
                "bad_definition": definition,
                "fold_count": int(part["fold"].nunique()),
                "bad_fold_count": bad_count,
                "good_fold_count": good_count,
                "bad_rank_ic_mean": _mean_for_mask(part, bad_mask, "rank_ic"),
                "good_rank_ic_mean": _mean_for_mask(part, good_mask, "rank_ic"),
                "bad_score_gap_mean": bad_score_gap,
                "good_score_gap_mean": good_score_gap,
                "score_gap_delta_bad_minus_good": score_gap_delta,
                "bad_score_ks_mean": _mean_for_mask(part, bad_mask, "score_ks"),
                "good_score_ks_mean": _mean_for_mask(part, good_mask, "score_ks"),
                "bad_label_long_rate_mean": bad_label_rate,
                "good_label_long_rate_mean": good_label_rate,
                "label_long_rate_delta_bad_minus_good": label_delta,
                "bad_official_f1_mean": bad_f1,
                "good_official_f1_mean": good_f1,
                "official_f1_delta_bad_minus_good": f1_delta,
                "bad_official_pred_long_rate_mean": _mean_for_mask(part, bad_mask, "official_pred_long_rate"),
                "good_official_pred_long_rate_mean": _mean_for_mask(part, good_mask, "official_pred_long_rate"),
                "bad_top_10_lift_mean": bad_top_lift,
                "good_top_10_lift_mean": good_top_lift,
                "bad_top_10_forward_return_mean": bad_top_return,
                "good_top_10_forward_return_mean": good_top_return,
                "bad_primary_issue_counts": json.dumps(issue_counts, sort_keys=True),
                "likely_signature": ";".join(signatures),
                "recommended_next_action": action,
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["candidate_type", "candidate", "bad_fold_count"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def _score_separation_markdown(score_forensics: pd.DataFrame, bad_signature: pd.DataFrame) -> str:
    lines = ["# Bad Fold Score-Separation Forensics", ""]
    if score_forensics.empty and bad_signature.empty:
        lines.append("No score-separation rows were produced.")
        return "\n".join(lines)
    lines.append(
        "These diagnostics explain whether weak folds are caused by score separation, label-rate shifts, "
        "threshold behavior, or top-score payoff reversal. They are diagnostics only."
    )
    if not bad_signature.empty:
        lines.append("")
        lines.append("## Bad Fold Signatures")
        display_cols = [
            "candidate",
            "fold_scope",
            "bad_fold_count",
            "bad_rank_ic_mean",
            "bad_score_gap_mean",
            "good_score_gap_mean",
            "bad_top_10_lift_mean",
            "bad_top_10_forward_return_mean",
            "likely_signature",
            "recommended_next_action",
        ]
        visible = bad_signature[[column for column in display_cols if column in bad_signature.columns]].copy()
        lines.append("| " + " | ".join(visible.columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
        for _, row in visible.iterrows():
            lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    if not score_forensics.empty:
        lines.append("")
        lines.append(f"Fold-level rows: {len(score_forensics)}. See `score_separation_forensics.csv` for detail.")
    return "\n".join(lines)


def _write_score_separation_forensics(
    path: Path,
    score_forensics: pd.DataFrame,
    bad_signature: pd.DataFrame,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    score_forensics.to_csv(path / "score_separation_forensics.csv", index=False)
    bad_signature.to_csv(path / "bad_fold_signature.csv", index=False)
    (path / "bad_fold_signature.md").write_text(
        _score_separation_markdown(score_forensics, bad_signature),
        encoding="utf-8",
    )
    _write_json(
        path / "bad_fold_signature.json",
        {
            "score_separation_forensics": score_forensics.to_dict(orient="records"),
            "bad_fold_signature": bad_signature.to_dict(orient="records"),
        },
    )


def _feature_family(feature: str) -> str:
    name = str(feature)
    timeframe = "4h" if name.startswith("4h_") else "1h"
    base = name[3:] if timeframe == "4h" else name
    if base.startswith("ih15_"):
        return "ih15_intrahour_order_flow"
    if base.startswith("fut_") or any(token in base for token in ("funding", "open_interest", "positioning")):
        return "futures_context"
    if "large_trade_pressure" in base or "signed_ltp" in base or base.startswith("ltp"):
        family = "large_trade_pressure"
    elif "cvd_pressure" in base:
        family = "cvd_pressure"
    elif any(token in base for token in ("taker", "imbalance", "cvd", "orderflow", "absorption", "pressure")):
        family = "order_flow"
    elif any(token in base for token in ("whale", "vpt", "vol_per_trade", "large_trade_ratio")):
        family = "whale_ticket_size"
    elif "volume" in base or "trade_share" in base:
        family = "volume_context"
    elif any(token in base for token in ("realized_vol", "gk_vol", "atr", "adx", "vwap", "return")):
        family = "volatility_structure"
    else:
        family = "other"
    return f"{timeframe}_{family}" if timeframe == "4h" else family


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    frame = pd.DataFrame({"left": left, "right": right}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 4:
        return np.nan
    if frame["left"].nunique(dropna=True) < 2 or frame["right"].nunique(dropna=True) < 2:
        return np.nan
    value = frame["left"].corr(frame["right"], method="spearman")
    return float(value) if np.isfinite(value) else np.nan


def _label_gap(frame: pd.DataFrame, feature: str) -> float:
    if frame.empty or feature not in frame.columns or "label" not in frame.columns:
        return np.nan
    values = pd.to_numeric(frame[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
    labels = pd.to_numeric(frame["label"], errors="coerce")
    pos = values.loc[labels == 1].dropna()
    neg = values.loc[labels == 0].dropna()
    if pos.empty or neg.empty:
        return np.nan
    return float(pos.mean() - neg.mean())


def _feature_drift_columns() -> list[str]:
    return [
        "candidate",
        "candidate_type",
        "fold_scope",
        "feature",
        "feature_family",
        "bad_definition",
        "bad_fold_count",
        "good_fold_count",
        "bad_fold_ids",
        "good_fold_ids",
        "bad_count",
        "good_count",
        "bad_mean",
        "good_mean",
        "drift_effect_size",
        "bad_std",
        "good_std",
        "bad_feature_return_ic",
        "good_feature_return_ic",
        "return_ic_delta_bad_minus_good",
        "bad_label_gap",
        "good_label_gap",
        "label_gap_delta_bad_minus_good",
        "return_ic_reversal",
        "label_gap_reversal",
        "distribution_drift_flag",
        "suspect_score",
        "likely_issue",
    ]


def _feature_drift_forensics_frame(
    entries: list[dict[str, Any]],
    score_forensics: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    columns = _feature_drift_columns()
    if score_forensics.empty:
        return pd.DataFrame(columns=columns)

    target_rank_ic = float(_cfg(config, ["validation", "target_rank_ic"], 0.03))
    rows: list[dict[str, Any]] = []
    score_groups = {
        (str(candidate), str(fold_scope)): part.copy()
        for (candidate, _candidate_type, fold_scope), part in score_forensics.groupby(
            ["candidate", "candidate_type", "fold_scope"],
            dropna=False,
        )
    }

    for entry in entries:
        candidate = str(entry.get("profile", ""))
        fold_scope = str(entry.get("fold_scope", ""))
        candidate_type = _diagnostic_candidate_type(fold_scope)
        if candidate_type != "profile" or fold_scope != "full":
            continue
        predictions = entry.get("predictions")
        if not isinstance(predictions, pd.DataFrame) or predictions.empty:
            continue
        feature_columns = [str(column) for column in entry.get("feature_columns", [])]
        usable_features = [
            column
            for column in feature_columns
            if column in predictions.columns and pd.api.types.is_numeric_dtype(predictions[column])
        ]
        if not usable_features:
            continue
        test_predictions = _test_predictions(predictions)
        required = {"fold", "label", "forward_return"}
        if test_predictions.empty or not required.issubset(test_predictions.columns):
            continue
        score_part = score_groups.get((candidate, fold_scope))
        if score_part is None or score_part.empty or "rank_ic" not in score_part.columns:
            continue
        rank_values = pd.to_numeric(score_part["rank_ic"], errors="coerce")
        bad_score = score_part.loc[rank_values < 0.0].copy()
        bad_definition = "negative_rank_ic"
        if bad_score.empty:
            bad_score = score_part.loc[rank_values < target_rank_ic].copy()
            bad_definition = f"rank_ic_below_target_{target_rank_ic:.3f}"
        good_score = score_part.loc[rank_values >= target_rank_ic].copy()
        bad_folds = sorted({int(fold) for fold in bad_score["fold"].dropna().astype(int).tolist()})
        good_folds = sorted({int(fold) for fold in good_score["fold"].dropna().astype(int).tolist()})
        if not bad_folds or not good_folds:
            continue
        frame = test_predictions.copy().replace([np.inf, -np.inf], np.nan)
        frame["fold"] = pd.to_numeric(frame["fold"], errors="coerce")
        frame["forward_return"] = pd.to_numeric(frame["forward_return"], errors="coerce")
        frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
        bad_frame = frame.loc[frame["fold"].isin(bad_folds)].copy()
        good_frame = frame.loc[frame["fold"].isin(good_folds)].copy()
        if bad_frame.empty or good_frame.empty:
            continue

        for feature in usable_features:
            bad_values = pd.to_numeric(bad_frame[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            good_values = pd.to_numeric(good_frame[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if bad_values.empty or good_values.empty:
                continue
            bad_mean = float(bad_values.mean())
            good_mean = float(good_values.mean())
            bad_std = float(bad_values.std(ddof=0))
            good_std = float(good_values.std(ddof=0))
            pooled_std = float(np.sqrt((bad_std**2 + good_std**2) / 2.0))
            drift_effect = float((bad_mean - good_mean) / pooled_std) if pooled_std > 0 else np.nan
            bad_return_ic = _safe_spearman(bad_frame[feature], bad_frame["forward_return"])
            good_return_ic = _safe_spearman(good_frame[feature], good_frame["forward_return"])
            return_ic_delta = (
                bad_return_ic - good_return_ic
                if np.isfinite(bad_return_ic) and np.isfinite(good_return_ic)
                else np.nan
            )
            bad_gap = _label_gap(bad_frame, feature)
            good_gap = _label_gap(good_frame, feature)
            label_gap_delta = bad_gap - good_gap if np.isfinite(bad_gap) and np.isfinite(good_gap) else np.nan
            return_reversal = bool(
                np.isfinite(bad_return_ic)
                and np.isfinite(good_return_ic)
                and bad_return_ic * good_return_ic < 0
                and abs(bad_return_ic) >= 0.02
                and abs(good_return_ic) >= 0.02
            )
            label_reversal = bool(
                np.isfinite(bad_gap)
                and np.isfinite(good_gap)
                and bad_gap * good_gap < 0
                and abs(bad_gap) >= 0.03
                and abs(good_gap) >= 0.03
            )
            distribution_drift = bool(np.isfinite(drift_effect) and abs(drift_effect) >= 0.50)
            suspect_score = 0.0
            if np.isfinite(drift_effect):
                suspect_score += min(abs(drift_effect), 3.0)
            if np.isfinite(return_ic_delta):
                suspect_score += min(abs(return_ic_delta), 2.0)
            if np.isfinite(label_gap_delta):
                suspect_score += min(abs(label_gap_delta), 2.0) * 0.5
            if return_reversal:
                suspect_score += 1.0
            if label_reversal:
                suspect_score += 0.75
            if distribution_drift:
                suspect_score += 0.5

            if return_reversal and label_reversal:
                issue = "feature_signal_and_label_gap_reversal"
            elif return_reversal:
                issue = "feature_return_ic_reversal"
            elif label_reversal:
                issue = "feature_label_gap_reversal"
            elif distribution_drift:
                issue = "bad_fold_distribution_drift"
            elif np.isfinite(return_ic_delta) and return_ic_delta < -0.08:
                issue = "bad_fold_feature_return_ic_degrades"
            else:
                issue = "monitor"
            rows.append(
                {
                    "candidate": candidate,
                    "candidate_type": candidate_type,
                    "fold_scope": fold_scope,
                    "feature": feature,
                    "feature_family": _feature_family(feature),
                    "bad_definition": bad_definition,
                    "bad_fold_count": len(bad_folds),
                    "good_fold_count": len(good_folds),
                    "bad_fold_ids": ",".join(str(fold) for fold in bad_folds),
                    "good_fold_ids": ",".join(str(fold) for fold in good_folds),
                    "bad_count": int(len(bad_values)),
                    "good_count": int(len(good_values)),
                    "bad_mean": bad_mean,
                    "good_mean": good_mean,
                    "drift_effect_size": drift_effect,
                    "bad_std": bad_std,
                    "good_std": good_std,
                    "bad_feature_return_ic": bad_return_ic,
                    "good_feature_return_ic": good_return_ic,
                    "return_ic_delta_bad_minus_good": return_ic_delta,
                    "bad_label_gap": bad_gap,
                    "good_label_gap": good_gap,
                    "label_gap_delta_bad_minus_good": label_gap_delta,
                    "return_ic_reversal": return_reversal,
                    "label_gap_reversal": label_reversal,
                    "distribution_drift_flag": distribution_drift,
                    "suspect_score": float(suspect_score),
                    "likely_issue": issue,
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["candidate", "fold_scope", "suspect_score"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def _feature_family_drift_summary_frame(feature_drift: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "feature_family",
        "feature_count",
        "top_suspect_feature",
        "top_suspect_score",
        "mean_abs_drift_effect",
        "mean_bad_return_ic",
        "mean_good_return_ic",
        "return_ic_reversal_count",
        "label_gap_reversal_count",
        "distribution_drift_count",
        "top_likely_issue",
        "recommended_next_action",
    ]
    if feature_drift.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (candidate, candidate_type, fold_scope, family), part in feature_drift.groupby(
        ["candidate", "candidate_type", "fold_scope", "feature_family"],
        dropna=False,
    ):
        sorted_part = part.sort_values("suspect_score", ascending=False)
        top = sorted_part.iloc[0].to_dict()
        reversal_count = int(part["return_ic_reversal"].astype(bool).sum())
        label_reversal_count = int(part["label_gap_reversal"].astype(bool).sum())
        drift_count = int(part["distribution_drift_flag"].astype(bool).sum())
        if reversal_count > 0 or label_reversal_count > 0:
            action = "inspect_or_ablate_family_in_pre_registered_future_oos_candidate"
        elif drift_count > 0:
            action = "prefer_stable_bounded_transforms_or_family_ablation_hypothesis"
        else:
            action = "monitor"
        rows.append(
            {
                "candidate": candidate,
                "candidate_type": candidate_type,
                "fold_scope": fold_scope,
                "feature_family": family,
                "feature_count": int(part["feature"].nunique()),
                "top_suspect_feature": str(top.get("feature", "")),
                "top_suspect_score": _float(top, "suspect_score"),
                "mean_abs_drift_effect": float(pd.to_numeric(part["drift_effect_size"], errors="coerce").abs().mean()),
                "mean_bad_return_ic": float(pd.to_numeric(part["bad_feature_return_ic"], errors="coerce").mean()),
                "mean_good_return_ic": float(pd.to_numeric(part["good_feature_return_ic"], errors="coerce").mean()),
                "return_ic_reversal_count": reversal_count,
                "label_gap_reversal_count": label_reversal_count,
                "distribution_drift_count": drift_count,
                "top_likely_issue": str(top.get("likely_issue", "")),
                "recommended_next_action": action,
            }
        )
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(
            ["candidate", "top_suspect_score", "return_ic_reversal_count", "label_gap_reversal_count"],
            ascending=[True, False, False, False],
        )
        .reset_index(drop=True)
    )


def _feature_drift_markdown(detail: pd.DataFrame, summary: pd.DataFrame) -> str:
    lines = ["# Bad Fold Feature Drift Forensics", ""]
    if detail.empty and summary.empty:
        lines.append("No feature drift rows were produced.")
        return "\n".join(lines)
    lines.append(
        "These diagnostics compare bad folds against good folds using existing OOS predictions. "
        "They identify distribution drift and feature/return or feature/label signal reversals. "
        "They are diagnostic evidence, not automatic feature-selection output."
    )
    if not summary.empty:
        lines.append("")
        lines.append("## Feature Family Summary")
        display_cols = [
            "candidate",
            "fold_scope",
            "feature_family",
            "feature_count",
            "top_suspect_feature",
            "top_suspect_score",
            "return_ic_reversal_count",
            "label_gap_reversal_count",
            "distribution_drift_count",
            "recommended_next_action",
        ]
        visible = summary[[column for column in display_cols if column in summary.columns]].copy()
        lines.append("| " + " | ".join(visible.columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
        for _, row in visible.iterrows():
            lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    if not detail.empty:
        lines.append("")
        lines.append(f"Feature-level rows: {len(detail)}. See `feature_drift_forensics.csv` for detail.")
    return "\n".join(lines)


def _write_feature_drift_forensics(path: Path, detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    detail.to_csv(path / "feature_drift_forensics.csv", index=False)
    summary.to_csv(path / "feature_family_drift_summary.csv", index=False)
    (path / "feature_drift_forensics.md").write_text(
        _feature_drift_markdown(detail, summary),
        encoding="utf-8",
    )
    _write_json(
        path / "feature_drift_forensics.json",
        {
            "feature_drift_forensics": detail.to_dict(orient="records"),
            "feature_family_drift_summary": summary.to_dict(orient="records"),
        },
    )


def _clean_probability_inputs(labels: pd.Series, scores: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "label": pd.to_numeric(labels, errors="coerce"),
            "score": pd.to_numeric(scores, errors="coerce"),
        }
    )
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return frame
    frame["label"] = frame["label"].astype(int)
    frame["score"] = frame["score"].clip(1e-6, 1.0 - 1e-6)
    return frame


def _calibration_error(labels: pd.Series, scores: pd.Series, *, bins: int, strategy: str) -> tuple[float, float, int]:
    frame = _clean_probability_inputs(labels, scores)
    if frame.empty:
        return np.nan, np.nan, 0
    bins = max(2, int(bins))
    if strategy == "equal_count":
        try:
            frame["bin"] = pd.qcut(frame["score"].rank(method="first"), q=min(bins, len(frame)), labels=False, duplicates="drop")
        except ValueError:
            return np.nan, np.nan, 0
    else:
        edges = np.linspace(0.0, 1.0, bins + 1)
        frame["bin"] = pd.cut(frame["score"], bins=edges, labels=False, include_lowest=True)
    frame = frame.dropna(subset=["bin"]).copy()
    if frame.empty:
        return np.nan, np.nan, 0
    total = float(len(frame))
    weighted_error = 0.0
    max_error = 0.0
    used_bins = 0
    for _, part in frame.groupby("bin"):
        if part.empty:
            continue
        predicted = float(part["score"].mean())
        actual = float(part["label"].mean())
        error = abs(actual - predicted)
        weighted_error += (len(part) / total) * error
        max_error = max(max_error, error)
        used_bins += 1
    return float(weighted_error), float(max_error), int(used_bins)


def _safe_average_precision(labels: pd.Series, scores: pd.Series) -> float:
    frame = _clean_probability_inputs(labels, scores)
    if frame.empty or frame["label"].nunique(dropna=True) < 2:
        return np.nan
    try:
        return float(average_precision_score(frame["label"], frame["score"]))
    except ValueError:
        return np.nan


def _binary_probability_metrics(labels: pd.Series, scores: pd.Series, *, bins: int) -> dict[str, float]:
    frame = _clean_probability_inputs(labels, scores)
    if frame.empty:
        return {
            "brier_score": np.nan,
            "log_loss": np.nan,
            "average_precision": np.nan,
            "ece_equal_width": np.nan,
            "mce_equal_width": np.nan,
            "ece_equal_count": np.nan,
            "mce_equal_count": np.nan,
            "score_entropy_mean": np.nan,
            "score_sharpness_mean": np.nan,
            "prob_long_mean": np.nan,
            "prob_long_std": np.nan,
            "prob_long_iqr": np.nan,
        }
    labels_array = frame["label"].astype(float)
    scores_array = frame["score"].astype(float)
    ece_width, mce_width, _ = _calibration_error(labels_array, scores_array, bins=bins, strategy="equal_width")
    ece_count, mce_count, _ = _calibration_error(labels_array, scores_array, bins=bins, strategy="equal_count")
    entropy = -(scores_array * np.log(scores_array) + (1.0 - scores_array) * np.log(1.0 - scores_array))
    return {
        "brier_score": float(np.mean((scores_array - labels_array) ** 2)),
        "log_loss": float(-np.mean(labels_array * np.log(scores_array) + (1.0 - labels_array) * np.log(1.0 - scores_array))),
        "average_precision": _safe_average_precision(labels_array, scores_array),
        "ece_equal_width": ece_width,
        "mce_equal_width": mce_width,
        "ece_equal_count": ece_count,
        "mce_equal_count": mce_count,
        "score_entropy_mean": float(entropy.mean()),
        "score_sharpness_mean": float((scores_array * (1.0 - scores_array)).mean()),
        "prob_long_mean": float(scores_array.mean()),
        "prob_long_std": float(scores_array.std(ddof=0)),
        "prob_long_iqr": float(scores_array.quantile(0.75) - scores_array.quantile(0.25)),
    }


def _probability_quality_forensics_frame(entries: list[dict[str, Any]], config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "fold",
        "start",
        "end",
        "count",
        "label_long_rate",
        "rank_ic",
        "rank_ic_bucket",
        "brier_score",
        "log_loss",
        "average_precision",
        "ece_equal_width",
        "mce_equal_width",
        "ece_equal_count",
        "mce_equal_count",
        "score_entropy_mean",
        "score_sharpness_mean",
        "prob_long_mean",
        "prob_long_std",
        "prob_long_iqr",
        "official_threshold",
        "official_f1",
        "official_pred_long_rate",
        "primary_issue",
    ]
    rows: list[dict[str, Any]] = []
    target_rank_ic = float(_cfg(config, ["validation", "target_rank_ic"], 0.03))
    min_long_f1 = float(_cfg(config, ["validation", "min_long_f1"], 0.45))
    max_pred_rate = float(_cfg(config, ["validation", "threshold_checks", "max_pred_long_rate"], 0.70))
    bins = int(_cfg(config, ["validation", "calibration_bins"], 10))
    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if not _is_stability_scope(fold_scope):
            continue
        predictions = entry.get("predictions")
        if not isinstance(predictions, pd.DataFrame) or predictions.empty:
            continue
        test_predictions = _test_predictions(predictions)
        if test_predictions.empty or not {"fold", "label", "prob_long"}.issubset(test_predictions.columns):
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
        candidate = str(entry.get("profile", ""))
        candidate_type = _diagnostic_candidate_type(fold_scope)
        for fold, part in test_predictions.groupby("fold"):
            fold_id = int(fold)
            part = part.replace([np.inf, -np.inf], np.nan).dropna(subset=["label", "prob_long"]).copy()
            if part.empty:
                continue
            rank_row = fold_by_id.get(fold_id, {})
            threshold_row = threshold_by_id.get(fold_id, {})
            rank_ic_value = _float(rank_row, "rank_ic")
            metrics = _binary_probability_metrics(part["label"], part["prob_long"], bins=bins)
            official_f1 = _float(threshold_row, "test_f1_at_official_threshold", _float(threshold_row, "test_f1_at_constrained_threshold"))
            official_rate = _float(
                threshold_row,
                "test_pred_long_rate_at_official_threshold",
                _float(threshold_row, "test_pred_long_rate_at_constrained_threshold"),
            )
            if np.isfinite(rank_ic_value) and rank_ic_value < 0.0:
                issue = "negative_rank_ic"
            elif np.isfinite(metrics["ece_equal_count"]) and metrics["ece_equal_count"] > 0.10:
                issue = "calibration_error"
            elif np.isfinite(official_f1) and official_f1 < min_long_f1:
                issue = "official_f1_gap"
            elif np.isfinite(official_rate) and official_rate > max_pred_rate:
                issue = "pred_long_rate_guardrail"
            else:
                issue = "ok"
            bucket = (
                "negative_rank_ic"
                if np.isfinite(rank_ic_value) and rank_ic_value < 0.0
                else ("below_target_rank_ic" if np.isfinite(rank_ic_value) and rank_ic_value < target_rank_ic else "rank_ic_ok")
            )
            rows.append(
                {
                    "candidate": candidate,
                    "candidate_type": candidate_type,
                    "fold_scope": fold_scope,
                    "fold": fold_id,
                    "start": str(part["timestamp"].min()) if "timestamp" in part.columns else str(rank_row.get("start", "")),
                    "end": str(part["timestamp"].max()) if "timestamp" in part.columns else str(rank_row.get("end", "")),
                    "count": int(len(part)),
                    "label_long_rate": float(pd.to_numeric(part["label"], errors="coerce").mean()),
                    "rank_ic": rank_ic_value,
                    "rank_ic_bucket": bucket,
                    **metrics,
                    "official_threshold": _float(threshold_row, "official_threshold"),
                    "official_f1": official_f1,
                    "official_pred_long_rate": official_rate,
                    "primary_issue": issue,
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["candidate_type", "candidate", "fold_scope", "rank_ic"], ascending=[True, True, True, True])
        .reset_index(drop=True)
    )


def _probability_quality_summary_frame(probability_quality: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "fold_count",
        "bad_definition",
        "bad_fold_count",
        "good_fold_count",
        "mean_brier_score",
        "mean_log_loss",
        "mean_average_precision",
        "mean_ece_equal_count",
        "mean_score_entropy",
        "bad_brier_score_mean",
        "good_brier_score_mean",
        "bad_average_precision_mean",
        "good_average_precision_mean",
        "bad_ece_equal_count_mean",
        "good_ece_equal_count_mean",
        "bad_score_entropy_mean",
        "good_score_entropy_mean",
        "probability_quality_issue",
        "recommended_next_action",
    ]
    if probability_quality.empty:
        return pd.DataFrame(columns=columns)
    target_rank_ic = float(_cfg(config, ["validation", "target_rank_ic"], 0.03))
    rows: list[dict[str, Any]] = []
    for (candidate, candidate_type, fold_scope), part in probability_quality.groupby(
        ["candidate", "candidate_type", "fold_scope"],
        dropna=False,
    ):
        rank = pd.to_numeric(part["rank_ic"], errors="coerce")
        bad_mask = rank < 0.0
        bad_definition = "negative_rank_ic"
        if int(bad_mask.sum()) < 1:
            bad_mask = rank < target_rank_ic
            bad_definition = f"rank_ic_below_target_{target_rank_ic:.3f}"
        good_mask = rank >= target_rank_ic
        bad_count = int(bad_mask.sum())
        good_count = int(good_mask.sum())
        bad_brier = _mean_for_mask(part, bad_mask, "brier_score")
        good_brier = _mean_for_mask(part, good_mask, "brier_score")
        bad_ap = _mean_for_mask(part, bad_mask, "average_precision")
        good_ap = _mean_for_mask(part, good_mask, "average_precision")
        bad_ece = _mean_for_mask(part, bad_mask, "ece_equal_count")
        good_ece = _mean_for_mask(part, good_mask, "ece_equal_count")
        bad_entropy = _mean_for_mask(part, bad_mask, "score_entropy_mean")
        good_entropy = _mean_for_mask(part, good_mask, "score_entropy_mean")
        if np.isfinite(bad_ap) and np.isfinite(good_ap) and bad_ap < good_ap - 0.05:
            issue = "bad_folds_lose_ranking_resolution"
            action = "prioritize_score_separation_features_over_threshold_smoothing"
        elif np.isfinite(bad_ece) and np.isfinite(good_ece) and bad_ece > good_ece + 0.03:
            issue = "bad_folds_calibration_worsens"
            action = "review_calibration_by_fold_but_do_not_fit_on_test_or_holdout"
        elif np.isfinite(bad_entropy) and np.isfinite(good_entropy) and bad_entropy > good_entropy + 0.03:
            issue = "bad_folds_scores_become_uncertain"
            action = "inspect_feature_drift_and_score_distribution_shift"
        else:
            issue = "monitor"
            action = "monitor"
        rows.append(
            {
                "candidate": candidate,
                "candidate_type": candidate_type,
                "fold_scope": fold_scope,
                "fold_count": int(part["fold"].nunique()),
                "bad_definition": bad_definition,
                "bad_fold_count": bad_count,
                "good_fold_count": good_count,
                "mean_brier_score": float(pd.to_numeric(part["brier_score"], errors="coerce").mean()),
                "mean_log_loss": float(pd.to_numeric(part["log_loss"], errors="coerce").mean()),
                "mean_average_precision": float(pd.to_numeric(part["average_precision"], errors="coerce").mean()),
                "mean_ece_equal_count": float(pd.to_numeric(part["ece_equal_count"], errors="coerce").mean()),
                "mean_score_entropy": float(pd.to_numeric(part["score_entropy_mean"], errors="coerce").mean()),
                "bad_brier_score_mean": bad_brier,
                "good_brier_score_mean": good_brier,
                "bad_average_precision_mean": bad_ap,
                "good_average_precision_mean": good_ap,
                "bad_ece_equal_count_mean": bad_ece,
                "good_ece_equal_count_mean": good_ece,
                "bad_score_entropy_mean": bad_entropy,
                "good_score_entropy_mean": good_entropy,
                "probability_quality_issue": issue,
                "recommended_next_action": action,
            }
        )
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["candidate_type", "candidate", "mean_average_precision"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def _population_stability_index(actual: pd.Series, expected: pd.Series, *, bins: int) -> float:
    actual_values = pd.to_numeric(actual, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    expected_values = pd.to_numeric(expected, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(actual_values) == 0 or len(expected_values) == 0:
        return np.nan
    bins = max(2, int(bins))
    try:
        edges = np.unique(np.quantile(expected_values, np.linspace(0.0, 1.0, bins + 1)))
    except ValueError:
        return np.nan
    if len(edges) < 3:
        low = min(float(np.min(actual_values)), float(np.min(expected_values)))
        high = max(float(np.max(actual_values)), float(np.max(expected_values)))
        if not np.isfinite(low) or not np.isfinite(high) or low == high:
            return 0.0
        edges = np.linspace(low, high, bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    actual_counts, _ = np.histogram(actual_values, bins=edges)
    expected_counts, _ = np.histogram(expected_values, bins=edges)
    epsilon = 1e-6
    actual_pct = np.maximum(actual_counts / max(1, actual_counts.sum()), epsilon)
    expected_pct = np.maximum(expected_counts / max(1, expected_counts.sum()), epsilon)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def _score_distribution_shift_frame(entries: list[dict[str, Any]], config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "fold",
        "rank_ic",
        "rank_ic_bucket",
        "reference_definition",
        "reference_fold_count",
        "score_ks_vs_reference",
        "score_psi_vs_reference",
        "prob_long_mean",
        "reference_prob_long_mean",
        "prob_long_mean_delta",
        "prob_long_std",
        "reference_prob_long_std",
        "prob_long_std_ratio",
        "prob_long_iqr",
        "reference_prob_long_iqr",
        "score_entropy_mean",
        "reference_score_entropy_mean",
        "score_entropy_delta",
        "score_shift_issue",
    ]
    rows: list[dict[str, Any]] = []
    target_rank_ic = float(_cfg(config, ["validation", "target_rank_ic"], 0.03))
    bins = int(_cfg(config, ["validation", "score_lift_bins"], 10))
    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if not _is_stability_scope(fold_scope):
            continue
        predictions = entry.get("predictions")
        if not isinstance(predictions, pd.DataFrame) or predictions.empty:
            continue
        test_predictions = _test_predictions(predictions)
        if test_predictions.empty or not {"fold", "prob_long"}.issubset(test_predictions.columns):
            continue
        diagnostics = entry.get("diagnostics", {}) or {}
        fold_metrics = diagnostics.get("fold_metrics")
        fold_by_id = (
            {int(row["fold"]): row.to_dict() for _, row in fold_metrics.dropna(subset=["fold"]).iterrows()}
            if isinstance(fold_metrics, pd.DataFrame) and not fold_metrics.empty and "fold" in fold_metrics.columns
            else {}
        )
        rank_by_id = {fold: _float(row, "rank_ic") for fold, row in fold_by_id.items()}
        all_folds = sorted({int(fold) for fold in pd.to_numeric(test_predictions["fold"], errors="coerce").dropna().astype(int)})
        candidate = str(entry.get("profile", ""))
        candidate_type = _diagnostic_candidate_type(fold_scope)
        for fold in all_folds:
            part = test_predictions.loc[pd.to_numeric(test_predictions["fold"], errors="coerce") == fold].copy()
            scores = pd.to_numeric(part["prob_long"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if scores.empty:
                continue
            good_reference_folds = [
                other_fold
                for other_fold in all_folds
                if other_fold != fold and np.isfinite(rank_by_id.get(other_fold, np.nan)) and rank_by_id[other_fold] >= target_rank_ic
            ]
            reference_definition = f"other_folds_rank_ic_ge_{target_rank_ic:.3f}"
            if not good_reference_folds:
                good_reference_folds = [other_fold for other_fold in all_folds if other_fold != fold]
                reference_definition = "all_other_folds"
            reference = test_predictions.loc[test_predictions["fold"].astype(int).isin(good_reference_folds), "prob_long"]
            reference_scores = pd.to_numeric(reference, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if reference_scores.empty:
                continue
            rank_ic_value = rank_by_id.get(fold, np.nan)
            metrics = _binary_probability_metrics(pd.Series(np.zeros(len(scores))), scores, bins=bins)
            reference_metrics = _binary_probability_metrics(pd.Series(np.zeros(len(reference_scores))), reference_scores, bins=bins)
            score_std = float(scores.std(ddof=0))
            ref_std = float(reference_scores.std(ddof=0))
            score_iqr = float(scores.quantile(0.75) - scores.quantile(0.25))
            ref_iqr = float(reference_scores.quantile(0.75) - reference_scores.quantile(0.25))
            ks = _score_ks_statistic(scores, reference_scores)
            psi = _population_stability_index(scores, reference_scores, bins=bins)
            entropy_delta = metrics["score_entropy_mean"] - reference_metrics["score_entropy_mean"]
            if np.isfinite(psi) and psi >= 0.25:
                issue = "major_score_distribution_shift"
            elif np.isfinite(ks) and ks >= 0.20:
                issue = "large_score_distribution_ks_shift"
            elif np.isfinite(entropy_delta) and abs(entropy_delta) >= 0.05:
                issue = "score_uncertainty_shift"
            else:
                issue = "monitor"
            bucket = (
                "negative_rank_ic"
                if np.isfinite(rank_ic_value) and rank_ic_value < 0.0
                else ("below_target_rank_ic" if np.isfinite(rank_ic_value) and rank_ic_value < target_rank_ic else "rank_ic_ok")
            )
            rows.append(
                {
                    "candidate": candidate,
                    "candidate_type": candidate_type,
                    "fold_scope": fold_scope,
                    "fold": fold,
                    "rank_ic": rank_ic_value,
                    "rank_ic_bucket": bucket,
                    "reference_definition": reference_definition,
                    "reference_fold_count": len(good_reference_folds),
                    "score_ks_vs_reference": ks,
                    "score_psi_vs_reference": psi,
                    "prob_long_mean": float(scores.mean()),
                    "reference_prob_long_mean": float(reference_scores.mean()),
                    "prob_long_mean_delta": float(scores.mean() - reference_scores.mean()),
                    "prob_long_std": score_std,
                    "reference_prob_long_std": ref_std,
                    "prob_long_std_ratio": float(score_std / ref_std) if ref_std > 0 else np.nan,
                    "prob_long_iqr": score_iqr,
                    "reference_prob_long_iqr": ref_iqr,
                    "score_entropy_mean": metrics["score_entropy_mean"],
                    "reference_score_entropy_mean": reference_metrics["score_entropy_mean"],
                    "score_entropy_delta": entropy_delta,
                    "score_shift_issue": issue,
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["candidate_type", "candidate", "fold_scope", "score_psi_vs_reference"], ascending=[True, True, True, False])
        .reset_index(drop=True)
    )


def _score_distribution_shift_summary_frame(score_shift: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "fold_scope",
        "fold_count",
        "bad_fold_count",
        "mean_score_ks",
        "max_score_ks",
        "mean_score_psi",
        "max_score_psi",
        "bad_score_psi_mean",
        "good_score_psi_mean",
        "bad_score_ks_mean",
        "good_score_ks_mean",
        "high_shift_fold_count",
        "high_shift_folds",
        "score_shift_issue",
        "recommended_next_action",
    ]
    if score_shift.empty:
        return pd.DataFrame(columns=columns)
    target_rank_ic = float(_cfg(config, ["validation", "target_rank_ic"], 0.03))
    rows: list[dict[str, Any]] = []
    for (candidate, candidate_type, fold_scope), part in score_shift.groupby(
        ["candidate", "candidate_type", "fold_scope"],
        dropna=False,
    ):
        rank = pd.to_numeric(part["rank_ic"], errors="coerce")
        bad_mask = rank < target_rank_ic
        good_mask = rank >= target_rank_ic
        high_shift = (pd.to_numeric(part["score_psi_vs_reference"], errors="coerce") >= 0.25) | (
            pd.to_numeric(part["score_ks_vs_reference"], errors="coerce") >= 0.20
        )
        high_folds = sorted(part.loc[high_shift, "fold"].dropna().astype(int).tolist())
        bad_psi = _mean_for_mask(part, bad_mask, "score_psi_vs_reference")
        good_psi = _mean_for_mask(part, good_mask, "score_psi_vs_reference")
        bad_ks = _mean_for_mask(part, bad_mask, "score_ks_vs_reference")
        good_ks = _mean_for_mask(part, good_mask, "score_ks_vs_reference")
        if high_folds and np.isfinite(bad_psi) and np.isfinite(good_psi) and bad_psi > good_psi + 0.05:
            issue = "bad_folds_show_score_distribution_shift"
            action = "inspect_feature_drift_for_shifted_folds_before_new_profile_search"
        elif high_folds:
            issue = "score_distribution_shift_not_specific_to_bad_folds"
            action = "monitor_score_distribution_shift"
        else:
            issue = "monitor"
            action = "monitor"
        rows.append(
            {
                "candidate": candidate,
                "candidate_type": candidate_type,
                "fold_scope": fold_scope,
                "fold_count": int(part["fold"].nunique()),
                "bad_fold_count": int(bad_mask.sum()),
                "mean_score_ks": float(pd.to_numeric(part["score_ks_vs_reference"], errors="coerce").mean()),
                "max_score_ks": float(pd.to_numeric(part["score_ks_vs_reference"], errors="coerce").max()),
                "mean_score_psi": float(pd.to_numeric(part["score_psi_vs_reference"], errors="coerce").mean()),
                "max_score_psi": float(pd.to_numeric(part["score_psi_vs_reference"], errors="coerce").max()),
                "bad_score_psi_mean": bad_psi,
                "good_score_psi_mean": good_psi,
                "bad_score_ks_mean": bad_ks,
                "good_score_ks_mean": good_ks,
                "high_shift_fold_count": len(high_folds),
                "high_shift_folds": ",".join(str(fold) for fold in high_folds),
                "score_shift_issue": issue,
                "recommended_next_action": action,
            }
        )
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["candidate_type", "candidate", "max_score_psi"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


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


def _fold_reliability_gate_markdown(summary: pd.DataFrame) -> str:
    lines = ["# Fold Reliability Gates", ""]
    lines.append(
        "These rows test causal validation-fold reliability gates. They are diagnostics only: "
        "a gate can be considered for future unseen OOS only if it improves CV stability without using holdout feedback."
    )
    if summary.empty:
        lines.append("")
        lines.append("No fold-reliability gate rows were produced.")
        return "\n".join(lines)
    display_cols = [
        "candidate",
        "fold_scope",
        "gate_name",
        "accepted_fold_count",
        "accepted_fraction",
        "accepted_rank_ic_mean",
        "accepted_rank_ic_std",
        "accepted_positive_ic_fraction",
        "accepted_official_f1_mean",
        "rejected_negative_fold_capture_rate",
        "gate_passed_cv",
        "reject_reason",
        "next_action",
    ]
    visible = summary[[column for column in display_cols if column in summary.columns]].copy()
    lines.append("")
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    return "\n".join(lines)


def _probability_quality_markdown(summary: pd.DataFrame) -> str:
    lines = ["# Probability Quality Forensics", ""]
    if summary.empty:
        lines.append("No probability-quality rows were produced.")
        return "\n".join(lines)
    display_cols = [
        "candidate",
        "fold_scope",
        "mean_brier_score",
        "mean_average_precision",
        "mean_ece_equal_count",
        "bad_average_precision_mean",
        "good_average_precision_mean",
        "bad_ece_equal_count_mean",
        "good_ece_equal_count_mean",
        "probability_quality_issue",
        "recommended_next_action",
    ]
    visible = summary[[column for column in display_cols if column in summary.columns]].copy()
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    return "\n".join(lines)


def _score_distribution_shift_markdown(summary: pd.DataFrame) -> str:
    lines = ["# Score Distribution Shift", ""]
    if summary.empty:
        lines.append("No score-distribution shift rows were produced.")
        return "\n".join(lines)
    display_cols = [
        "candidate",
        "fold_scope",
        "mean_score_ks",
        "max_score_ks",
        "mean_score_psi",
        "max_score_psi",
        "bad_score_psi_mean",
        "good_score_psi_mean",
        "high_shift_folds",
        "score_shift_issue",
        "recommended_next_action",
    ]
    visible = summary[[column for column in display_cols if column in summary.columns]].copy()
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    return "\n".join(lines)


def _write_probability_quality_forensics(path: Path, detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    detail.to_csv(path / "probability_quality_forensics.csv", index=False)
    summary.to_csv(path / "probability_quality_summary.csv", index=False)
    (path / "probability_quality_forensics.md").write_text(_probability_quality_markdown(summary), encoding="utf-8")
    _write_json(
        path / "probability_quality_forensics.json",
        {
            "probability_quality_forensics": detail.to_dict(orient="records"),
            "probability_quality_summary": summary.to_dict(orient="records"),
        },
    )


def _write_score_distribution_shift(path: Path, detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    detail.to_csv(path / "score_distribution_shift.csv", index=False)
    summary.to_csv(path / "score_distribution_shift_summary.csv", index=False)
    (path / "score_distribution_shift.md").write_text(_score_distribution_shift_markdown(summary), encoding="utf-8")
    _write_json(
        path / "score_distribution_shift.json",
        {
            "score_distribution_shift": detail.to_dict(orient="records"),
            "score_distribution_shift_summary": summary.to_dict(orient="records"),
        },
    )


def _write_fold_reliability_gate(path: Path, detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    detail.to_csv(path / "fold_reliability_gate.csv", index=False)
    summary.to_csv(path / "fold_reliability_gate_summary.csv", index=False)
    (path / "fold_reliability_gate.md").write_text(_fold_reliability_gate_markdown(summary), encoding="utf-8")
    _write_json(
        path / "fold_reliability_gate.json",
        {
            "fold_reliability_gate": detail.to_dict(orient="records"),
            "fold_reliability_gate_summary": summary.to_dict(orient="records"),
        },
    )


def _forensics_markdown(title: str, frame: pd.DataFrame) -> str:
    return _table_markdown(title, frame)


def _write_forensics_reports(
    path: Path,
    *,
    fold_stability_forensics: pd.DataFrame,
    fold_stability_summary: pd.DataFrame,
    threshold_forensics: pd.DataFrame,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    reports = [
        ("fold_stability_forensics", "Fold Stability Forensics", fold_stability_forensics),
        ("fold_stability_summary", "Fold Stability Summary", fold_stability_summary),
        ("threshold_forensics", "Threshold Forensics", threshold_forensics),
    ]
    for stem, title, frame in reports:
        frame.to_csv(path / f"{stem}.csv", index=False)
        (path / f"{stem}.md").write_text(_forensics_markdown(title, frame), encoding="utf-8")
        _write_json(path / f"{stem}.json", {"rows": frame.to_dict(orient="records")})


def _assign_payoff_score_bins(predictions: pd.DataFrame, *, score_column: str, bins: int) -> pd.DataFrame:
    required = {"label", score_column}
    if predictions.empty or not required.issubset(predictions.columns):
        return pd.DataFrame()
    frame = predictions.copy().replace([np.inf, -np.inf], np.nan).dropna(subset=["label", score_column])
    if frame.empty:
        return frame
    q = max(1, min(int(bins), len(frame)))
    frame["score_bin"] = pd.qcut(
        frame[score_column].rank(method="first"),
        q=q,
        labels=False,
        duplicates="drop",
    )
    return frame.dropna(subset=["score_bin"]).copy()


def _resolve_payoff_score_bands(config: dict[str, Any], actual_bins: int) -> list[dict[str, Any]]:
    max_bin = max(0, int(actual_bins) - 1)
    configured = _cfg(config, ["validation", "score_bands"], None)
    if not configured:
        configured = [
            {"name": "top_10", "min_bin": max_bin, "max_bin": max_bin},
            {"name": "top_20", "min_bin": max(0, int(np.floor(actual_bins * 0.80))), "max_bin": max_bin},
            {"name": "top_30", "min_bin": max(0, int(np.floor(actual_bins * 0.70))), "max_bin": max_bin},
            {"name": "upper_half", "min_bin": max(0, int(np.floor(actual_bins * 0.50))), "max_bin": max_bin},
            {
                "name": "mid_upper_40_90",
                "min_bin": max(0, int(np.floor(actual_bins * 0.40))),
                "max_bin": max(0, max_bin - 1),
            },
        ]
    bands = []
    for item in configured:
        name = str(item.get("name", f"bins_{item.get('min_bin')}_{item.get('max_bin')}"))
        min_bin = min(max(int(item.get("min_bin", max_bin)), 0), max_bin)
        max_item_bin = min(max(int(item.get("max_bin", max_bin)), 0), max_bin)
        if min_bin <= max_item_bin:
            bands.append({"name": name, "min_bin": min_bin, "max_bin": max_item_bin})
    return bands


def _numeric_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns or frame.empty:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.mean()) if not values.empty else np.nan


def _hit_rate(frame: pd.DataFrame, hit_type: str) -> float:
    if "hit_type" not in frame.columns or frame.empty:
        return np.nan
    return float(frame["hit_type"].astype(str).eq(hit_type).mean())


def _payoff_alignment_blockers(row: dict[str, Any]) -> str:
    reasons = []
    if _float(row, "label_lift_vs_base") <= 1.0:
        reasons.append("label_lift_not_above_base")
    if _float(row, "mean_forward_return") <= 0.0:
        reasons.append("forward_return_not_positive")
    if np.isfinite(_float(row, "mean_tb_return")) and _float(row, "mean_tb_return") <= 0.0:
        reasons.append("tb_return_not_positive")
    if np.isfinite(_float(row, "sl_rate_delta_vs_base")) and _float(row, "sl_rate_delta_vs_base") > 0.0:
        reasons.append("sl_rate_above_base")
    if _float(row, "selection_rate") <= 0.0:
        reasons.append("empty_selection")
    return ";".join(reasons)


def _payoff_alignment_action(row: dict[str, Any]) -> str:
    blockers = str(row.get("payoff_blockers", ""))
    if not blockers:
        return "candidate_band_payoff_aligned_monitor_future_oos"
    if "label_lift_not_above_base" in blockers:
        return "weak_label_discrimination_do_not_use_band"
    if "forward_return_not_positive" in blockers or "tb_return_not_positive" in blockers:
        return "investigate_payoff_mismatch_before_new_profile_search"
    if "sl_rate_above_base" in blockers:
        return "inspect_stop_loss_regime_exposure"
    return "monitor"


def _payoff_alignment_rows_for_entry(
    entry: dict[str, Any],
    config: dict[str, Any],
    *,
    evaluation_scope: str,
) -> list[dict[str, Any]]:
    predictions = entry.get("predictions", pd.DataFrame())
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        return []
    frame = _test_predictions(predictions)
    score_bins = int(_cfg(config, ["validation", "score_lift_bins"], _cfg(config, ["validation", "calibration_bins"], 10)))
    frame = _assign_payoff_score_bins(frame, score_column="prob_long", bins=score_bins)
    if frame.empty:
        return []

    actual_bins = int(pd.to_numeric(frame["score_bin"], errors="coerce").max()) + 1
    bands = _resolve_payoff_score_bands(config, actual_bins)
    profile = str(entry.get("profile", ""))
    fold_scope = str(entry.get("fold_scope", ""))
    candidate_type = "blend" if fold_scope.startswith("blend_") or profile.startswith("blend_") else "profile"
    base_count = int(len(frame))
    base_long_rate = float(pd.to_numeric(frame["label"], errors="coerce").mean())
    base_forward_return = _numeric_mean(frame, "forward_return")
    base_tb_return = _numeric_mean(frame, "tb_return")
    base_tp_rate = _hit_rate(frame, "tp")
    base_sl_rate = _hit_rate(frame, "sl") + _hit_rate(frame, "both_sl_first")
    base_time_rate = _hit_rate(frame, "time")
    rows = []
    for band in bands:
        part = frame.loc[
            (pd.to_numeric(frame["score_bin"], errors="coerce") >= int(band["min_bin"]))
            & (pd.to_numeric(frame["score_bin"], errors="coerce") <= int(band["max_bin"]))
        ].copy()
        if part.empty:
            continue
        selected_count = int(len(part))
        selected_long_rate = float(pd.to_numeric(part["label"], errors="coerce").mean())
        mean_forward_return = _numeric_mean(part, "forward_return")
        mean_tb_return = _numeric_mean(part, "tb_return")
        tp_rate = _hit_rate(part, "tp")
        sl_rate = _hit_rate(part, "sl") + _hit_rate(part, "both_sl_first")
        time_rate = _hit_rate(part, "time")
        row = {
            "candidate": profile,
            "candidate_type": candidate_type,
            "evaluation_scope": evaluation_scope,
            "fold_scope": fold_scope,
            "band": str(band["name"]),
            "min_bin": int(band["min_bin"]),
            "max_bin": int(band["max_bin"]),
            "base_count": base_count,
            "selected_count": selected_count,
            "selection_rate": float(selected_count / base_count) if base_count else np.nan,
            "mean_prob_long": _numeric_mean(part, "prob_long"),
            "base_long_rate": base_long_rate,
            "selected_long_rate": selected_long_rate,
            "label_lift_vs_base": float(selected_long_rate / base_long_rate) if base_long_rate > 0 else np.nan,
            "base_forward_return": base_forward_return,
            "mean_forward_return": mean_forward_return,
            "forward_return_delta_vs_base": mean_forward_return - base_forward_return,
            "base_tb_return": base_tb_return,
            "mean_tb_return": mean_tb_return,
            "tb_return_delta_vs_base": mean_tb_return - base_tb_return,
            "base_tp_rate": base_tp_rate,
            "tp_rate": tp_rate,
            "tp_rate_delta_vs_base": tp_rate - base_tp_rate,
            "base_sl_rate": base_sl_rate,
            "sl_rate": sl_rate,
            "sl_rate_delta_vs_base": sl_rate - base_sl_rate,
            "base_time_rate": base_time_rate,
            "time_rate": time_rate,
            "time_rate_delta_vs_base": time_rate - base_time_rate,
            "label_lift_positive_payoff_mismatch": bool(
                selected_long_rate > base_long_rate and np.isfinite(mean_forward_return) and mean_forward_return <= 0.0
            ),
        }
        row["payoff_blockers"] = _payoff_alignment_blockers(row)
        row["payoff_alignment_pass"] = not bool(row["payoff_blockers"])
        row["next_action"] = _payoff_alignment_action(row)
        rows.append(row)
    return rows


def _payoff_alignment_frame(
    entries: list[dict[str, Any]],
    holdout_entries: list[dict[str, Any]],
    config: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "evaluation_scope",
        "fold_scope",
        "band",
        "min_bin",
        "max_bin",
        "base_count",
        "selected_count",
        "selection_rate",
        "mean_prob_long",
        "base_long_rate",
        "selected_long_rate",
        "label_lift_vs_base",
        "base_forward_return",
        "mean_forward_return",
        "forward_return_delta_vs_base",
        "base_tb_return",
        "mean_tb_return",
        "tb_return_delta_vs_base",
        "base_tp_rate",
        "tp_rate",
        "tp_rate_delta_vs_base",
        "base_sl_rate",
        "sl_rate",
        "sl_rate_delta_vs_base",
        "base_time_rate",
        "time_rate",
        "time_rate_delta_vs_base",
        "label_lift_positive_payoff_mismatch",
        "payoff_alignment_pass",
        "payoff_blockers",
        "next_action",
    ]
    rows: list[dict[str, Any]] = []
    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if fold_scope == "full" or fold_scope.startswith("blend_"):
            rows.extend(_payoff_alignment_rows_for_entry(entry, config, evaluation_scope="cv_test"))
    for entry in holdout_entries:
        rows.extend(_payoff_alignment_rows_for_entry(entry, config, evaluation_scope="holdout"))
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows)
        .reindex(columns=columns)
        .sort_values(["evaluation_scope", "candidate_type", "candidate", "min_bin", "max_bin"])
        .reset_index(drop=True)
    )


def _payoff_alignment_summary_frame(payoff_alignment: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "evaluation_scope",
        "top_10_label_lift_vs_base",
        "top_10_mean_forward_return",
        "top_10_mean_tb_return",
        "top_10_tp_rate",
        "top_10_sl_rate",
        "top_10_payoff_alignment_pass",
        "top_10_payoff_blockers",
        "best_forward_return_band",
        "best_forward_return",
        "best_forward_return_label_lift",
        "best_lift_band",
        "best_lift",
        "best_lift_forward_return",
        "payoff_aligned_band_count",
        "payoff_mismatch_band_count",
        "next_action",
    ]
    if payoff_alignment.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for (candidate, candidate_type, evaluation_scope), part in payoff_alignment.groupby(
        ["candidate", "candidate_type", "evaluation_scope"],
        dropna=False,
    ):
        part = part.copy()
        top_10 = part.loc[part["band"].astype(str).eq("top_10")]
        top = top_10.iloc[0].to_dict() if not top_10.empty else {}
        best_return = part.sort_values("mean_forward_return", ascending=False).iloc[0].to_dict()
        best_lift = part.sort_values("label_lift_vs_base", ascending=False).iloc[0].to_dict()
        mismatch_count = int(part["label_lift_positive_payoff_mismatch"].astype(bool).sum())
        aligned_count = int(part["payoff_alignment_pass"].astype(bool).sum())
        if top and bool(top.get("payoff_alignment_pass", False)):
            action = "top_10_payoff_aligned_monitor_future_oos"
        elif top and bool(top.get("label_lift_positive_payoff_mismatch", False)):
            action = "top_10_label_lift_payoff_mismatch_investigate"
        elif aligned_count > 0:
            action = "review_non_top10_payoff_aligned_band_before_future_oos"
        else:
            action = "no_payoff_aligned_band_do_not_promote"
        rows.append(
            {
                "candidate": str(candidate),
                "candidate_type": str(candidate_type),
                "evaluation_scope": str(evaluation_scope),
                "top_10_label_lift_vs_base": _float(top, "label_lift_vs_base") if top else np.nan,
                "top_10_mean_forward_return": _float(top, "mean_forward_return") if top else np.nan,
                "top_10_mean_tb_return": _float(top, "mean_tb_return") if top else np.nan,
                "top_10_tp_rate": _float(top, "tp_rate") if top else np.nan,
                "top_10_sl_rate": _float(top, "sl_rate") if top else np.nan,
                "top_10_payoff_alignment_pass": bool(top.get("payoff_alignment_pass", False)) if top else False,
                "top_10_payoff_blockers": str(top.get("payoff_blockers", "")) if top else "missing_top_10",
                "best_forward_return_band": str(best_return.get("band", "")),
                "best_forward_return": _float(best_return, "mean_forward_return"),
                "best_forward_return_label_lift": _float(best_return, "label_lift_vs_base"),
                "best_lift_band": str(best_lift.get("band", "")),
                "best_lift": _float(best_lift, "label_lift_vs_base"),
                "best_lift_forward_return": _float(best_lift, "mean_forward_return"),
                "payoff_aligned_band_count": aligned_count,
                "payoff_mismatch_band_count": mismatch_count,
                "next_action": action,
            }
        )
    return (
        pd.DataFrame(rows)
        .reindex(columns=columns)
        .sort_values(["evaluation_scope", "candidate_type", "top_10_mean_forward_return"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def _payoff_alignment_markdown(summary: pd.DataFrame, detail: pd.DataFrame) -> str:
    lines = ["# Payoff Alignment", ""]
    if summary.empty:
        lines.append("No payoff alignment rows were produced.")
        return "\n".join(lines)
    display_cols = [
        "candidate",
        "candidate_type",
        "evaluation_scope",
        "top_10_label_lift_vs_base",
        "top_10_mean_forward_return",
        "top_10_mean_tb_return",
        "top_10_tp_rate",
        "top_10_sl_rate",
        "top_10_payoff_alignment_pass",
        "top_10_payoff_blockers",
        "best_forward_return_band",
        "best_forward_return",
        "next_action",
    ]
    visible = summary[[column for column in display_cols if column in summary.columns]].copy()
    lines.append("## Summary")
    lines.append("")
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    mismatch = detail.loc[detail.get("label_lift_positive_payoff_mismatch", pd.Series(dtype=bool)).astype(bool)]
    if not mismatch.empty:
        lines.extend(["", "## Label Lift / Payoff Mismatches", ""])
        mismatch_cols = [
            "candidate",
            "evaluation_scope",
            "band",
            "label_lift_vs_base",
            "mean_forward_return",
            "mean_tb_return",
            "payoff_blockers",
        ]
        visible_mismatch = mismatch[[column for column in mismatch_cols if column in mismatch.columns]].copy()
        lines.append("| " + " | ".join(visible_mismatch.columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(visible_mismatch.columns)) + " |")
        for _, row in visible_mismatch.iterrows():
            lines.append("| " + " | ".join(str(row[column]) for column in visible_mismatch.columns) + " |")
    return "\n".join(lines)


def _write_payoff_alignment(path: Path, detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    detail.to_csv(path / "payoff_alignment.csv", index=False)
    summary.to_csv(path / "payoff_alignment_summary.csv", index=False)
    (path / "payoff_alignment.md").write_text(_payoff_alignment_markdown(summary, detail), encoding="utf-8")
    _write_json(
        path / "payoff_alignment.json",
        {
            "summary": summary.to_dict(orient="records"),
            "detail": detail.to_dict(orient="records"),
        },
    )


def _payoff_policy_rows_for_frame(
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    candidate: str,
    candidate_type: str,
    evaluation_scope: str,
    fold_scope: str,
    fold: int,
) -> list[dict[str, Any]]:
    scored = _assign_payoff_score_bins(frame, score_column="prob_long", bins=int(_cfg(config, ["validation", "score_lift_bins"], 10)))
    if scored.empty:
        return []

    actual_bins = int(pd.to_numeric(scored["score_bin"], errors="coerce").max()) + 1
    bands = _resolve_payoff_score_bands(config, actual_bins)
    base_count = int(len(scored))
    base_long_rate = float(pd.to_numeric(scored["label"], errors="coerce").mean())
    base_forward_return = _numeric_mean(scored, "forward_return")
    base_tb_return = _numeric_mean(scored, "tb_return")
    base_tp_rate = _hit_rate(scored, "tp")
    base_sl_rate = _hit_rate(scored, "sl") + _hit_rate(scored, "both_sl_first")
    base_time_rate = _hit_rate(scored, "time")
    rows = []
    for band in bands:
        part = scored.loc[
            (pd.to_numeric(scored["score_bin"], errors="coerce") >= int(band["min_bin"]))
            & (pd.to_numeric(scored["score_bin"], errors="coerce") <= int(band["max_bin"]))
        ].copy()
        if part.empty:
            continue
        selected_count = int(len(part))
        selected_long_rate = float(pd.to_numeric(part["label"], errors="coerce").mean())
        mean_forward_return = _numeric_mean(part, "forward_return")
        mean_tb_return = _numeric_mean(part, "tb_return")
        tp_rate = _hit_rate(part, "tp")
        sl_rate = _hit_rate(part, "sl") + _hit_rate(part, "both_sl_first")
        time_rate = _hit_rate(part, "time")
        row = {
            "candidate": candidate,
            "candidate_type": candidate_type,
            "evaluation_scope": evaluation_scope,
            "fold_scope": fold_scope,
            "fold": int(fold),
            "band": str(band["name"]),
            "min_bin": int(band["min_bin"]),
            "max_bin": int(band["max_bin"]),
            "base_count": base_count,
            "selected_count": selected_count,
            "selection_rate": float(selected_count / base_count) if base_count else np.nan,
            "base_long_rate": base_long_rate,
            "selected_long_rate": selected_long_rate,
            "label_lift_vs_base": float(selected_long_rate / base_long_rate) if base_long_rate > 0 else np.nan,
            "base_forward_return": base_forward_return,
            "mean_forward_return": mean_forward_return,
            "forward_return_delta_vs_base": mean_forward_return - base_forward_return,
            "base_tb_return": base_tb_return,
            "mean_tb_return": mean_tb_return,
            "tb_return_delta_vs_base": mean_tb_return - base_tb_return,
            "base_tp_rate": base_tp_rate,
            "tp_rate": tp_rate,
            "tp_rate_delta_vs_base": tp_rate - base_tp_rate,
            "base_sl_rate": base_sl_rate,
            "sl_rate": sl_rate,
            "sl_rate_delta_vs_base": sl_rate - base_sl_rate,
            "base_time_rate": base_time_rate,
            "time_rate": time_rate,
            "time_rate_delta_vs_base": time_rate - base_time_rate,
        }
        row["payoff_blockers"] = _payoff_alignment_blockers(row)
        row["payoff_alignment_pass"] = not bool(row["payoff_blockers"])
        rows.append(row)
    return rows


def _payoff_policy_robustness_frame(
    entries: list[dict[str, Any]],
    holdout_entries: list[dict[str, Any]],
    config: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "evaluation_scope",
        "fold_scope",
        "fold",
        "band",
        "min_bin",
        "max_bin",
        "base_count",
        "selected_count",
        "selection_rate",
        "base_long_rate",
        "selected_long_rate",
        "label_lift_vs_base",
        "base_forward_return",
        "mean_forward_return",
        "forward_return_delta_vs_base",
        "base_tb_return",
        "mean_tb_return",
        "tb_return_delta_vs_base",
        "base_tp_rate",
        "tp_rate",
        "tp_rate_delta_vs_base",
        "base_sl_rate",
        "sl_rate",
        "sl_rate_delta_vs_base",
        "base_time_rate",
        "time_rate",
        "time_rate_delta_vs_base",
        "payoff_alignment_pass",
        "payoff_blockers",
    ]
    rows = []
    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if fold_scope != "full" and not fold_scope.startswith("blend_"):
            continue
        predictions = entry.get("predictions", pd.DataFrame())
        if not isinstance(predictions, pd.DataFrame) or predictions.empty or "fold" not in predictions.columns:
            continue
        test_predictions = _test_predictions(predictions)
        candidate = str(entry.get("profile", ""))
        candidate_type = "blend" if fold_scope.startswith("blend_") or candidate.startswith("blend_") else "profile"
        for fold, part in test_predictions.groupby("fold", dropna=False):
            rows.extend(
                _payoff_policy_rows_for_frame(
                    part,
                    config,
                    candidate=candidate,
                    candidate_type=candidate_type,
                    evaluation_scope="cv_test",
                    fold_scope=fold_scope,
                    fold=int(fold),
                )
            )
    for entry in holdout_entries:
        predictions = entry.get("predictions", pd.DataFrame())
        if not isinstance(predictions, pd.DataFrame) or predictions.empty:
            continue
        candidate = str(entry.get("profile", ""))
        fold_scope = str(entry.get("fold_scope", ""))
        candidate_type = "blend" if fold_scope.startswith("blend_") or candidate.startswith("blend_") else "profile"
        rows.extend(
            _payoff_policy_rows_for_frame(
                _test_predictions(predictions),
                config,
                candidate=candidate,
                candidate_type=candidate_type,
                evaluation_scope="holdout",
                fold_scope=fold_scope,
                fold=0,
            )
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows)
        .reindex(columns=columns)
        .sort_values(["evaluation_scope", "candidate_type", "candidate", "band", "fold"])
        .reset_index(drop=True)
    )


def _payoff_policy_reject_reasons(row: dict[str, Any], config: dict[str, Any]) -> str:
    gates = _cfg(config, ["validation", "payoff_policy_robustness"], {}) or {}
    reasons = []
    if _float(row, "mean_label_lift_vs_base") < float(gates.get("min_mean_label_lift_vs_base", 1.05)):
        reasons.append("mean_label_lift_vs_base")
    if _float(row, "positive_label_lift_fold_rate") < float(gates.get("min_positive_label_lift_fold_rate", 0.60)):
        reasons.append("positive_label_lift_fold_rate")
    if _float(row, "mean_forward_return") <= float(gates.get("min_mean_forward_return", 0.0)):
        reasons.append("mean_forward_return")
    if _float(row, "positive_forward_return_fold_rate") < float(gates.get("min_positive_forward_return_fold_rate", 0.60)):
        reasons.append("positive_forward_return_fold_rate")
    if _float(row, "mean_tb_return") <= float(gates.get("min_mean_tb_return", 0.0)):
        reasons.append("mean_tb_return")
    if _float(row, "positive_tb_return_fold_rate") < float(gates.get("min_positive_tb_return_fold_rate", 0.55)):
        reasons.append("positive_tb_return_fold_rate")
    if _float(row, "payoff_alignment_fold_rate") < float(gates.get("min_payoff_alignment_fold_rate", 0.50)):
        reasons.append("payoff_alignment_fold_rate")
    if _float(row, "sl_rate_above_base_fold_rate") > float(gates.get("max_sl_rate_above_base_fold_rate", 0.70)):
        reasons.append("sl_rate_above_base_fold_rate")
    return ";".join(reasons)


def _payoff_policy_robustness_summary_frame(robustness: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "candidate",
        "candidate_type",
        "evaluation_scope",
        "band",
        "folds",
        "mean_selection_rate",
        "mean_label_lift_vs_base",
        "positive_label_lift_fold_rate",
        "mean_forward_return",
        "positive_forward_return_fold_rate",
        "mean_tb_return",
        "positive_tb_return_fold_rate",
        "mean_tp_rate",
        "mean_sl_rate",
        "sl_rate_above_base_fold_rate",
        "payoff_alignment_fold_rate",
        "future_oos_policy_candidate",
        "reject_reason",
        "next_action",
    ]
    if robustness.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    frame = robustness.copy()
    numeric_columns = [
        "selection_rate",
        "label_lift_vs_base",
        "mean_forward_return",
        "mean_tb_return",
        "tp_rate",
        "sl_rate",
        "sl_rate_delta_vs_base",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for (candidate, candidate_type, evaluation_scope, band), part in frame.groupby(
        ["candidate", "candidate_type", "evaluation_scope", "band"],
        dropna=False,
    ):
        row = {
            "candidate": str(candidate),
            "candidate_type": str(candidate_type),
            "evaluation_scope": str(evaluation_scope),
            "band": str(band),
            "folds": int(part["fold"].nunique()),
            "mean_selection_rate": float(part["selection_rate"].mean()),
            "mean_label_lift_vs_base": float(part["label_lift_vs_base"].mean()),
            "positive_label_lift_fold_rate": float((part["label_lift_vs_base"] > 1.0).mean()),
            "mean_forward_return": float(part["mean_forward_return"].mean()),
            "positive_forward_return_fold_rate": float((part["mean_forward_return"] > 0.0).mean()),
            "mean_tb_return": float(part["mean_tb_return"].mean()),
            "positive_tb_return_fold_rate": float((part["mean_tb_return"] > 0.0).mean()),
            "mean_tp_rate": float(part["tp_rate"].mean()),
            "mean_sl_rate": float(part["sl_rate"].mean()),
            "sl_rate_above_base_fold_rate": float((part["sl_rate_delta_vs_base"] > 0.0).mean()),
            "payoff_alignment_fold_rate": float(part["payoff_alignment_pass"].astype(bool).mean()),
        }
        reject_reason = _payoff_policy_reject_reasons(row, config)
        row["future_oos_policy_candidate"] = bool(str(evaluation_scope) == "cv_test" and not reject_reason)
        row["reject_reason"] = reject_reason
        if str(evaluation_scope) == "holdout":
            row["next_action"] = "diagnostic_only_do_not_select_from_current_holdout"
        elif row["future_oos_policy_candidate"]:
            row["next_action"] = "pre_register_for_future_oos_review"
        elif "mean_forward_return" in reject_reason or "mean_tb_return" in reject_reason:
            row["next_action"] = "payoff_not_robust_do_not_pre_register"
        else:
            row["next_action"] = "monitor_not_pre_registered"
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows)
        .reindex(columns=columns)
        .sort_values(
            ["evaluation_scope", "future_oos_policy_candidate", "mean_forward_return", "mean_label_lift_vs_base"],
            ascending=[True, False, False, False],
        )
        .reset_index(drop=True)
    )


def _payoff_policy_robustness_markdown(summary: pd.DataFrame) -> str:
    lines = ["# Payoff Policy Robustness", ""]
    if summary.empty:
        lines.append("No score-band payoff policy robustness rows were produced.")
        return "\n".join(lines)
    display_cols = [
        "candidate",
        "evaluation_scope",
        "band",
        "folds",
        "mean_label_lift_vs_base",
        "positive_label_lift_fold_rate",
        "mean_forward_return",
        "positive_forward_return_fold_rate",
        "mean_tb_return",
        "positive_tb_return_fold_rate",
        "mean_sl_rate",
        "sl_rate_above_base_fold_rate",
        "payoff_alignment_fold_rate",
        "future_oos_policy_candidate",
        "reject_reason",
        "next_action",
    ]
    visible = summary[[column for column in display_cols if column in summary.columns]].copy()
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    return "\n".join(lines)


def _write_payoff_policy_robustness(path: Path, detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    detail.to_csv(path / "payoff_policy_robustness.csv", index=False)
    summary.to_csv(path / "payoff_policy_robustness_summary.csv", index=False)
    (path / "payoff_policy_robustness.md").write_text(
        _payoff_policy_robustness_markdown(summary),
        encoding="utf-8",
    )
    _write_json(
        path / "payoff_policy_robustness.json",
        {
            "summary": summary.to_dict(orient="records"),
            "detail": detail.to_dict(orient="records"),
        },
    )


def _frozen_policy_robustness_frame(entries: list[dict[str, Any]], config: dict[str, Any]) -> pd.DataFrame:
    policy_review = _cfg(config, ["experiments", "policy_review"], {}) or {}
    robustness = policy_review.get("robustness", {}) or {}
    columns = [
        "window",
        "start",
        "end",
        "candidate",
        "policy_type",
        "policy_name",
        "rows",
        "selected_rows",
        "base_long_rate",
        "policy_precision",
        "policy_recall",
        "policy_f1",
        "policy_lift_vs_base",
        "policy_forward_return",
        "rank_ic",
        "window_pass",
        "reject_reason",
    ]
    if not bool(policy_review.get("enabled", False)) or not bool(robustness.get("enabled", False)):
        return pd.DataFrame(columns=columns)

    frozen_candidate = str(policy_review.get("frozen_candidate", ""))
    policy_type = str(policy_review.get("policy_type", ""))
    policy_name = str(policy_review.get("policy_name", ""))
    entry = next((item for item in entries if str(item.get("profile", "")) == frozen_candidate), None)
    if entry is None:
        return pd.DataFrame(
            [
                {
                    "window": "all",
                    "start": "",
                    "end": "",
                    "candidate": frozen_candidate,
                    "policy_type": policy_type,
                    "policy_name": policy_name,
                    "rows": 0,
                    "selected_rows": 0,
                    "base_long_rate": np.nan,
                    "policy_precision": np.nan,
                    "policy_recall": np.nan,
                    "policy_f1": np.nan,
                    "policy_lift_vs_base": np.nan,
                    "policy_forward_return": np.nan,
                    "rank_ic": np.nan,
                    "window_pass": False,
                    "reject_reason": "missing_frozen_candidate_predictions",
                }
            ],
            columns=columns,
        )

    predictions = _test_predictions(entry.get("predictions", pd.DataFrame())).copy()
    if predictions.empty or "timestamp" not in predictions.columns:
        return pd.DataFrame(
            [
                {
                    "window": "all",
                    "start": "",
                    "end": "",
                    "candidate": frozen_candidate,
                    "policy_type": policy_type,
                    "policy_name": policy_name,
                    "rows": int(len(predictions)),
                    "selected_rows": 0,
                    "base_long_rate": np.nan,
                    "policy_precision": np.nan,
                    "policy_recall": np.nan,
                    "policy_f1": np.nan,
                    "policy_lift_vs_base": np.nan,
                    "policy_forward_return": np.nan,
                    "rank_ic": np.nan,
                    "window_pass": False,
                    "reject_reason": "missing_frozen_candidate_timestamps",
                }
            ],
            columns=columns,
        )

    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], utc=True)
    windows = robustness.get("windows", []) or []
    if not windows:
        windows = [
            {
                "name": "all_available_cv",
                "start": str(predictions["timestamp"].min()),
                "end": str(predictions["timestamp"].max()),
            }
        ]

    min_rows = int(robustness.get("min_rows", 0) or 0)
    min_selected_rows = int(robustness.get("min_selected_rows", 0) or 0)
    min_rank_ic = float(robustness.get("min_rank_ic", 0.0))
    min_lift = float(robustness.get("min_lift_vs_base", 1.0))
    min_forward_return = float(robustness.get("min_forward_return", 0.0))
    policy = {"policy_type": policy_type, "policy_name": policy_name}
    rows = []
    for spec in windows:
        if not isinstance(spec, dict):
            continue
        name = str(spec.get("name", "window"))
        start_raw = str(spec.get("start", ""))
        end_raw = str(spec.get("end", ""))
        try:
            start = pd.to_datetime(start_raw, utc=True)
            end = pd.to_datetime(end_raw, utc=True)
        except (TypeError, ValueError):
            rows.append(
                {
                    "window": name,
                    "start": start_raw,
                    "end": end_raw,
                    "candidate": frozen_candidate,
                    "policy_type": policy_type,
                    "policy_name": policy_name,
                    "rows": 0,
                    "selected_rows": 0,
                    "base_long_rate": np.nan,
                    "policy_precision": np.nan,
                    "policy_recall": np.nan,
                    "policy_f1": np.nan,
                    "policy_lift_vs_base": np.nan,
                    "policy_forward_return": np.nan,
                    "rank_ic": np.nan,
                    "window_pass": False,
                    "reject_reason": "invalid_window_bounds",
                }
            )
            continue

        part = predictions.loc[(predictions["timestamp"] >= start) & (predictions["timestamp"] <= end)].copy()
        metrics = _evaluate_score_policy_on_holdout(part, policy, config) if not part.empty else {}
        rows_count = int(len(part))
        selection_rate = _float(metrics, "selection_rate", 0.0)
        selected_rows = int(round(rows_count * selection_rate))
        rank_ic = _rank_ic_for_frame(part)
        base_long_rate = float(part["label"].mean()) if rows_count and "label" in part.columns else np.nan
        reasons = []
        if rows_count < min_rows:
            reasons.append("rows")
        if selected_rows < min_selected_rows:
            reasons.append("selected_rows")
        if not bool(metrics.get("pass", False)):
            reasons.append(str(metrics.get("reject_reason", "policy")).strip(";") or "policy")
        if not np.isfinite(rank_ic) or rank_ic <= min_rank_ic:
            reasons.append("rank_ic")
        lift = _float(metrics, "lift_vs_base")
        forward_return = _float(metrics, "forward_return")
        if not np.isfinite(lift) or lift <= min_lift:
            reasons.append("lift_vs_base")
        if not np.isfinite(forward_return) or forward_return <= min_forward_return:
            reasons.append("forward_return")
        rows.append(
            {
                "window": name,
                "start": str(start),
                "end": str(end),
                "candidate": frozen_candidate,
                "policy_type": policy_type,
                "policy_name": policy_name,
                "rows": rows_count,
                "selected_rows": selected_rows,
                "base_long_rate": base_long_rate,
                "policy_precision": _float(metrics, "precision"),
                "policy_recall": _float(metrics, "recall"),
                "policy_f1": _float(metrics, "f1"),
                "policy_lift_vs_base": lift,
                "policy_forward_return": forward_return,
                "rank_ic": rank_ic,
                "window_pass": len(reasons) == 0,
                "reject_reason": ";".join(reason for reason in reasons if reason),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _write_frozen_policy_robustness(path: Path, frame: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path / "frozen_policy_robustness.csv", index=False)
    (path / "frozen_policy_robustness.md").write_text(
        _table_markdown("Frozen Policy Robustness", frame),
        encoding="utf-8",
    )
    _write_json(path / "frozen_policy_robustness.json", {"rows": frame.to_dict(orient="records")})


def _evaluate_holdout_candidates(
    *,
    profile_entries: list[dict[str, Any]],
    cv_blend_entries: list[dict[str, Any]] | None = None,
    settings: dict[str, Any],
    config: dict[str, Any],
    decision: dict[str, Any],
    holdout_boundary_passed: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    holdout_context, holdout_start = _read_holdout_context(settings, config)
    if holdout_context.empty or holdout_start is None:
        holdout_decision = {
            "available": False,
            "reason": "missing_holdout_frame_or_metadata",
            "policy": "holdout result must remain separate from profile selection",
            "holdout_boundary_passed": bool(holdout_boundary_passed),
        }
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), holdout_decision, []

    full_entries = [
        entry
        for entry in profile_entries
        if str(entry.get("fold_scope", "")) == "full"
        and not str(entry.get("profile", "")).startswith("blend_")
    ]
    holdout_entries: list[dict[str, Any]] = []
    score_band_rows = []
    threshold_rows = []
    evaluation_rows = []
    cv_entry_by_profile = {
        str(entry.get("profile", "")): entry
        for entry in [*profile_entries, *(cv_blend_entries or [])]
        if str(entry.get("fold_scope", "")) == "full" or str(entry.get("fold_scope", "")).startswith("blend_")
    }

    for entry in full_entries:
        scope_dir = Path(entry["scope_dir"])
        manifest = _read_json(scope_dir / "training_manifest.json")
        raw_predictions = _predict_holdout_for_profile(
            scope_dir=scope_dir,
            manifest=manifest,
            holdout_context=holdout_context,
            holdout_start=holdout_start,
            config=config,
        )
        predictions = _aggregate_holdout_predictions(raw_predictions, profile=str(entry["profile"]))
        if predictions.empty:
            continue
        diagnostics = summarize_profile_predictions(
            predictions,
            config,
            profile=str(entry["profile"]),
            feature_columns=list(entry["feature_columns"]),
            fold_scope="holdout_profile",
        )
        row = dict(diagnostics["row"])
        row["candidate"] = str(entry["profile"])
        row["candidate_type"] = "profile"
        row["source_profiles"] = str(entry["profile"])
        row["blend_method"] = ""
        row["blend_weights"] = ""
        cv_entry = cv_entry_by_profile.get(str(entry["profile"]))
        row = _attach_holdout_cv_threshold_metrics(row, predictions, cv_entry)
        row = _attach_holdout_policy_metrics(row, predictions, cv_entry, config)
        row = _attach_holdout_soft_pass(row, config)
        row = _attach_holdout_policy_consistency(row)
        evaluation_rows.append(row)
        bands = diagnostics["score_band_summary"].copy()
        if not bands.empty:
            bands.insert(0, "candidate", row["candidate"])
            score_band_rows.append(bands)
        thresholds = diagnostics["threshold_summary"].copy()
        if not thresholds.empty:
            thresholds.insert(0, "candidate", row["candidate"])
            threshold_rows.append(thresholds)
        holdout_entries.append(
            {
                "profile": str(entry["profile"]),
                "fold_scope": "holdout_profile",
                "feature_columns": list(entry["feature_columns"]),
                "predictions": predictions,
                "diagnostics": diagnostics,
                "summary": row,
                "config": entry.get("config", config),
            }
        )

    blend_source_entries = [{**entry, "fold_scope": "full"} for entry in holdout_entries]
    blend_entries = _profile_blend_entries(blend_source_entries, config)
    for entry in blend_entries:
        diagnostics = entry["diagnostics"]
        row = dict(diagnostics["row"])
        row["candidate"] = str(entry["profile"])
        row["candidate_type"] = "blend"
        row["source_profiles"] = row.get("blend_profiles", "")
        row["blend_method"] = row.get("blend_method", "")
        row["blend_weights"] = row.get("blend_weights", "")
        cv_entry = cv_entry_by_profile.get(str(entry["profile"]))
        row = _attach_holdout_cv_threshold_metrics(row, entry["predictions"], cv_entry)
        row = _attach_holdout_policy_metrics(row, entry["predictions"], cv_entry, config)
        row = _attach_holdout_soft_pass(row, config)
        row = _attach_holdout_policy_consistency(row)
        evaluation_rows.append(row)
        holdout_diagnostics = dict(diagnostics)
        holdout_diagnostics["row"] = row
        holdout_entries.append(
            {
                **entry,
                "fold_scope": str(entry.get("fold_scope", "")),
                "diagnostics": holdout_diagnostics,
                "summary": row,
            }
        )
        bands = diagnostics["score_band_summary"].copy()
        if not bands.empty:
            bands.insert(0, "candidate", row["candidate"])
            score_band_rows.append(bands)
        thresholds = diagnostics["threshold_summary"].copy()
        if not thresholds.empty:
            thresholds.insert(0, "candidate", row["candidate"])
            threshold_rows.append(thresholds)

    holdout_evaluation = pd.DataFrame(evaluation_rows)
    holdout_score_bands = pd.concat(score_band_rows, ignore_index=True) if score_band_rows else pd.DataFrame()
    holdout_thresholds = pd.concat(threshold_rows, ignore_index=True) if threshold_rows else pd.DataFrame()
    if holdout_evaluation.empty:
        holdout_decision = {
            "available": False,
            "reason": "no_holdout_predictions",
            "policy": "holdout result must remain separate from profile selection",
            "holdout_boundary_passed": bool(holdout_boundary_passed),
        }
        return holdout_evaluation, holdout_score_bands, holdout_thresholds, holdout_decision, holdout_entries

    available_candidates = set(holdout_evaluation["candidate"].astype(str))
    policy_review = _cfg(config, ["experiments", "policy_review"], {}) or {}
    configured_frozen = str(policy_review.get("frozen_candidate", "")).strip()
    configured_available = bool(configured_frozen and configured_frozen in available_candidates)
    frozen_selection = str(settings.get("control_profile", ""))
    frozen_selection_source = "control_profile"
    best_blend = decision.get("best_profile_blend") or {}
    best_candidate = decision.get("best_candidate") or {}
    if configured_available:
        frozen_selection = configured_frozen
        frozen_selection_source = "configured_policy_review"
    elif configured_frozen:
        frozen_selection_source = "configured_policy_review_missing_fallback_control_profile"
    elif best_blend:
        frozen_selection = str(best_blend.get("profile") or frozen_selection)
        frozen_selection_source = "best_profile_blend"
    elif best_candidate:
        frozen_selection = str(best_candidate.get("profile") or frozen_selection)
        frozen_selection_source = "best_candidate"
    holdout_evaluation["frozen_selection"] = holdout_evaluation["candidate"].astype(str).eq(frozen_selection)

    sortable = holdout_evaluation.copy()
    sortable["signal_pass_sort"] = sortable["holdout_signal_pass"].astype(bool).astype(int)
    sortable["threshold_pass_sort"] = sortable["holdout_threshold_pass"].astype(bool).astype(int)
    sortable = sortable.sort_values(
        [
            "signal_pass_sort",
            "threshold_pass_sort",
            "mean_rank_ic",
            "top_10_lift_global",
            "holdout_cv_threshold_f1",
        ],
        ascending=[False, False, False, False, False],
    )
    observed_best = sortable.iloc[0].to_dict()
    frozen_rows = holdout_evaluation.loc[holdout_evaluation["frozen_selection"].astype(bool)]
    frozen_row = frozen_rows.iloc[0].to_dict() if not frozen_rows.empty else {}
    policy_sortable = holdout_evaluation.copy()
    policy_sortable["policy_consistency_sort"] = policy_sortable["holdout_policy_consistency_pass"].astype(bool).astype(int)
    policy_sortable = policy_sortable.sort_values(
        [
            "policy_consistency_sort",
            "holdout_policy_forward_return",
            "holdout_policy_lift_vs_base",
            "mean_rank_ic",
        ],
        ascending=[False, False, False, False],
    )
    observed_best_policy = policy_sortable.iloc[0].to_dict()
    observed_best_name = str(observed_best.get("candidate", ""))
    observed_best_warning = ""
    if observed_best_name and observed_best_name != frozen_selection:
        observed_best_warning = (
            "Observed-best holdout candidate is diagnostic only; do not promote it "
            "or tune blend weights against this same reserved holdout."
        )
    observed_best_policy_name = str(observed_best_policy.get("candidate", ""))
    observed_best_policy_warning = ""
    if observed_best_policy_name and observed_best_policy_name != frozen_selection:
        observed_best_policy_warning = (
            "Observed-best holdout policy candidate is diagnostic only; keep the frozen "
            "pre-holdout selection unless a future out-of-sample window confirms it."
        )
    if frozen_row and bool(frozen_row.get("holdout_policy_consistency_pass", False)):
        score_policy_recommendation = "review_frozen_score_band_policy"
    elif bool(observed_best_policy.get("holdout_policy_consistency_pass", False)):
        score_policy_recommendation = "holdout_only_diagnostic_policy_candidate"
    else:
        score_policy_recommendation = "keep_control_profile"
    holdout_decision = {
        "available": True,
        "policy": "one_shot_final_validation; do not tune profiles or weights against this same holdout",
        "holdout_boundary_passed": bool(holdout_boundary_passed),
        "holdout_start": str(pd.to_datetime(holdout_start, utc=True)),
        "holdout_rows": int(len(holdout_context.loc[pd.to_datetime(holdout_context["timestamp"], utc=True) >= holdout_start])),
        "candidate_count": int(len(holdout_evaluation)),
        "frozen_selection": frozen_selection,
        "frozen_selection_source": frozen_selection_source,
        "configured_frozen_candidate_available": configured_available,
        "frozen_selection_metrics": _json_ready(frozen_row),
        "frozen_policy_validation": _json_ready(frozen_row),
        "observed_best_holdout_candidate": _json_ready(observed_best),
        "observed_best_holdout_warning": observed_best_warning,
        "observed_best_policy_candidate": _json_ready(observed_best_policy),
        "observed_best_policy_warning": observed_best_policy_warning,
        "score_policy_recommendation": score_policy_recommendation,
    }
    policy_validation = _holdout_policy_decision_frame(holdout_decision, config)
    holdout_decision["policy_validation"] = (
        _json_ready(policy_validation.iloc[0].to_dict())
        if not policy_validation.empty
        else {}
    )
    return holdout_evaluation, holdout_score_bands, holdout_thresholds, holdout_decision, holdout_entries


def _fold_delta_frame(entry: dict[str, Any]) -> pd.DataFrame:
    diagnostics = entry["diagnostics"]
    fold_metrics = diagnostics["fold_metrics"].copy()
    columns = ["fold", "rank_ic", "long_f1", "prauc"]
    optional = [column for column in ("start", "end") if column in fold_metrics.columns]
    frame = fold_metrics[["fold", *optional, *columns[1:]]].copy()
    score_lift = diagnostics["score_lift_by_fold"]
    if score_lift is not None and not score_lift.empty:
        lift_columns = [column for column in ("top_lift_vs_base", "top_minus_bottom_forward_return") if column in score_lift.columns]
        if lift_columns:
            frame = frame.merge(score_lift[["fold", *lift_columns]], on="fold", how="left")
    thresholds = diagnostics["threshold_metrics"]
    if thresholds is not None and not thresholds.empty and "test_f1_at_selected_threshold" in thresholds.columns:
        merge_columns = [
            column
            for column in (
                "fold",
                "test_f1_at_selected_threshold",
                "test_f1_at_constrained_threshold",
                "test_pred_long_rate_at_constrained_threshold",
            )
            if column in thresholds.columns
        ]
        frame = frame.merge(thresholds[merge_columns], on="fold", how="left")
    return frame


def _profile_delta_vs_control(entries: list[dict[str, Any]], control_profile: str) -> pd.DataFrame:
    columns = [
        "profile",
        "fold_scope",
        "fold",
        "start",
        "end",
        "control_rank_ic",
        "candidate_rank_ic",
        "rank_ic_delta",
        "control_top_10_lift",
        "candidate_top_10_lift",
        "top_10_lift_delta",
        "control_threshold_f1",
        "candidate_threshold_f1",
        "threshold_f1_delta",
        "control_prauc",
        "candidate_prauc",
        "prauc_delta",
        "control_long_f1",
        "candidate_long_f1",
        "long_f1_delta",
    ]
    controls = {
        str(entry["fold_scope"]): _fold_delta_frame(entry)
        for entry in entries
        if str(entry["profile"]) == control_profile
    }
    rows = []
    for entry in entries:
        profile = str(entry["profile"])
        fold_scope = str(entry["fold_scope"])
        if profile == control_profile or fold_scope not in controls:
            continue
        control = controls[fold_scope].copy()
        candidate = _fold_delta_frame(entry).copy()
        merged = control.merge(candidate, on="fold", how="inner", suffixes=("_control", "_candidate"))
        for _, row in merged.iterrows():
            start = row.get("start_candidate", row.get("start_control", ""))
            end = row.get("end_candidate", row.get("end_control", ""))
            control_top = _float(row.to_dict(), "top_lift_vs_base_control")
            candidate_top = _float(row.to_dict(), "top_lift_vs_base_candidate")
            control_threshold_f1 = _float(row.to_dict(), "test_f1_at_selected_threshold_control")
            candidate_threshold_f1 = _float(row.to_dict(), "test_f1_at_selected_threshold_candidate")
            control_prauc = _float(row.to_dict(), "prauc_control")
            candidate_prauc = _float(row.to_dict(), "prauc_candidate")
            control_long_f1 = _float(row.to_dict(), "long_f1_control")
            candidate_long_f1 = _float(row.to_dict(), "long_f1_candidate")
            control_rank_ic = _float(row.to_dict(), "rank_ic_control")
            candidate_rank_ic = _float(row.to_dict(), "rank_ic_candidate")
            rows.append(
                {
                    "profile": profile,
                    "fold_scope": fold_scope,
                    "fold": int(row["fold"]),
                    "start": start,
                    "end": end,
                    "control_rank_ic": control_rank_ic,
                    "candidate_rank_ic": candidate_rank_ic,
                    "rank_ic_delta": candidate_rank_ic - control_rank_ic,
                    "control_top_10_lift": control_top,
                    "candidate_top_10_lift": candidate_top,
                    "top_10_lift_delta": candidate_top - control_top,
                    "control_threshold_f1": control_threshold_f1,
                    "candidate_threshold_f1": candidate_threshold_f1,
                    "threshold_f1_delta": candidate_threshold_f1 - control_threshold_f1,
                    "control_prauc": control_prauc,
                    "candidate_prauc": candidate_prauc,
                    "prauc_delta": candidate_prauc - control_prauc,
                    "control_long_f1": control_long_f1,
                    "candidate_long_f1": candidate_long_f1,
                    "long_f1_delta": candidate_long_f1 - control_long_f1,
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["fold_scope", "profile", "fold"]).reset_index(drop=True)


def _write_profile_delta(path: Path, profile_delta: pd.DataFrame | None) -> None:
    if profile_delta is None:
        return
    profile_delta.to_csv(path / "profile_delta_vs_control.csv", index=False)


def _seed_audit_scope(seed: int) -> str:
    return f"seed_audit_seed_{int(seed):03d}"


def _seed_audit_entries_to_frames(entries: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if not fold_scope.startswith("seed_audit_seed_"):
            continue
        seed_text = fold_scope.rsplit("_", 1)[-1]
        try:
            seed = int(seed_text)
        except ValueError:
            seed = np.nan
        row = dict(entry["diagnostics"]["row"])
        row["seed"] = seed
        row["audit_scope"] = fold_scope
        rows.append(row)
    seed_audit = pd.DataFrame(rows).sort_values(["profile", "seed"]).reset_index(drop=True) if rows else pd.DataFrame()
    return seed_audit, _seed_stability_frame(seed_audit)


def _seed_stability_frame(seed_audit: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "profile",
        "seed_count",
        "feature_count",
        "fold_count",
        "mean_rank_ic_seed_mean",
        "mean_rank_ic_seed_std",
        "positive_ic_fraction_seed_mean",
        "positive_ic_fraction_seed_std",
        "std_rank_ic_seed_mean",
        "top_10_lift_global_seed_mean",
        "top_10_lift_global_seed_std",
        "test_f1_at_selected_threshold_seed_mean",
        "test_f1_at_selected_threshold_seed_std",
        "test_f1_at_constrained_threshold_seed_mean",
        "test_f1_at_constrained_threshold_seed_std",
        "test_pred_long_rate_at_constrained_threshold_seed_mean",
        "test_pred_long_rate_at_constrained_threshold_seed_std",
        "worst_5_rank_ic_mean_seed_mean",
        "worst_5_rank_ic_mean_seed_std",
    ]
    if seed_audit.empty:
        return pd.DataFrame(columns=columns)

    metric_columns = [
        "mean_rank_ic",
        "positive_ic_fraction",
        "std_rank_ic",
        "top_10_lift_global",
        "test_f1_at_selected_threshold",
        "test_f1_at_constrained_threshold",
        "test_pred_long_rate_at_constrained_threshold",
        "worst_5_rank_ic_mean",
    ]
    frame = seed_audit.copy()
    for column in metric_columns:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    grouped = frame.groupby("profile", as_index=False).agg(
        seed_count=("seed", "nunique"),
        feature_count=("feature_count", "first"),
        fold_count=("fold_count", "first"),
        mean_rank_ic_seed_mean=("mean_rank_ic", "mean"),
        mean_rank_ic_seed_std=("mean_rank_ic", "std"),
        positive_ic_fraction_seed_mean=("positive_ic_fraction", "mean"),
        positive_ic_fraction_seed_std=("positive_ic_fraction", "std"),
        std_rank_ic_seed_mean=("std_rank_ic", "mean"),
        top_10_lift_global_seed_mean=("top_10_lift_global", "mean"),
        top_10_lift_global_seed_std=("top_10_lift_global", "std"),
        test_f1_at_selected_threshold_seed_mean=("test_f1_at_selected_threshold", "mean"),
        test_f1_at_selected_threshold_seed_std=("test_f1_at_selected_threshold", "std"),
        test_f1_at_constrained_threshold_seed_mean=("test_f1_at_constrained_threshold", "mean"),
        test_f1_at_constrained_threshold_seed_std=("test_f1_at_constrained_threshold", "std"),
        test_pred_long_rate_at_constrained_threshold_seed_mean=("test_pred_long_rate_at_constrained_threshold", "mean"),
        test_pred_long_rate_at_constrained_threshold_seed_std=("test_pred_long_rate_at_constrained_threshold", "std"),
        worst_5_rank_ic_mean_seed_mean=("worst_5_rank_ic_mean", "mean"),
        worst_5_rank_ic_mean_seed_std=("worst_5_rank_ic_mean", "std"),
    )
    return grouped[columns].reset_index(drop=True)


def _seed_audit_markdown(seed_audit: pd.DataFrame, seed_stability: pd.DataFrame) -> str:
    lines = ["# Seed Audit", ""]
    if seed_audit.empty:
        lines.append("Seed audit was disabled or produced no completed runs.")
    else:
        display_cols = [
            "profile",
            "seed",
            "fold_count",
            "mean_rank_ic",
            "std_rank_ic",
            "positive_ic_fraction",
            "top_10_lift_global",
            "test_f1_at_selected_threshold",
            "test_f1_at_constrained_threshold",
            "test_pred_long_rate_at_constrained_threshold",
            "worst_5_rank_ic_mean",
        ]
        lines.extend(["## Per Seed", ""])
        visible = seed_audit[[column for column in display_cols if column in seed_audit.columns]].copy()
        lines.append("| " + " | ".join(visible.columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
        for _, row in visible.iterrows():
            lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")

    lines.extend(["", "## Stability", ""])
    if seed_stability.empty:
        lines.append("No stability summary available.")
    else:
        lines.append("| " + " | ".join(seed_stability.columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(seed_stability.columns)) + " |")
        for _, row in seed_stability.iterrows():
            lines.append("| " + " | ".join(str(row[column]) for column in seed_stability.columns) + " |")
    return "\n".join(lines)


def _write_seed_audit_files(path: Path, seed_audit: pd.DataFrame, seed_stability: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    seed_audit.to_csv(path / "seed_audit.csv", index=False)
    seed_stability.to_csv(path / "seed_stability.csv", index=False)
    (path / "seed_audit.md").write_text(_seed_audit_markdown(seed_audit, seed_stability), encoding="utf-8")
    _write_json(path / "seed_audit.json", {"rows": seed_audit.to_dict(orient="records")})
    _write_json(path / "seed_stability.json", {"rows": seed_stability.to_dict(orient="records")})


def _seed_from_scope(fold_scope: str) -> int | None:
    if not fold_scope.startswith("seed_audit_seed_"):
        return None
    seed_text = fold_scope.rsplit("_", 1)[-1]
    try:
        return int(seed_text)
    except ValueError:
        return None


def _seed_ensemble_predictions(seed_entries: list[dict[str, Any]]) -> pd.DataFrame:
    if len(seed_entries) < 2:
        return pd.DataFrame()

    frames = []
    seeds = []
    for entry in seed_entries:
        seed = _seed_from_scope(str(entry.get("fold_scope", "")))
        if seed is None:
            continue
        prediction = entry["predictions"].copy()
        prediction["_ensemble_seed"] = seed
        frames.append(prediction)
        seeds.append(seed)
    if len(frames) < 2:
        return pd.DataFrame()

    stacked = pd.concat(frames, ignore_index=True)
    required_keys = ["fold", "timestamp"]
    if "split" in stacked.columns:
        required_keys.insert(0, "split")
    if "source_row_position" in stacked.columns:
        required_keys.append("source_row_position")
    key_columns = [column for column in required_keys if column in stacked.columns]
    if not {"fold", "timestamp"}.issubset(key_columns):
        return pd.DataFrame()

    seed_count = len(set(seeds))
    grouped = stacked.groupby(key_columns, dropna=False)
    stats = grouped["prob_long"].agg(
        prob_long_ensemble="mean",
        prob_long_seed_std="std",
        prob_long_seed_min="min",
        prob_long_seed_max="max",
        ensemble_seed_count="count",
    ).reset_index()
    stats = stats.loc[stats["ensemble_seed_count"] == seed_count].copy()
    if stats.empty:
        return pd.DataFrame()

    base = grouped.first().reset_index()
    base = base.merge(stats, on=key_columns, how="inner")
    base["prob_long"] = base["prob_long_ensemble"]
    regime_columns = [column for column in stacked.columns if column.startswith("regime_prob_")]
    if regime_columns:
        regime_avg = grouped[regime_columns].mean().reset_index()
        base = base.drop(columns=[column for column in regime_columns if column in base.columns]).merge(
            regime_avg,
            on=key_columns,
            how="left",
        )
    base = base.drop(columns=["_ensemble_seed", "prob_long_ensemble"], errors="ignore")
    base["ensemble_seeds"] = ",".join(str(seed) for seed in sorted(set(seeds)))
    return base.sort_values(key_columns).reset_index(drop=True)


def _seed_ensemble_entries(entries: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        fold_scope = str(entry.get("fold_scope", ""))
        if _seed_from_scope(fold_scope) is None:
            continue
        grouped.setdefault(str(entry["profile"]), []).append(entry)

    ensemble_entries = []
    for profile, seed_entries in grouped.items():
        predictions = _seed_ensemble_predictions(seed_entries)
        if predictions.empty:
            continue
        feature_columns = list(seed_entries[0]["feature_columns"])
        diagnostics = summarize_profile_predictions(
            predictions,
            config,
            profile=profile,
            feature_columns=feature_columns,
            fold_scope="seed_ensemble",
        )
        ensemble_entries.append(
            {
                "profile": profile,
                "fold_scope": "seed_ensemble",
                "feature_columns": feature_columns,
                "predictions": predictions,
                "diagnostics": diagnostics,
                "summary": diagnostics["row"],
            }
        )
    return ensemble_entries


def _seed_ensemble_frame(entries: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for entry in entries:
        if str(entry.get("fold_scope", "")) != "seed_ensemble":
            continue
        row = dict(entry["diagnostics"]["row"])
        predictions = entry.get("predictions", pd.DataFrame())
        if isinstance(predictions, pd.DataFrame) and "ensemble_seed_count" in predictions.columns and not predictions.empty:
            row["seed_count"] = int(predictions["ensemble_seed_count"].max())
            row["prob_long_seed_std_mean"] = float(pd.to_numeric(predictions["prob_long_seed_std"], errors="coerce").mean())
            row["prob_long_seed_std_p90"] = float(pd.to_numeric(predictions["prob_long_seed_std"], errors="coerce").quantile(0.90))
            row["ensemble_seeds"] = str(predictions["ensemble_seeds"].iloc[0]) if "ensemble_seeds" in predictions.columns else ""
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()


def _seed_ensemble_markdown(seed_ensemble: pd.DataFrame) -> str:
    lines = ["# Seed Ensemble", ""]
    if seed_ensemble.empty:
        lines.append("No seed ensemble was produced.")
        return "\n".join(lines)
    display_cols = [
        "profile",
        "fold_scope",
        "seed_count",
        "fold_count",
        "mean_rank_ic",
        "std_rank_ic",
        "positive_ic_fraction",
        "top_10_lift_global",
        "test_f1_at_selected_threshold",
        "test_f1_at_constrained_threshold",
        "test_pred_long_rate_at_constrained_threshold",
        "prob_long_seed_std_mean",
        "prob_long_seed_std_p90",
    ]
    visible = seed_ensemble[[column for column in display_cols if column in seed_ensemble.columns]].copy()
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    return "\n".join(lines)


def _write_seed_ensemble_files(path: Path, seed_ensemble: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    seed_ensemble.to_csv(path / "seed_ensemble.csv", index=False)
    (path / "seed_ensemble.md").write_text(_seed_ensemble_markdown(seed_ensemble), encoding="utf-8")
    _write_json(path / "seed_ensemble.json", {"rows": seed_ensemble.to_dict(orient="records")})


def _prediction_key_columns(frame: pd.DataFrame) -> list[str]:
    keys = []
    if "split" in frame.columns:
        keys.append("split")
    for column in ("fold", "timestamp", "source_row_position"):
        if column in frame.columns:
            keys.append(column)
    return keys


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


def _profile_blend_markdown(profile_blend: pd.DataFrame) -> str:
    lines = ["# Profile Blend Diagnostics", ""]
    if profile_blend.empty:
        lines.append("No full-profile blends were produced.")
        return "\n".join(lines)
    display_cols = [
        "profile",
        "blend_method",
        "blend_weights",
        "profile_count",
        "fold_count",
        "mean_rank_ic",
        "std_rank_ic",
        "positive_ic_fraction",
        "mean_rank_ic_delta_vs_control",
        "std_rank_ic_delta_vs_control",
        "positive_ic_fraction_delta_vs_control",
        "top_10_lift_global",
        "top_10_lift_global_delta_vs_control",
        "test_f1_at_selected_threshold",
        "test_f1_at_constrained_threshold",
        "test_pred_long_rate_at_constrained_threshold",
        "reviewable",
        "review_reason",
        "balanced_eligible",
        "balanced_reason",
        "tail_lift_eligible",
        "tail_lift_reason",
        "stability_eligible",
        "stability_reason",
        "leader_roles",
        "prob_long_profile_std_mean",
        "prob_long_profile_std_p90",
        "blend_profiles",
    ]
    visible = profile_blend[[column for column in display_cols if column in profile_blend.columns]].copy()
    lines.append("| " + " | ".join(visible.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    return "\n".join(lines)


def _write_profile_blend_files(path: Path, profile_blend: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    profile_blend.to_csv(path / "profile_blend.csv", index=False)
    (path / "profile_blend.md").write_text(_profile_blend_markdown(profile_blend), encoding="utf-8")
    _write_json(path / "profile_blend.json", {"rows": profile_blend.to_dict(orient="records")})


def _write_profile_diagnostic_summaries(path: Path, entries: list[dict[str, Any]]) -> None:
    path.mkdir(parents=True, exist_ok=True)

    def tagged_frame(entry: dict[str, Any], key: str) -> pd.DataFrame:
        frame = entry["diagnostics"].get(key)
        if frame is None or frame.empty:
            return pd.DataFrame()
        out = frame.copy()
        out.insert(0, "fold_scope", str(entry["fold_scope"]))
        out.insert(0, "profile", str(entry["profile"]))
        return out

    for key, filename in [
        ("fold_metrics", "profile_fold_metrics.csv"),
        ("threshold_summary", "profile_threshold_summary.csv"),
        ("calibrated_threshold_summary", "profile_calibrated_threshold_summary.csv"),
        ("threshold_grid_summary", "profile_threshold_grid_summary.csv"),
        ("score_band_summary", "profile_score_band_summary.csv"),
        ("score_policy_grid", "profile_score_policy_grid.csv"),
        ("score_policy_selection", "profile_score_policy_selection.csv"),
        ("feature_groups", "profile_feature_groups.csv"),
    ]:
        frames = [tagged_frame(entry, key) for entry in entries]
        frames = [frame for frame in frames if not frame.empty]
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(path / filename, index=False)


def _write_experiment_bundle(
    *,
    output_dir: Path,
    run_id: str,
    report_dir: Path,
    zip_paths: list[str],
) -> tuple[Path, Path]:
    bundle_path = output_dir / f"phase1_experiment_bundle_{run_id}.zip"
    latest_path = output_dir / "phase1_latest_experiment_bundle.zip"
    summary_files = [
        "profile_comparison.csv",
        "profile_comparison.md",
        "profile_delta_vs_control.csv",
        "seed_audit.csv",
        "seed_audit.md",
        "seed_audit.json",
        "seed_stability.csv",
        "seed_stability.json",
        "seed_ensemble.csv",
        "seed_ensemble.md",
        "seed_ensemble.json",
        "profile_blend.csv",
        "profile_blend.md",
        "profile_blend.json",
        "profile_fold_metrics.csv",
        "profile_threshold_summary.csv",
        "profile_calibrated_threshold_summary.csv",
        "profile_threshold_grid_summary.csv",
        "profile_score_band_summary.csv",
        "profile_score_policy_grid.csv",
        "profile_score_policy_selection.csv",
        "profile_feature_groups.csv",
        "experiment_selection.csv",
        "experiment_selection.md",
        "experiment_selection.json",
        "missing_selected_profiles.csv",
        "missing_selected_profiles.md",
        "missing_selected_profiles.json",
        "holdout_reservation.csv",
        "holdout_reservation.md",
        "holdout_reservation.json",
        "holdout_boundary_audit.csv",
        "holdout_boundary_audit.md",
        "holdout_boundary_audit.json",
        "holdout_evaluation.csv",
        "holdout_evaluation.md",
        "holdout_evaluation.json",
        "holdout_score_band_summary.csv",
        "holdout_threshold_summary.csv",
        "holdout_policy_evaluation.csv",
        "holdout_policy_consistency.csv",
        "holdout_policy_consistency.md",
        "holdout_policy_consistency.json",
        "holdout_policy_decision.csv",
        "holdout_policy_decision.md",
        "holdout_policy_decision.json",
        "frozen_policy_robustness.csv",
        "frozen_policy_robustness.md",
        "frozen_policy_robustness.json",
        "frozen_policy_monitoring_plan.csv",
        "frozen_policy_monitoring_plan.md",
        "frozen_policy_monitoring_plan.json",
        "experiment_policy_guard.csv",
        "experiment_policy_guard.md",
        "experiment_policy_guard.json",
        "future_oos_candidate_plan.csv",
        "future_oos_candidate_plan.md",
        "future_oos_candidate_plan.json",
        "phase1_blocker_action_plan.csv",
        "phase1_blocker_action_plan.md",
        "phase1_blocker_action_plan.json",
        "performance_gap_analysis.csv",
        "performance_gap_analysis.md",
        "performance_gap_analysis.json",
        "fold_stability_forensics.csv",
        "fold_stability_forensics.md",
        "fold_stability_forensics.json",
        "fold_stability_summary.csv",
        "fold_stability_summary.md",
        "fold_stability_summary.json",
        "score_separation_forensics.csv",
        "bad_fold_signature.csv",
        "bad_fold_signature.md",
        "bad_fold_signature.json",
        "feature_drift_forensics.csv",
        "feature_family_drift_summary.csv",
        "feature_drift_forensics.md",
        "feature_drift_forensics.json",
        "probability_quality_forensics.csv",
        "probability_quality_summary.csv",
        "probability_quality_forensics.md",
        "probability_quality_forensics.json",
        "score_distribution_shift.csv",
        "score_distribution_shift_summary.csv",
        "score_distribution_shift.md",
        "score_distribution_shift.json",
        "fold_reliability_gate.csv",
        "fold_reliability_gate_summary.csv",
        "fold_reliability_gate.md",
        "fold_reliability_gate.json",
        "regime_threshold_policy_by_fold.csv",
        "regime_threshold_policy_summary.csv",
        "regime_threshold_policy.md",
        "regime_threshold_policy.json",
        "regime_stability_forensics.csv",
        "regime_stability_summary.csv",
        "regime_stability.md",
        "regime_stability.json",
        "threshold_forensics.csv",
        "threshold_forensics.md",
        "threshold_forensics.json",
        "threshold_policy_review.csv",
        "threshold_policy_review.md",
        "threshold_policy_review.json",
        "threshold_transfer_review.csv",
        "threshold_transfer_review.md",
        "threshold_transfer_review.json",
        "threshold_transfer_by_fold.csv",
        "payoff_alignment.csv",
        "payoff_alignment_summary.csv",
        "payoff_alignment.md",
        "payoff_alignment.json",
        "payoff_policy_robustness.csv",
        "payoff_policy_robustness_summary.csv",
        "payoff_policy_robustness.md",
        "payoff_policy_robustness.json",
        "training_execution_summary.json",
        "auto_review.md",
        "auto_review.json",
        "next_actions.json",
        "phase2_readiness.md",
        "phase2_readiness.json",
        "phase1_transition_plan.md",
        "phase1_transition_plan.json",
        "decision_report.json",
        "best_candidate.json",
    ]
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in summary_files:
            path = report_dir / filename
            if path.exists():
                archive.write(path, f"{run_id}/{filename}")
        for item in zip_paths:
            path = Path(item)
            if path.exists():
                archive.write(path, f"{run_id}/diagnostics/{path.name}")
    shutil.copyfile(bundle_path, latest_path)
    return bundle_path, latest_path


def _write_experiment_slim_bundle(*, output_dir: Path, run_id: str, report_dir: Path) -> tuple[Path, Path]:
    slim_path = output_dir / f"phase1_experiment_slim_bundle_{run_id}.zip"
    latest_slim_path = output_dir / "phase1_latest_experiment_slim_bundle.zip"
    with zipfile.ZipFile(slim_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(report_dir.glob("*")):
            if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md"}:
                archive.write(path, f"{run_id}/{path.name}")
    shutil.copyfile(slim_path, latest_slim_path)
    return slim_path, latest_slim_path


def run_experiment_matrix(
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    checkpoint_dir: str | Path,
    run_id: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    settings = experiment_settings(config)
    settings = _resolve_holdout_settings(settings, config)
    settings = _apply_experiment_policy_guard(settings, config)
    frame = _selection_frame_before_holdout(frame, settings)
    settings = _preflight_experiment_profiles(settings, frame, config)
    settings = _apply_experiment_policy_guard(settings, config)
    signature = _experiment_signature(config, settings)
    signature_hash = _hash_payload(signature)
    run_id, run_id_source = resolve_experiment_run_id(checkpoint_dir, config, settings, run_id)
    run_dir = experiment_root(checkpoint_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "experiment_manifest.json",
        {
            "run_id": run_id,
            "run_id_source": run_id_source,
            "signature": signature,
            "signature_hash": signature_hash,
            "settings": settings,
        },
    )
    experiment_selection = _write_experiment_selection(run_dir, settings)
    holdout_reservation = _write_holdout_reservation(run_dir, settings)
    experiment_policy_guard = _experiment_policy_guard_frame(settings, config)
    _write_experiment_policy_guard(run_dir, experiment_policy_guard)
    future_oos_candidate_plan = _future_oos_candidate_plan_frame(settings, config)
    _write_future_oos_candidate_plan(run_dir, future_oos_candidate_plan)

    triage_fold_ids = [int(fold_id) for fold_id in settings.get("triage_fold_ids", [])]
    resume_existing = bool(settings.get("resume_existing", True))
    force_retrain = bool(settings.get("force_retrain", False))
    rows: list[dict[str, Any]] = []
    profile_results = []
    for profile in settings["profiles"]:
        result = run_profile_experiment(
            frame,
            config,
            profile=profile,
            checkpoint_dir=checkpoint_dir,
            run_id=run_id,
            fold_scope="triage",
            fold_ids=triage_fold_ids or None,
            resume_existing=resume_existing,
            force_retrain=force_retrain,
            device=device,
        )
        profile_results.append(result)
        rows.append(result["summary"])

    triage_rows = _decision_rows(rows, config, scope="triage")
    full_profiles_setting = settings.get("full_cv_profiles", "auto")
    if full_profiles_setting == "auto":
        full_profiles = _auto_full_profiles(settings, triage_rows)
    else:
        full_profiles = [str(profile) for profile in full_profiles_setting]

    full_rows = []
    for profile in dict.fromkeys(full_profiles):
        result = run_profile_experiment(
            frame,
            config,
            profile=profile,
            checkpoint_dir=checkpoint_dir,
            run_id=run_id,
            fold_scope="full",
            fold_ids=None,
            resume_existing=resume_existing,
            force_retrain=force_retrain,
            device=device,
        )
        profile_results.append(result)
        full_rows.append(result["summary"])

    full_rows = _decision_rows(full_rows, config, scope="full") if full_rows else []
    comparison = _comparison_frame([*triage_rows, *full_rows])
    seed_results: list[dict[str, Any]] = []
    seed_audit_cfg = settings.get("seed_audit", {}) or {}
    if bool(seed_audit_cfg.get("enabled", False)):
        audit_profiles = [str(profile) for profile in seed_audit_cfg.get("profiles", []) or [settings["control_profile"]]]
        audit_seeds = [int(seed) for seed in seed_audit_cfg.get("seeds", []) or []]
        audit_fold_ids = seed_audit_cfg.get("fold_ids", triage_fold_ids)
        audit_fold_ids = [int(fold_id) for fold_id in audit_fold_ids] if audit_fold_ids else None
        for profile in audit_profiles:
            for seed in audit_seeds:
                seed_cfg = copy.deepcopy(config)
                _set_cfg(seed_cfg, ["project", "random_seed"], seed)
                result = run_profile_experiment(
                    frame,
                    seed_cfg,
                    profile=profile,
                    checkpoint_dir=checkpoint_dir,
                    run_id=run_id,
                    fold_scope=_seed_audit_scope(seed),
                    fold_ids=audit_fold_ids,
                    resume_existing=resume_existing,
                    force_retrain=force_retrain,
                    device=device,
                )
                result["summary"]["seed"] = seed
                seed_results.append(result)
    seed_ensemble_results = _seed_ensemble_entries(seed_results, config)
    profile_blend_results = _profile_blend_entries(profile_results, config)
    all_results = [*profile_results, *seed_results, *seed_ensemble_results, *profile_blend_results]
    executed_results = [result for result in [*profile_results, *seed_results] if not bool(result.get("skipped", False))]
    skipped_results = [result for result in [*profile_results, *seed_results] if bool(result.get("skipped", False))]
    seed_audit, seed_stability = _seed_audit_entries_to_frames(all_results)
    seed_ensemble = _seed_ensemble_frame(all_results)
    profile_blend = _profile_blend_frame(all_results)
    profile_blend = _profile_blend_review_frame(profile_blend, comparison, config, settings["control_profile"])
    performance_gap_analysis = _performance_gap_analysis_frame(
        all_results,
        pd.DataFrame(),
        config,
        settings,
    )
    fold_stability_forensics = _fold_stability_forensics_frame(all_results, config)
    fold_stability_summary = _fold_stability_summary_frame(fold_stability_forensics, config)
    score_separation_forensics = _score_separation_forensics_frame(all_results, config)
    bad_fold_signature = _bad_fold_signature_frame(score_separation_forensics, config)
    feature_drift_forensics = _feature_drift_forensics_frame(all_results, score_separation_forensics, config)
    feature_family_drift_summary = _feature_family_drift_summary_frame(feature_drift_forensics)
    probability_quality_forensics = _probability_quality_forensics_frame(all_results, config)
    probability_quality_summary = _probability_quality_summary_frame(probability_quality_forensics, config)
    score_distribution_shift = _score_distribution_shift_frame(all_results, config)
    score_distribution_shift_summary = _score_distribution_shift_summary_frame(score_distribution_shift, config)
    fold_reliability_gate = _fold_reliability_gate_frame(all_results, config)
    fold_reliability_gate_summary = _fold_reliability_gate_summary_frame(fold_reliability_gate, config)
    regime_threshold_policy_by_fold, regime_threshold_policy_summary = _regime_threshold_policy_frames(
        all_results,
        config,
    )
    regime_stability_forensics, regime_stability_summary = _regime_stability_frames(all_results, config)
    threshold_forensics = _threshold_forensics_frame(all_results, config)
    threshold_policy_review = _threshold_policy_review_frame(all_results, config)
    threshold_transfer_review, threshold_transfer_by_fold = _threshold_transfer_review_frames(all_results, config)
    payoff_alignment = _payoff_alignment_frame(all_results, [], config)
    payoff_alignment_summary = _payoff_alignment_summary_frame(payoff_alignment)
    payoff_policy_robustness = _payoff_policy_robustness_frame(all_results, [], config)
    payoff_policy_robustness_summary = _payoff_policy_robustness_summary_frame(payoff_policy_robustness, config)
    future_oos_candidate_plan = _future_oos_candidate_plan_frame(
        settings,
        config,
        payoff_policy_robustness_summary,
    )
    phase1_blocker_action_plan = _phase1_blocker_action_plan_frame(
        comparison=comparison,
        profile_blend=profile_blend,
        performance_gap_analysis=performance_gap_analysis,
        fold_stability_forensics=fold_stability_forensics,
        fold_stability_summary=fold_stability_summary,
        threshold_forensics=threshold_forensics,
        payoff_policy_robustness_summary=payoff_policy_robustness_summary,
        future_oos_candidate_plan=future_oos_candidate_plan,
        phase2_readiness={},
        config=config,
        settings=settings,
    )
    _write_future_oos_candidate_plan(run_dir, future_oos_candidate_plan)
    _write_seed_audit_files(run_dir, seed_audit, seed_stability)
    _write_seed_ensemble_files(run_dir, seed_ensemble)
    _write_profile_blend_files(run_dir, profile_blend)
    _write_performance_gap_analysis(run_dir, performance_gap_analysis)
    _write_phase1_blocker_action_plan(run_dir, phase1_blocker_action_plan)
    _write_forensics_reports(
        run_dir,
        fold_stability_forensics=fold_stability_forensics,
        fold_stability_summary=fold_stability_summary,
        threshold_forensics=threshold_forensics,
    )
    _write_score_separation_forensics(run_dir, score_separation_forensics, bad_fold_signature)
    _write_feature_drift_forensics(run_dir, feature_drift_forensics, feature_family_drift_summary)
    _write_probability_quality_forensics(run_dir, probability_quality_forensics, probability_quality_summary)
    _write_score_distribution_shift(run_dir, score_distribution_shift, score_distribution_shift_summary)
    _write_fold_reliability_gate(run_dir, fold_reliability_gate, fold_reliability_gate_summary)
    _write_regime_threshold_policy(run_dir, regime_threshold_policy_by_fold, regime_threshold_policy_summary)
    _write_regime_stability(run_dir, regime_stability_forensics, regime_stability_summary)
    _write_threshold_policy_review(run_dir, threshold_policy_review)
    _write_threshold_transfer_review(run_dir, threshold_transfer_review, threshold_transfer_by_fold)
    _write_payoff_alignment(run_dir, payoff_alignment, payoff_alignment_summary)
    _write_payoff_policy_robustness(run_dir, payoff_policy_robustness, payoff_policy_robustness_summary)
    profile_delta = _profile_delta_vs_control(profile_results, settings["control_profile"])
    best = _best_candidate(comparison, settings["control_profile"])
    blend_leaders = _profile_blend_leaders(profile_blend)
    best_blend = _best_profile_blend(profile_blend)
    missing_selected = _missing_selected_profiles(experiment_selection, comparison)
    _write_missing_selected_profiles(run_dir, missing_selected)
    training_execution = _training_execution_summary(
        run_id=run_id,
        run_id_source=run_id_source,
        executed_results=executed_results,
        skipped_results=skipped_results,
        profile_results=profile_results,
        seed_results=seed_results,
    )
    _write_json(_training_execution_summary_path(run_dir), training_execution)
    recommendation = "fix_missing_selected_profiles" if not missing_selected.empty else (
        "promote_best_candidate" if best else ("review_profile_blend" if best_blend else "keep_control_profile")
    )
    decision = {
        "run_id": run_id,
        "control_profile": settings["control_profile"],
        "best_candidate": best,
        "best_profile_blend": best_blend,
        "profile_blend_leaders": blend_leaders,
        "full_profiles": full_profiles,
        "seed_audit_profiles": [str(profile) for profile in seed_audit_cfg.get("profiles", [])] if seed_audit_cfg else [],
        "seed_audit_seeds": [int(seed) for seed in seed_audit_cfg.get("seeds", [])] if seed_audit_cfg else [],
        "skipped_profiles": settings.get("skipped_profiles", []) or [],
        **{key: training_execution[key] for key in _TRAINING_EXECUTION_KEYS},
        "executed_training_scopes": training_execution["executed_training_scopes"],
        "training_execution_metadata_source": "run_experiment_matrix",
        "training_execution_metadata_available": True,
        "missing_selected_profiles": missing_selected.to_dict(orient="records"),
        "experiment_complete": bool(missing_selected.empty),
        "holdout": settings.get("holdout", {}) or {},
        "experiment_policy_guard": settings.get("experiment_policy_guard", {}) or {},
        "future_oos_candidate_plan": future_oos_candidate_plan.to_dict(orient="records"),
        "performance_gap_analysis": performance_gap_analysis.to_dict(orient="records"),
        "phase1_blocker_action_plan": phase1_blocker_action_plan.to_dict(orient="records"),
        "fold_stability_summary": fold_stability_summary.to_dict(orient="records"),
        "score_separation_forensics": score_separation_forensics.to_dict(orient="records"),
        "bad_fold_signature": bad_fold_signature.to_dict(orient="records"),
        "feature_family_drift_summary": feature_family_drift_summary.to_dict(orient="records"),
        "probability_quality_summary": probability_quality_summary.to_dict(orient="records"),
        "score_distribution_shift_summary": score_distribution_shift_summary.to_dict(orient="records"),
        "fold_reliability_gate_summary": fold_reliability_gate_summary.to_dict(orient="records"),
        "regime_threshold_policy_summary": regime_threshold_policy_summary.to_dict(orient="records"),
        "regime_stability_summary": regime_stability_summary.to_dict(orient="records"),
        "threshold_forensics_summary": (
            threshold_forensics["primary_issue"].value_counts().to_dict()
            if not threshold_forensics.empty and "primary_issue" in threshold_forensics.columns
            else {}
        ),
        "threshold_policy_review": threshold_policy_review.to_dict(orient="records"),
        "threshold_transfer_review": threshold_transfer_review.to_dict(orient="records"),
        "payoff_alignment_summary": payoff_alignment_summary.to_dict(orient="records"),
        "payoff_policy_robustness_summary": payoff_policy_robustness_summary.to_dict(orient="records"),
        "recommendation": _recommendation_with_policy_guard(recommendation, settings),
    }
    _write_decision_files(run_dir, comparison, decision)
    _write_profile_delta(run_dir, profile_delta)
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "profile_results": all_results,
        "comparison": comparison,
        "profile_delta": profile_delta,
        "seed_audit": seed_audit,
        "seed_stability": seed_stability,
        "seed_ensemble": seed_ensemble,
        "profile_blend": profile_blend,
        "performance_gap_analysis": performance_gap_analysis,
        "phase1_blocker_action_plan": phase1_blocker_action_plan,
        "fold_stability_forensics": fold_stability_forensics,
        "fold_stability_summary": fold_stability_summary,
        "score_separation_forensics": score_separation_forensics,
        "bad_fold_signature": bad_fold_signature,
        "feature_drift_forensics": feature_drift_forensics,
        "feature_family_drift_summary": feature_family_drift_summary,
        "probability_quality_forensics": probability_quality_forensics,
        "probability_quality_summary": probability_quality_summary,
        "score_distribution_shift": score_distribution_shift,
        "score_distribution_shift_summary": score_distribution_shift_summary,
        "fold_reliability_gate": fold_reliability_gate,
        "fold_reliability_gate_summary": fold_reliability_gate_summary,
        "regime_threshold_policy_by_fold": regime_threshold_policy_by_fold,
        "regime_threshold_policy_summary": regime_threshold_policy_summary,
        "regime_stability_forensics": regime_stability_forensics,
        "regime_stability_summary": regime_stability_summary,
        "threshold_forensics": threshold_forensics,
        "threshold_policy_review": threshold_policy_review,
        "threshold_transfer_review": threshold_transfer_review,
        "threshold_transfer_by_fold": threshold_transfer_by_fold,
        "payoff_alignment": payoff_alignment,
        "payoff_alignment_summary": payoff_alignment_summary,
        "payoff_policy_robustness": payoff_policy_robustness,
        "payoff_policy_robustness_summary": payoff_policy_robustness_summary,
        "experiment_selection": experiment_selection,
        "holdout_reservation": holdout_reservation,
        "experiment_policy_guard": experiment_policy_guard,
        "future_oos_candidate_plan": future_oos_candidate_plan,
        **{key: training_execution[key] for key in _TRAINING_EXECUTION_KEYS},
        "executed_training_scopes": training_execution["executed_training_scopes"],
        "training_execution_metadata_source": "run_experiment_matrix",
        "training_execution_metadata_available": True,
        "missing_selected_profiles": missing_selected,
        "decision": decision,
    }


def _profile_dirs(run_dir: Path) -> list[Path]:
    paths = []
    for profile_dir in run_dir.iterdir():
        if not profile_dir.is_dir():
            continue
        for scope_dir in profile_dir.iterdir():
            if (scope_dir / "training_manifest.json").exists() and (scope_dir / "predictions_all.parquet").exists():
                paths.append(scope_dir)
    return sorted(paths)


def write_experiment_diagnostics(
    *,
    checkpoint_dir: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
    run_id: str | None = None,
    write_full_bundles: bool | None = None,
) -> dict[str, Any]:
    if write_full_bundles is None:
        write_full_bundles = bool(
            _cfg(config, ["experiments", "diagnostics", "write_full_bundles"], default=False)
        )
    run_dir = experiment_root(checkpoint_dir) / run_id if run_id else latest_experiment_run(checkpoint_dir)
    run_manifest_path = run_dir / "experiment_manifest.json"
    run_manifest = _read_json(run_manifest_path) if run_manifest_path.exists() else {}
    training_execution = _load_training_execution_summary(run_dir, run_manifest)
    settings = copy.deepcopy(run_manifest.get("settings") or experiment_settings(config))
    settings = _resolve_holdout_settings(settings, config)
    diagnostic_config = copy.deepcopy(config)
    current_experiment_cfg = copy.deepcopy(_cfg(diagnostic_config, ["experiments"], default={}) or {})
    experiment_cfg = copy.deepcopy(current_experiment_cfg)
    experiment_cfg.update(settings)
    # Training run settings are historical metadata, but policy review is a
    # diagnostics-time decision. Keep it sourced from the current config so a
    # failed/retired frozen policy cannot be resurrected by an old run manifest.
    if "policy_review" in current_experiment_cfg:
        experiment_cfg["policy_review"] = copy.deepcopy(current_experiment_cfg["policy_review"])
    _set_cfg(diagnostic_config, ["experiments"], experiment_cfg)
    settings = _apply_experiment_policy_guard(settings, diagnostic_config)
    scope_dirs = _profile_dirs(run_dir)
    if not scope_dirs:
        root = experiment_root(checkpoint_dir)
        recent_runs = sorted([path.name for path in root.glob("*") if path.is_dir()], reverse=True)[:8]
        hint = (
            f"No completed profile runs found under {run_dir}.\n"
            "This usually means notebook `04_training_walk_forward.ipynb` has not finished (or wrote to a different CHECKPT_DIR).\n"
            f"Expected files like: {run_dir}/<profile>/<fold_scope>/training_manifest.json and predictions_all.parquet.\n"
            f"Recent experiment run directories: {recent_runs}"
        )
        raise FileNotFoundError(hint)

    entries = []
    for scope_dir in scope_dirs:
        manifest = _read_json(scope_dir / "training_manifest.json")
        profile = str(manifest["profile"])
        fold_scope = str(manifest["fold_scope"])
        feature_columns = list(manifest["feature_columns"])
        predictions = pd.read_parquet(scope_dir / "predictions_all.parquet")
        diagnostics = summarize_profile_predictions(
            predictions,
            diagnostic_config,
            profile=profile,
            feature_columns=feature_columns,
            fold_scope=fold_scope,
        )
        entries.append(
            {
                "scope_dir": scope_dir,
                "profile": profile,
                "fold_scope": fold_scope,
                "feature_columns": feature_columns,
                "predictions": predictions,
                "diagnostics": diagnostics,
            }
        )
    profile_entries = list(entries)
    seed_ensemble_entries = _seed_ensemble_entries(profile_entries, diagnostic_config)
    profile_blend_entries = _profile_blend_entries(profile_entries, diagnostic_config)
    entries = [*profile_entries, *seed_ensemble_entries, *profile_blend_entries]

    rows = [entry["diagnostics"]["row"] for entry in entries]
    triage_rows = _decision_rows(
        [row for row in rows if row.get("fold_scope") == "triage"],
        diagnostic_config,
        scope="triage",
    )
    full_rows = _decision_rows(
        [row for row in rows if row.get("fold_scope") == "full"],
        diagnostic_config,
        scope="full",
    )
    comparison = _comparison_frame([*triage_rows, *full_rows])
    profile_delta = _profile_delta_vs_control(profile_entries, settings["control_profile"])
    seed_audit, seed_stability = _seed_audit_entries_to_frames(entries)
    seed_ensemble = _seed_ensemble_frame(entries)
    profile_blend = _profile_blend_frame(entries)
    profile_blend = _profile_blend_review_frame(profile_blend, comparison, diagnostic_config, settings["control_profile"])
    holdout_boundary_audit = _holdout_boundary_audit_frame(entries, settings)
    holdout_boundary_passed = bool(holdout_boundary_audit.empty or holdout_boundary_audit["passed"].astype(bool).all())
    frozen_policy_robustness = _frozen_policy_robustness_frame(entries, diagnostic_config)
    frozen_policy_monitoring_plan = _frozen_policy_monitoring_plan_frame(diagnostic_config, settings)
    experiment_policy_guard = _experiment_policy_guard_frame(settings, diagnostic_config)
    future_oos_candidate_plan = _future_oos_candidate_plan_frame(settings, diagnostic_config)
    decision_lookup = {
        (str(row["profile"]), str(row["fold_scope"])): row
        for row in [*triage_rows, *full_rows]
    }

    zip_paths = []
    if write_full_bundles:
        for entry in entries:
            profile = str(entry["profile"])
            fold_scope = str(entry["fold_scope"])
            feature_columns = list(entry["feature_columns"])
            predictions = entry["predictions"]
            diagnostics = entry["diagnostics"]
            entry_config = entry.get("config", config)
            decided = decision_lookup.get((profile, fold_scope), {})
            ledger = diagnostics["ledger"].copy()
            for column in ("promotable", "reject_reason"):
                if column in decided:
                    ledger.loc[:, column] = decided[column]
            zip_path = write_phase1_diagnostic_bundle(
                output_dir=Path(output_dir) / "experiments" / run_dir.name / _slug(profile) / fold_scope,
                report=diagnostics["report"],
                predictions=_test_predictions(predictions),
                calibration=diagnostics["calibration"],
                calibrated_report=diagnostics.get("calibrated_report"),
                calibrated_calibration=diagnostics.get("calibrated_calibration"),
                calibrated_predictions=diagnostics.get("calibrated_predictions"),
                fold_metrics=diagnostics["fold_metrics"],
                regime_metrics=diagnostics["regime_metrics"],
                regime_by_fold=diagnostics["regime_by_fold"],
                bad_fold_regime=diagnostics["bad_fold_regime"],
                threshold_metrics=diagnostics["threshold_metrics"],
                threshold_summary=diagnostics["threshold_summary"],
                mtf_leakage=diagnostics["mtf_leakage"],
                stationarity_policy=diagnostics["stationarity_policy"],
                score_lift=diagnostics["score_lift"],
                score_lift_by_fold=diagnostics["score_lift_by_fold"],
                score_band_lift=diagnostics["score_band_lift"],
                score_band_by_fold=diagnostics["score_band_by_fold"],
                score_band_summary=diagnostics["score_band_summary"],
                recent_fold_summary=diagnostics["recent_fold_summary"],
                feature_groups=diagnostics["feature_groups"],
                feature_profile=diagnostics["feature_profile"],
                experiment_ledger=ledger,
                model_feature_columns=feature_columns,
                config=profile_config(entry_config, profile),
                prefix=f"phase1_diagnostics_{_slug(profile)}_{fold_scope}",
            )
            zip_paths.append(str(zip_path))

    best = _best_candidate(comparison, settings["control_profile"])
    blend_leaders = _profile_blend_leaders(profile_blend)
    best_blend = _best_profile_blend(profile_blend)
    report_dir = Path(output_dir) / "experiments" / run_dir.name
    report_dir.mkdir(parents=True, exist_ok=True)
    experiment_selection = _write_experiment_selection(report_dir, settings)
    holdout_reservation = _write_holdout_reservation(report_dir, settings)
    missing_selected = _missing_selected_profiles(experiment_selection, comparison)
    _write_missing_selected_profiles(report_dir, missing_selected)
    if not missing_selected.empty:
        recommendation = "fix_missing_selected_profiles"
    elif not holdout_boundary_passed:
        recommendation = "rerun_training_with_holdout_split"
    elif best:
        recommendation = "promote_best_candidate"
    elif best_blend:
        recommendation = "review_profile_blend"
    else:
        recommendation = "keep_control_profile"
    guarded_recommendation = _recommendation_with_policy_guard(recommendation, settings)
    decision = {
        "run_id": run_dir.name,
        "control_profile": settings["control_profile"],
        "best_candidate": best,
        "best_profile_blend": best_blend,
        "profile_blend_leaders": blend_leaders,
        "full_profiles": [str(profile) for profile in settings.get("always_full_profiles", [])],
        "seed_audit_profiles": [
            str(profile) for profile in (settings.get("seed_audit", {}) or {}).get("profiles", [])
        ],
        "seed_audit_seeds": [int(seed) for seed in (settings.get("seed_audit", {}) or {}).get("seeds", [])],
        "skipped_profiles": settings.get("skipped_profiles", []) or [],
        **{key: training_execution.get(key) for key in _TRAINING_EXECUTION_KEYS},
        "executed_training_scopes": training_execution.get("executed_training_scopes", []),
        "training_execution_metadata_source": training_execution.get("training_execution_metadata_source"),
        "training_execution_metadata_available": bool(training_execution.get("training_execution_metadata_available", False)),
        "missing_selected_profiles": missing_selected.to_dict(orient="records"),
        "experiment_complete": bool(missing_selected.empty),
        "holdout_boundary_passed": holdout_boundary_passed,
        "holdout_boundary_audit": holdout_boundary_audit.to_dict(orient="records"),
        "frozen_policy_robustness": frozen_policy_robustness.to_dict(orient="records"),
        "frozen_policy_monitoring_plan": frozen_policy_monitoring_plan.to_dict(orient="records"),
        "experiment_policy_guard": settings.get("experiment_policy_guard", {}) or {},
        "future_oos_candidate_plan": future_oos_candidate_plan.to_dict(orient="records"),
        "holdout": settings.get("holdout", {}) or {},
        "recommendation": guarded_recommendation,
        "diagnostic_zips": zip_paths,
    }
    holdout_evaluation, holdout_score_bands, holdout_thresholds, holdout_decision, holdout_entries = (
        _evaluate_holdout_candidates(
            profile_entries=profile_entries,
            cv_blend_entries=profile_blend_entries,
            settings=settings,
            config=diagnostic_config,
            decision=decision,
            holdout_boundary_passed=holdout_boundary_passed,
        )
    )
    decision["holdout_evaluation"] = holdout_decision
    decision["holdout_evaluation_available"] = bool(holdout_decision.get("available", False))
    performance_gap_analysis = _performance_gap_analysis_frame(
        entries,
        holdout_evaluation,
        diagnostic_config,
        settings,
    )
    fold_stability_forensics = _fold_stability_forensics_frame(entries, diagnostic_config)
    fold_stability_summary = _fold_stability_summary_frame(fold_stability_forensics, diagnostic_config)
    score_separation_forensics = _score_separation_forensics_frame(entries, diagnostic_config)
    bad_fold_signature = _bad_fold_signature_frame(score_separation_forensics, diagnostic_config)
    feature_drift_forensics = _feature_drift_forensics_frame(entries, score_separation_forensics, diagnostic_config)
    feature_family_drift_summary = _feature_family_drift_summary_frame(feature_drift_forensics)
    probability_quality_forensics = _probability_quality_forensics_frame(entries, diagnostic_config)
    probability_quality_summary = _probability_quality_summary_frame(probability_quality_forensics, diagnostic_config)
    score_distribution_shift = _score_distribution_shift_frame(entries, diagnostic_config)
    score_distribution_shift_summary = _score_distribution_shift_summary_frame(score_distribution_shift, diagnostic_config)
    fold_reliability_gate = _fold_reliability_gate_frame(entries, diagnostic_config)
    fold_reliability_gate_summary = _fold_reliability_gate_summary_frame(fold_reliability_gate, diagnostic_config)
    regime_threshold_policy_by_fold, regime_threshold_policy_summary = _regime_threshold_policy_frames(
        entries,
        diagnostic_config,
    )
    regime_stability_forensics, regime_stability_summary = _regime_stability_frames(entries, diagnostic_config)
    threshold_forensics = _threshold_forensics_frame(entries, diagnostic_config)
    threshold_policy_review = _threshold_policy_review_frame(entries, diagnostic_config)
    threshold_transfer_review, threshold_transfer_by_fold = _threshold_transfer_review_frames(entries, diagnostic_config)
    payoff_alignment = _payoff_alignment_frame(entries, holdout_entries, diagnostic_config)
    payoff_alignment_summary = _payoff_alignment_summary_frame(payoff_alignment)
    payoff_policy_robustness = _payoff_policy_robustness_frame(entries, holdout_entries, diagnostic_config)
    payoff_policy_robustness_summary = _payoff_policy_robustness_summary_frame(payoff_policy_robustness, diagnostic_config)
    future_oos_candidate_plan = _future_oos_candidate_plan_frame(
        settings,
        diagnostic_config,
        payoff_policy_robustness_summary,
    )
    decision["future_oos_candidate_plan"] = future_oos_candidate_plan.to_dict(orient="records")
    decision["performance_gap_analysis"] = performance_gap_analysis.to_dict(orient="records")
    decision["fold_stability_summary"] = fold_stability_summary.to_dict(orient="records")
    decision["bad_fold_signature"] = bad_fold_signature.to_dict(orient="records")
    decision["feature_family_drift_summary"] = feature_family_drift_summary.to_dict(orient="records")
    decision["probability_quality_summary"] = probability_quality_summary.to_dict(orient="records")
    decision["score_distribution_shift_summary"] = score_distribution_shift_summary.to_dict(orient="records")
    decision["fold_reliability_gate_summary"] = fold_reliability_gate_summary.to_dict(orient="records")
    decision["regime_threshold_policy_summary"] = regime_threshold_policy_summary.to_dict(orient="records")
    decision["regime_stability_summary"] = regime_stability_summary.to_dict(orient="records")
    decision["threshold_forensics_summary"] = (
        threshold_forensics["primary_issue"].value_counts().to_dict()
        if not threshold_forensics.empty and "primary_issue" in threshold_forensics.columns
        else {}
    )
    decision["threshold_policy_review"] = threshold_policy_review.to_dict(orient="records")
    decision["threshold_transfer_review"] = threshold_transfer_review.to_dict(orient="records")
    decision["payoff_alignment_summary"] = payoff_alignment_summary.to_dict(orient="records")
    decision["payoff_policy_robustness_summary"] = payoff_policy_robustness_summary.to_dict(orient="records")
    bundle_path = Path(output_dir) / f"phase1_experiment_bundle_{run_dir.name}.zip" if write_full_bundles else None
    latest_bundle_path = Path(output_dir) / "phase1_latest_experiment_bundle.zip" if write_full_bundles else None
    slim_bundle_path = Path(output_dir) / f"phase1_experiment_slim_bundle_{run_dir.name}.zip"
    latest_slim_bundle_path = Path(output_dir) / "phase1_latest_experiment_slim_bundle.zip"
    decision["write_full_bundles"] = bool(write_full_bundles)
    decision["bundle_zip"] = str(bundle_path) if bundle_path is not None else None
    decision["latest_bundle_zip"] = str(latest_bundle_path) if latest_bundle_path is not None else None
    decision["slim_bundle_zip"] = str(slim_bundle_path)
    decision["latest_slim_bundle_zip"] = str(latest_slim_bundle_path)
    _write_decision_files(report_dir, comparison, decision)
    _write_json(report_dir / "training_execution_summary.json", training_execution)
    _write_profile_delta(report_dir, profile_delta)
    _write_seed_audit_files(report_dir, seed_audit, seed_stability)
    _write_seed_ensemble_files(report_dir, seed_ensemble)
    _write_profile_blend_files(report_dir, profile_blend)
    _write_performance_gap_analysis(report_dir, performance_gap_analysis)
    _write_forensics_reports(
        report_dir,
        fold_stability_forensics=fold_stability_forensics,
        fold_stability_summary=fold_stability_summary,
        threshold_forensics=threshold_forensics,
    )
    _write_score_separation_forensics(report_dir, score_separation_forensics, bad_fold_signature)
    _write_feature_drift_forensics(report_dir, feature_drift_forensics, feature_family_drift_summary)
    _write_probability_quality_forensics(report_dir, probability_quality_forensics, probability_quality_summary)
    _write_score_distribution_shift(report_dir, score_distribution_shift, score_distribution_shift_summary)
    _write_fold_reliability_gate(report_dir, fold_reliability_gate, fold_reliability_gate_summary)
    _write_regime_threshold_policy(report_dir, regime_threshold_policy_by_fold, regime_threshold_policy_summary)
    _write_regime_stability(report_dir, regime_stability_forensics, regime_stability_summary)
    _write_threshold_policy_review(report_dir, threshold_policy_review)
    _write_threshold_transfer_review(report_dir, threshold_transfer_review, threshold_transfer_by_fold)
    _write_payoff_alignment(report_dir, payoff_alignment, payoff_alignment_summary)
    _write_payoff_policy_robustness(report_dir, payoff_policy_robustness, payoff_policy_robustness_summary)
    _write_profile_diagnostic_summaries(report_dir, entries)
    _write_holdout_boundary_audit(report_dir, holdout_boundary_audit)
    _write_frozen_policy_robustness(report_dir, frozen_policy_robustness)
    _write_frozen_policy_monitoring_plan(report_dir, frozen_policy_monitoring_plan)
    _write_experiment_policy_guard(report_dir, experiment_policy_guard)
    _write_future_oos_candidate_plan(report_dir, future_oos_candidate_plan)
    _write_holdout_files(
        report_dir,
        holdout_evaluation=holdout_evaluation,
        holdout_score_bands=holdout_score_bands,
        holdout_thresholds=holdout_thresholds,
        holdout_decision=holdout_decision,
        config=diagnostic_config,
    )
    from yenibot.automation import write_auto_review

    auto_review = write_auto_review(report_dir)
    auto_review_path = Path(auto_review["auto_review_path"])
    auto_review_json_path = Path(auto_review["auto_review_json_path"])
    next_actions_path = Path(auto_review["next_actions_path"])
    phase2_readiness_path = Path(auto_review["phase2_readiness_path"])
    phase2_readiness_md_path = Path(auto_review["phase2_readiness_md_path"])
    phase1_transition_plan_path = Path(auto_review["phase1_transition_plan_path"])
    phase1_transition_plan_md_path = Path(auto_review["phase1_transition_plan_md_path"])
    decision["auto_review_path"] = str(auto_review_path)
    decision["auto_review_json_path"] = str(auto_review_json_path)
    decision["next_actions_path"] = str(next_actions_path)
    decision["phase2_readiness_path"] = str(phase2_readiness_path)
    decision["phase2_readiness_md_path"] = str(phase2_readiness_md_path)
    decision["phase2_readiness"] = auto_review["review"].get("phase2_readiness", {})
    decision["phase1_transition_plan_path"] = str(phase1_transition_plan_path)
    decision["phase1_transition_plan_md_path"] = str(phase1_transition_plan_md_path)
    decision["phase1_transition_plan"] = auto_review["review"].get("phase1_transition_plan", {})
    phase1_blocker_action_plan = _phase1_blocker_action_plan_frame(
        comparison=comparison,
        profile_blend=profile_blend,
        performance_gap_analysis=performance_gap_analysis,
        fold_stability_forensics=fold_stability_forensics,
        fold_stability_summary=fold_stability_summary,
        threshold_forensics=threshold_forensics,
        payoff_policy_robustness_summary=payoff_policy_robustness_summary,
        future_oos_candidate_plan=future_oos_candidate_plan,
        phase2_readiness=decision["phase2_readiness"],
        config=diagnostic_config,
        settings=settings,
    )
    decision["phase1_blocker_action_plan"] = phase1_blocker_action_plan.to_dict(orient="records")
    _write_phase1_blocker_action_plan(report_dir, phase1_blocker_action_plan)
    _write_decision_files(report_dir, comparison, decision)
    _write_decision_files(run_dir, comparison, decision)
    _write_json(_training_execution_summary_path(run_dir), training_execution)
    _write_profile_delta(run_dir, profile_delta)
    _write_seed_audit_files(run_dir, seed_audit, seed_stability)
    _write_seed_ensemble_files(run_dir, seed_ensemble)
    _write_profile_blend_files(run_dir, profile_blend)
    _write_performance_gap_analysis(run_dir, performance_gap_analysis)
    _write_phase1_blocker_action_plan(run_dir, phase1_blocker_action_plan)
    _write_forensics_reports(
        run_dir,
        fold_stability_forensics=fold_stability_forensics,
        fold_stability_summary=fold_stability_summary,
        threshold_forensics=threshold_forensics,
    )
    _write_score_separation_forensics(run_dir, score_separation_forensics, bad_fold_signature)
    _write_feature_drift_forensics(run_dir, feature_drift_forensics, feature_family_drift_summary)
    _write_probability_quality_forensics(run_dir, probability_quality_forensics, probability_quality_summary)
    _write_score_distribution_shift(run_dir, score_distribution_shift, score_distribution_shift_summary)
    _write_fold_reliability_gate(run_dir, fold_reliability_gate, fold_reliability_gate_summary)
    _write_regime_threshold_policy(run_dir, regime_threshold_policy_by_fold, regime_threshold_policy_summary)
    _write_regime_stability(run_dir, regime_stability_forensics, regime_stability_summary)
    _write_threshold_policy_review(run_dir, threshold_policy_review)
    _write_threshold_transfer_review(run_dir, threshold_transfer_review, threshold_transfer_by_fold)
    _write_payoff_alignment(run_dir, payoff_alignment, payoff_alignment_summary)
    _write_payoff_policy_robustness(run_dir, payoff_policy_robustness, payoff_policy_robustness_summary)
    _write_profile_diagnostic_summaries(run_dir, entries)
    _write_experiment_selection(run_dir, settings)
    _write_holdout_reservation(run_dir, settings)
    _write_missing_selected_profiles(run_dir, missing_selected)
    _write_holdout_boundary_audit(run_dir, holdout_boundary_audit)
    _write_frozen_policy_robustness(run_dir, frozen_policy_robustness)
    _write_frozen_policy_monitoring_plan(run_dir, frozen_policy_monitoring_plan)
    _write_experiment_policy_guard(run_dir, experiment_policy_guard)
    _write_future_oos_candidate_plan(run_dir, future_oos_candidate_plan)
    _write_holdout_files(
        run_dir,
        holdout_evaluation=holdout_evaluation,
        holdout_score_bands=holdout_score_bands,
        holdout_thresholds=holdout_thresholds,
        holdout_decision=holdout_decision,
        config=diagnostic_config,
    )
    if write_full_bundles:
        bundle_path, latest_bundle_path = _write_experiment_bundle(
            output_dir=Path(output_dir),
            run_id=run_dir.name,
            report_dir=report_dir,
            zip_paths=zip_paths,
        )
    else:
        bundle_path = None
        latest_bundle_path = None
    slim_bundle_path, latest_slim_bundle_path = _write_experiment_slim_bundle(
        output_dir=Path(output_dir),
        run_id=run_dir.name,
        report_dir=report_dir,
    )
    return {
        "run_id": run_dir.name,
        "run_dir": run_dir,
        "comparison": comparison,
        "profile_delta": profile_delta,
        "seed_audit": seed_audit,
        "seed_stability": seed_stability,
        "seed_ensemble": seed_ensemble,
        "profile_blend": profile_blend,
        "performance_gap_analysis": performance_gap_analysis,
        "phase1_blocker_action_plan": phase1_blocker_action_plan,
        "fold_stability_forensics": fold_stability_forensics,
        "fold_stability_summary": fold_stability_summary,
        "score_separation_forensics": score_separation_forensics,
        "bad_fold_signature": bad_fold_signature,
        "feature_drift_forensics": feature_drift_forensics,
        "feature_family_drift_summary": feature_family_drift_summary,
        "probability_quality_forensics": probability_quality_forensics,
        "probability_quality_summary": probability_quality_summary,
        "score_distribution_shift": score_distribution_shift,
        "score_distribution_shift_summary": score_distribution_shift_summary,
        "fold_reliability_gate": fold_reliability_gate,
        "fold_reliability_gate_summary": fold_reliability_gate_summary,
        "regime_threshold_policy_by_fold": regime_threshold_policy_by_fold,
        "regime_threshold_policy_summary": regime_threshold_policy_summary,
        "regime_stability_forensics": regime_stability_forensics,
        "regime_stability_summary": regime_stability_summary,
        "threshold_forensics": threshold_forensics,
        "threshold_policy_review": threshold_policy_review,
        "threshold_transfer_review": threshold_transfer_review,
        "threshold_transfer_by_fold": threshold_transfer_by_fold,
        "payoff_alignment": payoff_alignment,
        "payoff_alignment_summary": payoff_alignment_summary,
        "payoff_policy_robustness": payoff_policy_robustness,
        "payoff_policy_robustness_summary": payoff_policy_robustness_summary,
        "experiment_selection": experiment_selection,
        "holdout_reservation": holdout_reservation,
        "holdout_boundary_audit": holdout_boundary_audit,
        "frozen_policy_robustness": frozen_policy_robustness,
        "frozen_policy_monitoring_plan": frozen_policy_monitoring_plan,
        "experiment_policy_guard": experiment_policy_guard,
        "future_oos_candidate_plan": future_oos_candidate_plan,
        "holdout_evaluation": holdout_evaluation,
        "holdout_score_bands": holdout_score_bands,
        "holdout_thresholds": holdout_thresholds,
        "holdout_entries": holdout_entries,
        "missing_selected_profiles": missing_selected,
        "decision": decision,
        "zip_paths": zip_paths,
        "write_full_bundles": bool(write_full_bundles),
        "bundle_zip": str(bundle_path) if bundle_path is not None else None,
        "latest_bundle_zip": str(latest_bundle_path) if latest_bundle_path is not None else None,
        "slim_bundle_zip": str(slim_bundle_path),
        "latest_slim_bundle_zip": str(latest_slim_bundle_path),
        "auto_review": auto_review["review"],
        "auto_review_path": str(auto_review_path),
        "auto_review_json_path": str(auto_review_json_path),
        "next_actions_path": str(next_actions_path),
        "phase2_readiness_path": str(phase2_readiness_path),
        "phase2_readiness_md_path": str(phase2_readiness_md_path),
        "phase1_transition_plan_path": str(phase1_transition_plan_path),
        "phase1_transition_plan_md_path": str(phase1_transition_plan_md_path),
    }
