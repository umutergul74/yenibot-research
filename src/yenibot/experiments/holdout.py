from __future__ import annotations

from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from .utils import _cfg, _future_oos_monitor_state, _holdout_signal_pass_reasons, _holdout_threshold_pass_reasons

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

