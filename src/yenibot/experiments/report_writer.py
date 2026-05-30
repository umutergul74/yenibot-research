from __future__ import annotations

from __future__ import annotations
import shutil
import zipfile
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from .utils import _cfg, _table_markdown, _float, _numeric_mean, _write_json, _test_predictions

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

def _write_frozen_policy_robustness(path: Path, frame: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path / "frozen_policy_robustness.csv", index=False)
    (path / "frozen_policy_robustness.md").write_text(
        _table_markdown("Frozen Policy Robustness", frame),
        encoding="utf-8",
    )
    _write_json(path / "frozen_policy_robustness.json", {"rows": frame.to_dict(orient="records")})

def _write_profile_delta(path: Path, profile_delta: pd.DataFrame | None) -> None:
    if profile_delta is None:
        return
    profile_delta.to_csv(path / "profile_delta_vs_control.csv", index=False)

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

