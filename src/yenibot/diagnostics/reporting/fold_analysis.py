from __future__ import annotations

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.stats import ks_2samp
from yenibot.diagnostics.metrics import rank_ic

from .utils import (
    _diagnostic_feature_columns, _feature_target_rank_ic, _top_joined_tokens, classify_feature_column
)
from .fold_metrics import good_bad_fold_summary

def regime_diagnostics(predictions: pd.DataFrame, *, threshold: float = 0.5) -> pd.DataFrame:
    regime_columns = [column for column in predictions.columns if column.startswith("regime_prob_")]
    if not regime_columns:
        return pd.DataFrame()

    frame = predictions.copy()
    frame["regime"] = frame[regime_columns].idxmax(axis=1).str.rsplit("_", n=1).str[-1].astype(int)
    rows = []
    for regime, part in frame.groupby("regime"):
        y_true = part["label"].astype(int)
        y_pred = (part["prob_long"] >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "regime": int(regime),
                "count": int(len(part)),
                "rank_ic": rank_ic(part["prob_long"], part["forward_return"]),
                "label_long_rate": float(part["label"].mean()),
                "pred_long_rate_050": float(y_pred.mean()),
                "precision_050": float(precision),
                "recall_050": float(recall),
                "long_f1_050": float(f1),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)

def regime_by_fold_diagnostics(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    *,
    threshold: float = 0.5,
    bad_ic: float = -0.08,
) -> pd.DataFrame:
    regime_columns = [column for column in predictions.columns if column.startswith("regime_prob_")]
    if not regime_columns or "fold" not in predictions.columns or fold_metrics.empty:
        return pd.DataFrame()

    fold_rank_ic = fold_metrics.set_index("fold")["rank_ic"].to_dict() if "rank_ic" in fold_metrics.columns else {}
    frame = predictions.copy()
    frame["regime"] = frame[regime_columns].idxmax(axis=1).str.rsplit("_", n=1).str[-1].astype(int)
    rows = []
    for (fold, regime), part in frame.groupby(["fold", "regime"]):
        y_true = part["label"].astype(int)
        y_pred = (part["prob_long"] >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        fold_ic = float(fold_rank_ic.get(fold, np.nan))
        rows.append(
            {
                "fold": int(fold),
                "regime": int(regime),
                "count": int(len(part)),
                "fold_rank_ic": fold_ic,
                "is_bad_fold": bool(pd.notna(fold_ic) and fold_ic <= bad_ic),
                "rank_ic": rank_ic(part["prob_long"], part["forward_return"]),
                "label_long_rate": float(part["label"].mean()),
                "pred_long_rate_050": float(y_pred.mean()),
                "prob_long_mean": float(part["prob_long"].mean()),
                "forward_return_mean": float(part["forward_return"].mean()) if "forward_return" in part.columns else np.nan,
                "precision_050": float(precision),
                "recall_050": float(recall),
                "long_f1_050": float(f1),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["fold", "regime"]).reset_index(drop=True)

def bad_fold_regime_diagnostics(regime_by_fold: pd.DataFrame) -> pd.DataFrame:
    if regime_by_fold is None or regime_by_fold.empty or "is_bad_fold" not in regime_by_fold.columns:
        return pd.DataFrame()

    rows = []
    for regime, part in regime_by_fold.groupby("regime"):
        bad = part[part["is_bad_fold"].astype(bool)]
        other = part[~part["is_bad_fold"].astype(bool)]
        row = {
            "regime": int(regime),
            "bad_fold_rows": int(len(bad)),
            "other_fold_rows": int(len(other)),
            "bad_count": int(bad["count"].sum()) if not bad.empty else 0,
            "other_count": int(other["count"].sum()) if not other.empty else 0,
            "bad_mean_rank_ic": float(bad["rank_ic"].mean()) if not bad.empty else np.nan,
            "other_mean_rank_ic": float(other["rank_ic"].mean()) if not other.empty else np.nan,
            "bad_mean_long_f1_050": float(bad["long_f1_050"].mean()) if not bad.empty else np.nan,
            "other_mean_long_f1_050": float(other["long_f1_050"].mean()) if not other.empty else np.nan,
            "bad_mean_label_long_rate": float(bad["label_long_rate"].mean()) if not bad.empty else np.nan,
            "other_mean_label_long_rate": float(other["label_long_rate"].mean()) if not other.empty else np.nan,
            "bad_mean_pred_long_rate_050": float(bad["pred_long_rate_050"].mean()) if not bad.empty else np.nan,
            "other_mean_pred_long_rate_050": float(other["pred_long_rate_050"].mean()) if not other.empty else np.nan,
            "bad_mean_forward_return": float(bad["forward_return_mean"].mean()) if not bad.empty else np.nan,
            "other_mean_forward_return": float(other["forward_return_mean"].mean()) if not other.empty else np.nan,
        }
        row["rank_ic_gap_bad_minus_other"] = row["bad_mean_rank_ic"] - row["other_mean_rank_ic"]
        row["long_f1_gap_bad_minus_other"] = row["bad_mean_long_f1_050"] - row["other_mean_long_f1_050"]
        row["pred_long_rate_gap_bad_minus_other"] = row["bad_mean_pred_long_rate_050"] - row["other_mean_pred_long_rate_050"]
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("rank_ic_gap_bad_minus_other").reset_index(drop=True)

def bad_fold_feature_forensics(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    good_ic: float = 0.10,
    bad_ic: float = -0.08,
    target_column: str = "forward_return",
    min_abs_reference_ic: float = 0.02,
) -> pd.DataFrame:
    required = {"fold", target_column}
    if not required.issubset(predictions.columns) or fold_metrics.empty:
        return pd.DataFrame()

    summary = good_bad_fold_summary(fold_metrics, good_ic=good_ic, bad_ic=bad_ic)
    good_folds = set(summary["good_folds"])
    bad_folds = set(summary["bad_folds"])
    if not good_folds or not bad_folds:
        return pd.DataFrame()

    if feature_columns is None:
        feature_columns = _diagnostic_feature_columns(predictions)
    feature_columns = [column for column in feature_columns if column in predictions.columns]
    good = predictions[predictions["fold"].isin(good_folds)]
    fold_rank_ic = fold_metrics.set_index("fold")["rank_ic"].to_dict()
    rows = []
    for bad_fold in sorted(bad_folds):
        bad = predictions[predictions["fold"] == bad_fold]
        if bad.empty:
            continue
        for feature in feature_columns:
            if not pd.api.types.is_numeric_dtype(predictions[feature]):
                continue
            good_values = good[feature].replace([np.inf, -np.inf], np.nan).dropna()
            bad_values = bad[feature].replace([np.inf, -np.inf], np.nan).dropna()
            if len(good_values) < 3 or len(bad_values) < 3:
                continue
            pooled_std = float(pd.concat([good_values, bad_values]).std(ddof=0))
            good_feature_ic = _feature_target_rank_ic(good, feature, target_column)
            bad_feature_ic = _feature_target_rank_ic(bad, feature, target_column)
            delta_feature_ic = (
                float(bad_feature_ic - good_feature_ic)
                if pd.notna(good_feature_ic) and pd.notna(bad_feature_ic)
                else np.nan
            )
            timeframe, family = classify_feature_column(feature)
            rows.append(
                {
                    "bad_fold": int(bad_fold),
                    "bad_fold_rank_ic": float(fold_rank_ic.get(bad_fold, np.nan)),
                    "feature": feature,
                    "timeframe": timeframe,
                    "family": family,
                    "good_feature_ic": good_feature_ic,
                    "bad_feature_ic": bad_feature_ic,
                    "delta_feature_ic_bad_minus_good": delta_feature_ic,
                    "signal_reversal": bool(
                        pd.notna(good_feature_ic)
                        and pd.notna(bad_feature_ic)
                        and abs(good_feature_ic) >= min_abs_reference_ic
                        and abs(bad_feature_ic) >= min_abs_reference_ic
                        and np.sign(good_feature_ic) != np.sign(bad_feature_ic)
                    ),
                    "good_mean": float(good_values.mean()),
                    "bad_mean": float(bad_values.mean()),
                    "mean_diff_bad_minus_good": float(bad_values.mean() - good_values.mean()),
                    "abs_standardized_diff": abs(float(bad_values.mean() - good_values.mean())) / pooled_std
                    if pooled_std > 0
                    else 0.0,
                }
            )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["bad_fold", "signal_reversal", "abs_standardized_diff", "delta_feature_ic_bad_minus_good"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)

def bad_fold_feature_forensics_summary(
    feature_forensics: pd.DataFrame,
    *,
    min_bad_fold_count: int = 2,
    min_signal_reversal_rate: float = 0.34,
    min_mean_abs_delta_ic: float = 0.05,
) -> pd.DataFrame:
    """Aggregate repeated bad-fold feature failures into a stable pruning watchlist."""

    columns = [
        "feature",
        "timeframe",
        "family",
        "bad_fold_count",
        "bad_folds",
        "mean_good_feature_ic",
        "mean_bad_feature_ic",
        "mean_delta_feature_ic_bad_minus_good",
        "mean_abs_delta_feature_ic",
        "signal_reversal_rate",
        "mean_abs_standardized_diff",
        "max_abs_standardized_diff",
        "recommended_action",
    ]
    if feature_forensics is None or feature_forensics.empty:
        return pd.DataFrame(columns=columns)

    frame = feature_forensics.copy()
    rows = []
    for (feature, timeframe, family), part in frame.groupby(["feature", "timeframe", "family"], dropna=False):
        bad_folds = sorted(pd.to_numeric(part["bad_fold"], errors="coerce").dropna().astype(int).unique().tolist())
        mean_abs_delta = float(pd.to_numeric(part["delta_feature_ic_bad_minus_good"], errors="coerce").abs().mean())
        reversal_rate = float(part["signal_reversal"].astype(bool).mean())
        mean_shift = float(pd.to_numeric(part["abs_standardized_diff"], errors="coerce").mean())
        recommended = (
            len(bad_folds) >= min_bad_fold_count
            and (reversal_rate >= min_signal_reversal_rate or mean_abs_delta >= min_mean_abs_delta_ic)
        )
        rows.append(
            {
                "feature": str(feature),
                "timeframe": str(timeframe),
                "family": str(family),
                "bad_fold_count": int(len(bad_folds)),
                "bad_folds": ",".join(map(str, bad_folds)),
                "mean_good_feature_ic": float(pd.to_numeric(part["good_feature_ic"], errors="coerce").mean()),
                "mean_bad_feature_ic": float(pd.to_numeric(part["bad_feature_ic"], errors="coerce").mean()),
                "mean_delta_feature_ic_bad_minus_good": float(
                    pd.to_numeric(part["delta_feature_ic_bad_minus_good"], errors="coerce").mean()
                ),
                "mean_abs_delta_feature_ic": mean_abs_delta,
                "signal_reversal_rate": reversal_rate,
                "mean_abs_standardized_diff": mean_shift,
                "max_abs_standardized_diff": float(
                    pd.to_numeric(part["abs_standardized_diff"], errors="coerce").max()
                ),
                "recommended_action": "ablate_or_bound" if recommended else "monitor",
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["recommended_action", "bad_fold_count", "signal_reversal_rate", "mean_abs_delta_feature_ic"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

def bad_fold_group_forensics(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    good_ic: float = 0.10,
    bad_ic: float = -0.08,
) -> pd.DataFrame:
    feature_frame = bad_fold_feature_forensics(
        predictions,
        fold_metrics,
        feature_columns=feature_columns,
        good_ic=good_ic,
        bad_ic=bad_ic,
    )
    if feature_frame.empty:
        return pd.DataFrame()

    rows = []
    for (bad_fold, bad_rank_ic, timeframe, family), part in feature_frame.groupby(
        ["bad_fold", "bad_fold_rank_ic", "timeframe", "family"],
        dropna=False,
    ):
        ranked_delta = part.reindex(part["delta_feature_ic_bad_minus_good"].abs().sort_values(ascending=False).index)
        shifted = part.sort_values("abs_standardized_diff", ascending=False)
        rows.append(
            {
                "bad_fold": int(bad_fold),
                "bad_fold_rank_ic": float(bad_rank_ic),
                "timeframe": timeframe,
                "family": family,
                "feature_count": int(part["feature"].nunique()),
                "mean_good_feature_ic": float(part["good_feature_ic"].mean()),
                "mean_bad_feature_ic": float(part["bad_feature_ic"].mean()),
                "mean_delta_feature_ic_bad_minus_good": float(part["delta_feature_ic_bad_minus_good"].mean()),
                "mean_abs_delta_feature_ic": float(part["delta_feature_ic_bad_minus_good"].abs().mean()),
                "signal_reversal_rate": float(part["signal_reversal"].mean()),
                "mean_abs_standardized_diff": float(part["abs_standardized_diff"].mean()),
                "top_delta_features": ",".join(ranked_delta["feature"].head(5).astype(str).tolist()),
                "top_shifted_features": ",".join(shifted["feature"].head(5).astype(str).tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["bad_fold", "signal_reversal_rate", "mean_abs_delta_feature_ic", "mean_abs_standardized_diff"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

def bad_fold_group_forensics_summary(
    group_forensics: pd.DataFrame,
    *,
    min_bad_fold_count: int = 2,
    min_signal_reversal_rate: float = 0.25,
    min_mean_abs_delta_ic: float = 0.05,
) -> pd.DataFrame:
    """Aggregate repeated family-level bad-fold failures for experiment planning."""

    columns = [
        "timeframe",
        "family",
        "bad_fold_count",
        "bad_folds",
        "mean_feature_count",
        "mean_delta_feature_ic_bad_minus_good",
        "mean_abs_delta_feature_ic",
        "signal_reversal_rate",
        "mean_abs_standardized_diff",
        "top_delta_features",
        "top_shifted_features",
        "recommended_action",
    ]
    if group_forensics is None or group_forensics.empty:
        return pd.DataFrame(columns=columns)

    frame = group_forensics.copy()
    rows = []
    for (timeframe, family), part in frame.groupby(["timeframe", "family"], dropna=False):
        bad_folds = sorted(pd.to_numeric(part["bad_fold"], errors="coerce").dropna().astype(int).unique().tolist())
        mean_abs_delta = float(pd.to_numeric(part["mean_abs_delta_feature_ic"], errors="coerce").mean())
        reversal_rate = float(pd.to_numeric(part["signal_reversal_rate"], errors="coerce").mean())
        recommended = (
            len(bad_folds) >= min_bad_fold_count
            and (reversal_rate >= min_signal_reversal_rate or mean_abs_delta >= min_mean_abs_delta_ic)
        )
        rows.append(
            {
                "timeframe": str(timeframe),
                "family": str(family),
                "bad_fold_count": int(len(bad_folds)),
                "bad_folds": ",".join(map(str, bad_folds)),
                "mean_feature_count": float(pd.to_numeric(part["feature_count"], errors="coerce").mean()),
                "mean_delta_feature_ic_bad_minus_good": float(
                    pd.to_numeric(part["mean_delta_feature_ic_bad_minus_good"], errors="coerce").mean()
                ),
                "mean_abs_delta_feature_ic": mean_abs_delta,
                "signal_reversal_rate": reversal_rate,
                "mean_abs_standardized_diff": float(
                    pd.to_numeric(part["mean_abs_standardized_diff"], errors="coerce").mean()
                ),
                "top_delta_features": _top_joined_tokens(part.get("top_delta_features", pd.Series(dtype=str))),
                "top_shifted_features": _top_joined_tokens(part.get("top_shifted_features", pd.Series(dtype=str))),
                "recommended_action": "ablate_or_split" if recommended else "monitor",
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["recommended_action", "bad_fold_count", "signal_reversal_rate", "mean_abs_delta_feature_ic"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

def good_bad_feature_audit(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    *,
    good_ic: float = 0.10,
    bad_ic: float = -0.08,
    top_n: int = 30,
) -> pd.DataFrame:
    summary = good_bad_fold_summary(fold_metrics, good_ic=good_ic, bad_ic=bad_ic)
    good_folds = set(summary["good_folds"])
    bad_folds = set(summary["bad_folds"])
    if not good_folds or not bad_folds:
        return pd.DataFrame()

    good = predictions[predictions["fold"].isin(good_folds)]
    bad = predictions[predictions["fold"].isin(bad_folds)]
    rows = []
    for column in _diagnostic_feature_columns(predictions):
        good_values = good[column].replace([np.inf, -np.inf], np.nan).dropna()
        bad_values = bad[column].replace([np.inf, -np.inf], np.nan).dropna()
        if len(good_values) < 3 or len(bad_values) < 3:
            continue
        pooled_std = float(pd.concat([good_values, bad_values]).std(ddof=0))
        mean_diff = float(good_values.mean() - bad_values.mean())
        rows.append(
            {
                "feature": column,
                "good_mean": float(good_values.mean()),
                "bad_mean": float(bad_values.mean()),
                "mean_diff_good_minus_bad": mean_diff,
                "abs_standardized_diff": abs(mean_diff) / pooled_std if pooled_std > 0 else 0.0,
                "ks_stat": float(ks_2samp(good_values, bad_values).statistic),
                "good_folds": ",".join(map(str, sorted(good_folds))),
                "bad_folds": ",".join(map(str, sorted(bad_folds))),
            }
        )
    columns = [
        "feature",
        "good_mean",
        "bad_mean",
        "mean_diff_good_minus_bad",
        "abs_standardized_diff",
        "ks_stat",
        "good_folds",
        "bad_folds",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["ks_stat", "abs_standardized_diff"], ascending=False).head(top_n).reset_index(drop=True)

