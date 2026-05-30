from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score


def rank_ic(prob_long: np.ndarray | pd.Series, forward_return: np.ndarray | pd.Series) -> float:
    data = pd.DataFrame({"prob": prob_long, "ret": forward_return}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3 or data["prob"].nunique() < 2 or data["ret"].nunique() < 2:
        return 0.0
    value = data["prob"].corr(data["ret"], method="spearman")
    if pd.isna(value):
        return 0.0
    return float(value)


def classification_metrics(
    labels: np.ndarray | pd.Series,
    prob_long: np.ndarray | pd.Series,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true = np.asarray(labels).astype(int)
    probs = np.asarray(prob_long).astype(float)
    y_pred = (probs >= threshold).astype(int)
    out = {
        "long_f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) == 2:
        out["prauc"] = float(average_precision_score(y_true, probs))
    else:
        out["prauc"] = 0.0
    return out


def calibration_table(
    labels: np.ndarray | pd.Series,
    prob_long: np.ndarray | pd.Series,
    *,
    bins: int = 10,
) -> pd.DataFrame:
    df = pd.DataFrame({"label": labels, "prob_long": prob_long}).replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        return pd.DataFrame(columns=["bin", "count", "mean_prob_long", "actual_long_rate"])
    df["bin"] = pd.qcut(
        df["prob_long"].rank(method="first"),
        q=min(bins, len(df)),
        labels=False,
        duplicates="drop",
    )
    grouped = df.groupby("bin", observed=True).agg(
        count=("label", "size"),
        mean_prob_long=("prob_long", "mean"),
        actual_long_rate=("label", "mean"),
    )
    return grouped.reset_index()


def phase1_report(predictions: pd.DataFrame, config: object) -> dict[str, object]:
    if predictions.empty:
        raise ValueError("No predictions supplied")

    rank_by_fold = predictions.groupby("fold").apply(
        lambda part: rank_ic(part["prob_long"], part["forward_return"]),
        include_groups=False,
    )
    class_by_fold = predictions.groupby("fold").apply(
        lambda part: pd.Series(classification_metrics(part["label"], part["prob_long"])),
        include_groups=False,
    )

    mean_rank_ic = float(rank_by_fold.mean())
    std_rank_ic = float(rank_by_fold.std(ddof=0))
    positive_fraction = float((rank_by_fold > 0).mean())
    mean_long_f1 = float(class_by_fold["long_f1"].mean())
    mean_prauc = float(class_by_fold["prauc"].mean())
    actual_long = predictions.loc[predictions["label"] == 1, "prob_long"].mean()
    actual_not_long = predictions.loc[predictions["label"] == 0, "prob_long"].mean()
    calibration_separation = float(actual_long - actual_not_long)

    validation = config["validation"] if isinstance(config, dict) else config.validation
    checks = {
        "rank_ic_mean": mean_rank_ic > float(validation["target_rank_ic"]),
        "rank_ic_std": std_rank_ic < float(validation["max_rank_ic_std"]),
        "positive_ic_fraction": positive_fraction > float(validation["min_positive_ic_fraction"]),
        "long_f1": mean_long_f1 > float(validation["min_long_f1"]),
        "calibration_separation": calibration_separation > 0,
    }
    alerts: list[str] = []
    if abs(mean_rank_ic) <= float(validation["random_like_rank_ic"]):
        alerts.append("Rank IC is near random; improve features before tuning hyperparameters.")
    if mean_rank_ic >= float(validation["suspicious_rank_ic"]):
        alerts.append("Rank IC is suspiciously high; audit MTF alignment, causality, and scaler fitting.")

    return {
        "rank_ic_by_fold": rank_by_fold.to_dict(),
        "mean_rank_ic": mean_rank_ic,
        "std_rank_ic": std_rank_ic,
        "positive_ic_fraction": positive_fraction,
        "mean_long_f1": mean_long_f1,
        "mean_prauc": mean_prauc,
        "calibration_separation": calibration_separation,
        "checks": checks,
        "passed": all(checks.values()),
        "alerts": alerts,
    }
