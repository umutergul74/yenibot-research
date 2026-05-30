from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from numba import njit
except Exception:  # pragma: no cover - numba is optional at import time
    njit = None


HIT_TYPES = {
    0: "time",
    1: "tp",
    2: "sl",
    3: "both_sl_first",
}


@dataclass(frozen=True)
class LabelQuality:
    long_mean_forward_return: float
    not_long_mean_forward_return: float
    long_pct: float


def _label_loop_python(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    tp_multiplier: float,
    sl_multiplier: float,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    labels = np.full(n, -1, dtype=np.int8)
    tb_return = np.full(n, np.nan, dtype=np.float64)
    hit_code = np.full(n, -1, dtype=np.int8)
    exit_bar = np.full(n, -1, dtype=np.int64)

    for i in range(n - horizon):
        if not np.isfinite(atr[i]) or atr[i] <= 0 or not np.isfinite(close[i]):
            continue
        entry = close[i]
        tp = entry + tp_multiplier * atr[i]
        sl = entry - sl_multiplier * atr[i]
        label = 0
        code = 0
        exit_idx = i + horizon
        exit_price = close[exit_idx]

        for j in range(1, horizon + 1):
            idx = i + j
            hit_tp = high[idx] >= tp
            hit_sl = low[idx] <= sl
            if hit_tp and hit_sl:
                label = 0
                code = 3
                exit_idx = idx
                exit_price = sl
                break
            if hit_sl:
                label = 0
                code = 2
                exit_idx = idx
                exit_price = sl
                break
            if hit_tp:
                label = 1
                code = 1
                exit_idx = idx
                exit_price = tp
                break

        labels[i] = label
        hit_code[i] = code
        exit_bar[i] = exit_idx
        tb_return[i] = (exit_price - entry) / entry
    return labels, tb_return, hit_code, exit_bar


if njit is not None:
    _label_loop_numba = njit(cache=True)(_label_loop_python)
else:
    _label_loop_numba = None


def add_long_only_labels(
    frame: pd.DataFrame,
    *,
    atr_column: str = "atr_14",
    tp_multiplier: float = 2.0,
    sl_multiplier: float = 5.0,
    max_holding_bars: int = 10,
) -> pd.DataFrame:
    """Add binary long-only triple-barrier labels to a feature frame."""

    required = ["timestamp", "close", "high", "low", atr_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns for labeling: {missing}")

    df = frame.copy().reset_index(drop=True)
    close = df["close"].astype(float).to_numpy()
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    atr = df[atr_column].astype(float).to_numpy()
    loop = _label_loop_numba or _label_loop_python
    labels, tb_return, hit_code, exit_bar = loop(
        close,
        high,
        low,
        atr,
        float(tp_multiplier),
        float(sl_multiplier),
        int(max_holding_bars),
    )

    df["label"] = labels
    df["tb_return"] = tb_return
    df["hit_type"] = [HIT_TYPES.get(int(code), "unlabelable") for code in hit_code]
    df["exit_bar"] = exit_bar
    df["exit_timestamp"] = pd.Series(pd.NaT, index=df.index, dtype=df["timestamp"].dtype)
    valid_exit = exit_bar >= 0
    if valid_exit.any():
        df.loc[valid_exit, "exit_timestamp"] = df["timestamp"].iloc[exit_bar[valid_exit]].to_list()
    df[f"fwd_return_{max_holding_bars}h"] = df["close"].shift(-max_holding_bars) / df["close"] - 1.0
    if max_holding_bars == 10:
        df["fwd_return_10h"] = df[f"fwd_return_{max_holding_bars}h"]

    df = df[df["label"] >= 0].copy()
    df["label"] = df["label"].astype("int8")
    return df.reset_index(drop=True)


def validate_label_quality(
    frame: pd.DataFrame,
    *,
    forward_return_column: str = "fwd_return_10h",
    min_long_forward_return: float = 0.003,
    max_not_long_forward_return: float = 0.001,
    min_long_pct: float = 0.20,
    max_long_pct: float = 0.50,
) -> LabelQuality:
    """Validate that long-only labels have basic economic meaning."""

    if "label" not in frame.columns:
        raise ValueError("label column is required")
    if forward_return_column not in frame.columns:
        raise ValueError(f"{forward_return_column} column is required")
    grouped = frame.groupby("label")[forward_return_column].mean()
    if 1 not in grouped.index or 0 not in grouped.index:
        raise ValueError("Both long and not-long labels must be present")

    long_mean = float(grouped.loc[1])
    not_long_mean = float(grouped.loc[0])
    long_pct = float(frame["label"].mean())

    if long_mean <= min_long_forward_return:
        raise ValueError(
            f"Long labels are not profitable enough: {long_mean:.6f} <= {min_long_forward_return:.6f}"
        )
    if not_long_mean >= max_not_long_forward_return:
        raise ValueError(
            f"Not-long labels have too much forward return: {not_long_mean:.6f} >= {max_not_long_forward_return:.6f}"
        )
    if not (min_long_pct < long_pct < max_long_pct):
        raise ValueError(f"Bad long label distribution: {long_pct:.2%}")

    return LabelQuality(
        long_mean_forward_return=long_mean,
        not_long_mean_forward_return=not_long_mean,
        long_pct=long_pct,
    )
