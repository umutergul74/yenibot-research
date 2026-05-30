from __future__ import annotations

import re
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Iterable

import numpy as np
import pandas as pd

from yenibot.features.wavelet import causal_wavelet_denoise

RAW_COLUMNS = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "num_trades",
    "taker_buy_base_vol",
    "taker_buy_quote_vol",
    "ignore",
}

METADATA_COLUMNS = {
    "4h_source_timestamp",
    "4h_available_timestamp",
}

LABEL_COLUMNS = {
    "label",
    "fwd_return_10h",
    "tb_return",
    "hit_type",
    "exit_timestamp",
    "exit_bar",
}


@dataclass(frozen=True)
class FeatureResult:
    frame: pd.DataFrame
    feature_columns: list[str]


def _safe_divide(numerator: pd.Series, denominator: pd.Series, default: float = 0.0) -> pd.Series:
    result = numerator.astype(float) / denominator.replace(0, np.nan).astype(float)
    return result.replace([np.inf, -np.inf], np.nan).fillna(default)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    return ((series - mean) / std.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _rolling_rank_score(series: pd.Series, window: int) -> pd.Series:
    def rank_last(values: np.ndarray) -> float:
        if np.isnan(values[-1]):
            return np.nan
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            return np.nan
        current = valid[-1]
        rank = ((valid < current).sum() + 0.5 * (valid == current).sum()) / len(valid)
        return float(2.0 * rank - 1.0)

    return series.rolling(window, min_periods=window).apply(rank_last, raw=True)


def _log_return(series: pd.Series) -> pd.Series:
    positive = series.where(series > 0)
    return np.log(positive / positive.shift(1)).replace([np.inf, -np.inf], np.nan)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x = x - x.mean()
    denom = float(np.dot(x, x))

    def slope(values: np.ndarray) -> float:
        y = values.astype(float)
        y = y - y.mean()
        return float(np.dot(x, y) / denom)

    return series.rolling(window, min_periods=window).apply(slope, raw=True)


def _array_slope(values: np.ndarray) -> float:
    valid = np.asarray(values, dtype=float)
    valid = valid[~np.isnan(valid)]
    if len(valid) < 2:
        return np.nan
    x = np.arange(len(valid), dtype=float)
    x = x - x.mean()
    y = valid - valid.mean()
    denom = float(np.dot(x, x))
    if denom == 0.0:
        return np.nan
    return float(np.dot(x, y) / denom)


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period, min_periods=period).mean()


def _adx(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    alpha = 1.0 / period
    atr = true_range.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()


def _config_get(config: object, path: Iterable[str], default: object) -> object:
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


def compute_bar_features(frame: pd.DataFrame, config: object) -> FeatureResult:
    """Compute causal microstructure features for a single timeframe."""

    df = frame.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    wavelet_enabled = bool(_config_get(config, ["features", "wavelet", "enabled"], True))
    if wavelet_enabled:
        wavelet_cfg = _config_get(config, ["features", "wavelet"], {})
        df["close_denoised"] = causal_wavelet_denoise(
            df["close"],
            window=int(_config_get(wavelet_cfg, ["window"], 256)),
            wavelet=str(_config_get(wavelet_cfg, ["wavelet"], "db4")),
            level=int(_config_get(wavelet_cfg, ["level"], 2)),
            threshold_scale=float(_config_get(wavelet_cfg, ["threshold_scale"], 0.5)),
        )
        df["volume_denoised"] = causal_wavelet_denoise(
            df["volume"],
            window=int(_config_get(wavelet_cfg, ["window"], 256)),
            wavelet=str(_config_get(wavelet_cfg, ["wavelet"], "db4")),
            level=int(_config_get(wavelet_cfg, ["level"], 2)),
            threshold_scale=float(_config_get(wavelet_cfg, ["threshold_scale"], 0.5)),
        )

    cvd_window = int(_config_get(config, ["features", "order_flow", "cvd_zscore_window"], 100))
    cvd_rate_window = int(_config_get(config, ["features", "order_flow", "cvd_rate_window"], 100))
    imbalance_span = int(_config_get(config, ["features", "order_flow", "imbalance_ema_span"], 14))
    vpt_window = int(_config_get(config, ["features", "whale", "vpt_zscore_window"], 100))
    whale_threshold = float(_config_get(config, ["features", "whale", "whale_zscore_threshold"], 2.0))
    whale_buy_ratio = float(_config_get(config, ["features", "whale", "whale_buy_ratio"], 0.55))
    whale_sell_ratio = float(_config_get(config, ["features", "whale", "whale_sell_ratio"], 0.45))
    large_trade_window = int(_config_get(config, ["features", "whale", "large_trade_window"], 100))
    realized_vol_window = int(_config_get(config, ["features", "structure", "realized_vol_window"], 14))
    gk_window = int(_config_get(config, ["features", "structure", "gk_vol_window"], 14))
    atr_period = int(_config_get(config, ["features", "structure", "atr_period"], 14))
    adx_period = int(_config_get(config, ["features", "structure", "adx_period"], 14))
    vwap_window = int(_config_get(config, ["features", "structure", "vwap_window"], 24))
    stationarity_cfg = _config_get(config, ["features", "stationarity"], {})
    stationarity_enabled = bool(_config_get(stationarity_cfg, ["enabled"], True))
    stationarity_window = int(_config_get(stationarity_cfg, ["normalization_window"], cvd_window))

    df["taker_buy_ratio"] = _safe_divide(df["taker_buy_base_vol"], df["volume"], default=0.5).clip(0.0, 1.0)
    df["taker_sell_ratio"] = 1.0 - df["taker_buy_ratio"]
    taker_sell_base = df["volume"] - df["taker_buy_base_vol"]
    df["true_cvd_delta"] = df["taker_buy_base_vol"] - taker_sell_base
    df["true_cvd_zscore"] = _rolling_zscore(df["true_cvd_delta"], cvd_window)
    cumulative_cvd = df["true_cvd_delta"].cumsum()
    df["cvd_cumulative_rate"] = _rolling_slope(cumulative_cvd, cvd_rate_window)
    df["buy_sell_imbalance_ema"] = (df["taker_buy_ratio"] - 0.5).ewm(
        span=imbalance_span,
        adjust=False,
        min_periods=imbalance_span,
    ).mean()

    df["vol_per_trade"] = _safe_divide(df["volume"], df["num_trades"])
    df["vpt_zscore"] = _rolling_zscore(df["vol_per_trade"], vpt_window)
    df["whale_buy_flag"] = ((df["vpt_zscore"] > whale_threshold) & (df["taker_buy_ratio"] > whale_buy_ratio)).astype(float)
    df["whale_sell_flag"] = ((df["vpt_zscore"] > whale_threshold) & (df["taker_buy_ratio"] < whale_sell_ratio)).astype(float)
    large_trade_volume = df["volume"].where(df["vpt_zscore"] > whale_threshold, 0.0)
    df["large_trade_ratio"] = _safe_divide(
        large_trade_volume.rolling(large_trade_window, min_periods=large_trade_window).sum(),
        df["volume"].rolling(large_trade_window, min_periods=large_trade_window).sum(),
    )

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["realized_vol_14"] = df["log_return"].rolling(realized_vol_window, min_periods=realized_vol_window).std(ddof=0)
    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])
    gk_var = 0.5 * log_hl.pow(2) - (2.0 * np.log(2.0) - 1.0) * log_co.pow(2)
    df["gk_vol_14"] = np.sqrt(gk_var.clip(lower=0).rolling(gk_window, min_periods=gk_window).mean())
    df["atr_14"] = _atr(df, atr_period)
    df["adx_14"] = _adx(df, adx_period)
    rolling_vwap = _safe_divide(
        df["quote_volume"].rolling(vwap_window, min_periods=vwap_window).sum(),
        df["volume"].rolling(vwap_window, min_periods=vwap_window).sum(),
        default=np.nan,
    )
    df["vwap_dist_atr"] = (df["close"] - rolling_vwap) / df["atr_14"].replace(0, np.nan)

    if stationarity_enabled:
        df = _add_stationary_features(df, config, stationarity_window)

    feature_columns = select_feature_columns(df)
    return FeatureResult(df, feature_columns)


def compute_intrahour_order_flow_features(intrabar_frame: pd.DataFrame, config: object) -> FeatureResult:
    """Aggregate lower-timeframe full-klines into causal 1H order-flow shape features."""

    cfg = _config_get(config, ["features", "intrahour_order_flow"], {})
    if not bool(_config_get(cfg, ["enabled"], False)):
        empty = pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns, UTC]")})
        return FeatureResult(empty, [])

    interval = str(_config_get(cfg, ["interval"], "15m")).lower()
    prefix = str(_config_get(cfg, ["prefix"], f"ih{interval}")).replace("-", "_")
    expected_bars = int(_config_get(cfg, ["expected_bars_per_hour"], 4))
    min_bars = int(_config_get(cfg, ["min_bars_per_hour"], expected_bars))
    stable_window = int(_config_get(cfg, ["stable_window"], _config_get(config, ["features", "stationarity", "normalization_window"], 100)))
    stable_clip_abs = float(_config_get(cfg, ["stable_clip_abs"], 5.0))
    stable_transforms = set(_config_get(cfg, ["stable_transforms"], ["zscore", "rank"]) or [])
    stable_tanh_scale = float(_config_get(cfg, ["stable_tanh_scale"], 2.0))

    df = intrabar_frame.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["hour_timestamp"] = df["timestamp"].dt.floor("h")
    df["taker_buy_ratio"] = _safe_divide(df["taker_buy_base_vol"], df["volume"], default=0.5).clip(0.0, 1.0)
    df["taker_imbalance"] = (2.0 * df["taker_buy_ratio"] - 1.0).clip(-1.0, 1.0)
    df["taker_sell_base_vol"] = (df["volume"] - df["taker_buy_base_vol"]).clip(lower=0.0)
    df["true_cvd_delta"] = df["taker_buy_base_vol"] - df["taker_sell_base_vol"]
    df["vol_per_trade"] = _safe_divide(df["volume"], df["num_trades"], default=np.nan)

    rows: list[dict[str, float | pd.Timestamp]] = []
    for hour, part in df.groupby("hour_timestamp", sort=True):
        part = part.sort_values("timestamp")
        count = int(len(part))
        if count < min_bars:
            continue
        volume = part["volume"].astype(float)
        trades = part["num_trades"].astype(float)
        taker_buy = part["taker_buy_base_vol"].astype(float)
        imbalance = part["taker_imbalance"].astype(float)
        buy_ratio = part["taker_buy_ratio"].astype(float)
        cvd = part["true_cvd_delta"].astype(float)
        vpt = part["vol_per_trade"].astype(float)
        close = part["close"].astype(float)
        open_ = part["open"].astype(float)
        high = part["high"].astype(float)
        low = part["low"].astype(float)
        total_volume = float(volume.sum())
        total_trades = float(trades.sum())
        total_buy = float(taker_buy.sum())
        first_half = part.iloc[: max(1, count // 2)]
        second_half = part.iloc[max(1, count // 2) :]
        early_volume = float(first_half["volume"].sum())
        late_volume = float(second_half["volume"].sum())
        early_cvd = float(first_half["true_cvd_delta"].sum())
        late_cvd = float(second_half["true_cvd_delta"].sum())
        early_vpt = float(first_half["volume"].sum() / first_half["num_trades"].replace(0, np.nan).sum())
        late_vpt = float(second_half["volume"].sum() / second_half["num_trades"].replace(0, np.nan).sum())
        hour_vpt = total_volume / total_trades if total_trades > 0 else np.nan
        last_volume = float(volume.iloc[-1])
        last_trades = float(trades.iloc[-1])
        last_buy = float(taker_buy.iloc[-1])
        last_imbalance = float(imbalance.iloc[-1])
        hour_open = float(open_.iloc[0])
        hour_close = float(close.iloc[-1])
        hour_high = float(high.max())
        hour_low = float(low.min())
        hour_range = max(hour_high - hour_low, 0.0)
        hour_range_pct = hour_range / hour_open if hour_open > 0 else np.nan
        hour_return = float(np.log(hour_close / hour_open)) if hour_open > 0 and hour_close > 0 else np.nan
        price_move_norm = (
            float(np.clip(hour_return / (hour_range_pct + 1e-12), -1.0, 1.0))
            if np.isfinite(hour_range_pct) and hour_range_pct > 0
            else np.nan
        )
        cvd_pressure_norm = float(cvd.sum() / total_volume) if total_volume > 0 else np.nan
        early_return = (
            float(np.log(float(first_half["close"].iloc[-1]) / float(first_half["open"].iloc[0])))
            if float(first_half["open"].iloc[0]) > 0 and float(first_half["close"].iloc[-1]) > 0
            else np.nan
        )
        late_return = (
            float(np.log(float(second_half["close"].iloc[-1]) / float(second_half["open"].iloc[0])))
            if float(second_half["open"].iloc[0]) > 0 and float(second_half["close"].iloc[-1]) > 0
            else np.nan
        )
        early_range = max(float(first_half["high"].max()) - float(first_half["low"].min()), 0.0)
        late_range = max(float(second_half["high"].max()) - float(second_half["low"].min()), 0.0)
        early_range_pct = early_range / float(first_half["open"].iloc[0]) if float(first_half["open"].iloc[0]) > 0 else np.nan
        late_range_pct = late_range / float(second_half["open"].iloc[0]) if float(second_half["open"].iloc[0]) > 0 else np.nan
        early_price_move_norm = (
            float(np.clip(early_return / (early_range_pct + 1e-12), -1.0, 1.0))
            if np.isfinite(early_range_pct) and early_range_pct > 0
            else np.nan
        )
        late_price_move_norm = (
            float(np.clip(late_return / (late_range_pct + 1e-12), -1.0, 1.0))
            if np.isfinite(late_range_pct) and late_range_pct > 0
            else np.nan
        )
        early_cvd_norm = float(early_cvd / total_volume) if total_volume > 0 else np.nan
        late_cvd_norm = float(late_cvd / total_volume) if total_volume > 0 else np.nan
        flow_alignment = (
            float(np.sign(cvd_pressure_norm) * price_move_norm)
            if np.isfinite(cvd_pressure_norm) and np.isfinite(price_move_norm)
            else np.nan
        )
        buy_absorption = (
            float(max(cvd_pressure_norm, 0.0) * max(-price_move_norm, 0.0))
            if np.isfinite(cvd_pressure_norm) and np.isfinite(price_move_norm)
            else np.nan
        )
        sell_absorption = (
            float(max(-cvd_pressure_norm, 0.0) * max(price_move_norm, 0.0))
            if np.isfinite(cvd_pressure_norm) and np.isfinite(price_move_norm)
            else np.nan
        )
        late_buy_absorption = (
            float(max(late_cvd_norm, 0.0) * max(-late_price_move_norm, 0.0))
            if np.isfinite(late_cvd_norm) and np.isfinite(late_price_move_norm)
            else np.nan
        )
        late_sell_absorption = (
            float(max(-late_cvd_norm, 0.0) * max(late_price_move_norm, 0.0))
            if np.isfinite(late_cvd_norm) and np.isfinite(late_price_move_norm)
            else np.nan
        )
        late_flow_reversal = (
            float(max(-np.sign(early_cvd_norm) * late_cvd_norm, 0.0))
            if np.isfinite(early_cvd_norm) and np.isfinite(late_cvd_norm) and early_cvd_norm != 0
            else 0.0
        )
        late_price_reversal = (
            float(max(-np.sign(early_price_move_norm) * late_price_move_norm, 0.0))
            if np.isfinite(early_price_move_norm) and np.isfinite(late_price_move_norm) and early_price_move_norm != 0
            else 0.0
        )
        late_flow_price_alignment = (
            float(np.sign(late_cvd_norm) * late_price_move_norm)
            if np.isfinite(late_cvd_norm) and np.isfinite(late_price_move_norm)
            else np.nan
        )

        row = {
            "timestamp": hour,
            f"{prefix}_bar_count": float(count),
            f"{prefix}_coverage": float(count / expected_bars) if expected_bars > 0 else np.nan,
            f"{prefix}_taker_imbalance_mean": float(imbalance.mean()),
            f"{prefix}_taker_imbalance_last": last_imbalance,
            f"{prefix}_taker_imbalance_late_minus_early": float(second_half["taker_imbalance"].mean() - first_half["taker_imbalance"].mean()),
            f"{prefix}_taker_imbalance_slope": _array_slope(imbalance.to_numpy()),
            f"{prefix}_buy_ratio_range": float(buy_ratio.max() - buy_ratio.min()),
            f"{prefix}_cvd_pressure_norm": cvd_pressure_norm,
            f"{prefix}_cvd_pressure_late_minus_early_norm": float((late_cvd - early_cvd) / total_volume) if total_volume > 0 else np.nan,
            f"{prefix}_cvd_slope_norm": float(_array_slope(cvd.cumsum().to_numpy()) / total_volume) if total_volume > 0 else np.nan,
            f"{prefix}_price_move_norm": price_move_norm,
            f"{prefix}_flow_price_alignment": flow_alignment,
            f"{prefix}_buy_absorption": buy_absorption,
            f"{prefix}_sell_absorption": sell_absorption,
            f"{prefix}_absorption_imbalance": buy_absorption - sell_absorption if np.isfinite(buy_absorption) and np.isfinite(sell_absorption) else np.nan,
            f"{prefix}_late_buy_absorption": late_buy_absorption,
            f"{prefix}_late_sell_absorption": late_sell_absorption,
            f"{prefix}_late_absorption_imbalance": late_buy_absorption - late_sell_absorption if np.isfinite(late_buy_absorption) and np.isfinite(late_sell_absorption) else np.nan,
            f"{prefix}_late_flow_reversal": late_flow_reversal,
            f"{prefix}_late_price_reversal": late_price_reversal,
            f"{prefix}_late_flow_price_alignment": late_flow_price_alignment,
            f"{prefix}_volume_share_last": float(last_volume / total_volume) if total_volume > 0 else np.nan,
            f"{prefix}_volume_share_late": float(late_volume / total_volume) if total_volume > 0 else np.nan,
            f"{prefix}_trade_share_last": float(last_trades / total_trades) if total_trades > 0 else np.nan,
            f"{prefix}_vpt_last_vs_hour": float(vpt.iloc[-1] / hour_vpt - 1.0) if hour_vpt and np.isfinite(hour_vpt) else np.nan,
            f"{prefix}_vpt_late_vs_early": float(late_vpt / early_vpt - 1.0) if early_vpt and np.isfinite(early_vpt) else np.nan,
            f"{prefix}_large_trade_concentration": float(vpt.max() / vpt.mean() - 1.0) if float(vpt.mean()) > 0 else np.nan,
            f"{prefix}_buy_volume_share_last": float(last_buy / total_buy) if total_buy > 0 else np.nan,
            f"{prefix}_aggressive_buy_burst": float(max(last_imbalance, 0.0) * last_volume / total_volume) if total_volume > 0 else np.nan,
            f"{prefix}_aggressive_sell_burst": float(max(-last_imbalance, 0.0) * last_volume / total_volume) if total_volume > 0 else np.nan,
        }
        rows.append(row)

    if not rows:
        empty = pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns, UTC]")})
        return FeatureResult(empty, [])

    out = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    stable_columns = [str(column) for column in (_config_get(cfg, ["stable_columns"], []) or [])]
    if stable_columns:
        stable_scores = _stable_rolling_score_frame(
            out,
            stable_columns,
            stable_window,
            stable_clip_abs,
            stable_transforms,
            error_context="features.intrahour_order_flow.stable_window",
            missing_ok=True,
            tanh_scale=stable_tanh_scale,
        )
        if not stable_scores.empty:
            out = pd.concat([out, stable_scores], axis=1)

    feature_columns = [column for column in select_feature_columns(out) if column.startswith(f"{prefix}_")]
    return FeatureResult(out, feature_columns)


def _add_stationary_features(df: pd.DataFrame, config: object, window: int) -> pd.DataFrame:
    if window <= 1:
        raise ValueError("features.stationarity.normalization_window must be greater than 1")

    rolling_volume = df["volume"].rolling(window, min_periods=window).mean()
    df["volume_log_zscore"] = _rolling_zscore(np.log1p(df["volume"].clip(lower=0)), window)
    df["true_cvd_delta_norm"] = _safe_divide(df["true_cvd_delta"], rolling_volume, default=np.nan)
    df["cvd_cumulative_rate_norm"] = _safe_divide(df["cvd_cumulative_rate"], rolling_volume, default=np.nan)
    df["vol_per_trade_log_zscore"] = _rolling_zscore(np.log1p(df["vol_per_trade"].clip(lower=0)), window)
    df["atr_14_pct"] = _safe_divide(df["atr_14"], df["close"], default=np.nan)

    if "close_denoised" in df.columns:
        df["close_denoised_log_return"] = _log_return(df["close_denoised"])
    if "volume_denoised" in df.columns:
        volume_denoised = df["volume_denoised"].clip(lower=0)
        df["volume_denoised_log_zscore"] = _rolling_zscore(np.log1p(volume_denoised), window)

    df = _add_structure_stability_features(df, config, window)
    df = _add_order_flow_v2_features(df, config, window)
    df = _add_bad_fold_context_features(df, config, window)
    return _add_order_flow_volatility_interaction_features(df, config)


def _add_structure_stability_features(df: pd.DataFrame, config: object, default_window: int) -> pd.DataFrame:
    cfg = _config_get(config, ["features", "structure_stability"], {})
    if not bool(_config_get(cfg, ["enabled"], False)):
        return df

    stable_window = int(_config_get(cfg, ["stable_window"], default_window))
    stable_clip_abs = float(_config_get(cfg, ["stable_clip_abs"], 5.0))
    stable_transforms = set(_config_get(cfg, ["stable_transforms"], ["zscore", "rank"]) or [])
    stable_tanh_scale = float(_config_get(cfg, ["stable_tanh_scale"], 2.0))
    source_columns = list(
        _config_get(
            cfg,
            ["source_columns"],
            [
                "log_return",
                "close_denoised_log_return",
                "realized_vol_14",
                "gk_vol_14",
                "atr_14_pct",
                "adx_14",
                "vwap_dist_atr",
                "volume_log_zscore",
                "volume_denoised_log_zscore",
            ],
        )
        or []
    )
    stable_scores = _stable_rolling_score_frame(
        df,
        [str(column) for column in source_columns],
        stable_window,
        stable_clip_abs,
        stable_transforms,
        error_context="features.structure_stability.stable_window",
        missing_ok=True,
        tanh_scale=stable_tanh_scale,
    )
    if stable_scores.empty:
        return df
    return pd.concat([df, stable_scores], axis=1)


def _add_order_flow_v2_features(df: pd.DataFrame, config: object, default_window: int) -> pd.DataFrame:
    cfg = _config_get(config, ["features", "order_flow_v2"], {})
    if not bool(_config_get(cfg, ["enabled"], False)):
        return df

    ratio_window = int(_config_get(cfg, ["ratio_zscore_window"], default_window))
    ratio_delta_periods = int(_config_get(cfg, ["ratio_delta_periods"], 1))
    slope_window = int(_config_get(cfg, ["imbalance_slope_window"], 24))
    pressure_windows = list(_config_get(cfg, ["pressure_windows"], [3, 6, 12, 24]) or [])
    efficiency_epsilon = float(_config_get(cfg, ["efficiency_epsilon"], 1e-3))
    stable_window = int(_config_get(cfg, ["stable_window"], default_window))
    stable_clip_abs = float(_config_get(cfg, ["stable_clip_abs"], 5.0))
    stable_transforms = set(_config_get(cfg, ["stable_transforms"], ["zscore", "rank"]) or [])
    stable_tanh_scale = float(_config_get(cfg, ["stable_tanh_scale"], 2.0))
    pressure_spread_pairs = list(_config_get(cfg, ["pressure_spread_pairs"], [[24, 12]]) or [])

    taker_imbalance = (df["taker_buy_ratio"] - df["taker_sell_ratio"]).clip(-1.0, 1.0)
    large_trade_intensity = (df["vpt_zscore"] - df["vpt_zscore"].rolling(default_window, min_periods=default_window).median()).clip(lower=0.0)
    signed_large_trade_pressure = large_trade_intensity * taker_imbalance
    normalized_return = _safe_divide(
        df["log_return"],
        df["realized_vol_14"].replace(0, np.nan),
        default=0.0,
    ).clip(-10.0, 10.0)
    abs_cvd_pressure = df["true_cvd_delta_norm"].abs()

    df["taker_imbalance"] = taker_imbalance
    df["taker_buy_ratio_zscore"] = _rolling_zscore(df["taker_buy_ratio"], ratio_window)
    df["taker_buy_ratio_delta"] = df["taker_buy_ratio"].diff(ratio_delta_periods)
    df["taker_imbalance_slope"] = _rolling_slope(taker_imbalance, slope_window)
    df["signed_large_trade_pressure"] = signed_large_trade_pressure
    df["orderflow_efficiency"] = _safe_divide(
        normalized_return,
        abs_cvd_pressure + efficiency_epsilon,
        default=0.0,
    ).clip(-10.0, 10.0)
    df["absorption_pressure"] = (-normalized_return * taker_imbalance).clip(-10.0, 10.0)
    df["cvd_price_divergence"] = (df["true_cvd_delta_norm"] - normalized_return).clip(-10.0, 10.0)
    stable_frames = [
        _stable_rolling_score_frame(
            df,
            [
                "signed_large_trade_pressure",
                "orderflow_efficiency",
                "absorption_pressure",
                "cvd_price_divergence",
            ],
            stable_window,
            stable_clip_abs,
            stable_transforms,
            error_context="features.order_flow_v2.stable_window",
            tanh_scale=stable_tanh_scale,
        )
    ]

    for item in pressure_windows:
        window = int(item)
        if window <= 1:
            raise ValueError("features.order_flow_v2.pressure_windows values must be greater than 1")
        df[f"taker_imbalance_mean_{window}"] = taker_imbalance.rolling(window, min_periods=window).mean()
        df[f"cvd_pressure_{window}"] = df["true_cvd_delta_norm"].rolling(window, min_periods=window).sum()
        df[f"large_trade_pressure_{window}"] = signed_large_trade_pressure.rolling(window, min_periods=window).mean()
        df[f"absorption_pressure_{window}"] = df["absorption_pressure"].rolling(window, min_periods=window).mean()
        df[f"cvd_price_divergence_{window}"] = df["cvd_price_divergence"].rolling(window, min_periods=window).mean()
        stable_frames.append(
            _stable_rolling_score_frame(
                df,
                [
                    f"cvd_pressure_{window}",
                    f"large_trade_pressure_{window}",
                    f"absorption_pressure_{window}",
                    f"cvd_price_divergence_{window}",
                ],
                stable_window,
                stable_clip_abs,
                stable_transforms,
                error_context="features.order_flow_v2.stable_window",
                tanh_scale=stable_tanh_scale,
            )
        )
    for pair in pressure_spread_pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("features.order_flow_v2.pressure_spread_pairs values must be [slow, fast] pairs")
        slow, fast = int(pair[0]), int(pair[1])
        if slow <= 1 or fast <= 1:
            raise ValueError("features.order_flow_v2.pressure_spread_pairs values must be greater than 1")
        slow_column = f"large_trade_pressure_{slow}"
        fast_column = f"large_trade_pressure_{fast}"
        if slow_column not in df.columns or fast_column not in df.columns:
            raise ValueError(
                "features.order_flow_v2.pressure_spread_pairs must reference configured pressure_windows; "
                f"missing {slow_column!r} or {fast_column!r}"
            )
        spread_column = f"large_trade_pressure_{slow}_minus_{fast}"
        df[spread_column] = df[slow_column] - df[fast_column]
        stable_frames.append(
            _stable_rolling_score_frame(
                df,
                [spread_column],
                stable_window,
                stable_clip_abs,
                stable_transforms,
                error_context="features.order_flow_v2.stable_window",
                tanh_scale=stable_tanh_scale,
            )
        )
    stable_frames = [frame for frame in stable_frames if not frame.empty]
    if stable_frames:
        return pd.concat([df, *stable_frames], axis=1)
    return df


def _feature_alias(column: str) -> str:
    aliases = {
        "signed_large_trade_pressure_stable_rank": "signed_ltp",
        "realized_vol_14_stable_rank": "rv14_rank",
        "gk_vol_14_stable_rank": "gk14_rank",
        "atr_14_pct_stable_rank": "atr14_rank",
        "large_trade_ratio": "ltr",
        "large_trade_ratio_stable_rank": "ltr_rank",
        "large_trade_ratio_stable_tanh": "ltr_tanh",
    }
    if column in aliases:
        return aliases[column]
    match = re.fullmatch(r"taker_imbalance_mean_(\d+)(?:_stable_(rank|zscore|tanh))?", column)
    if match:
        suffix = f"_{match.group(2)}" if match.group(2) else ""
        return f"taker_mean{match.group(1)}{suffix}"
    match = re.fullmatch(r"large_trade_pressure_(\d+)_stable_(rank|zscore|tanh)", column)
    if match:
        return f"ltp{match.group(1)}_{match.group(2)}"
    return column.replace("_stable_rank", "_rank").replace("_stable_zscore", "_zscore")


def _bounded_interaction_source(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)


def _stable_transform_column(column: str, transform: str) -> str:
    normalized = transform.strip().lower()
    if normalized in {"rank", "zscore", "tanh"}:
        return f"{column}_stable_{normalized}"
    if normalized in {"stable_rank", "stable_zscore", "stable_tanh"}:
        return f"{column}_{normalized}"
    raise ValueError(f"Unsupported bad-fold context transform: {transform}")


def _add_bad_fold_context_features(df: pd.DataFrame, config: object, default_window: int) -> pd.DataFrame:
    """Add narrow conditional context features for bad-fold suspects.

    These are deliberately separate from the broad flow-volatility interaction
    family. They target the observed May 2026 failure mode where an otherwise
    useful 4H taker-flow feature reverses when large-trade context shifts.
    """

    cfg = _config_get(config, ["features", "bad_fold_context"], {})
    if not bool(_config_get(cfg, ["enabled"], False)):
        return df

    stable_sources = [str(column) for column in (_config_get(cfg, ["stable_source_columns"], []) or [])]
    interaction_pairs = list(_config_get(cfg, ["interaction_pairs"], []) or [])
    if not stable_sources and not interaction_pairs:
        return df

    stable_window = int(_config_get(cfg, ["stable_window"], default_window))
    stable_clip_abs = float(_config_get(cfg, ["stable_clip_abs"], 5.0))
    stable_transforms = set(_config_get(cfg, ["stable_transforms"], ["rank", "tanh"]) or [])
    stable_tanh_scale = float(_config_get(cfg, ["stable_tanh_scale"], 2.0))

    stable_scores = _stable_rolling_score_frame(
        df,
        stable_sources,
        stable_window,
        stable_clip_abs,
        stable_transforms,
        error_context="features.bad_fold_context.stable_window",
        missing_ok=True,
        tanh_scale=stable_tanh_scale,
    )
    if not stable_scores.empty:
        df = pd.concat([df, stable_scores], axis=1)

    additions: dict[str, pd.Series] = {}
    for item in interaction_pairs:
        if not isinstance(item, dict):
            raise ValueError("features.bad_fold_context.interaction_pairs entries must be mappings")
        source_column = str(item.get("source", ""))
        context_column = str(item.get("context", ""))
        if not source_column or not context_column:
            raise ValueError("features.bad_fold_context interaction pairs require source and context")
        if source_column not in df.columns:
            continue

        transforms = [str(transform) for transform in (item.get("context_transforms") or ["stable_rank"])]
        modes = {str(mode) for mode in (item.get("modes") or ["signed", "high", "low"])}
        source = _bounded_interaction_source(df[source_column])
        source_name = _feature_alias(source_column)
        for transform in transforms:
            transformed_context = _stable_transform_column(context_column, transform)
            if transformed_context not in df.columns:
                continue
            context = _bounded_interaction_source(df[transformed_context])
            context_name = _feature_alias(transformed_context)
            base_name = f"{source_name}_x_{context_name}"
            if "signed" in modes:
                additions[f"{base_name}_signed"] = source * context
            if "high" in modes:
                additions[f"{base_name}_high"] = source * context.clip(lower=0.0)
            if "low" in modes:
                additions[f"{base_name}_low"] = source * (-context.clip(upper=0.0))

    if not additions:
        return df
    return pd.concat([df, pd.DataFrame(additions, index=df.index)], axis=1)


def _add_order_flow_volatility_interaction_features(df: pd.DataFrame, config: object) -> pd.DataFrame:
    cfg = _config_get(config, ["features", "order_flow_volatility_interactions"], {})
    if not bool(_config_get(cfg, ["enabled"], False)):
        return df

    flow_columns = [str(column) for column in (_config_get(cfg, ["flow_columns"], []) or [])]
    volatility_columns = [str(column) for column in (_config_get(cfg, ["volatility_columns"], []) or [])]
    modes = {str(mode) for mode in (_config_get(cfg, ["modes"], ["signed", "high", "low"]) or [])}
    if not flow_columns or not volatility_columns or not modes:
        return df

    additions: dict[str, pd.Series] = {}
    for flow_column in flow_columns:
        if flow_column not in df.columns:
            continue
        flow = _bounded_interaction_source(df[flow_column])
        flow_name = _feature_alias(flow_column)
        for volatility_column in volatility_columns:
            if volatility_column not in df.columns:
                continue
            volatility = _bounded_interaction_source(df[volatility_column])
            volatility_name = _feature_alias(volatility_column)
            base_name = f"{flow_name}_x_{volatility_name}"
            if "signed" in modes:
                additions[f"{base_name}_signed"] = flow * volatility
            if "high" in modes:
                additions[f"{base_name}_high"] = flow * volatility.clip(lower=0.0)
            if "low" in modes:
                additions[f"{base_name}_low"] = flow * (-volatility.clip(upper=0.0))

    if not additions:
        return df
    return pd.concat([df, pd.DataFrame(additions, index=df.index)], axis=1)


def _stable_rolling_score_frame(
    df: pd.DataFrame,
    columns: list[str],
    window: int,
    clip_abs: float,
    transforms: set[str],
    *,
    error_context: str,
    missing_ok: bool = False,
    tanh_scale: float = 2.0,
) -> pd.DataFrame:
    if window <= 1:
        raise ValueError(f"{error_context} must be greater than 1")
    if tanh_scale <= 0:
        raise ValueError(f"{error_context} tanh_scale must be greater than 0")
    additions: dict[str, pd.Series] = {}
    for column in columns:
        if column not in df.columns:
            if missing_ok:
                continue
            raise KeyError(f"Stable source column is missing: {column}")
        series = df[column].replace([np.inf, -np.inf], np.nan)
        zscore = None
        if "zscore" in transforms or "tanh" in transforms:
            zscore = _rolling_zscore(series, window).clip(-clip_abs, clip_abs)
        if "zscore" in transforms:
            additions[f"{column}_stable_zscore"] = zscore
        if "rank" in transforms:
            additions[f"{column}_stable_rank"] = _rolling_rank_score(series, window)
        if "tanh" in transforms:
            additions[f"{column}_stable_tanh"] = np.tanh(zscore / tanh_scale)
    return pd.DataFrame(additions, index=df.index)


def select_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = RAW_COLUMNS | METADATA_COLUMNS | LABEL_COLUMNS
    excluded_prefixes = ("pred_", "regime_", "fold")
    columns: list[str] = []
    for column in frame.columns:
        if column in excluded:
            continue
        if any(column.startswith(prefix) for prefix in excluded_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    return sorted(columns)


def filter_feature_columns(feature_columns: list[str], config: object) -> list[str]:
    exclude_columns = set(_config_get(config, ["features", "exclude_columns"], []) or [])
    exclude_patterns = list(_config_get(config, ["features", "exclude_patterns"], []) or [])
    profile = resolve_feature_profile(config)
    include_patterns = list(profile.get("include_patterns", []) or [])
    profile_exclude_patterns = list(profile.get("exclude_patterns", []) or [])
    stationarity_cfg = _config_get(config, ["features", "stationarity"], {})
    if bool(_config_get(stationarity_cfg, ["exclude_nonstationary"], False)):
        exclude_patterns.extend(list(_config_get(stationarity_cfg, ["exclude_patterns"], []) or []))
    exclude_patterns.extend(profile_exclude_patterns)
    filtered = []
    raw_order_flow_v2_columns = raw_order_flow_v2_model_exclusions(config)
    for column in feature_columns:
        if column in exclude_columns:
            continue
        if column in raw_order_flow_v2_columns:
            continue
        if include_patterns and not any(fnmatch(column, pattern) for pattern in include_patterns):
            continue
        if any(fnmatch(column, pattern) for pattern in exclude_patterns):
            continue
        filtered.append(column)
    if not filtered:
        raise ValueError("Feature filtering removed every feature column")
    return filtered


def feature_availability_columns(feature_columns: list[str], config: object) -> list[str]:
    """Columns that must be finite before saving the feature matrix.

    The model profile controls trainable inputs, while HMM and labeling need a
    few non-model columns later. Inactive experimental features must not change
    the row universe simply because their warmup window is longer.
    """

    available = set(feature_columns)
    required = list(filter_feature_columns(feature_columns, config))
    extra_columns = list(_config_get(config, ["hmm", "features"], []) or [])
    atr_column = str(_config_get(config, ["labeling", "atr_column"], "atr_14"))
    extra_columns.append(atr_column)

    missing = [column for column in extra_columns if column not in available]
    if missing:
        raise ValueError(f"Missing required non-model feature columns: {missing}")
    for column in extra_columns:
        if column not in required:
            required.append(column)
    return required


def resolve_feature_profile(config: object) -> dict[str, object]:
    active = _config_get(config, ["features", "active_profile"], None)
    if not active:
        return {"name": None, "include_patterns": [], "exclude_patterns": []}

    profiles = _config_get(config, ["features", "profiles"], {}) or {}
    if not isinstance(profiles, dict) or active not in profiles:
        raise ValueError(f"Unknown features.active_profile: {active}")

    def load_profile(name: str, seen: set[str] | None = None) -> dict[str, object]:
        seen = set() if seen is None else seen
        if name in seen:
            raise ValueError(f"Cyclic feature profile inheritance detected at {name}")
        seen.add(name)
        current = profiles[name]
        if not isinstance(current, dict):
            raise ValueError(f"Feature profile must be a mapping: {name}")
        parent_name = current.get("inherit")
        parent = load_profile(str(parent_name), seen) if parent_name else {"include_patterns": [], "exclude_patterns": []}
        include_patterns = _dedupe_patterns(
            list(parent.get("include_patterns", []) or []) + list(current.get("include_patterns", []) or [])
        )
        if "exclude_patterns" in current:
            exclude_patterns = list(current.get("exclude_patterns", []) or [])
        else:
            exclude_patterns = list(parent.get("exclude_patterns", []) or [])
        exclude_patterns.extend(list(current.get("append_exclude_patterns", []) or []))
        return {
            "name": name,
            "description": current.get("description", ""),
            "include_patterns": include_patterns,
            "exclude_patterns": _dedupe_patterns(exclude_patterns),
        }

    return load_profile(str(active))


def _dedupe_patterns(patterns: list[object]) -> list[str]:
    deduped: list[str] = []
    for pattern in patterns:
        text = str(pattern)
        if text not in deduped:
            deduped.append(text)
    return deduped


def raw_order_flow_v2_model_exclusions(config: object) -> set[str]:
    cfg = _config_get(config, ["features", "order_flow_v2"], {})
    if not bool(_config_get(cfg, ["enabled"], False)):
        return set()
    if not bool(_config_get(cfg, ["stable_only"], False)):
        return set()
    pressure_windows = [int(item) for item in (list(_config_get(cfg, ["pressure_windows"], [3, 6, 12, 24]) or []))]
    pressure_spread_pairs = list(_config_get(cfg, ["pressure_spread_pairs"], [[24, 12]]) or [])
    base = {
        "signed_large_trade_pressure",
        "orderflow_efficiency",
        "absorption_pressure",
        "cvd_price_divergence",
    }
    for window in pressure_windows:
        base.update(
            {
                f"cvd_pressure_{window}",
                f"large_trade_pressure_{window}",
                f"absorption_pressure_{window}",
                f"cvd_price_divergence_{window}",
            }
        )
    for pair in pressure_spread_pairs:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            base.add(f"large_trade_pressure_{int(pair[0])}_minus_{int(pair[1])}")
    return base | {f"4h_{column}" for column in base}


def compute_futures_metrics_features(metrics_frame: pd.DataFrame, config: object) -> FeatureResult:
    """Compute causal futures positioning features from Binance Vision metrics snapshots."""

    cfg = _config_get(config, ["features", "futures_context"], {})
    if not bool(_config_get(cfg, ["enabled"], False)) or metrics_frame is None or metrics_frame.empty:
        empty = pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns, UTC]")})
        return FeatureResult(empty, [])

    prefix = str(_config_get(cfg, ["prefix"], "fut")).replace("-", "_")
    stable_window = int(_config_get(cfg, ["stable_window"], 288))
    stable_clip_abs = float(_config_get(cfg, ["stable_clip_abs"], 5.0))
    stable_transforms = set(_config_get(cfg, ["stable_transforms"], ["zscore", "rank"]) or [])
    stable_tanh_scale = float(_config_get(cfg, ["stable_tanh_scale"], 2.0))
    oi_change_windows = [int(item) for item in (_config_get(cfg, ["oi_change_windows"], [12, 36, 96, 288]) or [])]

    df = metrics_frame.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    numeric_columns = [
        "sum_open_interest",
        "sum_open_interest_value",
        "count_toptrader_long_short_ratio",
        "sum_toptrader_long_short_ratio",
        "count_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    oi_log = np.log(df["sum_open_interest"].where(df["sum_open_interest"] > 0))
    oi_value_log = np.log(df["sum_open_interest_value"].where(df["sum_open_interest_value"] > 0))
    df[f"{prefix}_oi_log_return"] = oi_log.diff()
    df[f"{prefix}_oi_value_log_return"] = oi_value_log.diff()

    stable_sources = [f"{prefix}_oi_log_return", f"{prefix}_oi_value_log_return"]
    for window in oi_change_windows:
        if window <= 0:
            raise ValueError("features.futures_context.oi_change_windows values must be positive")
        df[f"{prefix}_oi_change_{window}"] = oi_log - oi_log.shift(window)
        df[f"{prefix}_oi_value_change_{window}"] = oi_value_log - oi_value_log.shift(window)
        stable_sources.extend([f"{prefix}_oi_change_{window}", f"{prefix}_oi_value_change_{window}"])

    ratio_sources = {
        f"{prefix}_toptrader_count_long_short_log_ratio": "count_toptrader_long_short_ratio",
        f"{prefix}_toptrader_sum_long_short_log_ratio": "sum_toptrader_long_short_ratio",
        f"{prefix}_global_long_short_log_ratio": "count_long_short_ratio",
        f"{prefix}_taker_long_short_vol_log_ratio": "sum_taker_long_short_vol_ratio",
    }
    for output, source in ratio_sources.items():
        if source not in df.columns:
            continue
        df[output] = np.log(df[source].where(df[source] > 0))
        stable_sources.append(output)

    stable_scores = _stable_rolling_score_frame(
        df,
        stable_sources,
        stable_window,
        stable_clip_abs,
        stable_transforms,
        error_context="features.futures_context.stable_window",
        missing_ok=True,
        tanh_scale=stable_tanh_scale,
    )
    if not stable_scores.empty:
        df = pd.concat([df, stable_scores], axis=1)

    feature_columns = select_feature_columns(df)
    return FeatureResult(df, feature_columns)


def compute_funding_rate_features(funding_frame: pd.DataFrame, config: object) -> FeatureResult:
    """Compute causal funding-rate context features."""

    cfg = _config_get(config, ["features", "futures_context"], {})
    if not bool(_config_get(cfg, ["enabled"], False)) or funding_frame is None or funding_frame.empty:
        empty = pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns, UTC]")})
        return FeatureResult(empty, [])

    prefix = str(_config_get(cfg, ["prefix"], "fut")).replace("-", "_")
    stable_window = int(_config_get(cfg, ["funding_stable_window"], _config_get(cfg, ["stable_window"], 288)))
    stable_clip_abs = float(_config_get(cfg, ["stable_clip_abs"], 5.0))
    stable_transforms = set(_config_get(cfg, ["stable_transforms"], ["zscore", "rank"]) or [])
    stable_tanh_scale = float(_config_get(cfg, ["stable_tanh_scale"], 2.0))
    funding_windows = [int(item) for item in (_config_get(cfg, ["funding_windows"], [3, 6, 12]) or [])]

    df = funding_frame.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    df[f"{prefix}_funding_rate"] = df["funding_rate"].clip(-0.05, 0.05)
    df[f"{prefix}_funding_rate_delta"] = df[f"{prefix}_funding_rate"].diff()
    stable_sources = [f"{prefix}_funding_rate", f"{prefix}_funding_rate_delta"]

    for window in funding_windows:
        if window <= 0:
            raise ValueError("features.futures_context.funding_windows values must be positive")
        df[f"{prefix}_funding_sum_{window}"] = df[f"{prefix}_funding_rate"].rolling(window, min_periods=window).sum()
        df[f"{prefix}_funding_mean_{window}"] = df[f"{prefix}_funding_rate"].rolling(window, min_periods=window).mean()
        stable_sources.extend([f"{prefix}_funding_sum_{window}", f"{prefix}_funding_mean_{window}"])

    stable_scores = _stable_rolling_score_frame(
        df,
        stable_sources,
        stable_window,
        stable_clip_abs,
        stable_transforms,
        error_context="features.futures_context.funding_stable_window",
        missing_ok=True,
        tanh_scale=stable_tanh_scale,
    )
    if not stable_scores.empty:
        df = pd.concat([df, stable_scores], axis=1)

    feature_columns = select_feature_columns(df)
    return FeatureResult(df, feature_columns)


def build_feature_matrix(
    primary_frame: pd.DataFrame,
    htf_frame: pd.DataFrame,
    config: object,
    intrabar_frame: pd.DataFrame | None = None,
    futures_metrics_frame: pd.DataFrame | None = None,
    funding_frame: pd.DataFrame | None = None,
) -> FeatureResult:
    """Build 1H, delayed 4H, and optional causal futures-context features."""

    primary = compute_bar_features(primary_frame, config).frame
    htf_result = compute_bar_features(htf_frame, config)
    htf = htf_result.frame

    shift_hours = int(_config_get(config, ["features", "mtf", "shift_hours"], 4))
    htf_features = htf[["timestamp", *htf_result.feature_columns]].copy()
    htf_features["4h_source_timestamp"] = htf_features["timestamp"]
    htf_features["timestamp"] = htf_features["timestamp"] + pd.Timedelta(hours=shift_hours)
    htf_features["4h_available_timestamp"] = htf_features["timestamp"]
    htf_features = htf_features.rename(
        columns={column: f"4h_{column}" for column in htf_result.feature_columns}
    )

    merged = pd.merge_asof(
        primary.sort_values("timestamp"),
        htf_features.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )

    if intrabar_frame is not None and bool(_config_get(config, ["features", "intrahour_order_flow", "enabled"], False)):
        intrahour_result = compute_intrahour_order_flow_features(intrabar_frame, config)
        if intrahour_result.feature_columns:
            intrahour_features = intrahour_result.frame[["timestamp", *intrahour_result.feature_columns]].copy()
            merged = merged.merge(intrahour_features, on="timestamp", how="left")
            missing_col = f"{_config_get(config, ['features', 'intrahour_order_flow', 'prefix'], 'ih15')}_missing"
            missing = merged[intrahour_result.feature_columns].isna().all(axis=1)
            merged[missing_col] = missing.astype(float)
            intrahour_fill_columns = [*intrahour_result.feature_columns, missing_col]
        else:
            intrahour_fill_columns = []
    else:
        intrahour_fill_columns = []

    context_fill_columns: list[str] = []
    if futures_metrics_frame is not None and bool(_config_get(config, ["features", "futures_context", "enabled"], False)):
        metrics_result = compute_futures_metrics_features(futures_metrics_frame, config)
        if metrics_result.feature_columns:
            metrics_features = metrics_result.frame[["timestamp", *metrics_result.feature_columns]].copy()
            tolerance_minutes = int(_config_get(config, ["features", "futures_context", "metrics_merge_tolerance_minutes"], 90))
            merged = pd.merge_asof(
                merged.sort_values("timestamp"),
                metrics_features.sort_values("timestamp"),
                on="timestamp",
                direction="backward",
                tolerance=pd.Timedelta(minutes=tolerance_minutes),
            )
            missing = merged[metrics_result.feature_columns].isna().all(axis=1)
            missing_col = f"{_config_get(config, ['features', 'futures_context', 'prefix'], 'fut')}_metrics_missing"
            merged[missing_col] = missing.astype(float)
            context_fill_columns.extend([*metrics_result.feature_columns, missing_col])

    if funding_frame is not None and bool(_config_get(config, ["features", "futures_context", "enabled"], False)):
        funding_result = compute_funding_rate_features(funding_frame, config)
        if funding_result.feature_columns:
            funding_features = funding_result.frame[["timestamp", *funding_result.feature_columns]].copy()
            tolerance_minutes = int(_config_get(config, ["features", "futures_context", "funding_merge_tolerance_minutes"], 540))
            merged = pd.merge_asof(
                merged.sort_values("timestamp"),
                funding_features.sort_values("timestamp"),
                on="timestamp",
                direction="backward",
                tolerance=pd.Timedelta(minutes=tolerance_minutes),
            )
            missing = merged[funding_result.feature_columns].isna().all(axis=1)
            missing_col = f"{_config_get(config, ['features', 'futures_context', 'prefix'], 'fut')}_funding_missing"
            merged[missing_col] = missing.astype(float)
            context_fill_columns.extend([*funding_result.feature_columns, missing_col])

    warmup_rows = int(_config_get(config, ["features", "warmup_rows"], 300))
    if warmup_rows > 0:
        merged = merged.iloc[warmup_rows:].copy()
    neutral_fill_columns = [*intrahour_fill_columns, *context_fill_columns]
    if neutral_fill_columns:
        passthrough_columns = [column for column in merged.columns if column not in neutral_fill_columns]
        merged[passthrough_columns] = merged[passthrough_columns].ffill()
        merged[neutral_fill_columns] = merged[neutral_fill_columns].fillna(0.0)
    else:
        merged = merged.ffill()
    feature_columns = select_feature_columns(merged)
    model_feature_columns = filter_feature_columns(feature_columns, config)
    availability_columns = feature_availability_columns(feature_columns, config)
    merged = merged.dropna(subset=availability_columns).reset_index(drop=True)
    if merged[availability_columns].isna().any().any():
        bad = merged[availability_columns].columns[merged[availability_columns].isna().any()].tolist()
        raise ValueError(f"Feature matrix contains NaNs after warmup/fill: {bad}")
    return FeatureResult(merged, model_feature_columns)
