from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from yenibot.features import (
    build_feature_matrix,
    compute_funding_rate_features,
    compute_futures_metrics_features,
    compute_intrahour_order_flow_features,
)
from yenibot.features.wavelet import causal_wavelet_denoise


def test_4h_alignment_delays_bar_until_complete(synthetic_klines, tiny_config) -> None:
    primary = synthetic_klines(36, "1h")
    htf = synthetic_klines(10, "4h")
    result = build_feature_matrix(primary, htf, tiny_config)
    frame = result.frame

    row_23 = frame.loc[frame["timestamp"] == pd.Timestamp("2022-01-01 23:00", tz="UTC")].iloc[0]
    row_24 = frame.loc[frame["timestamp"] == pd.Timestamp("2022-01-02 00:00", tz="UTC")].iloc[0]

    assert row_23["4h_source_timestamp"] == pd.Timestamp("2022-01-01 16:00", tz="UTC")
    assert row_24["4h_source_timestamp"] == pd.Timestamp("2022-01-01 20:00", tz="UTC")
    assert row_24["4h_available_timestamp"] == pd.Timestamp("2022-01-02 00:00", tz="UTC")


def test_causal_wavelet_value_unchanged_when_future_appended() -> None:
    pytest.importorskip("pywt")
    series = pd.Series(range(320), dtype=float)
    extended = pd.Series(range(380), dtype=float)
    base = causal_wavelet_denoise(series, window=64)
    future = causal_wavelet_denoise(extended, window=64)
    pd.testing.assert_series_equal(base.dropna(), future.iloc[: len(base)].dropna(), check_names=False)


def test_stationary_features_are_causal_when_future_rows_appended(synthetic_klines, tiny_config) -> None:
    primary = synthetic_klines(80, "1h")
    htf = synthetic_klines(24, "4h")
    extended_primary = synthetic_klines(96, "1h")
    extended_htf = synthetic_klines(28, "4h")

    base = build_feature_matrix(primary, htf, tiny_config).frame
    extended = build_feature_matrix(extended_primary, extended_htf, tiny_config).frame
    timestamp = pd.Timestamp("2022-01-03 12:00", tz="UTC")
    columns = [
        "volume_log_zscore",
        "true_cvd_delta_norm",
        "cvd_cumulative_rate_norm",
        "vol_per_trade_log_zscore",
        "atr_14_pct",
        "4h_true_cvd_delta_norm",
        "4h_cvd_cumulative_rate_norm",
        "4h_atr_14_pct",
    ]

    base_row = base.loc[base["timestamp"] == timestamp, columns].iloc[0]
    extended_row = extended.loc[extended["timestamp"] == timestamp, columns].iloc[0]
    pd.testing.assert_series_equal(base_row, extended_row, check_names=False)


def test_structure_stability_features_are_causal_when_future_rows_appended(synthetic_klines, tiny_config) -> None:
    config = copy.deepcopy(tiny_config)
    config["features"]["structure_stability"] = {
        "enabled": True,
        "stable_window": 4,
        "stable_clip_abs": 3.0,
        "stable_transforms": ["zscore", "rank"],
        "source_columns": [
            "log_return",
            "realized_vol_14",
            "gk_vol_14",
            "atr_14_pct",
            "adx_14",
            "vwap_dist_atr",
            "volume_log_zscore",
        ],
    }
    primary = synthetic_klines(112, "1h")
    htf = synthetic_klines(34, "4h")
    extended_primary = synthetic_klines(128, "1h")
    extended_htf = synthetic_klines(38, "4h")

    base = build_feature_matrix(primary, htf, config).frame
    extended = build_feature_matrix(extended_primary, extended_htf, config).frame
    timestamp = pd.Timestamp("2022-01-03 12:00", tz="UTC")
    columns = [
        "log_return_stable_zscore",
        "gk_vol_14_stable_rank",
        "atr_14_pct_stable_zscore",
        "vwap_dist_atr_stable_rank",
        "volume_log_zscore_stable_zscore",
        "4h_gk_vol_14_stable_rank",
        "4h_volume_log_zscore_stable_zscore",
    ]

    assert set(columns).issubset(base.columns)
    base_row = base.loc[base["timestamp"] == timestamp, columns].iloc[0]
    extended_row = extended.loc[extended["timestamp"] == timestamp, columns].iloc[0]
    pd.testing.assert_series_equal(base_row, extended_row, check_names=False)


def test_inactive_stable_features_do_not_change_feature_matrix_rows(synthetic_klines, tiny_config) -> None:
    base_config = copy.deepcopy(tiny_config)
    base_config["features"]["profiles"] = {
        "baseline": {
            "include_patterns": [
                "*log_return",
                "*gk_vol_14",
                "*adx_14",
                "*true_cvd_zscore",
                "*vwap_dist_atr",
                "*atr_14_pct",
            ],
            "exclude_patterns": ["*_stable_*"],
        }
    }
    base_config["features"]["active_profile"] = "baseline"
    base_config["labeling"]["atr_column"] = "atr_14"
    stable_config = copy.deepcopy(base_config)
    stable_config["features"]["structure_stability"] = {
        "enabled": True,
        "stable_window": 24,
        "stable_clip_abs": 3.0,
        "stable_transforms": ["zscore", "rank"],
        "source_columns": ["gk_vol_14", "atr_14_pct", "vwap_dist_atr"],
    }
    primary = synthetic_klines(96, "1h")
    htf = synthetic_klines(30, "4h")

    base = build_feature_matrix(primary, htf, base_config)
    stable = build_feature_matrix(primary, htf, stable_config)

    assert base.feature_columns == stable.feature_columns
    assert len(base.frame) == len(stable.frame)
    pd.testing.assert_series_equal(base.frame["timestamp"], stable.frame["timestamp"])
    assert stable.frame["gk_vol_14_stable_zscore"].isna().any()
    assert not stable.frame[stable.feature_columns].isna().any().any()


def test_order_flow_v2_features_are_causal_when_future_rows_appended(synthetic_klines, tiny_config) -> None:
    config = copy.deepcopy(tiny_config)
    config["features"]["order_flow_v2"] = {
        "enabled": True,
        "ratio_zscore_window": 4,
        "ratio_delta_periods": 1,
        "imbalance_slope_window": 4,
        "pressure_windows": [3, 4],
        "pressure_spread_pairs": [[4, 3]],
        "efficiency_epsilon": 0.001,
        "stable_only": True,
        "stable_window": 4,
        "stable_clip_abs": 3.0,
        "stable_tanh_scale": 2.0,
        "stable_transforms": ["zscore", "rank", "tanh"],
    }
    config["features"]["structure_stability"] = {
        "enabled": True,
        "stable_window": 4,
        "stable_clip_abs": 3.0,
        "stable_transforms": ["zscore", "rank"],
        "source_columns": ["realized_vol_14", "gk_vol_14", "atr_14_pct"],
    }
    config["features"]["order_flow_volatility_interactions"] = {
        "enabled": True,
        "flow_columns": [
            "taker_imbalance",
            "signed_large_trade_pressure_stable_rank",
            "large_trade_pressure_4_stable_rank",
        ],
        "volatility_columns": ["realized_vol_14_stable_rank", "gk_vol_14_stable_rank"],
        "modes": ["signed", "high", "low"],
    }
    config["features"]["bad_fold_context"] = {
        "enabled": True,
        "stable_window": 4,
        "stable_clip_abs": 3.0,
        "stable_tanh_scale": 2.0,
        "stable_transforms": ["rank"],
        "stable_source_columns": ["taker_imbalance_mean_4", "large_trade_ratio"],
        "interaction_pairs": [
            {
                "source": "taker_imbalance_mean_4",
                "context": "large_trade_ratio",
                "context_transforms": ["stable_rank"],
                "modes": ["signed", "high", "low"],
            }
        ],
    }
    primary = synthetic_klines(160, "1h")
    htf = synthetic_klines(50, "4h")
    extended_primary = synthetic_klines(180, "1h")
    extended_htf = synthetic_klines(55, "4h")

    base = build_feature_matrix(primary, htf, config).frame
    extended = build_feature_matrix(extended_primary, extended_htf, config).frame
    timestamp = pd.Timestamp("2022-01-05 12:00", tz="UTC")
    columns = [
        "taker_imbalance",
        "taker_buy_ratio_zscore",
        "taker_imbalance_slope",
        "signed_large_trade_pressure",
        "signed_large_trade_pressure_stable_zscore",
        "signed_large_trade_pressure_stable_rank",
        "signed_large_trade_pressure_stable_tanh",
        "large_trade_pressure_4_minus_3",
        "large_trade_pressure_4_minus_3_stable_rank",
        "large_trade_pressure_4_minus_3_stable_tanh",
        "taker_imbalance_mean_4_stable_rank",
        "taker_mean4_x_ltr_rank_signed",
        "taker_imbalance_x_rv14_rank_signed",
        "signed_ltp_x_rv14_rank_high",
        "ltp4_rank_x_gk14_rank_low",
        "orderflow_efficiency",
        "absorption_pressure_3",
        "absorption_pressure_3_stable_rank",
        "cvd_price_divergence_4",
        "cvd_price_divergence_4_stable_zscore",
        "4h_taker_imbalance",
        "4h_cvd_pressure_3",
        "4h_cvd_pressure_3_stable_rank",
        "4h_taker_imbalance_x_rv14_rank_signed",
        "4h_taker_imbalance_mean_4_stable_rank",
        "4h_taker_mean4_x_ltr_rank_high",
        "4h_ltp4_rank_x_gk14_rank_high",
        "4h_large_trade_pressure_4_minus_3_stable_tanh",
    ]

    assert set(columns).issubset(base.columns)
    base_row = base.loc[base["timestamp"] == timestamp, columns].iloc[0]
    extended_row = extended.loc[extended["timestamp"] == timestamp, columns].iloc[0]
    pd.testing.assert_series_equal(base_row, extended_row, check_names=False)

    assert "signed_large_trade_pressure" not in build_feature_matrix(primary, htf, config).feature_columns
    assert "signed_large_trade_pressure_stable_zscore" in build_feature_matrix(primary, htf, config).feature_columns
    assert "large_trade_pressure_4_minus_3" not in build_feature_matrix(primary, htf, config).feature_columns
    assert "large_trade_pressure_4_minus_3_stable_tanh" in build_feature_matrix(primary, htf, config).feature_columns
    assert "ltp4_rank_x_gk14_rank_low" in build_feature_matrix(primary, htf, config).feature_columns


def test_intrahour_order_flow_features_are_causal_when_future_rows_appended(synthetic_klines, tiny_config) -> None:
    config = copy.deepcopy(tiny_config)
    config["features"]["intrahour_order_flow"] = {
        "enabled": True,
        "interval": "15m",
        "prefix": "ih15",
        "expected_bars_per_hour": 4,
        "min_bars_per_hour": 4,
        "stable_window": 4,
        "stable_clip_abs": 3.0,
        "stable_transforms": ["zscore", "rank"],
        "stable_columns": [
            "ih15_cvd_pressure_norm",
            "ih15_cvd_slope_norm",
            "ih15_flow_price_alignment",
            "ih15_absorption_imbalance",
            "ih15_late_flow_reversal",
            "ih15_vpt_last_vs_hour",
            "ih15_aggressive_buy_burst",
        ],
    }
    primary = synthetic_klines(96, "1h")
    htf = synthetic_klines(30, "4h")
    intrabar = synthetic_klines(96 * 4, "15m")
    extended_primary = synthetic_klines(112, "1h")
    extended_htf = synthetic_klines(34, "4h")
    extended_intrabar = synthetic_klines(112 * 4, "15m")

    intrahour = compute_intrahour_order_flow_features(intrabar, config)
    assert {
        "ih15_taker_imbalance_late_minus_early",
        "ih15_cvd_pressure_norm",
        "ih15_flow_price_alignment",
        "ih15_buy_absorption",
        "ih15_absorption_imbalance_stable_rank",
        "ih15_cvd_pressure_norm_stable_rank",
        "ih15_aggressive_buy_burst_stable_zscore",
    }.issubset(set(intrahour.feature_columns))

    base = build_feature_matrix(primary, htf, config, intrabar_frame=intrabar).frame
    extended = build_feature_matrix(extended_primary, extended_htf, config, intrabar_frame=extended_intrabar).frame
    timestamp = pd.Timestamp("2022-01-03 12:00", tz="UTC")
    columns = [
        "ih15_taker_imbalance_mean",
        "ih15_taker_imbalance_late_minus_early",
        "ih15_cvd_pressure_norm",
        "ih15_cvd_slope_norm",
        "ih15_price_move_norm",
        "ih15_flow_price_alignment",
        "ih15_buy_absorption",
        "ih15_sell_absorption",
        "ih15_absorption_imbalance_stable_rank",
        "ih15_late_flow_reversal",
        "ih15_volume_share_last",
        "ih15_vpt_last_vs_hour",
        "ih15_aggressive_buy_burst",
        "ih15_cvd_pressure_norm_stable_rank",
        "ih15_aggressive_buy_burst_stable_zscore",
    ]

    assert set(columns).issubset(base.columns)
    base_row = base.loc[base["timestamp"] == timestamp, columns].iloc[0]
    extended_row = extended.loc[extended["timestamp"] == timestamp, columns].iloc[0]
    pd.testing.assert_series_equal(base_row, extended_row, check_names=False)


def test_intrahour_missing_hours_are_neutral_not_forward_filled(synthetic_klines, tiny_config) -> None:
    config = copy.deepcopy(tiny_config)
    config["features"]["intrahour_order_flow"] = {
        "enabled": True,
        "interval": "15m",
        "prefix": "ih15",
        "expected_bars_per_hour": 4,
        "min_bars_per_hour": 4,
        "stable_window": 4,
        "stable_clip_abs": 3.0,
        "stable_transforms": ["zscore", "rank"],
        "stable_columns": ["ih15_absorption_imbalance", "ih15_flow_price_alignment"],
    }
    primary = synthetic_klines(48, "1h")
    htf = synthetic_klines(18, "4h")
    intrabar = synthetic_klines(48 * 4, "15m")
    missing_hour = pd.Timestamp("2022-01-02 04:00", tz="UTC")
    intrabar = intrabar[intrabar["timestamp"].dt.floor("h") != missing_hour].reset_index(drop=True)

    features = build_feature_matrix(primary, htf, config, intrabar_frame=intrabar).frame
    row = features.loc[features["timestamp"] == missing_hour].iloc[0]

    assert row["ih15_missing"] == 1.0
    assert row["ih15_coverage"] == 0.0
    assert row["ih15_taker_imbalance_mean"] == 0.0
    assert row["ih15_absorption_imbalance"] == 0.0


def test_futures_context_features_are_causal_when_future_rows_appended(synthetic_klines, tiny_config) -> None:
    config = copy.deepcopy(tiny_config)
    config["features"]["futures_context"] = {
        "enabled": True,
        "prefix": "fut",
        "stable_window": 4,
        "funding_stable_window": 3,
        "stable_clip_abs": 3.0,
        "stable_tanh_scale": 2.0,
        "stable_transforms": ["zscore", "rank", "tanh"],
        "oi_change_windows": [2, 4],
        "funding_windows": [2, 3],
        "metrics_merge_tolerance_minutes": 90,
        "funding_merge_tolerance_minutes": 540,
    }
    primary = synthetic_klines(96, "1h")
    htf = synthetic_klines(30, "4h")
    extended_primary = synthetic_klines(112, "1h")
    extended_htf = synthetic_klines(34, "4h")
    metrics = _synthetic_futures_metrics(96 * 12)
    extended_metrics = _synthetic_futures_metrics(112 * 12)
    funding = _synthetic_funding_rates(16)
    extended_funding = _synthetic_funding_rates(20)

    metrics_features = compute_futures_metrics_features(metrics, config)
    assert {
        "fut_oi_change_4_stable_rank",
        "fut_toptrader_count_long_short_log_ratio_stable_zscore",
        "fut_taker_long_short_vol_log_ratio_stable_tanh",
    }.issubset(set(metrics_features.feature_columns))

    funding_features = compute_funding_rate_features(funding, config)
    assert {
        "fut_funding_rate_stable_rank",
        "fut_funding_sum_3_stable_zscore",
        "fut_funding_mean_2_stable_tanh",
    }.issubset(set(funding_features.feature_columns))

    base = build_feature_matrix(
        primary,
        htf,
        config,
        futures_metrics_frame=metrics,
        funding_frame=funding,
    ).frame
    extended = build_feature_matrix(
        extended_primary,
        extended_htf,
        config,
        futures_metrics_frame=extended_metrics,
        funding_frame=extended_funding,
    ).frame
    timestamp = pd.Timestamp("2022-01-03 12:00", tz="UTC")
    columns = [
        "fut_oi_change_4_stable_rank",
        "fut_oi_value_change_4_stable_tanh",
        "fut_toptrader_count_long_short_log_ratio_stable_zscore",
        "fut_global_long_short_log_ratio_stable_rank",
        "fut_taker_long_short_vol_log_ratio_stable_tanh",
        "fut_funding_rate_stable_rank",
        "fut_funding_sum_3_stable_zscore",
        "fut_funding_mean_2_stable_tanh",
        "fut_metrics_missing",
        "fut_funding_missing",
    ]

    assert set(columns).issubset(base.columns)
    base_row = base.loc[base["timestamp"] == timestamp, columns].iloc[0]
    extended_row = extended.loc[extended["timestamp"] == timestamp, columns].iloc[0]
    pd.testing.assert_series_equal(base_row, extended_row, check_names=False)
    assert float(base_row["fut_metrics_missing"]) == 0.0
    assert float(base_row["fut_funding_missing"]) == 0.0


def _synthetic_futures_metrics(periods: int) -> pd.DataFrame:
    ts = pd.date_range("2022-01-01", periods=periods, freq="5min", tz="UTC")
    idx = pd.Series(range(periods), dtype=float)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": "BTCUSDT",
            "sum_open_interest": 100000.0 + 100.0 * idx + 500.0 * np.sin(idx / 20.0),
            "sum_open_interest_value": 4_000_000_000.0 + 2_000_000.0 * idx,
            "count_toptrader_long_short_ratio": 1.0 + 0.1 * np.sin(idx / 13.0),
            "sum_toptrader_long_short_ratio": 1.0 + 0.08 * np.cos(idx / 17.0),
            "count_long_short_ratio": 1.0 + 0.05 * np.sin(idx / 11.0),
            "sum_taker_long_short_vol_ratio": 1.0 + 0.12 * np.cos(idx / 7.0),
        }
    )


def _synthetic_funding_rates(periods: int) -> pd.DataFrame:
    ts = pd.date_range("2022-01-01", periods=periods, freq="8h", tz="UTC")
    idx = pd.Series(range(periods), dtype=float)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": "BTCUSDT",
            "funding_rate": 0.0001 * np.sin(idx / 3.0),
            "mark_price": 40000.0 + 50.0 * idx,
        }
    )
