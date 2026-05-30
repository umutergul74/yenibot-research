from __future__ import annotations

import pytest

from yenibot.config import load_config
from yenibot.features import filter_feature_columns, resolve_feature_profile


def test_filter_feature_columns_uses_configured_names_and_patterns() -> None:
    columns = ["true_cvd_zscore", "realized_vol_14", "4h_log_return", "4h_true_cvd_zscore"]
    config = {
        "features": {
            "exclude_columns": ["realized_vol_14"],
            "exclude_patterns": ["4h_log_return"],
        }
    }

    assert filter_feature_columns(columns, config) == ["true_cvd_zscore", "4h_true_cvd_zscore"]


def test_filter_feature_columns_rejects_empty_result() -> None:
    with pytest.raises(ValueError, match="removed every feature"):
        filter_feature_columns(["a"], {"features": {"exclude_patterns": ["*"]}})


def test_filter_feature_columns_applies_stationarity_policy() -> None:
    columns = [
        "close_denoised",
        "close_denoised_log_return",
        "4h_atr_14",
        "4h_atr_14_pct",
        "true_cvd_delta",
        "true_cvd_delta_norm",
    ]
    config = {
        "features": {
            "stationarity": {
                "exclude_nonstationary": True,
                "exclude_patterns": ["*close_denoised", "*atr_14", "*true_cvd_delta"],
            }
        }
    }

    assert filter_feature_columns(columns, config) == [
        "close_denoised_log_return",
        "4h_atr_14_pct",
        "true_cvd_delta_norm",
    ]


def test_filter_feature_columns_applies_order_flow_v2_stable_only_policy() -> None:
    columns = [
        "signed_large_trade_pressure",
        "signed_large_trade_pressure_stable_zscore",
        "cvd_pressure_3",
        "cvd_pressure_3_stable_rank",
        "4h_absorption_pressure_12",
        "4h_absorption_pressure_12_stable_zscore",
        "taker_imbalance_mean_3",
    ]
    config = {
        "features": {
            "order_flow_v2": {
                "enabled": True,
                "stable_only": True,
                "pressure_windows": [3, 12],
            }
        }
    }

    assert filter_feature_columns(columns, config) == [
        "signed_large_trade_pressure_stable_zscore",
        "cvd_pressure_3_stable_rank",
        "4h_absorption_pressure_12_stable_zscore",
        "taker_imbalance_mean_3",
    ]


def test_filter_feature_columns_applies_active_profile_with_inherited_includes() -> None:
    columns = [
        "gk_vol_14",
        "4h_gk_vol_14",
        "taker_imbalance",
        "4h_taker_imbalance_mean_24",
        "4h_large_trade_ratio",
        "4h_cvd_pressure_24_stable_rank",
    ]
    config = {
        "features": {
            "active_profile": "baseline_plus_4h_bounded_whale",
            "profiles": {
                "baseline_40": {
                    "include_patterns": ["*gk_vol_14", "*large_trade_ratio"],
                    "exclude_patterns": ["*taker_imbalance*", "*_stable_*"],
                },
                "baseline_plus_4h_bounded_whale": {
                    "inherit": "baseline_40",
                    "include_patterns": ["4h_taker_imbalance_mean_*"],
                    "exclude_patterns": ["*_stable_*"],
                },
            },
        }
    }

    profile = resolve_feature_profile(config)
    assert profile["name"] == "baseline_plus_4h_bounded_whale"
    assert filter_feature_columns(columns, config) == [
        "gk_vol_14",
        "4h_gk_vol_14",
        "4h_taker_imbalance_mean_24",
        "4h_large_trade_ratio",
    ]


def test_filter_feature_columns_rejects_unknown_active_profile() -> None:
    with pytest.raises(ValueError, match="Unknown features.active_profile"):
        filter_feature_columns(["gk_vol_14"], {"features": {"active_profile": "missing", "profiles": {}}})


def test_filter_feature_columns_can_drop_inherited_4h_tier1_family() -> None:
    columns = [
        "taker_buy_ratio",
        "4h_taker_buy_ratio",
        "true_cvd_zscore",
        "4h_true_cvd_zscore",
        "4h_taker_imbalance_mean_6",
        "4h_whale_buy_flag",
        "4h_gk_vol_14",
    ]
    config = {
        "features": {
            "active_profile": "baseline_plus_4h_bounded_whale_no_4h_tier1",
            "profiles": {
                "baseline_40": {
                    "include_patterns": [
                        "*taker_buy_ratio",
                        "*true_cvd_zscore",
                        "*gk_vol_14",
                        "*whale_buy_flag",
                    ],
                    "exclude_patterns": [],
                },
                "baseline_plus_4h_bounded_whale": {
                    "inherit": "baseline_40",
                    "include_patterns": ["4h_taker_imbalance_mean_*"],
                    "exclude_patterns": [],
                },
                "baseline_plus_4h_bounded_whale_no_4h_tier1": {
                    "inherit": "baseline_plus_4h_bounded_whale",
                    "exclude_patterns": [
                        "4h_taker_buy_ratio",
                        "4h_true_cvd_zscore",
                    ],
                },
            },
        }
    }

    assert filter_feature_columns(columns, config) == [
        "taker_buy_ratio",
        "true_cvd_zscore",
        "4h_taker_imbalance_mean_6",
        "4h_whale_buy_flag",
        "4h_gk_vol_14",
    ]


def test_filter_feature_columns_can_replace_raw_structure_with_stable_structure() -> None:
    columns = [
        "taker_buy_ratio",
        "true_cvd_zscore",
        "gk_vol_14",
        "gk_vol_14_stable_zscore",
        "gk_vol_14_stable_rank",
        "4h_gk_vol_14",
        "4h_gk_vol_14_stable_zscore",
        "4h_taker_imbalance_mean_6",
        "4h_true_cvd_zscore",
        "signed_large_trade_pressure_stable_zscore",
    ]
    config = {
        "features": {
            "active_profile": "baseline_plus_4h_bounded_whale_stable_structure",
            "profiles": {
                "baseline_40": {
                    "include_patterns": [
                        "*taker_buy_ratio",
                        "*true_cvd_zscore",
                        "*gk_vol_14",
                    ],
                    "exclude_patterns": [],
                },
                "baseline_plus_4h_bounded_whale": {
                    "inherit": "baseline_40",
                    "include_patterns": ["4h_taker_imbalance_mean_*"],
                    "exclude_patterns": [],
                },
                "baseline_plus_4h_bounded_whale_no_4h_tier1": {
                    "inherit": "baseline_plus_4h_bounded_whale",
                    "exclude_patterns": ["4h_true_cvd_zscore"],
                },
                "baseline_plus_4h_bounded_whale_stable_structure": {
                    "inherit": "baseline_plus_4h_bounded_whale_no_4h_tier1",
                    "include_patterns": [
                        "*gk_vol_14_stable_zscore",
                        "*gk_vol_14_stable_rank",
                    ],
                    "exclude_patterns": [
                        "4h_true_cvd_zscore",
                        "*large_trade_pressure*",
                        "*gk_vol_14",
                    ],
                },
            },
        }
    }

    assert filter_feature_columns(columns, config) == [
        "taker_buy_ratio",
        "true_cvd_zscore",
        "gk_vol_14_stable_zscore",
        "gk_vol_14_stable_rank",
        "4h_gk_vol_14_stable_zscore",
        "4h_taker_imbalance_mean_6",
    ]


def test_repo_default_active_profile_is_stability_control() -> None:
    config = load_config("config.yaml")

    assert config["features"]["active_profile"] == "baseline_plus_4h_bounded_whale_no_4h_tier1"


def test_overlay_profile_keeps_raw_structure_and_only_allowed_4h_stable_overlay() -> None:
    config = load_config("config.yaml")
    config["features"]["active_profile"] = "baseline_plus_4h_bounded_whale_no_4h_tier1_plus_4h_stable_overlay"
    columns = [
        "taker_buy_ratio",
        "true_cvd_zscore",
        "gk_vol_14",
        "4h_gk_vol_14",
        "4h_vwap_dist_atr",
        "4h_taker_imbalance_mean_6",
        "realized_vol_14_stable_zscore",
        "gk_vol_14_stable_rank",
        "4h_realized_vol_14_stable_zscore",
        "4h_realized_vol_14_stable_rank",
        "4h_gk_vol_14_stable_zscore",
        "4h_gk_vol_14_stable_rank",
        "4h_atr_14_pct_stable_zscore",
        "4h_atr_14_pct_stable_rank",
        "4h_adx_14_stable_zscore",
        "4h_adx_14_stable_rank",
        "4h_vwap_dist_atr_stable_zscore",
        "4h_vwap_dist_atr_stable_rank",
        "4h_log_return_stable_zscore",
        "4h_close_denoised_log_return_stable_rank",
        "4h_volume_log_zscore_stable_zscore",
        "4h_volume_denoised_log_zscore_stable_rank",
        "4h_true_cvd_zscore",
    ]

    filtered = filter_feature_columns(columns, config)
    stable_columns = sorted(column for column in filtered if "_stable_" in column)

    assert filtered == [
        "taker_buy_ratio",
        "true_cvd_zscore",
        "gk_vol_14",
        "4h_gk_vol_14",
        "4h_vwap_dist_atr",
        "4h_taker_imbalance_mean_6",
        "4h_realized_vol_14_stable_zscore",
        "4h_realized_vol_14_stable_rank",
        "4h_gk_vol_14_stable_zscore",
        "4h_gk_vol_14_stable_rank",
        "4h_atr_14_pct_stable_zscore",
        "4h_atr_14_pct_stable_rank",
        "4h_adx_14_stable_zscore",
        "4h_adx_14_stable_rank",
        "4h_vwap_dist_atr_stable_zscore",
        "4h_vwap_dist_atr_stable_rank",
    ]
    assert stable_columns == sorted(
        [
            "4h_realized_vol_14_stable_zscore",
            "4h_realized_vol_14_stable_rank",
            "4h_gk_vol_14_stable_zscore",
            "4h_gk_vol_14_stable_rank",
            "4h_atr_14_pct_stable_zscore",
            "4h_atr_14_pct_stable_rank",
            "4h_adx_14_stable_zscore",
            "4h_adx_14_stable_rank",
            "4h_vwap_dist_atr_stable_zscore",
            "4h_vwap_dist_atr_stable_rank",
        ]
    )
