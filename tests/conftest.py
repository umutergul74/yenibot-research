from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def synthetic_klines(periods: int = 160, interval: str = "1h", start: str = "2022-01-01") -> pd.DataFrame:
    def factory(periods: int = 160, interval: str = "1h", start: str = "2022-01-01") -> pd.DataFrame:
        freq = {"15m": "15min", "1h": "1h", "4h": "4h"}[interval]
        ts = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
        idx = np.arange(periods, dtype=float)
        close = 100.0 + 0.04 * idx + np.sin(idx / 5.0)
        open_ = close + 0.05 * np.sin(idx / 3.0)
        high = np.maximum(open_, close) + 0.6
        low = np.minimum(open_, close) - 0.6
        volume = 1000.0 + 50.0 * np.sin(idx / 7.0) + idx
        taker_buy = volume * (0.5 + 0.08 * np.sin(idx / 4.0))
        quote_volume = volume * close
        return pd.DataFrame(
            {
                "timestamp": ts,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "close_time": ts
                + pd.to_timedelta({"15m": 15, "1h": 60, "4h": 240}[interval], unit="m")
                - pd.to_timedelta(1, unit="ms"),
                "quote_volume": quote_volume,
                "num_trades": (100 + (idx % 20)).astype(int),
                "taker_buy_base_vol": taker_buy,
                "taker_buy_quote_vol": taker_buy * close,
                "ignore": 0,
            }
        )

    return factory


@pytest.fixture
def tiny_config() -> dict:
    return {
        "features": {
            "warmup_rows": 0,
            "wavelet": {"enabled": False},
            "order_flow": {"cvd_zscore_window": 3, "cvd_rate_window": 3, "imbalance_ema_span": 2},
            "whale": {
                "vpt_zscore_window": 3,
                "whale_zscore_threshold": 2.0,
                "whale_buy_ratio": 0.55,
                "whale_sell_ratio": 0.45,
                "large_trade_window": 3,
            },
            "structure": {
                "realized_vol_window": 3,
                "gk_vol_window": 3,
                "atr_period": 3,
                "adx_period": 3,
                "vwap_window": 3,
            },
            "mtf": {"shift_hours": 4},
        },
        "labeling": {"max_holding_bars": 10},
        "model": {
            "seq_len": 8,
            "tcn_channels": 8,
            "tcn_kernel_size": 3,
            "tcn_dilations": [1, 2],
            "gru_hidden": 8,
            "gru_layers": 1,
            "dropout": 0.0,
            "fusion_hidden": 8,
        },
        "training": {
            "batch_size": 16,
            "epochs": 1,
            "early_stop_patience": 1,
            "optimizer": {"lr": 0.001, "weight_decay": 0.0001},
            "scheduler": {"T_0": 2, "T_mult": 1},
            "grad_clip": 1.0,
            "loss": {"focal_gamma": 2.0, "focal_alpha": 0.6, "rank_ic_weight": 0.2},
        },
        "walk_forward": {
            "train_bars": 80,
            "val_bars": 32,
            "test_bars": 32,
            "step_bars": 32,
            "purge_bars": 4,
            "embargo_bars": 2,
        },
        "hmm": {
            "n_states": 3,
            "covariance_type": "full",
            "n_iter": 20,
            "random_state": 7,
            "gamma_floor": 0.02,
            "state_weight_floor": 0.08,
            "n_ratio_alarm": 15.0,
            "suppress_convergence_warnings": True,
            "features": ["log_return", "gk_vol_14", "adx_14", "true_cvd_zscore", "vwap_dist_atr"],
        },
        "validation": {
            "target_rank_ic": 0.03,
            "max_rank_ic_std": 0.03,
            "min_positive_ic_fraction": 0.75,
            "min_long_f1": 0.45,
            "suspicious_rank_ic": 0.10,
            "random_like_rank_ic": 0.01,
            "calibration_bins": 10,
        },
    }
