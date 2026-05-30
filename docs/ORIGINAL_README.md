# yeniBot

Phase 1 ML foundation for BTC/USDT perpetual futures direction modeling.

The project intentionally stops at validated model research. It does not include backtesting, trade execution, live deployment, XGBoost meta-learning, or 3-class short/hold/long labels.

## What This Builds

- Binance USDT-M full-kline downloader for `BTCUSDT` 1H and 4H data.
- Binance Vision fallback for Colab or other hosts that receive HTTP 451 from the REST API.
- Config-controlled dropping of rare zero-volume/no-trade archive bars before feature generation.
- Bias-safe microstructure feature engineering.
- Stationary transforms for price/volume/order-flow scale features, with raw level columns excluded from model inputs by config.
- Continuous order-flow v2 features for taker imbalance, CVD pressure, large-trade pressure, absorption, and price-flow divergence.
- Stable rolling rank/z-score order-flow v2 model inputs, with high-variance raw pressure/divergence columns kept out of training when `stable_only` is enabled.
- Stable rolling rank/z-score volatility and structure inputs for controlled overlay experiments rather than full-profile replacement.
- Config-driven feature profiles for ablation runs such as `baseline_40`, `baseline_plus_bounded_v2`, `baseline_plus_4h_bounded_whale_no_4h_tier1`, `baseline_no_4h_tier1_4h_large_trade_pressure_long`, and targeted champion-pruning variants.
- Correct 4H-to-1H alignment by shifting 4H bars forward before merge.
- Long-only binary triple-barrier labels.
- Binary TCN+GRU sequence model with focal and rank-correlation losses.
- Purged walk-forward CV with train-only scaling.
- Forward-only HMM regime diagnostics.
- Colab notebooks `01` through `05`.

## Colab Workflow

Run notebooks in strict order:

1. `01_data_preparation.ipynb`
2. `02_feature_engineering.ipynb`
3. `03_labeling.ipynb`
4. `04_training_walk_forward.ipynb`
5. `05_diagnostics_validation.ipynb`

After any `git pull` in Colab, use `Runtime -> Restart session` before re-running cells. Python keeps imported modules in memory and will otherwise use stale code.

Feature-engineering changes, including changes to `features.structure_stability`,
require re-running notebooks `02` through `05` so the processed feature parquet
matches the active training profile. Pure profile-selection changes that only
switch among already-generated columns can be re-tested with notebooks `04`
and `05` after a Colab session restart.

The saved feature matrix drops warmup rows only when active model inputs, HMM
features, or the labeling ATR column are unavailable. Inactive experimental
columns may contain warmup NaNs and must not change the walk-forward row
universe.

## Local Checks

```bash
pip install -r requirements.txt
pytest
```

## Phase 1 Gate

Proceed only when diagnostics show:

- Rank IC mean above `0.03`
- Long F1 above `0.45`
- Rank IC std below `0.03`
- Positive Rank IC in more than `75%` of folds
- Calibration separation between actual long and non-long outcomes
- No leakage alerts

Notebook `05` writes one shareable archive under `Drive/yeniBot/reports/`:
`phase1_latest_experiment_bundle.zip`. Send that single bundle when a run needs
review. It contains the profile comparison, decision report, best candidate,
and profile-specific diagnostics zips with threshold diagnostics, score-lift
tables, recent-fold health summaries, MTF alignment sentinels, regime metrics,
model feature columns, feature-profile diagnostics, stationarity checks,
good/bad fold forensic tables, and an `experiment_ledger` snapshot for
run-over-run comparisons.
