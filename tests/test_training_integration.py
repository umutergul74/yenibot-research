from __future__ import annotations

import copy

import numpy as np
import pytest

from yenibot.features import build_feature_matrix
from yenibot.training import PurgedWalkForwardCV, run_walk_forward_training, train_one_fold


def test_small_pipeline_runs_one_training_step(synthetic_klines, tiny_config, tmp_path) -> None:
    primary = synthetic_klines(190, "1h")
    htf = synthetic_klines(60, "4h")
    features = build_feature_matrix(primary, htf, tiny_config)
    frame = features.frame.copy().reset_index(drop=True)

    # Deterministic synthetic labels for integration wiring only.
    frame["label"] = (np.arange(len(frame)) % 3 == 0).astype(int)
    frame["fwd_return_10h"] = frame["close"].shift(-10) / frame["close"] - 1.0
    frame = frame.dropna(subset=["fwd_return_10h"]).reset_index(drop=True)

    cv = PurgedWalkForwardCV(**tiny_config["walk_forward"])
    fold = next(cv.split(len(frame)))
    result = train_one_fold(
        frame,
        fold,
        features.feature_columns,
        tiny_config,
        checkpoint_dir=tmp_path,
        device="cpu",
    )

    assert not result["predictions"].empty
    assert (tmp_path / "scaler_fold_000.pkl").exists()
    assert (tmp_path / "model_fold_000.pt").exists()
    assert (tmp_path / "hmm_fold_000.pkl").exists()
    assert (tmp_path / "predictions_fold_000.parquet").exists()


def test_train_one_fold_repeats_with_same_seed_on_cpu(synthetic_klines, tiny_config) -> None:
    config = copy.deepcopy(tiny_config)
    config["project"] = {"random_seed": 123, "deterministic": True}
    primary = synthetic_klines(190, "1h")
    htf = synthetic_klines(60, "4h")
    features = build_feature_matrix(primary, htf, config)
    frame = features.frame.copy().reset_index(drop=True)
    frame["label"] = (np.arange(len(frame)) % 3 == 0).astype(int)
    frame["fwd_return_10h"] = frame["close"].shift(-10) / frame["close"] - 1.0
    frame = frame.dropna(subset=["fwd_return_10h"]).reset_index(drop=True)
    fold = next(PurgedWalkForwardCV(**config["walk_forward"]).split(len(frame)))

    first = train_one_fold(frame, fold, features.feature_columns, config, device="cpu")
    second = train_one_fold(frame, fold, features.feature_columns, config, device="cpu")

    np.testing.assert_allclose(
        first["predictions"]["prob_long"].to_numpy(),
        second["predictions"]["prob_long"].to_numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_train_one_fold_supports_val_loss_early_stopping_as_explicit_experiment(synthetic_klines, tiny_config) -> None:
    config = copy.deepcopy(tiny_config)
    config["training"]["early_stop_metric"] = "val_loss"
    config["training"]["epochs"] = 2
    primary = synthetic_klines(190, "1h")
    htf = synthetic_klines(60, "4h")
    features = build_feature_matrix(primary, htf, config)
    frame = features.frame.copy().reset_index(drop=True)
    frame["label"] = (np.arange(len(frame)) % 3 == 0).astype(int)
    frame["fwd_return_10h"] = frame["close"].shift(-10) / frame["close"] - 1.0
    frame = frame.dropna(subset=["fwd_return_10h"]).reset_index(drop=True)
    fold = next(PurgedWalkForwardCV(**config["walk_forward"]).split(len(frame)))

    result = train_one_fold(frame, fold, features.feature_columns, config, device="cpu")

    history = result["history"]
    assert set(history["early_stop_metric"]) == {"val_loss"}
    assert history["early_stop_value"].notna().all()
    assert history["val_loss"].notna().all()


def test_train_one_fold_supports_pairwise_label_margin_loss(synthetic_klines, tiny_config) -> None:
    config = copy.deepcopy(tiny_config)
    config["training"]["loss"]["label_margin_weight"] = 0.05
    config["training"]["loss"]["label_margin"] = 0.25
    config["training"]["epochs"] = 2
    primary = synthetic_klines(190, "1h")
    htf = synthetic_klines(60, "4h")
    features = build_feature_matrix(primary, htf, config)
    frame = features.frame.copy().reset_index(drop=True)
    frame["label"] = (np.arange(len(frame)) % 3 == 0).astype(int)
    frame["fwd_return_10h"] = frame["close"].shift(-10) / frame["close"] - 1.0
    frame = frame.dropna(subset=["fwd_return_10h"]).reset_index(drop=True)
    fold = next(PurgedWalkForwardCV(**config["walk_forward"]).split(len(frame)))

    result = train_one_fold(frame, fold, features.feature_columns, config, device="cpu")

    assert not result["predictions"].empty
    assert result["history"]["train_loss"].notna().all()


def test_train_one_fold_rejects_unknown_early_stop_metric(synthetic_klines, tiny_config) -> None:
    config = copy.deepcopy(tiny_config)
    config["training"]["early_stop_metric"] = "not_a_real_metric"
    primary = synthetic_klines(190, "1h")
    htf = synthetic_klines(60, "4h")
    features = build_feature_matrix(primary, htf, config)
    frame = features.frame.copy().reset_index(drop=True)
    frame["label"] = (np.arange(len(frame)) % 3 == 0).astype(int)
    frame["fwd_return_10h"] = frame["close"].shift(-10) / frame["close"] - 1.0
    frame = frame.dropna(subset=["fwd_return_10h"]).reset_index(drop=True)
    fold = next(PurgedWalkForwardCV(**config["walk_forward"]).split(len(frame)))

    with pytest.raises(ValueError, match="Unsupported early_stop_metric"):
        train_one_fold(frame, fold, features.feature_columns, config, device="cpu")


def test_run_walk_forward_training_fails_fast_on_active_feature_nans(synthetic_klines, tiny_config) -> None:
    primary = synthetic_klines(190, "1h")
    htf = synthetic_klines(60, "4h")
    features = build_feature_matrix(primary, htf, tiny_config)
    frame = features.frame.copy().reset_index(drop=True)
    frame["label"] = (np.arange(len(frame)) % 3 == 0).astype(int)
    frame["fwd_return_10h"] = frame["close"].shift(-10) / frame["close"] - 1.0
    frame = frame.dropna(subset=["fwd_return_10h"]).reset_index(drop=True)
    frame.loc[0, features.feature_columns[0]] = np.nan

    with pytest.raises(ValueError, match="Re-run notebooks 02 and 03"):
        run_walk_forward_training(frame, tiny_config, feature_columns=features.feature_columns, max_folds=1, device="cpu")


def test_run_walk_forward_training_honors_selected_fold_ids(synthetic_klines, tiny_config, tmp_path) -> None:
    primary = synthetic_klines(260, "1h")
    htf = synthetic_klines(80, "4h")
    features = build_feature_matrix(primary, htf, tiny_config)
    frame = features.frame.copy().reset_index(drop=True)
    frame["label"] = (np.arange(len(frame)) % 3 == 0).astype(int)
    frame["fwd_return_10h"] = frame["close"].shift(-10) / frame["close"] - 1.0
    frame = frame.dropna(subset=["fwd_return_10h"]).reset_index(drop=True)

    result = run_walk_forward_training(
        frame,
        tiny_config,
        feature_columns=features.feature_columns,
        checkpoint_dir=tmp_path,
        fold_ids=[1],
        device="cpu",
    )

    assert sorted(result["predictions"]["fold"].unique().tolist()) == [1]
    assert not (tmp_path / "model_fold_000.pt").exists()
    assert (tmp_path / "model_fold_001.pt").exists()
