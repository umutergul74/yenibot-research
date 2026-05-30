from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader

from yenibot.diagnostics.metrics import classification_metrics, phase1_report, rank_ic
from yenibot.features.builder import filter_feature_columns, select_feature_columns
from yenibot.losses import FocalLossWithLogits, PairwiseLabelMarginLoss, RankICLoss
from yenibot.models import HybridEncoder
from yenibot.regime import OnlineGaussianHMM
from yenibot.training.dataset import SequenceDataset
from yenibot.training.walk_forward import FoldIndices, PurgedWalkForwardCV


def _cfg(config: Any, path: list[str], default: Any = None) -> Any:
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


def _device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_random_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and Torch for repeatable fold training."""

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def _base_seed(config: Any) -> int:
    return int(_cfg(config, ["project", "random_seed"], 42))


def _deterministic(config: Any) -> bool:
    return bool(_cfg(config, ["project", "deterministic"], False))


def _build_model(n_features: int, config: Any) -> HybridEncoder:
    model_cfg = _cfg(config, ["model"], {})
    return HybridEncoder(
        n_features,
        seq_len=int(_cfg(model_cfg, ["seq_len"], 64)),
        tcn_channels=int(_cfg(model_cfg, ["tcn_channels"], 64)),
        tcn_kernel_size=int(_cfg(model_cfg, ["tcn_kernel_size"], 3)),
        tcn_dilations=list(_cfg(model_cfg, ["tcn_dilations"], [1, 2, 4, 8, 16])),
        gru_hidden=int(_cfg(model_cfg, ["gru_hidden"], 128)),
        gru_layers=int(_cfg(model_cfg, ["gru_layers"], 2)),
        dropout=float(_cfg(model_cfg, ["dropout"], 0.2)),
        fusion_hidden=int(_cfg(model_cfg, ["fusion_hidden"], 128)),
    )


def _make_dataset(part: pd.DataFrame, feature_columns: list[str], config: Any) -> SequenceDataset:
    seq_len = int(_cfg(config, ["model", "seq_len"], 64))
    forward_column = f"fwd_return_{int(_cfg(config, ['labeling', 'max_holding_bars'], 10))}h"
    if forward_column not in part.columns:
        forward_column = "fwd_return_10h"
    return SequenceDataset(
        part[feature_columns].to_numpy(dtype=np.float32),
        part["label"].to_numpy(dtype=np.float32),
        part[forward_column].to_numpy(dtype=np.float32),
        seq_len=seq_len,
    )


def _forward_return_column(frame: pd.DataFrame, config: Any) -> str:
    forward_column = f"fwd_return_{int(_cfg(config, ['labeling', 'max_holding_bars'], 10))}h"
    if forward_column not in frame.columns:
        forward_column = "fwd_return_10h"
    return forward_column


def _assert_training_inputs_available(frame: pd.DataFrame, feature_columns: list[str], config: Any) -> None:
    required = list(feature_columns)
    required.extend(str(column) for column in list(_cfg(config, ["hmm", "features"], []) or []))
    required.extend(["label", _forward_return_column(frame, config)])
    required = list(dict.fromkeys(required))
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Training frame is missing required columns: {missing}")
    bad = [
        column
        for column in required
        if frame[column].replace([np.inf, -np.inf], np.nan).isna().any()
    ]
    if bad:
        raise ValueError(
            "Training frame contains NaNs in active model/HMM columns. "
            f"Re-run notebooks 02 and 03 with the active feature profile. Columns: {bad}"
        )


def _predict_dataset(
    model: HybridEncoder,
    dataset: SequenceDataset,
    source_part: pd.DataFrame,
    *,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    returns: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, y, fwd, pos in loader:
            x = x.to(device)
            prob = model(x).detach().cpu().numpy()
            probs.append(prob)
            labels.append(y.numpy())
            returns.append(fwd.numpy())
            positions.append(pos.numpy())

    row_positions = np.concatenate(positions)
    rows = source_part.iloc[row_positions].copy().reset_index(drop=True)
    rows["prob_long"] = np.concatenate(probs)
    rows["label"] = np.concatenate(labels).astype(int)
    rows["forward_return"] = np.concatenate(returns)
    rows["source_row_position"] = row_positions
    return rows


def _fit_hmm(train_part: pd.DataFrame, config: Any, *, random_state: int | None = None) -> OnlineGaussianHMM:
    hmm_cfg = _cfg(config, ["hmm"], {})
    hmm_features = list(_cfg(hmm_cfg, ["features"], []))
    missing = [column for column in hmm_features if column not in train_part.columns]
    if missing:
        raise ValueError(f"Missing HMM feature columns: {missing}")
    hmm = OnlineGaussianHMM(
        n_states=int(_cfg(hmm_cfg, ["n_states"], 3)),
        covariance_type=str(_cfg(hmm_cfg, ["covariance_type"], "full")),
        n_iter=int(_cfg(hmm_cfg, ["n_iter"], 200)),
        random_state=int(random_state if random_state is not None else _cfg(hmm_cfg, ["random_state"], 42)),
        gamma_floor=float(_cfg(hmm_cfg, ["gamma_floor"], 0.02)),
        state_weight_floor=float(_cfg(hmm_cfg, ["state_weight_floor"], 0.08)),
        n_ratio_alarm=float(_cfg(hmm_cfg, ["n_ratio_alarm"], 15.0)),
        suppress_convergence_warnings=bool(_cfg(hmm_cfg, ["suppress_convergence_warnings"], True)),
    )
    hmm.fit(train_part[hmm_features].to_numpy(dtype=float))
    return hmm


def _add_regime_probs(part: pd.DataFrame, hmm: OnlineGaussianHMM, config: Any) -> pd.DataFrame:
    hmm_features = list(_cfg(config, ["hmm", "features"], []))
    probs = hmm.predict_proba_online(part[hmm_features].to_numpy(dtype=float), update_stats=False)
    out = part.copy()
    for state in range(probs.shape[1]):
        out[f"regime_prob_{state}"] = probs[:, state]
    return out


def _early_stop_metric(config: Any) -> str:
    metric = str(_cfg(config, ["training", "early_stop_metric"], "rank_ic"))
    allowed = {"rank_ic", "val_rank_ic_rolling", "val_loss"}
    if metric not in allowed:
        raise ValueError(f"Unsupported early_stop_metric {metric!r}; expected one of {sorted(allowed)}")
    return metric


def _evaluate(
    model: HybridEncoder,
    dataset: SequenceDataset,
    source_part: pd.DataFrame,
    config: Any,
    device: torch.device,
    *,
    focal: FocalLossWithLogits | None = None,
    rank_loss: RankICLoss | None = None,
    margin_loss: PairwiseLabelMarginLoss | None = None,
    rank_weight: float = 0.2,
    margin_weight: float = 0.0,
) -> dict[str, float]:
    batch_size = int(_cfg(config, ["training", "batch_size"], 256))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    val_loss_sum = 0.0
    val_loss_count = 0
    model.eval()
    with torch.no_grad():
        for x, y, fwd, _ in loader:
            x = x.to(device)
            y = y.to(device)
            fwd = fwd.to(device)
            logits = model(x, return_logits=True)
            probs = torch.sigmoid(logits)
            if focal is not None and rank_loss is not None:
                loss = focal(logits, y) + rank_weight * rank_loss(probs, fwd)
                if margin_loss is not None and margin_weight > 0.0:
                    loss = loss + margin_weight * margin_loss(logits, y)
                batch_n = int(y.shape[0])
                val_loss_sum += float(loss.detach().cpu()) * batch_n
                val_loss_count += batch_n

    pred = _predict_dataset(model, dataset, source_part, batch_size=batch_size, device=device)
    metrics = classification_metrics(pred["label"], pred["prob_long"])
    metrics["rank_ic"] = rank_ic(pred["prob_long"], pred["forward_return"])
    if val_loss_count:
        metrics["val_loss"] = float(val_loss_sum / val_loss_count)
    return metrics


def train_one_fold(
    frame: pd.DataFrame,
    fold: FoldIndices,
    feature_columns: list[str],
    config: Any,
    *,
    checkpoint_dir: str | Path | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Train one purged walk-forward fold and return metrics plus predictions."""

    torch_device = _device(device)
    fold_seed = _base_seed(config) + int(fold.fold)
    set_random_seed(fold_seed, deterministic=_deterministic(config))
    train_part = frame.iloc[fold.train].copy().reset_index(drop=True)
    val_part = frame.iloc[fold.val].copy().reset_index(drop=True)
    test_part = frame.iloc[fold.test].copy().reset_index(drop=True)

    scaler = RobustScaler()
    scaler.fit(train_part[feature_columns])
    for part in (train_part, val_part, test_part):
        part.loc[:, feature_columns] = scaler.transform(part[feature_columns])

    hmm = _fit_hmm(train_part, config, random_state=fold_seed)
    val_part = _add_regime_probs(val_part, hmm, config)
    test_part = _add_regime_probs(test_part, hmm, config)

    train_dataset = _make_dataset(train_part, feature_columns, config)
    val_dataset = _make_dataset(val_part, feature_columns, config)
    test_dataset = _make_dataset(test_part, feature_columns, config)

    model = _build_model(len(feature_columns), config).to(torch_device)
    train_cfg = _cfg(config, ["training"], {})
    loss_cfg = _cfg(train_cfg, ["loss"], {})
    focal = FocalLossWithLogits(
        gamma=float(_cfg(loss_cfg, ["focal_gamma"], 2.0)),
        alpha=float(_cfg(loss_cfg, ["focal_alpha"], 0.6)),
    )
    rank_loss = RankICLoss()
    rank_weight = float(_cfg(loss_cfg, ["rank_ic_weight"], 0.2))
    margin_weight = float(_cfg(loss_cfg, ["label_margin_weight"], 0.0))
    margin_loss = PairwiseLabelMarginLoss(
        margin=float(_cfg(loss_cfg, ["label_margin"], 0.25)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(_cfg(train_cfg, ["optimizer", "lr"], 1e-3)),
        weight_decay=float(_cfg(train_cfg, ["optimizer", "weight_decay"], 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(_cfg(train_cfg, ["scheduler", "T_0"], 10)),
        T_mult=int(_cfg(train_cfg, ["scheduler", "T_mult"], 2)),
    )
    loader = DataLoader(
        train_dataset,
        batch_size=int(_cfg(train_cfg, ["batch_size"], 256)),
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(fold_seed),
    )

    early_stop_metric = _early_stop_metric(config)
    if early_stop_metric == "val_loss":
        best_metric = np.inf
    else:
        best_metric = -np.inf

    best_state: dict[str, torch.Tensor] | None = None
    patience = int(_cfg(train_cfg, ["early_stop_patience"], 15))
    epochs = int(_cfg(train_cfg, ["epochs"], 100))
    grad_clip = float(_cfg(train_cfg, ["grad_clip"], 1.0))
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        losses = []
        for x, y, fwd, _ in loader:
            x = x.to(torch_device)
            y = y.to(torch_device)
            fwd = fwd.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, return_logits=True)
            probs = torch.sigmoid(logits)
            loss = focal(logits, y) + rank_weight * rank_loss(probs, fwd)
            if margin_weight > 0.0:
                loss = loss + margin_weight * margin_loss(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step(epoch + 1)

        val_metrics = _evaluate(
            model,
            val_dataset,
            val_part,
            config,
            torch_device,
            focal=focal,
            rank_loss=rank_loss,
            margin_loss=margin_loss,
            rank_weight=rank_weight,
            margin_weight=margin_weight,
        )

        smoothing_epochs = max(1, int(_cfg(train_cfg, ["rank_ic_smoothing_epochs"], 5)))
        previous_rank_ics = [] if smoothing_epochs == 1 else [h["rank_ic"] for h in history[-(smoothing_epochs - 1):]]
        recent_rank_ics = previous_rank_ics + [val_metrics["rank_ic"]]
        val_rank_ic_rolling = float(np.mean(recent_rank_ics))
        val_metrics["val_rank_ic_rolling"] = val_rank_ic_rolling

        if early_stop_metric == "val_loss":
            current_metric = val_metrics.get("val_loss", np.inf)
            improved = current_metric < best_metric
        elif early_stop_metric == "val_rank_ic_rolling":
            current_metric = val_rank_ic_rolling
            improved = current_metric > best_metric
        else:  # default "rank_ic"
            current_metric = val_metrics["rank_ic"]
            improved = current_metric > best_metric

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(losses)),
                "early_stop_metric": early_stop_metric,
                "early_stop_value": float(current_metric),
                **val_metrics,
            }
        )

        if improved:
            best_metric = current_metric
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    batch_size = int(_cfg(train_cfg, ["batch_size"], 256))
    val_predictions = _predict_dataset(model, val_dataset, val_part, batch_size=batch_size, device=torch_device)
    test_predictions = _predict_dataset(model, test_dataset, test_part, batch_size=batch_size, device=torch_device)
    val_predictions["split"] = "val"
    test_predictions["split"] = "test"
    predictions = pd.concat([val_predictions, test_predictions], ignore_index=True)
    predictions["fold"] = fold.fold

    val_metrics = classification_metrics(val_predictions["label"], val_predictions["prob_long"])
    val_metrics["rank_ic"] = rank_ic(val_predictions["prob_long"], val_predictions["forward_return"])
    test_metrics = classification_metrics(test_predictions["label"], test_predictions["prob_long"])
    test_metrics["rank_ic"] = rank_ic(test_predictions["prob_long"], test_predictions["forward_return"])

    if checkpoint_dir is not None:
        output_dir = Path(checkpoint_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, output_dir / f"scaler_fold_{fold.fold:03d}.pkl")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "feature_columns": feature_columns,
                "config_model": _cfg(config, ["model"], {}),
                "random_seed": fold_seed,
                "deterministic": _deterministic(config),
            },
            output_dir / f"model_fold_{fold.fold:03d}.pt",
        )
        joblib.dump(hmm, output_dir / f"hmm_fold_{fold.fold:03d}.pkl")
        predictions.to_parquet(output_dir / f"predictions_fold_{fold.fold:03d}.parquet", index=False)

    return {
        "fold": fold.fold,
        "model": model,
        "scaler": scaler,
        "hmm": hmm,
        "history": pd.DataFrame(history),
        "predictions": predictions,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def run_walk_forward_training(
    frame: pd.DataFrame,
    config: Any,
    *,
    feature_columns: list[str] | None = None,
    checkpoint_dir: str | Path | None = None,
    max_folds: int | None = None,
    fold_ids: list[int] | tuple[int, ...] | set[int] | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    set_random_seed(_base_seed(config), deterministic=_deterministic(config))
    if feature_columns is None:
        feature_columns = select_feature_columns(frame)
    feature_columns = filter_feature_columns(feature_columns, config)
    _assert_training_inputs_available(frame, feature_columns, config)
    cv_cfg = _cfg(config, ["walk_forward"], {})
    cv = PurgedWalkForwardCV(
        train_bars=int(_cfg(cv_cfg, ["train_bars"], 5040)),
        val_bars=int(_cfg(cv_cfg, ["val_bars"], 1080)),
        test_bars=int(_cfg(cv_cfg, ["test_bars"], 720)),
        step_bars=int(_cfg(cv_cfg, ["step_bars"], 720)),
        purge_bars=int(_cfg(cv_cfg, ["purge_bars"], 24)),
        embargo_bars=int(_cfg(cv_cfg, ["embargo_bars"], 6)),
    )

    fold_results = []
    predictions = []
    selected_fold_ids = {int(fold_id) for fold_id in fold_ids} if fold_ids is not None else None
    for fold in cv.split(len(frame)):
        if selected_fold_ids is not None and int(fold.fold) not in selected_fold_ids:
            continue
        if max_folds is not None and len(fold_results) >= max_folds:
            break
        result = train_one_fold(
            frame,
            fold,
            feature_columns,
            config,
            checkpoint_dir=checkpoint_dir,
            device=device,
        )
        fold_results.append(result)
        predictions.append(result["predictions"])

    if not predictions:
        raise ValueError("No folds were produced; check dataset length and CV configuration")

    all_predictions = pd.concat(predictions, ignore_index=True)
    if checkpoint_dir is not None:
        output_dir = Path(checkpoint_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        all_predictions.to_parquet(output_dir / "predictions_all.parquet", index=False)
    report = phase1_report(all_predictions[all_predictions["split"] == "test"], config)
    return {
        "fold_results": fold_results,
        "predictions": all_predictions,
        "report": report,
    }
