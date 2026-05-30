from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE

from yenibot.diagnostics.metrics import rank_ic
from yenibot.diagnostics.reporting import classify_feature_column
from yenibot.models import HybridEncoder
from yenibot.training.dataset import SequenceDataset


def _cfg(mapping: Any, key: str, default: Any) -> Any:
    if isinstance(mapping, dict):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


def load_fold_model(checkpoint_path: str | Path, *, device: str | torch.device = "cpu") -> tuple[HybridEncoder, list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    feature_columns = list(checkpoint["feature_columns"])
    model_cfg = checkpoint.get("config_model", {})
    model = HybridEncoder(
        len(feature_columns),
        seq_len=int(_cfg(model_cfg, "seq_len", 64)),
        tcn_channels=int(_cfg(model_cfg, "tcn_channels", 64)),
        tcn_kernel_size=int(_cfg(model_cfg, "tcn_kernel_size", 3)),
        tcn_dilations=list(_cfg(model_cfg, "tcn_dilations", [1, 2, 4, 8, 16])),
        gru_hidden=int(_cfg(model_cfg, "gru_hidden", 128)),
        gru_layers=int(_cfg(model_cfg, "gru_layers", 2)),
        dropout=float(_cfg(model_cfg, "dropout", 0.2)),
        fusion_hidden=int(_cfg(model_cfg, "fusion_hidden", 128)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, feature_columns


def predict_probabilities(
    model: HybridEncoder,
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    seq_len: int,
    device: str | torch.device = "cpu",
    batch_size: int = 512,
) -> pd.DataFrame:
    working = frame.copy().reset_index(drop=True)
    _assert_feature_order(working, feature_columns)
    labels = working["label"].to_numpy() if "label" in working.columns else np.zeros(len(working))
    returns = (
        working["forward_return"].to_numpy()
        if "forward_return" in working.columns
        else working.get("fwd_return_10h", pd.Series(np.zeros(len(working)))).to_numpy()
    )
    dataset = SequenceDataset(
        working[feature_columns].to_numpy(dtype=np.float32),
        labels.astype(np.float32),
        returns.astype(np.float32),
        seq_len=seq_len,
    )
    probs: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    torch_device = torch.device(device)
    with torch.no_grad():
        for x, _, _, pos in loader:
            probs.append(model(x.to(torch_device)).cpu().numpy())
            positions.append(pos.numpy())
    out = working.iloc[np.concatenate(positions)].copy().reset_index(drop=True)
    out["prob_long_recomputed"] = np.concatenate(probs)
    return out


def extract_embeddings(
    model: HybridEncoder,
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    seq_len: int,
    device: str | torch.device = "cpu",
    batch_size: int = 512,
) -> pd.DataFrame:
    working = frame.copy().reset_index(drop=True)
    _assert_feature_order(working, feature_columns)
    labels = working["label"].to_numpy() if "label" in working.columns else np.zeros(len(working))
    returns = (
        working["forward_return"].to_numpy()
        if "forward_return" in working.columns
        else working.get("fwd_return_10h", pd.Series(np.zeros(len(working)))).to_numpy()
    )
    dataset = SequenceDataset(
        working[feature_columns].to_numpy(dtype=np.float32),
        labels.astype(np.float32),
        returns.astype(np.float32),
        seq_len=seq_len,
    )
    embeddings: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    torch_device = torch.device(device)
    with torch.no_grad():
        for x, _, _, pos in loader:
            embeddings.append(model.encode(x.to(torch_device)).cpu().numpy())
            positions.append(pos.numpy())
    out = working.iloc[np.concatenate(positions)].copy().reset_index(drop=True)
    emb = np.concatenate(embeddings)
    embedding_columns = [f"embedding_{idx}" for idx in range(emb.shape[1])]
    embedding_frame = pd.DataFrame(emb, columns=embedding_columns, index=out.index)
    return pd.concat([out, embedding_frame], axis=1)


def tsne_embeddings(embedding_frame: pd.DataFrame, *, random_state: int = 42) -> pd.DataFrame:
    embedding_columns = [column for column in embedding_frame.columns if column.startswith("embedding_")]
    if not embedding_columns:
        raise ValueError("No embedding columns found")
    perplexity = min(30, max(5, (len(embedding_frame) - 1) // 3))
    coords = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=random_state).fit_transform(
        embedding_frame[embedding_columns].to_numpy(dtype=np.float32)
    )
    out = embedding_frame.copy()
    out["tsne_x"] = coords[:, 0]
    out["tsne_y"] = coords[:, 1]
    return out


def permutation_importance_rank_ic(
    model: HybridEncoder,
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    seq_len: int,
    forward_return_column: str = "forward_return",
    n_repeats: int = 3,
    random_state: int = 42,
    device: str | torch.device = "cpu",
) -> pd.DataFrame:
    _assert_feature_order(frame, feature_columns)
    if forward_return_column not in frame.columns:
        forward_return_column = "fwd_return_10h"
    baseline = predict_probabilities(
        model,
        frame,
        feature_columns,
        seq_len=seq_len,
        device=device,
    )
    baseline_ic = rank_ic(baseline["prob_long_recomputed"], baseline[forward_return_column])
    rng = np.random.default_rng(random_state)
    rows = []
    for column in feature_columns:
        drops = []
        for _ in range(n_repeats):
            permuted = frame.copy()
            permuted[column] = rng.permutation(permuted[column].to_numpy())
            pred = predict_probabilities(
                model,
                permuted,
                feature_columns,
                seq_len=seq_len,
                device=device,
            )
            permuted_ic = rank_ic(pred["prob_long_recomputed"], pred[forward_return_column])
            drops.append(baseline_ic - permuted_ic)
        rows.append({"feature": column, "rank_ic_drop": float(np.mean(drops)), "baseline_rank_ic": baseline_ic})
    return pd.DataFrame(rows).sort_values("rank_ic_drop", ascending=False).reset_index(drop=True)


def permutation_group_importance_rank_ic(
    model: HybridEncoder,
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    seq_len: int,
    forward_return_column: str = "forward_return",
    n_repeats: int = 3,
    random_state: int = 42,
    device: str | torch.device = "cpu",
) -> pd.DataFrame:
    _assert_feature_order(frame, feature_columns)
    if forward_return_column not in frame.columns:
        forward_return_column = "fwd_return_10h"
    baseline = predict_probabilities(
        model,
        frame,
        feature_columns,
        seq_len=seq_len,
        device=device,
    )
    baseline_ic = rank_ic(baseline["prob_long_recomputed"], baseline[forward_return_column])
    grouped_columns: dict[tuple[str, str], list[str]] = {}
    for column in feature_columns:
        grouped_columns.setdefault(classify_feature_column(column), []).append(column)

    rng = np.random.default_rng(random_state)
    rows = []
    for (timeframe, family), columns in sorted(grouped_columns.items()):
        drops = []
        for _ in range(n_repeats):
            permuted = frame.copy()
            order = rng.permutation(len(permuted))
            for column in columns:
                permuted[column] = permuted[column].to_numpy()[order]
            pred = predict_probabilities(
                model,
                permuted,
                feature_columns,
                seq_len=seq_len,
                device=device,
            )
            permuted_ic = rank_ic(pred["prob_long_recomputed"], pred[forward_return_column])
            drops.append(baseline_ic - permuted_ic)
        rows.append(
            {
                "timeframe": timeframe,
                "family": family,
                "feature_count": len(columns),
                "rank_ic_drop": float(np.mean(drops)),
                "baseline_rank_ic": baseline_ic,
                "features": ",".join(columns),
            }
        )
    return pd.DataFrame(rows).sort_values("rank_ic_drop", ascending=False).reset_index(drop=True)


def _assert_feature_order(frame: pd.DataFrame, feature_columns: list[str]) -> None:
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Frame is missing saved feature columns: {missing}")
