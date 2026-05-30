from __future__ import annotations

import torch
from torch import nn

from yenibot.models.tcn import CausalTCN


class HybridEncoder(nn.Module):
    """TCN + GRU binary sequence encoder producing P(Long)."""

    def __init__(
        self,
        n_features: int,
        *,
        seq_len: int = 64,
        tcn_channels: int = 64,
        tcn_kernel_size: int = 3,
        tcn_dilations: list[int] | None = None,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.2,
        fusion_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        dilations = tcn_dilations or [1, 2, 4, 8, 16]
        self.tcn = CausalTCN(
            n_features,
            tcn_channels,
            kernel_size=tcn_kernel_size,
            dilations=dilations,
            dropout=dropout,
        )
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            dropout=dropout if gru_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        fusion_dim = tcn_channels + gru_hidden
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.fusion_hidden = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(fusion_hidden, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("HybridEncoder input must have shape (batch, seq_len, n_features)")
        tcn_last = self.tcn(x)[:, -1, :]
        gru_out, _ = self.gru(x)
        gru_last = gru_out[:, -1, :]
        fused = self.fusion_norm(torch.cat([tcn_last, gru_last], dim=-1))
        return self.fusion_hidden(fused)

    def forward(self, x: torch.Tensor, *, return_logits: bool = False) -> torch.Tensor:
        logits = self.output(self.encode(x)).squeeze(-1)
        if return_logits:
            return logits
        return torch.sigmoid(logits)
