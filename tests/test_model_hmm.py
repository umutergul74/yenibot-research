from __future__ import annotations

import logging

import numpy as np
import torch

from yenibot.models import HybridEncoder
from yenibot.regime import OnlineGaussianHMM
from yenibot.regime.hmm import _hmmlearn_convergence_warning_scope


def test_hybrid_encoder_returns_binary_probability_shape() -> None:
    model = HybridEncoder(
        5,
        seq_len=8,
        tcn_channels=8,
        tcn_dilations=[1, 2],
        gru_hidden=8,
        gru_layers=1,
        dropout=0.0,
        fusion_hidden=8,
    )
    out = model(torch.randn(4, 8, 5))
    assert out.shape == (4,)
    assert torch.all((out >= 0) & (out <= 1))


def test_hmm_online_probability_is_forward_only() -> None:
    rng = np.random.default_rng(42)
    x = np.vstack(
        [
            rng.normal(-1, 0.2, size=(30, 3)),
            rng.normal(0, 0.2, size=(30, 3)),
            rng.normal(1, 0.2, size=(30, 3)),
        ]
    )
    hmm = OnlineGaussianHMM(n_states=3, n_iter=20, random_state=42).fit(x)
    short = hmm.predict_proba_online(x[:10], update_stats=False)
    full = hmm.predict_proba_online(x[:30], update_stats=False)
    np.testing.assert_allclose(short[0], full[0])
    np.testing.assert_allclose(full.sum(axis=1), np.ones(len(full)))


def test_hmmlearn_convergence_warning_scope_suppresses_warning(caplog) -> None:
    logger = logging.getLogger("hmmlearn.base")
    caplog.set_level(logging.WARNING, logger="hmmlearn.base")

    with _hmmlearn_convergence_warning_scope(True):
        logger.warning("Model is not converging.")

    assert "Model is not converging" not in caplog.text

    with _hmmlearn_convergence_warning_scope(False):
        logger.warning("Model is not converging.")

    assert "Model is not converging" in caplog.text
