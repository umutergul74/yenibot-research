from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from hmmlearn.hmm import GaussianHMM


@dataclass
class OnlineStats:
    n: np.ndarray
    bars_since_update: int = 0

    def to_dict(self) -> dict[str, object]:
        return {"N": self.n.tolist(), "bars_since_update": self.bars_since_update}


class OnlineGaussianHMM:
    """Gaussian HMM with forward-only inference and anti-collapse stats."""

    def __init__(
        self,
        *,
        n_states: int = 3,
        covariance_type: str = "full",
        n_iter: int = 200,
        random_state: int = 42,
        gamma_floor: float = 0.02,
        state_weight_floor: float = 0.08,
        n_ratio_alarm: float = 15.0,
        suppress_convergence_warnings: bool = True,
    ) -> None:
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.gamma_floor = gamma_floor
        self.state_weight_floor = state_weight_floor
        self.n_ratio_alarm = n_ratio_alarm
        self.suppress_convergence_warnings = suppress_convergence_warnings
        self.model: GaussianHMM | None = None
        self._snapshot: dict[str, np.ndarray] | None = None
        self._online_stats: OnlineStats | None = None

    def fit(self, x: np.ndarray) -> "OnlineGaussianHMM":
        x = self._clean_x(x)
        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            min_covar=1e-6,
        )
        with _hmmlearn_convergence_warning_scope(self.suppress_convergence_warnings):
            self.model.fit(x)
        self._snapshot = {
            "means": self.model.means_.copy(),
            "covars": self.model.covars_.copy(),
            "startprob": self.model.startprob_.copy(),
            "transmat": self.model.transmat_.copy(),
        }
        self._init_online_stats()
        return self

    def predict_proba_train(self, x: np.ndarray) -> np.ndarray:
        self._require_model()
        return self.model.predict_proba(self._clean_x(x))

    def predict_proba_online(self, x: np.ndarray, *, update_stats: bool = True) -> np.ndarray:
        """Forward-filtered probabilities. Does not use future observations."""

        self._require_model()
        data = self._clean_x(x)
        emissions = self._emission_likelihood(data)
        startprob = np.maximum(self.model.startprob_, 1e-12)
        startprob = startprob / startprob.sum()
        transmat = np.maximum(self.model.transmat_, 1e-12)
        transmat = transmat / transmat.sum(axis=1, keepdims=True)

        probs = np.zeros((len(data), self.n_states), dtype=float)
        alpha = startprob * emissions[0]
        alpha = self._normalize(alpha)
        probs[0] = alpha
        if update_stats:
            self.update_online(alpha)

        for idx in range(1, len(data)):
            alpha = (alpha @ transmat) * emissions[idx]
            alpha = self._normalize(alpha)
            probs[idx] = alpha
            if update_stats:
                self.update_online(alpha)
        return probs

    def update_online(self, gamma: np.ndarray) -> None:
        if self._online_stats is None:
            self._init_online_stats()
        floored = np.maximum(gamma.astype(float), self.gamma_floor)
        floored = floored / floored.sum()
        self._online_stats.n += floored
        self._online_stats.bars_since_update += 1
        self._redistribute_if_needed()

    def get_online_stats_dict(self) -> dict[str, object]:
        if self._online_stats is None:
            self._init_online_stats()
        return self._online_stats.to_dict()

    def set_online_stats_dict(self, payload: dict[str, object]) -> None:
        n = np.asarray(payload.get("N", np.ones(self.n_states)), dtype=float)
        if n.shape != (self.n_states,):
            raise ValueError("Online stats N has the wrong shape")
        self._online_stats = OnlineStats(
            n=n,
            bars_since_update=int(payload.get("bars_since_update", 0)),
        )

    def reset_online_stats(self) -> None:
        self._init_online_stats()
        self._online_stats.bars_since_update = 0

    def _redistribute_if_needed(self) -> None:
        if self._online_stats is None or self._snapshot is None or self.model is None:
            return
        n = self._online_stats.n
        total = float(n.sum())
        if total <= 0:
            return
        shares = n / total
        ratio = float(n.max() / max(n.min(), 1e-12))
        starving = shares < self.state_weight_floor
        if ratio <= self.n_ratio_alarm and not starving.any():
            return

        for state in np.where(starving)[0]:
            self.model.means_[state] = self._snapshot["means"][state]
            self.model.covars_[state] = self._snapshot["covars"][state]
            n[state] = max(n[state], total * self.state_weight_floor)
        self._online_stats.n = n

    def _init_online_stats(self) -> None:
        self._online_stats = OnlineStats(n=np.ones(self.n_states, dtype=float))

    def _require_model(self) -> None:
        if self.model is None:
            raise RuntimeError("HMM must be fit before inference")

    @staticmethod
    def _clean_x(x: np.ndarray) -> np.ndarray:
        data = np.asarray(x, dtype=float)
        if data.ndim != 2:
            raise ValueError("HMM input must have shape (n_samples, n_features)")
        if len(data) == 0:
            raise ValueError("HMM input is empty")
        return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        total = float(values.sum())
        if not np.isfinite(total) or total <= 0:
            return np.full_like(values, 1.0 / len(values))
        return values / total

    def _emission_likelihood(self, x: np.ndarray) -> np.ndarray:
        self._require_model()
        likelihoods = np.zeros((len(x), self.n_states), dtype=float)
        for state in range(self.n_states):
            mean = self.model.means_[state]
            cov = self.model.covars_[state]
            if cov.ndim == 1:
                cov_matrix = np.diag(cov)
            else:
                cov_matrix = cov
            cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6
            likelihoods[:, state] = _multivariate_normal_pdf(x, mean, cov_matrix)
        return np.maximum(likelihoods, 1e-300)


def _multivariate_normal_pdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    dim = mean.shape[0]
    inv = np.linalg.pinv(cov)
    det = max(float(np.linalg.det(cov)), 1e-300)
    diff = x - mean
    exponent = -0.5 * np.sum(diff @ inv * diff, axis=1)
    normalizer = np.sqrt(((2.0 * np.pi) ** dim) * det)
    return np.exp(exponent) / normalizer


@contextmanager
def _hmmlearn_convergence_warning_scope(enabled: bool):
    logger = logging.getLogger("hmmlearn.base")
    previous_level = logger.level
    if enabled:
        logger.setLevel(max(previous_level, logging.ERROR))
    try:
        yield
    finally:
        if enabled:
            logger.setLevel(previous_level)
