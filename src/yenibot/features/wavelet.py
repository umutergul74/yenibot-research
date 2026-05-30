from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import pywt
except ImportError:  # pragma: no cover - exercised only in minimal local envs
    pywt = None


def _denoise_segment(
    segment: np.ndarray,
    *,
    wavelet: str,
    level: int,
    threshold_scale: float,
) -> np.ndarray:
    if pywt is None:
        raise ImportError("PyWavelets is required when features.wavelet.enabled is true")
    coeffs = pywt.wavedec(segment, wavelet=wavelet, level=level, mode="symmetric")
    detail = coeffs[-1]
    sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745 if len(detail) else 0.0
    threshold = threshold_scale * sigma * np.sqrt(2.0 * np.log(len(segment)))
    filtered = [coeffs[0]]
    filtered.extend(pywt.threshold(part, threshold, mode="soft") for part in coeffs[1:])
    reconstructed = pywt.waverec(filtered, wavelet=wavelet, mode="symmetric")
    return reconstructed[: len(segment)]


def causal_wavelet_denoise(
    series: pd.Series,
    *,
    window: int = 256,
    wavelet: str = "db4",
    level: int = 2,
    threshold_scale: float = 0.5,
) -> pd.Series:
    """Causal rolling wavelet denoising.

    The value at index i is computed from rows up to and including i. Future rows
    are never visible to the transform.
    """

    if window <= 1:
        raise ValueError("window must be greater than 1")
    values = series.astype(float).to_numpy()
    result = np.full(len(values), np.nan, dtype=float)
    for end in range(window, len(values) + 1):
        segment = values[end - window : end]
        result[end - 1] = _denoise_segment(
            segment,
            wavelet=wavelet,
            level=level,
            threshold_scale=threshold_scale,
        )[-1]
    return pd.Series(result, index=series.index, name=f"{series.name}_denoised")
