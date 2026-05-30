"""Feature engineering package."""

from yenibot.features.builder import (
    build_feature_matrix,
    compute_bar_features,
    compute_funding_rate_features,
    compute_futures_metrics_features,
    compute_intrahour_order_flow_features,
    feature_availability_columns,
    filter_feature_columns,
    raw_order_flow_v2_model_exclusions,
    resolve_feature_profile,
    select_feature_columns,
)
from yenibot.features.wavelet import causal_wavelet_denoise

__all__ = [
    "build_feature_matrix",
    "compute_bar_features",
    "compute_funding_rate_features",
    "compute_futures_metrics_features",
    "compute_intrahour_order_flow_features",
    "feature_availability_columns",
    "filter_feature_columns",
    "raw_order_flow_v2_model_exclusions",
    "resolve_feature_profile",
    "select_feature_columns",
    "causal_wavelet_denoise",
]
