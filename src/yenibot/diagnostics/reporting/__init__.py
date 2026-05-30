"""Phase 1 diagnostics reporting engine.

This package generates fold-level metrics, feature importance,
calibration analysis, threshold optimization, score-band diagnostics,
and markdown summaries.

The monolith ``_monolith.py`` module contains the full original
implementation.  Public APIs are re-exported here so callers can use::

    from yenibot.diagnostics.reporting import fold_diagnostics, phase1_report

A future refactoring pass will decompose ``_monolith`` into focused
sub-modules (fold_metrics, threshold_analysis, score_analysis, etc.).
"""

from yenibot.diagnostics.reporting._monolith import (  # noqa: F401
    attach_threshold_summary_to_phase1_report,
    bad_fold_feature_forensics,
    bad_fold_feature_forensics_summary,
    bad_fold_group_forensics,
    bad_fold_group_forensics_summary,
    bad_fold_regime_diagnostics,
    best_f1_threshold,
    classify_feature_column,
    constrained_f1_threshold,
    experiment_ledger_diagnostics,
    feature_group_diagnostics,
    feature_group_importance_summary,
    feature_profile_diagnostics,
    fold_diagnostics,
    good_bad_feature_audit,
    good_bad_fold_summary,
    model_feature_columns_frame,
    mtf_leakage_diagnostics,
    recent_fold_diagnostics,
    regime_by_fold_diagnostics,
    regime_diagnostics,
    score_band_by_fold_diagnostics,
    score_band_diagnostics,
    score_band_summary_diagnostics,
    score_lift_by_fold_diagnostics,
    score_lift_diagnostics,
    score_policy_grid_diagnostics,
    select_score_policy,
    stationarity_policy_diagnostics,
    threshold_diagnostics,
    threshold_grid_diagnostics,
    threshold_grid_summary_diagnostics,
    threshold_summary_diagnostics,
    write_phase1_diagnostic_bundle,
)

__all__ = [
    "attach_threshold_summary_to_phase1_report",
    "bad_fold_feature_forensics",
    "bad_fold_feature_forensics_summary",
    "bad_fold_group_forensics",
    "bad_fold_group_forensics_summary",
    "bad_fold_regime_diagnostics",
    "best_f1_threshold",
    "classify_feature_column",
    "constrained_f1_threshold",
    "experiment_ledger_diagnostics",
    "feature_group_diagnostics",
    "feature_group_importance_summary",
    "feature_profile_diagnostics",
    "fold_diagnostics",
    "good_bad_feature_audit",
    "good_bad_fold_summary",
    "model_feature_columns_frame",
    "mtf_leakage_diagnostics",
    "recent_fold_diagnostics",
    "regime_by_fold_diagnostics",
    "regime_diagnostics",
    "score_band_by_fold_diagnostics",
    "score_band_diagnostics",
    "score_band_summary_diagnostics",
    "score_lift_by_fold_diagnostics",
    "score_lift_diagnostics",
    "score_policy_grid_diagnostics",
    "select_score_policy",
    "stationarity_policy_diagnostics",
    "threshold_diagnostics",
    "threshold_grid_diagnostics",
    "threshold_grid_summary_diagnostics",
    "threshold_summary_diagnostics",
    "write_phase1_diagnostic_bundle",
]
