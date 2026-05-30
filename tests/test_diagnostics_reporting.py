from __future__ import annotations

import json
import zipfile

import pandas as pd

from yenibot.diagnostics import (
    attach_threshold_summary_to_phase1_report,
    bad_fold_feature_forensics,
    bad_fold_feature_forensics_summary,
    bad_fold_group_forensics,
    bad_fold_group_forensics_summary,
    calibrate_split_probabilities_from_val,
    calibrate_test_probabilities_from_val,
    calibration_table,
    experiment_ledger_diagnostics,
    feature_group_diagnostics,
    feature_group_importance_summary,
    feature_profile_diagnostics,
    fold_diagnostics,
    good_bad_feature_audit,
    score_band_by_fold_diagnostics,
    score_band_diagnostics,
    score_band_summary_diagnostics,
    score_policy_grid_diagnostics,
    select_score_policy,
    score_lift_diagnostics,
    score_lift_by_fold_diagnostics,
    mtf_leakage_diagnostics,
    recent_fold_diagnostics,
    bad_fold_regime_diagnostics,
    regime_by_fold_diagnostics,
    regime_diagnostics,
    stationarity_policy_diagnostics,
    threshold_diagnostics,
    threshold_summary_diagnostics,
    write_phase1_diagnostic_bundle,
)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2022-01-01", periods=12, freq="1h", tz="UTC"),
            "fold": [0] * 6 + [1] * 6,
            "label": [0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
            "prob_long": [0.1, 0.7, 0.2, 0.8, 0.4, 0.6, 0.3, 0.2, 0.5, 0.9, 0.1, 0.7],
            "forward_return": [-0.01, 0.03, -0.02, 0.02, 0.0, 0.01, -0.01, -0.02, 0.01, 0.04, 0.0, 0.02],
            "regime_prob_0": [0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1, 0.1, 0.2, 0.2, 0.3, 0.4],
            "regime_prob_1": [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
            "regime_prob_2": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.3],
            "4h_source_timestamp": pd.date_range("2021-12-31 20:00", periods=12, freq="1h", tz="UTC"),
            "4h_available_timestamp": pd.date_range("2022-01-01", periods=12, freq="1h", tz="UTC"),
            "true_cvd_zscore": [-1.0, 0.5, -0.8, 0.8, -0.5, 1.0, -0.3, -0.2, 0.4, 1.2, -0.9, 0.7],
            "4h_true_cvd_zscore": [-0.5, 0.4, -0.4, 0.7, -0.2, 0.8, -0.1, -0.2, 0.2, 0.9, -0.6, 0.5],
        }
    )


def test_calibration_bins_are_readable_numbers() -> None:
    table = calibration_table(_predictions()["label"], _predictions()["prob_long"], bins=4)

    assert table["bin"].tolist() == [0, 1, 2, 3]
    assert set(table.columns) == {"bin", "count", "mean_prob_long", "actual_long_rate"}


def test_diagnostic_bundle_contains_shareable_outputs(tmp_path) -> None:
    predictions = _predictions()
    calibration = calibration_table(predictions["label"], predictions["prob_long"], bins=4)
    fold_metrics = fold_diagnostics(predictions)
    fold_metrics.loc[fold_metrics["fold"] == 0, "rank_ic"] = 0.20
    fold_metrics.loc[fold_metrics["fold"] == 1, "rank_ic"] = -0.20
    regime_metrics = regime_diagnostics(predictions)
    threshold_metrics = threshold_diagnostics(predictions)
    report = {
        "passed": False,
        "mean_rank_ic": 0.01,
        "std_rank_ic": 0.10,
        "positive_ic_fraction": 0.50,
        "mean_long_f1": 0.20,
        "mean_prauc": 0.35,
        "calibration_separation": 0.01,
        "checks": {"rank_ic_mean": False},
        "alerts": [],
    }

    zip_path = write_phase1_diagnostic_bundle(
        output_dir=tmp_path,
        report=report,
        predictions=predictions,
        calibration=calibration,
        fold_metrics=fold_metrics,
        regime_metrics=regime_metrics,
        threshold_metrics=threshold_metrics,
        threshold_summary=threshold_summary_diagnostics(threshold_metrics),
        stationarity_policy=stationarity_policy_diagnostics(["true_cvd_zscore"], {"features": {"stationarity": {"exclude_patterns": ["*atr_14"]}}}),
        model_feature_columns=["true_cvd_zscore"],
        score_lift_by_fold=score_lift_by_fold_diagnostics(predictions, bins=4),
        recent_fold_summary=recent_fold_diagnostics(fold_metrics, recent_folds=1),
        feature_groups=feature_group_diagnostics(["true_cvd_zscore"]),
        feature_profile=feature_profile_diagnostics(
            ["true_cvd_zscore"],
            {"features": {"active_profile": "base", "profiles": {"base": {"include_patterns": ["*true_cvd*"], "exclude_patterns": []}}}},
        ),
        config={
            "project": {"name": "test"},
            "features": {"active_profile": "base", "profiles": {"base": {"include_patterns": ["*true_cvd*"], "exclude_patterns": []}}},
            "validation": {"calibration_bins": 4},
        },
    )

    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        assert "phase1_report.json" in names
        assert "summary.md" in names
        assert {"test_predictions.parquet", "test_predictions.csv"} & names
        assert "calibration.csv" in names
        assert "fold_metrics.csv" in names
        assert "regime_metrics.csv" in names
        assert "regime_by_fold.csv" in names
        assert "bad_fold_regime_diagnostics.csv" in names
        assert "threshold_summary.csv" in names
        assert "score_lift.csv" in names
        assert "score_band_lift.csv" in names
        assert "score_band_by_fold.csv" in names
        assert "score_band_summary.csv" in names
        assert "model_feature_columns.csv" in names
        assert "stationarity_policy.csv" in names
        assert "score_lift_by_fold.csv" in names
        assert "recent_fold_summary.csv" in names
        assert "feature_groups.csv" in names
        assert "feature_profile.csv" in names
        assert "bad_fold_feature_forensics.csv" in names
        assert "bad_fold_feature_forensics_summary.csv" in names
        assert "bad_fold_group_forensics.csv" in names
        assert "bad_fold_group_forensics_summary.csv" in names
        assert "experiment_ledger.csv" in names
        assert "experiment_ledger.json" in names
        payload = json.loads(archive.read("phase1_report.json"))
        ledger = json.loads(archive.read("experiment_ledger.json"))
    assert payload["passed"] is False
    assert ledger["profile"] == "base"
    assert ledger["feature_count"] == 1
    assert "test_f1_at_selected_threshold" in ledger


def test_calibration_threshold_and_leakage_diagnostics() -> None:
    predictions = pd.concat(
        [
            _predictions().assign(split="val"),
            _predictions().assign(split="test"),
        ],
        ignore_index=True,
    )
    config = {
        "validation": {
            "target_rank_ic": 0.03,
            "max_rank_ic_std": 0.03,
            "min_positive_ic_fraction": 0.75,
            "min_long_f1": 0.45,
            "suspicious_rank_ic": 0.10,
            "random_like_rank_ic": 0.01,
            "calibration_bins": 4,
        }
    }

    calibrated, report, calibrated_table = calibrate_test_probabilities_from_val(predictions, config)
    calibrated_splits = calibrate_split_probabilities_from_val(predictions)
    thresholds = threshold_diagnostics(predictions)
    calibrated_thresholds = threshold_diagnostics(calibrated_splits, score_column="prob_long_calibrated")
    leakage = mtf_leakage_diagnostics(predictions[predictions["split"] == "test"])

    assert "prob_long_calibrated" in calibrated.columns
    assert set(calibrated_splits["split"]) == {"val", "test"}
    assert "mean_rank_ic" in report
    assert len(calibrated_table) == 4
    assert {
        "selected_threshold",
        "constrained_threshold",
        "test_f1_at_constrained_threshold",
        "test_pred_long_rate_at_constrained_threshold",
        "test_oracle_best_f1",
    }.issubset(thresholds.columns)
    assert "test_f1_at_constrained_threshold" in calibrated_thresholds.columns
    assert leakage["passed"].all()


def test_constrained_threshold_caps_pred_long_rate() -> None:
    predictions = pd.DataFrame(
        {
            "fold": [0] * 12,
            "split": ["val"] * 6 + ["test"] * 6,
            "label": [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
            "prob_long": [0.91, 0.88, 0.82, 0.80, 0.78, 0.76, 0.93, 0.85, 0.84, 0.79, 0.72, 0.68],
        }
    )

    thresholds = threshold_diagnostics(
        predictions,
        max_pred_long_rate=0.50,
        min_precision=0.30,
    )

    row = thresholds.iloc[0]
    assert bool(row["constrained_threshold_constraints_satisfied"]) is True
    assert row["source_constrained_pred_long_rate"] <= 0.50
    assert row["test_pred_long_rate_at_constrained_threshold"] <= 0.50
    assert row["constrained_threshold_source"] == "constrained_f1"


def test_guarded_threshold_rejects_too_broad_selected_threshold() -> None:
    report = {
        "checks": {
            "rank_ic_mean": True,
            "rank_ic_std": True,
            "positive_ic_fraction": True,
            "long_f1": False,
            "calibration_separation": True,
        }
    }
    threshold_summary = pd.DataFrame(
        [
            {"metric": "selected_threshold", "mean": 0.30},
            {"metric": "test_f1_at_selected_threshold", "mean": 0.47},
            {"metric": "test_precision_at_selected_threshold", "mean": 0.34},
            {"metric": "test_recall_at_selected_threshold", "mean": 0.75},
            {"metric": "test_pred_long_rate_at_selected_threshold", "mean": 0.86},
            {"metric": "constrained_threshold", "mean": 0.45},
            {"metric": "test_f1_at_constrained_threshold", "mean": 0.43},
            {"metric": "test_precision_at_constrained_threshold", "mean": 0.36},
            {"metric": "test_recall_at_constrained_threshold", "mean": 0.53},
            {"metric": "test_pred_long_rate_at_constrained_threshold", "mean": 0.64},
        ]
    )

    updated = attach_threshold_summary_to_phase1_report(
        report,
        threshold_summary,
        {
            "validation": {
                "min_long_f1": 0.45,
                "threshold_checks": {"max_pred_long_rate": 0.70, "min_precision": 0.30},
            }
        },
    )

    assert updated["threshold_guarded"]["threshold_source"] == "validation_constrained_threshold"
    assert updated["threshold_guarded"]["test_f1_at_guarded_threshold"] == 0.43
    assert "selected_threshold_pred_long_rate_above_guardrail" in updated["threshold_guarded"]["reject_reason"]
    assert updated["passed_threshold_selected"] is False
    assert updated["passed_threshold_guarded"] is False


def test_regime_by_fold_diagnostics_compares_bad_folds() -> None:
    predictions = _predictions()
    fold_metrics = fold_diagnostics(predictions)
    fold_metrics.loc[fold_metrics["fold"] == 0, "rank_ic"] = 0.20
    fold_metrics.loc[fold_metrics["fold"] == 1, "rank_ic"] = -0.20

    by_fold = regime_by_fold_diagnostics(predictions, fold_metrics, bad_ic=-0.08)
    bad_summary = bad_fold_regime_diagnostics(by_fold)

    assert not by_fold.empty
    assert {"fold", "regime", "is_bad_fold", "rank_ic", "long_f1_050"}.issubset(by_fold.columns)
    assert by_fold.loc[by_fold["fold"] == 1, "is_bad_fold"].all()
    assert not bad_summary.empty
    assert {"rank_ic_gap_bad_minus_other", "pred_long_rate_gap_bad_minus_other"}.issubset(bad_summary.columns)


def test_good_bad_feature_audit_returns_ranked_feature_differences() -> None:
    predictions = _predictions()
    fold_metrics = fold_diagnostics(predictions)
    fold_metrics.loc[fold_metrics["fold"] == 0, "rank_ic"] = 0.20
    fold_metrics.loc[fold_metrics["fold"] == 1, "rank_ic"] = -0.20

    audit = good_bad_feature_audit(predictions, fold_metrics, top_n=5)

    assert not audit.empty
    assert {"feature", "ks_stat", "abs_standardized_diff"}.issubset(audit.columns)


def test_stationarity_policy_diagnostics_flags_raw_model_features() -> None:
    config = {
        "features": {
            "stationarity": {
                "exclude_patterns": ["*close_denoised", "*atr_14", "*true_cvd_delta"],
            }
        }
    }
    diagnostics = stationarity_policy_diagnostics(
        ["close_denoised_log_return", "4h_atr_14", "true_cvd_delta_norm"],
        config,
    )

    overall = diagnostics.loc[diagnostics["check"] == "stationarity_policy_overall"].iloc[0]
    assert not bool(overall["passed"])
    assert overall["matched_features"] == "4h_atr_14"


def test_stationarity_policy_diagnostics_flags_raw_order_flow_v2_inputs() -> None:
    config = {
        "features": {
            "order_flow_v2": {
                "enabled": True,
                "stable_only": True,
                "pressure_windows": [3],
            },
            "stationarity": {"exclude_patterns": []},
        }
    }
    diagnostics = stationarity_policy_diagnostics(
        ["cvd_pressure_3", "cvd_pressure_3_stable_rank"],
        config,
    )

    raw_check = diagnostics.loc[diagnostics["check"] == "order_flow_v2_stable_only"].iloc[0]
    assert not bool(raw_check["passed"])
    assert raw_check["matched_features"] == "cvd_pressure_3"


def test_score_lift_diagnostics_reports_top_bin_lift() -> None:
    lift = score_lift_diagnostics(_predictions(), bins=4)

    assert {"score_bin", "actual_long_rate", "base_long_rate", "lift_vs_base", "is_top_bin"}.issubset(lift.columns)
    assert lift["is_top_bin"].sum() == 1
    assert lift.loc[lift["is_top_bin"], "lift_vs_base"].iloc[0] > 1.0


def test_fold_lift_recent_and_feature_group_diagnostics() -> None:
    predictions = _predictions()
    fold_metrics = fold_diagnostics(predictions)
    lift_by_fold = score_lift_by_fold_diagnostics(predictions, bins=3)
    recent = recent_fold_diagnostics(fold_metrics, recent_folds=1)
    groups = feature_group_diagnostics(
        [
            "signed_large_trade_pressure_stable_zscore",
            "4h_cvd_pressure_3_stable_rank",
            "4h_gk_vol_14_stable_rank",
            "gk_vol_14",
        ]
    )
    group_importance = feature_group_importance_summary(
        pd.DataFrame(
            {
                "feature": ["signed_large_trade_pressure_stable_zscore", "gk_vol_14"],
                "rank_ic_drop": [0.05, 0.01],
            }
        )
    )
    profile = feature_profile_diagnostics(
        ["true_cvd_zscore", "gk_vol_14"],
        {"features": {"active_profile": "base", "profiles": {"base": {"include_patterns": ["*true_cvd*"], "exclude_patterns": ["*atr*"]}}}},
    )

    assert {"top_lift_vs_base", "bin_long_rate_spearman"}.issubset(lift_by_fold.columns)
    assert "recent_minus_all" in recent.columns
    assert set(groups["family"]) == {
        "order_flow_v2_stable",
        "volatility_structure",
        "volatility_structure_stable",
    }
    assert "mean_rank_ic_drop" in group_importance.columns
    assert "profile_include_pattern" in set(profile["check"])


def test_score_band_diagnostics_reports_upper_score_ranges() -> None:
    predictions = _predictions()
    bands = [
        {"name": "top_bin", "min_bin": 3, "max_bin": 3},
        {"name": "upper_half", "min_bin": 2, "max_bin": 3},
    ]

    band_lift = score_band_diagnostics(predictions, bins=4, bands=bands)
    band_by_fold = score_band_by_fold_diagnostics(predictions, bins=4, bands=bands)
    summary = score_band_summary_diagnostics(band_by_fold)

    assert band_lift["band"].tolist() == ["top_bin", "upper_half"]
    assert {"selection_rate", "lift_vs_base", "mean_forward_return", "recall", "f1"}.issubset(band_lift.columns)
    assert set(summary["band"]) == {"top_bin", "upper_half"}
    assert "positive_lift_fold_rate" in summary.columns


def test_score_policy_grid_selects_cv_policy() -> None:
    predictions = pd.concat(
        [
            _predictions().assign(split="val"),
            _predictions().assign(split="test"),
        ],
        ignore_index=True,
    )
    grid = score_policy_grid_diagnostics(
        predictions,
        bins=4,
        bands=[
            {"name": "top_bin", "min_bin": 3, "max_bin": 3},
            {"name": "upper_half", "min_bin": 2, "max_bin": 3},
        ],
        threshold_caps=[0.30, 0.50],
        min_precision=0.30,
    )
    selection = select_score_policy(
        grid,
        {
            "validation": {
                "threshold_checks": {"max_pred_long_rate": 0.70, "min_precision": 0.30},
                "policy_selection": {"min_positive_forward_return_fold_rate": 0.0},
            }
        },
    )

    assert {"score_band", "threshold_cap"}.issubset(set(grid["policy_type"]))
    assert not selection.empty
    assert bool(selection.iloc[0]["selected_policy"]) is True
    assert "policy_reject_reason" in selection.columns


def test_experiment_ledger_summarizes_profile_recent_ic_and_top_lift() -> None:
    recent_fold_summary = pd.DataFrame(
        [
            {"metric": "rank_ic", "recent_mean": 0.123},
            {"metric": "long_f1", "recent_mean": 0.456},
        ]
    )
    score_band_summary = pd.DataFrame(
        [
            {
                "band": "top_10",
                "mean_lift_vs_base": 1.12,
                "positive_lift_fold_rate": 0.75,
                "mean_forward_return": 0.002,
            },
            {"band": "top_20", "mean_lift_vs_base": 1.08, "positive_lift_fold_rate": 0.50, "mean_forward_return": 0.001},
        ]
    )
    score_band_lift = pd.DataFrame(
        [
            {"band": "top_10", "lift_vs_base": 0.98, "mean_forward_return": 0.0003},
            {"band": "top_20", "lift_vs_base": 1.04, "mean_forward_return": 0.0007},
        ]
    )
    threshold_summary = pd.DataFrame(
        [
            {"metric": "selected_threshold", "mean": 0.37},
            {"metric": "test_f1_at_selected_threshold", "mean": 0.46},
            {"metric": "test_precision_at_selected_threshold", "mean": 0.32},
            {"metric": "test_recall_at_selected_threshold", "mean": 0.88},
            {"metric": "test_pred_long_rate_at_selected_threshold", "mean": 0.86},
            {"metric": "test_oracle_best_f1", "mean": 0.49},
            {"metric": "test_f1_at_050", "mean": 0.27},
        ]
    )
    fold_metrics = pd.DataFrame(
        [
            {"fold": 0, "rank_ic": 0.12},
            {"fold": 1, "rank_ic": -0.10},
            {"fold": 2, "rank_ic": 0.04},
            {"fold": 3, "rank_ic": -0.02},
            {"fold": 4, "rank_ic": 0.08},
        ]
    )
    score_lift_by_fold = pd.DataFrame(
        [
            {"fold": 0, "top_lift_vs_base": 1.30},
            {"fold": 1, "top_lift_vs_base": 0.90},
            {"fold": 2, "top_lift_vs_base": 1.10},
            {"fold": 3, "top_lift_vs_base": 1.00},
            {"fold": 4, "top_lift_vs_base": 1.20},
        ]
    )

    ledger = experiment_ledger_diagnostics(
        report={
            "mean_rank_ic": 0.04,
            "std_rank_ic": 0.08,
            "positive_ic_fraction": 0.73,
            "mean_long_f1": 0.27,
            "mean_prauc": 0.34,
            "calibration_separation": 0.01,
        },
        config={"features": {"active_profile": "base", "profiles": {"base": {"include_patterns": ["*"], "exclude_patterns": []}}}},
        feature_columns=["a", "b"],
        fold_metrics=fold_metrics,
        recent_fold_summary=recent_fold_summary,
        threshold_summary=threshold_summary,
        score_band_lift=score_band_lift,
        score_lift_by_fold=score_lift_by_fold,
        score_band_summary=score_band_summary,
        fold_scope="triage",
        data_start="2022-01-01 00:00:00+00:00",
        data_end="2022-02-01 00:00:00+00:00",
        promotable=False,
        reject_reason="mean_rank_ic_delta",
        timestamp="2026-05-05T12:00:00+00:00",
    )

    row = ledger.iloc[0].to_dict()
    assert row["timestamp"] == "2026-05-05T12:00:00+00:00"
    assert row["profile"] == "base"
    assert row["feature_count"] == 2
    assert row["fold_scope"] == "triage"
    assert row["data_start"] == "2022-01-01 00:00:00+00:00"
    assert row["data_end"] == "2022-02-01 00:00:00+00:00"
    assert row["recent_rank_ic_mean"] == 0.123
    assert row["negative_ic_count"] == 2
    assert row["negative_ic_fraction"] == 0.4
    assert round(row["worst_5_rank_ic_mean"], 6) == 0.024
    assert row["rank_ic_cvar_20"] == -0.1
    assert row["bad_fold_rank_ic_mean"] == -0.1
    assert row["top_10_bad_fold_lift_mean"] == 0.9
    assert row["selected_threshold_mean"] == 0.37
    assert row["test_f1_at_selected_threshold"] == 0.46
    assert row["test_precision_at_selected_threshold"] == 0.32
    assert row["test_recall_at_selected_threshold"] == 0.88
    assert row["test_pred_long_rate_at_selected_threshold"] == 0.86
    assert row["test_oracle_best_f1"] == 0.49
    assert row["test_f1_at_050"] == 0.27
    assert row["top_10_lift"] == 1.12
    assert row["top_10_lift_fold_mean"] == 1.12
    assert row["top_10_lift_global"] == 0.98
    assert row["top_10_positive_lift_fold_rate"] == 0.75
    assert row["top_10_forward_return_fold_mean"] == 0.002
    assert row["top_10_forward_return_global"] == 0.0003
    assert row["passed_phase1"] is False
    assert row["passed_phase1_selected_threshold"] is False
    assert row["promotable"] is False
    assert row["reject_reason"] == "mean_rank_ic_delta"


def test_threshold_summary_augments_phase1_report_without_masking_core_checks() -> None:
    threshold_summary = pd.DataFrame(
        [
            {"metric": "selected_threshold", "mean": 0.32},
            {"metric": "test_f1_at_selected_threshold", "mean": 0.46},
            {"metric": "test_precision_at_selected_threshold", "mean": 0.32},
            {"metric": "test_recall_at_selected_threshold", "mean": 0.88},
            {"metric": "test_pred_long_rate_at_selected_threshold", "mean": 0.86},
            {"metric": "test_oracle_best_f1", "mean": 0.49},
            {"metric": "test_f1_at_050", "mean": 0.27},
        ]
    )
    report = {
        "checks": {
            "rank_ic_mean": True,
            "rank_ic_std": False,
            "positive_ic_fraction": True,
            "long_f1": False,
            "calibration_separation": True,
        },
        "passed": False,
    }

    updated = attach_threshold_summary_to_phase1_report(
        report,
        threshold_summary,
        {"validation": {"min_long_f1": 0.45}},
    )

    assert updated["threshold_selected"]["test_f1_at_selected_threshold"] == 0.46
    assert updated["checks"]["long_f1"] is False
    assert updated["checks_threshold_selected"]["long_f1_selected_threshold"] is True
    assert updated["passed"] is False
    assert updated["passed_threshold_selected"] is False


def test_bad_fold_forensics_reports_group_signal_changes() -> None:
    predictions = _predictions()
    fold_metrics = fold_diagnostics(predictions)
    fold_metrics.loc[fold_metrics["fold"] == 0, "rank_ic"] = 0.20
    fold_metrics.loc[fold_metrics["fold"] == 1, "rank_ic"] = -0.20

    feature_forensics = bad_fold_feature_forensics(
        predictions,
        fold_metrics,
        feature_columns=["true_cvd_zscore", "4h_true_cvd_zscore"],
    )
    group_forensics = bad_fold_group_forensics(
        predictions,
        fold_metrics,
        feature_columns=["true_cvd_zscore", "4h_true_cvd_zscore"],
    )

    assert {"bad_fold", "feature", "delta_feature_ic_bad_minus_good", "signal_reversal"}.issubset(
        feature_forensics.columns
    )
    assert {"timeframe", "family", "mean_abs_delta_feature_ic", "top_delta_features"}.issubset(
        group_forensics.columns
    )


def test_feature_group_diagnostics_classifies_intrahour_flow() -> None:
    groups = feature_group_diagnostics(["ih15_cvd_pressure_norm", "ih15_aggressive_buy_burst_stable_rank"])

    assert set(groups["timeframe"]) == {"intrahour"}
    assert set(groups["family"]) == {"order_flow_intrahour"}


def test_feature_group_diagnostics_classifies_futures_context() -> None:
    groups = feature_group_diagnostics(
        [
            "fut_oi_change_288_stable_rank",
            "fut_toptrader_count_long_short_log_ratio_stable_zscore",
            "fut_funding_sum_12_stable_tanh",
        ]
    )

    assert set(groups["timeframe"]) == {"futures"}
    assert set(groups["family"]) == {
        "futures_open_interest_context",
        "futures_positioning_context",
        "futures_funding_context",
    }


def test_bad_fold_forensics_summaries_flag_repeated_reversal_sources() -> None:
    feature_forensics = pd.DataFrame(
        [
            {
                "bad_fold": 21,
                "feature": "4h_taker_imbalance_mean_24",
                "timeframe": "4h",
                "family": "order_flow_v2_bounded",
                "good_feature_ic": 0.05,
                "bad_feature_ic": -0.09,
                "delta_feature_ic_bad_minus_good": -0.14,
                "signal_reversal": True,
                "abs_standardized_diff": 0.48,
            },
            {
                "bad_fold": 32,
                "feature": "4h_taker_imbalance_mean_24",
                "timeframe": "4h",
                "family": "order_flow_v2_bounded",
                "good_feature_ic": 0.04,
                "bad_feature_ic": -0.10,
                "delta_feature_ic_bad_minus_good": -0.14,
                "signal_reversal": True,
                "abs_standardized_diff": 0.31,
            },
            {
                "bad_fold": 21,
                "feature": "4h_gk_vol_14",
                "timeframe": "4h",
                "family": "volatility_structure",
                "good_feature_ic": 0.01,
                "bad_feature_ic": 0.02,
                "delta_feature_ic_bad_minus_good": 0.01,
                "signal_reversal": False,
                "abs_standardized_diff": 0.12,
            },
        ]
    )
    group_forensics = pd.DataFrame(
        [
            {
                "bad_fold": 21,
                "timeframe": "4h",
                "family": "order_flow_v2_bounded",
                "feature_count": 8,
                "mean_delta_feature_ic_bad_minus_good": -0.08,
                "mean_abs_delta_feature_ic": 0.08,
                "signal_reversal_rate": 0.50,
                "mean_abs_standardized_diff": 0.35,
                "top_delta_features": "4h_taker_imbalance_mean_24",
                "top_shifted_features": "4h_taker_imbalance_mean_24",
            },
            {
                "bad_fold": 32,
                "timeframe": "4h",
                "family": "order_flow_v2_bounded",
                "feature_count": 8,
                "mean_delta_feature_ic_bad_minus_good": -0.07,
                "mean_abs_delta_feature_ic": 0.07,
                "signal_reversal_rate": 0.40,
                "mean_abs_standardized_diff": 0.30,
                "top_delta_features": "4h_taker_imbalance_mean_12",
                "top_shifted_features": "4h_taker_imbalance_mean_12",
            },
        ]
    )

    feature_summary = bad_fold_feature_forensics_summary(feature_forensics)
    group_summary = bad_fold_group_forensics_summary(group_forensics)

    flagged = feature_summary[feature_summary["feature"] == "4h_taker_imbalance_mean_24"].iloc[0]
    assert flagged["bad_fold_count"] == 2
    assert flagged["recommended_action"] == "ablate_or_bound"
    assert group_summary.iloc[0]["recommended_action"] == "ablate_or_split"
