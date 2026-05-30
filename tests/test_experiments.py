from __future__ import annotations

import copy
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from yenibot.config import load_config
from yenibot.experiments import (
    _attach_holdout_soft_pass,
    _auto_full_profiles,
    _best_profile_blend,
    _experiment_selection_frame,
    _experiment_policy_guard_frame,
    _fold_reliability_gate_frame,
    _fold_reliability_gate_summary_frame,
    _fold_stability_forensics_frame,
    _fold_stability_summary_frame,
    _frozen_policy_monitoring_plan_frame,
    _frozen_policy_robustness_frame,
    _future_oos_candidate_plan_frame,
    _feature_drift_forensics_frame,
    _feature_family_drift_summary_frame,
    _holdout_boundary_audit_frame,
    _holdout_reservation_frame,
    _performance_gap_reasons,
    _missing_selected_profiles,
    _preflight_experiment_profiles,
    _passes_full,
    _passes_triage,
    _profile_blend_leaders,
    _profile_blend_predictions,
    _profile_blend_review_frame,
    _probability_quality_forensics_frame,
    _probability_quality_summary_frame,
    _regime_stability_frames,
    _regime_threshold_policy_frames,
    _score_distribution_shift_frame,
    _score_distribution_shift_summary_frame,
    _bad_fold_signature_frame,
    _score_separation_forensics_frame,
    _threshold_forensics_frame,
    _threshold_policy_review_frame,
    _threshold_transfer_review_frames,
    experiment_settings,
    prepare_training_holdout_split,
    profile_config,
    resolve_experiment_run_id,
    run_experiment_matrix,
    run_profile_experiment,
    write_experiment_diagnostics,
)
from yenibot.features import build_feature_matrix, filter_feature_columns, select_feature_columns


def _labeled_frame(synthetic_klines, config: dict, *, periods: int = 220) -> tuple[pd.DataFrame, list[str]]:
    primary = synthetic_klines(periods, "1h")
    htf = synthetic_klines(max(70, periods // 3), "4h")
    features = build_feature_matrix(primary, htf, config)
    frame = features.frame.copy().reset_index(drop=True)
    frame["label"] = (np.arange(len(frame)) % 3 == 0).astype(int)
    frame["fwd_return_10h"] = frame["close"].shift(-10) / frame["close"] - 1.0
    frame = frame.dropna(subset=["fwd_return_10h"]).reset_index(drop=True)
    return frame, features.feature_columns


def test_profile_config_overrides_active_profile_without_mutating_source(tiny_config) -> None:
    config = copy.deepcopy(tiny_config)
    config["features"]["active_profile"] = "base"
    config["features"]["profiles"] = {
        "base": {"include_patterns": ["*gk_vol*"], "exclude_patterns": []},
        "candidate": {"include_patterns": ["*true_cvd*"], "exclude_patterns": []},
    }

    updated = profile_config(config, "candidate")

    assert config["features"]["active_profile"] == "base"
    assert updated["features"]["active_profile"] == "candidate"


def test_profile_config_applies_nested_config_overrides_without_mutating_source(tiny_config) -> None:
    config = copy.deepcopy(tiny_config)
    config["features"]["active_profile"] = "base"
    config["training"]["loss"]["label_margin_weight"] = 0.0
    config["features"]["profiles"] = {
        "base": {"include_patterns": ["*"], "exclude_patterns": []},
        "margin_candidate": {
            "inherit": "base",
            "config_overrides": {"training": {"loss": {"label_margin_weight": 0.05, "label_margin": 0.25}}},
        },
    }

    updated = profile_config(config, "margin_candidate")

    assert config["training"]["loss"]["label_margin_weight"] == 0.0
    assert updated["features"]["active_profile"] == "margin_candidate"
    assert updated["training"]["loss"]["label_margin_weight"] == 0.05
    assert updated["training"]["loss"]["label_margin"] == 0.25


def test_experiment_settings_resolves_control_and_candidates() -> None:
    config = {
        "features": {"active_profile": "fallback"},
        "experiments": {
            "control_profile": "control",
            "candidate_profiles": ["candidate_a", "candidate_a", "candidate_b"],
        },
    }

    settings = experiment_settings(config)

    assert settings["control_profile"] == "control"
    assert settings["profiles"] == ["control", "candidate_a", "candidate_b"]
    assert settings["candidate_profiles"] == ["candidate_a", "candidate_b"]


def test_preflight_keeps_duplicate_feature_candidate_when_config_override_differs(synthetic_klines, tiny_config) -> None:
    config = copy.deepcopy(tiny_config)
    config["features"]["profiles"] = {
        "control": {"include_patterns": ["*"], "exclude_patterns": []},
        "margin_candidate": {
            "inherit": "control",
            "config_overrides": {"training": {"loss": {"label_margin_weight": 0.05}}},
        },
    }
    settings = {
        "control_profile": "control",
        "profiles": ["control", "margin_candidate"],
        "candidate_profiles": ["margin_candidate"],
        "always_full_profiles": ["control", "margin_candidate"],
        "skipped_profiles": [],
    }
    frame, _ = _labeled_frame(synthetic_klines, config, periods=190)

    updated = _preflight_experiment_profiles(settings, frame, config)

    assert updated["profiles"] == ["control", "margin_candidate"]
    assert updated["candidate_profiles"] == ["margin_candidate"]
    assert not updated["skipped_profiles"]


def test_experiment_settings_skips_historically_rejected_candidates() -> None:
    config = {
        "features": {"active_profile": "fallback"},
        "experiments": {
            "control_profile": "control",
            "candidate_profiles": ["candidate_a", "rejected", "candidate_b"],
            "always_full_profiles": ["control", "rejected", "candidate_a"],
            "seed_audit": {"enabled": True, "profiles": ["control", "rejected", "candidate_b"]},
            "experiment_memory": {
                "enabled": True,
                "reject_retests": True,
                "rejected_profiles": {"rejected": {"reason": "known_bad_run"}},
            },
        },
    }

    settings = experiment_settings(config)

    assert settings["candidate_profiles"] == ["candidate_a", "candidate_b"]
    assert settings["always_full_profiles"] == ["control", "candidate_a"]
    assert settings["seed_audit"]["profiles"] == ["control", "candidate_b"]
    assert settings["skipped_profiles"] == [
        {"profile": "rejected", "role": "candidate_profile", "skip_reason": "known_bad_run"},
        {"profile": "rejected", "role": "always_full_profile", "skip_reason": "known_bad_run"},
        {"profile": "rejected", "role": "seed_audit_profile", "skip_reason": "known_bad_run"},
    ]


def test_experiment_policy_guard_locks_profile_search_until_future_oos() -> None:
    config = {
        "features": {
            "active_profile": "control",
            "profiles": {
                "control": {"include_patterns": ["*"], "exclude_patterns": []},
                "benchmark": {"include_patterns": ["*"], "exclude_patterns": []},
                "new_candidate": {"include_patterns": ["*"], "exclude_patterns": []},
            },
        },
        "experiments": {
            "control_profile": "control",
            "candidate_profiles": ["new_candidate"],
            "always_full_profiles": ["control", "benchmark", "new_candidate"],
            "profile_blends": {
                "weighted": [
                    {
                        "name": "control_benchmark_65_35",
                        "method": "prob_weighted",
                        "profiles": ["control", "benchmark"],
                        "weights": [0.65, 0.35],
                    }
                ]
            },
            "holdout": {"holdout_data_end": "2026-05-13 08:00:00+00:00"},
            "policy_review": {
                "enabled": True,
                "status": "failed_clean_holdout_review",
                "future_oos_candidates": ["blend_control_benchmark_65_35"],
                "future_oos_monitor": {
                    "enabled": True,
                    "anchor_run_id": "anchor",
                    "anchor_data_end": "2026-05-13 08:00:00+00:00",
                    "min_new_bars": 720,
                    "preferred_new_bars": 2160,
                    "allow_holdout_roll_forward": False,
                },
            },
        },
    }

    settings = experiment_settings(config)

    assert settings["candidate_profiles"] == []
    assert settings["profiles"] == ["control"]
    assert settings["always_full_profiles"] == ["control", "benchmark"]
    guard = settings["experiment_policy_guard"]
    assert guard["profile_search_locked"] is True
    assert guard["action"] == "wait_for_new_unseen_bars_keep_control_profile"
    assert guard["future_oos_ready"] is False
    assert guard["blocked_candidate_profiles"] == ["new_candidate"]
    assert guard["blocked_full_profiles"] == ["new_candidate"]
    skipped = settings["skipped_profiles"]
    assert any(item["profile"] == "new_candidate" and item["role"] == "candidate_profile" for item in skipped)
    assert any(item["profile"] == "new_candidate" and item["role"] == "always_full_profile" for item in skipped)


def test_experiment_policy_guard_uses_latest_available_data_end_for_future_oos_count() -> None:
    config = {
        "features": {
            "active_profile": "control",
            "profiles": {
                "control": {"include_patterns": ["*"], "exclude_patterns": []},
                "benchmark": {"include_patterns": ["*"], "exclude_patterns": []},
            },
        },
        "experiments": {
            "control_profile": "control",
            "candidate_profiles": [],
            "always_full_profiles": ["control", "benchmark"],
            "holdout": {
                "holdout_data_end": "2026-05-13 08:00:00+00:00",
                "latest_available_data_end": "2026-05-23 21:00:00+00:00",
            },
            "policy_review": {
                "enabled": True,
                "status": "failed_clean_holdout_review",
                "future_oos_candidates": ["benchmark"],
                "future_oos_monitor": {
                    "enabled": True,
                    "anchor_run_id": "anchor",
                    "anchor_data_end": "2026-05-13 08:00:00+00:00",
                    "min_new_bars": 720,
                    "preferred_new_bars": 2160,
                    "allow_holdout_roll_forward": False,
                },
            },
        },
    }

    settings = experiment_settings(config)
    guard = settings["experiment_policy_guard"]
    guard_frame = _experiment_policy_guard_frame(settings, config)
    monitor_frame = _frozen_policy_monitoring_plan_frame(config, settings)

    assert guard["latest_available_data_end"] == "2026-05-23 21:00:00+00:00"
    assert guard["new_bars_since_anchor"] == 253
    assert guard["min_new_bars_remaining"] == 467
    assert guard["profile_search_locked"] is True
    assert guard_frame.loc[0, "new_bars_since_anchor"] == 253
    assert monitor_frame.loc[0, "new_bars_since_anchor"] == 253
    assert monitor_frame.loc[0, "current_holdout_data_end"] == "2026-05-13 08:00:00+00:00"
    assert monitor_frame.loc[0, "latest_available_data_end"] == "2026-05-23 21:00:00+00:00"


def test_experiment_policy_guard_unlocks_after_future_oos_minimum_window() -> None:
    config = {
        "features": {"active_profile": "control"},
        "experiments": {
            "control_profile": "control",
            "candidate_profiles": ["new_candidate"],
            "always_full_profiles": ["control", "new_candidate"],
            "holdout": {"holdout_data_end": "2026-06-13 08:00:00+00:00"},
            "policy_review": {
                "enabled": True,
                "status": "failed_clean_holdout_review",
                "future_oos_monitor": {
                    "enabled": True,
                    "anchor_run_id": "anchor",
                    "anchor_data_end": "2026-05-13 08:00:00+00:00",
                    "min_new_bars": 720,
                    "preferred_new_bars": 2160,
                    "allow_holdout_roll_forward": False,
                },
            },
        },
    }

    settings = experiment_settings(config)

    assert settings["candidate_profiles"] == ["new_candidate"]
    assert settings["always_full_profiles"] == ["control", "new_candidate"]
    assert settings["experiment_policy_guard"]["profile_search_locked"] is False
    assert settings["experiment_policy_guard"]["future_oos_ready"] is True


def test_future_oos_candidate_plan_records_ready_dates() -> None:
    config = {
        "features": {
            "active_profile": "control",
            "profiles": {
                "control": {"include_patterns": ["*"], "exclude_patterns": []},
                "benchmark": {"include_patterns": ["*"], "exclude_patterns": []},
            },
        },
        "experiments": {
            "control_profile": "control",
            "candidate_profiles": [],
            "always_full_profiles": ["control", "benchmark"],
            "profile_blends": {
                "weighted": [
                    {
                        "name": "control_benchmark_65_35",
                        "method": "prob_weighted",
                        "profiles": ["control", "benchmark"],
                        "weights": [0.65, 0.35],
                    }
                ]
            },
            "holdout": {"holdout_data_end": "2026-05-13 08:00:00+00:00"},
            "policy_review": {
                "enabled": True,
                "frozen_candidate": "retired_blend",
                "policy_type": "score_band",
                "status": "failed_clean_holdout_review",
                "future_oos_candidates": ["blend_control_benchmark_65_35"],
                "future_oos_monitor": {
                    "enabled": True,
                    "anchor_run_id": "anchor",
                    "anchor_data_end": "2026-05-13 08:00:00+00:00",
                    "min_new_bars": 720,
                    "preferred_new_bars": 2160,
                    "allow_holdout_roll_forward": False,
                },
            },
        },
    }

    settings = experiment_settings(config)
    plan = _future_oos_candidate_plan_frame(settings, config)

    assert set(plan["candidate"]) == {"control", "retired_blend", "blend_control_benchmark_65_35"}
    assert plan["plan_rank"].tolist() == [1, 2, 3]
    assert set(plan["candidate_label"]) == {"control", "retired_blend", "blend_control_benchmark_65_35"}
    assert plan.loc[plan["plan_rank"].eq(2), "candidate"].iloc[0] == "blend_control_benchmark_65_35"
    assert plan.loc[plan["candidate"].eq("retired_blend"), "candidate_status"].iloc[0] == "historical_retired_policy_do_not_promote"
    assert bool(plan.loc[plan["candidate"].eq("retired_blend"), "promotion_allowed_now"].iloc[0]) is False
    assert plan["min_ready_at"].eq("2026-06-12 08:00:00+00:00").all()
    assert plan["preferred_ready_at"].eq("2026-08-11 08:00:00+00:00").all()
    future = plan.loc[plan["candidate"] == "blend_control_benchmark_65_35"].iloc[0]
    assert future["required_profiles"] == "control,benchmark"
    assert bool(future["all_required_profiles_allowed"]) is True
    assert future["evaluation_status"] == "wait_for_future_oos"


def test_future_oos_candidate_plan_adds_cv_robust_policy_candidates() -> None:
    config = {
        "features": {
            "active_profile": "control",
            "profiles": {
                "control": {"include_patterns": ["*"], "exclude_patterns": []},
                "benchmark": {"include_patterns": ["*"], "exclude_patterns": []},
            },
        },
        "experiments": {
            "control_profile": "control",
            "candidate_profiles": [],
            "always_full_profiles": ["control", "benchmark"],
            "profile_blends": {
                "weighted": [
                    {
                        "name": "control_benchmark_65_35",
                        "method": "prob_weighted",
                        "profiles": ["control", "benchmark"],
                        "weights": [0.65, 0.35],
                    }
                ]
            },
            "holdout": {"holdout_data_end": "2026-05-13 08:00:00+00:00"},
            "policy_review": {
                "enabled": True,
                "status": "failed_clean_holdout_review",
                "future_oos_candidates": ["blend_control_benchmark_65_35"],
                "future_oos_monitor": {
                    "enabled": True,
                    "anchor_run_id": "anchor",
                    "anchor_data_end": "2026-05-13 08:00:00+00:00",
                    "min_new_bars": 720,
                    "preferred_new_bars": 2160,
                    "allow_holdout_roll_forward": False,
                },
            },
        },
    }
    summary = pd.DataFrame(
        [
            {
                "candidate": "control",
                "candidate_type": "profile",
                "evaluation_scope": "cv_test",
                "band": "top_30",
                "future_oos_policy_candidate": True,
                "mean_label_lift_vs_base": 1.11,
                "mean_forward_return": 0.0017,
                "payoff_alignment_fold_rate": 0.50,
            },
            {
                "candidate": "blend_control_benchmark_65_35",
                "candidate_type": "blend",
                "evaluation_scope": "cv_test",
                "band": "top_10",
                "future_oos_policy_candidate": "True",
                "mean_label_lift_vs_base": 1.15,
                "mean_forward_return": 0.0025,
                "payoff_alignment_fold_rate": 0.53,
            },
            {
                "candidate": "control",
                "candidate_type": "profile",
                "evaluation_scope": "holdout",
                "band": "top_30",
                "future_oos_policy_candidate": False,
                "mean_label_lift_vs_base": 1.08,
                "mean_forward_return": -0.0002,
                "mean_tb_return": -0.0005,
                "payoff_alignment_fold_rate": 0.0,
                "reject_reason": "diagnostic_only",
            },
            {
                "candidate": "control",
                "candidate_type": "profile",
                "evaluation_scope": "holdout",
                "band": "top_20",
                "future_oos_policy_candidate": True,
            },
            {
                "candidate": "benchmark",
                "candidate_type": "profile",
                "evaluation_scope": "cv_test",
                "band": "top_10",
                "future_oos_policy_candidate": False,
            },
        ]
    )

    settings = experiment_settings(config)
    plan = _future_oos_candidate_plan_frame(settings, config, summary)
    policy_rows = plan.loc[plan["stage"] == "future_oos_score_band_policy"]

    assert set(policy_rows["candidate"]) == {"control", "blend_control_benchmark_65_35"}
    assert policy_rows["candidate_id"].is_unique
    assert set(policy_rows["candidate_label"]) == {
        "control [top_30]",
        "blend_control_benchmark_65_35 [top_10]",
    }
    control_policy = policy_rows.loc[policy_rows["candidate"] == "control"].iloc[0]
    assert control_policy["candidate_id"] == "control::top_30"
    assert control_policy["policy_name"] == "top_30"
    assert control_policy["required_profiles"] == "control"
    assert control_policy["selection_source"] == "cv_payoff_policy_robustness"
    assert float(control_policy["cv_mean_forward_return"]) == pytest.approx(0.0017)
    assert bool(control_policy["current_holdout_diagnostic_only"]) is True
    assert float(control_policy["current_holdout_mean_forward_return"]) == pytest.approx(-0.0002)
    assert control_policy["current_holdout_reject_reason"] == "diagnostic_only"
    assert bool(control_policy["promotion_allowed_now"]) is False
    blend_policy = policy_rows.loc[policy_rows["candidate"] == "blend_control_benchmark_65_35"].iloc[0]
    assert blend_policy["candidate_id"] == "blend_control_benchmark_65_35::top_10"
    assert blend_policy["candidate_type"] == "weighted_blend_score_band"
    assert blend_policy["required_profiles"] == "control,benchmark"
    assert bool(blend_policy["all_required_profiles_allowed"]) is True


def test_holdout_signal_pass_is_separate_from_threshold_deployment_gate() -> None:
    config = {
        "validation": {
            "target_rank_ic": 0.03,
            "min_long_f1": 0.45,
            "threshold_checks": {"max_pred_long_rate": 0.70},
        }
    }
    row = {
        "mean_rank_ic": 0.06,
        "top_10_lift_global": 1.12,
        "top_10_forward_return_global": 0.002,
        "holdout_policy_pass": True,
        "calibration_separation": 0.01,
        "mtf_leakage_passed": True,
        "stationarity_policy_passed": True,
        "holdout_cv_threshold_f1": 0.48,
        "holdout_cv_threshold_pred_long_rate": 0.78,
    }

    evaluated = _attach_holdout_soft_pass(row, config)

    assert evaluated["holdout_signal_pass"] is True
    assert evaluated["holdout_signal_reject_reason"] == ""
    assert evaluated["holdout_threshold_pass"] is False
    assert evaluated["holdout_threshold_reject_reason"] == "holdout_cv_threshold_pred_long_rate"
    assert evaluated["holdout_soft_pass"] is False
    assert evaluated["holdout_reject_reason"] == "holdout_cv_threshold_pred_long_rate"


def test_performance_gap_reasons_separate_selected_and_constrained_f1() -> None:
    config = {
        "validation": {
            "target_rank_ic": 0.03,
            "max_rank_ic_std": 0.03,
            "min_positive_ic_fraction": 0.75,
            "min_long_f1": 0.45,
            "threshold_checks": {"max_pred_long_rate": 0.70},
        }
    }
    row = {
        "mean_rank_ic": 0.07,
        "std_rank_ic": 0.07,
        "positive_ic_fraction": 0.86,
        "mean_long_f1": 0.31,
        "test_f1_at_selected_threshold": 0.47,
        "test_pred_long_rate_at_selected_threshold": 0.86,
        "test_f1_at_constrained_threshold": 0.43,
        "test_pred_long_rate_at_constrained_threshold": 0.64,
        "top_10_lift_global": 1.13,
        "mtf_leakage_passed": True,
        "stationarity_policy_passed": True,
    }

    reasons = _performance_gap_reasons(row, config).split(";")

    assert "cv_selected_threshold_f1_below_target" not in reasons
    assert "cv_constrained_threshold_f1_below_target" in reasons
    assert "cv_selected_threshold_pred_long_rate_above_guardrail" in reasons
    assert "cv_rank_ic_std_above_phase1_target" in reasons


def test_fold_stability_forensics_identifies_std_drivers() -> None:
    config = {
        "validation": {
            "target_rank_ic": 0.03,
            "bad_fold_ic_threshold": -0.08,
            "min_long_f1": 0.45,
            "threshold_checks": {"max_pred_long_rate": 0.70},
        }
    }
    entry = {
        "profile": "control",
        "fold_scope": "full",
        "diagnostics": {
            "fold_metrics": pd.DataFrame(
                [
                    {"fold": 0, "rank_ic": 0.08, "long_f1": 0.30, "prauc": 0.35},
                    {"fold": 1, "rank_ic": -0.12, "long_f1": 0.20, "prauc": 0.30},
                    {"fold": 2, "rank_ic": 0.06, "long_f1": 0.31, "prauc": 0.36},
                ]
            ),
            "threshold_metrics": pd.DataFrame(
                [
                    {
                        "fold": 0,
                        "test_f1_at_selected_threshold": 0.47,
                        "test_pred_long_rate_at_selected_threshold": 0.80,
                        "test_f1_at_constrained_threshold": 0.44,
                        "test_pred_long_rate_at_constrained_threshold": 0.62,
                    },
                    {
                        "fold": 1,
                        "test_f1_at_selected_threshold": 0.40,
                        "test_pred_long_rate_at_selected_threshold": 0.82,
                        "test_f1_at_constrained_threshold": 0.38,
                        "test_pred_long_rate_at_constrained_threshold": 0.61,
                    },
                    {
                        "fold": 2,
                        "test_f1_at_selected_threshold": 0.46,
                        "test_pred_long_rate_at_selected_threshold": 0.78,
                        "test_f1_at_constrained_threshold": 0.46,
                        "test_pred_long_rate_at_constrained_threshold": 0.60,
                    },
                ]
            ),
            "score_band_by_fold": pd.DataFrame(
                [
                    {"fold": 0, "band": "top_10", "lift_vs_base": 1.2, "mean_forward_return": 0.002},
                    {"fold": 1, "band": "top_10", "lift_vs_base": 0.8, "mean_forward_return": -0.001},
                    {"fold": 2, "band": "top_10", "lift_vs_base": 1.1, "mean_forward_return": 0.001},
                ]
            ),
        },
    }

    forensics = _fold_stability_forensics_frame([entry], config)
    summary = _fold_stability_summary_frame(forensics, config)

    worst = forensics.sort_values("rank_ic").iloc[0]
    assert worst["fold"] == 1
    assert worst["rank_ic_bucket"] == "bad_rank_ic"
    assert worst["primary_issue"] == "bad_rank_ic"
    assert summary.iloc[0]["bad_fold_count"] == 1
    assert summary.iloc[0]["negative_fold_count"] == 1


def test_threshold_forensics_separates_selected_pred_rate_and_constrained_f1() -> None:
    config = {
        "validation": {
            "min_long_f1": 0.45,
            "threshold_checks": {"max_pred_long_rate": 0.70},
        }
    }
    entry = {
        "profile": "control",
        "fold_scope": "full",
        "diagnostics": {
            "threshold_metrics": pd.DataFrame(
                [
                    {
                        "fold": 0,
                        "selected_threshold": 0.30,
                        "test_f1_at_selected_threshold": 0.47,
                        "test_pred_long_rate_at_selected_threshold": 0.86,
                        "constrained_threshold": 0.42,
                        "test_f1_at_constrained_threshold": 0.43,
                        "test_pred_long_rate_at_constrained_threshold": 0.64,
                        "test_oracle_best_f1": 0.50,
                    }
                ]
            )
        },
    }

    frame = _threshold_forensics_frame([entry], config)

    row = frame.iloc[0]
    assert row["primary_issue"] == "official_f1"
    assert row["official_threshold_source"] == "validation_constrained_threshold"
    assert row["test_f1_at_official_threshold"] == row["test_f1_at_constrained_threshold"]
    assert row["selected_pred_rate_excess_vs_guardrail"] > 0
    assert row["constrained_pred_rate_excess_vs_guardrail"] < 0


def test_threshold_forensics_uses_calibrated_official_source_when_selected() -> None:
    config = {
        "validation": {
            "min_long_f1": 0.45,
            "threshold_checks": {"max_pred_long_rate": 0.70},
        }
    }
    entry = {
        "profile": "control",
        "fold_scope": "full",
        "diagnostics": {
            "row": {"official_threshold_source": "calibrated_validation_constrained_threshold"},
            "threshold_metrics": pd.DataFrame(
                [
                    {
                        "fold": 0,
                        "selected_threshold": 0.30,
                        "test_f1_at_selected_threshold": 0.47,
                        "test_pred_long_rate_at_selected_threshold": 0.86,
                        "constrained_threshold": 0.42,
                        "test_f1_at_constrained_threshold": 0.41,
                        "test_precision_at_constrained_threshold": 0.31,
                        "test_recall_at_constrained_threshold": 0.60,
                        "test_pred_long_rate_at_constrained_threshold": 0.64,
                        "test_oracle_best_f1": 0.50,
                    }
                ]
            ),
            "calibrated_threshold_metrics": pd.DataFrame(
                [
                    {
                        "fold": 0,
                        "constrained_threshold": 0.39,
                        "test_f1_at_constrained_threshold": 0.46,
                        "test_precision_at_constrained_threshold": 0.35,
                        "test_recall_at_constrained_threshold": 0.68,
                        "test_pred_long_rate_at_constrained_threshold": 0.62,
                    }
                ]
            ),
        },
    }

    frame = _threshold_forensics_frame([entry], config)

    row = frame.iloc[0]
    assert bool(row["official_threshold_uses_calibration"]) is True
    assert row["official_threshold"] == 0.39
    assert row["test_f1_at_official_threshold"] == 0.46
    assert row["primary_issue"] == "selected_threshold_too_broad"


def test_threshold_policy_review_lists_validation_selected_constrained_and_caps() -> None:
    config = {
        "validation": {
            "min_long_f1": 0.45,
            "threshold_checks": {"max_pred_long_rate": 0.70, "min_precision": 0.30},
        }
    }
    entry = {
        "profile": "control",
        "fold_scope": "full",
        "diagnostics": {
            "threshold_summary": pd.DataFrame(
                [
                    {"metric": "selected_threshold", "mean": 0.25},
                    {"metric": "source_best_f1", "mean": 0.49},
                    {"metric": "test_f1_at_selected_threshold", "mean": 0.47},
                    {"metric": "test_precision_at_selected_threshold", "mean": 0.32},
                    {"metric": "test_recall_at_selected_threshold", "mean": 0.90},
                    {"metric": "test_pred_long_rate_at_selected_threshold", "mean": 0.86},
                    {"metric": "constrained_threshold", "mean": 0.42},
                    {"metric": "source_constrained_f1", "mean": 0.46},
                    {"metric": "source_constrained_precision", "mean": 0.35},
                    {"metric": "source_constrained_pred_long_rate", "mean": 0.62},
                    {"metric": "test_f1_at_constrained_threshold", "mean": 0.43},
                    {"metric": "test_precision_at_constrained_threshold", "mean": 0.34},
                    {"metric": "test_recall_at_constrained_threshold", "mean": 0.68},
                    {"metric": "test_pred_long_rate_at_constrained_threshold", "mean": 0.64},
                ]
            ),
            "threshold_grid_summary": pd.DataFrame(
                [
                    {
                        "max_pred_long_rate": 0.5,
                        "threshold_mean": 0.45,
                        "mean_source_f1": 0.44,
                        "mean_source_precision": 0.36,
                        "mean_source_pred_long_rate": 0.50,
                        "mean_f1": 0.41,
                        "mean_precision": 0.35,
                        "mean_recall": 0.55,
                        "mean_selection_rate": 0.49,
                        "mean_lift_vs_base": 1.1,
                        "mean_forward_return": 0.001,
                        "positive_lift_fold_rate": 0.75,
                        "positive_forward_return_fold_rate": 0.70,
                        "constraints_satisfied_fold_rate": 1.0,
                    }
                ]
            ),
        },
    }

    frame = _threshold_policy_review_frame([entry], config)

    assert {
        "validation_selected_threshold",
        "validation_constrained_threshold",
        "validation_threshold_cap_0.50",
    }.issubset(set(frame["policy_name"]))
    selected = frame.loc[frame["policy_name"] == "validation_selected_threshold"].iloc[0]
    assert bool(selected["pred_long_rate_passed"]) is False
    constrained = frame.loc[frame["policy_name"] == "validation_constrained_threshold"].iloc[0]
    assert constrained["source_selection_metric"] == "source_constrained_f1"


def test_threshold_transfer_review_uses_prior_fold_thresholds_only() -> None:
    config = {
        "validation": {
            "min_long_f1": 0.45,
            "threshold_checks": {"max_pred_long_rate": 0.70, "min_precision": 0.30},
        }
    }
    rows = []
    for fold in range(3):
        for split in ("val", "test"):
            for idx, score in enumerate([0.20, 0.35, 0.55, 0.80]):
                rows.append(
                    {
                        "fold": fold,
                        "split": split,
                        "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=fold * 10 + idx),
                        "label": int(idx >= 2),
                        "prob_long": score + fold * 0.01,
                        "forward_return": 0.001 * (idx - 1),
                        "tb_return": 0.002 * idx,
                    }
                )
    entry = {
        "profile": "control",
        "fold_scope": "full",
        "predictions": pd.DataFrame(rows),
        "diagnostics": {
            "threshold_metrics": pd.DataFrame(
                [
                    {"fold": 0, "selected_threshold": 0.30, "constrained_threshold": 0.40},
                    {"fold": 1, "selected_threshold": 0.50, "constrained_threshold": 0.60},
                    {"fold": 2, "selected_threshold": 0.70, "constrained_threshold": 0.80},
                ]
            )
        },
    }

    summary, by_fold = _threshold_transfer_review_frames([entry], config)

    median_rows = by_fold.loc[by_fold["policy_name"] == "past_median_constrained_threshold"]
    assert median_rows["fold"].tolist() == [1, 2]
    assert float(median_rows.loc[median_rows["fold"] == 1, "threshold"].iloc[0]) == pytest.approx(0.40)
    assert float(median_rows.loc[median_rows["fold"] == 2, "threshold"].iloc[0]) == pytest.approx(0.50)
    assert set(median_rows["selection_guard"]) == {"past_validation_thresholds_only_first_fold_skipped"}
    summary_row = summary.loc[summary["policy_name"] == "past_median_constrained_threshold"].iloc[0]
    assert int(summary_row["fold_count"]) == 2
    assert "test_f1" in summary.columns


def test_score_separation_forensics_flags_bad_fold_signature() -> None:
    config = {
        "validation": {
            "target_rank_ic": 0.03,
            "min_long_f1": 0.45,
            "threshold_checks": {"max_pred_long_rate": 0.70, "min_precision": 0.30},
        }
    }
    rows = []
    for fold, scores in {
        0: [0.20, 0.30, 0.75, 0.85],
        1: [0.80, 0.70, 0.30, 0.20],
        2: [0.25, 0.35, 0.70, 0.90],
    }.items():
        for idx, score in enumerate(scores):
            rows.append(
                {
                    "fold": fold,
                    "split": "test",
                    "timestamp": pd.Timestamp("2024-02-01") + pd.Timedelta(hours=fold * 10 + idx),
                    "label": int(idx >= 2),
                    "prob_long": score,
                    "forward_return": 0.002 if idx >= 2 else -0.001,
                    "tb_return": 0.003 if idx >= 2 else 0.0,
                }
            )
    entry = {
        "profile": "control",
        "fold_scope": "full",
        "predictions": pd.DataFrame(rows),
        "diagnostics": {
            "fold_metrics": pd.DataFrame(
                [
                    {"fold": 0, "rank_ic": 0.12},
                    {"fold": 1, "rank_ic": -0.10},
                    {"fold": 2, "rank_ic": 0.09},
                ]
            ),
            "threshold_metrics": pd.DataFrame(
                [
                    {"fold": 0, "constrained_threshold": 0.5, "test_f1_at_constrained_threshold": 1.0},
                    {"fold": 1, "constrained_threshold": 0.5, "test_f1_at_constrained_threshold": 0.0},
                    {"fold": 2, "constrained_threshold": 0.5, "test_f1_at_constrained_threshold": 1.0},
                ]
            ),
            "score_band_by_fold": pd.DataFrame(
                [
                    {"fold": 0, "band": "top_10", "lift_vs_base": 1.5, "mean_forward_return": 0.002},
                    {"fold": 1, "band": "top_10", "lift_vs_base": 0.5, "mean_forward_return": -0.001},
                    {"fold": 2, "band": "top_10", "lift_vs_base": 1.4, "mean_forward_return": 0.002},
                ]
            ),
        },
    }

    score_forensics = _score_separation_forensics_frame([entry], config)
    signature = _bad_fold_signature_frame(score_forensics, config)

    bad_row = score_forensics.loc[score_forensics["fold"] == 1].iloc[0]
    assert bad_row["primary_issue"] == "negative_rank_ic"
    assert bad_row["score_gap_pos_minus_neg"] < 0
    assert not signature.empty
    signature_row = signature.iloc[0]
    assert signature_row["bad_fold_count"] == 1
    assert "score_separation_compresses_or_reverses" in signature_row["likely_signature"]
    assert "top_score_payoff_reverses" in signature_row["likely_signature"]


def test_feature_drift_forensics_flags_bad_fold_signal_reversal() -> None:
    config = {
        "validation": {
            "target_rank_ic": 0.03,
            "min_long_f1": 0.45,
            "threshold_checks": {"max_pred_long_rate": 0.70, "min_precision": 0.30},
        }
    }
    rows = []
    for fold in range(3):
        feature_values = [0.1, 0.2, 0.8, 0.9]
        returns = [0.0, 0.001, 0.003, 0.004] if fold != 1 else [0.004, 0.003, 0.001, 0.0]
        scores = [0.20, 0.30, 0.75, 0.85] if fold != 1 else [0.80, 0.70, 0.30, 0.20]
        for idx, (feature_value, forward_return, score) in enumerate(zip(feature_values, returns, scores)):
            rows.append(
                {
                    "fold": fold,
                    "split": "test",
                    "timestamp": pd.Timestamp("2024-02-01") + pd.Timedelta(hours=fold * 10 + idx),
                    "label": int(idx >= 2),
                    "prob_long": score,
                    "forward_return": forward_return,
                    "tb_return": max(forward_return, 0.0),
                    "taker_imbalance": feature_value,
                }
            )
    entry = {
        "profile": "control",
        "fold_scope": "full",
        "feature_columns": ["taker_imbalance"],
        "predictions": pd.DataFrame(rows),
        "diagnostics": {
            "fold_metrics": pd.DataFrame(
                [
                    {"fold": 0, "rank_ic": 0.12},
                    {"fold": 1, "rank_ic": -0.10},
                    {"fold": 2, "rank_ic": 0.09},
                ]
            ),
            "threshold_metrics": pd.DataFrame(
                [
                    {"fold": 0, "constrained_threshold": 0.5, "test_f1_at_constrained_threshold": 1.0},
                    {"fold": 1, "constrained_threshold": 0.5, "test_f1_at_constrained_threshold": 0.0},
                    {"fold": 2, "constrained_threshold": 0.5, "test_f1_at_constrained_threshold": 1.0},
                ]
            ),
            "score_band_by_fold": pd.DataFrame(
                [
                    {"fold": 0, "band": "top_10", "lift_vs_base": 1.5, "mean_forward_return": 0.002},
                    {"fold": 1, "band": "top_10", "lift_vs_base": 0.5, "mean_forward_return": -0.001},
                    {"fold": 2, "band": "top_10", "lift_vs_base": 1.4, "mean_forward_return": 0.002},
                ]
            ),
        },
    }

    score_forensics = _score_separation_forensics_frame([entry], config)
    drift = _feature_drift_forensics_frame([entry], score_forensics, config)
    summary = _feature_family_drift_summary_frame(drift)

    row = drift.loc[drift["feature"] == "taker_imbalance"].iloc[0]
    assert row["feature_family"] == "order_flow"
    assert bool(row["return_ic_reversal"]) is True
    assert row["likely_issue"] == "feature_return_ic_reversal"
    assert not summary.empty
    assert int(summary.iloc[0]["return_ic_reversal_count"]) == 1


def test_probability_quality_and_score_shift_forensics_explain_bad_folds() -> None:
    config = {
        "validation": {
            "target_rank_ic": 0.03,
            "min_long_f1": 0.45,
            "calibration_bins": 4,
            "score_lift_bins": 4,
            "threshold_checks": {"max_pred_long_rate": 0.70, "min_precision": 0.30},
        }
    }
    rows = []
    for fold in range(3):
        if fold == 1:
            scores = [0.91, 0.88, 0.82, 0.79, 0.18, 0.16, 0.12, 0.10]
            labels = [0, 0, 0, 0, 1, 1, 1, 1]
            returns = [0.004, 0.003, 0.002, 0.001, -0.001, -0.002, -0.003, -0.004]
        else:
            scores = [0.12, 0.18, 0.22, 0.28, 0.72, 0.78, 0.82, 0.88]
            labels = [0, 0, 0, 0, 1, 1, 1, 1]
            returns = [-0.004, -0.003, -0.002, -0.001, 0.001, 0.002, 0.003, 0.004]
        for idx, (score, label, forward_return) in enumerate(zip(scores, labels, returns)):
            rows.append(
                {
                    "fold": fold,
                    "split": "test",
                    "timestamp": pd.Timestamp("2024-03-01") + pd.Timedelta(hours=fold * 20 + idx),
                    "label": label,
                    "prob_long": score,
                    "forward_return": forward_return,
                    "tb_return": max(forward_return, 0.0),
                }
            )
    entry = {
        "profile": "control",
        "fold_scope": "full",
        "feature_columns": [],
        "predictions": pd.DataFrame(rows),
        "diagnostics": {
            "fold_metrics": pd.DataFrame(
                [
                    {"fold": 0, "rank_ic": 0.20},
                    {"fold": 1, "rank_ic": -0.20},
                    {"fold": 2, "rank_ic": 0.18},
                ]
            ),
            "threshold_metrics": pd.DataFrame(
                [
                    {"fold": 0, "official_threshold": 0.5, "test_f1_at_official_threshold": 1.0},
                    {"fold": 1, "official_threshold": 0.5, "test_f1_at_official_threshold": 0.0},
                    {"fold": 2, "official_threshold": 0.5, "test_f1_at_official_threshold": 1.0},
                ]
            ),
        },
    }

    quality = _probability_quality_forensics_frame([entry], config)
    quality_summary = _probability_quality_summary_frame(quality, config)
    score_shift = _score_distribution_shift_frame([entry], config)
    score_shift_summary = _score_distribution_shift_summary_frame(score_shift, config)

    bad_quality = quality.loc[quality["fold"] == 1].iloc[0]
    assert bad_quality["primary_issue"] == "negative_rank_ic"
    assert bad_quality["brier_score"] > quality.loc[quality["fold"] == 0, "brier_score"].iloc[0]
    assert not quality_summary.empty
    assert quality_summary.iloc[0]["probability_quality_issue"] in {
        "bad_folds_lose_ranking_resolution",
        "bad_folds_calibration_worsens",
        "monitor",
    }
    bad_shift = score_shift.loc[score_shift["fold"] == 1].iloc[0]
    assert bad_shift["score_ks_vs_reference"] > 0
    assert bad_shift["score_psi_vs_reference"] >= 0
    assert not score_shift_summary.empty


def test_fold_reliability_gate_uses_validation_only_to_mark_accepted_folds() -> None:
    config = {
        "validation": {
            "fold_reliability_gates": {
                "enabled": True,
                "min_accepted_fraction": 0.50,
                "min_accepted_folds": 1,
                "min_positive_ic_fraction": 0.50,
                "max_rank_ic_std": 0.50,
                "min_official_f1_delta": 0.0,
                "gates": [{"name": "val_rank_ic_positive", "min_val_rank_ic": 0.0}],
            }
        }
    }

    def rows(fold: int, split: str, probs: list[float], returns: list[float]) -> list[dict[str, object]]:
        labels = [0, 0, 1, 1]
        base = pd.Timestamp("2024-04-01", tz="UTC") + pd.Timedelta(hours=fold * 20)
        return [
            {
                "fold": fold,
                "split": split,
                "timestamp": base + pd.Timedelta(hours=idx),
                "label": label,
                "prob_long": prob,
                "forward_return": forward_return,
                "tb_return": max(forward_return, 0.0),
            }
            for idx, (label, prob, forward_return) in enumerate(zip(labels, probs, returns))
        ]

    predictions = pd.DataFrame(
        rows(0, "val", [0.10, 0.20, 0.80, 0.90], [-0.02, -0.01, 0.01, 0.02])
        + rows(0, "test", [0.10, 0.20, 0.80, 0.90], [-0.02, -0.01, 0.01, 0.02])
        + rows(1, "val", [0.90, 0.80, 0.20, 0.10], [-0.02, -0.01, 0.01, 0.02])
        + rows(1, "test", [0.90, 0.80, 0.20, 0.10], [-0.02, -0.01, 0.01, 0.02])
        + rows(2, "val", [0.15, 0.25, 0.75, 0.85], [-0.02, -0.01, 0.01, 0.02])
        + rows(2, "test", [0.15, 0.25, 0.75, 0.85], [-0.02, -0.01, 0.01, 0.02])
    )
    entry = {
        "profile": "control",
        "fold_scope": "full",
        "predictions": predictions,
        "diagnostics": {
            "fold_metrics": pd.DataFrame(
                [
                    {"fold": 0, "rank_ic": 0.20},
                    {"fold": 1, "rank_ic": -0.20},
                    {"fold": 2, "rank_ic": 0.10},
                ]
            ),
            "threshold_metrics": pd.DataFrame(
                [
                    {"fold": 0, "constrained_threshold": 0.5, "test_f1_at_constrained_threshold": 0.80},
                    {"fold": 1, "constrained_threshold": 0.5, "test_f1_at_constrained_threshold": 0.20},
                    {"fold": 2, "constrained_threshold": 0.5, "test_f1_at_constrained_threshold": 0.70},
                ]
            ),
            "score_band_by_fold": pd.DataFrame(
                [
                    {"fold": 0, "band": "top_10", "lift_vs_base": 1.20, "mean_forward_return": 0.002},
                    {"fold": 1, "band": "top_10", "lift_vs_base": 0.80, "mean_forward_return": -0.002},
                    {"fold": 2, "band": "top_10", "lift_vs_base": 1.10, "mean_forward_return": 0.001},
                ]
            ),
        },
    }

    detail = _fold_reliability_gate_frame([entry], config)
    summary = _fold_reliability_gate_summary_frame(detail, config)

    assert set(detail.loc[detail["gate_passed"], "fold"]) == {0, 2}
    row = summary.iloc[0]
    assert int(row["accepted_fold_count"]) == 2
    assert bool(row["gate_passed_cv"]) is True
    assert row["rejected_negative_fold_capture_rate"] == 1.0
    assert row["accepted_rank_ic_std_delta"] < 0
    assert row["next_action"] == "pre_register_reliability_gate_for_future_oos_review"


def test_regime_threshold_policy_uses_validation_regime_thresholds() -> None:
    config = {
        "validation": {
            "min_long_f1": 0.45,
            "bad_fold_ic_threshold": -0.08,
            "threshold_checks": {"max_pred_long_rate": 0.70, "min_precision": 0.30},
            "regime_threshold_policy": {
                "enabled": True,
                "min_regime_val_rows": 1,
                "min_regime_test_rows": 1,
                "min_regime_val_longs": 1,
                "max_pred_long_rate": 0.70,
                "min_precision": 0.30,
                "min_f1_delta_vs_official": 0.01,
                "min_policy_pass_fold_rate": 0.50,
                "min_positive_forward_return_fold_rate": 0.50,
            },
        }
    }

    def rows(split: str) -> list[dict[str, object]]:
        base = pd.Timestamp("2024-05-01", tz="UTC")
        values = [
            (0, 0, 0.10, -0.02),
            (0, 0, 0.20, -0.01),
            (0, 1, 0.35, 0.01),
            (0, 1, 0.40, 0.02),
            (1, 0, 0.55, -0.02),
            (1, 0, 0.58, -0.01),
            (1, 1, 0.85, 0.01),
            (1, 1, 0.90, 0.02),
        ]
        out = []
        for idx, (regime, label, prob, forward_return) in enumerate(values):
            out.append(
                {
                    "fold": 0,
                    "split": split,
                    "timestamp": base + pd.Timedelta(hours=idx + (0 if split == "val" else 12)),
                    "label": label,
                    "prob_long": prob,
                    "forward_return": forward_return,
                    "tb_return": max(forward_return, 0.0),
                    "regime_prob_0": 1.0 if regime == 0 else 0.0,
                    "regime_prob_1": 1.0 if regime == 1 else 0.0,
                }
            )
        return out

    entry = {
        "profile": "control",
        "fold_scope": "full",
        "predictions": pd.DataFrame(rows("val") + rows("test")),
        "diagnostics": {
            "fold_metrics": pd.DataFrame([{"fold": 0, "rank_ic": 0.20}]),
            "threshold_metrics": pd.DataFrame(
                [
                        {
                            "fold": 0,
                            "official_threshold": 0.50,
                            "constrained_threshold": 0.50,
                            "test_f1_at_constrained_threshold": 0.50,
                            "test_precision_at_constrained_threshold": 0.50,
                            "test_recall_at_constrained_threshold": 0.50,
                            "test_pred_long_rate_at_constrained_threshold": 0.50,
                            "test_f1_at_official_threshold": 0.50,
                            "test_precision_at_official_threshold": 0.50,
                            "test_recall_at_official_threshold": 0.50,
                        "test_pred_long_rate_at_official_threshold": 0.50,
                    }
                ]
            ),
        },
    }

    by_fold, summary = _regime_threshold_policy_frames([entry], config)
    regime_forensics, regime_summary = _regime_stability_frames([entry], config)

    assert by_fold.iloc[0]["regime_threshold_count"] == 2
    assert by_fold.iloc[0]["test_f1"] == 1.0
    assert by_fold.iloc[0]["f1_delta_vs_official"] == 0.5
    assert bool(summary.iloc[0]["reviewable"]) is True
    assert summary.iloc[0]["next_action"] == "pre_register_regime_threshold_policy_for_future_oos_review"
    assert not regime_forensics.empty
    assert set(regime_summary["regime"]) == {0, 1}


def test_experiment_selection_flags_selected_full_profile_without_output() -> None:
    settings = {
        "control_profile": "control",
        "candidate_profiles": [],
        "always_full_profiles": ["control", "missing_full"],
        "seed_audit": {"enabled": True, "profiles": ["control"], "seeds": [42]},
    }
    selection = _experiment_selection_frame(settings)
    comparison = pd.DataFrame(
        [
            {"profile": "control", "fold_scope": "triage"},
            {"profile": "control", "fold_scope": "full"},
        ]
    )

    missing = _missing_selected_profiles(selection, comparison)

    assert missing.to_dict(orient="records") == [
        {
            "profile": "missing_full",
            "role": "always_full_profile",
            "expected_fold_scope": "full",
            "reason": "missing_selected_profile_output",
        }
    ]


def test_holdout_reservation_frame_records_selection_and_holdout_window() -> None:
    settings = {
        "holdout": {
            "enabled": True,
            "holdout_bars": 4320,
            "selection_rows": 32000,
            "holdout_rows": 4320,
            "selection_data_start": "2022-01-01 00:00:00+00:00",
            "selection_data_end": "2025-11-01 00:00:00+00:00",
            "holdout_data_start": "2025-11-01 01:00:00+00:00",
            "holdout_data_end": "2026-05-01 00:00:00+00:00",
            "latest_available_data_end": "2026-05-03 00:00:00+00:00",
            "holdout_path": "/content/drive/MyDrive/yeniBot/data/processed/holdout_1h.parquet",
            "policy": "profile_selection_only_before_holdout",
        }
    }

    frame = _holdout_reservation_frame(settings)

    assert bool(frame.loc[0, "enabled"]) is True
    assert frame.loc[0, "holdout_bars"] == 4320
    assert frame.loc[0, "selection_data_end"] < frame.loc[0, "holdout_data_start"]
    assert frame.loc[0, "latest_available_data_end"] == "2026-05-03 00:00:00+00:00"


def test_prepare_training_holdout_split_freezes_failed_clean_holdout_anchor(tmp_path) -> None:
    timestamps = pd.date_range("2026-01-01", periods=140, freq="1h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": np.linspace(100.0, 120.0, len(timestamps)),
            "label": (np.arange(len(timestamps)) % 3 == 0).astype(int),
            "fwd_return_10h": 0.01,
        }
    )
    anchor = timestamps[110]
    config = {
        "experiments": {
            "holdout": {
                "enabled": True,
                "holdout_bars": 24,
                "policy": "profile_selection_only_before_holdout",
            },
            "policy_review": {
                "status": "failed_clean_holdout_review",
                "future_oos_monitor": {
                    "enabled": True,
                    "anchor_run_id": "failed_anchor",
                    "anchor_data_end": str(anchor),
                    "min_new_bars": 72,
                    "preferred_new_bars": 144,
                    "allow_holdout_roll_forward": False,
                },
            },
        }
    }

    selection, holdout, meta = prepare_training_holdout_split(
        frame,
        config,
        holdout_path=tmp_path / "holdout_1h.parquet",
    )

    assert meta["split_mode"] == "frozen_anchor_holdout"
    assert meta["holdout_roll_forward_locked"] is True
    assert meta["future_oos_ready"] is False
    assert meta["new_bars_since_anchor"] == 29
    assert meta["latest_available_data_end"] == str(timestamps.max())
    assert meta["unused_rows_after_anchor"] == 29
    assert pd.to_datetime(holdout["timestamp"], utc=True).max() == anchor
    assert pd.to_datetime(selection["timestamp"], utc=True).max() < pd.to_datetime(holdout["timestamp"], utc=True).min()
    assert pd.to_datetime(selection["timestamp"], utc=True).max() <= anchor
    assert (tmp_path / "holdout_1h.parquet").exists()


def test_holdout_boundary_audit_rejects_entries_inside_reserved_holdout() -> None:
    settings = {
        "holdout": {
            "enabled": True,
            "holdout_data_start": "2025-11-01 01:00:00+00:00",
        }
    }
    entries = [
        {
            "profile": "control",
            "fold_scope": "full",
            "diagnostics": {
                "row": {
                    "data_start": "2024-01-01 00:00:00+00:00",
                    "data_end": "2025-10-31 23:00:00+00:00",
                }
            },
        },
        {
            "profile": "old_run",
            "fold_scope": "full",
            "diagnostics": {
                "row": {
                    "data_start": "2024-01-01 00:00:00+00:00",
                    "data_end": "2026-04-25 09:00:00+00:00",
                }
            },
        },
    ]

    audit = _holdout_boundary_audit_frame(entries, settings)

    assert bool(audit.loc[audit["profile"].eq("control"), "passed"].iloc[0]) is True
    rejected = audit.loc[audit["profile"].eq("old_run")].iloc[0]
    assert bool(rejected["passed"]) is False
    assert rejected["reason"] == "entry_data_end_reaches_reserved_holdout"


def test_frozen_policy_robustness_uses_configured_score_band() -> None:
    timestamps = pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC")
    labels = np.zeros(100, dtype=int)
    labels[-10:] = 1
    predictions = pd.DataFrame(
        {
            "timestamp": timestamps,
            "split": "test",
            "fold": 0,
            "label": labels,
            "prob_long": np.linspace(0.0, 1.0, 100),
            "forward_return": np.linspace(-0.01, 0.02, 100),
        }
    )
    config = {
        "validation": {
            "score_lift_bins": 10,
            "calibration_bins": 10,
            "threshold_checks": {"min_precision": 0.30, "max_pred_long_rate": 0.70},
            "score_bands": [{"name": "top_10", "min_bin": 9, "max_bin": 9}],
        },
        "experiments": {
            "policy_review": {
                "enabled": True,
                "frozen_candidate": "blend_prob_mean_test",
                "policy_type": "score_band",
                "policy_name": "top_10",
                "robustness": {
                    "enabled": True,
                    "min_rows": 80,
                    "min_selected_rows": 5,
                    "min_rank_ic": 0.0,
                    "min_lift_vs_base": 1.0,
                    "min_forward_return": 0.0,
                    "windows": [
                        {
                            "name": "synthetic",
                            "start": "2024-01-01 00:00:00+00:00",
                            "end": "2024-01-05 03:00:00+00:00",
                        }
                    ],
                },
            }
        },
    }
    entries = [{"profile": "blend_prob_mean_test", "predictions": predictions}]

    frame = _frozen_policy_robustness_frame(entries, config)

    assert frame.loc[0, "window"] == "synthetic"
    assert frame.loc[0, "policy_name"] == "top_10"
    assert frame.loc[0, "selected_rows"] == 10
    assert frame.loc[0, "policy_lift_vs_base"] > 1.0
    assert bool(frame.loc[0, "window_pass"]) is True


def test_experiment_preflight_skips_intrahour_candidates_until_features_exist() -> None:
    config = {
        "features": {
            "active_profile": "control",
            "profiles": {
                "control": {"include_patterns": ["base_feature"], "exclude_patterns": []},
                "same": {"include_patterns": ["base_feature"], "exclude_patterns": []},
                "intrahour": {"inherit": "control", "include_patterns": ["ih15_*"], "exclude_patterns": []},
            },
        },
        "experiments": {
            "control_profile": "control",
            "candidate_profiles": ["same", "intrahour"],
            "always_full_profiles": ["control", "intrahour"],
        },
    }
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
            "base_feature": [1.0, 2.0, 3.0, 4.0],
        }
    )

    settings = _preflight_experiment_profiles(experiment_settings(config), frame, config)

    assert settings["profiles"] == ["control"]
    assert settings["candidate_profiles"] == []
    assert settings["always_full_profiles"] == ["control"]
    assert settings["skipped_profiles"] == [
        {"profile": "same", "role": "candidate_profile", "skip_reason": "duplicate_feature_signature:control"},
        {
            "profile": "intrahour",
            "role": "candidate_profile",
            "skip_reason": "missing_intrahour_features_rerun_01_02_03:ih15_*",
        },
        {
            "profile": "intrahour",
            "role": "always_full_profile",
            "skip_reason": "missing_intrahour_features_rerun_01_02_03",
        },
    ]


def test_experiment_preflight_skips_partial_intrahour_feature_sets() -> None:
    config = {
        "features": {
            "active_profile": "control",
            "profiles": {
                "control": {"include_patterns": ["base_feature"], "exclude_patterns": []},
                "intrahour": {
                    "inherit": "control",
                    "include_patterns": ["ih15_coverage", "ih15_absorption_imbalance_stable_rank"],
                    "exclude_patterns": [],
                },
            },
        },
        "experiments": {
            "control_profile": "control",
            "candidate_profiles": ["intrahour"],
            "always_full_profiles": ["control"],
        },
    }
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
            "base_feature": [1.0, 2.0, 3.0, 4.0],
            "ih15_coverage": [1.0, 1.0, 1.0, 1.0],
        }
    )

    settings = _preflight_experiment_profiles(experiment_settings(config), frame, config)

    assert settings["profiles"] == ["control"]
    assert settings["skipped_profiles"] == [
        {
            "profile": "intrahour",
            "role": "candidate_profile",
            "skip_reason": "missing_intrahour_features_rerun_01_02_03:ih15_absorption_imbalance_stable_rank",
        }
    ]


def test_experiment_preflight_skips_futures_context_profiles_until_features_exist() -> None:
    config = {
        "features": {
            "active_profile": "control",
            "profiles": {
                "control": {"include_patterns": ["base_feature"], "exclude_patterns": []},
                "futures_context": {
                    "inherit": "control",
                    "include_patterns": ["fut_*_stable_rank", "fut_metrics_missing"],
                    "exclude_patterns": [],
                },
            },
        },
        "experiments": {
            "control_profile": "control",
            "candidate_profiles": ["futures_context"],
            "always_full_profiles": ["control", "futures_context"],
            "seed_audit": {"enabled": True, "profiles": ["control", "futures_context"], "seeds": [42]},
        },
    }
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
            "base_feature": [1.0, 2.0, 3.0, 4.0],
        }
    )

    settings = _preflight_experiment_profiles(experiment_settings(config), frame, config)

    assert settings["profiles"] == ["control"]
    assert settings["candidate_profiles"] == []
    assert settings["always_full_profiles"] == ["control"]
    assert settings["seed_audit"]["profiles"] == ["control"]
    assert settings["skipped_profiles"] == [
        {
            "profile": "futures_context",
            "role": "candidate_profile",
            "skip_reason": "missing_futures_context_features_rerun_01_02_03",
        },
        {
            "profile": "futures_context",
            "role": "always_full_profile",
            "skip_reason": "missing_futures_context_features_rerun_01_02_03",
        },
        {
            "profile": "futures_context",
            "role": "seed_audit_profile",
            "skip_reason": "missing_futures_context_features_rerun_01_02_03",
        },
    ]


def test_auto_full_profiles_keeps_control_and_promotes_best_triage_candidates() -> None:
    settings = {
        "control_profile": "control",
        "always_full_profiles": ["control", "champion"],
        "max_auto_full_candidates": 1,
    }
    triage_rows = [
        {"profile": "control", "promotable": False, "mean_rank_ic": 0.03, "top_10_lift_global": 1.0},
        {"profile": "weak", "promotable": False, "mean_rank_ic": 0.09, "top_10_lift_global": 1.2},
        {"profile": "candidate_a", "promotable": True, "mean_rank_ic": 0.05, "top_10_lift_global": 1.4},
        {"profile": "candidate_b", "promotable": True, "mean_rank_ic": 0.07, "top_10_lift_global": 1.1},
    ]

    assert _auto_full_profiles(settings, triage_rows) == ["control", "champion", "candidate_b"]


def test_repo_experiment_profiles_keep_default_baseline_and_candidate_boundaries() -> None:
    config = load_config("config.yaml")
    assert config["features"]["active_profile"] == "baseline_plus_4h_bounded_whale_no_4h_tier1"
    assert config["model"]["dropout"] == 0.2
    assert config["training"]["early_stop_metric"] == "rank_ic"
    assert config["training"]["rank_ic_smoothing_epochs"] == 5
    assert config["training"]["optimizer"]["weight_decay"] == 0.0001
    assert config["walk_forward"]["val_bars"] == 1080
    assert (
        config["experiments"]["control_profile"]
        == "baseline_plus_4h_bounded_whale_no_4h_tier1_no_4h_pure_volatility_no_1h_pure_volatility"
    )
    assert config["experiments"]["full_cv_profiles"] == "auto"
    assert config["experiments"]["always_full_profiles"] == [
        "baseline_plus_4h_bounded_whale_no_4h_tier1_no_4h_pure_volatility_no_1h_pure_volatility",
        "baseline_stable_score_margin_loss",
        "baseline_no_4h_tier1_4h_large_trade_pressure_long",
    ]
    assert config["experiments"]["max_auto_full_candidates"] == 2
    assert config["experiments"]["candidate_profiles"] == []
    assert config["experiments"]["profile_blends"]["include_auto_rank_mean"] is False
    weighted_blends = config["experiments"]["profile_blends"]["weighted"]
    assert [item["name"] for item in weighted_blends] == ["control_long_pressure_65_35"]
    assert weighted_blends[0]["weights"] == [0.65, 0.35]
    assert config["experiments"]["holdout"]["enabled"] is True
    assert config["experiments"]["holdout"]["holdout_bars"] == 4320
    assert config["experiments"]["holdout"]["holdout_filename"] == "holdout_1h.parquet"
    assert config["experiments"]["policy_review"]["frozen_candidate"] == "blend_prob_mean_953a4ee825"
    assert config["experiments"]["policy_review"]["policy_type"] == "score_band"
    assert config["experiments"]["policy_review"]["policy_name"] == "top_10"
    assert config["experiments"]["policy_review"]["status"] == "failed_clean_holdout_review"
    assert config["experiments"]["policy_review"]["threshold_deployment_allowed"] is False
    assert config["experiments"]["policy_review"]["future_oos_candidates"] == [
        "blend_control_long_pressure_65_35",
        "baseline_stable_score_margin_loss",
    ]
    assert config["experiments"]["policy_review"]["future_oos_monitor"]["enabled"] is True
    assert config["experiments"]["policy_review"]["future_oos_monitor"]["anchor_run_id"] == "20260522_135424"
    assert config["experiments"]["policy_review"]["future_oos_monitor"]["anchor_data_end"] == "2026-05-13 08:00:00+00:00"
    assert config["experiments"]["policy_review"]["future_oos_monitor"]["min_new_bars"] == 720
    assert config["experiments"]["policy_review"]["future_oos_monitor"]["preferred_new_bars"] == 2160
    assert config["experiments"]["policy_review"]["future_oos_monitor"]["allow_holdout_roll_forward"] is False
    robustness = config["experiments"]["policy_review"]["robustness"]
    assert robustness["enabled"] is True
    assert robustness["min_rows"] == 720
    assert [item["name"] for item in robustness["windows"]] == [
        "post_ftx_rebuild",
        "pre_etf_recovery",
        "etf_to_halving_cycle",
        "late_2024_to_q1_2025",
        "pre_holdout_recent",
    ]
    assert config["experiments"]["seed_audit"]["enabled"] is True
    assert config["experiments"]["seed_audit"]["profiles"] == [
        "baseline_plus_4h_bounded_whale_no_4h_tier1_no_4h_pure_volatility_no_1h_pure_volatility",
    ]
    assert config["experiments"]["seed_audit"]["seeds"] == [42, 43, 44]
    notes = config["experiments"]["experiment_memory"]["reference_notes"]
    assert "weak as a standalone profile" in notes["baseline_no_4h_tier1_4h_large_trade_pressure_long"]
    assert "Retired frozen review selection" in notes["blend_prob_mean_953a4ee825"]
    assert "future out-of-sample" in notes["blend_control_long_pressure_65_35"]
    assert "pairwise label-margin loss" in notes["baseline_stable_score_margin_loss"]
    assert "Split into narrower" in notes["baseline_control_plus_futures_context"]
    assert config["features"]["profiles"]["baseline_stable_score_margin_loss"]["config_overrides"]["training"]["loss"] == {
        "label_margin_weight": 0.05,
        "label_margin": 0.25,
    }
    assert "failed CV stability" in notes["baseline_control_plus_futures_oi_change_context"]
    assert "failed CV stability" in notes["baseline_control_plus_futures_positioning_context"]
    assert "did not clear" in notes["baseline_control_plus_futures_funding_context"]
    assert "failed to improve control" in notes["baseline_stable_plus_4h_taker_mean12_ltr_context"]
    reliability = config["validation"]["fold_reliability_gates"]
    assert reliability["enabled"] is True
    assert reliability["min_accepted_folds"] == 12
    assert {gate["name"] for gate in reliability["gates"]} == {
        "val_rank_ic_positive",
        "val_score_gap_positive",
        "val_rank_ic_and_score_gap_positive",
        "val_rank_ic_score_gap_and_ks",
    }
    regime_threshold = config["validation"]["regime_threshold_policy"]
    assert regime_threshold["enabled"] is True
    assert regime_threshold["min_regime_val_rows"] == 80
    assert regime_threshold["max_pred_long_rate"] == 0.70
    assert {0, 2, 4, 8, 17, 21, 32, 39}.issubset(set(config["experiments"]["triage_fold_ids"]))
    columns = [
        "4h_large_trade_ratio",
        "4h_vpt_zscore",
        "4h_vol_per_trade_log_zscore",
        "4h_cvd_pressure_3",
        "4h_cvd_pressure_3_stable_rank",
        "4h_cvd_pressure_24_stable_zscore",
        "4h_signed_large_trade_pressure_stable_zscore",
        "4h_signed_large_trade_pressure_stable_rank",
        "4h_large_trade_pressure_3_stable_zscore",
        "4h_large_trade_pressure_3_stable_rank",
        "4h_large_trade_pressure_6_stable_zscore",
        "4h_large_trade_pressure_6_stable_rank",
        "4h_large_trade_pressure_12_stable_zscore",
        "4h_large_trade_pressure_12_stable_rank",
        "4h_large_trade_pressure_12_stable_tanh",
        "4h_large_trade_pressure_24_stable_zscore",
        "4h_large_trade_pressure_24_stable_rank",
        "4h_large_trade_pressure_24_stable_tanh",
        "4h_large_trade_pressure_24_minus_12_stable_zscore",
        "4h_large_trade_pressure_24_minus_12_stable_rank",
        "4h_large_trade_pressure_24_minus_12_stable_tanh",
        "4h_signed_large_trade_pressure_stable_tanh",
        "4h_gk_vol_14_stable_rank",
        "4h_gk_vol_14_stable_zscore",
        "4h_realized_vol_14_stable_rank",
        "4h_realized_vol_14_stable_zscore",
        "4h_atr_14_pct_stable_rank",
        "4h_atr_14_pct_stable_zscore",
        "4h_adx_14_stable_rank",
        "4h_vwap_dist_atr_stable_zscore",
        "4h_volume_log_zscore_stable_rank",
        "4h_gk_vol_14",
        "4h_realized_vol_14",
        "4h_atr_14_pct",
        "4h_adx_14",
        "4h_log_return",
        "4h_close_denoised_log_return",
        "4h_volume_log_zscore",
        "4h_volume_denoised_log_zscore",
        "4h_vwap_dist_atr",
        "4h_taker_imbalance",
        "4h_taker_buy_ratio_zscore",
        "4h_taker_buy_ratio_delta",
        "4h_taker_imbalance_slope",
        "4h_taker_imbalance_mean_12",
        "4h_taker_imbalance_mean_12_stable_rank",
        "4h_taker_imbalance_mean_24",
        "4h_large_trade_ratio_stable_rank",
        "4h_taker_mean12_x_ltr_rank_signed",
        "4h_taker_mean12_x_ltr_rank_high",
        "4h_taker_mean12_x_ltr_rank_low",
        "4h_whale_buy_flag",
        "4h_whale_sell_flag",
        "4h_taker_imbalance_x_rv14_rank_signed",
        "4h_taker_imbalance_slope_x_rv14_rank_high",
        "4h_signed_ltp_x_rv14_rank_low",
        "4h_ltp12_rank_x_rv14_rank_signed",
        "4h_ltp24_rank_x_rv14_rank_high",
        "4h_taker_imbalance_x_gk14_rank_signed",
        "4h_taker_imbalance_slope_x_gk14_rank_high",
        "4h_signed_ltp_x_gk14_rank_low",
        "4h_ltp12_rank_x_gk14_rank_signed",
        "4h_ltp24_rank_x_gk14_rank_high",
        "4h_taker_imbalance_x_atr14_rank_signed",
        "4h_taker_imbalance_slope_x_atr14_rank_high",
        "4h_signed_ltp_x_atr14_rank_low",
        "4h_ltp12_rank_x_atr14_rank_signed",
        "4h_ltp24_rank_x_atr14_rank_high",
        "taker_imbalance_x_rv14_rank_signed",
        "taker_imbalance_slope_x_rv14_rank_high",
        "signed_ltp_x_rv14_rank_low",
        "ltp12_rank_x_rv14_rank_signed",
        "ltp24_rank_x_rv14_rank_high",
        "taker_imbalance_x_gk14_rank_signed",
        "taker_imbalance_slope_x_gk14_rank_high",
        "signed_ltp_x_gk14_rank_low",
        "ltp12_rank_x_gk14_rank_signed",
        "ltp24_rank_x_gk14_rank_high",
        "taker_imbalance_x_atr14_rank_signed",
        "taker_imbalance_slope_x_atr14_rank_high",
        "signed_ltp_x_atr14_rank_low",
        "ltp12_rank_x_atr14_rank_signed",
        "ltp24_rank_x_atr14_rank_high",
        "gk_vol_14",
        "realized_vol_14",
        "atr_14_pct",
        "volume_log_zscore",
        "volume_denoised_log_zscore",
        "taker_buy_ratio",
        "cvd_cumulative_rate_norm",
        "ih15_coverage",
        "ih15_taker_imbalance_last",
        "ih15_taker_imbalance_late_minus_early",
        "ih15_taker_imbalance_slope",
        "ih15_buy_ratio_range",
        "ih15_cvd_pressure_norm",
        "ih15_cvd_pressure_late_minus_early_norm",
        "ih15_cvd_slope_norm",
        "ih15_volume_share_last",
        "ih15_trade_share_last",
        "ih15_vpt_last_vs_hour",
        "ih15_vpt_late_vs_early",
        "ih15_large_trade_concentration",
        "ih15_buy_volume_share_last",
        "ih15_aggressive_buy_burst",
        "ih15_aggressive_sell_burst",
        "ih15_missing",
        "ih15_price_move_norm",
        "ih15_flow_price_alignment",
        "ih15_buy_absorption",
        "ih15_sell_absorption",
        "ih15_absorption_imbalance",
        "ih15_late_buy_absorption",
        "ih15_late_sell_absorption",
        "ih15_late_absorption_imbalance",
        "ih15_late_flow_reversal",
        "ih15_late_price_reversal",
        "ih15_late_flow_price_alignment",
        "ih15_cvd_pressure_norm_stable_rank",
        "ih15_cvd_pressure_norm_stable_zscore",
        "ih15_cvd_slope_norm_stable_rank",
        "ih15_flow_price_alignment_stable_rank",
        "ih15_flow_price_alignment_stable_tanh",
        "ih15_buy_absorption_stable_rank",
        "ih15_sell_absorption_stable_rank",
        "ih15_absorption_imbalance_stable_rank",
        "ih15_absorption_imbalance_stable_tanh",
        "ih15_late_buy_absorption_stable_rank",
        "ih15_late_sell_absorption_stable_rank",
        "ih15_late_absorption_imbalance_stable_rank",
        "ih15_late_absorption_imbalance_stable_tanh",
        "ih15_late_flow_reversal_stable_rank",
        "ih15_late_flow_reversal_stable_tanh",
        "ih15_late_price_reversal_stable_rank",
        "ih15_late_price_reversal_stable_tanh",
        "ih15_late_flow_price_alignment_stable_rank",
        "ih15_late_flow_price_alignment_stable_tanh",
        "ih15_vpt_last_vs_hour_stable_rank",
        "ih15_vpt_last_vs_hour_stable_zscore",
        "ih15_large_trade_concentration_stable_rank",
        "ih15_large_trade_concentration_stable_zscore",
        "ih15_aggressive_buy_burst_stable_rank",
        "ih15_aggressive_sell_burst_stable_rank",
        "fut_oi_log_return",
        "fut_oi_change_288_stable_rank",
        "fut_oi_change_288_stable_zscore",
        "fut_oi_change_288_stable_tanh",
        "fut_oi_value_change_288_stable_rank",
        "fut_toptrader_count_long_short_log_ratio_stable_rank",
        "fut_toptrader_count_long_short_log_ratio_stable_zscore",
        "fut_global_long_short_log_ratio_stable_rank",
        "fut_taker_long_short_vol_log_ratio_stable_zscore",
        "fut_funding_rate",
        "fut_funding_rate_stable_rank",
        "fut_funding_sum_12_stable_zscore",
        "fut_funding_mean_12_stable_tanh",
        "fut_metrics_missing",
        "fut_funding_missing",
        "fut_toptrader_sum_long_short_log_ratio_stable_tanh",
    ]

    pruned = profile_config(config, "baseline_no_4h_tier1_pruned_whale")
    assert "4h_large_trade_ratio" not in filter_feature_columns(columns, pruned)
    assert "4h_vpt_zscore" in filter_feature_columns(columns, pruned)

    base_no_slow = profile_config(config, "baseline_plus_4h_bounded_whale_no_4h_tier1_no_slow_4h_bounded_flow")
    base_no_slow_columns = filter_feature_columns(columns, base_no_slow)
    assert "4h_taker_imbalance_mean_12" not in base_no_slow_columns
    assert "4h_taker_imbalance_mean_24" not in base_no_slow_columns
    assert "4h_taker_imbalance" in base_no_slow_columns

    base_no_bounded = profile_config(config, "baseline_plus_4h_bounded_whale_no_4h_tier1_no_4h_bounded_flow")
    base_no_bounded_columns = filter_feature_columns(columns, base_no_bounded)
    assert "4h_taker_imbalance" not in base_no_bounded_columns
    assert "4h_taker_imbalance_mean_24" not in base_no_bounded_columns
    assert "4h_large_trade_ratio" in base_no_bounded_columns

    base_no_ratio = profile_config(config, "baseline_plus_4h_bounded_whale_no_4h_tier1_no_4h_large_trade_ratio")
    assert "4h_large_trade_ratio" not in filter_feature_columns(columns, base_no_ratio)

    base_no_whale_zscores = profile_config(config, "baseline_plus_4h_bounded_whale_no_4h_tier1_no_4h_whale_zscores")
    base_no_whale_zscore_columns = filter_feature_columns(columns, base_no_whale_zscores)
    assert "4h_vpt_zscore" not in base_no_whale_zscore_columns
    assert "4h_vol_per_trade_log_zscore" not in base_no_whale_zscore_columns
    assert "4h_large_trade_ratio" in base_no_whale_zscore_columns

    base_no_volatility = profile_config(config, "baseline_plus_4h_bounded_whale_no_4h_tier1_no_4h_pure_volatility")
    base_no_volatility_columns = filter_feature_columns(columns, base_no_volatility)
    assert "4h_gk_vol_14" not in base_no_volatility_columns
    assert "4h_atr_14_pct" not in base_no_volatility_columns

    base_no_vol_no_slow = profile_config(
        config,
        "baseline_plus_4h_bounded_whale_no_4h_tier1_no_4h_pure_volatility_no_slow_4h_bounded_flow",
    )
    base_no_vol_no_slow_columns = filter_feature_columns(columns, base_no_vol_no_slow)
    assert "4h_gk_vol_14" not in base_no_vol_no_slow_columns
    assert "4h_taker_imbalance_mean_12" not in base_no_vol_no_slow_columns
    assert "4h_taker_imbalance" in base_no_vol_no_slow_columns

    base_no_vol_no_bounded = profile_config(
        config,
        "baseline_plus_4h_bounded_whale_no_4h_tier1_no_4h_pure_volatility_no_4h_bounded_flow",
    )
    base_no_vol_no_bounded_columns = filter_feature_columns(columns, base_no_vol_no_bounded)
    assert "4h_gk_vol_14" not in base_no_vol_no_bounded_columns
    assert "4h_taker_imbalance" not in base_no_vol_no_bounded_columns
    assert "4h_taker_imbalance_mean_24" not in base_no_vol_no_bounded_columns

    base_no_vol_no_1h_vol = profile_config(
        config,
        "baseline_plus_4h_bounded_whale_no_4h_tier1_no_4h_pure_volatility_no_1h_pure_volatility",
    )
    base_no_vol_no_1h_vol_columns = filter_feature_columns(columns, base_no_vol_no_1h_vol)
    assert "4h_gk_vol_14" not in base_no_vol_no_1h_vol_columns
    assert "gk_vol_14" not in base_no_vol_no_1h_vol_columns
    assert "4h_taker_imbalance_mean_24" in base_no_vol_no_1h_vol_columns

    futures_context = profile_config(config, "baseline_control_plus_futures_context")
    futures_columns = filter_feature_columns(columns, futures_context)
    assert "fut_oi_log_return" not in futures_columns
    assert "fut_funding_rate" not in futures_columns
    assert "fut_oi_change_288_stable_rank" in futures_columns
    assert "fut_oi_change_288_stable_tanh" in futures_columns
    assert "fut_toptrader_count_long_short_log_ratio_stable_zscore" in futures_columns
    assert "fut_funding_sum_12_stable_zscore" in futures_columns
    assert "fut_funding_mean_12_stable_tanh" in futures_columns
    assert "fut_metrics_missing" in futures_columns
    assert "4h_gk_vol_14" not in futures_columns
    assert "realized_vol_14" not in futures_columns

    futures_oi = profile_config(config, "baseline_control_plus_futures_oi_change_context")
    futures_oi_columns = filter_feature_columns(columns, futures_oi)
    assert "fut_oi_change_288_stable_rank" in futures_oi_columns
    assert "fut_oi_change_288_stable_tanh" in futures_oi_columns
    assert "fut_oi_log_return" not in futures_oi_columns
    assert "fut_oi_log_return_stable_rank" not in futures_oi_columns
    assert "fut_funding_sum_12_stable_zscore" not in futures_oi_columns
    assert "fut_toptrader_count_long_short_log_ratio_stable_zscore" not in futures_oi_columns

    futures_positioning = profile_config(config, "baseline_control_plus_futures_positioning_context")
    futures_positioning_columns = filter_feature_columns(columns, futures_positioning)
    assert "fut_toptrader_count_long_short_log_ratio_stable_zscore" in futures_positioning_columns
    assert "fut_toptrader_sum_long_short_log_ratio_stable_tanh" in futures_positioning_columns
    assert "fut_taker_long_short_vol_log_ratio_stable_zscore" in futures_positioning_columns
    assert "fut_oi_change_288_stable_rank" not in futures_positioning_columns
    assert "fut_funding_sum_12_stable_zscore" not in futures_positioning_columns

    futures_funding = profile_config(config, "baseline_control_plus_futures_funding_context")
    futures_funding_columns = filter_feature_columns(columns, futures_funding)
    assert "fut_funding_sum_12_stable_zscore" in futures_funding_columns
    assert "fut_funding_mean_12_stable_tanh" in futures_funding_columns
    assert "fut_metrics_missing" not in futures_funding_columns
    assert "fut_oi_change_288_stable_rank" not in futures_funding_columns
    assert "fut_toptrader_count_long_short_log_ratio_stable_zscore" not in futures_funding_columns

    stable_no_cvd = profile_config(config, "baseline_stable_no_1h_cvd_rate")
    stable_no_cvd_columns = filter_feature_columns(columns, stable_no_cvd)
    assert "cvd_cumulative_rate_norm" not in stable_no_cvd_columns
    assert "4h_taker_imbalance_mean_24" in stable_no_cvd_columns

    stable_no_slow_flow = profile_config(config, "baseline_stable_no_slow_4h_bounded_flow")
    stable_no_slow_flow_columns = filter_feature_columns(columns, stable_no_slow_flow)
    assert "4h_taker_imbalance_mean_12" not in stable_no_slow_flow_columns
    assert "4h_taker_imbalance_mean_24" not in stable_no_slow_flow_columns
    assert "cvd_cumulative_rate_norm" in stable_no_slow_flow_columns

    stable_no_mean12 = profile_config(config, "baseline_stable_no_4h_taker_mean12")
    stable_no_mean12_columns = filter_feature_columns(columns, stable_no_mean12)
    assert "4h_taker_imbalance_mean_12" not in stable_no_mean12_columns
    assert "4h_taker_imbalance_mean_24" in stable_no_mean12_columns
    assert "4h_large_trade_ratio" in stable_no_mean12_columns
    assert "gk_vol_14" not in stable_no_mean12_columns

    stable_no_4h_ratio = profile_config(config, "baseline_stable_no_4h_large_trade_ratio")
    stable_no_4h_ratio_columns = filter_feature_columns(columns, stable_no_4h_ratio)
    assert "4h_large_trade_ratio" not in stable_no_4h_ratio_columns
    assert "4h_taker_imbalance_mean_12" in stable_no_4h_ratio_columns

    stable_no_mean12_no_ratio = profile_config(
        config,
        "baseline_stable_no_4h_taker_mean12_no_4h_large_trade_ratio",
    )
    stable_no_mean12_no_ratio_columns = filter_feature_columns(columns, stable_no_mean12_no_ratio)
    assert "4h_taker_imbalance_mean_12" not in stable_no_mean12_no_ratio_columns
    assert "4h_taker_imbalance_mean_24" in stable_no_mean12_no_ratio_columns
    assert "4h_large_trade_ratio" not in stable_no_mean12_no_ratio_columns

    conditional_context = profile_config(config, "baseline_stable_plus_4h_taker_mean12_ltr_context")
    conditional_context_columns = filter_feature_columns(columns, conditional_context)
    assert "4h_taker_imbalance_mean_12" in conditional_context_columns
    assert "4h_large_trade_ratio" in conditional_context_columns
    assert "4h_taker_imbalance_mean_12_stable_rank" in conditional_context_columns
    assert "4h_large_trade_ratio_stable_rank" in conditional_context_columns
    assert "4h_taker_mean12_x_ltr_rank_signed" in conditional_context_columns
    assert "4h_taker_mean12_x_ltr_rank_low" in conditional_context_columns
    assert "4h_gk_vol_14" not in conditional_context_columns
    assert "4h_large_trade_pressure_12_stable_rank" not in conditional_context_columns

    conditional_no_raw_ltr = profile_config(config, "baseline_stable_plus_4h_taker_mean12_ltr_context_no_raw_ltr")
    conditional_no_raw_ltr_columns = filter_feature_columns(columns, conditional_no_raw_ltr)
    assert "4h_large_trade_ratio" not in conditional_no_raw_ltr_columns
    assert "4h_large_trade_ratio_stable_rank" in conditional_no_raw_ltr_columns
    assert "4h_taker_imbalance_mean_12" in conditional_no_raw_ltr_columns

    conditional_stable_only = profile_config(config, "baseline_stable_plus_4h_taker_mean12_ltr_context_stable_only")
    conditional_stable_only_columns = filter_feature_columns(columns, conditional_stable_only)
    assert "4h_taker_imbalance_mean_12" not in conditional_stable_only_columns
    assert "4h_large_trade_ratio" not in conditional_stable_only_columns
    assert "4h_taker_mean12_x_ltr_rank_high" in conditional_stable_only_columns

    stable_combined = profile_config(config, "baseline_stable_no_slow_4h_bounded_flow_no_1h_cvd_rate")
    stable_combined_columns = filter_feature_columns(columns, stable_combined)
    assert "cvd_cumulative_rate_norm" not in stable_combined_columns
    assert "4h_taker_imbalance_mean_24" not in stable_combined_columns

    stable_no_whale_zscores = profile_config(config, "baseline_stable_no_4h_whale_zscores")
    stable_no_whale_zscore_columns = filter_feature_columns(columns, stable_no_whale_zscores)
    assert "4h_vpt_zscore" not in stable_no_whale_zscore_columns
    assert "4h_vol_per_trade_log_zscore" not in stable_no_whale_zscore_columns
    assert "4h_large_trade_ratio" in stable_no_whale_zscore_columns

    stable_no_4h_volume = profile_config(config, "baseline_stable_no_4h_volume_context")
    stable_no_4h_volume_columns = filter_feature_columns(columns, stable_no_4h_volume)
    assert "4h_volume_log_zscore" not in stable_no_4h_volume_columns
    assert "volume_log_zscore" in stable_no_4h_volume_columns

    stable_no_1h_ltr = profile_config(config, "baseline_stable_no_1h_large_trade_ratio")
    stable_no_1h_ltr_columns = filter_feature_columns(columns, stable_no_1h_ltr)
    assert "large_trade_ratio" not in stable_no_1h_ltr_columns
    assert "4h_large_trade_ratio" in stable_no_1h_ltr_columns
    for profile_name in [
        "baseline_stable_no_1h_cvd_rate",
        "baseline_stable_no_slow_4h_bounded_flow",
        "baseline_stable_no_slow_4h_bounded_flow_no_1h_cvd_rate",
        "baseline_stable_no_4h_whale_zscores",
        "baseline_stable_no_4h_volume_context",
        "baseline_stable_no_1h_large_trade_ratio",
        "baseline_stable_no_4h_taker_mean12",
        "baseline_stable_no_4h_large_trade_ratio",
        "baseline_stable_no_4h_taker_mean12_no_4h_large_trade_ratio",
        "baseline_stable_plus_4h_taker_mean12_ltr_context",
        "baseline_stable_plus_4h_taker_mean12_ltr_context_no_raw_ltr",
        "baseline_stable_plus_4h_taker_mean12_ltr_context_stable_only",
        "baseline_control_plus_futures_oi_change_context",
        "baseline_control_plus_futures_positioning_context",
        "baseline_control_plus_futures_funding_context",
        "baseline_stable_plus_15m_late_order_flow",
        "baseline_stable_plus_15m_intrahour_pressure",
        "baseline_stable_plus_15m_whale_burst",
        "baseline_stable_plus_15m_order_flow_full",
        "baseline_stable_plus_15m_absorption_rank",
        "baseline_stable_plus_15m_late_reversal_tanh",
        "baseline_stable_plus_15m_absorption_efficiency_combo",
    ]:
        assert profile_name in config["experiments"]["experiment_memory"]["rejected_profiles"]

    intrahour_late = profile_config(config, "baseline_stable_plus_15m_late_order_flow")
    intrahour_late_columns = filter_feature_columns(columns, intrahour_late)
    assert "ih15_taker_imbalance_late_minus_early" in intrahour_late_columns
    assert "ih15_cvd_pressure_norm" not in intrahour_late_columns
    assert "gk_vol_14" not in intrahour_late_columns

    intrahour_pressure = profile_config(config, "baseline_stable_plus_15m_intrahour_pressure")
    intrahour_pressure_columns = filter_feature_columns(columns, intrahour_pressure)
    assert "ih15_cvd_pressure_norm_stable_rank" in intrahour_pressure_columns
    assert "ih15_vpt_last_vs_hour" not in intrahour_pressure_columns
    assert "4h_taker_imbalance_mean_24" in intrahour_pressure_columns

    intrahour_whale = profile_config(config, "baseline_stable_plus_15m_whale_burst")
    intrahour_whale_columns = filter_feature_columns(columns, intrahour_whale)
    assert "ih15_vpt_last_vs_hour_stable_rank" in intrahour_whale_columns
    assert "ih15_large_trade_concentration" in intrahour_whale_columns
    assert "ih15_cvd_pressure_norm" not in intrahour_whale_columns

    intrahour_full = profile_config(config, "baseline_stable_plus_15m_order_flow_full")
    intrahour_full_columns = filter_feature_columns(columns, intrahour_full)
    assert "ih15_cvd_pressure_norm" in intrahour_full_columns
    assert "ih15_aggressive_sell_burst_stable_rank" in intrahour_full_columns

    intrahour_absorption = profile_config(config, "baseline_stable_plus_15m_absorption_rank")
    intrahour_absorption_columns = filter_feature_columns(columns, intrahour_absorption)
    assert "ih15_absorption_imbalance_stable_rank" in intrahour_absorption_columns
    assert "ih15_late_absorption_imbalance_stable_rank" in intrahour_absorption_columns
    assert "ih15_cvd_pressure_norm" not in intrahour_absorption_columns
    assert "ih15_missing" in intrahour_absorption_columns

    intrahour_efficiency = profile_config(config, "baseline_stable_plus_15m_flow_efficiency_rank")
    intrahour_efficiency_columns = filter_feature_columns(columns, intrahour_efficiency)
    assert "ih15_flow_price_alignment_stable_rank" in intrahour_efficiency_columns
    assert "ih15_flow_price_alignment_stable_tanh" in intrahour_efficiency_columns
    assert "ih15_buy_absorption_stable_rank" not in intrahour_efficiency_columns

    intrahour_reversal = profile_config(config, "baseline_stable_plus_15m_late_reversal_tanh")
    intrahour_reversal_columns = filter_feature_columns(columns, intrahour_reversal)
    assert "ih15_late_flow_reversal_stable_tanh" in intrahour_reversal_columns
    assert "ih15_late_price_reversal_stable_rank" in intrahour_reversal_columns
    assert "ih15_taker_imbalance_late_minus_early" not in intrahour_reversal_columns

    intrahour_combo = profile_config(config, "baseline_stable_plus_15m_absorption_efficiency_combo")
    intrahour_combo_columns = filter_feature_columns(columns, intrahour_combo)
    assert "ih15_absorption_imbalance_stable_tanh" in intrahour_combo_columns
    assert "ih15_late_flow_price_alignment_stable_tanh" in intrahour_combo_columns
    assert "ih15_cvd_slope_norm" not in intrahour_combo_columns

    pure_volatility_columns = {
        "realized_vol_14",
        "gk_vol_14",
        "atr_14_pct",
        "4h_realized_vol_14",
        "4h_gk_vol_14",
        "4h_atr_14_pct",
    }
    readd_profiles = {
        "baseline_plus_4h_bounded_whale_no_4h_tier1_no_raw_pure_volatility_except_1h_atr": "atr_14_pct",
        "baseline_plus_4h_bounded_whale_no_4h_tier1_no_raw_pure_volatility_except_1h_gk": "gk_vol_14",
        "baseline_plus_4h_bounded_whale_no_4h_tier1_no_raw_pure_volatility_except_1h_realized": "realized_vol_14",
        "baseline_plus_4h_bounded_whale_no_4h_tier1_no_raw_pure_volatility_except_4h_atr": "4h_atr_14_pct",
        "baseline_plus_4h_bounded_whale_no_4h_tier1_no_raw_pure_volatility_except_4h_gk": "4h_gk_vol_14",
        "baseline_plus_4h_bounded_whale_no_4h_tier1_no_raw_pure_volatility_except_4h_realized": "4h_realized_vol_14",
    }
    for profile_name, retained_column in readd_profiles.items():
        readd = profile_config(config, profile_name)
        readd_columns = set(filter_feature_columns(columns, readd))
        assert retained_column in readd_columns
        assert not (pure_volatility_columns - {retained_column}).intersection(readd_columns)
        assert "4h_taker_imbalance_mean_24" in readd_columns
        assert profile_name in config["experiments"]["experiment_memory"]["rejected_profiles"]

    interaction_expectations = {
        "baseline_stable_plus_1h_flow_rv_interactions": (
            "taker_imbalance_x_rv14_rank_signed",
            "4h_taker_imbalance_x_rv14_rank_signed",
            "gk_vol_14",
        ),
        "baseline_stable_plus_1h_flow_gk_interactions": (
            "signed_ltp_x_gk14_rank_low",
            "4h_signed_ltp_x_gk14_rank_low",
            "gk_vol_14",
        ),
        "baseline_stable_plus_1h_flow_atr_interactions": (
            "ltp24_rank_x_atr14_rank_high",
            "4h_ltp24_rank_x_atr14_rank_high",
            "atr_14_pct",
        ),
        "baseline_stable_plus_4h_flow_rv_interactions": (
            "4h_ltp12_rank_x_rv14_rank_signed",
            "ltp12_rank_x_rv14_rank_signed",
            "4h_realized_vol_14",
        ),
        "baseline_stable_plus_4h_flow_gk_interactions": (
            "4h_taker_imbalance_slope_x_gk14_rank_high",
            "taker_imbalance_slope_x_gk14_rank_high",
            "4h_gk_vol_14",
        ),
        "baseline_stable_plus_4h_flow_atr_interactions": (
            "4h_signed_ltp_x_atr14_rank_low",
            "signed_ltp_x_atr14_rank_low",
            "4h_atr_14_pct",
        ),
    }
    for profile_name, (included, excluded_interaction, excluded_raw_volatility) in interaction_expectations.items():
        profile_columns = set(filter_feature_columns(columns, profile_config(config, profile_name)))
        assert included in profile_columns
        assert excluded_interaction not in profile_columns
        assert excluded_raw_volatility not in profile_columns
        assert "4h_taker_imbalance_mean_24" in profile_columns
        assert profile_name in config["experiments"]["experiment_memory"]["rejected_profiles"]

    base_no_raw_vol_4h_stable_overlay = profile_config(
        config,
        "baseline_plus_4h_bounded_whale_no_4h_tier1_no_4h_1h_pure_volatility_plus_4h_stable_vol_overlay",
    )
    base_no_raw_vol_4h_stable_overlay_columns = filter_feature_columns(columns, base_no_raw_vol_4h_stable_overlay)
    assert "4h_gk_vol_14" not in base_no_raw_vol_4h_stable_overlay_columns
    assert "gk_vol_14" not in base_no_raw_vol_4h_stable_overlay_columns
    assert "4h_gk_vol_14_stable_rank" in base_no_raw_vol_4h_stable_overlay_columns
    assert "4h_vwap_dist_atr_stable_zscore" in base_no_raw_vol_4h_stable_overlay_columns

    base_no_vol_no_4h_volume = profile_config(
        config,
        "baseline_plus_4h_bounded_whale_no_4h_tier1_no_4h_pure_volatility_no_4h_volume_context",
    )
    base_no_vol_no_4h_volume_columns = filter_feature_columns(columns, base_no_vol_no_4h_volume)
    assert "4h_gk_vol_14" not in base_no_vol_no_4h_volume_columns
    assert "4h_volume_log_zscore" not in base_no_vol_no_4h_volume_columns
    assert "volume_log_zscore" in base_no_vol_no_4h_volume_columns

    base_no_cvd_rate = profile_config(config, "baseline_plus_4h_bounded_whale_no_4h_tier1_no_1h_cvd_rate")
    base_no_cvd_rate_columns = filter_feature_columns(columns, base_no_cvd_rate)
    assert "cvd_cumulative_rate_norm" not in base_no_cvd_rate_columns

    base_guardrail = profile_config(config, "baseline_plus_4h_bounded_whale_no_4h_tier1_bad_fold_guardrail_light")
    base_guardrail_columns = filter_feature_columns(columns, base_guardrail)
    assert "4h_taker_imbalance_mean_24" not in base_guardrail_columns
    assert "4h_large_trade_ratio" not in base_guardrail_columns
    assert "4h_gk_vol_14" not in base_guardrail_columns
    assert "cvd_cumulative_rate_norm" not in base_guardrail_columns

    cvd = profile_config(config, "baseline_no_4h_tier1_4h_cvd_pressure_stable")
    cvd_columns = filter_feature_columns(columns, cvd)
    assert "4h_cvd_pressure_3" not in cvd_columns
    assert "4h_cvd_pressure_3_stable_rank" in cvd_columns
    assert "4h_gk_vol_14_stable_rank" not in cvd_columns

    large = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_stable")
    large_columns = filter_feature_columns(columns, large)
    assert "4h_signed_large_trade_pressure_stable_rank" in large_columns
    assert "4h_large_trade_pressure_12_stable_zscore" in large_columns
    assert "4h_cvd_pressure_3_stable_rank" not in large_columns

    pruned_large = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_stable_pruned_whale")
    pruned_large_columns = filter_feature_columns(columns, pruned_large)
    assert "4h_large_trade_ratio" not in pruned_large_columns
    assert "4h_signed_large_trade_pressure_stable_rank" in pruned_large_columns

    rank_only = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_rank_only")
    rank_columns = filter_feature_columns(columns, rank_only)
    assert "4h_large_trade_pressure_12_stable_rank" in rank_columns
    assert "4h_large_trade_pressure_12_stable_zscore" not in rank_columns

    zscore_only = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_zscore_only")
    zscore_columns = filter_feature_columns(columns, zscore_only)
    assert "4h_large_trade_pressure_12_stable_zscore" in zscore_columns
    assert "4h_large_trade_pressure_12_stable_rank" not in zscore_columns

    short = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_short")
    short_columns = filter_feature_columns(columns, short)
    assert "4h_large_trade_pressure_3_stable_rank" in short_columns
    assert "4h_large_trade_pressure_12_stable_rank" not in short_columns

    long = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long")
    long_columns = filter_feature_columns(columns, long)
    assert "4h_large_trade_pressure_24_stable_rank" in long_columns
    assert "4h_large_trade_pressure_6_stable_rank" not in long_columns

    no_ratio = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_4h_large_trade_ratio")
    no_ratio_columns = filter_feature_columns(columns, no_ratio)
    assert "4h_large_trade_ratio" not in no_ratio_columns
    assert "4h_large_trade_pressure_24_stable_rank" in no_ratio_columns

    no_whale_flags = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_4h_whale_flags")
    no_whale_columns = filter_feature_columns(columns, no_whale_flags)
    assert "4h_large_trade_ratio" not in no_whale_columns
    assert "4h_whale_buy_flag" not in no_whale_columns
    assert "4h_vpt_zscore" in no_whale_columns

    no_structure = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_4h_structure")
    no_structure_columns = filter_feature_columns(columns, no_structure)
    assert "4h_gk_vol_14" not in no_structure_columns
    assert "4h_vwap_dist_atr" not in no_structure_columns
    assert "4h_large_trade_pressure_24_stable_rank" in no_structure_columns

    stable_structure = profile_config(
        config,
        "baseline_no_4h_tier1_4h_large_trade_pressure_long_4h_structure_stable_overlay",
    )
    stable_structure_columns = filter_feature_columns(columns, stable_structure)
    assert "4h_gk_vol_14" not in stable_structure_columns
    assert "4h_gk_vol_14_stable_rank" in stable_structure_columns
    assert "4h_vwap_dist_atr_stable_zscore" in stable_structure_columns

    no_4h_volume = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_4h_volume_context")
    no_4h_volume_columns = filter_feature_columns(columns, no_4h_volume)
    assert "4h_volume_log_zscore" not in no_4h_volume_columns
    assert "volume_log_zscore" in no_4h_volume_columns
    assert "4h_vwap_dist_atr" in no_4h_volume_columns

    no_1h_volume = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_1h_volume_context")
    no_1h_volume_columns = filter_feature_columns(columns, no_1h_volume)
    assert "volume_log_zscore" not in no_1h_volume_columns
    assert "4h_volume_log_zscore" in no_1h_volume_columns

    no_volume = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_volume_context")
    no_volume_columns = filter_feature_columns(columns, no_volume)
    assert "volume_log_zscore" not in no_volume_columns
    assert "4h_volume_log_zscore" not in no_volume_columns
    assert "4h_large_trade_pressure_24_stable_rank" in no_volume_columns

    no_4h_volatility = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_4h_pure_volatility")
    no_4h_volatility_columns = filter_feature_columns(columns, no_4h_volatility)
    assert "4h_gk_vol_14" not in no_4h_volatility_columns
    assert "4h_atr_14_pct" not in no_4h_volatility_columns
    assert "4h_vwap_dist_atr" in no_4h_volatility_columns

    pressure_no_raw_volatility = profile_config(
        config,
        "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_4h_1h_pure_volatility",
    )
    pressure_no_raw_volatility_columns = filter_feature_columns(columns, pressure_no_raw_volatility)
    assert "4h_gk_vol_14" not in pressure_no_raw_volatility_columns
    assert "gk_vol_14" not in pressure_no_raw_volatility_columns
    assert "4h_large_trade_pressure_24_stable_rank" in pressure_no_raw_volatility_columns
    assert "4h_large_trade_pressure_12_stable_rank" in pressure_no_raw_volatility_columns

    vwap_only = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_4h_vwap_only_structure")
    vwap_only_columns = filter_feature_columns(columns, vwap_only)
    assert "4h_vwap_dist_atr" in vwap_only_columns
    assert "4h_gk_vol_14" not in vwap_only_columns
    assert "4h_adx_14" not in vwap_only_columns

    no_bounded_flow = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_4h_bounded_flow")
    no_bounded_flow_columns = filter_feature_columns(columns, no_bounded_flow)
    assert "4h_taker_imbalance_mean_24" not in no_bounded_flow_columns
    assert "4h_taker_buy_ratio_zscore" not in no_bounded_flow_columns
    assert "4h_large_trade_pressure_24_stable_rank" in no_bounded_flow_columns
    assert "4h_large_trade_ratio" in no_bounded_flow_columns

    stable_vol = profile_config(
        config,
        "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_4h_pure_volatility_4h_stable_vol_overlay",
    )
    stable_vol_columns = filter_feature_columns(columns, stable_vol)
    assert "4h_gk_vol_14" not in stable_vol_columns
    assert "4h_gk_vol_14_stable_rank" in stable_vol_columns
    assert "4h_realized_vol_14_stable_zscore" in stable_vol_columns
    assert "4h_adx_14_stable_rank" not in stable_vol_columns
    assert "4h_volume_log_zscore_stable_rank" not in stable_vol_columns

    no_whale_zscores = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_4h_whale_zscores")
    no_whale_zscore_columns = filter_feature_columns(columns, no_whale_zscores)
    assert "4h_vpt_zscore" not in no_whale_zscore_columns
    assert "4h_vol_per_trade_log_zscore" not in no_whale_zscore_columns
    assert "4h_large_trade_ratio" in no_whale_zscore_columns
    assert "4h_whale_buy_flag" in no_whale_zscore_columns

    pressure_24 = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_pressure_24_only")
    pressure_24_columns = filter_feature_columns(columns, pressure_24)
    assert "4h_large_trade_pressure_12_stable_rank" not in pressure_24_columns
    assert "4h_large_trade_pressure_24_stable_rank" in pressure_24_columns

    no_12_zscore = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_12_pressure_zscore")
    no_12_zscore_columns = filter_feature_columns(columns, no_12_zscore)
    assert "4h_large_trade_pressure_12_stable_zscore" not in no_12_zscore_columns
    assert "4h_large_trade_pressure_12_stable_rank" in no_12_zscore_columns
    assert "4h_large_trade_pressure_24_stable_zscore" in no_12_zscore_columns

    no_12_rank = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_12_pressure_rank")
    no_12_rank_columns = filter_feature_columns(columns, no_12_rank)
    assert "4h_large_trade_pressure_12_stable_rank" not in no_12_rank_columns
    assert "4h_large_trade_pressure_12_stable_zscore" in no_12_rank_columns
    assert "4h_large_trade_pressure_24_stable_rank" in no_12_rank_columns

    tanh_replacement = profile_config(
        config,
        "baseline_no_4h_tier1_4h_large_trade_pressure_long_12_pressure_tanh_replacement",
    )
    tanh_replacement_columns = filter_feature_columns(columns, tanh_replacement)
    assert "4h_large_trade_pressure_12_stable_zscore" not in tanh_replacement_columns
    assert "4h_large_trade_pressure_12_stable_tanh" in tanh_replacement_columns
    assert "4h_large_trade_pressure_12_stable_rank" in tanh_replacement_columns
    assert "4h_large_trade_pressure_24_stable_zscore" in tanh_replacement_columns

    tanh_pair = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_pressure_tanh_pair")
    tanh_pair_columns = filter_feature_columns(columns, tanh_pair)
    assert "4h_signed_large_trade_pressure_stable_tanh" in tanh_pair_columns
    assert "4h_large_trade_pressure_12_stable_tanh" in tanh_pair_columns
    assert "4h_large_trade_pressure_24_stable_tanh" in tanh_pair_columns
    assert "4h_large_trade_pressure_12_stable_zscore" not in tanh_pair_columns

    spread_overlay = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_pressure_spread_overlay")
    spread_overlay_columns = filter_feature_columns(columns, spread_overlay)
    assert "4h_large_trade_pressure_24_minus_12_stable_rank" in spread_overlay_columns
    assert "4h_large_trade_pressure_12_stable_zscore" in spread_overlay_columns

    tanh_spread = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_12_tanh_spread_overlay")
    tanh_spread_columns = filter_feature_columns(columns, tanh_spread)
    assert "4h_large_trade_pressure_12_stable_zscore" not in tanh_spread_columns
    assert "4h_large_trade_pressure_12_stable_tanh" in tanh_spread_columns
    assert "4h_large_trade_pressure_24_minus_12_stable_tanh" in tanh_spread_columns

    no_slow_flow = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_slow_4h_bounded_flow")
    no_slow_flow_columns = filter_feature_columns(columns, no_slow_flow)
    assert "4h_taker_imbalance_mean_12" not in no_slow_flow_columns
    assert "4h_taker_imbalance_mean_24" not in no_slow_flow_columns
    assert "4h_taker_imbalance" in no_slow_flow_columns
    assert "4h_taker_imbalance_slope" in no_slow_flow_columns

    no_cvd_rate = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_1h_cvd_rate")
    no_cvd_rate_columns = filter_feature_columns(columns, no_cvd_rate)
    assert "cvd_cumulative_rate_norm" not in no_cvd_rate_columns
    assert "4h_taker_imbalance_mean_24" in no_cvd_rate_columns

    combined_cvd_flow = profile_config(
        config,
        "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_slow_4h_bounded_flow_no_1h_cvd_rate",
    )
    combined_cvd_flow_columns = filter_feature_columns(columns, combined_cvd_flow)
    assert "cvd_cumulative_rate_norm" not in combined_cvd_flow_columns
    assert "4h_taker_imbalance_mean_24" not in combined_cvd_flow_columns

    guardrail = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_bad_fold_guardrail_light")
    guardrail_columns = filter_feature_columns(columns, guardrail)
    assert "cvd_cumulative_rate_norm" not in guardrail_columns
    assert "4h_taker_imbalance_mean_12" not in guardrail_columns
    assert "4h_large_trade_pressure_12_stable_rank" not in guardrail_columns
    assert "4h_gk_vol_14" not in guardrail_columns
    assert "4h_large_trade_pressure_24_stable_rank" in guardrail_columns

    no_vpt_only = profile_config(config, "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_4h_vpt_zscore_only")
    no_vpt_only_columns = filter_feature_columns(columns, no_vpt_only)
    assert "4h_vpt_zscore" not in no_vpt_only_columns
    assert "4h_vol_per_trade_log_zscore" in no_vpt_only_columns
    assert "4h_whale_buy_flag" in no_vpt_only_columns

    no_vpt_log_only = profile_config(
        config,
        "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_4h_vol_per_trade_log_zscore_only",
    )
    no_vpt_log_only_columns = filter_feature_columns(columns, no_vpt_log_only)
    assert "4h_vol_per_trade_log_zscore" not in no_vpt_log_only_columns
    assert "4h_vpt_zscore" in no_vpt_log_only_columns
    assert "4h_whale_sell_flag" in no_vpt_log_only_columns

    no_vol_pressure_24 = profile_config(
        config,
        "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_4h_pure_volatility_pressure_24_only",
    )
    no_vol_pressure_24_columns = filter_feature_columns(columns, no_vol_pressure_24)
    assert "4h_gk_vol_14" not in no_vol_pressure_24_columns
    assert "4h_large_trade_pressure_12_stable_rank" not in no_vol_pressure_24_columns
    assert "4h_large_trade_pressure_24_stable_rank" in no_vol_pressure_24_columns

    no_raw_vol_pressure_24 = profile_config(
        config,
        "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_4h_1h_pure_volatility_pressure_24_only",
    )
    no_raw_vol_pressure_24_columns = filter_feature_columns(columns, no_raw_vol_pressure_24)
    assert "4h_gk_vol_14" not in no_raw_vol_pressure_24_columns
    assert "gk_vol_14" not in no_raw_vol_pressure_24_columns
    assert "4h_large_trade_pressure_12_stable_rank" not in no_raw_vol_pressure_24_columns
    assert "4h_large_trade_pressure_24_stable_rank" in no_raw_vol_pressure_24_columns

    no_vol_no_whale_zscores = profile_config(
        config,
        "baseline_no_4h_tier1_4h_large_trade_pressure_long_no_4h_pure_volatility_no_4h_whale_zscores",
    )
    no_vol_no_whale_zscore_columns = filter_feature_columns(columns, no_vol_no_whale_zscores)
    assert "4h_gk_vol_14" not in no_vol_no_whale_zscore_columns
    assert "4h_vpt_zscore" not in no_vol_no_whale_zscore_columns
    assert "4h_whale_sell_flag" in no_vol_no_whale_zscore_columns


def test_full_promotion_gate_uses_threshold_selected_f1() -> None:
    config = {
        "experiments": {
            "promotion_gates": {
                "full": {
                    "min_mean_rank_ic_delta": 0.005,
                    "min_positive_ic_fraction_floor": 0.75,
                    "max_std_rank_ic_delta": 0.002,
                    "min_selected_threshold_f1": 0.45,
                    "min_selected_threshold_f1_delta": 0.0,
                    "min_long_f1_delta": None,
                    "min_top_10_lift_global_delta": 0.05,
                    "min_top_10_lift_global": 1.0,
                }
            }
        }
    }
    control = {
        "mean_rank_ic": 0.048,
        "positive_ic_fraction": 0.738,
        "std_rank_ic": 0.085,
        "mean_long_f1": 0.268,
        "test_f1_at_selected_threshold": 0.463,
        "top_10_lift_global": 1.044,
        "mtf_leakage_passed": True,
        "stationarity_policy_passed": True,
    }
    candidate = {
        "mean_rank_ic": 0.060,
        "positive_ic_fraction": 0.762,
        "std_rank_ic": 0.086,
        "mean_long_f1": 0.274,
        "test_f1_at_selected_threshold": 0.464,
        "top_10_lift_global": 1.141,
        "mtf_leakage_passed": True,
        "stationarity_policy_passed": True,
    }

    assert _passes_full(candidate, control, config) == (True, "")
    weak_top_lift = dict(candidate, top_10_lift_global=0.99)
    assert _passes_full(weak_top_lift, control, config) == (
        False,
        "top_10_lift_global_delta;top_10_lift_global",
    )


def test_triage_promotion_gate_uses_downside_metrics() -> None:
    config = {
        "experiments": {
            "promotion_gates": {
                "triage": {
                    "min_mean_rank_ic_delta": 0.005,
                    "max_std_rank_ic_delta": 0.005,
                    "min_top_10_lift_global": 1.05,
                    "min_top_10_positive_lift_fold_rate": 0.55,
                    "min_worst_5_rank_ic_delta": 0.0,
                    "max_negative_ic_fraction_delta": 0.0,
                    "min_top_10_bad_fold_lift_mean": 1.0,
                }
            }
        }
    }
    control = {
        "mean_rank_ic": 0.03,
        "std_rank_ic": 0.10,
        "positive_ic_fraction": 0.50,
        "top_10_lift_global": 1.07,
        "top_10_positive_lift_fold_rate": 0.56,
        "worst_5_rank_ic_mean": -0.11,
        "negative_ic_fraction": 0.50,
        "top_10_bad_fold_lift_mean": 0.95,
        "mtf_leakage_passed": True,
        "stationarity_policy_passed": True,
    }
    candidate = {
        "mean_rank_ic": 0.04,
        "std_rank_ic": 0.09,
        "positive_ic_fraction": 0.62,
        "top_10_lift_global": 1.16,
        "top_10_positive_lift_fold_rate": 0.70,
        "worst_5_rank_ic_mean": -0.08,
        "negative_ic_fraction": 0.38,
        "top_10_bad_fold_lift_mean": 1.05,
        "mtf_leakage_passed": True,
        "stationarity_policy_passed": True,
    }
    weaker_downside = dict(candidate, worst_5_rank_ic_mean=-0.13)

    assert _passes_triage(candidate, control, config) == (True, "")
    assert _passes_triage(weaker_downside, control, config) == (False, "worst_5_rank_ic_delta")


def test_profile_blend_review_separates_tail_lift_and_stability_leaders() -> None:
    config = {
        "experiments": {
            "profile_blend_review_gates": {
                "min_mean_rank_ic_delta": 0.005,
                "max_std_rank_ic_delta": 0.0,
                "min_positive_ic_fraction": 0.70,
                "min_top_10_lift_global_delta": 0.02,
            },
            "profile_blend_leader_gates": {
                "tail_lift": {
                    "min_mean_rank_ic_delta": 0.005,
                    "max_std_rank_ic_delta": 0.0,
                    "min_positive_ic_fraction": 0.70,
                    "min_top_10_lift_global_delta": 0.02,
                },
                "stability": {
                    "min_mean_rank_ic_delta": 0.005,
                    "max_std_rank_ic_delta": 0.0,
                    "min_positive_ic_fraction": 0.70,
                    "min_worst_5_rank_ic_delta": 0.02,
                },
            },
        }
    }
    comparison = pd.DataFrame(
        [
            {
                "profile": "control",
                "fold_scope": "full",
                "mean_rank_ic": 0.057,
                "std_rank_ic": 0.101,
                "positive_ic_fraction": 0.786,
                "test_f1_at_selected_threshold": 0.467,
                "mean_long_f1": 0.270,
                "worst_5_rank_ic_mean": -0.113,
                "top_10_lift_global": 1.139,
            }
        ]
    )
    profile_blend = pd.DataFrame(
        [
            {
                "profile": "tail",
                "mean_rank_ic": 0.065,
                "std_rank_ic": 0.093,
                "positive_ic_fraction": 0.714,
                "test_f1_at_selected_threshold": 0.464,
                "mean_long_f1": 0.267,
                "worst_5_rank_ic_mean": -0.077,
                "top_10_lift_global": 1.172,
                "mtf_leakage_passed": True,
                "stationarity_policy_passed": True,
            },
            {
                "profile": "stable",
                "mean_rank_ic": 0.068,
                "std_rank_ic": 0.085,
                "positive_ic_fraction": 0.738,
                "test_f1_at_selected_threshold": 0.465,
                "mean_long_f1": 0.270,
                "worst_5_rank_ic_mean": -0.063,
                "top_10_lift_global": 1.119,
                "mtf_leakage_passed": True,
                "stationarity_policy_passed": True,
            },
        ]
    )

    reviewed = _profile_blend_review_frame(profile_blend, comparison, config, "control")
    leaders = _profile_blend_leaders(reviewed)

    assert leaders["tail_lift_leader"]["profile"] == "tail"
    assert leaders["stability_leader"]["profile"] == "stable"
    assert bool(reviewed.loc[reviewed["profile"] == "tail", "tail_lift_leader"].item()) is True
    assert bool(reviewed.loc[reviewed["profile"] == "stable", "stability_leader"].item()) is True
    assert bool(reviewed.loc[reviewed["profile"] == "stable", "tail_lift_eligible"].item()) is False
    assert bool(reviewed.loc[reviewed["profile"] == "stable", "stability_eligible"].item()) is True


def test_profile_blend_predictions_support_weighted_probability_blends() -> None:
    timestamps = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    base = pd.DataFrame(
        {
            "fold": [0, 0, 0],
            "timestamp": timestamps,
            "label": [0, 1, 0],
            "fwd_return_10h": [0.0, 0.01, -0.01],
            "prob_long": [0.10, 0.20, 0.30],
        }
    )
    overlay = base.copy()
    overlay["prob_long"] = [0.90, 0.80, 0.70]
    entries = [
        {"profile": "control", "predictions": base},
        {"profile": "overlay", "predictions": overlay},
    ]

    blended = _profile_blend_predictions(entries, method="prob_weighted", weights=[0.75, 0.25])

    assert blended["blend_method"].unique().tolist() == ["prob_weighted"]
    assert blended["blend_profiles"].unique().tolist() == ["control,overlay"]
    assert blended["blend_weights"].unique().tolist() == ["0.75,0.25"]
    np.testing.assert_allclose(blended["prob_long"], [0.30, 0.35, 0.40])
    assert (blended["blend_profile_count"] == 2).all()


def test_profile_blend_review_selects_balanced_leader_before_tail_lift() -> None:
    config = {
        "experiments": {
            "profile_blend_review_gates": {
                "min_mean_rank_ic_delta": 0.005,
                "max_std_rank_ic_delta": 0.02,
                "min_positive_ic_fraction": 0.70,
                "min_top_10_lift_global_delta": 0.02,
            },
            "profile_blend_leader_gates": {
                "balanced": {
                    "min_mean_rank_ic_delta": 0.005,
                    "max_std_rank_ic_delta": 0.005,
                    "min_positive_ic_fraction": 0.75,
                    "min_top_10_lift_global_delta": 0.015,
                    "min_worst_5_rank_ic_delta": 0.0,
                },
                "tail_lift": {
                    "min_mean_rank_ic_delta": 0.005,
                    "max_std_rank_ic_delta": 0.02,
                    "min_positive_ic_fraction": 0.70,
                    "min_top_10_lift_global_delta": 0.02,
                },
            },
        }
    }
    comparison = pd.DataFrame(
        [
            {
                "profile": "control",
                "fold_scope": "full",
                "mean_rank_ic": 0.0596,
                "std_rank_ic": 0.0738,
                "positive_ic_fraction": 0.857,
                "worst_5_rank_ic_mean": -0.068,
                "top_10_lift_global": 1.126,
            }
        ]
    )
    profile_blend = pd.DataFrame(
        [
            {
                "profile": "tail",
                "mean_rank_ic": 0.0650,
                "std_rank_ic": 0.0926,
                "positive_ic_fraction": 0.714,
                "worst_5_rank_ic_mean": -0.077,
                "top_10_lift_global": 1.172,
                "mtf_leakage_passed": True,
                "stationarity_policy_passed": True,
            },
            {
                "profile": "balanced",
                "mean_rank_ic": 0.0664,
                "std_rank_ic": 0.0751,
                "positive_ic_fraction": 0.786,
                "worst_5_rank_ic_mean": -0.054,
                "top_10_lift_global": 1.148,
                "mtf_leakage_passed": True,
                "stationarity_policy_passed": True,
            },
        ]
    )

    reviewed = _profile_blend_review_frame(profile_blend, comparison, config, "control")
    leaders = _profile_blend_leaders(reviewed)

    assert leaders["balanced_leader"]["profile"] == "balanced"
    assert leaders["tail_lift_leader"]["profile"] == "tail"
    assert _best_profile_blend(reviewed)["profile"] == "balanced"
    assert bool(reviewed.loc[reviewed["profile"] == "balanced", "balanced_leader"].item()) is True
    assert bool(reviewed.loc[reviewed["profile"] == "tail", "balanced_eligible"].item()) is False


def test_profile_experiment_writes_isolated_outputs_and_resumes(synthetic_klines, tiny_config, tmp_path) -> None:
    config = copy.deepcopy(tiny_config)
    config["features"]["active_profile"] = "base"
    config["features"]["profiles"] = {"base": {"include_patterns": ["*"], "exclude_patterns": []}}
    frame, feature_columns = _labeled_frame(synthetic_klines, config, periods=220)
    assert feature_columns

    first = run_profile_experiment(
        frame,
        config,
        profile="base",
        checkpoint_dir=tmp_path,
        run_id="run_a",
        fold_scope="triage",
        fold_ids=[0],
        device="cpu",
    )
    second = run_profile_experiment(
        frame,
        config,
        profile="base",
        checkpoint_dir=tmp_path,
        run_id="run_a",
        fold_scope="triage",
        fold_ids=[0],
        device="cpu",
    )

    assert not first["skipped"]
    assert second["skipped"]
    assert first["output_dir"] != tmp_path
    assert (first["output_dir"] / "predictions_all.parquet").exists()
    assert (first["output_dir"] / "training_manifest.json").exists()


def test_experiment_matrix_and_diagnostics_write_profile_comparison(synthetic_klines, tiny_config, tmp_path) -> None:
    config = copy.deepcopy(tiny_config)
    config["features"]["profiles"] = {
        "control": {"include_patterns": ["*"], "exclude_patterns": ["4h_taker_buy_ratio"]},
        "candidate": {"include_patterns": ["*"], "exclude_patterns": ["4h_taker_sell_ratio"]},
    }
    config["experiments"] = {
        "mode": "staged",
        "control_profile": "control",
        "candidate_profiles": ["candidate"],
        "triage_fold_ids": [0],
        "full_cv_profiles": ["control", "candidate"],
        "always_full_profiles": ["control", "candidate"],
        "resume_existing": True,
        "force_retrain": False,
    }
    config["validation"]["calibration"] = {"enabled": True, "method": "isotonic"}
    config["validation"]["fold_reliability_gates"] = {
        "enabled": True,
        "min_accepted_fraction": 0.10,
        "min_accepted_folds": 1,
        "min_positive_ic_fraction": 0.0,
        "max_rank_ic_std": 2.0,
        "min_official_f1_delta": -1.0,
        "gates": [{"name": "val_rank_ic_positive", "min_val_rank_ic": -1.0}],
    }
    config["validation"]["regime_threshold_policy"] = {
        "enabled": True,
        "min_regime_val_rows": 1,
        "min_regime_test_rows": 1,
        "min_regime_val_longs": 1,
        "max_pred_long_rate": 0.90,
        "min_precision": 0.0,
        "min_f1_delta_vs_official": -1.0,
        "min_policy_pass_fold_rate": 0.0,
        "min_positive_forward_return_fold_rate": 0.0,
    }
    frame, _ = _labeled_frame(synthetic_klines, config, periods=220)

    result = run_experiment_matrix(frame, config, checkpoint_dir=tmp_path, run_id="matrix", device="cpu")
    diagnostics_config = copy.deepcopy(config)
    diagnostics_config["experiments"]["always_full_profiles"].append("ghost_profile_not_in_run")
    diagnostics = write_experiment_diagnostics(
        checkpoint_dir=tmp_path,
        config=diagnostics_config,
        output_dir=tmp_path / "reports",
        run_id="matrix",
        write_full_bundles=True,
    )

    assert set(result["comparison"]["profile"]) == {"control", "candidate"}
    assert not result["profile_delta"].empty
    assert not result["fold_reliability_gate"].empty
    assert not result["fold_reliability_gate_summary"].empty
    assert not result["regime_threshold_policy_by_fold"].empty
    assert not result["regime_threshold_policy_summary"].empty
    assert not result["regime_stability_forensics"].empty
    assert not result["regime_stability_summary"].empty
    assert {"rank_ic_delta", "top_10_lift_delta", "threshold_f1_delta"}.issubset(result["profile_delta"].columns)
    assert (tmp_path / "experiments" / "matrix" / "profile_comparison.csv").exists()
    assert (tmp_path / "experiments" / "matrix" / "profile_delta_vs_control.csv").exists()
    assert (tmp_path / "experiments" / "matrix" / "profile_blend.csv").exists()
    assert (tmp_path / "experiments" / "matrix" / "performance_gap_analysis.csv").exists()
    assert (tmp_path / "experiments" / "matrix" / "phase1_blocker_action_plan.csv").exists()
    assert (tmp_path / "experiments" / "matrix" / "fold_stability_forensics.csv").exists()
    assert (tmp_path / "experiments" / "matrix" / "fold_stability_summary.csv").exists()
    assert (tmp_path / "experiments" / "matrix" / "score_separation_forensics.csv").exists()
    assert (tmp_path / "experiments" / "matrix" / "bad_fold_signature.csv").exists()
    assert (tmp_path / "experiments" / "matrix" / "feature_drift_forensics.csv").exists()
    assert (tmp_path / "experiments" / "matrix" / "feature_family_drift_summary.csv").exists()
    assert (tmp_path / "experiments" / "matrix" / "probability_quality_forensics.csv").exists()
    assert (tmp_path / "experiments" / "matrix" / "probability_quality_summary.csv").exists()
    assert (tmp_path / "experiments" / "matrix" / "score_distribution_shift.csv").exists()
    assert (tmp_path / "experiments" / "matrix" / "score_distribution_shift_summary.csv").exists()
    assert (tmp_path / "experiments" / "matrix" / "fold_reliability_gate.csv").exists()
    assert (tmp_path / "experiments" / "matrix" / "fold_reliability_gate_summary.csv").exists()
    assert (tmp_path / "experiments" / "matrix" / "regime_threshold_policy_by_fold.csv").exists()
    assert (tmp_path / "experiments" / "matrix" / "regime_threshold_policy_summary.csv").exists()
    assert (tmp_path / "experiments" / "matrix" / "regime_stability_forensics.csv").exists()
    assert (tmp_path / "experiments" / "matrix" / "regime_stability_summary.csv").exists()
    assert (tmp_path / "experiments" / "matrix" / "threshold_forensics.csv").exists()
    assert (tmp_path / "experiments" / "matrix" / "threshold_policy_review.csv").exists()
    assert (tmp_path / "experiments" / "matrix" / "threshold_transfer_review.csv").exists()
    assert (tmp_path / "experiments" / "matrix" / "threshold_transfer_by_fold.csv").exists()
    assert (tmp_path / "experiments" / "matrix" / "payoff_alignment.csv").exists()
    assert (tmp_path / "experiments" / "matrix" / "payoff_alignment_summary.csv").exists()
    assert (tmp_path / "experiments" / "matrix" / "payoff_policy_robustness.csv").exists()
    assert (tmp_path / "experiments" / "matrix" / "payoff_policy_robustness_summary.csv").exists()
    assert (tmp_path / "experiments" / "matrix" / "experiment_selection.csv").exists()
    assert (tmp_path / "experiments" / "matrix" / "missing_selected_profiles.csv").exists()
    assert (tmp_path / "experiments" / "matrix" / "holdout_reservation.csv").exists()
    assert (tmp_path / "experiments" / "matrix" / "future_oos_candidate_plan.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "profile_comparison.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "profile_delta_vs_control.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "profile_blend.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "performance_gap_analysis.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "phase1_blocker_action_plan.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "fold_stability_forensics.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "fold_stability_summary.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "score_separation_forensics.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "bad_fold_signature.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "feature_drift_forensics.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "feature_family_drift_summary.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "probability_quality_forensics.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "probability_quality_summary.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "score_distribution_shift.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "score_distribution_shift_summary.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "fold_reliability_gate.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "fold_reliability_gate_summary.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "regime_threshold_policy_by_fold.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "regime_threshold_policy_summary.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "regime_stability_forensics.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "regime_stability_summary.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "threshold_forensics.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "threshold_policy_review.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "threshold_transfer_review.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "threshold_transfer_by_fold.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "payoff_alignment.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "payoff_alignment_summary.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "payoff_policy_robustness.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "payoff_policy_robustness_summary.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "experiment_selection.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "missing_selected_profiles.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "holdout_reservation.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "future_oos_candidate_plan.csv").exists()
    assert diagnostics["zip_paths"]
    assert not diagnostics["profile_delta"].empty
    assert not diagnostics["profile_blend"].empty
    assert not diagnostics["performance_gap_analysis"].empty
    assert not diagnostics["phase1_blocker_action_plan"].empty
    assert not diagnostics["fold_stability_forensics"].empty
    assert not diagnostics["fold_stability_summary"].empty
    assert not diagnostics["score_separation_forensics"].empty
    assert {"feature", "feature_family", "suspect_score", "likely_issue"}.issubset(
        diagnostics["feature_drift_forensics"].columns
    )
    assert {"feature_family", "top_suspect_feature", "recommended_next_action"}.issubset(
        diagnostics["feature_family_drift_summary"].columns
    )
    assert {"brier_score", "log_loss", "average_precision", "ece_equal_count"}.issubset(
        diagnostics["probability_quality_forensics"].columns
    )
    assert {"mean_brier_score", "mean_average_precision", "probability_quality_issue"}.issubset(
        diagnostics["probability_quality_summary"].columns
    )
    assert {"score_ks_vs_reference", "score_psi_vs_reference", "score_shift_issue"}.issubset(
        diagnostics["score_distribution_shift"].columns
    )
    assert {"max_score_psi", "high_shift_folds", "recommended_next_action"}.issubset(
        diagnostics["score_distribution_shift_summary"].columns
    )
    assert not diagnostics["fold_reliability_gate"].empty
    assert not diagnostics["fold_reliability_gate_summary"].empty
    assert {"val_rank_ic", "gate_passed", "test_rank_ic"}.issubset(
        diagnostics["fold_reliability_gate"].columns
    )
    assert {"accepted_rank_ic_std_delta", "next_action"}.issubset(
        diagnostics["fold_reliability_gate_summary"].columns
    )
    assert not diagnostics["regime_threshold_policy_by_fold"].empty
    assert not diagnostics["regime_threshold_policy_summary"].empty
    assert {"f1_delta_vs_official", "selection_guard"}.issubset(
        diagnostics["regime_threshold_policy_by_fold"].columns
    )
    assert {"reviewable", "next_action"}.issubset(
        diagnostics["regime_threshold_policy_summary"].columns
    )
    assert not diagnostics["regime_stability_forensics"].empty
    assert not diagnostics["regime_stability_summary"].empty
    assert "regime" in diagnostics["regime_stability_forensics"].columns
    assert {"suspect_score", "likely_issue"}.issubset(diagnostics["regime_stability_summary"].columns)
    assert not diagnostics["threshold_forensics"].empty
    assert not diagnostics["threshold_policy_review"].empty
    assert not diagnostics["threshold_transfer_review"].empty
    assert not diagnostics["threshold_transfer_by_fold"].empty
    assert not diagnostics["payoff_alignment"].empty
    assert not diagnostics["payoff_alignment_summary"].empty
    assert not diagnostics["payoff_policy_robustness"].empty
    assert not diagnostics["payoff_policy_robustness_summary"].empty
    assert {
        "candidate",
        "candidate_type",
        "cv_phase1_blockers",
        "holdout_blockers",
        "next_action",
        "research_track",
    }.issubset(diagnostics["performance_gap_analysis"].columns)
    assert {
        "blocker",
        "severity",
        "recommended_action",
        "promotion_allowed_now",
        "source_files",
    }.issubset(diagnostics["phase1_blocker_action_plan"].columns)
    assert {"fold_stability", "official_threshold_f1", "future_unseen_oos"}.issubset(
        set(diagnostics["phase1_blocker_action_plan"]["blocker"])
    )
    assert "test_f1_at_official_threshold" in diagnostics["comparison"].columns
    assert "score_gap_pos_minus_neg" in diagnostics["score_separation_forensics"].columns
    assert "likely_signature" in diagnostics["bad_fold_signature"].columns
    assert "bad_fold_signature" in diagnostics["decision"]
    assert "feature_family_drift_summary" in diagnostics["decision"]
    assert "probability_quality_summary" in diagnostics["decision"]
    assert "score_distribution_shift_summary" in diagnostics["decision"]
    assert "fold_reliability_gate_summary" in diagnostics["decision"]
    assert "regime_threshold_policy_summary" in diagnostics["decision"]
    assert "regime_stability_summary" in diagnostics["decision"]
    assert "test_f1_at_official_threshold" in diagnostics["threshold_forensics"].columns
    assert "validation_constrained_threshold" in set(diagnostics["threshold_policy_review"]["policy_name"])
    assert "past_median_constrained_threshold" in set(diagnostics["threshold_transfer_review"]["policy_name"])
    assert "threshold_transfer_review" in diagnostics["decision"]
    assert "profile_calibrated_threshold_summary.csv" in {
        path.name for path in (tmp_path / "reports" / "experiments" / "matrix").iterdir()
    }
    assert {
        "candidate",
        "evaluation_scope",
        "band",
        "label_lift_vs_base",
        "mean_forward_return",
        "mean_tb_return",
        "payoff_alignment_pass",
        "payoff_blockers",
    }.issubset(diagnostics["payoff_alignment"].columns)
    assert "cv_test" in set(diagnostics["payoff_alignment"]["evaluation_scope"])
    assert {
        "candidate",
        "evaluation_scope",
        "band",
        "fold",
        "label_lift_vs_base",
        "mean_forward_return",
        "mean_tb_return",
        "payoff_alignment_pass",
    }.issubset(diagnostics["payoff_policy_robustness"].columns)
    assert {
        "positive_forward_return_fold_rate",
        "positive_tb_return_fold_rate",
        "future_oos_policy_candidate",
        "reject_reason",
        "next_action",
    }.issubset(diagnostics["payoff_policy_robustness_summary"].columns)
    assert set(diagnostics["profile_blend"]["blend_method"]) == {"prob_mean", "rank_mean"}
    assert {
        "reviewable",
        "review_reason",
        "mean_rank_ic_delta_vs_control",
        "balanced_eligible",
        "tail_lift_eligible",
        "stability_eligible",
        "leader_roles",
    }.issubset(diagnostics["profile_blend"].columns)
    assert "best_profile_blend" in diagnostics["decision"]
    assert "profile_blend_leaders" in diagnostics["decision"]
    assert not diagnostics["experiment_selection"].empty
    assert diagnostics["missing_selected_profiles"].empty
    assert "ghost_profile_not_in_run" not in set(diagnostics["experiment_selection"]["profile"])
    assert diagnostics["decision"]["experiment_complete"] is True
    assert (tmp_path / "reports" / "phase1_experiment_bundle_matrix.zip").exists()
    assert (tmp_path / "reports" / "phase1_latest_experiment_bundle.zip").exists()
    assert (tmp_path / "reports" / "phase1_experiment_slim_bundle_matrix.zip").exists()
    assert (tmp_path / "reports" / "phase1_latest_experiment_slim_bundle.zip").exists()
    assert diagnostics["bundle_zip"].endswith("phase1_experiment_bundle_matrix.zip")
    assert diagnostics["latest_bundle_zip"].endswith("phase1_latest_experiment_bundle.zip")
    assert diagnostics["slim_bundle_zip"].endswith("phase1_experiment_slim_bundle_matrix.zip")
    assert diagnostics["latest_slim_bundle_zip"].endswith("phase1_latest_experiment_slim_bundle.zip")
    assert diagnostics["write_full_bundles"] is True
    assert (tmp_path / "reports" / "experiments" / "matrix" / "auto_review.md").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "next_actions.json").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "phase2_readiness.json").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "phase2_readiness.md").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "phase1_transition_plan.json").exists()
    assert (tmp_path / "reports" / "experiments" / "matrix" / "phase1_transition_plan.md").exists()
    assert "phase1_transition_plan" in diagnostics["decision"]
    assert diagnostics["decision"]["recommendation"] in {
        "keep_control_profile",
        "promote_best_candidate",
        "review_profile_blend",
        "rerun_training_with_holdout_split",
        "wait_for_new_unseen_bars_keep_control_profile",
    }
    with zipfile.ZipFile(tmp_path / "reports" / "phase1_experiment_bundle_matrix.zip") as archive:
        assert "matrix/profile_delta_vs_control.csv" in archive.namelist()
        assert "matrix/profile_blend.csv" in archive.namelist()
        assert "matrix/performance_gap_analysis.csv" in archive.namelist()
        assert "matrix/phase1_blocker_action_plan.csv" in archive.namelist()
        assert "matrix/profile_calibrated_threshold_summary.csv" in archive.namelist()
        assert "matrix/fold_stability_forensics.csv" in archive.namelist()
        assert "matrix/fold_stability_summary.csv" in archive.namelist()
        assert "matrix/score_separation_forensics.csv" in archive.namelist()
        assert "matrix/bad_fold_signature.csv" in archive.namelist()
        assert "matrix/feature_drift_forensics.csv" in archive.namelist()
        assert "matrix/feature_family_drift_summary.csv" in archive.namelist()
        assert "matrix/probability_quality_forensics.csv" in archive.namelist()
        assert "matrix/probability_quality_summary.csv" in archive.namelist()
        assert "matrix/score_distribution_shift.csv" in archive.namelist()
        assert "matrix/score_distribution_shift_summary.csv" in archive.namelist()
        assert "matrix/fold_reliability_gate.csv" in archive.namelist()
        assert "matrix/fold_reliability_gate_summary.csv" in archive.namelist()
        assert "matrix/regime_threshold_policy_by_fold.csv" in archive.namelist()
        assert "matrix/regime_threshold_policy_summary.csv" in archive.namelist()
        assert "matrix/regime_stability_forensics.csv" in archive.namelist()
        assert "matrix/regime_stability_summary.csv" in archive.namelist()
        assert "matrix/threshold_forensics.csv" in archive.namelist()
        assert "matrix/threshold_policy_review.csv" in archive.namelist()
        assert "matrix/threshold_transfer_review.csv" in archive.namelist()
        assert "matrix/threshold_transfer_by_fold.csv" in archive.namelist()
        assert "matrix/payoff_alignment.csv" in archive.namelist()
        assert "matrix/payoff_alignment_summary.csv" in archive.namelist()
        assert "matrix/payoff_policy_robustness.csv" in archive.namelist()
        assert "matrix/payoff_policy_robustness_summary.csv" in archive.namelist()
        assert "matrix/profile_fold_metrics.csv" in archive.namelist()
        assert "matrix/experiment_selection.csv" in archive.namelist()
        assert "matrix/missing_selected_profiles.csv" in archive.namelist()
        assert "matrix/holdout_reservation.csv" in archive.namelist()
        assert "matrix/holdout_boundary_audit.csv" in archive.namelist()
        assert "matrix/holdout_policy_consistency.csv" in archive.namelist()
        assert "matrix/holdout_policy_decision.csv" in archive.namelist()
        assert "matrix/frozen_policy_robustness.csv" in archive.namelist()
        assert "matrix/frozen_policy_monitoring_plan.csv" in archive.namelist()
        assert "matrix/future_oos_candidate_plan.csv" in archive.namelist()
        assert "matrix/auto_review.md" in archive.namelist()
        assert "matrix/next_actions.json" in archive.namelist()
        assert "matrix/phase2_readiness.json" in archive.namelist()
        assert "matrix/phase1_transition_plan.json" in archive.namelist()
    with zipfile.ZipFile(tmp_path / "reports" / "phase1_experiment_slim_bundle_matrix.zip") as archive:
        names = set(archive.namelist())
    assert "matrix/profile_comparison.csv" in names
    assert "matrix/performance_gap_analysis.csv" in names
    assert "matrix/phase1_blocker_action_plan.csv" in names
    assert "matrix/profile_calibrated_threshold_summary.csv" in names
    assert "matrix/fold_stability_forensics.csv" in names
    assert "matrix/fold_stability_summary.csv" in names
    assert "matrix/score_separation_forensics.csv" in names
    assert "matrix/bad_fold_signature.csv" in names
    assert "matrix/feature_drift_forensics.csv" in names
    assert "matrix/feature_family_drift_summary.csv" in names
    assert "matrix/probability_quality_forensics.csv" in names
    assert "matrix/probability_quality_summary.csv" in names
    assert "matrix/score_distribution_shift.csv" in names
    assert "matrix/score_distribution_shift_summary.csv" in names
    assert "matrix/fold_reliability_gate.csv" in names
    assert "matrix/fold_reliability_gate_summary.csv" in names
    assert "matrix/regime_threshold_policy_by_fold.csv" in names
    assert "matrix/regime_threshold_policy_summary.csv" in names
    assert "matrix/regime_stability_forensics.csv" in names
    assert "matrix/regime_stability_summary.csv" in names
    assert "matrix/threshold_forensics.csv" in names
    assert "matrix/threshold_policy_review.csv" in names
    assert "matrix/threshold_transfer_review.csv" in names
    assert "matrix/threshold_transfer_by_fold.csv" in names
    assert "matrix/payoff_alignment.csv" in names
    assert "matrix/payoff_alignment_summary.csv" in names
    assert "matrix/payoff_policy_robustness.csv" in names
    assert "matrix/payoff_policy_robustness_summary.csv" in names
    assert "matrix/profile_fold_metrics.csv" in names
    assert "matrix/missing_selected_profiles.csv" in names
    assert "matrix/holdout_reservation.csv" in names
    assert "matrix/holdout_boundary_audit.csv" in names
    assert "matrix/holdout_policy_consistency.csv" in names
    assert "matrix/holdout_policy_decision.csv" in names
    assert "matrix/frozen_policy_robustness.csv" in names
    assert "matrix/frozen_policy_monitoring_plan.csv" in names
    assert "matrix/future_oos_candidate_plan.csv" in names
    assert "matrix/auto_review.md" in names
    assert "matrix/next_actions.json" in names
    assert "matrix/phase2_readiness.json" in names
    assert "matrix/phase1_transition_plan.json" in names
    assert all("/diagnostics/" not in name for name in names)

    slim_only = write_experiment_diagnostics(
        checkpoint_dir=tmp_path,
        config=diagnostics_config,
        output_dir=tmp_path / "slim_reports",
        run_id="matrix",
        write_full_bundles=False,
    )
    assert slim_only["write_full_bundles"] is False
    assert slim_only["zip_paths"] == []
    assert slim_only["bundle_zip"] is None
    assert slim_only["latest_bundle_zip"] is None
    assert (tmp_path / "slim_reports" / "phase1_experiment_slim_bundle_matrix.zip").exists()
    assert not (tmp_path / "slim_reports" / "phase1_experiment_bundle_matrix.zip").exists()
    assert (tmp_path / "slim_reports" / "experiments" / "matrix" / "auto_review.md").exists()
    assert (tmp_path / "slim_reports" / "experiments" / "matrix" / "next_actions.json").exists()
    assert (tmp_path / "slim_reports" / "experiments" / "matrix" / "phase2_readiness.json").exists()
    assert (tmp_path / "slim_reports" / "experiments" / "matrix" / "phase1_transition_plan.json").exists()


def test_experiment_diagnostics_evaluates_reserved_holdout(synthetic_klines, tiny_config, tmp_path) -> None:
    config = copy.deepcopy(tiny_config)
    config["paths"] = {"data_dir": str(tmp_path / "data")}
    config["features"]["profiles"] = {
        "control": {"include_patterns": ["*"], "exclude_patterns": ["4h_taker_buy_ratio"]},
        "candidate": {"include_patterns": ["*"], "exclude_patterns": ["4h_taker_sell_ratio"]},
    }
    config["experiments"] = {
        "mode": "staged",
        "control_profile": "control",
        "candidate_profiles": [],
        "triage_fold_ids": [0],
        "full_cv_profiles": ["control", "candidate"],
        "always_full_profiles": ["control", "candidate"],
        "resume_existing": True,
        "force_retrain": False,
        "profile_blends": {
            "include_auto_equal_weight": True,
            "include_auto_rank_mean": False,
            "weighted": [
                {
                    "name": "control_candidate_65_35",
                    "method": "prob_weighted",
                    "profiles": ["control", "candidate"],
                    "weights": [0.65, 0.35],
                }
            ],
        },
        "policy_review": {
            "enabled": True,
            "frozen_candidate": "blend_control_candidate_65_35",
            "policy_type": "score_band",
            "policy_name": "top_10",
            "status": "score_band_review_only",
            "threshold_deployment_allowed": False,
        },
    }
    frame, _ = _labeled_frame(synthetic_klines, config, periods=260)
    holdout = frame.tail(48).copy().reset_index(drop=True)
    selection = frame.iloc[:-48].copy().reset_index(drop=True)
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    frame.to_parquet(processed_dir / "labeled_1h.parquet", index=False)
    holdout_path = processed_dir / "holdout_1h.parquet"
    holdout.to_parquet(holdout_path, index=False)
    config["experiments"]["holdout"] = {
        "enabled": True,
        "holdout_bars": int(len(holdout)),
        "selection_rows": int(len(selection)),
        "holdout_rows": int(len(holdout)),
        "selection_data_start": str(pd.to_datetime(selection["timestamp"], utc=True).min()),
        "selection_data_end": str(pd.to_datetime(selection["timestamp"], utc=True).max()),
        "holdout_data_start": str(pd.to_datetime(holdout["timestamp"], utc=True).min()),
        "holdout_data_end": str(pd.to_datetime(holdout["timestamp"], utc=True).max()),
        "holdout_path": str(holdout_path),
        "policy": "profile_selection_only_before_holdout",
    }
    config["experiments"]["policy_review"]["future_oos_monitor"] = {
        "enabled": True,
        "anchor_run_id": "old_holdout_run",
        "anchor_data_end": str(pd.to_datetime(selection["timestamp"], utc=True).max()),
        "min_new_bars": 24,
        "preferred_new_bars": 72,
        "policy": "test monitor",
    }

    run_experiment_matrix(selection, config, checkpoint_dir=tmp_path, run_id="holdout_run", device="cpu")
    config["experiments"]["policy_review"]["status"] = "failed_clean_holdout_review"
    diagnostics = write_experiment_diagnostics(
        checkpoint_dir=tmp_path,
        config=config,
        output_dir=tmp_path / "reports",
        run_id="holdout_run",
    )

    holdout_evaluation = diagnostics["holdout_evaluation"]
    assert not holdout_evaluation.empty
    assert diagnostics["decision"]["holdout_evaluation_available"] is True
    assert diagnostics["decision"]["holdout_evaluation"]["available"] is True
    candidates = set(holdout_evaluation["candidate"])
    assert {"control", "candidate", "blend_control_candidate_65_35"}.issubset(candidates)
    assert any(candidate.startswith("blend_prob_mean_") for candidate in candidates)
    assert {
        "holdout_cv_threshold",
        "holdout_cv_threshold_source",
        "holdout_cv_threshold_f1",
        "holdout_cv_threshold_precision",
        "holdout_cv_threshold_recall",
        "holdout_cv_threshold_pred_long_rate",
        "holdout_policy_name",
        "holdout_policy_selection_rate",
        "holdout_policy_forward_return",
        "holdout_policy_pass",
        "cv_policy_name",
        "cv_policy_lift_vs_base",
        "cv_policy_forward_return",
        "holdout_policy_lift_vs_base",
        "holdout_policy_lift_delta_vs_cv",
        "holdout_policy_forward_return_delta_vs_cv",
        "holdout_policy_consistency_pass",
        "holdout_policy_consistency_reject_reason",
        "holdout_signal_pass",
        "holdout_signal_reject_reason",
        "holdout_threshold_pass",
        "holdout_threshold_reject_reason",
        "holdout_soft_pass",
        "holdout_reject_reason",
    }.issubset(holdout_evaluation.columns)
    assert holdout_evaluation["holdout_cv_threshold_source"].eq("cv_constrained_threshold").all()
    assert holdout_evaluation["holdout_cv_threshold_pred_long_rate"].between(0.0, 1.0).all()
    assert (
        holdout_evaluation["holdout_soft_pass"].astype(bool)
        == (
            holdout_evaluation["holdout_signal_pass"].astype(bool)
            & holdout_evaluation["holdout_threshold_pass"].astype(bool)
        )
    ).all()
    assert holdout_evaluation["mtf_leakage_passed"].all()
    assert "holdout" in set(diagnostics["payoff_alignment"]["evaluation_scope"])
    assert "holdout" in set(diagnostics["payoff_policy_robustness"]["evaluation_scope"])
    assert {
        "top_10_label_lift_vs_base",
        "top_10_mean_forward_return",
        "top_10_payoff_blockers",
        "best_forward_return_band",
        "next_action",
    }.issubset(diagnostics["payoff_alignment_summary"].columns)
    assert diagnostics["decision"]["holdout_boundary_passed"] is True
    assert diagnostics["holdout_boundary_audit"]["passed"].astype(bool).all()
    assert (tmp_path / "reports" / "experiments" / "holdout_run" / "holdout_evaluation.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "holdout_run" / "holdout_score_band_summary.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "holdout_run" / "holdout_threshold_summary.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "holdout_run" / "holdout_policy_evaluation.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "holdout_run" / "holdout_policy_consistency.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "holdout_run" / "holdout_policy_decision.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "holdout_run" / "holdout_boundary_audit.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "holdout_run" / "frozen_policy_robustness.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "holdout_run" / "frozen_policy_monitoring_plan.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "holdout_run" / "performance_gap_analysis.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "holdout_run" / "payoff_alignment.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "holdout_run" / "payoff_alignment_summary.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "holdout_run" / "payoff_policy_robustness.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "holdout_run" / "payoff_policy_robustness_summary.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "holdout_run" / "profile_score_policy_grid.csv").exists()
    assert (tmp_path / "reports" / "experiments" / "holdout_run" / "profile_score_policy_selection.csv").exists()
    assert "frozen_policy_validation" in diagnostics["decision"]["holdout_evaluation"]
    assert "observed_best_policy_candidate" in diagnostics["decision"]["holdout_evaluation"]
    assert "policy_validation" in diagnostics["decision"]["holdout_evaluation"]
    policy_validation = diagnostics["decision"]["holdout_evaluation"]["policy_validation"]
    assert "configured_policy_match" in policy_validation
    assert "threshold_deployment_blocked_by_policy" in policy_validation
    assert "holdout_boundary_passed" in policy_validation
    assert policy_validation["frozen_selection"] == "blend_control_candidate_65_35"
    assert policy_validation["frozen_selection_source"] == "configured_policy_review"
    assert policy_validation["configured_frozen_candidate_available"] is True
    assert policy_validation["configured_status"] == "failed_clean_holdout_review"
    assert policy_validation["policy_action"] == "retired_frozen_policy_keep_control_profile"
    monitoring = diagnostics["frozen_policy_monitoring_plan"].iloc[0].to_dict()
    assert monitoring["status"] == "failed_clean_holdout_review"
    assert monitoring["anchor_run_id"] == "old_holdout_run"
    assert monitoring["new_bars_since_anchor"] >= 47
    assert monitoring["min_new_bars_remaining"] == 0
    assert monitoring["future_oos_ready"] is True
    assert monitoring["allow_holdout_roll_forward"] is False
    assert monitoring["holdout_roll_forward_locked"] is True
    assert monitoring["next_action"] == "future_oos_window_available"
    assert policy_validation["policy_action"] in {
        "review_frozen_threshold_and_score_policy",
        "review_frozen_score_band_policy_only_no_threshold_deployment",
        "holdout_only_candidate_do_not_promote_without_future_oos",
        "keep_control_profile",
        "invalid_holdout_training_boundary_rerun_04",
        "retired_frozen_policy_keep_control_profile",
    }
    assert diagnostics["decision"]["holdout_evaluation"]["score_policy_recommendation"] in {
        "review_frozen_score_band_policy",
        "holdout_only_diagnostic_policy_candidate",
        "keep_control_profile",
    }
    with zipfile.ZipFile(tmp_path / "reports" / "phase1_experiment_slim_bundle_holdout_run.zip") as archive:
        names = set(archive.namelist())
    assert "holdout_run/holdout_evaluation.csv" in names
    assert "holdout_run/holdout_score_band_summary.csv" in names
    assert "holdout_run/holdout_threshold_summary.csv" in names
    assert "holdout_run/holdout_policy_consistency.csv" in names
    assert "holdout_run/holdout_policy_decision.csv" in names
    assert "holdout_run/holdout_policy_evaluation.csv" in names
    assert "holdout_run/holdout_boundary_audit.csv" in names
    assert "holdout_run/frozen_policy_robustness.csv" in names
    assert "holdout_run/frozen_policy_monitoring_plan.csv" in names
    assert "holdout_run/performance_gap_analysis.csv" in names
    assert "holdout_run/payoff_alignment.csv" in names
    assert "holdout_run/payoff_alignment_summary.csv" in names
    assert "holdout_run/payoff_policy_robustness.csv" in names
    assert "holdout_run/payoff_policy_robustness_summary.csv" in names
    assert "holdout_run/profile_score_policy_grid.csv" in names
    assert "holdout_run/profile_score_policy_selection.csv" in names


def test_experiment_diagnostics_recovers_standard_holdout_when_manifest_lacks_metadata(
    synthetic_klines,
    tiny_config,
    tmp_path,
) -> None:
    config = copy.deepcopy(tiny_config)
    config["paths"] = {"data_dir": str(tmp_path / "data")}
    config["features"]["profiles"] = {
        "control": {"include_patterns": ["*"], "exclude_patterns": ["4h_taker_buy_ratio"]},
    }
    config["experiments"] = {
        "mode": "staged",
        "control_profile": "control",
        "candidate_profiles": [],
        "triage_fold_ids": [0],
        "full_cv_profiles": ["control"],
        "always_full_profiles": ["control"],
        "resume_existing": True,
        "force_retrain": False,
        "holdout": {
            "enabled": True,
            "holdout_bars": 48,
            "holdout_filename": "holdout_1h.parquet",
            "policy": "profile_selection_only_before_holdout",
        },
    }
    frame, _ = _labeled_frame(synthetic_klines, config, periods=260)
    holdout = frame.tail(48).copy().reset_index(drop=True)
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    frame.to_parquet(processed_dir / "labeled_1h.parquet", index=False)
    holdout.to_parquet(processed_dir / "holdout_1h.parquet", index=False)
    holdout_start = pd.to_datetime(holdout["timestamp"], utc=True).min()

    result = run_experiment_matrix(frame, config, checkpoint_dir=tmp_path, run_id="auto_holdout", device="cpu")
    full_rows = result["comparison"].loc[result["comparison"]["fold_scope"].eq("full")]
    assert not full_rows.empty
    assert (pd.to_datetime(full_rows["data_end"], utc=True) < holdout_start).all()

    manifest_path = tmp_path / "experiments" / "auto_holdout" / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["settings"].pop("holdout", None)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    diagnostics = write_experiment_diagnostics(
        checkpoint_dir=tmp_path,
        config=config,
        output_dir=tmp_path / "reports",
        run_id="auto_holdout",
    )

    holdout_evaluation = diagnostics["holdout_evaluation"]
    assert not holdout_evaluation.empty
    assert diagnostics["decision"]["holdout_evaluation_available"] is True
    assert diagnostics["decision"]["holdout_evaluation"]["available"] is True
    reservation = diagnostics["holdout_reservation"]
    assert reservation.loc[0, "holdout_rows"] == 48
    assert str(reservation.loc[0, "holdout_path"]).endswith("holdout_1h.parquet")


def test_seed_audit_writes_isolated_seed_summaries(synthetic_klines, tiny_config, tmp_path) -> None:
    config = copy.deepcopy(tiny_config)
    config["features"]["profiles"] = {
        "control": {"include_patterns": ["*"], "exclude_patterns": []},
    }
    config["experiments"] = {
        "mode": "staged",
        "control_profile": "control",
        "candidate_profiles": [],
        "triage_fold_ids": [0],
        "full_cv_profiles": [],
        "resume_existing": True,
        "force_retrain": False,
        "seed_audit": {
            "enabled": True,
            "profiles": ["control"],
            "seeds": [11, 12],
            "fold_ids": [0],
        },
    }
    frame, _ = _labeled_frame(synthetic_klines, config, periods=220)

    result = run_experiment_matrix(frame, config, checkpoint_dir=tmp_path, run_id="seeded", device="cpu")
    diagnostics = write_experiment_diagnostics(
        checkpoint_dir=tmp_path,
        config=config,
        output_dir=tmp_path / "reports",
        run_id="seeded",
        write_full_bundles=True,
    )

    assert set(result["seed_audit"]["seed"]) == {11, 12}
    assert result["seed_stability"].loc[0, "seed_count"] == 2
    assert (tmp_path / "experiments" / "seeded" / "seed_audit.csv").exists()
    assert (tmp_path / "experiments" / "seeded" / "seed_stability.csv").exists()
    assert (tmp_path / "experiments" / "seeded" / "seed_ensemble.csv").exists()
    assert not result["seed_ensemble"].empty
    assert result["seed_ensemble"].loc[0, "seed_count"] == 2
    assert not diagnostics["seed_audit"].empty
    assert not diagnostics["seed_stability"].empty
    assert not diagnostics["seed_ensemble"].empty
    with zipfile.ZipFile(tmp_path / "reports" / "phase1_experiment_bundle_seeded.zip") as archive:
        names = set(archive.namelist())
    assert "seeded/seed_audit.csv" in names
    assert "seeded/seed_stability.csv" in names
    assert "seeded/seed_ensemble.csv" in names


def test_experiment_run_id_reuses_latest_matching_signature(synthetic_klines, tiny_config, tmp_path) -> None:
    config = copy.deepcopy(tiny_config)
    config["features"]["profiles"] = {
        "control": {"include_patterns": ["*"], "exclude_patterns": []},
    }
    config["experiments"] = {
        "mode": "staged",
        "control_profile": "control",
        "candidate_profiles": [],
        "triage_fold_ids": [0],
        "full_cv_profiles": [],
        "resume_existing": True,
        "force_retrain": False,
    }
    frame, _ = _labeled_frame(synthetic_klines, config, periods=220)

    first = run_experiment_matrix(frame, config, checkpoint_dir=tmp_path, run_id="stable_run", device="cpu")
    run_id, source = resolve_experiment_run_id(tmp_path, config)
    second = run_experiment_matrix(frame, config, checkpoint_dir=tmp_path, device="cpu")

    assert first["run_id"] == "stable_run"
    assert run_id == "stable_run"
    assert source == "matching_existing"
    assert second["run_id"] == "stable_run"
    assert second["run_id_source"] == "matching_existing"
    assert second["training_executed_count"] == 0
    assert second["training_skipped_count"] > 0
    assert second["all_training_scopes_reused"] is True
    assert second["decision"]["all_training_scopes_reused"] is True
    execution_summary_path = tmp_path / "experiments" / "stable_run" / "training_execution_summary.json"
    assert execution_summary_path.exists()
    execution_summary = json.loads(execution_summary_path.read_text(encoding="utf-8"))
    assert execution_summary["run_id_source"] == "matching_existing"
    assert execution_summary["training_executed_count"] == 0
    assert execution_summary["training_skipped_count"] > 0

    diagnostics = write_experiment_diagnostics(
        checkpoint_dir=tmp_path,
        config=config,
        output_dir=tmp_path / "reports",
        run_id="stable_run",
    )
    assert diagnostics["decision"]["run_id_source"] == "matching_existing"
    assert diagnostics["decision"]["training_executed_count"] == 0
    assert diagnostics["decision"]["training_skipped_count"] > 0
    assert diagnostics["decision"]["all_training_scopes_reused"] is True
    assert diagnostics["decision"]["training_execution_metadata_source"] == "training_execution_summary"
    assert diagnostics["decision"]["training_execution_metadata_available"] is True
    assert diagnostics["experiment_policy_guard"].loc[0, "action"] == "normal_experiment_flow"
    assert not diagnostics["future_oos_candidate_plan"].empty
    with zipfile.ZipFile(tmp_path / "reports" / "phase1_experiment_slim_bundle_stable_run.zip") as archive:
        names = set(archive.namelist())
    assert "stable_run/training_execution_summary.json" in names
    assert "stable_run/experiment_policy_guard.csv" in names
    assert "stable_run/future_oos_candidate_plan.csv" in names


def test_write_experiment_diagnostics_raises_when_run_has_no_completed_profiles(tmp_path, tiny_config) -> None:
    config = copy.deepcopy(tiny_config)
    config["features"]["profiles"] = {"control": {"include_patterns": ["*"], "exclude_patterns": []}}
    config["experiments"] = {
        "mode": "staged",
        "control_profile": "control",
        "candidate_profiles": [],
        "triage_fold_ids": [0],
        "full_cv_profiles": [],
        "resume_existing": True,
        "force_retrain": False,
    }

    # Create an empty run directory (e.g. interrupted training that wrote no manifests/predictions).
    run_dir = tmp_path / "experiments" / "empty_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="No completed profile runs found"):
        write_experiment_diagnostics(
            checkpoint_dir=tmp_path,
            config=config,
            output_dir=tmp_path / "reports",
            run_id="empty_run",
        )
