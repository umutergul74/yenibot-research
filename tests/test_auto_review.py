from __future__ import annotations

import json

import pandas as pd

from yenibot.automation import review_experiment_report, write_auto_review


def _write_minimal_report(path, *, missing_selected: bool = False, future_oos_ready: bool = False) -> None:
    path.mkdir(parents=True, exist_ok=True)
    control = "control_profile"
    challenger = "candidate_profile"
    pd.DataFrame(
        [
            {
                "profile": control,
                "fold_scope": "full",
                "mean_rank_ic": 0.05,
                "std_rank_ic": 0.07,
                "positive_ic_fraction": 0.80,
                "mean_long_f1": 0.31,
                "test_f1_at_selected_threshold": 0.47,
                "test_f1_at_constrained_threshold": 0.46,
                "calibration_separation": 0.01,
                "top_10_lift_global": 1.10,
                "mtf_leakage_passed": True,
                "stationarity_policy_passed": True,
            },
            {
                "profile": challenger,
                "fold_scope": "full",
                "mean_rank_ic": 0.052,
                "std_rank_ic": 0.09,
                "positive_ic_fraction": 0.70,
                "mean_long_f1": 0.30,
                "test_f1_at_selected_threshold": 0.45,
                "test_f1_at_constrained_threshold": 0.44,
                "calibration_separation": 0.01,
                "top_10_lift_global": 1.14,
                "mtf_leakage_passed": True,
                "stationarity_policy_passed": True,
            },
        ]
    ).to_csv(path / "profile_comparison.csv", index=False)
    pd.DataFrame(
        [
            {
                "profile": "blend_prob_mean",
                "fold_scope": "blend_full",
                "mean_rank_ic": 0.055,
                "std_rank_ic": 0.075,
                "positive_ic_fraction": 0.78,
                "top_10_lift_global": 1.15,
            }
        ]
    ).to_csv(path / "profile_blend.csv", index=False)
    pd.DataFrame(
        [
            {
                "profile": challenger,
                "fold_scope": "holdout_profile",
                "mean_rank_ic": 0.06,
                "top_10_lift_global": 1.12,
                "top_10_forward_return_global": 0.002,
                "holdout_signal_pass": True,
            }
        ]
    ).to_csv(path / "holdout_evaluation.csv", index=False)
    pd.DataFrame(
        [
            {
                "plan_rank": 1,
                "candidate": challenger,
                "candidate_label": f"{challenger} [top_10]",
                "candidate_type": "profile",
                "stage": "future_oos_score_band_policy",
                "policy_name": "top_10",
                "future_oos_priority_score": 0.7,
                "cv_mean_rank_ic": 0.052,
                "holdout_mean_rank_ic": 0.06,
            }
        ]
    ).to_csv(path / "future_oos_candidate_plan.csv", index=False)
    pd.DataFrame(
        [
            {
                "status": "failed_clean_holdout_review",
                "action": "wait_for_new_unseen_bars_keep_control_profile",
                "future_oos_ready": future_oos_ready,
                "future_oos_preferred_ready": False,
                "new_bars_since_anchor": 250,
                "min_new_bars_remaining": 470,
                "preferred_new_bars_remaining": 1900,
                "min_ready_at": "2026-06-12 08:00:00+00:00",
                "preferred_ready_at": "2026-08-11 08:00:00+00:00",
                "holdout_roll_forward_locked": True,
            }
        ]
    ).to_csv(path / "experiment_policy_guard.csv", index=False)
    pd.DataFrame(
        [
            {
                "selected": True,
                "profile": control,
                "fold_scope": "full",
            }
        ]
    ).to_csv(path / "experiment_selection.csv", index=False)
    missing = pd.DataFrame(
        [{"profile": "missing_profile", "fold_scope": "full"}]
        if missing_selected
        else [],
        columns=["profile", "fold_scope"],
    )
    missing.to_csv(path / "missing_selected_profiles.csv", index=False)
    (path / "decision_report.json").write_text(
        json.dumps(
            {
                "run_id": "test_run",
                "control_profile": control,
                "holdout_boundary_passed": True,
                "recommendation": "keep_control_profile",
            }
        ),
        encoding="utf-8",
    )
    (path / "training_execution_summary.json").write_text(
        json.dumps(
            {
                "training_executed_count": 2,
                "training_skipped_count": 0,
                "all_training_scopes_reused": False,
            }
        ),
        encoding="utf-8",
    )


def test_auto_review_waits_for_future_oos_when_no_cv_candidate(tmp_path) -> None:
    _write_minimal_report(tmp_path)

    review = review_experiment_report(tmp_path)

    assert review["report_completeness"]["complete"] is True
    assert review["next_action"]["action"] == "wait_for_new_unseen_bars_keep_control"
    assert review["next_action"]["do_not_promote_from_current_holdout"] is True
    assert review["cv"]["control"]["profile"] == "control_profile"
    assert review["future_oos"]["best_candidate_plan_row"]["candidate_label"] == "candidate_profile [top_10]"
    assert review["phase2_readiness"]["ready_for_phase2"] is False
    assert "rank_ic_std_above_phase1_target" in review["phase2_readiness"]["blockers"]
    assert "long_f1_below_phase1_target" not in review["phase2_readiness"]["blockers"]
    assert review["phase2_readiness"]["long_f1_source"] == "validation_selected_threshold"
    assert "fixed_0_50_f1_below_target_calibration_issue" in review["phase2_readiness"]["advisories"]
    assert review["phase1_transition_plan"]["decision"] == "PHASE1_RESEARCH_READY_PHASE2_BLOCKED"
    assert "do_not_start_phase2_backtest" in review["phase1_transition_plan"]["blocked_actions"]


def test_auto_review_uses_guarded_f1_when_selected_threshold_is_too_broad(tmp_path) -> None:
    _write_minimal_report(tmp_path)
    comparison = pd.read_csv(tmp_path / "profile_comparison.csv")
    comparison.loc[comparison["profile"] == "control_profile", "test_pred_long_rate_at_selected_threshold"] = 0.86
    comparison.loc[comparison["profile"] == "control_profile", "test_pred_long_rate_at_constrained_threshold"] = 0.64
    comparison.loc[comparison["profile"] == "control_profile", "test_f1_at_constrained_threshold"] = 0.43
    comparison.to_csv(tmp_path / "profile_comparison.csv", index=False)

    review = review_experiment_report(tmp_path)

    assert review["phase2_readiness"]["long_f1_source"] == "validation_constrained_threshold"
    assert "long_f1_below_phase1_target" in review["phase2_readiness"]["blockers"]


def test_auto_review_uses_official_calibrated_threshold_when_available(tmp_path) -> None:
    _write_minimal_report(tmp_path)
    comparison = pd.read_csv(tmp_path / "profile_comparison.csv")
    mask = comparison["profile"] == "control_profile"
    comparison.loc[mask, "test_f1_at_guarded_threshold"] = 0.43
    comparison.loc[mask, "test_pred_long_rate_at_guarded_threshold"] = 0.64
    comparison.loc[mask, "guarded_threshold_source"] = "validation_constrained_threshold"
    comparison.loc[mask, "test_f1_at_official_threshold"] = 0.455
    comparison.loc[mask, "test_pred_long_rate_at_official_threshold"] = 0.63
    comparison.loc[mask, "official_threshold_source"] = "calibrated_validation_constrained_threshold"
    comparison.loc[mask, "official_threshold_uses_calibration"] = True
    comparison.to_csv(tmp_path / "profile_comparison.csv", index=False)

    review = review_experiment_report(tmp_path)

    assert review["phase2_readiness"]["long_f1_source"] == "calibrated_validation_constrained_threshold"
    assert "long_f1_below_phase1_target" not in review["phase2_readiness"]["blockers"]


def test_auto_review_flags_missing_selected_profiles(tmp_path) -> None:
    _write_minimal_report(tmp_path, missing_selected=True)

    review = review_experiment_report(tmp_path)

    assert review["report_completeness"]["complete"] is False
    assert review["next_action"]["action"] == "fix_missing_selected_profiles"


def test_write_auto_review_outputs_files(tmp_path) -> None:
    _write_minimal_report(tmp_path)

    result = write_auto_review(tmp_path)

    assert (tmp_path / "auto_review.md").exists()
    assert (tmp_path / "auto_review.json").exists()
    assert (tmp_path / "next_actions.json").exists()
    assert (tmp_path / "phase2_readiness.json").exists()
    assert (tmp_path / "phase2_readiness.md").exists()
    assert (tmp_path / "phase1_transition_plan.json").exists()
    assert (tmp_path / "phase1_transition_plan.md").exists()
    next_actions = json.loads((tmp_path / "next_actions.json").read_text(encoding="utf-8"))
    assert next_actions["action"] == "wait_for_new_unseen_bars_keep_control"
    phase2 = json.loads((tmp_path / "phase2_readiness.json").read_text(encoding="utf-8"))
    assert phase2["decision"] == "DO_NOT_PROCEED_TO_PHASE2"
    assert phase2["long_f1_source"] == "validation_selected_threshold"
    transition = json.loads((tmp_path / "phase1_transition_plan.json").read_text(encoding="utf-8"))
    assert transition["decision"] == "PHASE1_RESEARCH_READY_PHASE2_BLOCKED"
    assert transition["long_f1_source"] == "validation_selected_threshold"
    assert result["auto_review_path"].endswith("auto_review.md")
