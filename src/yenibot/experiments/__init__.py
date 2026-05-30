"""Experiment orchestration for Phase 1 research.

This package manages multi-profile experiment runs including triage,
full walk-forward CV, seed auditing, holdout evaluation, profile
blending, and automated report generation.

The monolith ``_monolith.py`` module contains the full original
implementation.  Public APIs are re-exported here so callers can use::

    from yenibot.experiments import run_experiment_matrix, experiment_settings

A future refactoring pass will decompose ``_monolith`` into focused
sub-modules (runner, triage, blend, holdout, report_writer, memory).
"""

from yenibot.experiments._monolith import (  # noqa: F401
    experiment_root,
    experiment_settings,
    latest_experiment_run,
    new_run_id,
    profile_config,
    profile_run_dir,
    resolve_experiment_run_id,
    run_experiment_matrix,
    write_experiment_diagnostics,
)

__all__ = [
    "experiment_root",
    "experiment_settings",
    "latest_experiment_run",
    "new_run_id",
    "profile_config",
    "profile_run_dir",
    "resolve_experiment_run_id",
    "run_experiment_matrix",
    "write_experiment_diagnostics",
]
