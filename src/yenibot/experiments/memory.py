from __future__ import annotations

from __future__ import annotations
from fnmatch import fnmatch
from typing import Any


def _profile_rejection_reason(profile: str, experiments: dict[str, Any]) -> str:
    memory = experiments.get("experiment_memory", {}) or {}
    if not bool(memory.get("enabled", False)) or not bool(memory.get("reject_retests", True)):
        return ""
    if profile in {str(item) for item in memory.get("allow_retest_profiles", []) or []}:
        return ""

    rejected_profiles = memory.get("rejected_profiles", {}) or {}
    if isinstance(rejected_profiles, dict) and profile in rejected_profiles:
        value = rejected_profiles[profile]
        if isinstance(value, dict):
            return str(value.get("reason") or "historically_rejected_profile")
        return str(value or "historically_rejected_profile")

    for item in memory.get("rejected_profile_patterns", []) or []:
        if isinstance(item, str):
            pattern = item
            reason = "historically_rejected_profile_pattern"
        elif isinstance(item, dict):
            pattern = str(item.get("pattern", ""))
            reason = str(item.get("reason") or "historically_rejected_profile_pattern")
        else:
            continue
        if pattern and fnmatch(profile, pattern):
            return reason
    return ""

