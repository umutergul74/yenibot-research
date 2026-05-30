from __future__ import annotations

from __future__ import annotations
import copy
from typing import Any
from yenibot.features import resolve_feature_profile

from .utils import _deep_update, _set_cfg, _cfg

def profile_config(config: dict[str, Any], profile: str) -> dict[str, Any]:
    """Return an in-memory config copy with the requested active feature profile."""

    updated = copy.deepcopy(config)
    _set_cfg(updated, ["features", "active_profile"], profile)
    resolve_feature_profile(updated)
    overrides = _profile_config_overrides(updated, profile)
    if overrides:
        _deep_update(updated, overrides)
    return updated

def _profile_config_overrides(config: dict[str, Any], profile: str) -> dict[str, Any]:
    profiles = _cfg(config, ["features", "profiles"], {}) or {}
    if not isinstance(profiles, dict):
        return {}

    def load(name: str, seen: set[str] | None = None) -> dict[str, Any]:
        seen = set() if seen is None else seen
        if name in seen:
            raise ValueError(f"Cyclic feature profile inheritance detected at {name}")
        seen.add(name)
        current = profiles.get(name)
        if not isinstance(current, dict):
            return {}
        parent_name = current.get("inherit")
        overrides = load(str(parent_name), seen) if parent_name else {}
        current_overrides = current.get("config_overrides", current.get("training_overrides", {})) or {}
        if not isinstance(current_overrides, dict):
            raise ValueError(f"Feature profile config_overrides must be a mapping: {name}")
        return _deep_update(overrides, current_overrides)

    return load(str(profile))

