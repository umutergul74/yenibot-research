from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigNode(dict):
    """Dictionary with attribute access for YAML configuration."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _to_node(value: Any) -> Any:
    if isinstance(value, Mapping):
        return ConfigNode({key: _to_node(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_node(item) for item in value]
    return value


@dataclass(frozen=True)
class LoadedConfig:
    path: Path
    data: ConfigNode


def load_config(path: str | Path = "config.yaml") -> ConfigNode:
    """Load project configuration from YAML.

    Supports loading a single YAML file or a directory of YAML files
    that are merged together (base.yaml first, then profiles).
    """
    config_path = Path(path)

    if config_path.is_dir():
        return _load_config_dir(config_path)

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    return _to_node(raw)


def _load_config_dir(config_dir: Path) -> ConfigNode:
    """Load and merge all YAML files in a config directory.

    Loads base.yaml first, then merges profiles/ directory files on top.
    """
    base_path = config_dir / "base.yaml"
    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")

    with base_path.open("r", encoding="utf-8") as handle:
        merged = yaml.safe_load(handle) or {}

    profiles_dir = config_dir / "profiles"
    if profiles_dir.is_dir():
        for profile_file in sorted(profiles_dir.glob("*.yaml")):
            with profile_file.open("r", encoding="utf-8") as handle:
                profile_data = yaml.safe_load(handle) or {}
            _deep_merge(merged, profile_data)

    return _to_node(merged)


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay dict into base dict."""
    for key, value in overlay.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def repo_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[2]
