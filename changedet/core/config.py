"""Configuration: ``configs/default.yaml`` + ``--set section.key=value`` overrides."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from changedet.core.cache import run_path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
RUN_CONFIG_RELPATH = "config.yaml"


class Config(dict):
    """Nested dict with attribute access: ``cfg.frames.fps == cfg["frames"]["fps"]``."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(f"config has no key '{name}'") from e

    @classmethod
    def from_dict(cls, data: dict) -> Config:
        return cls({k: cls.from_dict(v) if isinstance(v, dict) else v for k, v in data.items()})

    def to_dict(self) -> dict:
        return {k: v.to_dict() if isinstance(v, Config) else v for k, v in self.items()}


def apply_overrides(data: dict, overrides: Iterable[str]) -> dict:
    """Return a copy of ``data`` with ``"section.key=value"`` overrides applied.

    Values are parsed as YAML (``0.1`` → float, ``true`` → bool, ``[a, b]`` → list).
    Keys must already exist, so typos fail loudly instead of being silently ignored.
    """
    data = copy.deepcopy(data)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like section.key=value")
        dotted, raw = item.split("=", 1)
        keys = dotted.strip().split(".")
        node = data
        for i, key in enumerate(keys):
            if not isinstance(node, dict) or key not in node:
                raise KeyError(f"Unknown config key '{'.'.join(keys[: i + 1])}' in '{item}'")
            if i == len(keys) - 1:
                node[key] = yaml.safe_load(raw)
            else:
                node = node[key]
    return data


def load_config(path: str | Path | None = None, overrides: Iterable[str] = ()) -> Config:
    """Load a YAML config (default: ``configs/default.yaml``) and apply overrides."""
    data = yaml.safe_load(Path(path or DEFAULT_CONFIG_PATH).read_text()) or {}
    return Config.from_dict(apply_overrides(data, overrides))


def save_config(cfg: Config | dict, path: str | Path) -> Path:
    """Write a config snapshot as YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = cfg.to_dict() if isinstance(cfg, Config) else cfg
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def deep_merge(base: dict, override: dict) -> dict:
    """Copy of ``base`` with ``override``'s values on top, merging nested dicts key by key."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_run_config(run: str, overrides: Iterable[str] = ()) -> Config:
    """Load a run's config: its snapshot (``runs/<run>/config.yaml``) on top of the current
    defaults, so keys added to ``configs/default.yaml`` later are still available, then
    overrides."""
    path = run_path(run, RUN_CONFIG_RELPATH)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — create the run first with `init`")
    defaults = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text()) or {}
    snapshot = yaml.safe_load(path.read_text()) or {}
    return Config.from_dict(apply_overrides(deep_merge(defaults, snapshot), overrides))
