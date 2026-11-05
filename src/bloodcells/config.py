"""Load ``config.yaml`` and resolve project paths."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Return the config dict; entries under ``paths`` become absolute Paths."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {key: (PROJECT_ROOT / value).resolve() for key, value in cfg.get("paths", {}).items()}
    return cfg


def parse_overrides(spec: str) -> dict[tuple[str, str], object]:
    """'section.key=value;section.key=value' -> {(section, key): value} (values parsed as YAML)."""
    overrides = {}
    for item in filter(None, (s.strip() for s in spec.split(";"))):
        key, _, value = item.partition("=")
        section, _, param = key.strip().partition(".")
        overrides[(section, param)] = yaml.safe_load(value)
    return overrides


def with_overrides(cfg: dict, overrides: dict[tuple[str, str], object]) -> dict:
    """Deep copy of ``cfg`` with some ``section.key`` values replaced (unknown keys raise KeyError)."""
    out = copy.deepcopy(cfg)
    for (section, key), value in overrides.items():
        if section not in out or key not in out[section]:
            raise KeyError(f"unknown parameter {section}.{key}")
        out[section][key] = value
    return out
