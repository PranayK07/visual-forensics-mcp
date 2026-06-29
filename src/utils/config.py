"""Configuration loading and merging.

The configuration is a plain nested dict loaded from ``configs/default.yaml``.
Callers may override any subset of values at request time. There are no
hardcoded thresholds anywhere else in the codebase; every tunable lives in the
YAML file and flows through this module.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``.

    Dict values are merged key-by-key; all other values (including lists) are
    replaced wholesale by the override.
    """
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class Config:
    """Read-only-ish wrapper around the nested config dict with dotted access."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Fetch a nested value using a ``"a.b.c"`` dotted path."""
        node: Any = self._data
        for part in dotted_key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def section(self, name: str) -> dict[str, Any]:
        value = self._data.get(name, {})
        return value if isinstance(value, dict) else {}

    def with_overrides(self, overrides: dict[str, Any] | None) -> "Config":
        if not overrides:
            return Config(copy.deepcopy(self._data))
        return Config(deep_merge(self._data, overrides))

    def digest(self) -> dict[str, Any]:
        """A compact summary of the most relevant settings for the response."""
        return {
            "dpi": self.get("render.dpi"),
            "tile_size": self.get("tiling.tile_size"),
            "tile_overlap": self.get("tiling.tile_overlap"),
            "enable_ocr": self.get("features.enable_ocr"),
            "enable_pdf_analysis": self.get("features.enable_pdf_analysis"),
        }


def load_config(
    path: str | os.PathLike[str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Load configuration from YAML and apply optional overrides.

    Resolution order for the base file:

    1. The explicit ``path`` argument.
    2. The ``VISUAL_FORENSICS_CONFIG`` environment variable.
    3. The packaged ``configs/default.yaml``.
    """
    if path is None:
        env_path = os.environ.get("VISUAL_FORENSICS_CONFIG")
        path = env_path if env_path else _DEFAULT_CONFIG_PATH

    with open(path, "r", encoding="utf-8") as fh:
        base = yaml.safe_load(fh) or {}

    if not isinstance(base, dict):
        raise ValueError(f"Config root must be a mapping, got {type(base)!r}")

    merged = deep_merge(base, overrides or {})
    return Config(merged)
