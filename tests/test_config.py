"""Configuration source/package consistency tests."""

from pathlib import Path

import yaml

from src.utils.config import load_config


def test_packaged_and_editable_default_configs_match():
    root = Path(__file__).resolve().parents[1]
    with (root / "configs" / "default.yaml").open(encoding="utf-8") as stream:
        editable = yaml.safe_load(stream)
    with (root / "src" / "configs" / "default.yaml").open(encoding="utf-8") as stream:
        packaged = yaml.safe_load(stream)

    assert editable == packaged
    assert load_config().data == editable
