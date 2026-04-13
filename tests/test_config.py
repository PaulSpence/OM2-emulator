"""Tests for configuration parsing and validation."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from om2_experiment_config import (  # noqa: E402
    deep_merge,
    load_yaml_config,
    resolve_config,
)


class TestExperimentConfig(unittest.TestCase):
    """Validate baseline config behavior and validation checks."""

    def test_deep_merge_overrides_nested_keys(self) -> None:
        """Nested dictionaries should be recursively overridden."""
        base = {"a": {"b": 1, "c": 2}, "x": 4}
        overrides = {"a": {"b": 10}, "x": 5}
        merged = deep_merge(base, overrides)

        self.assertEqual(merged["a"]["b"], 10)
        self.assertEqual(merged["a"]["c"], 2)
        self.assertEqual(merged["x"], 5)

    def test_resolve_config_accepts_baseline(self) -> None:
        """Repository baseline config should resolve without error."""
        baseline = load_yaml_config(REPO_ROOT / "configs" / "baseline.yaml")
        resolved = resolve_config(baseline)

        self.assertIn("model", resolved)
        self.assertEqual(resolved["normalization"]["strategy"], "masked_zscore")

    def test_invalid_normalization_strategy_raises(self) -> None:
        """Unsupported normalization strategies must fail fast."""
        baseline = load_yaml_config(REPO_ROOT / "configs" / "baseline.yaml")
        baseline["normalization"]["strategy"] = "not_real"

        with self.assertRaises(ValueError):
            resolve_config(baseline)

    def test_yaml_round_trip_loads(self) -> None:
        """A minimal YAML file should parse and keep expected scalar values."""
        yaml_text = """
run:
  name: tiny
normalization:
  strategy: masked_zscore
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tiny.yaml"
            path.write_text(yaml_text, encoding="utf-8")
            loaded = load_yaml_config(path)

        self.assertEqual(loaded["run"]["name"], "tiny")
        self.assertEqual(loaded["normalization"]["strategy"], "masked_zscore")


if __name__ == "__main__":
    unittest.main()
