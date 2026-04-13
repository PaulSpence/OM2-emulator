"""Tests for one-factor ablation variant generation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from om2_experiment_ablation import generate_ablation_variants, get_nested_path  # noqa: E402
from om2_experiment_config import load_yaml_config, resolve_config  # noqa: E402


class TestAblationMatrix(unittest.TestCase):
    """Ensure matrix variants are generated and modified correctly."""

    def test_generate_variants_changes_targeted_paths(self) -> None:
        """Each variant should alter at least one declared factor path."""
        baseline = resolve_config(load_yaml_config(REPO_ROOT / "configs" / "baseline.yaml"))
        matrix = load_yaml_config(REPO_ROOT / "configs" / "ablation_matrix.yaml")

        variants = generate_ablation_variants(baseline, matrix)
        self.assertGreater(len(variants), 0)

        factor_paths = list(matrix["factors"].keys())
        for _, variant in variants:
            changed = any(get_nested_path(variant, path) != get_nested_path(baseline, path) for path in factor_paths)
            self.assertTrue(changed)


if __name__ == "__main__":
    unittest.main()
