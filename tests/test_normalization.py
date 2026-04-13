"""Tests for normalization strategy behavior and inverse transform."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from om2_experiment_normalization import denormalize_tensor, normalize_tensor  # noqa: E402


class TestNormalizationStrategies(unittest.TestCase):
    """Check finite outputs and round-trip consistency where applicable."""

    def setUp(self) -> None:
        """Create deterministic synthetic tensor and ocean mask."""
        torch.manual_seed(1)
        self.x = torch.rand(1, 2, 8, 10)
        self.mask = torch.ones(1, 1, 8, 10)

    def test_masked_zscore_round_trip(self) -> None:
        """Denormalization should reconstruct original values."""
        cfg = {"strategy": "masked_zscore", "eps": 1e-6}
        x_norm, stats = normalize_tensor(self.x, self.mask, cfg)
        x_back = denormalize_tensor(x_norm, stats)

        self.assertTrue(torch.all(torch.isfinite(x_norm)))
        self.assertTrue(torch.allclose(self.x, x_back, atol=1e-5, rtol=1e-5))

    def test_all_strategies_produce_finite_outputs(self) -> None:
        """Every supported strategy should avoid NaNs/Infs."""
        strategies = [
            {"strategy": "masked_zscore", "eps": 1e-6},
            {"strategy": "nan_zscore", "eps": 1e-6},
            {"strategy": "robust_median_iqr", "eps": 1e-6, "robust_quantiles": [0.25, 0.75]},
            {"strategy": "minmax", "eps": 1e-6},
        ]

        for cfg in strategies:
            with self.subTest(strategy=cfg["strategy"]):
                x_norm, _ = normalize_tensor(self.x, self.mask, cfg)
                self.assertTrue(torch.all(torch.isfinite(x_norm)))


if __name__ == "__main__":
    unittest.main()
