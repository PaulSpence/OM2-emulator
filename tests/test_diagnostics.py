"""Tests for diagnostics metric calculations."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from om2_experiment_diagnostics import compute_area_weighted_rmse_by_channel  # noqa: E402


class TestDiagnosticsMetrics(unittest.TestCase):
    """Validate channel-wise RMSE computation against simple inputs."""

    def test_area_weighted_rmse_by_channel(self) -> None:
        """RMSE should be zero for matching channels and positive otherwise."""
        prediction = np.zeros((1, 2, 3, 4), dtype=np.float64)
        target = np.zeros((1, 2, 3, 4), dtype=np.float64)

        # Introduce a deterministic offset in channel 1 only.
        prediction[0, 1, :, :] = 2.0

        area = np.ones((3, 4), dtype=np.float64)
        metrics = compute_area_weighted_rmse_by_channel(
            prediction,
            target,
            area,
            channel_names=["OHC", "HeatFlux"],
        )

        self.assertAlmostEqual(metrics["rmse_OHC"], 0.0)
        self.assertAlmostEqual(metrics["rmse_HeatFlux"], 2.0)


if __name__ == "__main__":
    unittest.main()
