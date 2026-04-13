"""Tests for configurable PartialConv model construction and shapes."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from om2_experiment_model import build_model_from_config  # noqa: E402


class TestConfigurableModel(unittest.TestCase):
    """Ensure model factory output remains shape-compatible."""

    def setUp(self) -> None:
        """Create a compact model config for deterministic shape tests."""
        self.model_config = {
            "input_channels": 2,
            "output_channels": 2,
            "encoder": [
                {"out_channels": 8, "kernel_size": 3, "stride": 2, "padding": 1},
                {"out_channels": 16, "kernel_size": 3, "stride": 2, "padding": 1},
            ],
            "decoder": [
                {
                    "out_channels": 8,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "upsample_factor": 2,
                },
                {
                    "out_channels": 2,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "upsample_factor": 2,
                },
            ],
            "activation": "relu",
            "final_activation": "identity",
            "upsample_mode": "bilinear",
        }

    def test_forward_matches_input_shape(self) -> None:
        """Model reconstruction should be cropped to input height/width."""
        model = build_model_from_config(self.model_config)
        x = torch.rand(1, 2, 30, 36)
        mask = torch.ones(1, 1, 30, 36)

        prediction, _ = model(x, mask, return_features=False)
        self.assertEqual(prediction.shape, x.shape)

    def test_forward_returns_latent_features(self) -> None:
        """Feature dict should include latent tensors when requested."""
        model = build_model_from_config(self.model_config)
        x = torch.rand(1, 2, 20, 24)
        mask = torch.ones(1, 1, 20, 24)

        _, features = model(x, mask, return_features=True)
        self.assertIn("latent", features)
        self.assertIn("latent_mask", features)


if __name__ == "__main__":
    unittest.main()
