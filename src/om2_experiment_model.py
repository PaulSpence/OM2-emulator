"""Configurable PartialConv autoencoder models for OM2 experiments.

The original notebook architecture used fixed layers. This module introduces a
config-driven variant where encoder/decoder blocks are defined in YAML. That
lets us run controlled ablations over channels, kernels, strides, and padding.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from om2_model_utils import PartialConv2d


class ConfigurablePartialConvAutoEncoder(nn.Module):
    """A PartialConv-based autoencoder built from block config lists.

    Parameters
    ----------
    input_channels
        Number of model input channels.
    output_channels
        Number of model output channels.
    encoder_blocks
        Sequence of block dictionaries with convolution parameters.
    decoder_blocks
        Sequence of block dictionaries with convolution parameters and optional
        ``upsample_factor`` before each decoder block.
    activation
        Hidden activation name. Currently supports ``"relu"``.
    final_activation
        Output activation name. Supports ``"sigmoid"`` or ``"identity"``.
    upsample_mode
        Interpolation mode for decoder feature maps.

    Notes
    -----
    The decoder applies nearest-neighbor interpolation to masks and configurable
    interpolation to feature maps. This preserves the semantics of binary mask
    support while allowing smooth feature upsampling.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        encoder_blocks: list[dict[str, Any]],
        decoder_blocks: list[dict[str, Any]],
        activation: str = "relu",
        final_activation: str = "sigmoid",
        upsample_mode: str = "bilinear",
    ) -> None:
        super().__init__()

        self.upsample_mode = upsample_mode
        self.hidden_activation = self._build_hidden_activation(activation)
        self.final_activation = self._build_final_activation(final_activation)

        self.encoder = nn.ModuleList()
        in_channels = input_channels
        for block in encoder_blocks:
            layer = PartialConv2d(
                in_channels,
                block["out_channels"],
                kernel_size=block["kernel_size"],
                stride=block["stride"],
                padding=block["padding"],
            )
            self.encoder.append(layer)
            in_channels = block["out_channels"]

        self.decoder = nn.ModuleList()
        self.decoder_upsample_factors: list[int] = []
        in_channels = self.encoder[-1].conv.out_channels
        for block in decoder_blocks:
            layer = PartialConv2d(
                in_channels,
                block["out_channels"],
                kernel_size=block["kernel_size"],
                stride=block["stride"],
                padding=block["padding"],
            )
            self.decoder.append(layer)
            self.decoder_upsample_factors.append(int(block.get("upsample_factor", 1)))
            in_channels = block["out_channels"]

        # Safety check: final decoder block must match requested output channels.
        if self.decoder[-1].conv.out_channels != output_channels:
            raise ValueError(
                "Final decoder block out_channels must equal model.output_channels; "
                f"got {self.decoder[-1].conv.out_channels} vs {output_channels}."
            )

    @staticmethod
    def _build_hidden_activation(name: str) -> nn.Module:
        """Construct hidden-layer activation module from a name string."""
        if name == "relu":
            return nn.ReLU()

        raise ValueError(f"Unsupported hidden activation: {name!r}")

    @staticmethod
    def _build_final_activation(name: str) -> nn.Module:
        """Construct output activation module from a name string."""
        if name == "sigmoid":
            return nn.Sigmoid()
        if name == "identity":
            return nn.Identity()

        raise ValueError(f"Unsupported final activation: {name!r}")

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run encoder stack and return latent tensor plus updated mask.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch, channels, height, width)``.
        mask
            Binary support mask of shape ``(batch, 1, height, width)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(latent, latent_mask)`` after all encoder blocks.
        """
        for layer in self.encoder:
            x, mask = layer(x, mask)
            x = self.hidden_activation(x)

        return x, mask

    def decode(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run decoder stack and return reconstruction plus final mask.

        Parameters
        ----------
        x
            Latent tensor from :meth:`encode`.
        mask
            Latent mask from :meth:`encode`.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(reconstruction, reconstruction_mask)``.
        """
        for index, layer in enumerate(self.decoder):
            upsample_factor = self.decoder_upsample_factors[index]

            # Upsampling is done before convolution to mirror the existing notebook logic.
            if upsample_factor > 1:
                x = F.interpolate(
                    x,
                    scale_factor=upsample_factor,
                    mode=self.upsample_mode,
                    align_corners=False if self.upsample_mode in {"bilinear", "bicubic"} else None,
                )
                mask = F.interpolate(mask, scale_factor=upsample_factor, mode="nearest")

            x, mask = layer(x, mask)

            # Apply hidden activation on all decoder layers except the final output layer.
            if index < len(self.decoder) - 1:
                x = self.hidden_activation(x)

        x = self.final_activation(x)
        return x, mask

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        return_features: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run full autoencoder forward pass.

        Parameters
        ----------
        x
            Input tensor.
        mask
            Binary support mask.
        return_features
            When ``True``, include latent features and masks in a dictionary.

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            Reconstruction and optional feature dictionary.

        Notes
        -----
        The output is cropped back to input height/width to handle any shape
        rounding introduced by interpolation and stride combinations.
        """
        input_height, input_width = x.shape[2], x.shape[3]

        latent, latent_mask = self.encode(x, mask)
        reconstruction, output_mask = self.decode(latent, latent_mask)

        reconstruction = reconstruction[:, :, :input_height, :input_width]
        output_mask = output_mask[:, :, :input_height, :input_width]

        features: dict[str, torch.Tensor] = {}
        if return_features:
            features = {
                "latent": latent,
                "latent_mask": latent_mask,
                "output_mask": output_mask,
            }

        return reconstruction, features


def build_model_from_config(model_config: dict[str, Any]) -> ConfigurablePartialConvAutoEncoder:
    """Instantiate a configurable autoencoder from resolved config.

    Parameters
    ----------
    model_config
        ``model`` section of the resolved experiment config.

    Returns
    -------
    ConfigurablePartialConvAutoEncoder
        Initialized model ready for training/evaluation.
    """
    return ConfigurablePartialConvAutoEncoder(
        input_channels=int(model_config["input_channels"]),
        output_channels=int(model_config["output_channels"]),
        encoder_blocks=model_config["encoder"],
        decoder_blocks=model_config["decoder"],
        activation=model_config.get("activation", "relu"),
        final_activation=model_config.get("final_activation", "sigmoid"),
        upsample_mode=model_config.get("upsample_mode", "bilinear"),
    )
