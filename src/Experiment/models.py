"""
Model construction.

``build_model(cfg, data)`` returns a ``ForwardEmulator``: a one-step
autoregressive predictor that maps the prior prognostic states plus the current
forcing to the next prognostic state. The backbone architecture and the
optional add-ons are chosen by name in ``cfg.model``.
"""

import random

import numpy as np
import torch
import torch.nn as nn

from Emulator import LatentResidualTuner, SpatialResidualHead, UNet

# Backbones by name: (in_channels, out_channels, latent_processor, padding_mode) -> nn.Module
# mapping (B, in, H, W) and a (B, 1, H, W) mask to (B, out, H, W).
ARCHITECTURES = {
    "unet": lambda in_ch, out_ch, latent_processor, padding_mode: UNet(
        input_channel_count=in_ch,
        output_channel_count=out_ch,
        latent_processor=latent_processor,
        padding_mode=padding_mode,
    ),
}

# Channels of the UNet bottleneck (enc3), which the latent processor refines.
UNET_LATENT_CHANNELS = 64


class ForwardEmulator(nn.Module):
    """
    One-step forward emulator.

    forward(prior, forcing, mask):
        prior   : (B, n_prior * P, H, W) prior states stacked oldest first
        forcing : (B, C, H, W)           forcing in the target month
        mask    : (H, W), (B, H, W) or (B, 1, H, W), 1 = ocean
    returns (B, P, H, W): the next state, always float32 (so the losses stay
    in full precision when training with 16-bit mixed precision).

    If ``output_processor`` is set it receives the backbone's prediction and
    all model inputs, and returns the corrected prediction
    (e.g. SpatialResidualHead: pred + residual_scale * correction).
    """

    def __init__(self, backbone, n_prior, n_prognostic, n_forcing, output_processor=None):
        super().__init__()
        self.backbone = backbone
        self.n_prior = n_prior
        self.n_prognostic = n_prognostic
        self.n_forcing = n_forcing
        self.output_processor = output_processor

    def forward(self, prior, forcing, mask):
        x = torch.cat([prior, forcing], dim=1)
        mask = mask.to(device=x.device, dtype=x.dtype)
        if mask.ndim == 2:
            mask = mask[None, None]
        elif mask.ndim == 3:
            mask = mask[:, None]
        mask = mask.expand(x.shape[0], 1, x.shape[2], x.shape[3])

        pred = self.backbone(x, mask)
        if self.output_processor is not None:
            pred = self.output_processor(pred, x, mask)
        return pred.float()


def build_model(cfg, data):
    """Build the ForwardEmulator described by cfg.model for the variables in data."""
    # Seed everything first so the initial weights (and later the shuffling of
    # the training data) are reproducible for a given cfg.train.seed.
    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    m, w = cfg.model, cfg.window
    n_prognostic, n_forcing = data.n_prognostic, data.n_forcing
    in_channels = w.n_prior * n_prognostic + n_forcing

    latent_processor = None
    if m.latent_processor == "latent_residual":
        latent_processor = LatentResidualTuner(
            channel_count=UNET_LATENT_CHANNELS,
            residual_scale=m.latent_residual_scale,
            padding_mode=m.padding_mode,
        )

    output_processor = None
    if m.output_head == "spatial_residual":
        output_processor = SpatialResidualHead(
            # the prediction + every model input (prior states and forcing)
            input_channel_count=n_prognostic + in_channels,
            output_channel_count=n_prognostic,
            hidden_channel_count=m.output_head_hidden_channels,
            residual_scale=m.output_head_residual_scale,
            padding_mode=m.padding_mode,
        )

    backbone = ARCHITECTURES[m.arch](in_channels, n_prognostic, latent_processor, m.padding_mode)
    return ForwardEmulator(backbone, w.n_prior, n_prognostic, n_forcing, output_processor)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
