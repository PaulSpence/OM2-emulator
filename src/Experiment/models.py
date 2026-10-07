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

from .config import _dirichlet_value

# Backbones by name: (in_channels, out_channels, latent_processor, model config,
# Dirichlet input channels) -> nn.Module mapping (B, in, H, W) and a (B, 1, H, W)
# mask to (B, out, H, W).
ARCHITECTURES = {
    "unet": lambda in_ch, out_ch, latent_processor, m, dirichlet: UNet(
        input_channel_count=in_ch,
        output_channel_count=out_ch,
        latent_processor=latent_processor,
        padding_mode=m.padding_mode,
        width_multiplier=m.width_multiplier,
        dirichlet=dirichlet,
    ),
}


class ForwardEmulator(nn.Module):
    """
    One-step forward emulator.

    forward(prior, forcing, mask):
        prior   : (B, n_prior * P, H, W) prior states stacked oldest first
        forcing : (B, C, H, W)           forcing in the target month
        mask    : (H, W), (B, H, W) or (B, 1, H, W), 1 = ocean
    returns (B, P, H, W): the next state, always float32 (so the losses stay
    in full precision when training with 16-bit mixed precision).

    The backbone predicts the CHANGE from the most recent prior state, so
    next = prior[-1] + backbone(...). Persisting the current anomaly then costs
    the network nothing, and it only has to learn the tendency. Land values
    are ignored: the losses and the evaluation mask them out.

    If ``output_processor`` is set it receives that prediction and all model
    inputs, and returns the corrected prediction
    (e.g. SpatialResidualHead: pred + residual_scale * correction).
    """

    def __init__(self, backbone, n_prior, n_prognostic, n_forcing, output_processor=None, channels_last=False):
        super().__init__()
        # channels_last: NHWC memory layout, faster for fp16 convolutions on
        # tensor cores (V100); no effect on the results.
        self.channels_last = channels_last
        self.backbone = backbone
        self.n_prior = n_prior
        self.n_prognostic = n_prognostic
        self.n_forcing = n_forcing
        self.output_processor = output_processor

    def forward(self, prior, forcing, mask):
        x = torch.cat([prior, forcing], dim=1)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        # One (1, 1, H, W) land mask broadcast over the batch: every partial
        # convolution then counts valid pixels once, not once per sample.
        mask = mask.to(device=x.device, dtype=x.dtype)
        if mask.ndim == 2:
            mask = mask[None, None]
        elif mask.ndim == 3:
            mask = mask[:, None]

        # Most recent prior state: the last n_prognostic channels (oldest first).
        pred = prior[:, -self.n_prognostic :] + self.backbone(x, mask)
        if self.output_processor is not None:
            pred = self.output_processor(pred, x, mask)
        return pred.float()


def dirichlet_channels(cfg):
    """
    {input channel: z-score value} for the variables with a Dirichlet land
    condition in cfg.data.boundary_conditions. Input channels are the prior
    states (n_prior time levels x prognostic variables, oldest first) followed
    by the forcing variables.
    """
    d = cfg.data
    n_prior, n_prognostic = cfg.window.n_prior, len(d.prognostic)
    channels = {}
    for name, condition in d.boundary_conditions.items():
        value = _dirichlet_value(name, condition)
        if value is None:
            continue
        if name in d.prognostic:
            p = d.prognostic.index(name)
            for level in range(n_prior):
                channels[level * n_prognostic + p] = value
        else:
            channels[n_prior * n_prognostic + d.forcing.index(name)] = value
    return channels


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
    dirichlet = dirichlet_channels(cfg)

    latent_processor = None
    if m.latent_processor == "latent_residual":
        latent_processor = LatentResidualTuner(
            # The UNet bottleneck width, which scales with in_channels.
            channel_count=UNet.channel_widths(in_channels, m.width_multiplier)[-1],
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

    backbone = ARCHITECTURES[m.arch](in_channels, n_prognostic, latent_processor, m, dirichlet)
    model = ForwardEmulator(backbone, w.n_prior, n_prognostic, n_forcing, output_processor, m.channels_last)
    if m.channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
