"""
Model construction.

``build_model(cfg, data)`` returns a ``ForwardEmulator``: a one-step
autoregressive predictor that maps the prior prognostic states plus the current
forcing to the next prognostic state. The architecture of its tendency
network is chosen by name in ``cfg.model.arch``.
"""

import random

import numpy as np
import torch
import torch.nn as nn

from Emulator import UNet
from Emulator.samudra_unet import SamudraUNet

# Tendency networks by name: (in_channels, out_channels, model config)
# -> nn.Module mapping (B, in, H, W) and a (B, 1, H, W)
# mask to (B, out, H, W).
# Extra inputs (noise channels) do not change the widths: see UNet.
ARCHITECTURES = {
    "unet": lambda in_ch, out_ch, m, extra=0: UNet(
        input_channel_count=in_ch + extra,
        output_channel_count=out_ch,
        width_multiplier=m.width_multiplier,
        width_channel_count=in_ch,
    ),
    # Samudra's ConvNeXt U-Net.
    "samudra": lambda in_ch, out_ch, m, extra=0: SamudraUNet(
        in_ch + extra, out_ch, ch_width=list(m.samudra_ch_width), dilation=list(m.samudra_dilation),
        width_channel_count=in_ch,
    ),
}


class ForwardEmulator(nn.Module):
    """
    One-step forward emulator: next state = most recent state + tendency_network(inputs).

    forward(prior, forcing, mask) -> (B, P, H, W) the next state, in float32 (so the losses stay in full
    precision when training with 16-bit mixed precision)
        prior   : (B, n_prior * P, H, W) prior states stacked oldest first
        forcing : (B, C, H, W) forcing of the month being predicted
        mask    : (H, W) ocean mask, 1 = ocean

    The tendency network (the UNet, or the Samudra ConvNeXt U-Net) predicts the CHANGE from the most
    recent state, so persisting the current anomaly costs it nothing and it only has to learn the
    tendency. Land values are ignored: the losses and the evaluation mask them out.

    noise_channels > 0 (the CRPS ensemble): the network also gets that many channels of fresh N(0, 1)
    noise at every call, after the forcing, so every call samples a different next state. With 0 no
    noise channel is added.
    """

    def __init__(self, tendency_network, n_prognostic, noise_channels=0):
        super().__init__()
        self.tendency_network = tendency_network
        self.n_prognostic = n_prognostic          # P: the most recent state is the last P channels of prior
        self.noise_channels = noise_channels

    def forward(self, prior, forcing, mask):
        noise = torch.randn(prior.shape[0], self.noise_channels, *prior.shape[-2:], device=prior.device, dtype=prior.dtype)
        # The inputs side by side, laid out channels-last in memory (NHWC): faster 16-bit convolutions on
        # tensor cores, same results.
        x = torch.cat([prior, forcing, noise], dim=1).contiguous(memory_format=torch.channels_last)
        # One (1, 1, H, W) land mask broadcast over the batch: every partial convolution then counts the
        # ocean cells once, not once per sample.
        mask = mask.to(device=x.device, dtype=x.dtype)[None, None]
        next_state = prior[:, -self.n_prognostic :] + self.tendency_network(x, mask)
        return next_state.float()


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

    noise_channels = cfg.ensemble.noise_channels if cfg.paradigm == "crps_ensemble" else 0
    tendency_network = ARCHITECTURES[m.arch](in_channels, n_prognostic, m, noise_channels)
    model = ForwardEmulator(tendency_network, n_prognostic, noise_channels)
    return model.to(memory_format=torch.channels_last)


def load_trunk(model, state_dict):
    """
    Load trained weights (e.g. the deterministic stage 1 network) into model, which may have extra
    noise input channels. Every weight is copied into the model's weight of the same name. The one
    that does not fit, the first convolution of a CRPS-ensemble model (its noise channels come after
    the physical inputs), gets the trained weights in its first channels and zeros for the noise
    channels. The model then starts out exactly as the trained network, with the noise switched off
    until training turns it on.
    """
    own = model.state_dict()
    with torch.no_grad():
        for name, value in state_dict.items():
            if value.shape == own[name].shape:
                own[name].copy_(value)
            else:
                own[name].zero_()
                own[name][:, : value.shape[1]].copy_(value)   # input channels: physical first, then noise
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
