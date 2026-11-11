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
    One-step forward emulator.

    forward(prior, forcing, mask):
        prior   : (B, n_prior * P, H, W) prior states stacked oldest first
        forcing : (B, C, H, W)           forcing in the target month
        mask    : (H, W), (B, H, W) or (B, 1, H, W), 1 = ocean
    returns (B, P, H, W): the next state, always float32 (so the losses stay
    in full precision when training with 16-bit mixed precision).

    tendency_network (the UNet, or the Samudra ConvNeXt U-Net) predicts the
    CHANGE from the most recent prior state, so
    next = prior[-1] + tendency_network(...). Persisting the current anomaly
    then costs the network nothing, and it only has to learn the tendency.
    Land values are ignored: the losses and the evaluation mask them out.

    The inputs are laid out channels-last in memory (NHWC): faster fp16
    convolutions on tensor cores, identical results.

    With ``noise_channels`` > 0 (the CRPS ensemble) the tendency network also
    gets that many channels of fresh standard Gaussian noise at every call,
    after the forcing, so every call samples a different next state.
    """

    def __init__(self, tendency_network, n_prior, n_prognostic, n_forcing, noise_channels=0):
        super().__init__()
        self.tendency_network = tendency_network
        self.n_prior = n_prior
        self.n_prognostic = n_prognostic
        self.n_forcing = n_forcing
        self.noise_channels = noise_channels

    def forward(self, prior, forcing, mask):
        inputs = [prior, forcing]
        if self.noise_channels:
            inputs.append(torch.randn(prior.shape[0], self.noise_channels, *prior.shape[-2:],
                                      device=prior.device, dtype=prior.dtype))
        x = torch.cat(inputs, dim=1).contiguous(memory_format=torch.channels_last)
        # One (1, 1, H, W) land mask broadcast over the batch: every partial
        # convolution then counts valid pixels once, not once per sample.
        mask = mask.to(device=x.device, dtype=x.dtype)
        if mask.ndim == 2:
            mask = mask[None, None]
        elif mask.ndim == 3:
            mask = mask[:, None]

        # Most recent prior state: the last n_prognostic channels (oldest first).
        pred = prior[:, -self.n_prognostic :] + self.tendency_network(x, mask)
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

    noise_channels = cfg.ensemble.noise_channels if cfg.paradigm == "crps_ensemble" else 0
    tendency_network = ARCHITECTURES[m.arch](in_channels, n_prognostic, m, noise_channels)
    model = ForwardEmulator(tendency_network, w.n_prior, n_prognostic, n_forcing, noise_channels)
    return model.to(memory_format=torch.channels_last)


def load_trunk(model, state_dict):
    """
    Load trained weights (e.g. the shared deterministic trunk) into model.

    Weights of the same shape are copied. A weight that only has extra input
    channels in the model (the first convolution of a CRPS-ensemble model,
    whose noise channels come after the physical inputs) gets the trained
    weights for the existing channels and zeros for the new ones, so the model
    starts out exactly as the trained network, with the noise switched off
    until training turns it on. Anything else that does not fit is an error.
    """
    own = model.state_dict()
    unknown = set(state_dict) - set(own)
    missing = set(own) - set(state_dict)
    if unknown or missing:
        raise KeyError(f"Weights do not match the model: unexpected {sorted(unknown)}, missing {sorted(missing)}")
    with torch.no_grad():
        for name, value in state_dict.items():
            target = own[name]
            if value.shape == target.shape:
                target.copy_(value)
            elif (value.ndim == target.ndim and value.ndim >= 2 and value.shape[0] == target.shape[0]
                  and value.shape[2:] == target.shape[2:] and value.shape[1] < target.shape[1]):
                target.zero_()
                target[:, : value.shape[1]].copy_(value)
            else:
                raise ValueError(f"{name}: shape {tuple(value.shape)} does not fit {tuple(target.shape)}")
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
