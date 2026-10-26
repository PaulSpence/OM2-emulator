# SPDX-FileCopyrightText: 2023 Matthias Karlbauer, Nathaniel Cresswell-Clay, Thorsten Kurth
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0 AND MIT
#
# Ported from Samudra (https://github.com/m2lines/Samudra, Apache-2.0):
# src/samudra/models/modules/{blocks,activations,unet_backbone}.py and
# src/samudra/models/samudra.py. The ConvNeXt block derives from dlwp-hpx (MIT).
# Kept: the ConvNeXt U-Net with its defaults (block, widths, dilations, capped
# GELU, batch norm, average-pool down, zonally periodic bilinear up, additive
# skips, circular padding in longitude and zero padding in latitude, a 3x3
# output convolution). Left out: training-time options (drop path,
# checkpointing modes, bfloat16, positional channels, pointwise-linear layers),
# all off by default in Samudra.
"""
The Samudra architecture as a backbone for ForwardEmulator (ModelConfig.arch =
"samudra"), so it can be compared with this package's UNet on the same data,
losses, residual update and budget constraint.

Differences from Samudra as published, all from the shared design here: the
wrapper adds the prediction to the last state (Samudra's pred_residuals
defaults to False), and land is handled as in Samudra itself (inputs are zero
on land, nothing is filled), with the losses, not the network, masking it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CappedGELU(nn.Module):
    """GELU clamped at cap_value."""

    def __init__(self, cap_value=10.0):
        super().__init__()
        self.gelu = nn.GELU()
        self.register_buffer("cap", torch.tensor(cap_value, dtype=torch.float32))

    def forward(self, x):
        return torch.clamp(self.gelu(x), max=self.cap)


def _pad(x, n):
    """Pad n cells: periodic in longitude (last axis), zeros in latitude."""
    x = F.pad(x, (n, n, 0, 0), mode="circular")
    return F.pad(x, (0, 0, n, n), mode="constant")


class ConvNeXtBlock(nn.Module):
    """
    Samudra's ConvNeXt block: two dilated 3x3 convolutions at upscale_factor x
    the input width (batch norm + capped GELU after each), a 1x1 projection to
    out_channels, and a skip (identity, or 1x1 when the widths differ).
    """

    def __init__(self, in_channels, out_channels, dilation=1, kernel_size=3, upscale_factor=4):
        super().__init__()
        inner = int(in_channels * upscale_factor)
        self.n_pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        self.conv1 = nn.Conv2d(in_channels, inner, kernel_size, dilation=dilation)
        self.norm1 = nn.BatchNorm2d(inner)
        self.conv2 = nn.Conv2d(inner, inner, kernel_size, dilation=dilation)
        self.norm2 = nn.BatchNorm2d(inner)
        self.act = CappedGELU()
        self.project = nn.Conv2d(inner, out_channels, 1)

    def forward(self, x):
        y = self.act(self.norm1(self.conv1(_pad(x, self.n_pad))))
        y = self.act(self.norm2(self.conv2(_pad(y, self.n_pad))))
        return self.skip(x) + self.project(y)


def zonally_periodic_upsample(x):
    """2x bilinear upsampling with periodic padding in longitude (Samudra's default upsampler)."""
    width = x.shape[-1]
    y = F.interpolate(F.pad(x, (1, 1, 0, 0), mode="circular"), scale_factor=2, mode="bilinear", align_corners=False)
    return y[..., 2 : 2 + 2 * width]


def _match(x, like):
    """Zero-pad x to like's spatial size: pooling an odd size loses a row or column."""
    dh, dw = like.shape[-2] - x.shape[-2], like.shape[-1] - x.shape[-1]
    return F.pad(x, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2))


class SamudraUNet(nn.Module):
    """
    forward(x, mask) -> (B, out_channels, H, W). ``mask`` is accepted for the
    ForwardEmulator interface and not used (see the module docstring).

    ch_width / dilation: one entry per level (Samudra's defaults:
    [200, 250, 300, 400] and [1, 2, 4, 8], about 127 million parameters for
    eight inputs). Going down, each level is a ConvNeXt block then a 2x average
    pool; a middle block; going up, a 2x upsample plus the skip from that
    level, then a block; a final block and a 3x3 output convolution.
    """

    def __init__(self, in_channels, out_channels, ch_width=(200, 250, 300, 400), dilation=(1, 2, 4, 8)):
        super().__init__()
        if len(ch_width) != len(dilation) or len(ch_width) < 2:
            raise ValueError("ch_width and dilation need one entry per level, and at least two levels")
        widths = [in_channels, *ch_width]
        self.down = nn.ModuleList(ConvNeXtBlock(a, b, d) for a, b, d in zip(widths[:-1], widths[1:], dilation))
        self.middle = ConvNeXtBlock(widths[-1], widths[-1], dilation[-1])
        rev, rev_dilation = widths[::-1], list(dilation)[::-1]
        self.up = nn.ModuleList(ConvNeXtBlock(a, b, d) for a, b, d in zip(rev[:-2], rev[1:-1], rev_dilation))
        # As in Samudra, the final block reuses the last upward block's dilation.
        self.final = ConvNeXtBlock(widths[1], widths[1], rev_dilation[len(ch_width) - 2])
        self.output = nn.Conv2d(widths[1], out_channels, 3)

    def forward(self, x, mask=None):
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = F.avg_pool2d(x, 2)
        x = self.middle(x)
        # Each upsample joins the skip of the level it returns to (deepest first).
        for block, skip in zip(self.up, reversed(skips[1:])):
            x = block(_match(zonally_periodic_upsample(x), skip) + skip)
        x = _match(zonally_periodic_upsample(x), skips[0]) + skips[0]
        x = self.final(x)
        return self.output(_pad(x, 1))
