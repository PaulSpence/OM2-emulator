"""
Utility functions and modules for OM2 emulator models.

This module contains neural network components and utilities for building
and training autoencoder models on ACCESS-OM2 data.
"""

import torch
import torch.nn as nn


class PartialConv2d(nn.Module):
    """
    Mask-aware (partial) 2D convolution layer.

    This layer implements a simplified version of the partial convolution
    described in:

        Liu et al. (2018)
        "Image Inpainting for Irregular Holes Using Partial Convolutions"

    The key idea is that convolution is performed using only valid pixels
    defined by a binary mask.

    In this implementation:
        mask = 1  -> valid pixel (ocean)
        mask = 0  -> invalid pixel (land)

    The layer performs three operations:

    1. Mask the input
       x_masked = x * mask

    2. Apply a standard convolution
       out = Conv2D(x_masked)

    3. Renormalize by the number of valid pixels contributing to each
       convolution window.

       If N pixels were valid in a KxK window, the output is multiplied by:

            kernel_area / N

       This prevents the convolution amplitude from shrinking near masked
       regions (coastlines).

    4. Update the mask

       The new mask is defined as:

            new_mask = 1 if any valid pixel existed in the window
                     = 0 otherwise

       This allows the mask to propagate through the network.

    Parameters
    ----------
    in_ch : int
        Number of input channels.

    out_ch : int
        Number of output channels.

    kernel_size : int
        Size of convolution kernel.

    stride : int
        Convolution stride. When stride > 1 the output spatial dimensions
        shrink, which allows use in encoder networks.

    padding : int
        Padding size.

    Notes
    -----
    * padding_mode='replicate' is used to avoid artificial zeros near the
      boundary.
    * The mask convolution is fixed (all weights = 1) and has no gradients.
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode="replicate",
        )

        # convolution used only to count valid pixels
        self.mask_conv = nn.Conv2d(
            1,
            1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode="replicate",
            bias=False,
        )

        self.mask_conv.weight.data[:] = 1.0
        self.mask_conv.requires_grad_(False)

        self.kernel_area = kernel_size * kernel_size
        self.eps = 1e-6

    def forward(self, x, mask):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape:

                (batch, channels, height, width)

        mask : torch.Tensor
            Binary mask tensor:

                (batch, 1, height, width)

            1 = valid pixel
            0 = invalid pixel

        Returns
        -------
        out : torch.Tensor
            Convolution output

                (batch, out_channels, new_height, new_width)

        new_mask : torch.Tensor
            Updated mask

                (batch, 1, new_height, new_width)
        """

        # apply mask to input
        x_masked = x * mask

        # convolution
        out = self.conv(x_masked)

        # count valid pixels in each convolution window
        with torch.no_grad():
            mask_sum = self.mask_conv(mask)

        # renormalize output
        out = torch.where(
            mask_sum > 0,
            out * (self.kernel_area / (mask_sum + self.eps)),
            torch.zeros_like(out),
        )

        # updated mask
        new_mask = (mask_sum > 0).float()

        return out, new_mask
