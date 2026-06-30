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

# The below is a Lightning wrapper that is used to train the autoencoder. 
# This was implemented so that we could interface with PET's training workflow
# Which uses Lightning. 

# See GH issue: https://github.com/ACCESS-Community-Hub/PyEarthTools/issues/266
# NB: This wrapper should probably also move out of this notebook

class LightingWrapper(L.LightningModule):
    def __init__(self, model, mask, lr=1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.L1Loss()

        # Persist mask with device movement/checkpoints
        self.register_buffer(
            "mask",
            torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x):
        # x: [B, C, H, W]
        mask = self.mask.expand(x.shape[0], 1, x.shape[2], x.shape[3])
        return self.model(x, mask)
        
    def _shared_step(self, batch):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        if x.ndim == 5 and x.shape[1] == 1:
            x = x[:, 0]
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_hat = self.forward(x)
        mask = self.mask.expand_as(x[:, :1])
        loss = torch.abs(x_hat - x)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)
    
    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("valid_loss", loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch

        if x.ndim == 5 and x.shape[1] == 1:
            x = x[:, 0]
            
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        x_hat = self.forward(x)

        return {
            "x": x.detach().cpu(),
            "x_hat": x_hat.detach().cpu(),
        }


    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

        # Define the Autoencoder using PyTorch -- probably should also be moved out :) 

class AutoEncoder(nn.Module):

    def __init__(self,
                 input_channel_count=2,
                 output_channel_count=2):

        super(AutoEncoder, self).__init__()

        # ---------- Encoder ----------
        self.enc1 = PartialConv2d(input_channel_count, 16, kernel_size=4, stride=2, padding=1)
        self.enc2 = PartialConv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.enc3 = PartialConv2d(32, 64, kernel_size=7, stride=1, padding=3)

        # ---------- Decoder ----------
        self.dec1 = PartialConv2d(64, 32, kernel_size=7, stride=1, padding=3)
        self.dec2 = PartialConv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.dec3 = PartialConv2d(16, output_channel_count, kernel_size=4, stride=1, padding=2)

        self.relu = nn.ReLU()
    
    def encode(self, x, mask):
        """Encoder: returns latent representation and updated mask"""
        x, mask = self.enc1(x, mask)
        x = self.relu(x)
        x, mask = self.enc2(x, mask)
        x = self.relu(x)
        x, mask = self.enc3(x, mask)
        x = self.relu(x)
        return x, mask
    
    def decode(self, x, mask):
        """Decoder: takes latent representation and reconstructs output"""
        # upsample 1
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        mask = F.interpolate(mask, scale_factor=2, mode="nearest")

        x, mask = self.dec1(x, mask)
        x = self.relu(x)

        # upsample 2
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        mask = F.interpolate(mask, scale_factor=2, mode="nearest")

        x, mask = self.dec2(x, mask)
        x = self.relu(x)

        # final conv
        x, mask = self.dec3(x, mask)

        # Note that here we were using nn.Sigmoid, which 
        # Forces the prediction to be positive, hence the single-signed estimates.
        reconstructed = x
        
        return reconstructed, mask
    
    def forward(self, x, mask):
        # Store input size for later
        input_h, input_w = x.shape[2], x.shape[3]

        # Encoder
        latent, latent_mask = self.encode(x, mask)

        # Decoder
        reconstructed, final_mask = self.decode(latent, latent_mask)
        
        # Crop to match input size (handles rounding errors from upsampling)
        reconstructed = reconstructed[:, :, :input_h, :input_w]

        return reconstructed


class UNet(nn.Module):

    def __init__(self,
                 input_channel_count=2,
                 output_channel_count=2):

        super(UNet, self).__init__()

        # ---------- Encoder ----------
        self.enc1 = PartialConv2d(input_channel_count, 16, kernel_size=4, stride=2, padding=1)
        self.enc2 = PartialConv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.enc3 = PartialConv2d(32, 64, kernel_size=7, stride=1, padding=3)

        # ---------- Decoder ----------
        # After first upsample, latent has 64 channels and skip2 has 32 channels
        self.dec1 = PartialConv2d(64 + 32, 32, kernel_size=7, stride=1, padding=3)

        # After second upsample, dec1 has 32 channels and skip1 has 16 channels
        self.dec2 = PartialConv2d(32 + 16, 16, kernel_size=3, stride=1, padding=1)

        self.dec3 = PartialConv2d(16, output_channel_count, kernel_size=4, stride=1, padding=2)

        self.relu = nn.ReLU()

    def encode(self, x, mask):
        """Encoder: returns latent representation, updated mask, and skip features"""

        x1, mask1 = self.enc1(x, mask)
        x1 = self.relu(x1)

        x2, mask2 = self.enc2(x1, mask1)
        x2 = self.relu(x2)

        latent, latent_mask = self.enc3(x2, mask2)
        latent = self.relu(latent)

        return latent, latent_mask, x1, mask1, x2, mask2

    def decode(self, x, mask, x1, mask1, x2, mask2):
        """Decoder with U-Net skip connections"""

        # ---------- Upsample to enc2 resolution ----------
        x = F.interpolate(x, size=x2.shape[2:], mode="bilinear", align_corners=False)
        mask = F.interpolate(mask, size=mask2.shape[2:], mode="nearest")

        x = torch.cat([x, x2], dim=1)
        # mask = torch.cat([mask, mask2], dim=1)

        x, mask = self.dec1(x, mask)
        x = self.relu(x)

        # ---------- Upsample to enc1 resolution ----------
        x = F.interpolate(x, size=x1.shape[2:], mode="bilinear", align_corners=False)
        mask = F.interpolate(mask, size=mask1.shape[2:], mode="nearest")

        x = torch.cat([x, x1], dim=1)
        # mask = torch.cat([mask, mask1], dim=1)

        x, mask = self.dec2(x, mask)
        x = self.relu(x)

        # ---------- Final upsample to input resolution ----------
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        mask = F.interpolate(mask, scale_factor=2, mode="nearest")

        x, mask = self.dec3(x, mask)

        return x, mask

    def forward(self, x, mask):
        input_h, input_w = x.shape[2], x.shape[3]

        latent, latent_mask, x1, mask1, x2, mask2 = self.encode(x, mask)

        reconstructed, final_mask = self.decode(
            latent, latent_mask,
            x1, mask1,
            x2, mask2,
        )

        reconstructed = reconstructed[:, :, :input_h, :input_w]

        return reconstructed