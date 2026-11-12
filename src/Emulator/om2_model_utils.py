"""
Machine Learning modules for OM2 emulator.

This module contains neural network components for building
and training autoencoder models on ACCESS-OM2 data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L


class PartialConv2d(nn.Module):
    """
    A convolution that sees only ocean cells: a partial convolution (Liu et al. 2018,
    "Image Inpainting for Irregular Holes Using Partial Convolutions").

    forward(x, mask) -> (out, new_mask)
        x        : (B, C_in, H, W)
        mask     : (B, 1, H, W), or (1, 1, H, W) shared by the whole batch; 1 = ocean, 0 = land
        out      : (B, C_out, H_out, W_out)
        new_mask : 1 wherever the convolution window held at least one ocean cell

    Three steps:
    1. Pad x and the mask with ``padding`` cells on every side, following the ACCESS-OM2 tripolar grid:
         * east/west: longitude is periodic, so the columns wrap around. (With non-periodic padding the
           seam acted as a coastline neither side could see across; in the autoregressive OHC emulator
           98% of the fastest-growing perturbation sat at that seam.)
         * north: the tripole fold. Top-row cell i neighbours top-row cell nx - 1 - i, so the ghost rows
           above the top row are the top rows reversed in x and in y.
         * south (Antarctica): rows of zeros, marked as land, so the edge acts as a coastline.
    2. Convolve x * mask, i.e. with land set to 0.
    3. Multiply by (cells in the window) / (ocean cells in the window), so values next to a coast are
       not damped by the land around them. 0 where the window holds no ocean cell.
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride)
        # Counts the ocean cells in every window: all-ones weights, never trained.
        self.mask_conv = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, bias=False)
        self.mask_conv.weight.data[:] = 1.0
        self.mask_conv.requires_grad_(False)
        self.kernel_area = kernel_size * kernel_size

    def forward(self, x, mask):
        p = self.padding
        # 1. Padding. Along y: [p zero rows (south), field, fold rows (north)], the fold rows being the top
        # p rows flipped in y and x (rows run south to north). Along x: [last p columns, field, first p
        # columns], i.e. longitude wraps around. Built with torch.cat only: the same values as
        # F.pad(mode="circular"), which copies with a slow strided kernel (training step 9% faster).
        x_fold = torch.flip(x[..., -p:, :], dims=(-2, -1))
        x = torch.cat([x.new_zeros(*x.shape[:-2], p, x.shape[-1]), x, x_fold], dim=-2)
        x = torch.cat([x[..., -p:], x, x[..., :p]], dim=-1)
        mask_fold = torch.flip(mask[..., -p:, :], dims=(-2, -1))
        mask = torch.cat([mask.new_zeros(*mask.shape[:-2], p, mask.shape[-1]), mask, mask_fold], dim=-2)
        mask = torch.cat([mask[..., -p:], mask, mask[..., :p]], dim=-1)

        # 2 and 3. Convolve the ocean cells only, and scale up by (cells / ocean cells) in each window.
        ocean_cells = self.mask_conv(mask)
        scale = torch.where(ocean_cells > 0, self.kernel_area / (ocean_cells + 1e-6), torch.zeros_like(ocean_cells))
        new_mask = (ocean_cells > 0).to(mask.dtype)
        return self.conv(x * mask) * scale, new_mask

def upsample_periodic_x(x, size):
    """
    Bilinear upsampling of x (B, C, h, w) to size = (H, W), with longitude (the last dimension)
    periodic: wrap one column around on each side, interpolate, then cut the extra columns off again,
    so the east and west edges interpolate across the seam. W must be a whole multiple of w.
    """
    scale = size[1] // x.shape[-1]          # e.g. 2: every input column becomes 2 output columns
    x = torch.cat([x[..., -1:], x, x[..., :1]], dim=-1)    # wrap one column around on each side
    x = F.interpolate(x, size=(size[0], size[1] + 2 * scale), mode="bilinear", align_corners=False)
    return x[..., scale:-scale]


# The below is a Lightning wrapper that is used to train the autoencoder. 
# This was implemented so that we could interface with PET's training workflow
# Which uses Lightning. 

# See GH issue: https://github.com/ACCESS-Community-Hub/PyEarthTools/issues/266
# NB: This wrapper should probably also move out of this notebook

class LightningWrapper(L.LightningModule):
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
        x = upsample_periodic_x(x, (2 * x.shape[-2], 2 * x.shape[-1]))
        mask = F.interpolate(mask, scale_factor=2, mode="nearest")

        x, mask = self.dec1(x, mask)
        x = self.relu(x)

        # upsample 2
        x = upsample_periodic_x(x, (2 * x.shape[-2], 2 * x.shape[-1]))
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
    """
    Mask-aware U-Net: two stride-2 encoder levels, a bottleneck, and a decoder with skip connections at
    every resolution. Every convolution is a PartialConv2d, so land never enters the computation.

    forward(x, mask) -> (B, output_channel_count, H, W)
        x    : (B, input_channel_count, H, W) inputs (prior states, forcing, [noise])
        mask : (1, 1, H, W) or (B, 1, H, W) ocean mask, 1 = ocean

    Channel widths scale with the number of inputs, so adding predictors widens the network instead of
    squeezing them through a fixed bottleneck:
        level 1 (1/2 resolution)   : width_multiplier * width_channel_count   (32 for 8 inputs and 4)
        level 2 (1/4 resolution)   : 2 x level 1
        bottleneck (1/4 resolution): 4 x level 1
    width_channel_count defaults to input_channel_count. Setting it lets extra inputs (the noise
    channels of the CRPS ensemble) leave every layer but the first unchanged, so a trained network's
    weights still fit. width_multiplier must be >= 2 for the first layer to carry every signed input
    through its ReLU (each needs a +/- pair of channels).

    Design notes:
      * The bottleneck (enc3a-c) and the first decoder layer (dec1a-c) are each three 3x3 convolutions
        with a ReLU after each: together a 7x7 window (3 -> 5 -> 7), with 27 instead of 49 weights per
        channel pair and two extra non-linearities.
      * The output layer sees only the upsampled decoder features, not the raw input: fed the raw input
        as well, it learned a sharpening stencil that grew grid-scale noise by ~17% per step. The
        residual wrapper (next = state + output) carries the state's grid-scale content forward instead.
    """

    def __init__(self,
                 input_channel_count=2,
                 output_channel_count=2,
                 width_multiplier=4,
                 width_channel_count=None):
        super().__init__()

        # Channel widths, scaled from the number of (physical) input channels.
        width1 = int(round(width_multiplier * (width_channel_count or input_channel_count)))  # 1/2 resolution
        width2 = 2 * width1                                                                     # 1/4 resolution
        latent_width = 4 * width1                                                               # bottleneck

        # ---------- Encoder ----------
        self.enc1 = PartialConv2d(input_channel_count, width1, kernel_size=4, stride=2, padding=1)
        self.enc2 = PartialConv2d(width1, width2, kernel_size=3, stride=2, padding=1)
        # Bottleneck: three 3x3 layers, the receptive field of one 7x7.
        self.enc3a = PartialConv2d(width2, width2, kernel_size=3, stride=1, padding=1)
        self.enc3b = PartialConv2d(width2, width2, kernel_size=3, stride=1, padding=1)
        self.enc3c = PartialConv2d(width2, latent_width, kernel_size=3, stride=1, padding=1)

        # ---------- Decoder ----------
        # After the first upsample: latent + skip from enc2 (three 3x3 layers, as a 7x7).
        self.dec1a = PartialConv2d(latent_width + width2, width2, kernel_size=3, stride=1, padding=1)
        self.dec1b = PartialConv2d(width2, width2, kernel_size=3, stride=1, padding=1)
        self.dec1c = PartialConv2d(width2, width2, kernel_size=3, stride=1, padding=1)

        # After the second upsample: dec1 + skip from enc1.
        self.dec2 = PartialConv2d(width2 + width1, width1, kernel_size=3, stride=1, padding=1)

        # Full resolution: dec2 upsampled.
        self.dec3 = PartialConv2d(width1, output_channel_count, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mask):
        # The whole network on one batch. Every self.encN / self.decN(...) call runs one PartialConv2d
        # with that layer's own weights and returns (features, mask). Example widths: 8 inputs, width_multiplier 4.
        # ================================ Encoder ================================
        # PartialConv2d.forward, 4x4 stride 2: full -> 1/2 resolution, C_in -> 32 channels.
        x1, mask1 = self.enc1(x, mask)
        x1 = self.relu(x1)

        # PartialConv2d.forward, 3x3 stride 2: 1/2 -> 1/4 resolution, 32 -> 64 channels.
        x2, mask2 = self.enc2(x1, mask1)
        x2 = self.relu(x2)

        # Bottleneck, at 1/4 resolution: three 3x3 PartialConv2d.forward calls, a ReLU after each.
        latent, latent_mask = self.enc3a(x2, mask2)                  # 64 -> 64 channels
        latent = self.relu(latent)
        latent, latent_mask = self.enc3b(latent, latent_mask)        # 64 -> 64
        latent = self.relu(latent)
        latent, latent_mask = self.enc3c(latent, latent_mask)        # 64 -> 128
        latent = self.relu(latent)

        # ================================ Decoder ================================
        # x1 and x2 (the encoder features at 1/2 and 1/4 resolution) are the skip
        # connections: they are joined back in at their own resolution.

        # The bottleneck is already at x2's 1/4 resolution: upsample_periodic_x
        # interpolates to x2's grid size, wrapping around in longitude.
        y = upsample_periodic_x(latent, x2.shape[2:])
        y_mask = F.interpolate(latent_mask, size=mask2.shape[2:], mode="nearest")
        y = torch.cat([y, x2], dim=1)                                # skip from enc2: 128 + 64 = 192 channels

        # Three 3x3 PartialConv2d.forward calls at 1/4 resolution, a ReLU after each.
        y, y_mask = self.dec1a(y, y_mask)                            # 192 -> 64
        y = self.relu(y)
        y, y_mask = self.dec1b(y, y_mask)                            # 64 -> 64
        y = self.relu(y)
        y, y_mask = self.dec1c(y, y_mask)                            # 64 -> 64
        y = self.relu(y)

        # Upsample 1/4 -> 1/2 resolution and join the skip from enc1.
        y = upsample_periodic_x(y, x1.shape[2:])
        y_mask = F.interpolate(y_mask, size=mask1.shape[2:], mode="nearest")
        y = torch.cat([y, x1], dim=1)                                # 64 + 32 = 96 channels

        # PartialConv2d.forward, 3x3 at 1/2 resolution.
        y, y_mask = self.dec2(y, y_mask)                             # 96 -> 32
        y = self.relu(y)

        # Upsample 1/2 -> full resolution (the input's exact size, odd grids included).
        y = upsample_periodic_x(y, x.shape[2:])

        # PartialConv2d.forward, 3x3 at full resolution with the original ocean mask:
        # the output, one channel per predicted variable. No ReLU: the change can be negative.
        change, _ = self.dec3(y, mask)                               # 32 -> C_out
        return change


# Backwards-compatible alias for notebooks that used the old misspelling.
LightingWrapper = LightningWrapper
