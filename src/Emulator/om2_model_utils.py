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

    padding_mode : str
        How the grid edges are padded, for both the data and the mask
        convolution: "replicate" (default) or "zeros".

        * "replicate" copies the edge values outwards, avoiding artificial
          zeros near the boundary. It is slow in training on the GPU: PyTorch
          pads explicitly and then calls cuDNN without padding, and for these
          shapes cuDNN picks slow backward kernels (worst for large kernels).
        * "zeros" is ~2x faster per training step and uses ~35% less memory
          (measured on a V100). It is also consistent with the partial
          convolution: the zero-padded mask marks the padded cells as invalid,
          so the kernel_area / mask_sum renormalisation corrects the edges the
          same way it corrects coastlines.

    Notes
    -----
    * The mask convolution is fixed (all weights = 1) and has no gradients.
    * The renormalisation depends only on the mask. When every sample shares
      the land mask, pass it as (1, 1, H, W): the mask convolution then runs
      once and is broadcast over the batch.
    * The masking and the renormalisation are folded into one multiplication
      by a precomputed factor (0 where no valid pixel exists), so each layer
      keeps one fewer full-size activation for the backward pass.
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, padding_mode="replicate"):
        super().__init__()

        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
        )

        # convolution used only to count valid pixels
        self.mask_conv = nn.Conv2d(
            1,
            1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
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

                (batch, 1, height, width), or (1, 1, height, width) to
                share one mask across the batch

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

        with torch.no_grad():
            # count valid pixels in each convolution window
            mask_sum = self.mask_conv(mask)
            valid = mask_sum > 0
            # renormalisation factor, 0 where the window has no valid pixel
            scale = torch.where(valid, self.kernel_area / (mask_sum + self.eps), torch.zeros_like(mask_sum))
            # updated mask
            new_mask = valid.to(mask.dtype)

        # mask the input, convolve, renormalise
        out = self.conv(x * mask) * scale

        return out, new_mask

class PartialConvStack(nn.Module):
    """
    n_layers 3x3 partial convolutions with ReLUs in between (not after the
    last): the receptive field and mask growth of one (2 * n_layers + 1)^2
    partial convolution, e.g. a 7x7 for n_layers=3.

    It replaces large kernels, which cost (k / 3)^2 more per channel pair and
    get slow cuDNN kernels on the V100, while adding depth. in_channels ->
    hidden_channels -> ... -> out_channels.
    """

    def __init__(self, in_channels, out_channels, hidden_channels, n_layers=3, padding_mode="replicate"):
        super().__init__()
        widths = [in_channels] + [hidden_channels] * (n_layers - 1) + [out_channels]
        self.layers = nn.ModuleList(
            PartialConv2d(a, b, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
            for a, b in zip(widths[:-1], widths[1:])
        )
        self.relu = nn.ReLU(inplace=True)
        self.in_channels, self.out_channels = in_channels, out_channels

    def forward(self, x, mask):
        for i, layer in enumerate(self.layers):
            if i:
                x = self.relu(x)
            x, mask = layer(x, mask)
        return x, mask


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


class IdentityLatentProcessor(nn.Module):
    """
    Default latent-space processor used when no refinement module is requested.

    It preserves the existing UNet behaviour while giving the forward pass a
    stable hook for optional latent-space modules.
    """

    def forward(self, latent, latent_mask):
        return latent, latent_mask


class LatentResidualTuner(nn.Module):
    """
    Deterministic latent-space residual refiner for the forward UNet.

    This is not a stochastic denoising-diffusion sampler. It is a small,
    mask-aware residual module that operates on the encoded UNet latent state
    before decoding. The input and output shapes are identical, so it can be
    swapped for another latent processor without changing the decoder contract.
    """

    def __init__(
        self,
        channel_count=64,
        hidden_channel_count=None,
        residual_scale=0.1,
        padding_mode="replicate",
    ):
        super().__init__()

        hidden_channel_count = hidden_channel_count or channel_count
        self.diff1 = PartialConv2d(channel_count, hidden_channel_count, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
        self.diff2 = PartialConv2d(hidden_channel_count, hidden_channel_count, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
        self.diff3 = PartialConv2d(hidden_channel_count, channel_count, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
        self.relu = nn.ReLU(inplace=True)
        self.residual_scale = residual_scale

    def forward(self, latent, latent_mask):
        correction, mask = self.diff1(latent, latent_mask)
        correction = self.relu(correction)

        correction, mask = self.diff2(correction, mask)
        correction = self.relu(correction)

        correction, mask = self.diff3(correction, mask)

        return latent + self.residual_scale * correction, mask


class SpatialResidualHead(nn.Module):
    """
    Mask-aware output-space residual correction head.

    This operates on the full-resolution prediction grid, optionally conditioned
    on the original forward-emulator inputs. Unlike LatentResidualTuner, this
    module directly corrects the predicted OHC spatial pattern.
    """

    def __init__(
        self,
        input_channel_count,
        output_channel_count=1,
        hidden_channel_count=16,
        residual_scale=0.1,
        padding_mode="replicate",
    ):
        super().__init__()

        self.head1 = PartialConv2d(input_channel_count, hidden_channel_count, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
        self.head2 = PartialConv2d(hidden_channel_count, hidden_channel_count, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
        self.head3 = PartialConv2d(hidden_channel_count, output_channel_count, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
        self.relu = nn.ReLU(inplace=True)
        self.residual_scale = residual_scale

    def forward(self, prediction, conditioning, mask):
        if conditioning is None:
            x = prediction
        else:
            x = torch.cat([prediction, conditioning], dim=1)

        correction, mask = self.head1(x, mask)
        correction = self.relu(correction)

        correction, mask = self.head2(correction, mask)
        correction = self.relu(correction)

        correction, mask = self.head3(correction, mask)

        return prediction + self.residual_scale * correction


class UNet(nn.Module):
    """
    Mask-aware U-Net: two stride-2 encoder levels, a bottleneck, and a decoder
    with skip connections at every resolution.

    Channel widths scale with the number of input channels, so adding
    predictors or predicted variables widens the network instead of squeezing
    them through a fixed bottleneck:

        level 1 (1/2 resolution)   : width_multiplier * input_channel_count
        level 2 (1/4 resolution)   : 2 x level 1
        bottleneck (1/4 resolution): 4 x level 1

    width_multiplier must be >= 2 for the first layer to be able to carry every
    signed input field through its ReLU (each needs a +/- pair of channels).

    The bottleneck and the first decoder layer are stacks of three 3x3 partial
    convolutions (PartialConvStack): the receptive field of the original 7x7
    layers at a fraction of the cost.

    The output layer works at full resolution on the upsampled decoder features
    concatenated with the raw input, so grid-scale structure in the inputs
    (e.g. the current state) reaches the output without passing through the
    downsampled levels.
    """

    def __init__(self,
                 input_channel_count=2,
                 output_channel_count=2,
                 latent_processor=None,
                 padding_mode="replicate",
                 width_multiplier=4):
        # padding_mode is passed to every PartialConv2d layer; see
        # PartialConv2d for the "replicate" vs "zeros" trade-off.

        super(UNet, self).__init__()

        width1, width2, latent_width = self.channel_widths(input_channel_count, width_multiplier)
        self.latent_channel_count = latent_width

        # ---------- Encoder ----------
        self.enc1 = PartialConv2d(input_channel_count, width1, kernel_size=4, stride=2, padding=1, padding_mode=padding_mode)
        self.enc2 = PartialConv2d(width1, width2, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode)
        # Bottleneck: three 3x3 layers, the receptive field of one 7x7.
        self.enc3 = PartialConvStack(width2, latent_width, hidden_channels=width2, padding_mode=padding_mode)

        # ---------- Decoder ----------
        # After the first upsample: latent + skip from enc2 (three 3x3 layers, as a 7x7).
        self.dec1 = PartialConvStack(latent_width + width2, width2, hidden_channels=width2, padding_mode=padding_mode)

        # After the second upsample: dec1 + skip from enc1.
        self.dec2 = PartialConv2d(width2 + width1, width1, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)

        # Full resolution: dec2 upsampled + the raw input (full-resolution skip).
        self.dec3 = PartialConv2d(width1 + input_channel_count, output_channel_count, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)

        self.latent_processor = latent_processor or IdentityLatentProcessor()
        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def channel_widths(input_channel_count, width_multiplier=4):
        """(level 1, level 2, bottleneck) channel widths for this many input channels."""
        if width_multiplier < 1:
            raise ValueError(f"width_multiplier must be >= 1, got {width_multiplier}")
        width1 = int(round(width_multiplier * input_channel_count))
        return width1, 2 * width1, 4 * width1

    def encode(self, x, mask):
        """Encoder: returns latent representation, updated mask, and skip features"""

        x1, mask1 = self.enc1(x, mask)
        x1 = self.relu(x1)

        x2, mask2 = self.enc2(x1, mask1)
        x2 = self.relu(x2)

        latent, latent_mask = self.enc3(x2, mask2)
        latent = self.relu(latent)

        return latent, latent_mask, x1, mask1, x2, mask2

    def decode(self, x, mask, x1, mask1, x2, mask2, x0, mask0):
        """Decoder with U-Net skip connections; x0/mask0 are the full-resolution input"""

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
        # Upsample to the input's exact size (odd grids included) and append the
        # raw input as a full-resolution skip connection.
        x = F.interpolate(x, size=x0.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x0], dim=1)

        x, mask = self.dec3(x, mask0)

        return x, mask

    def forward(self, x, mask):
        latent, latent_mask, x1, mask1, x2, mask2 = self.encode(x, mask)
        latent, latent_mask = self.latent_processor(latent, latent_mask)

        reconstructed, final_mask = self.decode(
            latent, latent_mask,
            x1, mask1,
            x2, mask2,
            x, mask,
        )

        return reconstructed


# Backwards-compatible alias for notebooks that used the old misspelling.
LightingWrapper = LightningWrapper
