"""
Machine Learning modules for OM2 emulator.

This module contains neural network components for building
and training autoencoder models on ACCESS-OM2 data.
"""

import weakref

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L


# =============================================================================
# Land boundary conditions
#
# WHY: the partial-convolution renormalisation (Liu et al. 2018) multiplies each
# output by kernel_area / (number of ocean cells in the window). Next to land
# that factor is up to 9x (3x3) or 16x (4x4), independent of where the learned
# weights sit, so coastal cells get large gains the network must cancel
# exactly. In autoregressive rollouts of the OHC emulator the remaining growing
# mode (~2% per step) sat at the coastlines. Instead, before every convolution
# land cells are FILLED and an ordinary convolution is applied:
#
#   * Neumann (default, every layer): each land cell takes the value of its
#     nearest ocean cell, a zero normal gradient across the coast. This is the
#     physical no-flux condition for tracers (heat and freshwater content). It
#     creates no artificial fronts at the coast, so coastal stencils look like
#     open-ocean ones; for hidden-layer features, which have no physical
#     boundary condition, it is the neutral choice.
#   * Dirichlet (input layer only, per variable): land cells are held at a
#     fixed value in normalised (z-score) units, default 0. Use it for velocity
#     components (u, v): at a wall the normal velocity is 0 every month, so its
#     climatology and anomaly there are 0, i.e. z = 0 is u = 0 (no normal flow).
#     (This z = 0 <=> u = 0 argument is ours, not from the literature.)
#
# Source of the approach: Zhang, Perezhogin, Adcroft & Zanna (2024), "Addressing
# out-of-sample issues in multi-layer convolutional neural-network
# parameterization of mesoscale eddies applied near coastlines", arXiv:2411.01138.
# They replace land values with the nearest ocean values at every layer
# ("replicate padding", approximately a Neumann condition) and find it removes
# coastal artifacts and keeps online ocean simulations stable, whereas zero
# filling at every layer could intensify artifacts and inject energy.
# DLESyM-Ocean (arXiv:2608.11545) similarly imputes ocean fields across land
# before its convolutions.
# =============================================================================

_NEAREST_OCEAN_CACHE = {}


def _mask_key(mask2d):
    """Content key of an (H, W) mask: shape, device and two checksums."""
    h, w = mask2d.shape
    weights = torch.arange(1, h * w + 1, device=mask2d.device, dtype=torch.float64)
    flat = mask2d.reshape(-1).to(torch.float64)
    total, weighted = torch.stack([flat.sum(), (flat * weights).sum()]).tolist()
    return (h, w, str(mask2d.device), total, weighted)


def nearest_ocean_index(mask2d):
    """
    For an (H, W) ocean mask (1 = ocean), the flat index of each cell's nearest
    ocean cell (ocean cells map to themselves), as a LongTensor on the mask's
    device. Distances wrap around in longitude (x is periodic); the north
    tripole fold is not crossed. Computed once per mask (scipy, on the CPU) and
    cached.
    """
    key = _mask_key(mask2d)
    index = _NEAREST_OCEAN_CACHE.get(key)
    if index is None:
        from scipy.ndimage import distance_transform_edt

        h, w = mask2d.shape
        land = mask2d.detach().cpu().numpy() == 0
        flat = np.arange(h * w).reshape(h, w)
        if land.any() and not land.all():
            # Tile three copies side by side so the nearest ocean cell may lie
            # across the longitude seam, then take the middle copy.
            _, (iy, ix) = distance_transform_edt(np.tile(land, (1, 3)), return_indices=True)
            flat = iy[:, w : 2 * w] * w + ix[:, w : 2 * w] % w
        index = torch.as_tensor(flat.reshape(-1), dtype=torch.long, device=mask2d.device)
        if len(_NEAREST_OCEAN_CACHE) > 64:
            _NEAREST_OCEAN_CACHE.clear()
        _NEAREST_OCEAN_CACHE[key] = index
    return index


def fill_land(x, mask, dirichlet=None, index=None):
    """
    Fill the land cells of x (B, C, H, W) before a convolution (see the comment
    block above for why).

    mask      : (H, W), (1, 1, H, W) or (B, 1, H, W) ocean mask, 1 = ocean; one
                mask shared by the batch (the first one is used).
    dirichlet : optional {channel: value}. Those channels are set to `value`
                (z-score units) on land; every other channel gets the Neumann
                fill (nearest ocean value). Ocean cells are never changed.
    index     : optional precomputed nearest_ocean_index(mask). Passing it
                avoids the per-call mask lookup (which synchronises with the GPU).
    """
    mask2d = mask.reshape(-1, *mask.shape[-2:])[0]
    if index is None:
        index = nearest_ocean_index(mask2d)
    channels_last = x.dim() == 4 and not x.is_contiguous() and x.is_contiguous(memory_format=torch.channels_last)
    if channels_last:
        # channels_last: gather along the flattened (H*W) axis of the NHWC
        # storage so the result stays channels_last.
        b, c, h, w = x.shape
        nhwc = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        filled = nhwc.index_select(1, index).view(b, h, w, c).permute(0, 3, 1, 2)
    else:
        filled = x.flatten(-2)[..., index].view_as(x)
    if dirichlet:
        channels = torch.zeros(x.shape[1], dtype=torch.bool, device=x.device)
        values = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device)
        for channel, value in dirichlet.items():
            channels[channel] = True
            values[channel] = value
        land = (mask2d == 0).to(x.device)
        on_dirichlet_land = channels[:, None, None] & land
        filled = torch.where(on_dirichlet_land, values[:, None, None], filled)
        if channels_last:
            filled = filled.contiguous(memory_format=torch.channels_last)
    return filled


class PartialConv2d(nn.Module):
    """
    Mask-aware 2D convolution for ocean fields: fill land, then convolve.

    (The name is historical: this used to be a partial convolution with
    kernel_area / valid_count renormalisation. That renormalisation amplified
    perturbations at coastlines in autoregressive rollouts, so land is now
    filled before an ordinary convolution instead. See the "Land boundary
    conditions" comment block above for the reasoning and the source.)

    forward(x, mask) -> (out, new_mask)
      1. Fill land cells: Neumann (nearest ocean value) for every channel, or
         Dirichlet (fixed z-score value) for the channels in ``dirichlet``.
      2. Pad for the ACCESS-OM2 tripolar grid (see padding_mode).
      3. Ordinary convolution.
      4. new_mask: the same ocean mask for stride 1; for stride > 1, a coarse
         cell is ocean if any fine cell in it is ocean. Land is not dilated:
         it is refilled before every layer.

    Parameters
    ----------
    in_ch, out_ch : int
        Number of input and output channels.
    kernel_size, stride, padding : int
        As for nn.Conv2d (the padding is applied explicitly, see below).
    padding_mode : str
        How the south edge is padded: "replicate" (default) or "zeros". The
        other edges follow the ACCESS-OM2 tripolar grid topology:
          * x (longitude) is padded circularly. With non-periodic padding the
            seam acts as a coastline neither side can see across; in the
            autoregressive OHC emulator 98% of the fastest-growing perturbation
            sat at that seam.
          * the north edge is the tripole fold: row-top cell i neighbours
            row-top cell nx - 1 - i, so the ghost rows above the top row are
            the top rows reversed in x (and in y).
    dirichlet : dict, optional
        {input channel: value in z-score units} for channels with a Dirichlet
        land condition. Only meaningful on the first layer, whose channels are
        physical variables; hidden layers always use the Neumann fill.

    The mask may be (1, 1, H, W), shared by the whole batch; the
    nearest-ocean lookup is computed once per mask and cached. Callers that
    reuse one mask (the UNet) pass ``fill_index`` to skip that lookup.
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, padding_mode="replicate", dirichlet=None):
        super().__init__()

        # Padding is applied in forward: circular in x (longitude is periodic
        # on a global grid), the tripole fold at the north edge, padding_mode
        # at the south edge. The convolution itself does not pad.
        self.padding = padding
        self.y_pad_mode = {"zeros": "constant", "replicate": "replicate", "reflect": "reflect", "circular": "circular"}[padding_mode]
        self.dirichlet = dict(dirichlet or {})
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride)

    def forward(self, x, mask, fill_index=None):
        channels_last = x.dim() == 4 and not x.is_contiguous() and x.is_contiguous(memory_format=torch.channels_last)
        x = fill_land(x, mask, self.dirichlet, index=fill_index)
        if self.padding:
            x = self._pad(x)
            if channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
        out = self.conv(x)
        new_mask = mask if out.shape[-2:] == mask.shape[-2:] else F.adaptive_max_pool2d(mask, out.shape[-2:])
        return out, new_mask

    def _pad(self, t):
        """
        Pad for the ACCESS-OM2 tripolar grid: the north edge folds onto itself
        (the ghost rows above the top row are the top rows reversed in x),
        x is periodic, and the south edge uses padding_mode.
        """
        p = self.padding
        fold = torch.flip(t[..., -p:, :], dims=(-2, -1))
        t = torch.cat([F.pad(t, (0, 0, p, 0), mode=self.y_pad_mode), fold], dim=-2)
        return F.pad(t, (p, p, 0, 0), mode="circular")

def upsample_periodic_x(x, size):
    """
    Bilinear upsampling (align_corners=False) to ``size`` = (height, width),
    treating x (longitude, the last dimension) as periodic so the east and west
    edges interpolate across the seam. Falls back to plain interpolation if the
    target width is not a whole multiple of the input width.
    """
    height, width = size
    scale = width // x.shape[-1]
    if scale * x.shape[-1] != width:
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    x = F.pad(x, (1, 1, 0, 0), mode="circular")
    x = F.interpolate(x, size=(height, width + 2 * scale), mode="bilinear", align_corners=False)
    return x[..., scale:-scale]


class PartialConvStack(nn.Module):
    """
    n_layers 3x3 mask-aware convolutions with ReLUs in between (not after the
    last): the receptive field of one (2 * n_layers + 1)^2 convolution, e.g. a
    7x7 for n_layers=3.

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

    def forward(self, x, mask, fill_index=None):
        for i, layer in enumerate(self.layers):
            if i:
                x = self.relu(x)
            x, mask = layer(x, mask, fill_index)
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

    The output layer works at full resolution on the upsampled decoder
    features only. Feeding it the raw input as well was tried and removed: in
    autoregressive use it learned a sharpening stencil that grew grid-scale
    noise by ~17% per step. A residual wrapper (next = state + output) carries
    the state's grid-scale content forward unchanged instead.

    Land is filled before every convolution (see "Land boundary conditions"
    above). ``dirichlet`` = {input channel: z-score value} gives those input
    channels a Dirichlet land condition at the first layer; all other channels,
    and every hidden layer, use the Neumann (nearest-ocean) fill.
    """

    def __init__(self,
                 input_channel_count=2,
                 output_channel_count=2,
                 latent_processor=None,
                 padding_mode="replicate",
                 width_multiplier=4,
                 dirichlet=None):
        # padding_mode is passed to every PartialConv2d layer; see
        # PartialConv2d for the "replicate" vs "zeros" trade-off.

        super(UNet, self).__init__()

        width1, width2, latent_width = self.channel_widths(input_channel_count, width_multiplier)
        self.latent_channel_count = latent_width

        # ---------- Encoder ----------
        # The only layer that sees physical variables, so the only one with per-variable boundary conditions.
        self.enc1 = PartialConv2d(input_channel_count, width1, kernel_size=4, stride=2, padding=1, padding_mode=padding_mode, dirichlet=dirichlet)
        self.enc2 = PartialConv2d(width1, width2, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode)
        # Bottleneck: three 3x3 layers, the receptive field of one 7x7.
        self.enc3 = PartialConvStack(width2, latent_width, hidden_channels=width2, padding_mode=padding_mode)

        # ---------- Decoder ----------
        # After the first upsample: latent + skip from enc2 (three 3x3 layers, as a 7x7).
        self.dec1 = PartialConvStack(latent_width + width2, width2, hidden_channels=width2, padding_mode=padding_mode)

        # After the second upsample: dec1 + skip from enc1.
        self.dec2 = PartialConv2d(width2 + width1, width1, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)

        # Full resolution: dec2 upsampled.
        self.dec3 = PartialConv2d(width1, output_channel_count, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)

        self.latent_processor = latent_processor or IdentityLatentProcessor()
        self.relu = nn.ReLU(inplace=True)
        # {id(mask): (weakref(mask), mask version, levels)}; see land_levels.
        self._land_cache = {}

    def land_levels(self, mask):
        """
        Ocean mask and nearest-ocean index at each resolution level: full,
        1/2 (after enc1) and 1/4 (after enc2), as [(mask, index), ...]. A
        coarse cell is ocean if any fine cell in it is ocean.

        Computed once per mask tensor and cached: the land mask is fixed, and
        recomputing (or even checking) it in every layer forces the GPU to
        synchronise with the CPU. The cache entry is reused only while the
        same tensor object is alive and unmodified (its version counter is
        unchanged), so a new or edited mask is always recomputed.
        """
        mask = mask.reshape(1, 1, *mask.shape[-2:]) if mask.dim() != 4 else mask[:1]
        base = mask._base if mask._base is not None else mask
        entry = self._land_cache.get(id(base))
        if entry is not None and entry[0]() is base and entry[1] == base._version:
            return entry[2]

        levels = []
        for layer in (None, self.enc1, self.enc2):
            if layer is not None:
                k, s = layer.conv.kernel_size[0], layer.conv.stride[0]
                h, w = ((n + 2 * layer.padding - k) // s + 1 for n in mask.shape[-2:])
                mask = F.adaptive_max_pool2d(mask, (h, w))
            levels.append((mask, nearest_ocean_index(mask[0, 0])))
        if len(self._land_cache) > 8:
            self._land_cache.clear()
        self._land_cache[id(base)] = (weakref.ref(base), base._version, levels)
        return levels

    @staticmethod
    def channel_widths(input_channel_count, width_multiplier=4):
        """(level 1, level 2, bottleneck) channel widths for this many input channels."""
        if width_multiplier < 1:
            raise ValueError(f"width_multiplier must be >= 1, got {width_multiplier}")
        width1 = int(round(width_multiplier * input_channel_count))
        return width1, 2 * width1, 4 * width1

    def encode(self, x, levels):
        """Encoder: returns the latent representation and the skip features."""
        (mask0, index0), (mask1, index1), (mask2, index2) = levels

        x1, _ = self.enc1(x, mask0, index0)
        x1 = self.relu(x1)

        x2, _ = self.enc2(x1, mask1, index1)
        x2 = self.relu(x2)

        latent, _ = self.enc3(x2, mask2, index2)
        latent = self.relu(latent)

        return latent, x1, x2

    def decode(self, x, x1, x2, x0, levels):
        """Decoder with U-Net skip connections; x0 is the full-resolution input."""
        (mask0, index0), (mask1, index1), (mask2, index2) = levels

        # Each decoder level uses the ocean mask at that resolution.
        # ---------- Upsample to enc2 resolution ----------
        x = upsample_periodic_x(x, x2.shape[2:])
        x = torch.cat([x, x2], dim=1)
        x, _ = self.dec1(x, mask2, index2)
        x = self.relu(x)

        # ---------- Upsample to enc1 resolution ----------
        x = upsample_periodic_x(x, x1.shape[2:])
        x = torch.cat([x, x1], dim=1)
        x, _ = self.dec2(x, mask1, index1)
        x = self.relu(x)

        # ---------- Final upsample to input resolution ----------
        # Upsample to the input's exact size (odd grids included).
        x = upsample_periodic_x(x, x0.shape[2:])
        x, _ = self.dec3(x, mask0, index0)
        return x

    def forward(self, x, mask):
        levels = self.land_levels(mask)
        latent, x1, x2 = self.encode(x, levels)
        latent, _ = self.latent_processor(latent, levels[2][0])
        return self.decode(latent, x1, x2, x, levels)


# Backwards-compatible alias for notebooks that used the old misspelling.
LightingWrapper = LightningWrapper
