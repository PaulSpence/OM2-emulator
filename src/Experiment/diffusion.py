"""
Generative paradigm: residual diffusion on top of a frozen deterministic model.

After CorrDiff (Mardani et al. 2025, Communications Earth & Environment): a
deterministic emulator predicts the mean next state, and a diffusion model
samples the residual (truth - mean), conditioned on the same inputs and on
that mean. The deterministic part is the shared trunk, so the generative
branch trains only the denoiser.

The denoiser follows EDM (Karras, Aittala, Aila & Laine 2022, NeurIPS): the
preconditioned D(x; sigma) = c_skip x + c_out F(c_in x, c_noise), noise levels
sigma ~ lognormal(p_mean, p_std) in training, and Heun's second-order sampler
on the rho-spaced noise schedule. The noise level reaches the UNet as an extra
constant input channel, log(sigma) / 4.

    diffusion = build_diffusion(cfg, data, trunk_model)
    module    = build_diffusion_module(cfg, diffusion, data)
    trainer   = build_trainer(cfg)          # cfg.window.rollout_schedule unused
    trainer.fit(module, data.train_dl, data.valid_dl)

ResidualDiffusion(prior, forcing, mask) returns one sampled next state, with
the call signature of ForwardEmulator, so the rollouts and the skill test use
it unchanged (the budget constraint is applied after it, as for every model).
"""

import math

import lightning as L
import torch
import torch.nn as nn

from Emulator import UNet

from .config import rollout_spec
from .constraints import build_constraint
from .data import gather_windows
from .models import dirichlet_channels
from .training import check_finite


class ResidualDiffusion(nn.Module):
    def __init__(self, mean_model, denoiser, n_prognostic, settings, sigma_data=1.0):
        super().__init__()
        self.mean_model = mean_model.requires_grad_(False)
        self.denoiser = denoiser
        self.n_prognostic = n_prognostic
        self.settings = settings  # a DiffusionConfig
        self.register_buffer("sigma_data", torch.tensor(float(sigma_data)))

    # ------------------------------------------------------------- the model
    def train(self, mode=True):
        super().train(mode)
        self.mean_model.eval()  # frozen: no dropout/batch-norm updates, ever
        return self

    @torch.no_grad()
    def mean(self, prior, forcing, mask):
        return self.mean_model(prior, forcing, mask).float()

    def denoise(self, noisy, sigma, condition, mask):
        """EDM-preconditioned denoiser D(noisy; sigma). sigma: (B, 1, 1, 1)."""
        sd = self.sigma_data
        c_in = 1.0 / torch.sqrt(sigma**2 + sd**2)
        c_skip = sd**2 / (sigma**2 + sd**2)
        c_out = sigma * sd / torch.sqrt(sigma**2 + sd**2)
        c_noise = (torch.log(sigma) / 4.0).expand(-1, 1, *noisy.shape[-2:])
        x = torch.cat([c_in * noisy, condition, c_noise.to(noisy.dtype)], dim=1)
        mask = mask.to(device=x.device, dtype=x.dtype)
        mask = mask[None, None] if mask.ndim == 2 else mask
        return c_skip * noisy + c_out * self.denoiser(x, mask).float()

    @staticmethod
    def condition(prior, forcing, mean):
        return torch.cat([prior, forcing, mean], dim=1)

    # --------------------------------------------------------------- training
    def loss(self, prior, forcing, target, mask, area):
        """EDM denoising loss on the residual target - mean, area-weighted over the ocean."""
        mean = self.mean(prior, forcing, mask)
        residual = target.float() - mean
        s = self.settings
        sigma = torch.exp(s.p_mean + s.p_std * torch.randn(residual.shape[0], 1, 1, 1, device=residual.device))
        noisy = residual + torch.randn_like(residual) * sigma
        denoised = self.denoise(noisy, sigma, self.condition(prior, forcing, mean), mask)
        sd = self.sigma_data
        weight = (sigma**2 + sd**2) / (sigma * sd) ** 2
        cell = (area * mask).to(residual.dtype)
        cell = cell / cell.sum()
        return (weight * (denoised - residual) ** 2 * cell).sum() / (residual.shape[0] * residual.shape[1])

    # --------------------------------------------------------------- sampling
    def noise_levels(self, device, steps=None):
        """The sampler's decreasing noise levels sigma_0 > ... > sigma_{N-1}, then 0 (N = steps or sampler_steps)."""
        s = self.settings
        n = steps or s.sampler_steps
        i = torch.arange(n, device=device, dtype=torch.float64)
        ramp = i / max(n - 1, 1)
        smax, smin = s.sigma_max ** (1 / s.rho), s.sigma_min ** (1 / s.rho)
        sigmas = (smax + ramp * (smin - smax)) ** s.rho
        return torch.cat([sigmas, sigmas.new_zeros(1)]).float()

    @torch.no_grad()
    def sample_residual(self, condition, mask, shape, steps=None):
        """Heun (EDM Algorithm 1) from pure noise at sigma_max to a residual sample, in `steps` (default sampler_steps)."""
        sigmas = self.noise_levels(condition.device, steps)
        x = torch.randn(shape, device=condition.device) * sigmas[0]
        ones = torch.ones(shape[0], 1, 1, 1, device=condition.device)
        for current, nxt in zip(sigmas[:-1], sigmas[1:]):
            d = (x - self.denoise(x, current * ones, condition, mask)) / current
            x_next = x + (nxt - current) * d
            if nxt > 0:
                d_next = (x_next - self.denoise(x_next, nxt * ones, condition, mask)) / nxt
                x_next = x + (nxt - current) * 0.5 * (d + d_next)
            x = x_next
        return x

    def forward(self, prior, forcing, mask, sampler_steps=None):
        """One sampled next state: the deterministic mean plus a sampled residual."""
        mean = self.mean(prior, forcing, mask)
        residual = self.sample_residual(self.condition(prior, forcing, mean), mask, mean.shape, sampler_steps)
        return (mean + residual).float()


def build_diffusion(cfg, data, mean_model, sigma_data_batches=4):
    """
    The ResidualDiffusion for cfg: mean_model (the trained deterministic trunk,
    frozen) plus a fresh denoiser UNet. sigma_data is the RMS residual of the
    mean model over a few training batches (one-step, true inputs).
    """
    torch.manual_seed(cfg.train.seed)
    n_prior, p, c = cfg.window.n_prior, data.n_prognostic, data.n_forcing
    # Denoiser inputs: noisy residual (P), prior states (n_prior * P), forcing (C), mean (P), noise level (1).
    in_channels = p + n_prior * p + c + p + 1
    # The prior/forcing channels sit after the P residual channels.
    dirichlet = {channel + p: value for channel, value in dirichlet_channels(cfg).items()}
    denoiser = UNet(
        input_channel_count=in_channels,
        output_channel_count=p,
        padding_mode=cfg.model.padding_mode,
        width_multiplier=cfg.diffusion.width_multiplier,
        dirichlet=dirichlet,
    )
    diffusion = ResidualDiffusion(mean_model, denoiser, p, cfg.diffusion)
    diffusion.sigma_data.fill_(residual_rms(mean_model, data, sigma_data_batches))
    return diffusion


@torch.no_grad()
def residual_rms(mean_model, data, n_batches):
    """Area-weighted RMS of (truth - mean_model) over the ocean for one-step training windows."""
    f = data.fields
    device = next(mean_model.parameters()).device
    weight = (f["area"] * f["mask"]).float().to(device)
    weight = weight / weight.sum()
    squared, count = 0.0, 0
    was_training = mean_model.training
    mean_model.eval()
    for i, t0 in enumerate(data.train_dl):
        if i >= n_batches:
            break
        w = data.windows(t0, n_steps=1)
        mean = mean_model(w["prior"].flatten(1, 2).to(device), w["forcing"][:, 0].to(device), f["mask"].to(device))
        error = (w["target"][:, 0].to(device) - mean) ** 2
        squared += float((error * weight).sum())
        count += error.shape[0] * error.shape[1]
    mean_model.train(was_training)
    return math.sqrt(squared / max(count, 1))


class DiffusionModule(L.LightningModule):
    """
    Trains the denoiser of a ResidualDiffusion on one-step residuals. The mean
    model stays frozen.

    Rollout (pushforward) training: with rollout_steps [n_free, 1] (or a
    schedule of them; see WindowConfig) the full model first runs n_free months
    freely from the true state, without gradients: mean + sampled residual, with
    diffusion.pushforward_sampler_steps Heun steps and the budget constraint.
    The denoiser then learns the next month's residual from that drifted state,
    i.e. to correct the drift its own samples cause. Trained on true inputs only,
    a residual model adds independent noise every month and the variance of free
    runs grows without bound.

    Logs train_loss / val_loss; validation is one step from true states with a
    fixed random seed, so epochs are comparable.
    """

    def __init__(self, diffusion, data, cfg):
        super().__init__()
        self.diffusion = diffusion
        self.n_prior = cfg.window.n_prior
        self.n_prognostic = diffusion.n_prognostic
        # (n_free, 1); updated by RolloutSchedule, if used.
        self.train_steps = rollout_spec(cfg.window.rollout_steps)
        self.free_sampler_steps = cfg.diffusion.pushforward_sampler_steps
        self.constraint = build_constraint(cfg, data)
        tr = cfg.train
        self.lr, self.weight_decay = tr.lr, tr.weight_decay
        self.lr_scheduler, self.max_epochs = tr.lr_scheduler, tr.max_epochs
        f = data.fields
        self.register_buffer("prognostic_z", f["prognostic"].float(), persistent=False)
        self.register_buffer("forcing_z", f["forcing"].float(), persistent=False)
        self.register_buffer("mask", f["mask"].float())
        self.register_buffer("area", f["area"].float())

    def push_forward(self, t0, n_free):
        """
        Run the full model n_free months from the true states at t0, without
        gradients, and return the one-step window that follows (gather_windows
        layout) with the model's own states as the prior.
        """
        t0 = torch.as_tensor(t0, dtype=torch.long, device=self.prognostic_z.device)
        prior = gather_windows(self.prognostic_z, self.forcing_z, t0, self.n_prior, 0)["prior"].flatten(1, 2)
        with torch.no_grad():
            for k in range(1, n_free + 1):
                pred = self.diffusion(prior, self.forcing_z[t0 + k], self.mask, sampler_steps=self.free_sampler_steps)
                if self.constraint is not None:
                    pred = self.constraint(pred, prior[:, -self.n_prognostic:], t0 + k - 1, t0 + k)
                prior = torch.cat([prior[:, self.n_prognostic:], pred.to(prior.dtype)], dim=1)
        # Weight copies cached by autocast without gradients must not reach the loss (see EmulatorModule).
        torch.clear_autocast_cache()
        window = gather_windows(self.prognostic_z, self.forcing_z, t0 + n_free, self.n_prior, 1)
        window["prior"] = prior.unflatten(1, (self.n_prior, self.n_prognostic))
        return window

    def _loss(self, t0, n_free=0):
        if n_free:
            w = self.push_forward(t0, n_free)
        else:
            w = gather_windows(self.prognostic_z, self.forcing_z, t0, self.n_prior, 1)
        return self.diffusion.loss(w["prior"].flatten(1, 2), w["forcing"][:, 0], w["target"][:, 0],
                                   self.mask, self.area)

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch, self.train_steps[0])
        check_finite(loss, self.current_epoch)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(batch))
        return loss

    def validation_step(self, batch, batch_idx):
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(batch_idx)
            loss = self._loss(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(batch))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.diffusion.denoiser.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer


def build_diffusion_module(cfg, diffusion, data):
    return DiffusionModule(diffusion, data, cfg)
