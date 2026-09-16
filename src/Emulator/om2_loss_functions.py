"""
Composable rollout loss functions for OM2 emulator models.

Each public loss constructor returns a callable with a shared rollout-step
context signature. A training notebook can assemble any list of losses and pass
that list to ``total_rollout_loss`` without changing the rollout loop.
"""

import torch
import torch.nn.functional as F


def expand_ocean_mask(mask, like):
    valid_mask = mask.to(device=like.device, dtype=like.dtype)
    if valid_mask.ndim == 2:
        valid_mask = valid_mask.unsqueeze(0).unsqueeze(0)
    elif valid_mask.ndim == 3:
        valid_mask = valid_mask.unsqueeze(1)
    return valid_mask.expand_as(like)


def squeeze_field_axes(field):
    """Drop singleton variable/channel axes, preserving batch and spatial axes."""
    while field.ndim > 3:
        squeezed = False
        for dim in range(1, field.ndim - 2):
            if field.shape[dim] == 1:
                field = field.squeeze(dim)
                squeezed = True
                break
        if not squeezed:
            break
    return field


def physical_field(normalised, mean_lookup, std_lookup, time_index):
    normalised = squeeze_field_axes(normalised)
    mean_t = mean_lookup[time_index].to(device=normalised.device, dtype=normalised.dtype)
    std_t = std_lookup[time_index].to(device=normalised.device, dtype=normalised.dtype)
    return normalised * std_t + mean_t


def local_mse_loss(weight=1.0):
    """Return a weighted masked local MSE loss callable."""
    weight = float(weight)

    def loss_fn(*, pred_t, target_t, mask, **_):
        if weight == 0.0:
            return pred_t.new_zeros(())
        step_err = (pred_t - target_t) ** 2
        valid_mask = expand_ocean_mask(mask, step_err)
        loss = (step_err * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
        return weight * loss

    return loss_fn


def spectral_loss(weight=1.0, eps=1.0e-6):
    """Return a weighted masked log-amplitude spectral loss callable."""
    weight = float(weight)

    def loss_fn(*, pred_t, target_t, mask, **_):
        if weight == 0.0:
            return pred_t.new_zeros(())

        valid_mask = expand_ocean_mask(mask, pred_t)
        ocean_count = valid_mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)

        pred_mean = (pred_t * valid_mask).sum(dim=(-2, -1), keepdim=True) / ocean_count
        target_mean = (target_t * valid_mask).sum(dim=(-2, -1), keepdim=True) / ocean_count

        pred_anom = squeeze_field_axes((pred_t - pred_mean) * valid_mask)
        target_anom = squeeze_field_axes((target_t - target_mean) * valid_mask)

        pred_amp = torch.fft.rfft2(pred_anom, norm="ortho").abs()
        target_amp = torch.fft.rfft2(target_anom, norm="ortho").abs()

        loss = F.mse_loss(torch.log1p(pred_amp + eps), torch.log1p(target_amp + eps))
        return weight * loss

    return loss_fn


def global_closure_loss(
    weight=1.0,
    surface_flux_sign=1.0,
    closure_min_scale=1.0e20,
):
    """Return a weighted cumulative global heat-closure loss callable."""
    weight = float(weight)

    def loss_fn(
        *,
        initial_ohc_norm,
        pred_t,
        forcing_history,
        initial_time_index,
        target_time_index,
        forcing_time_indices,
        area,
        mask,
        ohc_mean,
        ohc_std,
        forcing_mean,
        forcing_std,
        dt_seconds,
        **_,
    ):
        if weight == 0.0:
            return pred_t.new_zeros(())

        area_t = area.to(device=pred_t.device, dtype=pred_t.dtype)
        ocean_mask = mask.to(device=pred_t.device, dtype=pred_t.dtype)
        area_t = area_t * ocean_mask

        initial_ohc = physical_field(initial_ohc_norm, ohc_mean, ohc_std, initial_time_index)
        predicted_ohc = physical_field(pred_t, ohc_mean, ohc_std, target_time_index)
        ohc_change_global = ((predicted_ohc - initial_ohc) * area_t).sum(dim=(-2, -1))

        forcing_integral_global = pred_t.new_zeros(ohc_change_global.shape)
        for history_step in range(forcing_history.shape[1]):
            forcing_t = physical_field(
                forcing_history[:, history_step],
                forcing_mean,
                forcing_std,
                forcing_time_indices[:, history_step],
            )
            forcing_integral_global = forcing_integral_global + (
                (forcing_t * area_t).sum(dim=(-2, -1)) * dt_seconds * surface_flux_sign
            )

        residual = ohc_change_global - forcing_integral_global
        scale = torch.maximum(
            forcing_integral_global.detach().abs().mean(),
            ohc_change_global.detach().abs().mean(),
        ).clamp_min(closure_min_scale)
        return weight * ((residual / scale) ** 2).mean()

    return loss_fn


def total_rollout_loss(
    model,
    initial_prior_states,
    forcing_sequence,
    target_sequence,
    mask,
    losses,
    n_steps=4,
    target_time_indices=None,
    area=None,
    ohc_mean=None,
    ohc_std=None,
    forcing_mean=None,
    forcing_std=None,
    dt_seconds=30 * 24 * 60 * 60,
):
    """Run an autoregressive rollout and average any supplied loss callables."""
    if not losses:
        raise ValueError("total_rollout_loss requires at least one loss function")

    prior_states = initial_prior_states
    initial_ohc_norm = initial_prior_states[:, 1:2]
    initial_time_index = None
    if target_time_indices is not None:
        initial_time_index = target_time_indices[:, 0].to(initial_prior_states.device) - 1

    total = initial_prior_states.new_zeros(())

    for step in range(n_steps):
        forcing_t = forcing_sequence[:, step : step + 1]
        target_t = target_sequence[:, step : step + 1]
        pred_t = model(prior_states, forcing_t, mask)

        target_time_index = None
        forcing_time_indices = None
        if target_time_indices is not None:
            target_time_index = target_time_indices[:, step].to(pred_t.device)
            forcing_time_indices = target_time_indices[:, : step + 1].to(pred_t.device)

        context = dict(
            pred_t=pred_t,
            target_t=target_t,
            mask=mask,
            initial_ohc_norm=initial_ohc_norm,
            forcing_history=forcing_sequence[:, : step + 1],
            initial_time_index=initial_time_index,
            target_time_index=target_time_index,
            forcing_time_indices=forcing_time_indices,
            area=area,
            ohc_mean=ohc_mean,
            ohc_std=ohc_std,
            forcing_mean=forcing_mean,
            forcing_std=forcing_std,
            dt_seconds=dt_seconds,
        )

        for loss_fn in losses:
            total = total + loss_fn(**context)

        prior_states = torch.cat([prior_states[:, 1:2], pred_t], dim=1)

    return total / n_steps
