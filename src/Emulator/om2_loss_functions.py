"""
Composable rollout loss functions for OM2 emulator models.

Each public loss constructor returns a callable with a shared rollout-step
context signature. A training notebook can assemble any list of losses and pass
that list to ``total_rollout_loss`` without changing the rollout loop.
"""

import torch
import torch.nn.functional as F


def step_weight(weight, rollout_step, like):
    """
    Return a scalar tensor weight for the current rollout step.

    ``weight`` may be either a scalar, applied to every rollout step, or a
    sequence/tensor with one entry per rollout step. This lets later forecast
    months be emphasized or de-emphasized without changing the rollout loop.
    Per-step weights are normalised to have mean 1, so they do not need to sum
    to 1 and do not change the overall loss scale just because the schedule has
    larger numbers.
    """
    if torch.is_tensor(weight):
        if weight.ndim == 0:
            return weight.to(device=like.device, dtype=like.dtype)
        if rollout_step is None:
            raise ValueError("Per-step loss weights require rollout_step in the loss context")
        if rollout_step >= weight.numel():
            raise ValueError(
                f"Per-step loss weights have length {weight.numel()}, "
                f"but rollout_step={rollout_step} was requested"
            )
        flat_weight = weight.flatten().to(device=like.device, dtype=like.dtype)
        normaliser = flat_weight.mean()
        if float(normaliser.detach().cpu()) == 0.0:
            raise ValueError("Per-step loss weights must have non-zero mean")
        return flat_weight[rollout_step] / normaliser

    if isinstance(weight, (list, tuple)):
        if rollout_step is None:
            raise ValueError("Per-step loss weights require rollout_step in the loss context")
        if rollout_step >= len(weight):
            raise ValueError(
                f"Per-step loss weights have length {len(weight)}, "
                f"but rollout_step={rollout_step} was requested"
            )
        normaliser = sum(weight) / len(weight)
        if normaliser == 0.0:
            raise ValueError("Per-step loss weights must have non-zero mean")
        return like.new_tensor(weight[rollout_step] / normaliser)

    return like.new_tensor(float(weight))


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


def local_mse_loss(weight=1.0):
    """Return a weighted masked local MSE loss callable."""

    def loss_fn(*, pred_t, target_t, mask, rollout_step=None, **_):
        current_weight = step_weight(weight, rollout_step, pred_t)
        if current_weight == 0.0:
            return pred_t.new_zeros(())
        step_err = (pred_t - target_t) ** 2
        valid_mask = expand_ocean_mask(mask, step_err)
        loss = (step_err * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
        return current_weight * loss

    return loss_fn


def spectral_loss(weight=1.0, eps=1.0e-6):
    """Return a weighted masked log-amplitude spectral loss callable."""

    def loss_fn(*, pred_t, target_t, mask, rollout_step=None, **_):
        current_weight = step_weight(weight, rollout_step, pred_t)
        if current_weight == 0.0:
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
        return current_weight * loss

    return loss_fn


def global_closure_loss(
    weight=1.0,
    surface_flux_sign=1.0,
    closure_min_scale=1.0e20,
    heat_flux_channel_index=0,
    ohc_channel_index=0,
):
    """
    Return a weighted cumulative global heat-closure loss callable.

    ``ohc_channel_index`` picks OHC out of the prediction and the initial state
    when the model predicts several prognostic variables, (B, P, H, W). With a
    single prognostic variable (P = 1) it has no effect.

    ``heat_flux_channel_index`` picks the surface heat flux out of a
    channel-stacked forcing tensor (e.g. heat flux + tau_x + tau_y). Only heat
    flux enters the heat budget; wind stress does not. ``forcing_std`` must be
    the heat-flux variable's own normalisation std.

    The closure is evaluated on ANOMALIES relative to the monthly spatial
    climatology, in PHYSICAL units. For a window from the initial month t0 to
    the target month t0+n:

        sum_xy area * (OHC'[t0+n] - OHC'[t0])
            ~= dt * sum_xy area * ( 1/2 F'[t0] + F'[t0+1] + ... + F'[t0+n-1] + 1/2 F'[t0+n] )

    where ' denotes the anomaly from the climatological mean for that calendar
    month, OHC is in J/m^2 and F (surface heat flux) is in W/m^2.

    Why the flux is averaged over two months (the 1/2 weights):
        OHC and F are both MONTHLY MEANS. The difference between two consecutive
        monthly-mean OHCs is the heat added between the two month centres, i.e.
        during the second half of month t and the first half of month t+1. The
        matching flux for one step is therefore (F[t] + F[t+1]) / 2, not F[t+1]
        alone. Summing that over the window gives the trapezoid weights above:
        the initial and target months count half, every month in between counts
        fully. On raw ACCESS-OM2 output (output360-362, full fields, degC OHC)
        this cuts the one-step global residual from ~34% of the OHC change
        (using F[t+1] only) to ~7%; the remaining ~7% comes from working with
        monthly means rather than snapshots.

        This needs the flux in the INITIAL month, F[t0], which is not part of
        the rollout forcing (that starts at the first target month). It must be
        supplied as ``initial_forcing`` (see total_rollout_loss).

    Why anomalies (i.e. why the climatological mean is NOT added back):
        The model is trained on, and predicts, anomalies. Writing each full field
        as anomaly + climatology, the full-field residual splits into
            [anomaly residual] + [climatology residual].
        The climatology residual depends only on which calendar months are in
        the window, so the model cannot change it. Including it would push the
        predicted anomalies to cancel any non-closure of the climatology (e.g.
        from the monthly-mean timing offset or the fixed 30-day dt), injecting a
        seasonal bias. It would also inflate `scale` below with the large
        seasonal cycle, weakening the penalty on the anomaly errors we care about.

    Why the standard deviation IS still multiplied back in:
        The model works in z-scores, z = anomaly / std, where std varies per grid
        cell and per calendar month, and differs between OHC and heat flux. The
        heat budget is linear in PHYSICAL anomalies, not z-scores:
          * area-integrating raw z-scores would weight each cell by 1/std,
            so quiet regions would dominate the "global" heat content;
          * OHC and flux are divided by different std fields, so equal z-score
            changes do not correspond to equal energy;
          * OHC std changes month to month, so z[target] - z[initial] is not
            an anomaly change even at a single grid point.
        Multiplying by std_t (and only std_t) converts z-scores back to physical
        anomalies (J/m^2 and W/m^2) so both sides of the budget are comparable.

    Note the contrast with `local_mse_loss` / `spectral_loss`: those are fitting
    objectives, not conservation laws, so they deliberately stay in z-score
    space, where every cell and month is weighted roughly equally.
    """

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
        ohc_std,
        forcing_std,
        dt_seconds,
        initial_forcing=None,
        rollout_step=None,
        **_,
    ):
        # ohc_mean / forcing_mean may still arrive via the rollout context; they
        # are swallowed by **_ because the anomaly closure never adds the
        # climatological mean back (see the docstring above).
        current_weight = step_weight(weight, rollout_step, pred_t)
        if current_weight == 0.0:
            return pred_t.new_zeros(())

        # Cell areas (m^2), zeroed over land so land never contributes to the
        # global integrals below.
        area_t = area.to(device=pred_t.device, dtype=pred_t.dtype)
        ocean_mask = mask.to(device=pred_t.device, dtype=pred_t.dtype)
        area_t = area_t * ocean_mask

        # --- OHC anomaly change over the rollout window --------------------
        # ohc_std is the per-month climatological std, tiled along the full time
        # axis, so indexing it with a (B,) tensor of time indices gives the
        # (B, H, W) std field for each sample's month. The initial and target
        # months generally differ, so each state needs its own std_t.
        # With several prognostic variables, keep only the OHC channel.
        if initial_ohc_norm.ndim == 4 and initial_ohc_norm.shape[1] > 1:
            initial_ohc_norm = initial_ohc_norm[:, ohc_channel_index]
        pred_ohc = pred_t[:, ohc_channel_index] if pred_t.ndim == 4 and pred_t.shape[1] > 1 else pred_t
        initial_ohc_norm = squeeze_field_axes(initial_ohc_norm)  # (B, H, W) z-score
        pred_ohc_norm = squeeze_field_axes(pred_ohc)              # (B, H, W) z-score
        initial_ohc_std_t = ohc_std[initial_time_index].to(device=pred_t.device, dtype=pred_t.dtype)
        target_ohc_std_t = ohc_std[target_time_index].to(device=pred_t.device, dtype=pred_t.dtype)

        # z-score * std = physical anomaly (J/m^2). No "+ mean": we stay in
        # anomaly space on purpose.
        initial_ohc_anom = initial_ohc_norm * initial_ohc_std_t
        predicted_ohc_anom = pred_ohc_norm * target_ohc_std_t

        # Area-integrate the anomaly change: J/m^2 * m^2 -> J, one value per sample.
        ohc_change_global = ((predicted_ohc_anom - initial_ohc_anom) * area_t).sum(dim=(-2, -1))

        # --- Accumulated surface heat-flux anomaly over the same window ----
        # Trapezoid rule over the monthly-mean fluxes (see the docstring):
        #   initial month t0            -> weight 1/2  (initial_forcing)
        #   months t0+1 ... t0+n-1      -> weight 1    (forcing_history[:, :-1])
        #   target month t0+n           -> weight 1/2  (forcing_history[:, -1])
        # forcing_history holds the flux for every month from the first target
        # up to the current target, so its last entry is the target month.
        if initial_forcing is None:
            raise ValueError(
                "global_closure_loss needs `initial_forcing`: the forcing in the "
                "initial month (time index target_time_indices[:, 0] - 1). The "
                "two-month flux average uses half of it for the first step. Add it "
                "to the rollout batch and pass it to total_rollout_loss."
            )
        n_history = forcing_history.shape[1]
        flux_terms = [(initial_forcing, initial_time_index, 0.5)] + [
            (
                forcing_history[:, history_step],
                forcing_time_indices[:, history_step],
                0.5 if history_step == n_history - 1 else 1.0,
            )
            for history_step in range(n_history)
        ]

        forcing_integral_global = pred_t.new_zeros(ohc_change_global.shape)
        for flux_norm, flux_time_index, trapezoid_weight in flux_terms:
            # Multi-channel forcing arrives as (B, C, H, W): keep only the heat
            # flux channel, since forcing_std is the heat-flux std and wind
            # stress has no place in a heat budget. Single-channel forcing is
            # already (B, H, W) or (B, 1, H, W) and is left as is.
            if flux_norm.ndim == 4 and flux_norm.shape[1] > 1:
                flux_norm = flux_norm[:, heat_flux_channel_index]
            flux_norm = squeeze_field_axes(flux_norm)  # (B, H, W) z-score
            flux_std_t = forcing_std[flux_time_index].to(device=pred_t.device, dtype=pred_t.dtype)
            # z-score * std = physical flux anomaly (W/m^2); again no "+ mean".
            flux_anom = flux_norm * flux_std_t

            # W/m^2 * m^2 * s -> J, times the trapezoid weight (1/2 at the two
            # ends of the window, 1 in between). surface_flux_sign flips the
            # convention if positive flux means heat leaving the ocean.
            forcing_integral_global = forcing_integral_global + (
                (flux_anom * area_t).sum(dim=(-2, -1))
                * dt_seconds
                * trapezoid_weight
                * surface_flux_sign
            )

        # --- Normalised squared residual -------------------------------------
        # Residual in Joules. Divide by the typical magnitude of the anomaly
        # signal in this batch (detached, so the scale itself is not optimised)
        # to make the penalty dimensionless. closure_min_scale is a floor so
        # windows with near-zero anomalies do not blow the penalty up.
        residual = ohc_change_global - forcing_integral_global
        scale = torch.maximum(
            forcing_integral_global.detach().abs().mean(),
            ohc_change_global.detach().abs().mean(),
        ).clamp_min(closure_min_scale)
        return current_weight * ((residual / scale) ** 2).mean()

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
    initial_forcing=None,
    n_prognostic=1,
):
    """
    Run an autoregressive rollout and average any supplied loss callables.

    ``initial_prior_states`` is (B, n_prior * n_prognostic, H, W): the prior
    states stacked oldest first, each contributing ``n_prognostic`` channels.
    After every step the oldest state is dropped and the prediction appended.

    ``target_sequence`` is either (B, n_steps, H, W) for one prognostic variable
    or (B, n_steps, n_prognostic, H, W).

    ``forcing_sequence`` is either (B, n_steps, H, W) for a single forcing
    variable, or (B, n_steps, C, H, W) for channel-stacked forcing. Each step
    passes the model a (B, C, H, W) forcing tensor either way.

    ``initial_forcing`` is the forcing in the initial month (the month of the
    last prior state, time index target_time_indices[:, 0] - 1), shaped like a
    single step of ``forcing_sequence``: (B, H, W) or (B, C, H, W). The model
    never sees it; it is only used by global_closure_loss for the two-month
    flux average in the first rollout step.
    """
    if not losses:
        raise ValueError("total_rollout_loss requires at least one loss function")

    prior_states = initial_prior_states
    # The most recent prior state (all prognostic variables): the rollout's
    # starting point for the closure term.
    initial_ohc_norm = initial_prior_states[:, -n_prognostic:]
    initial_time_index = None
    if target_time_indices is not None:
        initial_time_index = target_time_indices[:, 0].to(initial_prior_states.device) - 1

    total = initial_prior_states.new_zeros(())

    for step in range(n_steps):
        # (B, n_steps, C, H, W) -> (B, C, H, W); (B, n_steps, H, W) -> (B, 1, H, W).
        if forcing_sequence.ndim == 5:
            forcing_t = forcing_sequence[:, step]
        else:
            forcing_t = forcing_sequence[:, step : step + 1]
        # (B, n_steps, P, H, W) -> (B, P, H, W); (B, n_steps, H, W) -> (B, 1, H, W).
        if target_sequence.ndim == 5:
            target_t = target_sequence[:, step]
        else:
            target_t = target_sequence[:, step : step + 1]
        pred_t = model(prior_states, forcing_t, mask)

        target_time_index = None
        forcing_time_indices = None
        if target_time_indices is not None:
            target_time_index = target_time_indices[:, step].to(pred_t.device)
            forcing_time_indices = target_time_indices[:, : step + 1].to(pred_t.device)

        context = dict(
            rollout_step=step,
            pred_t=pred_t,
            target_t=target_t,
            mask=mask,
            initial_ohc_norm=initial_ohc_norm,
            forcing_history=forcing_sequence[:, : step + 1],
            initial_time_index=initial_time_index,
            target_time_index=target_time_index,
            forcing_time_indices=forcing_time_indices,
            area=area,
            # ohc_mean / forcing_mean are passed through for any loss that wants
            # full physical fields; global_closure_loss ignores them because it
            # works on anomalies (normalised * std only).
            ohc_mean=ohc_mean,
            ohc_std=ohc_std,
            forcing_mean=forcing_mean,
            forcing_std=forcing_std,
            dt_seconds=dt_seconds,
            initial_forcing=initial_forcing,
        )

        for loss_fn in losses:
            total = total + loss_fn(**context)

        # Drop the oldest prior state (n_prognostic channels), append the prediction.
        prior_states = torch.cat([prior_states[:, n_prognostic:], pred_t], dim=1)

    return total / n_steps
