"""
Composable rollout loss functions for OM2 emulator models.

Each public loss constructor returns a callable with a shared rollout-step
context signature. A training notebook can assemble any list of losses and pass
that list to ``total_rollout_loss`` without changing the rollout loop.
"""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


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


def budget_closure_loss(
    weight=1.0,
    budget=None,
    content_channel_index=0,
    flux_channel_index=0,
    surface_flux_sign=1.0,
    min_scale=1.0e20,
):
    """
    Return a weighted cumulative global budget-closure loss callable.

    A budget pairs a vertically integrated CONTENT (a prognostic variable, per
    unit area) with the SURFACE FLUX that changes it (a forcing variable, per
    unit area per second), e.g.

        heat       : OHC (J/m^2)                and surface heat flux (W/m^2)
        freshwater : freshwater content (kg/m^2) and surface freshwater flux (kg/m^2/s)

    The flux must be in content units per second; no unit conversion is applied.

    ``content_channel_index`` picks the content out of the prediction and the
    initial state when the model predicts several prognostic variables,
    (B, P, H, W). ``flux_channel_index`` picks the flux out of channel-stacked
    forcing (B, C, H, W). With a single channel neither has any effect.

    Normalisation std fields (to turn z-scores into physical anomalies):
        budget=None : the rollout context's ``ohc_std`` (content) and
                      ``forcing_std`` (flux), as in the original heat-only loss.
        budget=name : ``closure_std[name] = (content_std, flux_std)`` from the
                      rollout context, so several budgets can be used at once.
    Each std is (T, H, W), indexed with the sample's time indices.

    ``min_scale`` is a floor (in content units x m^2, e.g. J or kg) on the
    normalisation scale of the residual; see the end of loss_fn.

    The closure is evaluated on ANOMALIES relative to the monthly spatial
    climatology, in PHYSICAL units. For a window from the initial month t0 to
    the target month t0+n:

        sum_xy area * (C'[t0+n] - C'[t0])
            ~= dt * sum_xy area * ( 1/2 F'[t0] + F'[t0+1] + ... + F'[t0+n-1] + 1/2 F'[t0+n] )

    where ' denotes the anomaly from the climatological mean for that calendar
    month, C is the content and F the surface flux (times surface_flux_sign).

    Why the flux is averaged over two months (the 1/2 weights):
        C and F are both MONTHLY MEANS. The difference between two consecutive
        monthly-mean contents is what was added between the two month centres,
        i.e. during the second half of month t and the first half of month t+1.
        The matching flux for one step is therefore (F[t] + F[t+1]) / 2, not
        F[t+1] alone. Summing that over the window gives the trapezoid weights
        above: the initial and target months count half, every month in between
        counts fully. For heat, on raw ACCESS-OM2 output (output360-362, full
        fields, degC OHC) this cuts the one-step global residual from ~34% of
        the OHC change (using F[t+1] only) to ~7%; the remaining ~7% comes from
        working with monthly means rather than snapshots.

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
        cell and per calendar month, and differs between content and flux. The
        budget is linear in PHYSICAL anomalies, not z-scores:
          * area-integrating raw z-scores would weight each cell by 1/std,
            so quiet regions would dominate the "global" content;
          * content and flux are divided by different std fields, so equal
            z-score changes do not correspond to equal amounts;
          * the content std changes month to month, so z[target] - z[initial]
            is not an anomaly change even at a single grid point.
        Multiplying by std_t (and only std_t) converts z-scores back to physical
        anomalies so both sides of the budget are comparable.

    Note the contrast with `local_mse_loss` / `spectral_loss`: those are fitting
    objectives, not conservation laws, so they deliberately stay in z-score
    space, where every cell and month is weighted roughly equally.
    """
    name = "global_closure_loss" if budget is None else f"{budget} closure loss"

    def loss_fn(
        *,
        pred_t,
        forcing_history,
        initial_time_index,
        target_time_index,
        forcing_time_indices,
        area,
        mask,
        dt_seconds,
        initial_state_norm=None,
        initial_ohc_norm=None,
        ohc_std=None,
        forcing_std=None,
        closure_std=None,
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

        if budget is None:
            content_std, flux_std = ohc_std, forcing_std
        else:
            if closure_std is None or budget not in closure_std:
                raise ValueError(
                    f"The {name} needs closure_std[{budget!r}] = (content_std, flux_std) "
                    "in the rollout context (see total_rollout_loss)."
                )
            content_std, flux_std = closure_std[budget]

        # Cell areas (m^2), zeroed over land so land never contributes to the
        # global integrals below.
        area_t = area.to(device=pred_t.device, dtype=pred_t.dtype)
        ocean_mask = mask.to(device=pred_t.device, dtype=pred_t.dtype)
        area_t = area_t * ocean_mask

        # --- Content anomaly change over the rollout window ----------------
        # content_std is the per-month climatological std, tiled along the full
        # time axis, so indexing it with a (B,) tensor of time indices gives the
        # (B, H, W) std field for each sample's month. The initial and target
        # months generally differ, so each state needs its own std_t.
        # With several prognostic variables, keep only the content channel.
        # initial_ohc_norm is the older name of initial_state_norm.
        initial_norm = initial_state_norm if initial_state_norm is not None else initial_ohc_norm
        if initial_norm.ndim == 4 and initial_norm.shape[1] > 1:
            initial_norm = initial_norm[:, content_channel_index]
        pred_content = pred_t[:, content_channel_index] if pred_t.ndim == 4 and pred_t.shape[1] > 1 else pred_t
        initial_norm = squeeze_field_axes(initial_norm)        # (B, H, W) z-score
        pred_content_norm = squeeze_field_axes(pred_content)   # (B, H, W) z-score
        initial_std_t = content_std[initial_time_index].to(device=pred_t.device, dtype=pred_t.dtype)
        target_std_t = content_std[target_time_index].to(device=pred_t.device, dtype=pred_t.dtype)

        # z-score * std = physical anomaly (e.g. J/m^2). No "+ mean": we stay
        # in anomaly space on purpose.
        initial_anom = initial_norm * initial_std_t
        predicted_anom = pred_content_norm * target_std_t

        # Area-integrate the anomaly change (e.g. J/m^2 * m^2 -> J), one value per sample.
        content_change_global = ((predicted_anom - initial_anom) * area_t).sum(dim=(-2, -1))

        # --- Accumulated surface-flux anomaly over the same window ----------
        # Trapezoid rule over the monthly-mean fluxes (see the docstring):
        #   initial month t0            -> weight 1/2  (initial_forcing)
        #   months t0+1 ... t0+n-1      -> weight 1    (forcing_history[:, :-1])
        #   target month t0+n           -> weight 1/2  (forcing_history[:, -1])
        # forcing_history holds the flux for every month from the first target
        # up to the current target, so its last entry is the target month.
        if initial_forcing is None:
            raise ValueError(
                f"The {name} needs `initial_forcing`: the forcing in the "
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

        flux_integral_global = pred_t.new_zeros(content_change_global.shape)
        for flux_norm, flux_time_index, trapezoid_weight in flux_terms:
            # Multi-channel forcing arrives as (B, C, H, W): keep only this
            # budget's flux channel, since flux_std is that variable's std and
            # the other forcings (e.g. wind stress) have no place in the budget.
            # Single-channel forcing is already (B, H, W) or (B, 1, H, W).
            if flux_norm.ndim == 4 and flux_norm.shape[1] > 1:
                flux_norm = flux_norm[:, flux_channel_index]
            flux_norm = squeeze_field_axes(flux_norm)  # (B, H, W) z-score
            flux_std_t = flux_std[flux_time_index].to(device=pred_t.device, dtype=pred_t.dtype)
            # z-score * std = physical flux anomaly (e.g. W/m^2); again no "+ mean".
            flux_anom = flux_norm * flux_std_t

            # e.g. W/m^2 * m^2 * s -> J, times the trapezoid weight (1/2 at the
            # two ends of the window, 1 in between). surface_flux_sign flips the
            # convention if positive flux means leaving the ocean.
            flux_integral_global = flux_integral_global + (
                (flux_anom * area_t).sum(dim=(-2, -1))
                * dt_seconds
                * trapezoid_weight
                * surface_flux_sign
            )

        # --- Normalised squared residual -------------------------------------
        # Residual in content units x m^2 (e.g. J). Divide by the typical
        # magnitude of the anomaly signal in this batch (detached, so the scale
        # itself is not optimised) to make the penalty dimensionless. min_scale
        # is a floor so windows with near-zero anomalies do not blow it up.
        residual = content_change_global - flux_integral_global
        scale = torch.maximum(
            flux_integral_global.detach().abs().mean(),
            content_change_global.detach().abs().mean(),
        ).clamp_min(min_scale)
        return current_weight * ((residual / scale) ** 2).mean()

    return loss_fn


def heat_closure_loss(
    weight=1.0,
    ohc_channel_index=0,
    heat_flux_channel_index=0,
    surface_flux_sign=1.0,
    min_scale=1.0e20,
    budget="heat",
):
    """
    Global heat-budget closure: OHC (J/m^2) against the surface heat flux
    (W/m^2, positive into the ocean for surface_flux_sign=+1). min_scale in J.
    Reads closure_std[budget] from the rollout context; see budget_closure_loss.
    """
    return budget_closure_loss(
        weight=weight,
        budget=budget,
        content_channel_index=ohc_channel_index,
        flux_channel_index=heat_flux_channel_index,
        surface_flux_sign=surface_flux_sign,
        min_scale=min_scale,
    )


def freshwater_closure_loss(
    weight=1.0,
    freshwater_content_channel_index=0,
    freshwater_flux_channel_index=0,
    surface_flux_sign=1.0,
    min_scale=1.0e12,
    budget="freshwater",
):
    """
    Global freshwater-budget closure: freshwater content (kg/m^2, relative to
    a reference salinity) against the surface freshwater flux (kg/m^2/s,
    positive into the ocean for surface_flux_sign=+1). min_scale in kg.
    Reads closure_std[budget] from the rollout context; see budget_closure_loss.
    """
    return budget_closure_loss(
        weight=weight,
        budget=budget,
        content_channel_index=freshwater_content_channel_index,
        flux_channel_index=freshwater_flux_channel_index,
        surface_flux_sign=surface_flux_sign,
        min_scale=min_scale,
    )


def global_closure_loss(
    weight=1.0,
    surface_flux_sign=1.0,
    closure_min_scale=1.0e20,
    heat_flux_channel_index=0,
    ohc_channel_index=0,
):
    """
    The original heat-only closure, kept for the older notebooks: uses the
    rollout context's ``ohc_std`` and ``forcing_std`` (the heat-flux std).
    New code should use heat_closure_loss / freshwater_closure_loss.
    """
    return budget_closure_loss(
        weight=weight,
        budget=None,
        content_channel_index=ohc_channel_index,
        flux_channel_index=heat_flux_channel_index,
        surface_flux_sign=surface_flux_sign,
        min_scale=closure_min_scale,
    )


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
    closure_std=None,
    checkpoint_steps=False,
    steps_in_memory=0,
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

    ``closure_std`` is {budget: (content_std, flux_std)} for the closure losses
    built with a budget name (heat_closure_loss, freshwater_closure_loss, ...).
    ``ohc_std`` / ``forcing_std`` serve the original global_closure_loss.

    ``checkpoint_steps=True`` recomputes each step's model activations during
    the backward pass instead of keeping them (gradient checkpointing): peak
    memory is about one step's activations instead of n_steps', for roughly
    one extra forward pass of compute. The loss and gradients are unchanged.
    ``steps_in_memory`` keeps the activations of the last that many steps
    (no recompute for them) and checkpoints only the earlier ones: memory for
    about steps_in_memory + 1 steps, with proportionally less recompute.
    """
    if not losses:
        raise ValueError("total_rollout_loss requires at least one loss function")

    prior_states = initial_prior_states
    # The most recent prior state (all prognostic variables): the rollout's
    # starting point for the closure terms.
    initial_state_norm = initial_prior_states[:, -n_prognostic:]
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
        if checkpoint_steps and step < n_steps - steps_in_memory and torch.is_grad_enabled():
            pred_t = checkpoint(model, prior_states, forcing_t, mask, use_reentrant=False)
        else:
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
            initial_state_norm=initial_state_norm,
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
            closure_std=closure_std,
            dt_seconds=dt_seconds,
            initial_forcing=initial_forcing,
        )

        for loss_fn in losses:
            total = total + loss_fn(**context)

        # Drop the oldest prior state (n_prognostic channels), append the prediction.
        prior_states = torch.cat([prior_states[:, n_prognostic:], pred_t], dim=1)

    return total / n_steps
