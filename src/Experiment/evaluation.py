"""
Evaluation: known-closure check, skill-test and control rollouts, metrics.

Rollouts run on the cached normalised fields (no PET pipelines) and return
xarray Datasets in physical units:
    <var>_pred, <var>_truth            full fields
    <var>_pred_anom, <var>_truth_anom  anomalies from the normalisation mean
                                       (the monthly climatology for Spatial_climatology)
Land cells are NaN.
"""

import warnings

import numpy as np
import pandas as pd
import torch
import xarray as xr

from .losses import closure_loss, closure_std_fields


# =============================================================================
# Known closure (what the closure loss scores on the TRUE data)
# =============================================================================

@torch.no_grad()
def check_known_closure(cfg, data, dataloader=None, max_batches=None, warn_threshold=0.25):
    """
    Run every active closure loss on true targets instead of predictions.

    This is the lowest value each closure term can reach with a perfect model.
    Returns {budget: {"mean", "max", "by_lead"}}; warns for any budget whose
    mean exceeds warn_threshold (see issue #43: with monthly means the anomaly
    heat budget scores ~0.5).
    """
    f = data.fields
    closures = {budget: closure_loss(cfg, budget) for budget in cfg.active_closures()}
    dataloader = dataloader or data.train_dl
    common = dict(
        area=f["area"].float(),
        mask=f["mask"].float(),
        closure_std=closure_std_fields(cfg, data),
        dt_seconds=cfg.loss.seconds_per_step,
    )
    n_steps = cfg.window.posterior_steps
    by_lead = {budget: [[] for _ in range(n_steps)] for budget in closures}
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        if not isinstance(batch, dict):  # initial months -> windows
            batch = data.windows(batch, n_steps)
        tti = batch["target_time_index"]
        for step in range(n_steps):
            context = dict(
                initial_state_norm=batch["prior"][:, -1],      # (B, P, H, W) last prior state
                pred_t=batch["target"][:, step],               # the truth
                forcing_history=batch["forcing"][:, : step + 1],
                initial_forcing=batch["initial_forcing"],
                initial_time_index=tti[:, 0] - 1,
                target_time_index=tti[:, step],
                forcing_time_indices=tti[:, : step + 1],
                rollout_step=step,
                **common,
            )
            for budget, closure in closures.items():
                by_lead[budget][step].append(float(closure(**context)))

    results = {}
    for budget, values in by_lead.items():
        all_values = np.concatenate([np.array(v) for v in values])
        result = {
            "mean": float(all_values.mean()),
            "max": float(all_values.max()),
            "by_lead": [float(np.mean(v)) for v in values],
        }
        print(
            f"Known {budget} closure loss on the true data | mean {result['mean']:.3f}, max {result['max']:.3f}\n"
            f"  by lead: " + " ".join(f"{v:.2f}" for v in result["by_lead"])
        )
        if result["mean"] > warn_threshold:
            warnings.warn(
                f"The {budget} closure loss on the TRUE data averages {result['mean']:.3f} (> {warn_threshold}), "
                "so a perfect model would still be penalised by it (see issue #43). Consider a smaller "
                f"{budget}_closure weight."
            )
        results[budget] = result
    return results


# =============================================================================
# Rollouts
# =============================================================================

@torch.no_grad()
def autoregressive_rollout(model, prior, forcing_sequence, mask):
    """
    Free-running rollout from one initial state.

    prior            (n_prior, P, H, W) true initial states, oldest first
    forcing_sequence (N, C, H, W)       forcing for each predicted month
    returns          (N, P, H, W)       predictions (z-scores), on the CPU
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    n_prognostic = prior.shape[1]
    state = prior.flatten(0, 1)[None].to(device)  # (1, n_prior * P, H, W)
    mask = mask.to(device)
    predictions = []
    for k in range(forcing_sequence.shape[0]):
        pred = model(state, forcing_sequence[k : k + 1].to(device), mask)  # (1, P, H, W)
        predictions.append(pred[0].float().cpu())
        state = torch.cat([state[:, n_prognostic:], pred], dim=1)
    model.train(was_training)
    return torch.stack(predictions)


def _to_physical(z, time_indices, data):
    """z-scores (N, P, H, W) at the given time indices -> (anomaly, full field), physical units."""
    f = data.fields
    std = f["prognostic_std"][time_indices]
    mean = f["prognostic_mean"][time_indices]
    anomaly = z * std
    return anomaly, anomaly + mean


def _as_dataset(arrays, time_coord, data, time_dim="time"):
    """{name: (N, P, H, W) tensor} -> Dataset with one variable per (name, prognostic variable)."""
    f = data.fields
    land = f["mask"].numpy() == 0
    coords = {time_dim: time_coord, "latitude": f["latitude"], "longitude": f["longitude"]}
    variables = {}
    for suffix, values in arrays.items():
        for p, name in enumerate(f["prognostic_names"]):
            v = values[:, p].numpy().astype(np.float32)
            v[:, land] = np.nan
            variables[f"{name}_{suffix}"] = xr.DataArray(
                v, dims=(time_dim, "latitude", "longitude"), coords=coords,
                attrs={"units": f["prognostic_units"][p]},
            )
    ds = xr.Dataset(variables)
    ds["area"] = xr.DataArray(f["area"].numpy(), dims=("latitude", "longitude"))
    return ds


def run_skill_test(cfg, model, data):
    """
    Seed from the truth at time.test[0], then free-run with the real forcing
    through time.test[1] + 1 month. Returns a Dataset of predictions, truth and
    the persistence baseline (<var>_persist, <var>_persist_anom: the seed
    month's physical anomaly held fixed).
    """
    f, n_prior = data.fields, cfg.window.n_prior
    t0 = data.index_of(cfg.time.test[0])
    t_last = data.index_of(cfg.time.test[1]) + 1
    if t_last > len(data.months) - 1:
        raise ValueError(f"time.test ends at {cfg.time.test[1]}: its last target is past the time axis")
    if t0 - n_prior + 1 < 0:
        raise ValueError(f"time.test starts at {cfg.time.test[0]}: not enough prior months on the time axis")

    targets = torch.arange(t0 + 1, t_last + 1)
    z_pred = autoregressive_rollout(model, f["prognostic"][t0 - n_prior + 1 : t0 + 1], f["forcing"][targets], f["mask"])
    z_true = f["prognostic"][targets]
    pred_anom, pred = _to_physical(z_pred, targets, data)
    true_anom, true = _to_physical(z_true, targets, data)
    persist_anom = (f["prognostic"][t0] * f["prognostic_std"][t0]).expand_as(true_anom)
    persist = persist_anom + f["prognostic_mean"][targets]
    time = pd.to_datetime([data.months[i] for i in targets.tolist()])
    ds = _as_dataset(
        {"pred": pred, "truth": true, "pred_anom": pred_anom, "truth_anom": true_anom,
         "persist": persist, "persist_anom": persist_anom},
        time, data,
    )
    ds.attrs["description"] = f"Skill test seeded at {cfg.time.test[0]}, {len(targets)} months"
    print(f"Skill test: seeded at {cfg.time.test[0]}, predicted {data.months[t0 + 1]}..{data.months[t_last]} ({len(targets)} months)")
    return ds


def run_control(cfg, model, data):
    """
    Seed from the truth at time.control[0], then free-run with the forcing of
    the control months repeated time.control_repeats times. As in the original
    control run, the forcing for initial month t is the forcing at t+1.
    Returns a Dataset of predictions on a "step" axis (no truth to compare).
    """
    if cfg.time.control is None:
        raise ValueError("time.control is None: no control window configured")
    f, n_prior = data.fields, cfg.window.n_prior
    first, last = data.index_of(cfg.time.control[0]), data.index_of(cfg.time.control[1])
    if first - n_prior + 1 < 0 or last + 1 > len(data.months) - 1:
        raise ValueError("time.control needs n_prior months before it and one month after it on the time axis")

    window = torch.arange(first + 1, last + 2)  # forcing months of one repeat
    targets = window.repeat(cfg.time.control_repeats)
    z_pred = autoregressive_rollout(model, f["prognostic"][first - n_prior + 1 : first + 1], f["forcing"][targets], f["mask"])
    pred_anom, pred = _to_physical(z_pred, targets, data)
    steps = np.arange(1, len(targets) + 1)
    ds = _as_dataset({"pred": pred, "pred_anom": pred_anom}, steps, data, time_dim="step")
    ds = ds.assign_coords(forcing_month=("step", [data.months[i] for i in targets.tolist()]))
    ds.attrs["description"] = (
        f"Control: forcing {cfg.time.control[0]}..{cfg.time.control[1]} (+1 month) "
        f"repeated {cfg.time.control_repeats}x"
    )
    print(f"Control run: {len(targets)} steps ({cfg.time.control_repeats} repeats of {len(window)} months)")
    return ds


# =============================================================================
# Persistence baseline
# =============================================================================

def persistence_rmse(cfg, data, initial_indices=None, n_steps=None):
    """
    Per-variable RMSE of persistence over rollout windows, in the same units
    as the per-epoch validation RMSE (normalised, area-weighted over the
    ocean, averaged over windows and lead times, square-rooted).

    Persistence here holds the initial month's z-score fixed: z(t) = z(t0).
    That is what the residual ForwardEmulator predicts when it outputs zero
    change, so it is the null model for the training metric. (Holding the
    PHYSICAL anomaly fixed instead, z(t0) * std(t0) / std(t), is the usual
    forecast baseline and is what the skill test reports in physical units;
    in z-score units it blows up wherever a cell's std is much smaller in the
    target month than in the initial month.) Defaults: the validation windows
    and the validation rollout length. Returns {variable: rmse}.
    """
    f = data.fields
    indices = data.valid_indices if initial_indices is None else torch.as_tensor(initial_indices)
    n_steps = n_steps or cfg.window.valid_rollout_steps or cfg.window.rollout_steps
    weight = f["area"].float() * f["mask"].float()
    weight = weight / weight.sum()
    squared = torch.zeros(len(f["prognostic_names"]), dtype=torch.float64)
    count = 0
    for t0 in indices.tolist():
        z0 = f["prognostic"][t0]
        for t in range(t0 + 1, t0 + n_steps + 1):
            squared += (((z0 - f["prognostic"][t]) ** 2) * weight).sum(dim=(-2, -1)).double()
            count += 1
    return dict(zip(f["prognostic_names"], torch.sqrt(squared / count).tolist()))


# =============================================================================
# Stability diagnostic
# =============================================================================

@torch.no_grad()
def leading_growth_mode(model, data, month, n_iter=40, eps=1e-2, seed=0):
    """
    Fastest-growing small perturbation of the one-step map around the true
    state at ``month`` (forcing fixed at the following month), by power
    iteration with finite-difference Jacobian-vector products.

    With n_prior > 1 the iteration runs on the stacked prior states (the
    shift-and-append form of the rollout), so ``growth`` is the per-step
    amplification factor of autoregressive rollouts linearised about that
    state: > 1 means small perturbations grow by that factor every step.

    Returns {"growth", "history", "mode", "names", "month"}: ``mode`` is the
    (P, H, W) newest-state part of the leading perturbation, unit RMS, NaN on land.
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    f = data.fields
    n_prior = data.n_prior
    t0 = data.index_of(month)
    prior = f["prognostic"][t0 - n_prior + 1 : t0 + 1].to(device).float()  # (n_prior, P, H, W)
    forcing = f["forcing"][t0 + 1 : t0 + 2].to(device).float()           # (1, C, H, W)
    mask = f["mask"].to(device).float()

    def step(states):
        return model(states.flatten(0, 1)[None], forcing, mask)[0].float()

    base = step(prior)
    generator = torch.Generator().manual_seed(seed)
    v = torch.randn(prior.shape, generator=generator).to(device) * mask
    history = []
    for _ in range(n_iter):
        v = v / v.norm()
        newest = (step(prior + eps * v) - base) / eps
        v = torch.cat([v[1:], newest[None]]) * mask
        history.append(float(v.norm()))
    model.train(was_training)

    tail = history[-min(5, len(history)):]
    growth = float(np.exp(np.mean(np.log(np.maximum(tail, 1e-30)))))
    mode = v[-1] / v[-1].norm() * np.sqrt(mask.sum().item() * v.shape[1])
    mode = mode.cpu().numpy()
    mode[:, f["mask"].numpy() == 0] = np.nan
    return {"growth": growth, "history": history, "mode": mode,
            "names": list(f["prognostic_names"]), "month": str(month)}


# =============================================================================
# Metrics
# =============================================================================

def global_integral(field, area):
    """Area integral over the ocean (land NaN -> 0): J/m^2 -> J, etc."""
    return (field.fillna(0.0) * area).sum(("latitude", "longitude"))


def global_rmse(pred, truth, area):
    """Area-weighted RMSE over the ocean, one value per time."""
    ocean_area = area.where(pred.isel({pred.dims[0]: 0}).notnull(), 0.0)
    squared = ((pred - truth) ** 2).fillna(0.0)
    return np.sqrt((squared * ocean_area).sum(("latitude", "longitude")) / ocean_area.sum())
