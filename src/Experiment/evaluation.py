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

from Emulator import zonal_wavenumber_bands
from Emulator.om2_loss_functions import SPECTRAL_BANDS_PER_DECADE

from .config import rollout_spec
from .constraints import build_constraint
from .data import _open_time_axis, _rename_xy
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
        month_seconds=f["month_seconds"],
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
def autoregressive_rollout(model, prior, forcing_sequence, mask, target_indices=None, initial_index=None,
                           constraint=None, members=1):
    """
    Free-running rollout from one initial state, for ``members`` members at
    once (each a separate sample for a stochastic model; 1 for a deterministic
    one).

    prior            (n_prior, P, H, W) true initial states, oldest first
    forcing_sequence (N, C, H, W)       forcing for each predicted month
    target_indices   (N,)               time index of each predicted month, and
    initial_index    int                that of the last prior state: needed by
    constraint                          (e.g. GlobalBudgetConstraint), applied after every step
    returns          (members, N, P, H, W) predictions (z-scores), on the CPU
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    n_prognostic = prior.shape[1]
    state = prior.flatten(0, 1)[None].to(device).repeat(members, 1, 1, 1)  # (M, n_prior * P, H, W)
    mask = mask.to(device)
    if constraint is not None:
        if target_indices is None or initial_index is None:
            raise ValueError("A constraint needs target_indices and initial_index")
        constraint = constraint.to(device)
        previous = torch.full((members,), int(initial_index), dtype=torch.long, device=device)
    predictions = []
    for k in range(forcing_sequence.shape[0]):
        forcing = forcing_sequence[k : k + 1].to(device).expand(members, -1, -1, -1)
        pred = model(state, forcing, mask)  # (M, P, H, W)
        if constraint is not None:
            target = torch.full_like(previous, int(target_indices[k]))
            pred = constraint(pred, state[:, -n_prognostic:], previous, target)
            previous = target
        predictions.append(pred.float().cpu())
        state = torch.cat([state[:, n_prognostic:], pred.to(state.dtype)], dim=1)
    model.train(was_training)
    return torch.stack(predictions, dim=1)


def default_members(cfg):
    """Rollout members for cfg's paradigm: 1 if deterministic, else ensemble.eval_members."""
    return 1 if cfg.paradigm == "deterministic" else cfg.ensemble.eval_members


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


def run_skill_test(cfg, model, data, members=None):
    """
    Seed from the truth at time.test[0], then free-run with the real forcing
    through time.test[1] + 1 month, with the global budget constraint. Returns
    a Dataset of predictions, truth and the persistence baseline
    (<var>_persist, <var>_persist_anom: the seed month's physical anomaly held
    fixed).

    members (default: default_members(cfg)) rollouts are run. <var>_pred and
    <var>_pred_anom are the first member: one realisation, whose variability
    and spectra can be compared with the truth's. <var>_pred_mean(_anom) is the
    ensemble mean, for skill scores; with more than one member,
    <var>_ens_anom (member, time, latitude, longitude) holds every member.
    """
    f, n_prior = data.fields, cfg.window.n_prior
    t0 = data.index_of(cfg.time.test[0])
    t_last = data.index_of(cfg.time.test[1]) + 1
    if t_last > len(data.months) - 1:
        raise ValueError(f"time.test ends at {cfg.time.test[1]}: its last target is past the time axis")
    if t0 - n_prior + 1 < 0:
        raise ValueError(f"time.test starts at {cfg.time.test[0]}: not enough prior months on the time axis")

    members = members or default_members(cfg)
    targets = torch.arange(t0 + 1, t_last + 1)
    z_ens = autoregressive_rollout(model, f["prognostic"][t0 - n_prior + 1 : t0 + 1], f["forcing"][targets], f["mask"],
                                   targets, t0, build_constraint(cfg, data), members)
    z_true = f["prognostic"][targets]
    pred_anom, pred = _to_physical(z_ens[0], targets, data)
    mean_anom, mean = _to_physical(z_ens.mean(dim=0), targets, data)
    true_anom, true = _to_physical(z_true, targets, data)
    persist_anom = (f["prognostic"][t0] * f["prognostic_std"][t0]).expand_as(true_anom)
    persist = persist_anom + f["prognostic_mean"][targets]
    time = pd.to_datetime([data.months[i] for i in targets.tolist()])
    ds = _as_dataset(
        {"pred": pred, "truth": true, "pred_anom": pred_anom, "truth_anom": true_anom,
         "pred_mean": mean, "pred_mean_anom": mean_anom,
         "persist": persist, "persist_anom": persist_anom},
        time, data,
    )
    if members > 1:
        land = f["mask"].numpy() == 0
        std = f["prognostic_std"][targets]
        for p, name in enumerate(f["prognostic_names"]):
            ens = (z_ens[:, :, p] * std[:, p]).numpy().astype(np.float32)
            ens[..., land] = np.nan
            ds[f"{name}_ens_anom"] = xr.DataArray(
                ens, dims=("member", "time", "latitude", "longitude"),
                coords={"time": time, "latitude": f["latitude"], "longitude": f["longitude"]},
                attrs={"units": f["prognostic_units"][p]},
            )
    ds.attrs["description"] = f"Skill test seeded at {cfg.time.test[0]}, {len(targets)} months, {members} member(s)"
    ds.attrs["paradigm"] = cfg.paradigm
    ds.attrs["members"] = members
    print(f"Skill test: seeded at {cfg.time.test[0]}, predicted {data.months[t0 + 1]}..{data.months[t_last]} ({len(targets)} months)")
    return ds


def run_control(cfg, model, data):
    """
    Seed from the truth at time.control[0], then free-run with the forcing of
    the control months repeated time.control_repeats times. As in the original
    control run, the forcing for initial month t is the forcing at t+1.
    Returns a Dataset of predictions on a "step" axis (no truth to compare):
    one member, with the global budget constraint.
    """
    if cfg.time.control is None:
        raise ValueError("time.control is None: no control window configured")
    f, n_prior = data.fields, cfg.window.n_prior
    first, last = data.index_of(cfg.time.control[0]), data.index_of(cfg.time.control[1])
    if first - n_prior + 1 < 0 or last + 1 > len(data.months) - 1:
        raise ValueError("time.control needs n_prior months before it and one month after it on the time axis")

    window = torch.arange(first + 1, last + 2)  # forcing months of one repeat
    targets = window.repeat(cfg.time.control_repeats)
    z_pred = autoregressive_rollout(model, f["prognostic"][first - n_prior + 1 : first + 1], f["forcing"][targets], f["mask"],
                                    targets, first, build_constraint(cfg, data))[0]
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
    n_steps = n_steps or cfg.window.valid_rollout_steps or sum(rollout_spec(cfg.window.rollout_steps))
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


def climatology_rmse(cfg, data, initial_indices=None, n_steps=None):
    """
    Per-variable RMSE of the climatology forecast (zero anomaly, z = 0) over
    rollout windows, in the units of persistence_rmse: the RMS of the target
    z-scores. Where nothing is predictable it beats persistence by ~sqrt(2),
    so it is the stricter baseline at long leads. Same defaults as
    persistence_rmse. Returns {variable: rmse}.
    """
    f = data.fields
    indices = data.valid_indices if initial_indices is None else torch.as_tensor(initial_indices)
    n_steps = n_steps or cfg.window.valid_rollout_steps or sum(rollout_spec(cfg.window.rollout_steps))
    weight = f["area"].float() * f["mask"].float()
    weight = weight / weight.sum()
    squared = torch.zeros(len(f["prognostic_names"]), dtype=torch.float64)
    count = 0
    for t0 in indices.tolist():
        for t in range(t0 + 1, t0 + n_steps + 1):
            squared += ((f["prognostic"][t] ** 2) * weight).sum(dim=(-2, -1)).double()
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


def _area_mean(field, ocean_area):
    return (field.fillna(0.0) * ocean_area).sum(("latitude", "longitude")) / ocean_area.sum()


def skill_diagnostics(skill, variable):
    """
    Pattern, amplitude and error of the skill test for one variable, one value
    per month (area-weighted over the ocean, on anomalies):

      rmse_<emulator|persistence|climatology>  climatology = zero anomaly, so
                                               its RMSE is the truth's RMS anomaly
      acc_<emulator|persistence>   anomaly correlation (uncentred): the pattern
                                   match whatever its amplitude
      amplitude_<emulator|truth>   RMS anomaly: below the truth = fading
                                   (blurring), above = noise or drift growing

    Persistence keeps the seed month's anomaly. Where nothing is predictable,
    zero anomaly beats persistence by about sqrt(2), so beating persistence
    alone is not skill.
    """
    pred, truth, persist = (skill[f"{variable}_{s}_anom"] for s in ("pred", "truth", "persist"))
    area = skill["area"].where(truth.isel(time=0).notnull(), 0.0)
    mean = lambda x: _area_mean(x, area)
    rms = lambda x: np.sqrt(mean(x**2))
    acc = lambda x: mean(x * truth) / np.sqrt(mean(x**2) * mean(truth**2))
    return xr.Dataset({
        "rmse_emulator": rms(pred - truth),
        "rmse_persistence": rms(persist - truth),
        "rmse_climatology": rms(truth),
        "acc_emulator": acc(pred),
        "acc_persistence": acc(persist),
        "amplitude_emulator": rms(pred),
        "amplitude_truth": rms(truth),
    })


def skill_summary(skill, variables, acc_threshold=0.5):
    """
    One row per variable: the first lead month (1 = first prediction) at which
    the ACC drops below acc_threshold (emulator, persistence) and the emulator's
    RMSE exceeds persistence's / climatology's (None = never), plus the
    emulator / truth amplitude ratio over the last 12 months.
    """
    def first_month(condition):
        hits = np.flatnonzero(np.asarray(condition))
        return int(hits[0]) + 1 if hits.size else None

    rows = []
    for variable in variables:
        d = skill_diagnostics(skill, variable)
        rows.append({
            "variable": variable,
            "acc_below_emulator": first_month(d.acc_emulator < acc_threshold),
            "acc_below_persistence": first_month(d.acc_persistence < acc_threshold),
            "worse_than_persistence": first_month(d.rmse_emulator > d.rmse_persistence),
            "worse_than_climatology": first_month(d.rmse_emulator > d.rmse_climatology),
            "amplitude_ratio_last_year": float(d.amplitude_emulator[-12:].mean() / d.amplitude_truth[-12:].mean()),
        })
    return pd.DataFrame(rows).set_index("variable")


def zonal_spectrum(anomaly):
    """
    Time-mean zonal power spectral density of an anomaly (time, latitude,
    longitude) against physical zonal wavenumber, on the same bands as
    spectral_loss (Emulator.zonal_wavenumber_bands): returns (k in cycles/km,
    PSD in units^2 km). PSD is the area-weighted mean zonal variance per unit k,
    so it reads the same at 1 deg and 1/10 deg. Land is zero anomaly.
    """
    latitude = anomaly["latitude"].values
    rows, band, power_weight, k = zonal_wavenumber_bands(latitude, anomaly.sizes["longitude"])
    x = anomaly.fillna(0.0).values[:, rows, :]
    power = (np.abs(np.fft.rfft(x, axis=-1)) ** 2).mean(0)[:, 1:] * power_weight
    energy = np.bincount(band, power.ravel(), minlength=k.size) / np.cos(np.deg2rad(latitude[rows])).sum()
    half = 0.5 / SPECTRAL_BANDS_PER_DECADE
    return k, energy / (k * (10.0**half - 10.0**-half))


# =============================================================================
# Implied meridional transports (MHT, MFWT)
# =============================================================================

def implied_meridional_transport(content, flux, area, month_seconds, flux_sign=1.0):
    """
    Northward transport implied by a column budget, across the northern edge
    of each grid row, one value per month interval:

        d(content)/dt = surface flux - div(transport)
        T(j) = sum over rows <= j, all longitudes, of area * (flux - d(content)/dt)

    content (time, latitude, longitude) in X m^-2, flux in X m^-2 s^-1 (e.g. J
    and W, or kg and kg s^-1); positive flux = into the ocean after flux_sign.
    month_seconds: the length of each month (time,). Between the centres of
    months a and b the content changes by C_b - C_a and the flux integral is
    F_a L_a / 2 + F_b L_b / 2 (as in the budget constraint); both are divided by
    the interval (L_a + L_b) / 2 to give rates, labelled with the later month.
    Ocean = where content is defined.

    Returns a Dataset (time, latitude) in X s^-1: "transport", and its two
    parts "surface" (the cumulative surface flux) and "storage" (the
    cumulative content tendency), with transport = surface - storage.
    """
    seconds = xr.DataArray(np.asarray(month_seconds, dtype=float), dims="time", coords={"time": content.time})
    ocean_area = area.where(content.isel(time=0).notnull(), 0.0)
    flux_seconds = flux.fillna(0.0) * seconds
    later = slice(1, None)
    interval = (0.5 * (seconds + seconds.shift(time=1))).isel(time=later)
    flux_integral = (0.5 * (flux_seconds + flux_seconds.shift(time=1)) * flux_sign).isel(time=later)
    change = content.diff("time", label="upper").fillna(0.0)
    row = lambda x: (x * ocean_area).sum("longitude")
    surface = row(flux_integral / interval).cumsum("latitude")
    storage = row(change / interval).cumsum("latitude")
    return xr.Dataset({"transport": surface - storage, "surface": surface, "storage": storage})


def meridional_transports(cfg, data, skill):
    """
    Implied meridional transports through the skill test, for every budget in
    cfg.loss.closures whose content is predicted and whose flux is forcing
    (heat -> MHT, freshwater -> MFWT).

    The surface flux is the prescribed forcing, identical for the truth and the
    emulator, so the two differ only through the storage term (the content
    tendency). The truth is ACCESS-OM2's own implied transport: the data hold
    no diagnosed transports to compare against.

    Returns {budget: Dataset} with <truth|pred>_<transport|storage> and
    "surface" on (time, latitude), and attrs content_units.
    """
    budgets = {b: c for b, c in cfg.loss.closures.items()
               if c.content_variable in cfg.data.prognostic and c.flux_variable in cfg.data.forcing}
    months = [pd.Timestamp(t).strftime("%Y-%m") for t in skill.time.values]
    positions = [data.index_of(m) for m in months]
    result = {}
    with _open_time_axis(cfg) as ds:
        for budget, closure in budgets.items():
            flux = _rename_xy(ds[closure.flux_variable]).isel(time=positions)
            flux = flux.transpose("time", "latitude", "longitude").assign_coords(
                time=skill.time.values, latitude=skill.latitude.values, longitude=skill.longitude.values).load()
            out = {}
            for kind in ("truth", "pred"):
                t = implied_meridional_transport(skill[f"{closure.content_variable}_{kind}"], flux, skill["area"],
                                                 data.fields["month_seconds"][positions].numpy(),
                                                 closure.surface_flux_sign)
                out[f"{kind}_transport"], out[f"{kind}_storage"], out["surface"] = t.transport, t.storage, t.surface
            result[budget] = xr.Dataset(out, attrs={
                "content_variable": closure.content_variable,
                "content_units": skill[f"{closure.content_variable}_truth"].attrs.get("units", "")})
    return result
