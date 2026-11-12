"""
Long, repeat-year control runs: stability over centuries.

run_repeat_year_control drives an emulator with one forcing year repeated
(default ACCESS-OM2's RYF year, May 1990 - April 1991) for many years. With
balance=True the year's net global heat and freshwater fluxes are removed
(a uniform offset over the ocean, length-weighted so the year integrates to
zero), in both the network's input and the budget constraint. A constrained
emulator's global content then returns to the same value every year, so
anything that drifts or grows is the emulator's own doing.

12000 months of full fields per member would not fit in memory, so the run
keeps diagnostics only, computed on the device as it goes:
  global      ocean-integrated anomaly per month (content units x m^2)
  rms         area-weighted RMS anomaly per month
  max_abs_z   largest |z-score| over the ocean per month
plus, for member 0, the annual-mean anomaly in selected years (for spectra)
and every decade's mean anomaly (for slow pattern drift).
"""

import numpy as np
import pandas as pd
import torch
import xarray as xr

from .constraints import build_constraint


def balanced_year_forcing(cfg, data, month_indices, balance=True):
    """
    The forcing z-scores (12, C, H, W) of the given months, and the offset
    removed from each budget flux channel {channel: W m^-2 (or kg m^-2 s^-1)}.

    The offset is the length-weighted mean of the ocean-integrated flux over
    the months divided by the ocean area, so sum_i (G_i - offset * A) L_i = 0.
    It is subtracted from the physical flux at every ocean cell; in z-scores
    that is offset / std. Budget flux channels are those of every closure in
    cfg.loss.closures whose flux variable is forcing, enforced or not, so an
    unconstrained emulator gets the same balanced input.
    """
    f = data.fields
    idx = torch.as_tensor(month_indices)
    forcing = f["forcing"][idx].clone().float()
    offsets = {}
    if not balance:
        return forcing, offsets
    ocean = f["mask"] > 0
    ocean_area = float(f["area"].sum())
    seconds = f["month_seconds"][idx]
    for closure in cfg.loss.closures.values():
        if closure.flux_variable not in cfg.data.forcing:
            continue
        c = cfg.data.forcing.index(closure.flux_variable)
        g = f["forcing_global"][idx, c]
        offset = float((g * seconds).sum() / (seconds.sum() * ocean_area))
        std = f["forcing_std"][idx, c]
        shift = torch.where(ocean & (std > 0), offset / std.clamp_min(1e-30), torch.zeros_like(std))
        forcing[:, c] -= shift.float()
        offsets[c] = offset
    return forcing, offsets


def _balanced_constraint(cfg, data, month_indices, offsets):
    """build_constraint with each flux channel's global integral reduced by offset x ocean area."""
    constraint = build_constraint(cfg, data)
    if constraint is None or not offsets:
        return constraint
    flux_global = constraint.flux_global.clone()
    ocean_area = float(data.fields["area"].sum())
    idx = torch.as_tensor(month_indices)
    for c, offset in offsets.items():
        flux_global[idx, c] -= offset * ocean_area
    constraint.flux_global = flux_global
    return constraint


@torch.no_grad()
def run_repeat_year_control(cfg, model, data, first_month="1990-05", years=1000, members=1, balance=True,
                            snapshot_years=(1, 10, 100, 1000), seed=0):
    """
    Seed from the truth at the month before first_month, then run `years`
    repeats of the 12 months from first_month, `members` members at once.
    Returns an xarray Dataset of diagnostics (see the module docstring), with
    attrs giving the forcing year, the balance offsets and the member count.
    """
    torch.manual_seed(seed)
    f, n_prior = data.fields, cfg.window.n_prior
    first = data.index_of(first_month)
    idx = list(range(first, first + 12))
    if first - n_prior < 0 or idx[-1] > len(data.months) - 1:
        raise ValueError(f"{first_month}: need {n_prior} months before it and 12 from it on the time axis")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    forcing, offsets = balanced_year_forcing(cfg, data, idx, balance)
    forcing = forcing.to(device)
    constraint = _balanced_constraint(cfg, data, idx, offsets)
    if constraint is not None:
        constraint = constraint.to(device)
    mask = f["mask"].to(device)
    ocean = mask > 0
    area = f["area"].to(device)
    ocean_area = area.sum()
    std = f["prognostic_std"].to(device)
    p_count = f["prognostic"].shape[1]

    state = f["prognostic"][first - n_prior : first].flatten(0, 1)[None].to(device).repeat(members, 1, 1, 1)
    previous = torch.full((members,), first - 1, dtype=torch.long, device=device)
    n_steps = years * 12
    glob = torch.empty(n_steps, members, p_count, dtype=torch.float64)
    rms = torch.empty(n_steps, members, p_count, dtype=torch.float64)
    max_abs = torch.empty(n_steps, members, p_count)
    snapshot_years = sorted(y for y in snapshot_years if 1 <= y <= years)
    annual = {}
    decades = torch.zeros(years // 10, p_count, *mask.shape)
    year_sum = torch.zeros(p_count, *mask.shape, device=device)

    for step in range(n_steps):
        k = step % 12
        target = torch.full_like(previous, idx[k])
        pred = model(state, forcing[k : k + 1].expand(members, -1, -1, -1), mask).float()
        if constraint is not None:
            pred = constraint(pred, state[:, -p_count:], previous, target)
        anomaly = pred * std[idx[k]]                                  # (M, P, H, W) physical
        weighted = anomaly.double() * area
        glob[step] = weighted.sum(dim=(-2, -1)).cpu()
        rms[step] = ((anomaly.double() ** 2 * area).sum(dim=(-2, -1)) / ocean_area).sqrt().cpu()
        max_abs[step] = pred.abs().masked_fill(~ocean, 0).amax(dim=(-2, -1)).cpu()
        year_sum += anomaly[0]
        if k == 11:
            year = step // 12 + 1
            if year in snapshot_years:
                annual[year] = (year_sum / 12).cpu()
            decade = (year - 1) // 10
            if decade < decades.shape[0]:
                decades[decade] += (year_sum / 120).cpu()
            year_sum.zero_()
        state = torch.cat([state[:, p_count:], pred.to(state.dtype)], dim=1)
        previous = target
    model.train(was_training)

    names = list(f["prognostic_names"])
    land = ~ocean.cpu().numpy()
    def field(t):
        a = t.numpy().astype(np.float32)
        a[..., land] = np.nan
        return a
    ds = xr.Dataset(
        {
            "global": (("step", "member", "variable"), glob.numpy()),
            "rms": (("step", "member", "variable"), rms.numpy()),
            "max_abs_z": (("step", "member", "variable"), max_abs.numpy()),
            "decadal_mean": (("decade", "variable", "latitude", "longitude"), field(decades)),
        },
        coords={"step": np.arange(1, n_steps + 1), "member": np.arange(members), "variable": names,
                "decade": np.arange(1, decades.shape[0] + 1),
                "latitude": f["latitude"], "longitude": f["longitude"]},
    )
    if annual:
        ds["annual_mean"] = (("snapshot_year", "variable", "latitude", "longitude"),
                             field(torch.stack([annual[y] for y in sorted(annual)])))
        ds = ds.assign_coords(snapshot_year=sorted(annual))
    ds["area"] = (("latitude", "longitude"), f["area"].numpy())
    ds.attrs.update(forcing_months=f"{data.months[idx[0]]}..{data.months[idx[-1]]}", years=years, members=members,
                    balanced=int(balance), constrained=int(constraint is not None),
                    flux_offsets=str({cfg.data.forcing[c]: o for c, o in offsets.items()}))
    return ds


def reference_rms(data, variable, months=None):
    """The truth's area-weighted RMS anomaly of a variable, averaged over months (default: all)."""
    f = data.fields
    p = list(f["prognostic_names"]).index(variable)
    idx = range(len(data.months)) if months is None else months
    area = f["area"].double()
    values = [float((((f["prognostic"][t, p] * f["prognostic_std"][t, p]).double() ** 2 * area).sum() / area.sum()).sqrt())
              for t in idx]
    return float(np.mean(values))


def plot_long_controls(controls, reference=None):
    """
    controls = {emulator: run_repeat_year_control Dataset}. One row per
    variable; columns: annual-mean global anomaly, annual-mean RMS anomaly
    (member mean, shading: member range; dashed: the truth's typical RMS from
    `reference` = {variable: value}), annual maximum |z|, and each decade's
    mean pattern against the first decade's (RMS difference / RMS of the
    first decade).
    """
    import matplotlib.pyplot as plt
    from .intercomparison import MODEL_COLOURS

    variables = list(next(iter(controls.values())).variable.values)
    fig, axes = plt.subplots(len(variables), 4, figsize=(22, 3.8 * len(variables)), squeeze=False,
                             constrained_layout=True)
    titles = ("Global anomaly (annual mean)", "RMS anomaly (annual mean)", "Max |z| (annual max)",
              "Decadal pattern drift vs decade 1")
    for row, variable in zip(axes, variables):
        for i, (name, ds) in enumerate(controls.items()):
            colour = MODEL_COLOURS[i % len(MODEL_COLOURS)]
            d = ds.sel(variable=variable)
            years = d.step.size // 12
            by_year = lambda x: x.isel(step=slice(0, years * 12)).coarsen(step=12).mean()
            year_axis = np.arange(1, years + 1)
            row[0].plot(year_axis, by_year(d["global"]).mean("member"), color=colour, lw=1.4, label=name)
            r = by_year(d["rms"])
            row[1].plot(year_axis, r.mean("member"), color=colour, lw=1.4, label=name)
            if d.member.size > 1:
                row[1].fill_between(year_axis, r.min("member"), r.max("member"), color=colour, alpha=0.15, lw=0)
            row[2].plot(year_axis, d["max_abs_z"].isel(step=slice(0, years * 12)).coarsen(step=12).max()
                        .max("member"), color=colour, lw=1.2, label=name)
            dec = d["decadal_mean"]
            first = dec.isel(decade=0)
            w = ds["area"].where(first.notnull(), 0.0)
            ocean_rms = lambda x: float(np.sqrt((x.fillna(0.0) ** 2 * w).sum() / w.sum()))
            drift = [ocean_rms(dec.isel(decade=j) - first) / ocean_rms(first) for j in range(dec.decade.size)]
            row[3].plot(dec.decade.values * 10, drift, color=colour, lw=1.4, label=name)
        if reference and variable in reference:
            row[1].axhline(reference[variable], color="black", ls="--", lw=1, label="ACCESS-OM2 typical")
        for ax, title in zip(row, titles):
            ax.set_title(f"{variable}: {title}", fontsize=10)
            ax.set_xlabel("year")
            ax.grid(alpha=0.3)
            ax.legend(frameon=False, fontsize=8)
        row[2].set_yscale("log")
    return fig


def plot_long_control_spectra(controls):
    """Zonal spectra of member 0's annual-mean anomaly in each snapshot year, one panel per (variable, emulator)."""
    import matplotlib.pyplot as plt
    from .evaluation import zonal_spectrum

    variables = list(next(iter(controls.values())).variable.values)
    fig, axes = plt.subplots(len(variables), len(controls), figsize=(4.6 * len(controls), 3.8 * len(variables)),
                             squeeze=False, constrained_layout=True, sharey="row")
    for row, variable in zip(axes, variables):
        for ax, (name, ds) in zip(row, controls.items()):
            if "annual_mean" not in ds:
                continue
            years = ds.snapshot_year.values
            colours = plt.cm.viridis(np.linspace(0, 0.9, len(years)))
            for year, colour in zip(years, colours):
                field = ds["annual_mean"].sel(snapshot_year=year, variable=variable).expand_dims(time=[0])
                k, psd = zonal_spectrum(field.values, field["latitude"].values)
                ax.loglog(k, psd, color=colour, lw=1.4, label=f"year {year}")
            ax.set_title(f"{name}: {variable}", fontsize=10)
            ax.set_xlabel("zonal wavenumber (cycles/km)")
            ax.grid(alpha=0.3, which="both")
            ax.legend(frameon=False, fontsize=8)
    return fig
