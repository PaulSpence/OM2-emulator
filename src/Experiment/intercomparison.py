"""
Scores and figures that compare emulators (deterministic, CRPS ensemble,
diffusion) on the same skill test.

Every function takes ``skills`` = {name: Dataset from evaluation.run_skill_test}.
All scores are on anomalies (the field minus its monthly climatology),
area-weighted over the ocean:

  rmse_mean         RMSE of the ensemble mean (the forecast skill)
  rmse_climatology  RMSE of predicting zero anomaly (= the truth's RMS anomaly)
  acc_mean          anomaly correlation of the ensemble mean
  amplitude_member  RMS anomaly of one member (realism: compare amplitude_truth)
  spread            RMS ensemble standard deviation (0 for one member)
  spread_skill      sqrt((M + 1) / M) * spread / rmse_mean: 1 = reliable spread
  crps              fair CRPS of the ensemble (the MAE for one member)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from .evaluation import global_integral, zonal_spectrum

# Fixed colour per emulator, in this order (never cycled); the truth is black.
MODEL_COLOURS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22")
TRUTH_STYLE = dict(color="black", lw=1.6)


def _colour(skills, name):
    return MODEL_COLOURS[list(skills).index(name) % len(MODEL_COLOURS)]


def _ocean_mean(field, area):
    return (field.fillna(0.0) * area).sum(("latitude", "longitude")) / area.sum()


def ensemble_scores(skill, variable):
    """The scores in the module docstring for one variable, one value per month."""
    truth = skill[f"{variable}_truth_anom"]
    mean = skill[f"{variable}_pred_mean_anom"]
    member = skill[f"{variable}_pred_anom"]
    area = skill["area"].where(truth.isel(time=0).notnull(), 0.0)
    avg = lambda x: _ocean_mean(x, area)

    out = {
        "rmse_mean": np.sqrt(avg((mean - truth) ** 2)),
        "rmse_climatology": np.sqrt(avg(truth**2)),
        "acc_mean": avg(mean * truth) / np.sqrt(avg(mean**2) * avg(truth**2)),
        "amplitude_member": np.sqrt(avg(member**2)),
        "amplitude_truth": np.sqrt(avg(truth**2)),
    }
    ens_name = f"{variable}_ens_anom"
    if ens_name in skill:
        ens = skill[ens_name]
        m = ens.sizes["member"]
        out["spread"] = np.sqrt(avg(ens.var("member", ddof=1)))
        # Fair CRPS: mean |x_i - y| - sum_{i,j} |x_i - x_j| / (2 M (M - 1)), with the
        # pair sum from the sorted members: sum_{i,j} |x_i - x_j| = 2 sum_i (2i - M - 1) x_(i).
        # Computed 12 months at a time to bound memory (members x months x grid).
        ranks = (2 * np.arange(1, m + 1) - m - 1).reshape(-1, 1, 1, 1)
        crps = np.empty(truth.shape, dtype=np.float32)
        for start in range(0, truth.sizes["time"], 12):
            chunk = slice(start, start + 12)
            members = ens.values[:, chunk]
            pairs = 2 * (ranks * np.sort(members, axis=0)).sum(axis=0)
            crps[chunk] = np.abs(members - truth.values[None, chunk]).mean(axis=0) - pairs / (2 * m * (m - 1))
        out["crps"] = avg(truth.copy(data=crps))
        out["spread_skill"] = np.sqrt((m + 1) / m) * out["spread"] / out["rmse_mean"]
    else:
        out["spread"] = xr.zeros_like(out["rmse_mean"])
        out["crps"] = avg(np.abs(member - truth))
        out["spread_skill"] = xr.zeros_like(out["rmse_mean"])
    return xr.Dataset(out)


def global_anomaly(skill, variable, kind="pred"):
    """Ocean-integrated anomaly through the skill test (kind: pred, truth, pred_mean)."""
    return global_integral(skill[f"{variable}_{kind}_anom"], skill["area"])


def _trend_per_decade(series):
    years = (series.time - series.time[0]).dt.days.values / 365.25
    return float(np.polyfit(years, series.values, 1)[0] * 10)


def intercomparison_summary(skills, variables, leads=(12, 36)):
    """
    One row per (emulator, variable): ensemble-mean RMSE / climatology RMSE at
    the given lead months and over the last year, anomaly correlation of the
    ensemble mean at those leads, the member/truth amplitude ratio over the
    last year, mean spread-skill ratio and CRPS, the decadal pattern error
    (RMS error of the ensemble mean's last-10-year mean anomaly against the
    truth's, relative to the truth's: 0 = perfect, 1 = no better than zero
    anomaly), and the global-integral anomaly: RMS error against the truth and
    trend error (per decade, as a fraction of the truth's trend). Lower is
    better except acc (higher), amplitude_ratio and spread_skill (1 is best).
    """
    rows = []
    for name, skill in skills.items():
        for variable in variables:
            s = ensemble_scores(skill, variable)
            row = {"emulator": name, "variable": variable, "members": skill.attrs.get("members", 1)}
            for lead in leads:
                if lead <= s.sizes["time"]:
                    row[f"rmse/clim_{lead}mo"] = float(s.rmse_mean[lead - 1] / s.rmse_climatology[lead - 1])
                    row[f"acc_{lead}mo"] = float(s.acc_mean[lead - 1])
            last = slice(-12, None)
            row["rmse/clim_last_yr"] = float(s.rmse_mean[last].mean() / s.rmse_climatology[last].mean())
            row["amplitude_ratio_last_yr"] = float(s.amplitude_member[last].mean() / s.amplitude_truth[last].mean())
            row["spread_skill"] = float(s.spread_skill.mean())
            # The forced response: the last decade's mean anomaly pattern of the ensemble
            # mean against the truth's, as an RMS error relative to the truth's pattern.
            truth_decade = skill[f"{variable}_truth_anom"].isel(time=slice(-120, None)).mean("time")
            pred_decade = skill[f"{variable}_pred_mean_anom"].isel(time=slice(-120, None)).mean("time")
            area = skill["area"].where(truth_decade.notnull(), 0.0)
            ocean_rms = lambda x: float(np.sqrt((x.fillna(0.0) ** 2 * area).sum() / area.sum()))
            row["decadal_pattern_error"] = ocean_rms(pred_decade - truth_decade) / ocean_rms(truth_decade)
            row["crps"] = float(s.crps.mean())
            pred, truth = global_anomaly(skill, variable), global_anomaly(skill, variable, "truth")
            row["global_rms_error"] = float(np.sqrt(((pred - truth) ** 2).mean()))
            truth_trend = _trend_per_decade(truth)
            row["global_trend_error"] = (_trend_per_decade(pred) - truth_trend) / abs(truth_trend) if truth_trend else np.nan
            rows.append(row)
    return pd.DataFrame(rows).set_index(["variable", "emulator"]).sort_index()


def plot_intercomparison(skills, variables):
    """
    One row per variable; columns: ensemble-mean RMSE (dotted: climatology),
    anomaly correlation of the ensemble mean, member amplitude (black: truth),
    global-integrated anomaly (one member; shading: ensemble range), and the
    spread-skill ratio (1 = reliable).
    """
    titles = ("RMSE of ensemble mean", "Anomaly correlation (ens. mean)", "Anomaly amplitude (one member)",
              "Global-integrated anomaly", "Spread / skill")
    fig, axes = plt.subplots(len(variables), len(titles), figsize=(26, 3.8 * len(variables)), sharex=True,
                             squeeze=False, constrained_layout=True)
    for row, variable in zip(axes, variables):
        units = next(iter(skills.values()))[f"{variable}_truth_anom"].attrs.get("units", "")
        first = True
        for name, skill in skills.items():
            s, colour, time = ensemble_scores(skill, variable), _colour(skills, name), skill.time.values
            row[0].plot(time, s.rmse_mean, color=colour, lw=1.8, label=name)
            row[1].plot(time, s.acc_mean, color=colour, lw=1.8, label=name)
            row[2].plot(time, s.amplitude_member, color=colour, lw=1.8, label=name)
            row[3].plot(time, global_anomaly(skill, variable), color=colour, lw=1.5, label=name)
            if f"{variable}_ens_anom" in skill:
                ens = global_integral(skill[f"{variable}_ens_anom"], skill["area"])
                row[3].fill_between(time, ens.min("member"), ens.max("member"), color=colour, alpha=0.15, lw=0)
                row[4].plot(time, s.spread_skill, color=colour, lw=1.8, label=name)
            if first:
                row[0].plot(time, s.rmse_climatology, color="0.4", ls=":", lw=1.5, label="climatology")
                row[2].plot(time, s.amplitude_truth, label="ACCESS-OM2", **TRUTH_STYLE)
                row[3].plot(time, global_anomaly(skill, variable, "truth"), label="ACCESS-OM2", **TRUTH_STYLE)
                first = False
        row[1].axhline(0.5, color="0.7", lw=0.8)
        row[4].axhline(1.0, color="0.7", lw=0.8)
        ylabels = (f"RMSE ({units})", "ACC", f"RMS anomaly ({units})", f"{units} m$^2$", "ratio")
        for ax, title, ylabel in zip(row, titles, ylabels):
            ax.set_title(f"{variable}: {title}", fontsize=10)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            if ax.lines:
                ax.legend(frameon=False, fontsize=8)
    return fig


def plot_intercomparison_spectra(skills, variables):
    """Zonal spectra (evaluation.zonal_spectrum) of one member per emulator and the truth, first and last year."""
    periods = {"first year": slice(0, 12), "last year": slice(-12, None)}
    fig, axes = plt.subplots(len(variables), len(periods), figsize=(12, 3.8 * len(variables)), squeeze=False,
                             constrained_layout=True)
    for row, variable in zip(axes, variables):
        reference = next(iter(skills.values()))
        units = reference[f"{variable}_truth_anom"].attrs.get("units", "")
        for ax, (label, period) in zip(row, periods.items()):
            k, psd = zonal_spectrum(reference[f"{variable}_truth_anom"].isel(time=period))
            ax.loglog(k, psd, label="ACCESS-OM2", **TRUTH_STYLE)
            for name, skill in skills.items():
                k, psd = zonal_spectrum(skill[f"{variable}_pred_anom"].isel(time=period))
                ax.loglog(k, psd, color=_colour(skills, name), lw=1.6, ls="--", label=name)
            ax.set_title(f"{variable}: {label}", fontsize=10)
            ax.set_xlabel("zonal wavenumber (cycles/km)")
            ax.set_ylabel(f"PSD ({units}$^2$ km)")
            ax.grid(alpha=0.3, which="both")
            ax.legend(frameon=False, fontsize=8)
    return fig


def plot_intercomparison_transports(transports):
    """
    transports = {emulator: evaluation.meridional_transports(...)}. One row per
    budget (MHT, MFWT): time-mean storage term (black: ACCESS-OM2) and the RMS
    over time of each emulator's monthly transport minus the truth's (black:
    the truth's standard deviation over time).
    """
    from .plots import TRIPOLE_LATITUDE, _TRANSPORT_UNITS

    budgets = list(next(iter(transports.values())))
    labels = {"heat": "MHT", "freshwater": "MFWT"}
    fig, axes = plt.subplots(len(budgets), 2, figsize=(13, 3.8 * len(budgets)), squeeze=False, constrained_layout=True)
    for row, budget in zip(axes, budgets):
        reference = next(iter(transports.values()))[budget]
        scale, unit = _TRANSPORT_UNITS.get(reference.attrs.get("content_units", ""), (1.0, "content units s$^{-1}$"))
        lat = reference.latitude.values
        truth = reference.truth_transport / scale
        row[0].plot(lat, reference.truth_storage.mean("time") / scale, label="ACCESS-OM2 (implied)", **TRUTH_STYLE)
        row[1].plot(lat, truth.std("time"), label="ACCESS-OM2 std over time", **TRUTH_STYLE)
        for name, per_budget in transports.items():
            t, colour = per_budget[budget], MODEL_COLOURS[list(transports).index(name) % len(MODEL_COLOURS)]
            row[0].plot(lat, t.pred_storage.mean("time") / scale, color=colour, lw=1.6, ls="--", label=name)
            error = np.sqrt(((t.pred_transport / scale - truth) ** 2).mean("time"))
            row[1].plot(lat, error, color=colour, lw=1.6, ls="--", label=f"{name}: RMS error")
        for ax, title in zip(row, ("storage term, time mean", "month-to-month")):
            ax.axhline(0.0, color="0.5", lw=0.8)
            ax.axvspan(TRIPOLE_LATITUDE, lat.max(), color="0.9", zorder=0)
            ax.set_title(f"{labels.get(budget, budget)}: {title}")
            ax.set_xlabel("latitude (row north edge, deg)")
            ax.set_ylabel(f"northward ({unit})")
            ax.grid(alpha=0.3)
            ax.legend(frameon=False, fontsize=8)
    return fig


# =============================================================================
# Basin heat and freshwater content
# =============================================================================

# Basins as groups of COSIMA basin-mask ids (Collier, CSIRO 2021: 1 Southern
# Ocean, 2 Atlantic, 3 Pacific, 4 Arctic, 5 Indian, 6 Mediterranean, 7 Black
# Sea, 8 Hudson Bay, 9 Baltic, 10 Red Sea); marginal seas join their ocean.
BASIN_GROUPS = {
    "Southern Ocean": (1,),
    "Atlantic": (2, 6, 7, 8, 9),
    "Indian": (5, 10),
    "Pacific": (3,),
    "Arctic": (4,),
}
OM2_1DEG_BASIN_MASK = "/g/data/ik11/grids/om2_1deg_basinmask_yt_xt.nc"


def load_basins(path, latitude, longitude, groups=BASIN_GROUPS):
    """
    (basin, latitude, longitude) boolean masks from a basin-id file on the
    emulator's grid (e.g. OM2_1DEG_BASIN_MASK), one per entry of groups.
    """
    with xr.open_dataset(path) as ds:
        ids = ds["basin"].rename({d: n for d, n in zip(ds["basin"].dims, ("latitude", "longitude"))}).load()
    if ids.shape != (len(latitude), len(longitude)):
        raise ValueError(f"{path} is {ids.shape}, the grid is {(len(latitude), len(longitude))}")
    ids = ids.assign_coords(latitude=np.asarray(latitude), longitude=np.asarray(longitude))
    masks = [ids.isin(list(members)) for members in groups.values()]
    return xr.concat(masks, dim=pd.Index(list(groups), name="basin"))


def basin_anomaly(skill, variable, basins, kind="pred"):
    """Ocean-integrated anomaly per basin, (basin, time) (kind: pred, truth, pred_mean; or ens -> member too)."""
    field = skill[f"{variable}_{kind}_anom"].fillna(0.0)
    return (field * skill["area"] * basins).sum(("latitude", "longitude"))


def basin_scores(skills, variables, basins):
    """
    One row per (variable, basin, emulator): RMS error of one member's basin
    anomaly against the truth's, the correlation of the two time series, the
    truth's linear trend and the emulator's trend error, both per decade in the
    integrated units (e.g. J). The error is absolute, not a fraction of the
    truth's trend, which is near zero in some basins.
    """
    rows = []
    for variable in variables:
        for name, skill in skills.items():
            pred = basin_anomaly(skill, variable, basins)
            truth = basin_anomaly(skill, variable, basins, "truth")
            for basin in basins.basin.values:
                p, t = pred.sel(basin=basin), truth.sel(basin=basin)
                truth_trend = _trend_per_decade(t)
                rows.append({
                    "variable": variable, "basin": basin, "emulator": name,
                    "rms_error": float(np.sqrt(((p - t) ** 2).mean())),
                    "correlation": float(np.corrcoef(p.values, t.values)[0, 1]),
                    "truth_trend_per_decade": truth_trend,
                    "trend_error_per_decade": _trend_per_decade(p) - truth_trend,
                })
    return pd.DataFrame(rows).set_index(["variable", "basin", "emulator"])


def plot_basin_content(skills, variable, basins):
    """
    Integrated anomaly of one variable per basin through the skill test: the
    truth (black) and one member per emulator, with the ensemble range shaded
    where there is one. One panel per basin, each on its own scale.
    """
    reference = next(iter(skills.values()))
    units = reference[f"{variable}_truth_anom"].attrs.get("units", "")
    names = list(basins.basin.values)
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 3.8), sharex=True, squeeze=False,
                             constrained_layout=True)
    truth = basin_anomaly(reference, variable, basins, "truth")
    members = {name: basin_anomaly(skill, variable, basins) for name, skill in skills.items()}
    ranges = {name: basin_anomaly(skill, variable, basins, "ens") for name, skill in skills.items()
              if f"{variable}_ens_anom" in skill}
    for ax, basin in zip(axes[0], names):
        time = reference.time.values
        ax.plot(time, truth.sel(basin=basin), label="ACCESS-OM2", **TRUTH_STYLE)
        for name, series in members.items():
            colour = _colour(skills, name)
            ax.plot(time, series.sel(basin=basin), color=colour, lw=1.4, label=name)
            if name in ranges:
                ens = ranges[name].sel(basin=basin)
                ax.fill_between(time, ens.min("member"), ens.max("member"), color=colour, alpha=0.15, lw=0)
        ax.axhline(0.0, color="0.7", lw=0.8)
        ax.set_title(f"{basin}", fontsize=11)
        ax.set_ylabel(f"{variable} anomaly ({units} m$^2$)")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Basin-integrated {variable} anomaly through the skill test", fontsize=13)
    return fig
