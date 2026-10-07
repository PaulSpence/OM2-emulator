"""
Standard figures for the skill test, the control run and training history.

All functions take the Datasets returned by evaluation.run_skill_test /
run_control and return the matplotlib Figure. The plot_*_all_variables /
plot_skill_evaluation functions draw every prognostic variable in the Dataset.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluation import global_integral, global_rmse


def _default_variable(ds):
    return next(name[: -len("_pred")] for name in ds.data_vars if name.endswith("_pred"))


def _select_period(da, period):
    """A month ("YYYY-MM") or a year mean ("YYYY") of a (time, lat, lon) field."""
    text = str(period)
    if len(text) == 4 and text.isdigit():
        chosen = da.sel(time=text)
        if chosen.sizes["time"] == 0:
            raise KeyError(f"No months of {text} in {da.time.values[0]}..{da.time.values[-1]}")
        return chosen.mean("time"), f"{text} mean"
    month = pd.Period(text, freq="M")
    matches = np.flatnonzero(pd.PeriodIndex(pd.to_datetime(da.time.values), freq="M") == month)
    if len(matches) == 0:
        raise KeyError(f"{text} is not in the rollout ({str(da.time.values[0])[:7]}..{str(da.time.values[-1])[:7]})")
    return da.isel(time=int(matches[0])), text


def prognostic_variables(ds):
    """Every prognostic variable in a skill-test or control Dataset, in order."""
    return [name[: -len("_pred")] for name in ds.data_vars if name.endswith("_pred")]


def _scale_for(scale, variable, field, quantile=0.99):
    """
    Colour-scale limit for one variable: a number (all variables), a dict
    {variable: number}, or None. Variables without a number get the given
    quantile of |field| over the ocean.
    """
    if isinstance(scale, dict):
        scale = scale.get(variable)
    if scale is not None:
        return float(scale)
    values = np.abs(np.asarray(field.values, dtype=np.float64))
    values = values[np.isfinite(values)]
    return float(np.quantile(values, quantile)) if values.size and values.max() > 0 else 1.0


def _power_of_ten(values):
    """(scale, label) so that values / scale is of order 1-10."""
    peak = np.nanmax(np.abs(values))
    exponent = int(np.floor(np.log10(peak))) if np.isfinite(peak) and peak > 0 else 0
    return 10.0**exponent, f"$10^{{{exponent}}}$"


def plot_snapshots(skill, periods, anomaly_scale=None, difference_scale=None, variable=None):
    """
    Predicted anomaly, true anomaly and their difference, one row per period.
    Scales as in plot_skill_evaluation (None = automatic).
    """
    variable = variable or _default_variable(skill)
    pred, truth = skill[f"{variable}_pred_anom"], skill[f"{variable}_truth_anom"]
    units = pred.attrs.get("units", "")
    anomaly_scale = _scale_for(anomaly_scale, variable, truth)
    difference_scale = _scale_for(difference_scale, variable, pred - truth)
    fig, axes = plt.subplots(len(periods), 3, figsize=(15, 4 * len(periods)), squeeze=False, constrained_layout=True)
    for row, period in enumerate(periods):
        p, label = _select_period(pred, period)
        t, _ = _select_period(truth, period)
        anomaly_kw = dict(cmap="RdBu_r", vmin=-anomaly_scale, vmax=anomaly_scale, add_colorbar=False)
        image = p.plot(ax=axes[row, 0], **anomaly_kw)
        t.plot(ax=axes[row, 1], **anomaly_kw)
        diff = (p - t).plot(ax=axes[row, 2], cmap="RdBu_r", vmin=-difference_scale, vmax=difference_scale, add_colorbar=False)
        for ax, title in zip(axes[row], ("Predicted anomaly", "True anomaly", "Prediction minus truth")):
            ax.set_title(f"{title} | {label}")
        fig.colorbar(image, ax=axes[row, :2], shrink=0.9, label=f"{variable} anomaly ({units})")
        fig.colorbar(diff, ax=axes[row, 2], shrink=0.9, label=f"difference ({units})")
    return fig


def plot_skill_evaluation(skill, variable, periods, anomaly_scale=None, difference_scale=None):
    """
    One figure per variable: predicted anomaly, true anomaly and their
    difference (one row per period), with the global-integrated anomaly
    through the skill test as a full-width panel underneath.

    anomaly_scale / difference_scale: a number, {variable: number}, or None
    for an automatic scale (99th percentile of |truth anomaly| / |difference|).
    """
    pred, truth = skill[f"{variable}_pred_anom"], skill[f"{variable}_truth_anom"]
    units = pred.attrs.get("units", "")
    n_rows = len(periods)
    fig = plt.figure(figsize=(16, 4 * n_rows + 4.5), constrained_layout=True)
    grid = fig.add_gridspec(n_rows + 1, 3, height_ratios=[1] * n_rows + [1.1])

    vmax = _scale_for(anomaly_scale, variable, truth)
    dmax = _scale_for(difference_scale, variable, pred - truth)
    anomaly_kw = dict(cmap="RdBu_r", vmin=-vmax, vmax=vmax, add_colorbar=False)
    for row, period in enumerate(periods):
        axes = [fig.add_subplot(grid[row, col]) for col in range(3)]
        p, label = _select_period(pred, period)
        t, _ = _select_period(truth, period)
        image = p.plot(ax=axes[0], **anomaly_kw)
        t.plot(ax=axes[1], **anomaly_kw)
        diff = (p - t).plot(ax=axes[2], cmap="RdBu_r", vmin=-dmax, vmax=dmax, add_colorbar=False)
        for ax, title in zip(axes, ("Predicted anomaly", "True anomaly", "Prediction minus truth")):
            ax.set_title(f"{title} | {label}")
            ax.set_xlabel("")
            ax.set_ylabel("")
        fig.colorbar(image, ax=axes[:2], shrink=0.9, label=f"anomaly ({units})")
        fig.colorbar(diff, ax=axes[2], shrink=0.9, label=f"difference ({units})")

    _plot_global_anomaly(fig.add_subplot(grid[n_rows, :]), skill, variable)
    fig.suptitle(f"{variable}: skill test", fontsize=14)
    return fig


def plot_global_rmse_all_variables(skill, variables=None):
    """Area-weighted RMSE of the full field through the skill test, one panel per variable."""
    variables = variables or prognostic_variables(skill)
    fig, axes = plt.subplots(len(variables), 1, figsize=(11, 3.2 * len(variables)), sharex=True,
                             squeeze=False, constrained_layout=True)
    for ax, variable in zip(axes[:, 0], variables):
        rmse = global_rmse(skill[f"{variable}_pred"], skill[f"{variable}_truth"], skill["area"])
        ax.plot(skill.time.values, rmse.values, color="firebrick", lw=2)
        ax.set_title(f"Global area-weighted RMSE of {variable}")
        ax.set_ylabel(f"RMSE ({skill[f'{variable}_pred'].attrs.get('units', '')})")
        ax.grid(alpha=0.3)
    return fig


def plot_rmse_by_epoch(history, variables=None):
    """
    Train and validation RMSE (normalised units) per epoch, one panel per
    variable. history is EmulatorModule.rmse_history, or a run_dir whose
    metrics.csv has the {stage}_rmse_{variable} columns.
    """
    if not isinstance(history, pd.DataFrame):
        metrics = pd.read_csv(Path(history) / "metrics.csv")
        rows = []
        for column in metrics.columns:
            for stage in ("train", "val"):
                if column.startswith(f"{stage}_rmse_"):
                    series = metrics.dropna(subset=[column]).groupby("epoch")[column].mean()
                    rows += [{"epoch": e, "stage": stage, "variable": column[len(f"{stage}_rmse_"):], "rmse": v}
                             for e, v in series.items()]
        history = pd.DataFrame(rows, columns=["epoch", "stage", "variable", "rmse"])
    if history.empty:
        raise ValueError("No per-variable RMSE history: train the model first")
    variables = variables or list(dict.fromkeys(history["variable"]))
    fig, axes = plt.subplots(len(variables), 1, figsize=(11, 3.2 * len(variables)), sharex=True,
                             squeeze=False, constrained_layout=True)
    for ax, variable in zip(axes[:, 0], variables):
        for stage, label in (("train", "Training"), ("val", "Validation")):
            rows = history[(history["variable"] == variable) & (history["stage"] == stage)]
            if not rows.empty:
                ax.plot(rows["epoch"], rows["rmse"], label=label, lw=2)
        ax.set_yscale("log")
        ax.set_title(f"{variable}: area-weighted RMSE per epoch")
        ax.set_ylabel("RMSE (normalised)")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
    axes[-1, 0].set_xlabel("Epoch")
    return fig


def _plot_global_anomaly(ax, ds, variable, suffixes=(("pred_anom", "Predicted"), ("truth_anom", "Truth")),
                         x="time"):
    """Area-integrated anomaly series on ax, scaled by a power of ten (units x m^2)."""
    units = ds[f"{variable}_pred_anom"].attrs.get("units", "")
    series = {label: global_integral(ds[f"{variable}_{suffix}"], ds["area"]).values for suffix, label in suffixes}
    scale, scale_label = _power_of_ten(np.concatenate(list(series.values())))
    for label, values in series.items():
        ax.plot(ds[x].values, values / scale, label=label, lw=2)
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_ylabel(f"Integrated anomaly ({scale_label} {units} m$^2$)")
    ax.grid(alpha=0.3)
    if len(series) > 1:
        ax.legend(frameon=False)


def plot_global_timeseries(skill, variable=None):
    """Global-integrated anomaly, predicted vs true, through the skill test."""
    variable = variable or _default_variable(skill)
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    _plot_global_anomaly(ax, skill, variable)
    ax.set_title(f"Global-integrated {variable} anomaly through the skill test")
    return fig


def plot_global_rmse(skill, variable=None):
    """Area-weighted RMSE of the full field through the skill test."""
    variable = variable or _default_variable(skill)
    rmse = global_rmse(skill[f"{variable}_pred"], skill[f"{variable}_truth"], skill["area"])
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    ax.plot(skill.time.values, rmse.values, color="firebrick", lw=2)
    ax.set_title(f"Global area-weighted RMSE of {variable}")
    ax.set_ylabel(f"RMSE ({skill[f'{variable}_pred'].attrs.get('units', '')})")
    ax.grid(alpha=0.3)
    return fig


def plot_control(control, variables=None):
    """
    Global-integrated anomaly through the control run (drift under repeated
    forcing), one panel per variable (default: every prognostic variable).
    """
    variables = [variables] if isinstance(variables, str) else variables or prognostic_variables(control)
    fig, axes = plt.subplots(len(variables), 1, figsize=(11, 3.2 * len(variables)), sharex=True,
                             squeeze=False, constrained_layout=True)
    for ax, variable in zip(axes[:, 0], variables):
        _plot_global_anomaly(ax, control, variable, suffixes=(("pred_anom", "Predicted"),), x="step")
        ax.set_title(f"Control run: global-integrated {variable} anomaly")
    axes[-1, 0].set_xlabel("Rollout step (months)")
    return fig


def plot_training_history(run_dir):
    """Train/validation loss per epoch from the CSV log in run_dir (needs train.run_dir)."""
    metrics = pd.read_csv(Path(run_dir) / "metrics.csv")
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    for column in ("train_loss", "val_loss"):
        if column in metrics:
            series = metrics.dropna(subset=[column]).groupby("epoch")[column].mean()
            ax.plot(series.index, series.values, label=column, lw=2)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_title("Training history")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    return fig


def plot_growth_mode(result, data):
    """Maps of the leading growth mode from evaluation.leading_growth_mode, one panel per variable."""
    lat, lon = data.fields["latitude"], data.fields["longitude"]
    names = result["names"]
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4), squeeze=False, constrained_layout=True)
    for ax, name, field in zip(axes[0], names, result["mode"]):
        vmax = np.nanquantile(np.abs(field), 0.99) or 1.0
        image = ax.pcolormesh(lon, lat, field, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        ax.set_title(name)
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle(f"Leading growth mode around {result['month']}: growth factor {result['growth']:.3f} per step")
    return fig
