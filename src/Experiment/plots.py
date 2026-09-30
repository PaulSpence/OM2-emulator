"""
Standard figures for the skill test, the control run and training history.

All functions take the Datasets returned by evaluation.run_skill_test /
run_control and return the matplotlib Figure.
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


def plot_snapshots(skill, periods, anomaly_scale=2e9, difference_scale=2e9, variable=None):
    """Predicted anomaly, true anomaly and their difference, one row per period."""
    variable = variable or _default_variable(skill)
    pred, truth = skill[f"{variable}_pred_anom"], skill[f"{variable}_truth_anom"]
    units = pred.attrs.get("units", "")
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


def plot_global_timeseries(skill, variable=None, scale=1e22, scale_label="$10^{22}$"):
    """Global-integrated anomaly, predicted vs true, through the skill test."""
    variable = variable or _default_variable(skill)
    area = skill["area"]
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    for suffix, label in (("pred_anom", "Predicted"), ("truth_anom", "Truth")):
        ax.plot(skill.time.values, global_integral(skill[f"{variable}_{suffix}"], area).values / scale, label=label, lw=2)
    ax.set_title(f"Global-integrated {variable} anomaly through the skill test")
    ax.set_ylabel(f"Integrated anomaly ({scale_label} x area units)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
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


def plot_control(control, variable=None, scale=1e22, scale_label="$10^{22}$"):
    """Global-integrated anomaly through the control run (drift under repeated forcing)."""
    variable = variable or _default_variable(control)
    series = global_integral(control[f"{variable}_pred_anom"], control["area"])
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    ax.plot(control.step.values, series.values / scale, lw=2)
    ax.set_title(f"Control run: global-integrated {variable} anomaly")
    ax.set_xlabel("Rollout step (months)")
    ax.set_ylabel(f"Integrated anomaly ({scale_label} x area units)")
    ax.grid(alpha=0.3)
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
