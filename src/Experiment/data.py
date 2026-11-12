"""
Data loading, normalisation, caching and rollout datasets.

Pipeline:
  1. open the NetCDF file and select the configured time axis;
  2. compute the normalisation statistics over the fit period;
  3. normalise every prognostic and forcing variable once, over the whole time
     axis, into (time, variable, H, W) tensors ("fields");
  4. cache the fields on disk, keyed by a hash of every data setting;
  5. serve training/validation samples as windows into the fields, built on
     the fly by integer indexing (no per-sample copies are stored).

This replaces the PET pipelines and the per-date cache of the older notebooks.
All tensors are float32; land cells are 0 in the normalised fields.
"""

import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader

CACHE_FORMAT_VERSION = 3  # 3: month_seconds and global integrals for the budget constraint


# =============================================================================
# Normalisation statistics
# =============================================================================

def compute_normalisation(ds, variables, fit_period, mask_variable):
    """
    The monthly climatology of every variable: the mean and std of each grid
    cell for each calendar month, fitted over fit_period (first and last month,
    inclusive). Both are returned on ds's whole time axis: every month gets its
    calendar month's values. Also returns the ocean mask: 1 where
    mask_variable has data in the first month, 0 on land.

    ds has dimensions (time, latitude, longitude). The numbers are the same as
    the "Spatial_climatology" strategy of Data.build_normalisation (the
    PyEarthTools version used by the older notebooks); the test suite checks
    this.
    """
    mask = xr.where(np.isnan(ds[mask_variable].isel(time=0)), 0, 1)
    fit = ds[variables].where(mask == 1).sel(time=slice(*fit_period))

    # Mean and std of every cell for each calendar month, (month, latitude, longitude). The std is that of
    # the anomalies from the monthly mean.
    monthly_mean = fit.groupby("time.month").mean("time")
    monthly_std = (fit.groupby("time.month") - monthly_mean).groupby("time.month").std("time")

    # Give every month of ds its calendar month's values: (time, latitude, longitude).
    calendar_month = ds.time.dt.month
    mean = monthly_mean.sel(month=calendar_month).drop_vars("month")
    std = monthly_std.sel(month=calendar_month).drop_vars("month")
    return mean, std, mask


# =============================================================================
# Fields: normalised tensors over the whole time axis
# =============================================================================

def _open_time_axis(cfg):
    """Open the file and select the configured monthly time axis."""
    t = cfg.time
    months = pd.date_range(start=t.time_axis[0], end=t.time_axis[1], freq="MS")
    ds = xr.open_dataset(cfg.data.path)
    missing = [v for v in (*cfg.data.prognostic, *cfg.data.forcing, cfg.data.area_variable) if v not in ds]
    if missing:
        ds.close()
        raise KeyError(f"{cfg.data.path} is missing {missing}; it has {sorted(ds.data_vars)}")

    # Guard against the Kelvin-OHC bug (fixed in Extract_om2_data.ipynb).
    heat = cfg.active_closures().get("heat")
    ohc = heat.content_variable if heat is not None else None
    if ohc in ds and "273.15" not in str(ds[ohc].attrs.get("description", "")):
        warnings.warn(
            f"{ohc} in {cfg.data.path} does not say it was computed from Celsius "
            "temperature (description lacks '273.15'). OHC computed from Kelvin cannot "
            "close against the surface heat flux -- re-extract the dataset."
        )
    ds = ds.rename({"yt_ocean": "latitude", "xt_ocean": "longitude"})
    return ds.sel(time=months, method="nearest")


def build_fields(cfg):
    """Normalise every variable over the whole time axis. Returns a dict of tensors and metadata."""
    d = cfg.data
    variables = [*d.prognostic, *d.forcing]
    with _open_time_axis(cfg) as ds:
        mean, std, mask = compute_normalisation(ds, variables, d.normalisation.fit_period, d.mask_variable)

        # A variable missing at ocean cells is silently filled with zero anomaly
        # (z = 0) below: say so. (v1 of the dataset lost tau_x, tau_y and
        # surface_speed in a ring of coastal cells and the seam column to its
        # u-to-t regrid; see om2_grid.regrid_u_to_t.)
        ocean = np.asarray(mask.values) == 1
        for v in variables:
            missing = int((ocean & ~np.isfinite(ds[v].isel(time=0).values)).sum())
            if missing:
                warnings.warn(
                    f"{v} is missing at {missing} of {int(ocean.sum())} ocean cells (first month); "
                    "those cells get zero anomaly. Regenerate the dataset if this is not intended."
                )

        def normalised(v):
            field = ds[v].transpose("time", "latitude", "longitude")
            z = ((field - mean[v]) / std[v]).fillna(0.0)
            return z.transpose("time", "latitude", "longitude").load()

        def stack(names, fn):
            return torch.stack(
                [torch.as_tensor(np.asarray(fn(v).values), dtype=torch.float32) for v in names], dim=1
            )

        prognostic = stack(d.prognostic, normalised)
        forcing = stack(d.forcing, normalised)
        like = ds[d.prognostic[0]]
        stat = lambda s: lambda v: s[v].transpose("time", "latitude", "longitude").fillna(0.0)
        prognostic_mean = stack(d.prognostic, stat(mean))
        prognostic_std = stack(d.prognostic, stat(std))
        forcing_std = stack(d.forcing, stat(std))

        area = ds[d.area_variable]
        if "time" in area.dims:
            area = area.isel(time=0)
        mask_np = np.asarray(mask.values, dtype=np.float32)
        area_np = np.nan_to_num(np.asarray(area.values, dtype=np.float64)) * mask_np

        units = [str(ds[v].attrs.get("units", "")) for v in d.prognostic]
        months = [pd.Timestamp(x).strftime("%Y-%m") for x in ds.time.values]
        month_seconds = [pd.Period(m, freq="M").days_in_month * 86400.0 for m in months]

        # Ocean-integrated (area-weighted) totals per month, in physical units,
        # for the global budget constraint: every forcing variable (e.g. W, kg/s)
        # and the climatological mean of every prognostic variable (e.g. J, kg).
        ocean_area = xr.DataArray(area_np, dims=("latitude", "longitude"))
        def ocean_integral(field):
            field = field.transpose("time", "latitude", "longitude").fillna(0.0)
            return torch.as_tensor((field * ocean_area.values).sum(("latitude", "longitude")).values,
                                   dtype=torch.float64)
        forcing_global = torch.stack([ocean_integral(ds[v]) for v in d.forcing], dim=1)
        mean_global = (prognostic_mean.double() * torch.as_tensor(area_np)).sum(dim=(-2, -1))
        latitude = np.asarray(like["latitude"].values)
        longitude = np.asarray(like["longitude"].values)

    return {
        "format_version": CACHE_FORMAT_VERSION,
        "prognostic": prognostic,            # (T, P, H, W) z-scores
        "forcing": forcing,                  # (T, C, H, W) z-scores
        "prognostic_mean": prognostic_mean,  # (T, P, H, W) physical units
        "prognostic_std": prognostic_std,    # (T, P, H, W) physical units
        "forcing_std": forcing_std,          # (T, C, H, W) physical units (closure fluxes)
        "forcing_global": forcing_global,    # (T, C) float64 ocean integral of each forcing, physical units x m^2
        "prognostic_mean_global": mean_global,  # (T, P) float64 ocean integral of the climatological mean
        "month_seconds": torch.as_tensor(month_seconds, dtype=torch.float64),  # (T,) length of each month
        "mask": torch.as_tensor(mask_np),    # (H, W), 1 = ocean
        "area": torch.as_tensor(area_np, dtype=torch.float64),  # (H, W) m^2, 0 on land
        "months": months,
        "latitude": latitude,
        "longitude": longitude,
        "prognostic_names": list(d.prognostic),
        "prognostic_units": units,
        "forcing_names": list(d.forcing),
        "data_contract": cfg.data_contract(),
    }


def _file_stamp(path):
    st = os.stat(path)
    return f"{st.st_size}:{int(st.st_mtime)}"


def cache_path(cfg):
    cache_dir = Path(cfg.data.cache_dir) if cfg.data.cache_dir else Path(cfg.data.path).parent / "emulator_cache"
    return cache_dir / f"fields_{cfg.data_hash(_file_stamp(cfg.data.path))}.pt"


def load_or_build_fields(cfg, rebuild=False, verbose=True):
    """Load the cached fields for this config, building (and caching) them if needed."""
    path = cache_path(cfg)
    if path.exists() and not rebuild:
        fields = torch.load(path, map_location="cpu", weights_only=False)
        if fields.get("format_version") == CACHE_FORMAT_VERSION:
            if verbose:
                print(f"Loaded cached fields: {path}")
            return fields
    if verbose:
        print("Building normalised fields (one pass over the file)...")
    fields = build_fields(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(fields, tmp)
    tmp.rename(path)
    if verbose:
        print(f"Cached fields: {path}")
    return fields


# =============================================================================
# Rollout windows
# =============================================================================

def gather_windows(prognostic, forcing, t0, n_prior, n_steps):
    """
    Rollout windows for the initial months t0 (B,), cut from the normalised
    fields on the device they live on (no CPU work, no host-to-device copy
    when the fields are on the GPU). Returns:

      prior             (B, n_prior, P, H, W)  months t0-n_prior+1 ... t0
      target            (B, n_steps, P, H, W)  months t0+1 ... t0+n_steps
      forcing           (B, n_steps, C, H, W)  months t0+1 ... t0+n_steps
      initial_forcing   (B, C, H, W)           month t0 (closure's two-month flux average)
      target_time_index (B, n_steps)           time indices t0+1 ... t0+n_steps

    Only n_steps target/forcing months are cut, so short rollouts (e.g. early
    in a rollout_schedule) move proportionally less data.
    """
    device = prognostic.device
    t0 = torch.as_tensor(t0, dtype=torch.long).to(device)
    prior_index = t0[:, None] + torch.arange(-n_prior + 1, 1, device=device)
    target_index = t0[:, None] + torch.arange(1, n_steps + 1, device=device)
    return {
        "prior": prognostic[prior_index],
        "target": prognostic[target_index],
        "forcing": forcing[target_index],
        "initial_forcing": forcing[t0],
        "target_time_index": target_index,
    }


@dataclass
class ExperimentData:
    """
    Everything downstream code needs from the data, plus the dataloaders.

    The dataloaders yield batches of initial months (B,) only; turn them into
    windows with ``windows`` (on the CPU) or let EmulatorModule cut them on
    the GPU.
    """

    fields: dict
    month_index: dict
    train_indices: torch.Tensor
    valid_indices: torch.Tensor
    train_dl: DataLoader
    valid_dl: DataLoader
    n_prior: int
    posterior_steps: int

    @property
    def n_prognostic(self):
        return self.fields["prognostic"].shape[1]

    @property
    def n_forcing(self):
        return self.fields["forcing"].shape[1]

    @property
    def months(self):
        return self.fields["months"]

    def windows(self, t0, n_steps=None, n_prior=None):
        """Rollout windows (see gather_windows) for initial months t0, from the CPU fields."""
        f = self.fields
        return gather_windows(
            f["prognostic"], f["forcing"], t0,
            n_prior or self.n_prior, n_steps or self.posterior_steps,
        )

    def index_of(self, month):
        try:
            return self.month_index[str(pd.Period(month, freq="M"))]
        except KeyError as err:
            raise KeyError(f"{month} is not on the time axis {self.months[0]}..{self.months[-1]}") from err

    def summary(self):
        f = self.fields
        T, P, H, W = f["prognostic"].shape
        return (
            f"time axis   : {self.months[0]} .. {self.months[-1]} ({T} months), grid {H} x {W}\n"
            f"prognostic  : {f['prognostic_names']}\n"
            f"forcing     : {f['forcing_names']}\n"
            f"train       : {len(self.train_indices)} samples, {len(self.train_dl)} batches\n"
            f"valid       : {len(self.valid_indices)} samples, {len(self.valid_dl)} batches"
        )


def split_indices(month_index, date_range, n_prior, n_steps, n_time, name):
    """Time indices of the initial months in an inclusive date range, with bounds checks."""
    months = pd.period_range(date_range[0], date_range[1], freq="M")
    try:
        indices = np.array([month_index[str(m)] for m in months])
    except KeyError as err:
        raise ValueError(f"{name} range {date_range} is not on the time axis") from err
    if indices.min() - n_prior + 1 < 0:
        raise ValueError(f"{name} starts too early: its first sample needs {n_prior} prior months on the time axis")
    if indices.max() + n_steps > n_time - 1:
        raise ValueError(f"{name} ends too late: its last sample needs {n_steps} target months on the time axis")
    return torch.as_tensor(indices, dtype=torch.long)


def build_data(cfg, rebuild=False, verbose=True):
    """Load (or build) the cached fields and create the train/valid dataloaders."""
    fields = load_or_build_fields(cfg, rebuild=rebuild, verbose=verbose)
    month_index = {m: i for i, m in enumerate(fields["months"])}
    n_time = len(fields["months"])
    w = cfg.window
    train_indices = split_indices(month_index, cfg.time.train, w.n_prior, w.posterior_steps, n_time, "train")
    # Training windows must end inside the training period: otherwise the last
    # samples' targets (posterior_steps months ahead) are validation or test months.
    train_end = month_index[str(pd.Period(cfg.time.train[1], freq="M"))]
    train_indices = train_indices[train_indices + w.posterior_steps <= train_end]
    if len(train_indices) == 0:
        raise ValueError(f"time.train {cfg.time.train} is too short for {w.posterior_steps}-month windows "
                         "that end inside it")
    valid_indices = split_indices(month_index, cfg.time.valid, w.n_prior, w.posterior_steps, n_time, "valid")

    # A batch is just B initial months (time indices): the DataLoader batches the
    # tensor of indices directly. The windows are cut on the GPU (see
    # gather_windows), so no worker processes are needed.
    tr = cfg.train
    train_dl = DataLoader(torch.as_tensor(train_indices, dtype=torch.long), batch_size=tr.batch_size, shuffle=True,
                          num_workers=tr.num_workers)
    valid_dl = DataLoader(torch.as_tensor(valid_indices, dtype=torch.long), batch_size=tr.batch_size, shuffle=False,
                          num_workers=tr.num_workers)
    data = ExperimentData(fields, month_index, train_indices, valid_indices, train_dl, valid_dl,
                          w.n_prior, w.posterior_steps)
    if verbose:
        print(data.summary())
    return data
