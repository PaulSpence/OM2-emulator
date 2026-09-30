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
from torch.utils.data import DataLoader, Dataset

CACHE_FORMAT_VERSION = 1


# =============================================================================
# Normalisation statistics
# =============================================================================

def compute_normalisation(ds, variables, strategy, fit_period, mask_variable, area_variable, area_weight=True):
    """
    Mean and std of each variable, as xarray Datasets on (time, latitude, longitude).

    This is the statistics part of Data.build_normalisation (the PET-based
    version used by the older notebooks), ported to plain xarray so it runs
    without PyEarthTools. It must give identical numbers; the test suite checks
    this. ``ds`` has already been reduced to the configured time axis.

    Returns (mean, std, mask): mask is 1 over ocean and 0 over land, taken
    from where ``mask_variable`` is NaN at the first time.
    """
    area = ds[area_variable]
    stats = ds[list(variables)]

    mask = xr.where(np.isnan(stats[mask_variable].isel(time=0)), 0, 1)
    masked = stats.where(mask == 1)
    if area_weight:
        weight = area.where(mask == 1).fillna(0.0).load()
    else:
        weight = xr.ones_like(area).load()

    rename_map = {k: v for k, v in {"yt_ocean": "latitude", "xt_ocean": "longitude"}.items() if k in masked.dims}
    masked = masked.rename(rename_map)
    weight = weight.rename(rename_map)

    fit = masked.sel(time=slice(fit_period[0], fit_period[1]))
    spatial = ("latitude", "longitude")
    month_indexer = xr.DataArray(
        masked.time.dt.month.values, coords={"time": masked.time.values}, dims="time"
    )

    if strategy == "Global_in_time":
        mean = ((fit * weight).sum(dim=spatial, skipna=True) / weight.sum()).mean("time")
        std = (fit.std("time") * weight).sum(dim=spatial, skipna=True) / weight.sum()
        std = std.drop_vars(["time"], errors="ignore")
    elif strategy == "Spatial_in_time":
        mean = fit.mean(dim="time", skipna=True)
        std = fit.std(dim="time", skipna=True)
    elif strategy == "Spatial_climatology":
        monthly_mean = fit.groupby("time.month").mean(dim="time", skipna=True)
        monthly_std = (fit.groupby("time.month") - monthly_mean).groupby("time.month").std(dim="time", skipna=True)
        mean = monthly_mean.sel(month=month_indexer).drop_vars("month", errors="ignore")
        std = monthly_std.sel(month=month_indexer).drop_vars("month", errors="ignore")
    elif strategy == "Spatial_climatology_global_variance":
        monthly_mean = fit.groupby("time.month").mean(dim="time", skipna=True)
        std = ((fit.groupby("time.month") - monthly_mean).std("time") * weight).sum(dim=spatial, skipna=True) / weight.sum()
        std = std.drop_vars(["time"], errors="ignore")
        mean = monthly_mean.sel(month=month_indexer).drop_vars("month", errors="ignore")
    else:
        raise ValueError(f"Unknown normalisation strategy {strategy!r}")

    mean = xr.Dataset({v: mean[v] for v in variables})
    std = xr.Dataset({v: std[v] for v in variables})
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
    ohc = cfg.data.ohc_variable
    if ohc in ds and "273.15" not in str(ds[ohc].attrs.get("description", "")):
        warnings.warn(
            f"{ohc} in {cfg.data.path} does not say it was computed from Celsius "
            "temperature (description lacks '273.15'). OHC computed from Kelvin cannot "
            "close against the surface heat flux -- re-extract the dataset."
        )
    return ds.sel(time=months, method="nearest")


def _rename_xy(da):
    rename_map = {k: v for k, v in {"yt_ocean": "latitude", "xt_ocean": "longitude"}.items() if k in da.dims}
    return da.rename(rename_map)


def _broadcast_stat(stat, like):
    """Broadcast a (possibly scalar or time-less) statistic to like's (time, lat, lon)."""
    return xr.broadcast(stat, like)[0].transpose("time", "latitude", "longitude")


def build_fields(cfg):
    """Normalise every variable over the whole time axis. Returns a dict of tensors and metadata."""
    d = cfg.data
    variables = [*d.prognostic, *d.forcing]
    with _open_time_axis(cfg) as ds:
        mean, std, mask = compute_normalisation(
            ds,
            variables,
            d.normalisation.strategy,
            d.normalisation.fit_period,
            d.mask_variable,
            d.area_variable,
            d.normalisation.area_weight,
        )

        def normalised(v):
            field = _rename_xy(ds[v]).transpose("time", "latitude", "longitude")
            z = ((field - mean[v]) / std[v]).fillna(0.0)
            return z.transpose("time", "latitude", "longitude").load()

        def stack(names, fn):
            return torch.stack(
                [torch.as_tensor(np.asarray(fn(v).values), dtype=torch.float32) for v in names], dim=1
            )

        prognostic = stack(d.prognostic, normalised)
        forcing = stack(d.forcing, normalised)
        like = _rename_xy(ds[d.prognostic[0]]).transpose("time", "latitude", "longitude")
        stat = lambda s: lambda v: _broadcast_stat(s[v], like).fillna(0.0)
        prognostic_mean = stack(d.prognostic, stat(mean))
        prognostic_std = stack(d.prognostic, stat(std))
        heat_flux_std = None
        if d.heat_flux_variable in variables:
            heat_flux_std = torch.as_tensor(
                np.asarray(_broadcast_stat(std[d.heat_flux_variable], like).fillna(0.0).values), dtype=torch.float32
            )

        area = ds[d.area_variable]
        if "time" in area.dims:
            area = area.isel(time=0)
        mask_np = np.asarray(mask.values, dtype=np.float32)
        area_np = np.nan_to_num(np.asarray(area.values, dtype=np.float64)) * mask_np

        units = [str(ds[v].attrs.get("units", "")) for v in d.prognostic]
        months = [pd.Timestamp(x).strftime("%Y-%m") for x in ds.time.values]
        latitude = np.asarray(like["latitude"].values)
        longitude = np.asarray(like["longitude"].values)

    return {
        "format_version": CACHE_FORMAT_VERSION,
        "prognostic": prognostic,            # (T, P, H, W) z-scores
        "forcing": forcing,                  # (T, C, H, W) z-scores
        "prognostic_mean": prognostic_mean,  # (T, P, H, W) physical units
        "prognostic_std": prognostic_std,    # (T, P, H, W) physical units
        "heat_flux_std": heat_flux_std,      # (T, H, W) or None
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

class RolloutWindowDataset(Dataset):
    """
    Rollout samples as windows into the normalised fields.

    Sample i has initial month t0 = initial_indices[i] and returns:
      prior           (n_prior, P, H, W)  months t0-n_prior+1 ... t0
      target          (S, P, H, W)        months t0+1 ... t0+S
      forcing         (S, C, H, W)        months t0+1 ... t0+S
      initial_forcing (C, H, W)           month t0 (closure's two-month flux average)
      target_time_index (S,)              time indices t0+1 ... t0+S
    """

    def __init__(self, prognostic, forcing, initial_indices, n_prior, n_steps):
        self.prognostic = prognostic
        self.forcing = forcing
        self.initial_indices = torch.as_tensor(initial_indices, dtype=torch.long)
        self.n_prior = n_prior
        self.n_steps = n_steps

    def __len__(self):
        return len(self.initial_indices)

    def __getitem__(self, i):
        t0 = int(self.initial_indices[i])
        first, last = t0 + 1, t0 + 1 + self.n_steps
        return {
            "prior": self.prognostic[t0 - self.n_prior + 1 : t0 + 1],
            "target": self.prognostic[first:last],
            "forcing": self.forcing[first:last],
            "initial_forcing": self.forcing[t0],
            "target_time_index": torch.arange(first, last),
        }


@dataclass
class ExperimentData:
    """Everything downstream code needs from the data, plus the dataloaders."""

    fields: dict
    month_index: dict
    train_indices: torch.Tensor
    valid_indices: torch.Tensor
    train_dl: DataLoader
    valid_dl: DataLoader

    @property
    def n_prognostic(self):
        return self.fields["prognostic"].shape[1]

    @property
    def n_forcing(self):
        return self.fields["forcing"].shape[1]

    @property
    def months(self):
        return self.fields["months"]

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
    valid_indices = split_indices(month_index, cfg.time.valid, w.n_prior, w.posterior_steps, n_time, "valid")

    make = lambda idx: RolloutWindowDataset(fields["prognostic"], fields["forcing"], idx, w.n_prior, w.posterior_steps)
    tr = cfg.train
    train_dl = DataLoader(make(train_indices), batch_size=tr.batch_size, shuffle=True, num_workers=tr.num_workers)
    valid_dl = DataLoader(make(valid_indices), batch_size=tr.batch_size, shuffle=False, num_workers=tr.num_workers)
    data = ExperimentData(fields, month_index, train_indices, valid_indices, train_dl, valid_dl)
    if verbose:
        print(data.summary())
    return data
