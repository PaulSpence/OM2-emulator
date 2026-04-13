"""Data and pipeline helpers for OM2 emulator experiments.

This module wraps the existing ACCESS_OHC PyEarthTools accessor into a
configurable pipeline. Keeping this logic in one place makes it easier to
reuse exactly the same data path for baseline and ablation runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

import pyearthtools.data as petdata
import pyearthtools.pipeline as petpipe

from om2_data_utils import ACCESS_OHC


def build_pipeline(data_config: dict[str, Any]) -> petpipe.Pipeline:
    """Build a PyEarthTools pipeline from experiment config.

    Parameters
    ----------
    data_config
        ``data`` section of a resolved experiment config.

    Returns
    -------
    petpipe.Pipeline
        Time-iterable pipeline producing numpy arrays in ``t c h w`` format.

    Notes
    -----
    The steps mirror the notebook workflow but are now parameterized so we can
    evaluate many model variants with identical preprocessing.
    """
    data_root = Path(data_config["root"]).expanduser().resolve()

    accessor = ACCESS_OHC(
        data_config["variables"],
        root=data_root,
    )

    pipeline = petpipe.Pipeline(
        accessor,
        petdata.transforms.coordinates.Drop(data_config["drop_coordinates"]),
        petdata.transforms.variables.Drop(data_config["drop_variables"]),
        petpipe.operations.xarray.Sort(order=data_config["sort_order"]),
        petpipe.operations.xarray.values.FillNan(data_config["fill_nan"]),
        petpipe.operations.xarray.conversion.ToNumpy(),
        petpipe.operations.numpy.reshape.Rearrange(data_config["reshape_pattern"]),
        iterator=petpipe.iterators.DateRange(
            data_config["time_start"],
            data_config["time_end"],
            interval=data_config["time_interval"],
        ),
    )

    return pipeline


def load_mask_and_area(data_config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Load static land-sea mask and cell areas from netCDF.

    Parameters
    ----------
    data_config
        ``data`` section of the resolved experiment config.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(mask, area_weighted_mask)`` arrays with shape ``(h, w)``.

    Notes
    -----
    - The mask is derived from the first timestep of heat flux where NaN means
      land and finite values mean ocean.
    - ``area_weighted_mask`` is used for area-weighted RMSE diagnostics.
    """
    data_root = Path(data_config["root"]).expanduser().resolve()
    dataset_path = data_root / data_config["dataset_file"]

    with xr.open_dataset(dataset_path) as dataset:
        mask_xr = xr.where(np.isnan(dataset.total_surface_heat_flx.isel(time=0)), 0.0, 1.0)
        mask = mask_xr.values.astype(np.float32)

        area_t = dataset.area_t.values.astype(np.float64)

    area_weighted_mask = area_t * mask
    return mask, area_weighted_mask


def take_pipeline_sample(pipeline: petpipe.Pipeline, sample_index: int = 0) -> np.ndarray:
    """Extract a deterministic sample from a pipeline iterator.

    Parameters
    ----------
    pipeline
        Pipeline created by :func:`build_pipeline`.
    sample_index
        Zero-based sample offset to fetch.

    Returns
    -------
    np.ndarray
        Sample tensor in ``(1, c, h, w)`` shape.

    Raises
    ------
    IndexError
        Raised if the pipeline has fewer samples than ``sample_index + 1``.

    Notes
    -----
    This helper gives a stable diagnostics snapshot across runs, which is
    important when visually comparing error and latent feature maps.
    """
    iterator = iter(pipeline)
    sample: np.ndarray | None = None
    for _ in range(sample_index + 1):
        sample = next(iterator, None)
        if sample is None:
            raise IndexError(
                f"Pipeline ended before sample_index={sample_index}; no diagnostics sample available."
            )

    return sample
