"""
Data utilities for ACCESS-OM2 emulator.

Provides custom data archive accessors for loading ocean heat content and surface heat flux data from ACCESS-OM2.
"""

from pathlib import Path
from typing import Any, Literal

import pyearthtools.data as petdata
from pyearthtools.data.transforms import TransformCollection
import pyearthtools.data.archive as archive
from pyearthtools.data.indexes import ArchiveIndex
from pyearthtools.data.exceptions import DataNotFoundError as PetDataNotFoundError
import xarray as xr


class DataNotFoundError(PetDataNotFoundError):
    """Exception raised when expected data file is not found."""


@archive.register_archive(
    "ACCESS_OHC",
    sample_kwargs=dict(
        variables="ocean_heat_content_2d",
        root="",
    ),
)
class ACCESS_OHC(ArchiveIndex):
    """User-defined ACCESS Ocean Heat Content archive.

    This archive provides access to ocean heat content data from the
    ACCESS-OM2 model. It supports querying by year, month, or specific datetime.
    """

    @property
    def _desc_(self):
        return {
            "singleline": "ACCESS Ocean Heat Content (user extension)",
            "Range": "2000–2018",
            "Resolution": "1-degree",
        }

    def __init__(
        self,
        variables: str | list[str],
        *,
        root: str | Path,
        transforms=None,
        data_interval: tuple[int, str] | int | str = (1, "month"),
        surface_flux_mode: Literal["flux", "cumsum_energy"] = "flux",
        surface_flux_variable: str = "total_surface_heat_flx",
        cumsum_default_step_seconds: float = 30 * 24 * 3600,
        ohc_mode: Literal["absolute", "anomaly_from_first"] = "absolute",
        ohc_variable: str = "ocean_heat_content_2d",
        baseline_time: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the ACCESS_OHC archive.

        Parameters
        ----------
        variables : str or list[str]
            Variable name(s) to extract from the archive.
        root : str or Path
            Root directory containing the ocean heat data.
        transforms : TransformCollection, optional
            Data transformations to apply.
        data_interval : tuple[int, str] | int | str, optional
            Nominal temporal interval for AdvancedTimeIndex-style operations,
            by default (1, "month").
        surface_flux_mode : {"flux", "cumsum_energy"}, optional
            "flux" keeps surface flux in W/m^2. "cumsum_energy" converts
            surface flux to cumulative energy (J/m^2) via time integration.
        surface_flux_variable : str, optional
            Name of the surface-flux variable to convert/mask from,
            by default "total_surface_heat_flx".
        cumsum_default_step_seconds : float, optional
            Fallback integration step in seconds if datetime metadata is
            unavailable for deriving month lengths.
        ohc_mode : {"absolute", "anomaly_from_first"}, optional
            "absolute" keeps ocean heat content unchanged.
            "anomaly_from_first" converts to OHC(t) - OHC(t0).
        ohc_variable : str, optional
            Name of the OHC variable to convert, by default
            "ocean_heat_content_2d".
        baseline_time : str or None, optional
            Time label used as shared reference for cumulative/anomaly
            channels. If None, the first available time in file is used.
        **kwargs
            Additional arguments passed to parent class.
        """
        self.variables = [variables] if isinstance(variables, str) else list(variables)
        self.root = Path(root)
        self.surface_flux_mode = surface_flux_mode
        self.surface_flux_variable = surface_flux_variable
        self.cumsum_default_step_seconds = float(cumsum_default_step_seconds)
        self.ohc_mode = ohc_mode
        self.ohc_variable = ohc_variable
        self.baseline_time = baseline_time

        if self.surface_flux_mode not in {"flux", "cumsum_energy"}:
            raise ValueError(
                "surface_flux_mode must be one of {'flux', 'cumsum_energy'}."
            )

        if self.ohc_mode not in {"absolute", "anomaly_from_first"}:
            raise ValueError(
                "ohc_mode must be one of {'absolute', 'anomaly_from_first'}."
            )

        self._ocean_mask: xr.DataArray | None = None
        self._surface_flux_energy_cumsum: xr.DataArray | None = None
        self._ohc_anomaly_from_first: xr.DataArray | None = None

        base = petdata.transforms.variables.Trim(self.variables)

        super().__init__(
            transforms=base + (transforms or TransformCollection()),
            data_interval=data_interval,
            **kwargs,
        )
        self.record_initialisation()

    def _dataset_path(self) -> Path:
        """Return the canonical ACCESS_OHC NetCDF path."""
        return self.root / "1deg_ocean_heat_emulator_data.nc"

    def _get_ocean_mask(self) -> xr.DataArray:
        """Compute and cache ocean-valid mask (1=ocean, 0=land)."""
        if self._ocean_mask is None:
            path = self._dataset_path()
            if not path.exists():
                raise DataNotFoundError(f"ACCESS_OHC file not found at {path!r}")

            with xr.open_dataset(path) as ds_mask:
                if self.surface_flux_variable not in ds_mask:
                    raise DataNotFoundError(
                        f"ACCESS_OHC variable {self.surface_flux_variable!r} not found for mask creation."
                    )

                mask = xr.where(
                    ds_mask[self.surface_flux_variable].isel(time=0).isnull(),
                    0,
                    1,
                )
                self._ocean_mask = mask.load()

        return self._ocean_mask

    def _apply_land_nan_mask(self, data: xr.Dataset | xr.DataArray):
        """Set land points to NaN using mask derived from surface heat flux."""
        ocean_mask = self._get_ocean_mask()
        ocean_valid = ocean_mask == 1

        if isinstance(data, xr.Dataset):
            masked = data.copy()
            for name, da in masked.data_vars.items():
                if {"yt_ocean", "xt_ocean"}.issubset(da.dims):
                    masked[name] = da.where(ocean_valid)
            return masked

        if isinstance(data, xr.DataArray) and {"yt_ocean", "xt_ocean"}.issubset(data.dims):
            return data.where(ocean_valid)

        return data

    def _deduplicate_time_index(self, da: xr.DataArray) -> xr.DataArray:
        """Ensure unique time index for robust alignment/selection."""
        if "time" not in da.dims:
            return da

        idx = da.indexes.get("time", None)
        if idx is not None and getattr(idx, "has_duplicates", False):
            # Collapse duplicate timestamps deterministically.
            da = da.groupby("time").first()

        return da

    def _select_baseline(self, da: xr.DataArray) -> xr.DataArray:
        """Select baseline slice along time using baseline_time when provided."""
        if "time" not in da.dims:
            return da

        da = self._deduplicate_time_index(da)

        if self.baseline_time is None:
            baseline = da.isel(time=0)
        else:
            try:
                baseline = da.sel(time=self.baseline_time)
            except Exception:
                # Fallback to nearest if exact label does not exist.
                baseline = da.sel(time=self.baseline_time, method="nearest")

        # Crucial: drop time coordinate so subtraction broadcasts over all times
        # without triggering time-index alignment side effects.
        if "time" in baseline.dims:
            baseline = baseline.isel(time=0, drop=True)

        return baseline

    def _subtract_baseline(self, da: xr.DataArray) -> xr.DataArray:
        """Subtract selected baseline from time-varying data array."""
        baseline = self._select_baseline(da)
        return da - baseline

    def _align_time_like(self, source: xr.DataArray, target: xr.DataArray) -> xr.DataArray:
        """Align `source` time coordinates to match `target` robustly."""
        if "time" not in source.dims or "time" not in target.dims:
            return source

        source = self._deduplicate_time_index(source)
        target_time = target["time"]

        # Preferred: exact coordinate match.
        try:
            return source.sel(time=target_time)
        except Exception:
            pass

        # Fallback: nearest neighbour alignment (month-start vs mid-month labels).
        try:
            return source.sel(time=target_time, method="nearest")
        except Exception as exc:
            raise DataNotFoundError(
                "Unable to align transformed channel time coordinates to requested sample times."
            ) from exc

    def _get_surface_flux_energy_cumsum(self) -> xr.DataArray:
        """Return cached cumulative surface energy (J/m^2) from flux (W/m^2)."""
        if self._surface_flux_energy_cumsum is not None:
            return self._surface_flux_energy_cumsum

        path = self._dataset_path()
        if not path.exists():
            raise DataNotFoundError(f"ACCESS_OHC file not found at {path!r}")

        with xr.open_dataset(path) as ds:
            if self.surface_flux_variable not in ds:
                raise DataNotFoundError(
                    f"ACCESS_OHC variable {self.surface_flux_variable!r} not found for cumulative conversion."
                )

            flux = ds[self.surface_flux_variable]
            if "time" not in flux.dims:
                raise DataNotFoundError(
                    f"ACCESS_OHC variable {self.surface_flux_variable!r} has no 'time' dimension."
                )

            try:
                dt_seconds = flux["time"].dt.days_in_month.astype("float64") * 86400.0
            except Exception:
                dt_seconds = xr.DataArray(
                    [self.cumsum_default_step_seconds] * flux.sizes["time"],
                    coords={"time": flux["time"]},
                    dims=("time",),
                )

            energy_increment = flux.astype("float64") * dt_seconds
            energy_cumsum = energy_increment.cumsum(dim="time")

            # Align baseline with OHC anomaly convention so both channels are
            # referenced to the same baseline_time.
            energy_cumsum = self._subtract_baseline(energy_cumsum)

            self._surface_flux_energy_cumsum = energy_cumsum.load()

        return self._surface_flux_energy_cumsum

    def _convert_surface_flux_channel(self, data: xr.Dataset | xr.DataArray):
        """Optionally convert surface flux channel from W/m^2 to cumulative J/m^2."""
        if self.surface_flux_mode != "cumsum_energy":
            return data

        cumulative = self._get_surface_flux_energy_cumsum()

        if isinstance(data, xr.Dataset):
            if self.surface_flux_variable not in data:
                return data

            converted = data.copy()
            target = converted[self.surface_flux_variable]
            if "time" in target.dims:
                converted[self.surface_flux_variable] = self._align_time_like(cumulative, target)
            else:
                converted[self.surface_flux_variable] = cumulative
            return converted

        if isinstance(data, xr.DataArray) and data.name == self.surface_flux_variable:
            if "time" in data.dims:
                return self._align_time_like(cumulative, data)
            return cumulative

        return data

    def _get_ohc_anomaly_from_first(self) -> xr.DataArray:
        """Return cached OHC anomaly: OHC(t) - OHC(t0)."""
        if self._ohc_anomaly_from_first is not None:
            return self._ohc_anomaly_from_first

        path = self._dataset_path()
        if not path.exists():
            raise DataNotFoundError(f"ACCESS_OHC file not found at {path!r}")

        with xr.open_dataset(path) as ds:
            if self.ohc_variable not in ds:
                raise DataNotFoundError(
                    f"ACCESS_OHC variable {self.ohc_variable!r} not found for OHC anomaly conversion."
                )

            ohc = ds[self.ohc_variable]
            if "time" not in ohc.dims:
                raise DataNotFoundError(
                    f"ACCESS_OHC variable {self.ohc_variable!r} has no 'time' dimension."
                )

            ohc = ohc.astype("float64")
            self._ohc_anomaly_from_first = self._subtract_baseline(ohc).load()

        return self._ohc_anomaly_from_first

    def _convert_ohc_channel(self, data: xr.Dataset | xr.DataArray):
        """Optionally convert OHC channel to anomaly from first timestep."""
        if self.ohc_mode != "anomaly_from_first":
            return data

        ohc_anomaly = self._get_ohc_anomaly_from_first()

        if isinstance(data, xr.Dataset):
            if self.ohc_variable not in data:
                return data

            converted = data.copy()
            target = converted[self.ohc_variable]
            if "time" in target.dims:
                converted[self.ohc_variable] = self._align_time_like(ohc_anomaly, target)
            else:
                converted[self.ohc_variable] = ohc_anomaly
            return converted

        if isinstance(data, xr.DataArray) and data.name == self.ohc_variable:
            if "time" in data.dims:
                return self._align_time_like(ohc_anomaly, data)
            return ohc_anomaly

        return data

    def filesystem(self, querytime, **kwargs):
        """Resolve archive path for a given query time.

        The ACCESS_OHC dataset is time-complete and stored in one NetCDF file,
        so all query times map to the same path.
        """
        path = self._dataset_path()
        if not path.exists():
            raise DataNotFoundError(f"ACCESS_OHC file not found at {path!r}")

        # Return a single file path (not {var: path} mapping). The per-variable
        # selection is already handled by the Trim transform in __init__, and
        # returning duplicate paths via a dict can push xarray/open_mfdataset
        # into an unnecessary concat path during pipeline iteration.
        return path

    def get(self, querytime, **kwargs):
        """Retrieve data, optionally convert flux channel, and mask land points."""
        data = super().get(querytime, **kwargs)
        data = self._convert_surface_flux_channel(data)
        data = self._convert_ohc_channel(data)
        return self._apply_land_nan_mask(data)
