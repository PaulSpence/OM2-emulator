"""
Data utilities for ACCESS-OM2 emulator.

Provides custom data archive accessors for loading ocean heat content and surface heat flux data from ACCESS-OM2.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import pyearthtools.data as petdata
from pyearthtools.data.transforms import TransformCollection
import pyearthtools.data.archive as archive
from pyearthtools.data.indexes import DataFileSystemIndex
from pyearthtools.data.time import Petdt


class DataNotFoundError(Exception):
    """Exception raised when expected data file is not found."""
    pass


@archive.register_archive(
    "ACCESS_OHC",
    sample_kwargs=dict(
        variables="ocean_heat_content_2d",
        root="",
    ),
)
class ACCESS_OHC(DataFileSystemIndex):
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
        **kwargs
            Additional arguments passed to parent class.
        """
        self.variables = [variables] if isinstance(variables, str) else list(variables)
        self.root = Path(root)

        base = petdata.transforms.variables.Trim(self.variables)

        super().__init__(
            transforms=base + (transforms or TransformCollection()),
            **kwargs,
        )
        self.record_initialisation()

    def search(self, *args, **kwargs):
        """Search for data files.
        
        Ignore time entirely: dataset is time-complete.
        """
        path = self.root / "1deg_ocean_heat_emulator_data.nc"
        if not path.exists():
            raise DataNotFoundError(f"ACCESS_OHC file not found at {path!r}")

        # Map each requested variable to the same file
        return {v: path for v in self.variables}

    def get(self, querytime, **kwargs):
        """Retrieve data for a specific time or time range.
        
        Parameters
        ----------
        querytime : str
            Time specification. Can be:
            - "YYYY" for full year
            - "YYYY-MM" for specific month
            - datetime string for nearest match
            
        Returns
        -------
        xr.Dataset
            Dataset containing requested variables for the specified time.
        """
        path = self.root / "1deg_ocean_heat_emulator_data.nc"
        if not path.exists():
            raise DataNotFoundError(f"ACCESS_OHC file not found at {path!r}")
    
        ds = xr.open_dataset(path)
    
        # Keep only requested variables
        keep = [v for v in self.variables if v in ds.data_vars]
        if keep:
            ds = ds[keep]
    
        qt = str(querytime)
    
        # -------------------------
        # Case 1: "YYYY" → full year
        # -------------------------
        if len(qt) == 4 and qt.isdigit():
            start = pd.Timestamp(f"{qt}-01-01")
            end = pd.Timestamp(f"{int(qt)+1}-01-01")
            return ds.sel(time=slice(
                np.datetime64(start),
                np.datetime64(end),
            ))
    
        # -------------------------
        # Case 2: "YYYY-MM" → month
        # -------------------------
        if len(qt) == 7 and qt[4] == "-":
            start = pd.Timestamp(f"{qt}-01")
            end = start + pd.offsets.MonthBegin(1)
            return ds.sel(time=slice(
                np.datetime64(start),
                np.datetime64(end),
            ))
    
        # -------------------------
        # Case 3: exact datetime → nearest
        # -------------------------
        qt_dt = np.datetime64(str(Petdt(querytime)))
        return ds.sel(time=qt_dt, method="nearest")
