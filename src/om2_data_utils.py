"""
Data utilities for ACCESS-OM2 emulator.

Provides custom data archive accessors for loading ocean heat content and surface heat flux data from ACCESS-OM2.
"""

from pathlib import Path
from typing import Any

import pyearthtools.data as petdata
from pyearthtools.data.transforms import TransformCollection
import pyearthtools.data.archive as archive
from pyearthtools.data.indexes import ArchiveIndex
from pyearthtools.data.exceptions import DataNotFoundError as PetDataNotFoundError


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
        **kwargs
            Additional arguments passed to parent class.
        """
        self.variables = [variables] if isinstance(variables, str) else list(variables)
        self.root = Path(root)

        base = petdata.transforms.variables.Trim(self.variables)

        super().__init__(
            transforms=base + (transforms or TransformCollection()),
            data_interval=data_interval,
            **kwargs,
        )
        self.record_initialisation()

    def filesystem(self, querytime, **kwargs):
        """Resolve archive path for a given query time.

        The ACCESS_OHC dataset is time-complete and stored in one NetCDF file,
        so all query times map to the same path.
        """
        path = self.root / "1deg_ocean_heat_emulator_data.nc"
        if not path.exists():
            raise DataNotFoundError(f"ACCESS_OHC file not found at {path!r}")

        # Return a single file path (not {var: path} mapping). The per-variable
        # selection is already handled by the Trim transform in __init__, and
        # returning duplicate paths via a dict can push xarray/open_mfdataset
        # into an unnecessary concat path during pipeline iteration.
        return path
