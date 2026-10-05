"""
Build the small test fixture used by test_global_closure.py.

GitHub's CI runners cannot see /g/data, so the closure tests run on the
globally integrated time series instead of the full gridded dataset. The global
heat budget only needs these integrals, and they are a few KB instead of ~1.3 GB.

Run this on NCI after re-extracting the emulator dataset, then commit the
updated test/data/global_heat_budget.npz:

    python test/make_global_budget_fixture.py [path/to/1deg_ocean_heat_emulator_data.nc]
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr

DEFAULT_DATAPATH = "/g/data/nm47/txs156/OM2-emulator/data/1deg_ocean_heat_emulator_data.nc"
OUTPUT = Path(__file__).parent / "data" / "global_heat_budget.npz"


def main(datapath):
    with xr.open_dataset(datapath) as ds:
        # Same ocean mask as build_normalisation and the extraction check:
        # land is NaN in the surface heat flux.
        ocean = ds["total_surface_heat_flx"].isel(time=0).notnull()
        cell_area = ds["area_t"]
        if "time" in cell_area.dims:
            cell_area = cell_area.isel(time=0)
        cell_area = cell_area.astype("float64").where(ocean, 0.0).fillna(0.0)

        # float64 throughout: global OHC is ~1e25 J and its monthly changes ~1e22 J.
        def global_integral(name):
            field = ds[name].astype("float64").fillna(0.0)
            return (field * cell_area).sum(["yt_ocean", "xt_ocean"]).values

        np.savez(
            OUTPUT,
            months=ds["time"].dt.strftime("%Y-%m").values.astype(str),
            days_in_month=ds["time"].dt.days_in_month.values.astype(np.int64),
            global_ohc_J=global_integral("ocean_heat_content_2d"),
            global_flux_W=global_integral("total_surface_heat_flx"),
            ohc_description=np.array(ds["ocean_heat_content_2d"].attrs.get("description", "")),
            source=np.array(str(datapath)),
        )
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATAPATH)
