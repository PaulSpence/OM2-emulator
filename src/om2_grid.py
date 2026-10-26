"""
Grid operations for ACCESS-OM2 (MOM5 B-grid) output, with no dependencies
beyond xarray, for use by the extraction notebook and the tests.
"""

import xarray as xr


def regrid_u_to_t(data_array, target_ds):
    """
    Average a u-cell (xu_ocean/yu_ocean) field onto the t-cell
    (xt_ocean/yt_ocean) grid of ``target_ds``.

    On MOM5's B-grid, u(i, j) is the north-east corner of t(i, j), so each
    t-cell has four surrounding u-corners: (i, j), (i-1, j), (i, j-1) and
    (i-1, j-1). Each t-cell takes the mean of the corners that are defined
    (ocean), with i-1 wrapping around in longitude; the southernmost row has
    only its two northern corners.

    This replaces bilinear interpolation (xarray .interp), which returned NaN
    at every t-cell with a land corner (a ring of coastal ocean cells: 5892 of
    the 69809 at 1 degree) and in the westernmost column (-279.5 lies outside
    the u-grid's -279.0 .. 80.0), so tau_x, tau_y and surface_speed were
    missing there.
    """
    ne = data_array
    nw = data_array.roll(xu_ocean=1, roll_coords=False)
    corners = xr.concat([ne, nw, ne.shift(yu_ocean=1), nw.shift(yu_ocean=1)], dim="corner")
    averaged = corners.mean("corner", skipna=True).rename({"xu_ocean": "xt_ocean", "yu_ocean": "yt_ocean"})
    return averaged.assign_coords(xt_ocean=target_ds["xt_ocean"].values, yt_ocean=target_ds["yt_ocean"].values)
