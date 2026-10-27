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


def surface_freshwater_flux(pme_net, runoff, sfc_salt_flux_ice, s_ref=35.0):
    """
    Surface freshwater flux (kg m^-2 s^-1, > 0 into the ocean) consistent with
    the freshwater content sum_z(rho_dzt * (S_ref - S) / S_ref), S in g/kg.

    Freshwater content = mass - (salt in g) / S_ref, so its tendency is the
    mass flux (pme_net + runoff; pme_net already holds the restoring and the
    sea-ice melt mass) minus the salt flux in g m^-2 s^-1 over S_ref. MOM5's
    sfc_salt_flux_ice is in kg of salt, hence the factor 1000. (v1 and v2 of
    the dataset divided by S_ref only, making the sea-ice term 1000 times too
    small: the 1970-2018 global freshwater budget then mis-closed by
    -1.13e15 kg, 22% of the content change, against -2.5e14 kg with this flux.)
    """
    return pme_net + runoff - 1000.0 * sfc_salt_flux_ice / s_ref
