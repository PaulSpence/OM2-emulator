"""
The coupler: the surface fluxes the emulated ocean feels, computed the way ACCESS-OM2's own coupler computes
them, from the JRA55-do atmosphere and the emulator's OWN ocean surface.

Why: in ACCESS-OM2 the fluxes depend on the ocean surface. A warm SST loses more heat to the air, so it cools
back down (the air-sea feedback). An emulator forced with ACCESS-OM2's fluxes never feels that: if its SST
drifts warm, nothing pulls it back. With the coupler in the loop it does.

What ACCESS-OM2 does (1deg_jra55_iaf_omip2_cycle6; cice5 9864af2, libaccessom2 44e8821, MOM5 af3a94d):
  - libaccessom2 (YATM) reads the 3-hourly JRA55-do v1.4.0 fields, remaps them to the ocean/ice grid with the
    SCRIP weights in common_1deg_jra55 (patch weights for tas, huss, uas, vas, psl), and holds each record
    for 3 hours (no time interpolation).
  - CICE computes the open-ocean fluxes (drivers/auscom/cpl_forcing_handler.F90 gfdl_ocean_fluxes ->
    GFDL surface_flux_1d, ncar_ocean_fluxes "corrected" branch, Large & Yeager): sensible heat, evaporation
    and its latent heat, upward longwave (sigma T^4), and the wind stress, from the wind RELATIVE to the
    surface current (+ 1 m/s gust), tas and huss used as 10 m values, and the sea-level pressure. Winds are
    used as given (east/north), not rotated onto the grid. These enter the ocean weighted by the open-water
    fraction 1 - aice.
  - Everything else the ocean receives does not depend on the ocean surface: downward shortwave (fixed
    albedo) and longwave, rain, snow, runoff, the sea-ice terms (melt water, salt, heat, shortwave through
    the ice, ice-ocean stress), frazil.

How the emulator's coupler does the same, per month:
    flux = ACCESS-OM2's flux of the month  -  B(ACCESS-OM2's surface)  +  B(the emulator's surface)
where B is the open-ocean part above (the only part that depends on the ocean), computed exactly as CICE
does from every 3-hourly JRA55-do record of the month and averaged. So the coupler gives ACCESS-OM2's own
fluxes when the emulator's surface equals ACCESS-OM2's, and responds to any difference exactly as ACCESS-OM2's
bulk formulae do. Offline check (2010, MOM's daily SST and currents): this port reproduces MOM's recorded
open-ocean sensible, latent, longwave and evaporation fluxes with R^2 = 1.0000.

The surface B sees: SST and the surface currents u_surf, v_surf (all three predicted by the emulator), and the
month's sea-ice concentration (prescribed: CICE monthly aice_m of the same run).

Lagged surface: to predict month t+1 the emulator needs the fluxes of month t+1, which depend on the surface of
month t+1, which it has not predicted yet. So B uses the surface ANOMALY (z-score) of month t laid on the
climatology of month t+1, for the emulator and for ACCESS-OM2 alike: the two lags cancel when the emulator is
right, and the flux responds to the emulator's error the same way either way.

Inputs are read straight from where they live (nothing is copied): JRA55-do from /g/data/qv56, the remapping
weights and the CICE output from /g/data/ik11. They are read once, for the months a run needs, and kept on the
GPU if they fit, otherwise in host memory (half precision, stored relative to an offset; ~170 MB per month).

    inputs  = read_coupler_inputs(months_YYYY_MM, time_indices, mask, device, weather_device)
    coupler = PhysicalCoupler(inputs, ...)
    forcing_z, flux_totals = coupler(latest_state, months_to_predict)
"""

import glob
import multiprocessing
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as recompute_in_backward

# ---------------------------------------------------------------------------------------------------------
# Where the inputs live
# ---------------------------------------------------------------------------------------------------------
JRA55_DIR = "/g/data/qv56/replicas/input4MIPs/CMIP6/OMIP/MRI/MRI-JRA55-do-1-4-0/atmos/3hrPt"
REMAP_WEIGHTS = "/g/data/ik11/inputs/access-om2/input_20201102/common_1deg_jra55/rmp_jra55_cice_patch.nc"
ACCESS_OM2_RUN = "/g/data/ik11/outputs/access-om2/1deg_jra55_iaf_omip2_cycle6"

# The JRA55-do fields B needs, and how each is stored (half precision): (value - offset) * factor.
ATMOSPHERE = (
    ("tas", 273.15, 1.0),       # air temperature (K) -> degC
    ("huss", 0.0, 1000.0),      # specific humidity (kg/kg) -> g/kg
    ("uas", 0.0, 1.0),          # eastward wind (m/s)
    ("vas", 0.0, 1.0),          # northward wind (m/s)
    ("psl", 101325.0, 0.01),    # sea-level pressure (Pa) -> hPa from 1013.25
)
MAX_RECORDS = 31 * 8            # 3-hourly records in the longest month

# ---------------------------------------------------------------------------------------------------------
# Constants, as in cice5 drivers/auscom/ice_constants.F90 and GFDL surface_flux_mod
# ---------------------------------------------------------------------------------------------------------
RDGAS, RVGAS, CP_AIR, GRAV, VONKAR = 287.04, 461.50, 1005.0, 9.80616, 0.4
LVAP = 2.501e6                  # latent heat of vaporisation in CICE (J/kg)
HLV_MOM = 2.500e6               # MOM's: MOM's evaporation (kg/m2/s) = CICE's latent heat / HLV_MOM
SIGMA = 567.0e-10               # Stefan-Boltzmann (W/m2/K4)
TFFRESH = 273.15
D622 = RDGAS / RVGAS
D378 = 1.0 - D622
D608 = D378 / D622
KAPPA = RDGAS / CP_AIR
GUST = 1.0                      # m/s, added to the relative wind speed
Z_REF = 10.0                    # m, height of the atmospheric values


# =========================================================================================================
# Reading the inputs (CPU, once per run)
# =========================================================================================================

def _remap_matrix(ocean_cells):
    """The SCRIP patch weights as a sparse matrix (ocean cells x JRA55 cells)."""
    import scipy.sparse as sp
    import xarray as xr
    w = xr.open_dataset(REMAP_WEIGHTS)
    matrix = sp.csr_matrix(
        (w.remap_matrix.values[:, 0], (w.dst_address.values - 1, w.src_address.values - 1)),
        shape=(w.sizes["dst_grid_size"], w.sizes["src_grid_size"]),
    )
    w.close()
    return matrix[ocean_cells]


def _jra55_file(field, year):
    files = glob.glob(f"{JRA55_DIR}/{field}/gr/*/{field}_*_gr_{year}01010000-*.nc")
    if len(files) != 1:
        raise FileNotFoundError(f"expected one JRA55-do {field} file for {year}, found {files}")
    return files[0]


_WORKER = {}


def _start_worker(ocean_cells):
    _WORKER["remap"] = _remap_matrix(ocean_cells)


def _read_year(job):
    """One year's wanted months: {"YYYY-MM": ((5, MAX_RECORDS, cells) float16, number of records)}."""
    import cftime
    import netCDF4
    year, wanted = job
    remap = _WORKER["remap"]
    out = {m: np.zeros((len(ATMOSPHERE), MAX_RECORDS, remap.shape[0]), dtype=np.float16) for m in wanted}
    n_records = {}
    for k, (field, offset, factor) in enumerate(ATMOSPHERE):
        with netCDF4.Dataset(_jra55_file(field, year)) as ds:
            t = ds["time"]
            dates = cftime.num2date(t[:], t.units, t.calendar)
            record_month = np.array([f"{d.year:04d}-{d.month:02d}" for d in dates])
            var = ds[field]
            var.set_auto_mask(False)
            for m in wanted:
                records = np.flatnonzero(record_month == m)
                assert len(records) and np.all(np.diff(records) == 1), (field, m)
                x = var[records[0] : records[-1] + 1].astype(np.float64).reshape(len(records), -1)
                remapped = np.asarray(remap @ x.T).T                    # (records, ocean cells)
                out[m][k, : len(records)] = ((remapped - offset) * factor).astype(np.float16)
                n_records[m] = len(records)
    return {m: (out[m], n_records[m]) for m in wanted}


def read_coupler_inputs(months, time_indices, mask, device, weather_device, processes=12):
    """
    Everything the coupler reads from outside the emulator, for the forcing months `months`.

    months       : ["YYYY-MM", ...] the forcing months to read (no repeats)
    time_indices : their positions on the emulator's time axis
    mask         : (H, W) 1 = ocean
    device       : where to keep the small inputs (the GPU)
    weather_device : where to keep the weather, ~170 MB a month: the GPU if it fits there (an 80-141 GB A100 or
                   H200), otherwise "cpu" (a 32 GB V100), and the coupler copies each batch's months to the GPU
    returns {
        "time_index": (M,) long, the months' positions on the emulator's time axis
        "atmosphere": (M, 5, MAX_RECORDS, cells) float16 on `weather_device`, every 3-hourly record of tas, huss,
                      uas, vas, psl remapped to the ocean cells as libaccessom2 does, stored as
                      (x - offset) * factor (see ATMOSPHERE); records past the end of a month are 0
        "n_records" : (M,) long on `device`, 3-hourly records in each month
        "open_water": (M, cells) float32 on `device`, 1 - CICE's monthly-mean sea-ice concentration
    }
    """
    import xarray as xr
    months = list(months)
    assert len(set(months)) == len(months), "repeated months"
    ocean_cells = np.flatnonzero(np.asarray(mask).ravel() > 0)
    slot = {m: k for k, m in enumerate(months)}

    atmosphere = torch.zeros((len(months), len(ATMOSPHERE), MAX_RECORDS, len(ocean_cells)), dtype=torch.float16, device=weather_device)
    n_records = torch.zeros(len(months), dtype=torch.long)
    by_year = {}
    for m in months:
        by_year.setdefault(int(m[:4]), []).append(m)
    # Processes, not threads (HDF5 reads hold the GIL). Forked: the workers never touch the GPU.
    with multiprocessing.get_context("fork").Pool(processes, initializer=_start_worker, initargs=(ocean_cells,)) as pool:
        for done in pool.imap_unordered(_read_year, sorted(by_year.items())):
            for m, (values, n) in done.items():
                atmosphere[slot[m]] = torch.from_numpy(values).to(weather_device)
                n_records[slot[m]] = n

    # The month's sea-ice concentration, CICE monthly mean (iceh.YYYY-MM.nc of the run).
    ice_files = {Path(f).name[5:12]: f for f in glob.glob(f"{ACCESS_OM2_RUN}/output*/ice/OUTPUT/iceh.????-??.nc")}
    open_water = np.zeros((len(months), len(ocean_cells)), dtype=np.float32)
    for m in months:
        with xr.open_dataset(ice_files[m], decode_times=False) as ice:
            aice = np.nan_to_num(ice["aice_m"].values[0]).ravel()[ocean_cells]
        open_water[slot[m]] = 1.0 - np.clip(aice, 0.0, 1.0)

    return {
        "time_index": torch.as_tensor(time_indices, dtype=torch.long),
        "atmosphere": atmosphere,
        "n_records": n_records.to(device),
        "open_water": torch.from_numpy(open_water).to(device),
    }


# =========================================================================================================
# The open-ocean bulk formulae (GPU, differentiable)
# =========================================================================================================

def saturation_vapour_pressure(t):
    """GFDL compute_es_k (Goff-Gratch; water above 0 C, ice below -20 C, linear blend between), Pa; t in K.
    (The model looks it up in a 0.1 K table of this function.)"""
    t_ice, t_boil = TFFRESH, TFFRESH + 100.0
    log_ice = (-9.09718 * (t_ice / t - 1.0) - 3.56654 * torch.log10(t_ice / t)
               + 0.876793 * (1.0 - t / t_ice) + np.log10(610.71))
    log_water = (-7.90298 * (t_boil / t - 1) + 5.02808 * torch.log10(t_boil / t)
                 - 1.3816e-07 * (10 ** ((1 - t / t_boil) * 11.344) - 1)
                 + 8.1328e-03 * (10 ** ((t_boil / t - 1) * (-3.49149)) - 1) + np.log10(101324.60))
    es_ice, es_water = 10.0 ** log_ice, 10.0 ** log_water
    blend = 0.05 * ((t_ice - t) * es_ice + (t - t_ice + 20.0) * es_water)
    return torch.where(t <= t_ice - 20.0, es_ice, torch.where(t >= t_ice, es_water, blend))


def open_ocean_fluxes(tas, huss, uas, vas, psl, sst, u_ocean, v_ocean):
    """
    CICE's gfdl_ocean_fluxes -> surface_flux_1d (ncar_ocean_fluxes, corrected branch), for open water.

    tas (K), huss (kg/kg), uas, vas (m/s, east/north), psl (Pa), sst (K, as MOM writes it), u_ocean, v_ocean (m/s): tensors of
    broadcastable shapes, (B, records, cells) here. Returns (B, 4, records, cells), all positive INTO the ocean:
        heat        W/m2      -(sensible + latent + sigma SST^4)
        freshwater  kg/m2/s   -evaporation, as MOM receives it
        tau_x, tau_y  N/m2    stress on the ocean
    """
    # Air density and pressure at 10 m (cpl_forcing_handler.F90); the surface pressure is psl.
    tv0 = tas * (1.0 + D608 * huss)
    p_atm = psl - psl / (RDGAS * tv0) * GRAV * Z_REF
    t_surf = sst                                         # MOM's sst is in K
    e_sat = saturation_vapour_pressure(t_surf)
    q_sat = D622 * e_sat / (psl - D378 * e_sat)          # use_mixing_ratio = F, no 0.98 factor
    th_atm = tas * (psl / p_atm) ** KAPPA                # potential temperature
    u_rel = u_ocean - uas                                # relative wind (absolute_wind = F)
    v_rel = v_ocean - vas
    w_atm = torch.sqrt(u_rel ** 2 + v_rel ** 2 + GUST ** 2)

    # ncar_ocean_fluxes(u_del=w_atm, t=th_atm, ts=t_surf, q=huss, qs=q_sat, z=10), corrected branch
    t, ts, q, qs = th_atm, t_surf, huss, q_sat
    tv = t * (1 + 0.608 * q)
    u = torch.clamp(w_atm, min=0.5)
    u10 = u
    cd_n10 = (2.7 / u10 + 0.142 + 0.0764 * u10) / 1e3
    cd_n10_rt = torch.sqrt(cd_n10)
    ce_n10 = 34.6 * cd_n10_rt / 1e3
    stable = (t >= ts).to(t.dtype)                       # 0.5 + sign(0.5, t - ts): sign(0) = +
    ch_n10 = (18.0 * stable + 32.7 * (1 - stable)) * cd_n10_rt / 1e3
    cd, ch, ce = cd_n10, ch_n10, ce_n10
    for _ in range(2):
        cd_rt = torch.sqrt(cd)
        ustar = cd_rt * u
        tstar = (ch / cd_rt) * (t - ts)
        qstar = (ce / cd_rt) * (q - qs)
        bstar = GRAV * (tstar / tv + qstar / (q + 1 / 0.608))
        zeta = VONKAR * bstar * Z_REF / (ustar * ustar)
        zeta = torch.copysign(torch.clamp(zeta.abs(), max=10.0), zeta)
        x2 = torch.clamp(torch.sqrt((1 - 16 * zeta).abs()), min=1.0)
        x = torch.sqrt(x2)
        is_stable = zeta > 0
        psi_m = torch.where(is_stable, -5 * zeta,
                            torch.log((1 + 2 * x + x2) * (1 + x2) / 8) - 2 * (torch.atan(x) - np.arctan(1.0)))
        psi_h = torch.where(is_stable, -5 * zeta, 2 * torch.log((1 + x2) / 2))
        u10 = u / (1 + cd_n10_rt * (np.log(Z_REF / 10) - psi_m) / VONKAR)
        cd_n10 = (2.7 / u10 + 0.142 + 0.0764 * u10) / 1e3
        cd_n10_rt = torch.sqrt(cd_n10)
        ce_n10 = 34.6 * cd_n10_rt / 1e3
        stable = (zeta >= 0).to(t.dtype)
        ch_n10 = (18.0 * stable + 32.7 * (1 - stable)) * cd_n10_rt / 1e3
        xx = (np.log(Z_REF / 10) - psi_m) / VONKAR
        cd = cd_n10 / (1 + cd_n10_rt * xx) ** 2
        xx = (np.log(Z_REF / 10) - psi_h) / VONKAR
        ch = ch_n10 / (1 + ch_n10 * xx / cd_n10_rt) * torch.sqrt(cd / cd_n10)
        ce = ce_n10 / (1 + ce_n10 * xx / cd_n10_rt) * torch.sqrt(cd / cd_n10)

    # Back in surface_flux_1d (rough_scale = 1)
    rho = p_atm / (RDGAS * tas * (1.0 + D608 * huss))
    sensible = CP_AIR * ch * w_atm * rho * (t_surf - th_atm)        # W/m2, upward
    evaporation = ce * w_atm * rho * (q_sat - huss)                 # kg/m2/s, upward
    heat = -(sensible + LVAP * evaporation + SIGMA * t_surf ** 4)
    freshwater = -evaporation * LVAP / HLV_MOM
    tau_x = -cd * w_atm * rho * u_rel
    tau_y = -cd * w_atm * rho * v_rel
    return torch.stack([heat, freshwater, tau_x, tau_y], dim=1)


def _records_sum(chunk, valid, sst, u_ocean, v_ocean):
    """open_ocean_fluxes over a chunk of 3-hourly records, summed over the records inside the month: (B, 4, cells).
    chunk (B, 5, records, cells) float16 as stored by read_coupler_inputs; valid (B, records) 1 inside the month."""
    tas, huss, uas, vas, psl = (chunk[:, k].float() / factor + offset for k, (_, offset, factor) in enumerate(ATMOSPHERE))
    fluxes = open_ocean_fluxes(tas, huss, uas, vas, psl, sst, u_ocean, v_ocean)   # (B, 4, records, cells)
    return (fluxes * valid[:, None, :, None]).sum(dim=2)


# The same, compiled for the GPU: torch.compile fuses the bulk formulae's many elementwise steps into a few kernels,
# so their intermediates never go through GPU memory (faster, and a fraction of the memory). On the CPU (the tests)
# it runs as written above.
_records_sum_compiled = torch.compile(_records_sum)


def monthly_open_ocean_fluxes(atmosphere, slots, n_records, open_water, surface, records_per_chunk=8):
    """
    The month-mean of open_ocean_fluxes over every 3-hourly record, times the open-water fraction.

    atmosphere : (M, 5, MAX_RECORDS, cells) float16, the weather of every month read, as stored by
                 read_coupler_inputs (on the GPU, or in host memory)
    slots      : (B,) which of those months each sample is in (samples may share a month)
    n_records  : (B,) records in each sample's month
    open_water : (B, cells)
    surface    : (B, 3, cells) SST (K), u_surf, v_surf (m/s), held over the month
    returns      (B, 4, cells) heat, freshwater, tau_x, tau_y into the ocean

    In chunks of records, each fetched from `atmosphere` (to the GPU) only when it is needed. With gradients nothing
    of a chunk is kept for the backward pass: it is fetched and computed again there. (Keeping it would hold every
    rollout month's weather on the GPU, ~170 MB per sample-month: more than any GPU has for a 60-month rollout.)
    """
    sst, u_ocean, v_ocean = (surface[:, k, None] for k in range(3))     # each (B, 1, cells)
    records_sum = _records_sum_compiled if surface.is_cuda else _records_sum
    # Each distinct month is fetched once (rollout twins and ensemble members share their month), then handed to
    # every sample in it.
    distinct, sample_month = slots.unique(return_inverse=True)
    distinct = distinct.to(atmosphere.device)

    def chunk_sum(first, sst, u_ocean, v_ocean):
        chunk = atmosphere[:, :, first : first + records_per_chunk][distinct].to(sst.device)[sample_month]
        record = torch.arange(first, first + chunk.shape[2], device=sst.device)
        valid = (record[None, :] < n_records[:, None]).float()                         # (B, records)
        return records_sum(chunk, valid, sst, u_ocean, v_ocean)

    total = 0.0
    with torch.autocast(device_type=surface.device.type, enabled=False):
        for first in range(0, int(n_records.max()), records_per_chunk):
            if torch.is_grad_enabled() and surface.requires_grad:
                total = total + recompute_in_backward(chunk_sum, first, sst, u_ocean, v_ocean, use_reentrant=False)
            else:
                total = total + chunk_sum(first, sst, u_ocean, v_ocean)
    return total / n_records[:, None, None] * open_water[:, None]


# =========================================================================================================
# The coupler
# =========================================================================================================

class PhysicalCoupler(nn.Module):
    """
    coupler(latest_state, months) -> (forcing_z, flux_totals)

        latest_state : (B, P, H, W) the emulator's latest state (z-scores), month t
        months       : (B,) time index of the month to compute the fluxes for, t+1 (one of the months read
                       by read_coupler_inputs)
        forcing_z    : (B, 4, H, W) those fluxes as the emulator's forcing input (z-scores)
        flux_totals  : (B, 4) float64 their ocean integrals (e.g. W), for the budget constraint

    Gradients flow through it (from the fluxes back to the SST and currents that made them), so in training the
    emulator learns how its own surface changes the fluxes it will feel next month.
    """

    def __init__(self, inputs, true_state_z, true_fluxes, surface_channels, state_mean, state_std,
                 forcing_mean, forcing_scale, mask, area):
        """
        inputs           : from read_coupler_inputs (the weather on the GPU or in host memory, the rest on the GPU)
        true_state_z     : (T, P, H, W) ACCESS-OM2's states (z-scores), the whole time axis
        true_fluxes      : (T, 4, H, W) ACCESS-OM2's fluxes in physical units, in the emulator's forcing order:
                           heat (W/m2), freshwater (kg/m2/s), tau_x, tau_y (N/m2)
        surface_channels : positions of sst, u_surf, v_surf among the state variables
        state_mean,
        state_std        : (T, P, H, W) climatological mean and std of the state (to undo the z-scores)
        forcing_mean     : (T, 4, H, W) climatological mean of the fluxes, physical units
        forcing_scale    : (4,) the one scale of each flux (forcing z = (flux - mean) / scale)
        mask, area       : (H, W) ocean mask, cell area (m^2)
        """
        super().__init__()
        self.surface_channels = list(surface_channels)
        self.grid_shape = tuple(mask.shape)
        device = inputs["open_water"].device
        ocean = torch.nonzero(torch.as_tensor(mask).flatten() > 0).squeeze(1)
        at_ocean = lambda field: torch.as_tensor(field).flatten(-2)[..., ocean.cpu()].float()

        slot = torch.full((true_state_z.shape[0],), -1, dtype=torch.long)
        slot[inputs["time_index"]] = torch.arange(len(inputs["time_index"]))
        buffers = {
            "ocean_index": ocean,
            "slot": slot,                                                  # time index -> position in the inputs
            "n_records": inputs["n_records"],                              # (M,)
            "open_water": inputs["open_water"],                            # (M, cells)
            "surface_mean": at_ocean(state_mean[:, self.surface_channels]),   # (T, 3, cells)
            "surface_std": at_ocean(state_std[:, self.surface_channels]),
            "forcing_mean": at_ocean(forcing_mean),                        # (T, 4, cells)
            "forcing_scale": torch.as_tensor(forcing_scale, dtype=torch.float32),
            "cell_area": at_ocean(area).double(),                          # (cells,)
            "mask": torch.as_tensor(mask).float(),
        }
        # persistent=False: rebuilt from the data, never written into checkpoints.
        for name, value in buffers.items():
            self.register_buffer(name, value.to(device), persistent=False)
        # The weather, (M, 5, MAX_RECORDS, cells) float16: a plain attribute, not a buffer, so it stays where
        # read_coupler_inputs put it when the coupler is moved, and is never written into checkpoints.
        self.atmosphere = inputs["atmosphere"]

        # The rest of every month's flux: ACCESS-OM2's flux minus B on ACCESS-OM2's own (lagged) surface.
        rest = []
        with torch.no_grad():
            for k, t in enumerate(inputs["time_index"].tolist()):
                z_true = at_ocean(true_state_z[t - 1, self.surface_channels])[None].to(device)
                b_true = self._open_ocean(z_true, torch.tensor([t], device=device), torch.tensor([k], device=device))
                rest.append(at_ocean(true_fluxes[t])[None].to(device) - b_true)
        self.register_buffer("rest", torch.cat(rest), persistent=False)    # (M, 4, cells)

    def _open_ocean(self, z_surface, months, slots):
        """B: the open-ocean fluxes (B, 4, cells) of `months` from the surface z-scores (B, 3, cells) of the month
        before, laid on the climatology of `months`."""
        surface = z_surface * self.surface_std[months] + self.surface_mean[months]
        return monthly_open_ocean_fluxes(self.atmosphere, slots, self.n_records[slots], self.open_water[slots], surface)

    def fluxes(self, latest_state, months):
        """(B, 4, cells) physical fluxes of `months` from the emulator's latest state (the month before)."""
        slots = self.slot[months]
        if (slots < 0).any():
            raise KeyError(f"the coupler has no inputs for time indices {months[slots < 0].tolist()}")
        z_surface = latest_state[:, self.surface_channels].flatten(2)[:, :, self.ocean_index].float()
        return self.rest[slots] + self._open_ocean(z_surface, months, slots)

    def forcing_z(self, fluxes, months):
        """(B, 4, H, W) the emulator's forcing input: (flux - climatology) / scale, 0 on land."""
        z = (fluxes - self.forcing_mean[months]) / self.forcing_scale[None, :, None]
        full = z.new_zeros(z.shape[0], z.shape[1], self.grid_shape[0] * self.grid_shape[1])
        full[:, :, self.ocean_index] = z
        return full.view(z.shape[0], z.shape[1], *self.grid_shape)

    def ocean_totals(self, fluxes):
        """(B, 4) float64 ocean integral of each flux (flux x cell area, summed), e.g. W for the heat flux."""
        return (fluxes.double() * self.cell_area).sum(dim=-1)

    def forward(self, latest_state, months):
        fluxes = self.fluxes(latest_state, months)
        return self.forcing_z(fluxes, months), self.ocean_totals(fluxes)


@torch.no_grad()
def coupled_rollout(model, coupler, prior, target_indices, initial_index, constraint, initial_flux_totals, members=1):
    """
    Free-running rollout like autoregressive_rollout, but every month's forcing comes from the coupler,
    computed from the emulator's own previous month.

    prior               (n_prior, P, H, W) the true initial states, oldest first
    target_indices      (N,)  time index of each predicted month (sets the atmosphere and sea ice used)
    initial_index       int   time index of the last prior state
    constraint          the GlobalBudgetConstraint, closed with the coupled fluxes
    initial_flux_totals (C,)  ACCESS-OM2's ocean-total fluxes of the starting month (the budget of the
                              first step averages them with the first coupled month's)
    returns  predictions (members, N, P, H, W) z-scores, and the ocean-total coupled fluxes
             (members, N, C) float64, both on the CPU
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    coupler, constraint = coupler.to(device), constraint.to(device)
    n_prognostic = prior.shape[1]
    state = prior.flatten(0, 1)[None].to(device).repeat(members, 1, 1, 1)
    previous = torch.full((members,), int(initial_index), device=device)
    totals_previous = torch.as_tensor(initial_flux_totals, device=device).double()[None].repeat(members, 1)
    predictions, totals = [], []
    for target in target_indices:
        target = torch.full_like(previous, int(target))
        latest = state[:, -n_prognostic:]
        forcing, totals_target = coupler(latest, target)                     # fluxes from the emulator's own surface
        pred = model(state, forcing, coupler.mask)
        pred = constraint(pred, latest, previous, target, totals_previous, totals_target)
        predictions.append(pred.float().cpu())
        totals.append(totals_target.cpu())
        state = torch.cat([state[:, n_prognostic:], pred.to(state.dtype)], dim=1)
        previous, totals_previous = target, totals_target
    model.train(was_training)
    return torch.stack(predictions, dim=1), torch.stack(totals, dim=1)
