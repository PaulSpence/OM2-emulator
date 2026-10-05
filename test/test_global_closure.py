"""
Global heat-budget closure tests.

Globally, ocean-interior heat transports cancel, so the change in area-integrated
ocean heat content (OHC) must match the time-integrated surface heat flux. These
tests check that on the real ACCESS-OM2 data, and check that the training loss
(src/Emulator/om2_loss_functions.py: global_closure_loss) computes it correctly.

Real-data tests use test/data/global_heat_budget.npz: the globally integrated OHC
(J) and surface heat flux (W) for every month of the emulator dataset. It is
built from the full NetCDF by test/make_global_budget_fixture.py; regenerate and
commit it whenever the dataset is re-extracted.

Monthly means: the difference between two consecutive monthly-mean OHCs is the
heat added between the two month centres, so each step is paired with the
average flux of the two months, (F[t] + F[t+1]) / 2.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

FIXTURE = Path(__file__).parent / "data" / "global_heat_budget.npz"

# --- Tolerances ---------------------------------------------------------------
# Full fields: with degC OHC and the two-month flux average the budget closes to
# ~7% of the monthly global OHC change (the rest comes from using monthly means).
FULL_FIELD_MAX_RESIDUAL_FRACTION = 0.10
FULL_FIELD_MIN_CORRELATION = 0.99
# Long-term drift: 1e19 J/month is ~0.01 W/m^2 over the global ocean.
FULL_FIELD_MAX_DRIFT_J_PER_MONTH = 1.0e19
# Anomalies, as seen by the training loss: the known-closure loss on the TRUE
# data must be small, otherwise the closure term pushes the model away from the
# truth. 0.25 is the notebook's known-closure warning threshold.
ANOMALY_MAX_KNOWN_CLOSURE_LOSS = 0.25

# --- Settings mirroring Ocean_Emulator_UNet.ipynb -----------------------------
TIME_START, TRAIN_END, TIME_END = "1970-01", "2004-02", "2018-12"
N_TRAIN_SAMPLES = 408          # initial months 1970-01 ... 2003-12 (dates 1970-02 ... 2004-01)
N_ROLLOUT_STEPS = 12
BATCH_SIZE = 32
SECONDS_PER_MONTH = 30 * 24 * 60 * 60
CLOSURE_MIN_SCALE = 1.0e20


@pytest.fixture(scope="module")
def budget():
    data = np.load(FIXTURE)
    return {name: data[name] for name in data.files}


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


# =============================================================================
# Real data
# =============================================================================

def test_fixture_ohc_is_celsius_referenced(budget):
    """OHC must be built from temp - 273.15; Kelvin OHC cannot close (see Extract_om2_data.ipynb)."""
    assert "273.15" in str(budget["ohc_description"]), (
        "The OHC in the fixture was not computed from Celsius temperature: "
        f"{budget['ohc_description']!r}. Re-extract the dataset and regenerate the fixture."
    )


def test_full_field_global_budget_closes(budget):
    """Global OHC change between monthly means matches the two-month-average surface flux."""
    ohc, flux = budget["global_ohc_J"], budget["global_flux_W"]
    seconds = budget["days_in_month"] * 86400.0

    d_ohc = np.diff(ohc)
    # Heat added between month-t and month-(t+1) centres: second half of month t
    # plus first half of month t+1, using the real month lengths.
    flux_integral = 0.5 * (flux[:-1] * seconds[:-1] + flux[1:] * seconds[1:])
    residual = d_ohc - flux_integral

    fraction = rms(residual) / rms(d_ohc)
    correlation = np.corrcoef(d_ohc, flux_integral)[0, 1]
    drift = residual.mean()
    print(f"full-field residual {fraction:.1%} of dOHC | corr {correlation:.4f} | drift {drift:.2e} J/month")

    assert fraction < FULL_FIELD_MAX_RESIDUAL_FRACTION, f"residual is {fraction:.1%} of the monthly OHC change"
    assert correlation > FULL_FIELD_MIN_CORRELATION, f"dOHC vs flux correlation only {correlation:.4f}"
    assert abs(drift) < FULL_FIELD_MAX_DRIFT_J_PER_MONTH, f"long-term drift {drift:.2e} J/month"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known limitation: with monthly-mean data the anomaly budget leaves a residual "
        "comparable to the anomaly signal (known-closure loss ~0.5). strict=True makes "
        "this test FAIL once it starts passing, so the marker gets removed when the "
        "closure loss is fixed."
    ),
)
def test_anomaly_known_closure_loss_is_small(budget, loss_functions):
    """
    Run the training closure loss on the TRUE anomalies, as check_known_global_closure does.

    This builds the same quantities the notebook feeds the loss, but for the
    global integral directly: a 1x1 "grid" whose single cell holds the global
    anomaly, with area = 1 and std = 1 (so z-score * std = the anomaly in J or W).
    """
    months = budget["months"]
    keep = (months >= TIME_START) & (months <= TIME_END)
    months = months[keep]
    ohc, flux = budget["global_ohc_J"][keep], budget["global_flux_W"][keep]

    # Anomalies from the monthly climatology over the training period, as in
    # build_normalisation's Spatial_climatology (the global integral of the
    # gridded anomaly equals the anomaly of the global integral).
    calendar_month = np.array([int(m[5:7]) for m in months])
    train = months <= TRAIN_END
    ohc_anom, flux_anom = ohc.copy(), flux.copy()
    for m in range(1, 13):
        sel = calendar_month == m
        ohc_anom[sel] -= ohc[train & sel].mean()
        flux_anom[sel] -= flux[train & sel].mean()

    n_time = len(months)
    as_grid = lambda x: torch.tensor(x, dtype=torch.float64).reshape(*x.shape, 1, 1)
    ohc_z, flux_z = as_grid(ohc_anom), as_grid(flux_anom)
    ones_std = torch.ones(n_time, 1, 1, dtype=torch.float64)
    area = torch.ones(1, 1, dtype=torch.float64)
    mask = torch.ones(1, 1, dtype=torch.float64)

    closure = loss_functions.global_closure_loss(weight=1.0, closure_min_scale=CLOSURE_MIN_SCALE)

    initial = torch.arange(N_TRAIN_SAMPLES)          # initial-month index of each training window
    losses = []
    for start in range(0, N_TRAIN_SAMPLES, BATCH_SIZE):
        i0 = initial[start : start + BATCH_SIZE]
        targets = i0[:, None] + torch.arange(1, N_ROLLOUT_STEPS + 1)[None]
        forcing = flux_z[targets]                     # (B, n_steps, 1, 1)
        for step in range(N_ROLLOUT_STEPS):
            losses.append(
                closure(
                    initial_ohc_norm=ohc_z[i0][:, None],
                    pred_t=ohc_z[targets[:, step]][:, None],   # the truth
                    forcing_history=forcing[:, : step + 1],
                    initial_forcing=flux_z[i0],
                    initial_time_index=i0,
                    target_time_index=targets[:, step],
                    forcing_time_indices=targets[:, : step + 1],
                    area=area,
                    mask=mask,
                    ohc_std=ones_std,
                    forcing_std=ones_std,
                    dt_seconds=SECONDS_PER_MONTH,
                    rollout_step=step,
                ).item()
            )
    losses = np.array(losses)
    print(f"known-closure loss on true anomalies | mean {losses.mean():.3f}, max {losses.max():.3f}")

    assert losses.mean() < ANOMALY_MAX_KNOWN_CLOSURE_LOSS, (
        f"On the true data the anomaly closure loss averages {losses.mean():.3f} "
        f"(max {losses.max():.3f}), above {ANOMALY_MAX_KNOWN_CLOSURE_LOSS}. The monthly-mean "
        "anomaly budget does not close tightly enough for this term to be minimised at the truth."
    )


# =============================================================================
# Loss implementation on synthetic data
# =============================================================================

def _exactly_closing_batch(n_steps=12, batch=3, channels=3, h=6, w=8, n_time=40, seed=0):
    """Fields whose anomaly budget closes exactly under the two-month flux average."""
    g = torch.Generator().manual_seed(seed)
    dt = 2.6e6
    area = torch.rand(h, w, generator=g, dtype=torch.float64) * 1e10
    mask = (torch.rand(h, w, generator=g, dtype=torch.float64) > 0.2).double()
    ohc_std = torch.rand(n_time, h, w, generator=g, dtype=torch.float64) * 1e9 + 1e8
    flux_std = torch.rand(n_time, h, w, generator=g, dtype=torch.float64) * 20 + 5

    i0 = torch.tensor([3, 10, 20])[:batch]
    targets = i0[:, None] + torch.arange(1, n_steps + 1)[None]
    months = torch.cat([i0[:, None], targets], dim=1)

    flux = torch.randn(batch, n_steps + 1, h, w, generator=g, dtype=torch.float64) * 30
    ohc = torch.zeros(batch, n_steps + 1, h, w, dtype=torch.float64)
    ohc[:, 0] = torch.randn(batch, h, w, generator=g, dtype=torch.float64) * 1e9
    for j in range(n_steps):
        ohc[:, j + 1] = ohc[:, j] + 0.5 * (flux[:, j] + flux[:, j + 1]) * dt

    flux_z = flux / flux_std[months]
    ohc_z = ohc / ohc_std[months]
    # Channel 0 is heat flux; channels 1.. stand in for wind stress.
    forcing = torch.randn(batch, n_steps, channels, h, w, generator=g, dtype=torch.float64)
    forcing[:, :, 0] = flux_z[:, 1:]
    initial_forcing = torch.randn(batch, channels, h, w, generator=g, dtype=torch.float64)
    initial_forcing[:, 0] = flux_z[:, 0]

    def context(step, forcing=forcing, initial_forcing=initial_forcing):
        return dict(
            initial_ohc_norm=ohc_z[:, 0:1],
            pred_t=ohc_z[:, step + 1 : step + 2],
            forcing_history=forcing[:, : step + 1],
            initial_forcing=initial_forcing,
            initial_time_index=i0,
            target_time_index=targets[:, step],
            forcing_time_indices=targets[:, : step + 1],
            area=area,
            mask=mask,
            ohc_std=ohc_std,
            forcing_std=flux_std,
            dt_seconds=dt,
            rollout_step=step,
        )

    return context, forcing, initial_forcing


def test_closure_loss_is_zero_when_budget_closes(loss_functions):
    context, _, _ = _exactly_closing_batch()
    closure = loss_functions.global_closure_loss(weight=1.0, closure_min_scale=1.0)
    for step in range(12):
        assert closure(**context(step)).item() < 1e-20, f"non-zero loss at lead {step + 1}"


def test_closure_loss_uses_only_heat_flux_channel(loss_functions):
    context, forcing, initial_forcing = _exactly_closing_batch()
    closure = loss_functions.global_closure_loss(weight=1.0, closure_min_scale=1.0, heat_flux_channel_index=0)
    junk_forcing, junk_initial = forcing.clone(), initial_forcing.clone()
    junk_forcing[:, :, 1:] = 1e3
    junk_initial[:, 1:] = 1e3
    for step in range(12):
        clean = closure(**context(step))
        junk = closure(**context(step, forcing=junk_forcing, initial_forcing=junk_initial))
        assert torch.allclose(clean, junk)


def test_closure_loss_detects_wrong_initial_flux(loss_functions):
    context, _, initial_forcing = _exactly_closing_batch()
    closure = loss_functions.global_closure_loss(weight=1.0, closure_min_scale=1.0)
    wrong = initial_forcing.clone()
    wrong[:, 0] = 0.0
    assert closure(**context(0, initial_forcing=wrong)).item() > 1e-6


def test_closure_loss_requires_initial_forcing(loss_functions):
    context, _, _ = _exactly_closing_batch()
    closure = loss_functions.global_closure_loss(weight=1.0)
    ctx = context(0)
    del ctx["initial_forcing"]
    with pytest.raises(ValueError, match="initial_forcing"):
        closure(**ctx)
