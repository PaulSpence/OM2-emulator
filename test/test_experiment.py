"""
Tests for the config-driven experiment package (src/Experiment).

Everything runs on a small synthetic NetCDF file shaped like the emulator
dataset, so the suite needs no /g/data access and no GPU.
"""

import importlib.util
import sys
import types

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd
import pytest
import torch
import xarray as xr

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "src"))

from Experiment import (  # noqa: E402
    ClosureConfig,
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    LossConfig,
    ModelConfig,
    NormalisationConfig,
    TimeConfig,
    TrainConfig,
    WindowConfig,
    build_data,
    build_losses,
    build_model,
    build_module,
    build_trainer,
    check_known_closure,
    compute_normalisation,
    plot_global_rmse_all_variables,
    plot_rmse_by_epoch,
    plot_skill_evaluation,
    run_control,
    run_skill_test,
)
from Experiment.training import PerVariableError, RolloutSchedule  # noqa: E402

PROGNOSTIC = ["ocean_heat_content_2d"]
FORCING = ["total_surface_heat_flx", "tau_x", "tau_y"]
H, W = 16, 20


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def synthetic_file(tmp_path_factory):
    """72 months (2000-01 .. 2005-12) on a 16 x 20 grid with land, like the real file."""
    rng = np.random.default_rng(0)
    n_time = 72
    times = pd.date_range("2000-01-01", periods=n_time, freq="MS") + pd.Timedelta(days=14)
    land = rng.random((H, W)) < 0.25
    month = times.month.values - 1
    seasonal = lambda amp: amp * np.sin(2 * np.pi * month / 12)[:, None, None]

    def field(mean, amp, noise):
        x = mean + seasonal(amp) + rng.normal(0, noise, (n_time, H, W))
        x[:, land] = np.nan
        return x.astype(np.float32)

    dims = ("time", "yt_ocean", "xt_ocean")
    ds = xr.Dataset(
        {
            "ocean_heat_content_2d": (dims, field(5e10, 2e9, 5e8), {"units": "J/m2", "description": "cp * ((temp - 273.15) * rho_dzt)"}),
            "total_surface_heat_flx": (dims, field(0.0, 80.0, 20.0), {"units": "W/m2"}),
            "ocean_freshwater_content_2d": (dims, field(-3e3, 50.0, 10.0), {"units": "kg m-2"}),
            "ocean_freshwater_flux": (dims, field(0.0, 2e-5, 5e-6), {"units": "kg m-2 s-1"}),
            "tau_x": (dims, field(0.0, 0.05, 0.02)),
            "tau_y": (dims, field(0.0, 0.03, 0.02)),
            "area_t": (dims, np.repeat(rng.uniform(1e9, 1e10, (1, H, W)), n_time, 0).astype(np.float32)),
        },
        coords={"time": times, "yt_ocean": np.linspace(-70, 80, H), "xt_ocean": np.linspace(-280, 79, W)},
    )
    path = tmp_path_factory.mktemp("data") / "synthetic_emulator.nc"
    ds.to_netcdf(path)
    return path


def make_cfg(path, tmp_path, **overrides):
    """A small, fast config on the synthetic file (overrides replace whole sections)."""
    sections = dict(
        data=DataConfig(
            path=str(path),
            prognostic=list(PROGNOSTIC),
            forcing=list(FORCING),
            normalisation=NormalisationConfig(fit_period=("2000-01", "2003-12")),
            cache_dir=str(tmp_path / "cache"),
        ),
        time=TimeConfig(
            time_axis=("2000-01", "2005-12"),
            train=("2000-02", "2003-06"),
            valid=("2003-07", "2003-12"),
            test=("2004-01", "2005-10"),
            control=("2002-01", "2002-12"),
            control_repeats=2,
        ),
        window=WindowConfig(n_prior=2, posterior_steps=4, rollout_steps=4),
        loss=LossConfig(terms={"local_mse": 1.0, "spectral": 0.0, "heat_closure": 0.1}),
        model=ModelConfig(padding_mode="zeros"),
        train=TrainConfig(batch_size=8, max_epochs=1, gpu_check=False),
        eval=EvalConfig(snapshot_periods=["2004-06"]),
    )
    sections.update(overrides)
    return ExperimentConfig(**sections)


@pytest.fixture(scope="module")
def data_and_cfg(synthetic_file, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("run")
    cfg = make_cfg(synthetic_file, tmp)
    return build_data(cfg, verbose=False), cfg


# =============================================================================
# Normalisation matches the original PET-based build_normalisation
# =============================================================================

def _load_original_build_normalisation():
    """src/Data/om2_normalisation_strategies.py with PyEarthTools replaced by a stub."""
    stub = types.ModuleType("pyearthtools.pipeline")
    stub.operations = types.SimpleNamespace(
        xarray=types.SimpleNamespace(
            normalisation=types.SimpleNamespace(
                Evaluated=lambda **kw: types.SimpleNamespace(_initialisation={"mean": kw["mean"], "deviation": kw["deviation"]})
            )
        )
    )
    saved = {k: sys.modules.get(k) for k in ("pyearthtools", "pyearthtools.pipeline")}
    sys.modules["pyearthtools"] = types.ModuleType("pyearthtools")
    sys.modules["pyearthtools.pipeline"] = stub
    try:
        path = REPO_ROOT / "src" / "Data" / "om2_normalisation_strategies.py"
        spec = importlib.util.spec_from_file_location("original_normalisation", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return module.build_normalisation


@pytest.mark.parametrize(
    "strategy",
    ["Spatial_climatology", "Spatial_climatology_global_variance", "Spatial_in_time", "Global_in_time"],
)
def test_normalisation_matches_original(synthetic_file, strategy):
    build_normalisation = _load_original_build_normalisation()
    variables = [*PROGNOSTIC, *FORCING]  # heat flux at index 1 = the original's land-mask variable
    with pytest.warns(UserWarning):
        mask_old, norm = build_normalisation(
            str(synthetic_file), strategy, variables,
            time_window=dict(start="2000-01", end="2005-12", freq="1MS"), train_end="2003-12", mask=True,
        )
    months = pd.date_range("2000-01", "2005-12", freq="MS")
    with xr.open_dataset(synthetic_file) as ds:
        mean, std, mask = compute_normalisation(
            ds.sel(time=months, method="nearest"), variables, strategy, ("2000-01", "2003-12"),
            "total_surface_heat_flx", "area_t",
        )
    np.testing.assert_array_equal(mask.values, mask_old.values)
    for v in variables:
        xr.testing.assert_allclose(mean[v], norm._initialisation["mean"][v])
        xr.testing.assert_allclose(std[v], norm._initialisation["deviation"][v])


# =============================================================================
# Data: fields, windows, cache
# =============================================================================

def test_windows_match_direct_normalisation(data_and_cfg, synthetic_file):
    data, cfg = data_and_cfg
    months = pd.date_range("2000-01", "2005-12", freq="MS")
    variables = [*PROGNOSTIC, *FORCING]
    with xr.open_dataset(synthetic_file) as raw:
        ds = raw.sel(time=months, method="nearest")
        mean, std, _ = compute_normalisation(ds, variables, "Spatial_climatology", ("2000-01", "2003-12"), "total_surface_heat_flx", "area_t")
        z = {
            v: ((ds[v].rename(yt_ocean="latitude", xt_ocean="longitude") - mean[v]) / std[v]).fillna(0.0).values
            for v in variables
        }

    sample = {k: v[0] for k, v in data.windows(data.train_indices[5:6]).items()}
    t0 = int(data.train_indices[5])
    assert data.months[t0] == "2000-07"  # train starts 2000-02, sample 5 -> 2000-07
    np.testing.assert_allclose(sample["prior"][:, 0].numpy(), z["ocean_heat_content_2d"][t0 - 1 : t0 + 1], rtol=1e-6)
    np.testing.assert_allclose(sample["target"][:, 0].numpy(), z["ocean_heat_content_2d"][t0 + 1 : t0 + 5], rtol=1e-6)
    for c, v in enumerate(FORCING):
        np.testing.assert_allclose(sample["forcing"][:, c].numpy(), z[v][t0 + 1 : t0 + 5], rtol=1e-6)
        np.testing.assert_allclose(sample["initial_forcing"][c].numpy(), z[v][t0], rtol=1e-6)
    assert sample["target_time_index"].tolist() == list(range(t0 + 1, t0 + 5))


def test_split_sizes_and_bounds(data_and_cfg):
    data, _ = data_and_cfg
    assert len(data.train_indices) == 41  # 2000-02 .. 2003-06
    assert len(data.valid_indices) == 6   # 2003-07 .. 2003-12


def test_cache_reused_and_invalidated(synthetic_file, tmp_path):
    cfg = make_cfg(synthetic_file, tmp_path)
    build_data(cfg, verbose=False)
    build_data(cfg, verbose=False)
    cached = sorted((tmp_path / "cache").glob("fields_*.pt"))
    assert len(cached) == 1, "second build should reuse the cache"

    heat_flux_only = make_cfg(
        synthetic_file, tmp_path,
        data=DataConfig(path=str(synthetic_file), forcing=["total_surface_heat_flx"],
                        normalisation=NormalisationConfig(fit_period=("2000-01", "2003-12")),
                        cache_dir=str(tmp_path / "cache")),
    )
    data = build_data(heat_flux_only, verbose=False)
    assert data.n_forcing == 1
    assert len(sorted((tmp_path / "cache").glob("fields_*.pt"))) == 2, "changed data settings need a new cache"


def test_split_outside_time_axis_raises(synthetic_file, tmp_path):
    cfg = make_cfg(synthetic_file, tmp_path, time=TimeConfig(
        time_axis=("2000-01", "2005-12"), train=("2000-01", "2003-06"), valid=("2003-07", "2003-12"),
        test=("2004-01", "2005-10"),
    ))
    with pytest.raises(ValueError, match="prior months"):
        build_data(cfg, verbose=False)


# =============================================================================
# Config validation
# =============================================================================

@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(window=WindowConfig(rollout_steps=13, posterior_steps=12)), "rollout_steps"),
        (dict(loss=LossConfig(terms={"local_mse": 0.0})), "at least one loss term"),
        (dict(loss=LossConfig(terms={"mae": 1.0})), "Unknown loss terms"),
        (dict(model=ModelConfig(output_head="resnet")), "Unknown output_head"),
        (dict(window=WindowConfig(rollout_schedule={5: 4})), "epoch 0"),
    ],
)
def test_config_rejects_inconsistent_settings(synthetic_file, tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        make_cfg(synthetic_file, tmp_path, **overrides)


def test_config_rejects_variable_in_both_roles(synthetic_file, tmp_path):
    with pytest.raises(ValueError, match="both prognostic and forcing"):
        make_cfg(synthetic_file, tmp_path, data=DataConfig(path=str(synthetic_file), prognostic=["tau_x"], forcing=["tau_x"]))


# =============================================================================
# Model and losses
# =============================================================================

@pytest.mark.parametrize("output_head", [None, "spatial_residual"])
@pytest.mark.parametrize("latent_processor", [None, "latent_residual"])
def test_model_options_forward_backward(data_and_cfg, output_head, latent_processor):
    data, cfg = data_and_cfg
    cfg.model.output_head, cfg.model.latent_processor = output_head, latent_processor
    try:
        model = build_model(cfg, data)
    finally:
        cfg.model.output_head, cfg.model.latent_processor = None, None
    prior = torch.randn(2, cfg.window.n_prior * data.n_prognostic, H, W, requires_grad=True)
    out = model(prior, torch.randn(2, data.n_forcing, H, W), data.fields["mask"])
    assert out.shape == (2, data.n_prognostic, H, W) and out.dtype == torch.float32
    out.pow(2).mean().backward()
    assert torch.isfinite(prior.grad).all()


TWO_PROGNOSTIC = ["ocean_heat_content_2d", "tau_x"]


def two_prognostic_cfg(synthetic_file, tmp_path, **overrides):
    """tau_x as a second prognostic variable (tau_y stays forcing)."""
    return make_cfg(
        synthetic_file, tmp_path,
        data=DataConfig(path=str(synthetic_file), prognostic=list(TWO_PROGNOSTIC),
                        forcing=["total_surface_heat_flx", "tau_y"],
                        normalisation=NormalisationConfig(fit_period=("2000-01", "2003-12")),
                        cache_dir=str(tmp_path / "cache")),
        **overrides,
    )


def test_two_prognostic_variables(synthetic_file, tmp_path):
    """tau_x as a second prognostic variable: channels, rollout and closure all follow."""
    cfg = two_prognostic_cfg(synthetic_file, tmp_path)
    data = build_data(cfg, verbose=False)
    module = build_module(cfg, build_model(cfg, data), build_losses(cfg, data), data)
    batch = data.windows(next(iter(data.train_dl)))
    assert batch["prior"].shape[1:3] == (2, 2)
    loss = module._step(batch, 4, "train")
    loss.backward()
    assert torch.isfinite(loss)


def test_loss_step_weights(data_and_cfg):
    data, cfg = data_and_cfg
    cfg.loss.terms = {"local_mse": [1.0, 1.0, 2.0, 2.0], "heat_closure": 0.5}
    cfg.loss.step_weights = [1.0, 1.0, 1.0, 3.0]
    try:
        terms = {t.name: t.step_weights for t in build_losses(cfg, data)}
    finally:
        cfg.loss.terms = {"local_mse": 1.0, "spectral": 0.0, "heat_closure": 0.1}
        cfg.loss.step_weights = None
    relative = [2 / 3, 2 / 3, 2 / 3, 2.0]  # step_weights / mean(step_weights)
    assert set(terms) == {"local_mse", "heat_closure"}  # spectral (0) dropped
    np.testing.assert_allclose(terms["local_mse"], [a * r for a, r in zip([1, 1, 2, 2], relative)])
    np.testing.assert_allclose(terms["heat_closure"], [0.5 * r for r in relative])


# =============================================================================
# Training and evaluation
# =============================================================================

def test_one_epoch_trains(data_and_cfg):
    data, cfg = data_and_cfg
    model = build_model(cfg, data)
    module = build_module(cfg, model, build_losses(cfg, data), data)
    before = [p.detach().clone() for p in model.parameters()]
    trainer = build_trainer(cfg)
    trainer.fit(module, data.train_dl, data.valid_dl)
    assert torch.isfinite(trainer.callback_metrics["val_loss"])
    assert "train_heat_closure" in trainer.callback_metrics
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters())), "weights did not change"


def test_rollout_schedule():
    schedule = RolloutSchedule({0: 1, 3: 4, 10: 12})
    assert [schedule.steps_for(e) for e in (0, 2, 3, 9, 10, 50)] == [1, 1, 4, 4, 12, 12]


def test_skill_test_and_control(data_and_cfg):
    data, cfg = data_and_cfg
    model = build_model(cfg, data)
    skill = run_skill_test(cfg, model, data)
    assert skill.sizes["time"] == 22  # seeded 2004-01, predicts 2004-02 .. 2005-11
    assert str(skill.time.values[0])[:7] == "2004-02"
    name = "ocean_heat_content_2d"
    land = data.fields["mask"].numpy() == 0
    assert np.isnan(skill[f"{name}_pred"].values[:, land]).all()
    ocean = ~land
    # truth anomaly + climatology = truth; truth matches the file
    truth_z = data.fields["prognostic"][data.index_of("2004-02"), 0].numpy()
    std = data.fields["prognostic_std"][data.index_of("2004-02"), 0].numpy()
    np.testing.assert_allclose(skill[f"{name}_truth_anom"].values[0][ocean], (truth_z * std)[ocean], rtol=1e-5)

    control = run_control(cfg, model, data)
    assert control.sizes["step"] == 24
    assert list(control.forcing_month.values[:2]) == ["2002-02", "2002-03"]


def test_known_closure_runs(data_and_cfg):
    data, cfg = data_and_cfg
    results = check_known_closure(cfg, data, warn_threshold=np.inf)
    assert set(results) == {"heat"}  # freshwater_closure is off in make_cfg
    assert len(results["heat"]["by_lead"]) == cfg.window.posterior_steps
    assert np.isfinite(results["heat"]["mean"])


# =============================================================================
# Per-variable RMSE and the multi-variable evaluation figures
# =============================================================================

def test_per_variable_error_is_area_weighted_and_adds_no_loss():
    rng = torch.Generator().manual_seed(0)
    area = torch.rand(H, W, generator=rng) + 0.5
    mask = (torch.rand(H, W, generator=rng) > 0.3).float()
    pred = torch.randn(4, 3, H, W, generator=rng)
    target = pred.clone()
    target[:, 1] += 2.0
    target[:, 2] += 3.0 * (1 - mask)  # error on land only: ignored
    metric = PerVariableError(3)
    for _ in range(3):
        assert metric(pred_t=pred, target_t=target, mask=mask, area=area).item() == 0.0
    np.testing.assert_allclose(metric.pop_rmse().numpy(), [0.0, 2.0, 0.0], atol=1e-5)
    assert metric.pop_rmse() is None  # reset after popping


def test_rmse_history_per_variable(synthetic_file, tmp_path):
    """Training records train/val RMSE per variable and epoch, in memory and in metrics.csv."""
    run_dir = tmp_path / "run"
    cfg = two_prognostic_cfg(
        synthetic_file, tmp_path,
        train=TrainConfig(batch_size=8, max_epochs=2, gpu_check=False, run_dir=str(run_dir)),
    )
    data = build_data(cfg, verbose=False)
    module = build_module(cfg, build_model(cfg, data), build_losses(cfg, data), data)
    build_trainer(cfg).fit(module, data.train_dl, data.valid_dl)

    history = module.rmse_history
    assert len(history) == 2 * 2 * len(TWO_PROGNOSTIC)  # epochs x stages x variables
    assert set(history["variable"]) == set(TWO_PROGNOSTIC)
    assert set(history["stage"]) == {"train", "val"}
    assert sorted(set(history["epoch"])) == [0, 1]
    assert np.isfinite(history["rmse"]).all() and (history["rmse"] > 0).all()

    metrics = pd.read_csv(run_dir / "metrics.csv")
    for stage in ("train", "val"):
        for name in TWO_PROGNOSTIC:
            assert f"{stage}_rmse_{name}" in metrics

    for source in (history, run_dir):
        fig = plot_rmse_by_epoch(source, TWO_PROGNOSTIC)
        assert len(fig.axes) == len(TWO_PROGNOSTIC)
        assert all(len(ax.lines) == 2 for ax in fig.axes)  # training + validation
        plt.close(fig)


def test_evaluation_figures_for_every_variable(synthetic_file, tmp_path):
    cfg = two_prognostic_cfg(synthetic_file, tmp_path)
    data = build_data(cfg, verbose=False)
    skill = run_skill_test(cfg, build_model(cfg, data), data)
    periods = ["2004-06", "2005-01", "2005"]

    for name in TWO_PROGNOSTIC:
        fig = plot_skill_evaluation(skill, name, periods, {"ocean_heat_content_2d": 2e9}, None)
        maps = [ax for ax in fig.axes if ax.collections and ax.get_label() != "<colorbar>"]
        assert len(maps) == 3 * len(periods)
        timeseries = [ax for ax in fig.axes if len(ax.lines) >= 2]
        assert len(timeseries) == 1  # predicted + truth, full width underneath
        path = tmp_path / f"{name}_skill_evaluation.png"
        fig.savefig(path)
        assert path.stat().st_size > 0
        plt.close(fig)

    fig = plot_global_rmse_all_variables(skill, TWO_PROGNOSTIC)
    assert len(fig.axes) == len(TWO_PROGNOSTIC)
    assert all(np.isfinite(ax.lines[0].get_ydata()).all() for ax in fig.axes)
    plt.close(fig)


# =============================================================================
# Heat and freshwater closures as independent loss terms
# =============================================================================

HEAT_AND_FRESHWATER = dict(
    prognostic=["ocean_heat_content_2d", "ocean_freshwater_content_2d"],
    forcing=["total_surface_heat_flx", "ocean_freshwater_flux", "tau_x"],
)


def closure_cfg(synthetic_file, tmp_path, terms, **data_overrides):
    variables = {**HEAT_AND_FRESHWATER, **data_overrides}
    return make_cfg(
        synthetic_file, tmp_path,
        data=DataConfig(path=str(synthetic_file), **variables,
                        normalisation=NormalisationConfig(fit_period=("2000-01", "2003-12")),
                        cache_dir=str(tmp_path / "cache")),
        loss=LossConfig(terms=terms),
    )


def test_global_closure_term_is_renamed(synthetic_file, tmp_path):
    with pytest.raises(ValueError, match="heat_closure"):
        closure_cfg(synthetic_file, tmp_path, {"local_mse": 1.0, "global_closure": 0.1})


def test_closure_needs_its_variables(synthetic_file, tmp_path):
    with pytest.raises(ValueError, match="ocean_freshwater_content_2d"):
        closure_cfg(synthetic_file, tmp_path, {"local_mse": 1.0, "freshwater_closure": 0.1},
                    prognostic=["ocean_heat_content_2d"])
    with pytest.raises(ValueError, match="ocean_freshwater_flux"):
        closure_cfg(synthetic_file, tmp_path, {"local_mse": 1.0, "freshwater_closure": 0.1},
                    forcing=["total_surface_heat_flx", "tau_x"],
                    prognostic=["ocean_heat_content_2d", "ocean_freshwater_content_2d"])
    # An inactive closure (weight 0) needs nothing.
    closure_cfg(synthetic_file, tmp_path, {"local_mse": 1.0, "freshwater_closure": 0.0},
                prognostic=["ocean_heat_content_2d"])


def test_custom_budget_becomes_a_loss_term(synthetic_file, tmp_path):
    with pytest.raises(ValueError, match="Unknown loss terms"):
        closure_cfg(synthetic_file, tmp_path, {"local_mse": 1.0, "fw2_closure": 0.2})
    cfg = closure_cfg(synthetic_file, tmp_path, {"local_mse": 1.0})
    cfg.loss.closures["fw2"] = ClosureConfig("ocean_freshwater_content_2d", "ocean_freshwater_flux", 1.0, 1.0)
    cfg.loss.terms["fw2_closure"] = 0.2
    cfg.validate()
    assert set(cfg.active_closures()) == {"fw2"}


def test_heat_and_freshwater_closures_train_with_independent_weights(synthetic_file, tmp_path):
    cfg = closure_cfg(synthetic_file, tmp_path,
                      {"local_mse": 1.0, "heat_closure": 0.1, "freshwater_closure": [0.0, 0.0, 0.3, 0.3]})
    data = build_data(cfg, verbose=False)
    assert data.fields["forcing_std"].shape[1] == len(HEAT_AND_FRESHWATER["forcing"])
    losses = build_losses(cfg, data)
    weights = {t.name: t.step_weights for t in losses}
    assert weights["heat_closure"] == [0.1] * 4
    assert weights["freshwater_closure"] == [0.0, 0.0, 0.3, 0.3]

    module = build_module(cfg, build_model(cfg, data), losses, data)
    assert set(module.closure_std) == {"heat", "freshwater"}
    # Each budget gets its own content and flux std (OHC and FWC are very different sizes).
    assert not torch.equal(module.closure_std["heat"][0], module.closure_std["freshwater"][0])
    batch = data.windows(next(iter(data.train_dl)))
    loss = module._step(batch, 4, "train")
    loss.backward()
    assert torch.isfinite(loss)

    results = check_known_closure(cfg, data, warn_threshold=np.inf, max_batches=2)
    assert set(results) == {"heat", "freshwater"}
    assert all(np.isfinite(r["mean"]) for r in results.values())


def test_only_active_closures_use_gpu_buffers(synthetic_file, tmp_path):
    cfg = closure_cfg(synthetic_file, tmp_path, {"local_mse": 1.0, "freshwater_closure": 0.1})
    data = build_data(cfg, verbose=False)
    module = build_module(cfg, build_model(cfg, data), build_losses(cfg, data), data)
    assert module.budgets == ["freshwater"]
    assert not hasattr(module, "heat_content_std")


def test_plot_scales_are_automatic(data_and_cfg):
    """No fixed colour limits: each variable's maps scale to its own anomalies."""
    from Experiment import plot_snapshots

    data, cfg = data_and_cfg
    skill = run_skill_test(cfg, build_model(cfg, data), data)
    fig = plot_snapshots(skill, ["2004-06"])
    vmax = fig.axes[0].collections[0].get_clim()[1]
    truth = np.abs(skill["ocean_heat_content_2d_truth_anom"].values)
    assert vmax == pytest.approx(np.nanquantile(truth, 0.99))
    plt.close(fig)


# =============================================================================
# Architecture: widths scale with the variables, residual prediction,
# full-resolution input skip
# =============================================================================

def test_unet_widths_scale_with_input_channels(synthetic_file, tmp_path):
    for cfg in (make_cfg(synthetic_file, tmp_path), two_prognostic_cfg(synthetic_file, tmp_path)):
        data = build_data(cfg, verbose=False)
        unet = build_model(cfg, data).backbone
        in_ch = cfg.window.n_prior * data.n_prognostic + data.n_forcing
        width1 = cfg.model.width_multiplier * in_ch
        assert unet.enc1.conv.in_channels == in_ch
        assert unet.enc1.conv.out_channels == width1
        assert unet.enc2.conv.out_channels == 2 * width1
        assert unet.latent_channel_count == unet.enc3.out_channels == 4 * width1
        assert unet.dec3.conv.in_channels == width1 + in_ch  # full-resolution input skip
        assert unet.dec3.conv.out_channels == data.n_prognostic


def test_width_multiplier_must_be_positive(synthetic_file, tmp_path):
    with pytest.raises(ValueError, match="width_multiplier"):
        make_cfg(synthetic_file, tmp_path, model=ModelConfig(width_multiplier=0))


def test_zero_backbone_output_persists_last_state(data_and_cfg):
    """The emulator predicts the change: a backbone that outputs 0 persists the last prior state."""
    data, cfg = data_and_cfg
    model = build_model(cfg, data)
    with torch.no_grad():
        model.backbone.dec3.conv.weight.zero_()
        model.backbone.dec3.conv.bias.zero_()
    prior = torch.randn(2, cfg.window.n_prior * data.n_prognostic, H, W)
    out = model(prior, torch.randn(2, data.n_forcing, H, W), data.fields["mask"])
    torch.testing.assert_close(out, prior[:, -data.n_prognostic :])


def test_unet_full_resolution_skip_can_copy_the_input():
    """
    The output layer sees the raw input at full resolution, so the UNet can
    reproduce grid-scale structure exactly (impossible through the 1/2-resolution path).
    """
    from Emulator import UNet

    channels, h, w = 3, 15, 21  # odd grid: the output must still match the input size
    unet = UNet(input_channel_count=channels, output_channel_count=channels, padding_mode="zeros")
    width1 = unet.dec3.conv.in_channels - channels
    with torch.no_grad():
        unet.dec3.conv.weight.zero_()
        unet.dec3.conv.bias.zero_()
        for c in range(channels):
            unet.dec3.conv.weight[c, width1 + c, 1, 1] = 1.0  # centre tap on raw input channel c
    x = torch.randn(2, channels, h, w)
    out = unet(x, torch.ones(2, 1, h, w))
    assert out.shape == x.shape
    # Interior cells (the edges are renormalised by the zero padding).
    torch.testing.assert_close(out[..., 1:-1, 1:-1], x[..., 1:-1, 1:-1])


# =============================================================================
# Speed and memory: lean partial convolution, 3x3 stacks, gradient
# checkpointing, channels_last, compile
# =============================================================================

def _reference_partial_conv(layer, x, mask):
    """The original PartialConv2d forward: mask, convolve, where(renormalise, 0)."""
    out = layer.conv(x * mask)
    mask_sum = layer.mask_conv(mask)
    out = torch.where(mask_sum > 0, out * (layer.kernel_area / (mask_sum + layer.eps)), torch.zeros_like(out))
    return out, (mask_sum > 0).float()


@pytest.mark.parametrize("kernel_size,stride,padding", [(3, 1, 1), (4, 2, 1), (7, 1, 3)])
def test_partial_conv_matches_original(kernel_size, stride, padding):
    from Emulator import PartialConv2d

    torch.manual_seed(0)
    layer = PartialConv2d(5, 6, kernel_size, stride, padding, padding_mode="zeros")
    x = torch.randn(3, 5, H, W)
    mask = (torch.rand(1, 1, H, W) > 0.3).float()
    expected, expected_mask = _reference_partial_conv(layer, x, mask.expand(3, 1, H, W))
    for m in (mask, mask.expand(3, 1, H, W)):  # shared (1, 1, H, W) or per-sample mask
        out, new_mask = layer(x, m)
        torch.testing.assert_close(out, expected)
        torch.testing.assert_close(new_mask.expand_as(expected_mask), expected_mask)


def test_conv_stack_grows_the_mask_like_a_7x7():
    from Emulator import PartialConv2d, PartialConvStack

    mask = (torch.rand(1, 1, H, W, generator=torch.Generator().manual_seed(1)) > 0.85).float()
    _, mask7 = PartialConv2d(2, 2, kernel_size=7, padding=3, padding_mode="zeros")(torch.randn(1, 2, H, W), mask)
    stack = PartialConvStack(2, 3, hidden_channels=4, padding_mode="zeros")
    out, mask_stack = stack(torch.randn(2, 2, H, W), mask)
    assert out.shape == (2, 3, H, W)
    torch.testing.assert_close(mask_stack, mask7)


def test_unet_has_no_large_kernels(data_and_cfg):
    data, cfg = data_and_cfg
    sizes = {m.kernel_size for m in build_model(cfg, data).modules() if isinstance(m, torch.nn.Conv2d)}
    assert max(max(k) for k in sizes) <= 4


def test_checkpointed_rollout_gives_same_loss_and_gradients(data_and_cfg):
    from Emulator import total_rollout_loss

    data, cfg = data_and_cfg
    model = build_model(cfg, data)
    losses = build_losses(cfg, data)
    module = build_module(cfg, model, losses, data)
    batch = data.windows(next(iter(data.train_dl)))
    results = []
    # No checkpointing, every step checkpointed, and steps 0-1 checkpointed with 2-3 kept.
    for checkpoint_steps, steps_in_memory in ((False, 0), (True, 0), (True, 2)):
        model.zero_grad()
        loss = total_rollout_loss(
            model=model,
            initial_prior_states=batch["prior"].flatten(1, 2),
            forcing_sequence=batch["forcing"],
            target_sequence=batch["target"],
            mask=module.mask,
            losses=losses,
            n_steps=4,
            target_time_indices=batch["target_time_index"],
            area=module.area,
            closure_std=module.closure_std,
            dt_seconds=cfg.loss.seconds_per_step,
            initial_forcing=batch["initial_forcing"],
            n_prognostic=data.n_prognostic,
            checkpoint_steps=checkpoint_steps,
            steps_in_memory=steps_in_memory,
        )
        loss.backward()
        for term in losses:
            term.pop_running()
        results.append((loss.detach(), [p.grad.clone() for p in model.parameters()]))
    for loss, grads in results[1:]:
        torch.testing.assert_close(loss, results[0][0])
        for a, b in zip(grads, results[0][1]):
            torch.testing.assert_close(a, b)


def test_checkpointed_training_runs(synthetic_file, tmp_path):
    cfg = make_cfg(synthetic_file, tmp_path,
                   train=TrainConfig(batch_size=8, max_epochs=1, gpu_check=False, checkpoint_rollout_steps=True,
                                     rollout_steps_in_memory=2))
    data = build_data(cfg, verbose=False)
    module = build_module(cfg, build_model(cfg, data), build_losses(cfg, data), data)
    build_trainer(cfg).fit(module, data.train_dl, data.valid_dl)
    assert len(module.rmse_history) > 0


def test_channels_last_gives_same_output(synthetic_file, tmp_path):
    outputs = []
    prior = torch.randn(2, 2, H, W)
    forcing = torch.randn(2, len(FORCING), H, W)
    for channels_last in (False, True):
        cfg = make_cfg(synthetic_file, tmp_path, model=ModelConfig(channels_last=channels_last))
        data = build_data(cfg, verbose=False)
        model = build_model(cfg, data)  # same seed -> same weights
        outputs.append(model(prior, forcing, data.fields["mask"]))
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-4, atol=1e-5)


def test_compiled_model_keeps_plain_state_dict(data_and_cfg):
    """torch.compile is lazy, so building the wrapper is cheap; checkpoints must not change keys."""
    data, cfg = data_and_cfg
    model = build_model(cfg, data)
    plain = build_module(cfg, model, build_losses(cfg, data), data)
    cfg.train.compile_model = True
    try:
        compiled = build_module(cfg, model, build_losses(cfg, data), data)
    finally:
        cfg.train.compile_model = False
    assert compiled._run_model is not compiled.model
    assert list(compiled.state_dict()) == list(plain.state_dict())


# =============================================================================
# Data pipeline: index batches, windows cut on the module's device
# =============================================================================

def test_dataloaders_yield_initial_months(data_and_cfg):
    data, cfg = data_and_cfg
    batch = next(iter(data.valid_dl))
    assert batch.dtype == torch.long and batch.ndim == 1
    torch.testing.assert_close(batch, data.valid_indices[: cfg.train.batch_size])


def test_windows_cut_only_the_steps_in_use(data_and_cfg):
    data, cfg = data_and_cfg
    t0 = data.train_indices[:3]
    full = data.windows(t0)
    short = data.windows(t0, n_steps=2)
    assert full["target"].shape[1] == cfg.window.posterior_steps
    assert short["target"].shape[1] == short["forcing"].shape[1] == 2
    for key in ("prior", "initial_forcing"):
        torch.testing.assert_close(short[key], full[key])
    torch.testing.assert_close(short["target"], full["target"][:, :2])
    torch.testing.assert_close(short["forcing"], full["forcing"][:, :2])
    torch.testing.assert_close(short["target_time_index"], full["target_time_index"][:, :2])


def test_module_cuts_windows_from_its_own_fields(data_and_cfg):
    """An index batch gives the same loss as the equivalent window dict; the fields stay out of checkpoints."""
    data, cfg = data_and_cfg
    module = build_module(cfg, build_model(cfg, data), build_losses(cfg, data), data)
    t0 = next(iter(data.train_dl))
    with torch.no_grad():
        from_indices = module._step(t0, 3, "val")
        from_windows = module._step(data.windows(t0, n_steps=3), 3, "val")
    torch.testing.assert_close(from_indices, from_windows)
    keys = set(module.state_dict())
    assert "prognostic_z" not in keys and "forcing_z" not in keys
    assert module.prognostic_z.data_ptr() == data.fields["prognostic"].data_ptr()  # no CPU copy


def test_rollout_steps_in_memory_must_be_non_negative(synthetic_file, tmp_path):
    with pytest.raises(ValueError, match="rollout_steps_in_memory"):
        make_cfg(synthetic_file, tmp_path, train=TrainConfig(rollout_steps_in_memory=-1))
