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
        loss=LossConfig(terms={"local_mse": 1.0, "spectral": 0.0, "global_closure": 0.1}),
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

    sample = data.train_dl.dataset[5]
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
    batch = next(iter(data.train_dl))
    assert batch["prior"].shape[1:3] == (2, 2)
    loss = module._step(batch, 4, "train")
    loss.backward()
    assert torch.isfinite(loss)


def test_loss_step_weights(data_and_cfg):
    data, cfg = data_and_cfg
    cfg.loss.terms = {"local_mse": [1.0, 1.0, 2.0, 2.0], "global_closure": 0.5}
    cfg.loss.step_weights = [1.0, 1.0, 1.0, 3.0]
    try:
        terms = {t.name: t.step_weights for t in build_losses(cfg, data)}
    finally:
        cfg.loss.terms = {"local_mse": 1.0, "spectral": 0.0, "global_closure": 0.1}
        cfg.loss.step_weights = None
    relative = [2 / 3, 2 / 3, 2 / 3, 2.0]  # step_weights / mean(step_weights)
    assert set(terms) == {"local_mse", "global_closure"}  # spectral (0) dropped
    np.testing.assert_allclose(terms["local_mse"], [a * r for a, r in zip([1, 1, 2, 2], relative)])
    np.testing.assert_allclose(terms["global_closure"], [0.5 * r for r in relative])


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
    assert "train_global_closure" in trainer.callback_metrics
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
    result = check_known_closure(cfg, data, warn_threshold=np.inf)
    assert len(result["by_lead"]) == cfg.window.posterior_steps
    assert np.isfinite(result["mean"])


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
