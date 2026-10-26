"""
Config-driven emulator experiments.

A notebook defines one ExperimentConfig and calls the builders below; the data
pipeline, model, losses, training and evaluation all live in this package.

    cfg     = ExperimentConfig(data=DataConfig(path=...), ...)
    data    = build_data(cfg)
    model   = build_model(cfg, data)
    losses  = build_losses(cfg, data)
    module  = build_module(cfg, model, losses, data)
    trainer = build_trainer(cfg)
    trainer.fit(module, data.train_dl, data.valid_dl)
    skill   = run_skill_test(cfg, model, data)

Paradigms (cfg.paradigm): "deterministic", "crps_ensemble" (build_model adds
noise inputs; trained with the "crps" term) and "diffusion" (build_diffusion on
a trained deterministic model, trained with build_diffusion_module). Branches
start from shared weights with load_trunk; intercomparison.py compares them.
"""

from .config import (
    ClosureConfig,
    DataConfig,
    DiffusionConfig,
    EnsembleConfig,
    EvalConfig,
    ExperimentConfig,
    LossConfig,
    ModelConfig,
    NormalisationConfig,
    TimeConfig,
    TrainConfig,
    WindowConfig,
)
from .constraints import GlobalBudgetConstraint, build_constraint
from .data import ExperimentData, InitialMonthDataset, build_data, compute_normalisation, gather_windows
from .evaluation import (
    check_known_closure,
    climatology_rmse,
    default_members,
    global_integral,
    global_rmse,
    implied_meridional_transport,
    leading_growth_mode,
    meridional_transports,
    persistence_rmse,
    run_control,
    run_skill_test,
    skill_diagnostics,
    skill_summary,
    zonal_spectrum,
)
from .diffusion import ResidualDiffusion, build_diffusion, build_diffusion_module
from .intercomparison import (
    BASIN_GROUPS,
    OM2_1DEG_BASIN_MASK,
    basin_anomaly,
    basin_scores,
    ensemble_scores,
    load_basins,
    plot_basin_content,
    global_anomaly,
    intercomparison_summary,
    plot_intercomparison,
    plot_intercomparison_spectra,
    plot_intercomparison_transports,
)
from .losses import build_losses
from .models import ForwardEmulator, build_model, count_parameters, load_trunk
from .plots import (
    plot_control,
    plot_growth_mode,
    plot_meridional_transports,
    plot_global_rmse,
    plot_global_rmse_all_variables,
    plot_global_timeseries,
    plot_rmse_by_epoch,
    plot_skill_diagnostics,
    plot_skill_evaluation,
    plot_snapshots,
    plot_spectra,
    plot_training_history,
    prognostic_variables,
)
from .training import (
    EmulatorModule, build_module, build_trainer, check_gpu, last_checkpoint, prepare_resume, read_metrics,
    training_minutes,
)

__all__ = [
    "DiffusionConfig",
    "EnsembleConfig",
    "GlobalBudgetConstraint",
    "build_constraint",
    "default_members",
    "ResidualDiffusion",
    "build_diffusion",
    "build_diffusion_module",
    "ensemble_scores",
    "BASIN_GROUPS",
    "OM2_1DEG_BASIN_MASK",
    "basin_anomaly",
    "basin_scores",
    "load_basins",
    "plot_basin_content",
    "global_anomaly",
    "intercomparison_summary",
    "plot_intercomparison",
    "plot_intercomparison_spectra",
    "plot_intercomparison_transports",
    "load_trunk",
    "ClosureConfig",
    "DataConfig",
    "EvalConfig",
    "ExperimentConfig",
    "LossConfig",
    "ModelConfig",
    "NormalisationConfig",
    "TimeConfig",
    "TrainConfig",
    "WindowConfig",
    "ExperimentData",
    "InitialMonthDataset",
    "build_data",
    "gather_windows",
    "compute_normalisation",
    "check_known_closure",
    "climatology_rmse",
    "global_integral",
    "global_rmse",
    "implied_meridional_transport",
    "leading_growth_mode",
    "meridional_transports",
    "persistence_rmse",
    "run_control",
    "run_skill_test",
    "skill_diagnostics",
    "skill_summary",
    "zonal_spectrum",
    "build_losses",
    "ForwardEmulator",
    "build_model",
    "count_parameters",
    "plot_control",
    "plot_growth_mode",
    "plot_meridional_transports",
    "plot_global_rmse",
    "plot_global_rmse_all_variables",
    "plot_global_timeseries",
    "plot_rmse_by_epoch",
    "plot_skill_diagnostics",
    "plot_skill_evaluation",
    "plot_snapshots",
    "plot_spectra",
    "plot_training_history",
    "prognostic_variables",
    "EmulatorModule",
    "build_module",
    "build_trainer",
    "check_gpu",
    "last_checkpoint",
    "prepare_resume",
    "read_metrics",
    "training_minutes",
]
