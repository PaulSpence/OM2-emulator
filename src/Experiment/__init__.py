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
"""

from .config import (
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
)
from .data import ExperimentData, InitialMonthDataset, build_data, compute_normalisation, gather_windows
from .evaluation import (
    check_known_closure,
    global_integral,
    global_rmse,
    leading_growth_mode,
    persistence_rmse,
    run_control,
    run_skill_test,
)
from .losses import build_losses
from .models import ForwardEmulator, build_model, count_parameters
from .plots import (
    plot_control,
    plot_growth_mode,
    plot_global_rmse,
    plot_global_rmse_all_variables,
    plot_global_timeseries,
    plot_rmse_by_epoch,
    plot_skill_evaluation,
    plot_snapshots,
    plot_training_history,
    prognostic_variables,
)
from .training import EmulatorModule, build_module, build_trainer, check_gpu

__all__ = [
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
    "global_integral",
    "global_rmse",
    "leading_growth_mode",
    "persistence_rmse",
    "run_control",
    "run_skill_test",
    "build_losses",
    "ForwardEmulator",
    "build_model",
    "count_parameters",
    "plot_control",
    "plot_growth_mode",
    "plot_global_rmse",
    "plot_global_rmse_all_variables",
    "plot_global_timeseries",
    "plot_rmse_by_epoch",
    "plot_skill_evaluation",
    "plot_snapshots",
    "plot_training_history",
    "prognostic_variables",
    "EmulatorModule",
    "build_module",
    "build_trainer",
    "check_gpu",
]
