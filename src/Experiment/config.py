"""
Experiment configuration: every user-facing setting of an emulator run.

The notebook builds one ``ExperimentConfig`` and passes it to the builders in
this package (``build_data``, ``build_model``, ``build_losses``,
``build_module``, ``build_trainer`` and the evaluation functions). Nothing else
in the notebook should need editing to change an experiment.

Dates are "YYYY-MM" strings. All date ranges are INCLUSIVE.

Sample dates: a training/validation sample is identified by its INITIAL month,
the month of its last prior state. A sample with initial month t uses the prior
states t-n_prior+1 ... t, and predicts months t+1 ... t+posterior_steps, driven
by the forcing in those same months.
"""

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Names accepted by the registries in models.py and losses.py. Kept here so the
# config can be validated before anything expensive runs.
KNOWN_NORMALISATION_STRATEGIES = (
    "Spatial_climatology",
    "Spatial_climatology_global_variance",
    "Spatial_in_time",
    "Global_in_time",
)
KNOWN_ARCHITECTURES = ("unet",)
KNOWN_OUTPUT_HEADS = (None, "spatial_residual")
KNOWN_LATENT_PROCESSORS = (None, "latent_residual")
KNOWN_LOSS_TERMS = ("local_mse", "spectral", "global_closure")


def _month(value):
    """Parse a "YYYY-MM" string to a pandas month period, with a clear error."""
    try:
        return pd.Period(value, freq="M")
    except (ValueError, TypeError) as err:
        raise ValueError(f"Expected a 'YYYY-MM' date, got {value!r}") from err


def _check_range(name, date_range):
    start, end = (_month(d) for d in date_range)
    if end < start:
        raise ValueError(f"{name} ends ({date_range[1]}) before it starts ({date_range[0]})")


@dataclass
class NormalisationConfig:
    # How each variable is standardised: z = (x - mean) / std.
    #   Spatial_climatology                 -- per-grid-cell, per-calendar-month mean and std
    #   Spatial_climatology_global_variance -- per-cell monthly mean, one area-weighted std
    #   Spatial_in_time                     -- per-grid-cell mean and std over the fit period
    #   Global_in_time                      -- one area-weighted mean and std per variable
    strategy: str = "Spatial_climatology"
    # Months used to fit the statistics (inclusive). Keep this inside the
    # training period so no validation/test information leaks into them.
    fit_period: tuple = ("1970-01", "2004-02")
    # Area-weight the global statistics (only used by the *global* strategies).
    area_weight: bool = True


@dataclass
class DataConfig:
    # The emulator NetCDF file (see notebooks/Extract_om2_data.ipynb).
    path: str
    # Prognostic variables: predicted by the model AND fed back as its inputs at
    # the next rollout step. These are both the output variables and the
    # autoregressive state inputs.
    prognostic: list = field(default_factory=lambda: ["ocean_heat_content_2d"])
    # Forcing variables: prescribed from the data at every step, never predicted.
    forcing: list = field(
        default_factory=lambda: ["total_surface_heat_flx", "tau_x", "tau_y"]
    )
    # Variables the global heat-closure loss needs.
    ohc_variable: str = "ocean_heat_content_2d"
    heat_flux_variable: str = "total_surface_heat_flx"
    # Land mask: cells where this variable is NaN at the first time are land.
    mask_variable: str = "total_surface_heat_flx"
    area_variable: str = "area_t"
    normalisation: NormalisationConfig = field(default_factory=NormalisationConfig)
    # Where the normalised fields are cached. None -> "<data dir>/emulator_cache".
    # The cache file name includes a hash of every data setting, so changing
    # any of them builds a new cache automatically.
    cache_dir: str | None = None


@dataclass
class TimeConfig:
    # Full time axis loaded from the file (inclusive). All other dates must lie
    # inside it, including the prior states and targets of every sample.
    time_axis: tuple = ("1970-01", "2018-12")
    # Initial months of the training and validation samples (inclusive).
    train: tuple = ("1970-02", "2004-01")
    valid: tuple = ("2004-02", "2005-01")
    # Skill test: a single free-running rollout seeded from the truth at
    # test[0], driven by the real forcing, predicting test[0]+1 ... test[1]+1.
    test: tuple = ("2005-02", "2018-11")
    # Control run: the forcing of these months (inclusive) repeated
    # control_repeats times, seeded from the truth at control[0]. Isolates
    # drift under (near-)constant forcing. None disables it.
    control: tuple | None = ("2003-01", "2003-12")
    control_repeats: int = 10


@dataclass
class WindowConfig:
    # Number of past states the model sees (t-n_prior+1 ... t).
    n_prior: int = 2
    # Targets available per sample (the posterior window). Rollouts can use up
    # to this many steps.
    posterior_steps: int = 12
    # Autoregressive steps per training sample.
    rollout_steps: int = 12
    # Optional curriculum: {first_epoch: rollout_steps}, e.g. {0: 1, 20: 4, 50: 12}.
    # Training cost scales linearly with rollout length. None = always rollout_steps.
    rollout_schedule: dict | None = None
    # Steps used for validation. None = rollout_steps. Keep it fixed so val_loss
    # stays comparable across epochs when a schedule is used.
    valid_rollout_steps: int | None = None


@dataclass
class LossConfig:
    # Loss terms and their weights, summed at every rollout step and averaged
    # over the steps. A weight of 0 removes the term entirely.
    #   local_mse      -- ocean-masked MSE in z-score space (primary fit)
    #   spectral       -- MSE of log spatial amplitude spectra (penalises over-smoothing)
    #   global_closure -- global heat-budget closure on physical anomalies
    #                     (see issue #43: the true data score ~0.5 on it)
    # A weight may also be a list with one ABSOLUTE weight per rollout step.
    terms: dict = field(
        default_factory=lambda: {"local_mse": 1.0, "spectral": 0.0, "global_closure": 0.1}
    )
    # Optional relative emphasis per rollout step, applied to every term and
    # rescaled to mean 1 (so it changes the balance across lead times, not the
    # overall size), e.g. [1] * 6 + [2] * 6. Length must be posterior_steps.
    step_weights: list | None = None
    # Closure settings.
    surface_flux_sign: float = 1.0  # +1 if positive flux means heat INTO the ocean
    closure_min_scale: float = 1.0e20  # J, floor on the closure normalisation scale
    seconds_per_step: float = 30 * 24 * 60 * 60  # one month


@dataclass
class ModelConfig:
    arch: str = "unet"
    # Edge padding of every partial convolution. "zeros" trains ~2x faster than
    # "replicate" and uses ~35% less GPU memory (see PartialConv2d).
    padding_mode: str = "zeros"
    # Full-resolution learned correction of the prediction (the former ResNet
    # notebook): None or "spatial_residual".
    output_head: str | None = None
    output_head_hidden_channels: int = 16
    output_head_residual_scale: float = 1.0
    # Residual refinement of the UNet bottleneck: None or "latent_residual".
    latent_processor: str | None = None
    latent_residual_scale: float = 0.1


@dataclass
class TrainConfig:
    batch_size: int = 64
    # Effective batch = batch_size * accumulate_grad_batches.
    accumulate_grad_batches: int = 1
    max_epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 0.0
    lr_scheduler: str | None = None  # None or "cosine"
    gradient_clip_val: float | None = None
    # Lightning precision: "32-true", or "16-mixed" (fp16, ~1.5x faster on a
    # V100). Predictions are cast back to float32 before the losses.
    precision: str = "32-true"
    seed: int = 42
    num_workers: int = 0
    check_val_every_n_epoch: int = 1
    # Directory for the config, CSV logs and checkpoints. None = no logging or
    # checkpointing (training still works; the model stays in memory).
    run_dir: str | None = None
    # Warn before training if other processes are using the GPU.
    gpu_check: bool = True


@dataclass
class EvalConfig:
    run_skill_test: bool = True
    run_control: bool = False
    # Snapshot maps: months ("YYYY-MM") or years ("YYYY"), one row each.
    snapshot_periods: list = field(default_factory=lambda: ["2005-05", "2010-12", "2015-12"])
    anomaly_scale: float = 2e9
    difference_scale: float = 2e9


@dataclass
class ExperimentConfig:
    data: DataConfig
    time: TimeConfig = field(default_factory=TimeConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self):
        self.validate()

    # ------------------------------------------------------------------ checks
    def validate(self):
        """Raise ValueError on any inconsistent setting, before any work is done."""
        d, t, w, lo, m, tr = self.data, self.time, self.window, self.loss, self.model, self.train

        if not d.prognostic:
            raise ValueError("data.prognostic must list at least one variable")
        overlap = set(d.prognostic) & set(d.forcing)
        if overlap:
            raise ValueError(f"{sorted(overlap)} are listed as both prognostic and forcing")
        for name in (*d.prognostic, *d.forcing):
            if d.prognostic.count(name) + d.forcing.count(name) > 1:
                raise ValueError(f"{name!r} is listed more than once")
        if d.normalisation.strategy not in KNOWN_NORMALISATION_STRATEGIES:
            raise ValueError(
                f"Unknown normalisation strategy {d.normalisation.strategy!r}; "
                f"use one of {KNOWN_NORMALISATION_STRATEGIES}"
            )
        if d.mask_variable not in (*d.prognostic, *d.forcing):
            raise ValueError(f"mask_variable {d.mask_variable!r} must be a prognostic or forcing variable")

        for name in ("time_axis", "train", "valid", "test"):
            _check_range(f"time.{name}", getattr(t, name))
        _check_range("data.normalisation.fit_period", d.normalisation.fit_period)
        if t.control is not None:
            _check_range("time.control", t.control)
            if t.control_repeats < 1:
                raise ValueError("time.control_repeats must be >= 1")
        axis_start, axis_end = (_month(x) for x in t.time_axis)
        for name, (start, end) in {
            "normalisation fit_period": d.normalisation.fit_period,
            "train": t.train,
            "valid": t.valid,
            "test": t.test,
        }.items():
            if _month(start) < axis_start or _month(end) > axis_end:
                raise ValueError(f"{name} {start}..{end} lies outside time_axis {t.time_axis}")

        if w.n_prior < 1:
            raise ValueError("window.n_prior must be >= 1")
        if not 1 <= w.rollout_steps <= w.posterior_steps:
            raise ValueError("window.rollout_steps must be between 1 and posterior_steps")
        if w.valid_rollout_steps is not None and not 1 <= w.valid_rollout_steps <= w.posterior_steps:
            raise ValueError("window.valid_rollout_steps must be between 1 and posterior_steps")
        if w.rollout_schedule is not None:
            if 0 not in w.rollout_schedule:
                raise ValueError("window.rollout_schedule must define epoch 0")
            for epoch, steps in w.rollout_schedule.items():
                if not 1 <= steps <= w.posterior_steps:
                    raise ValueError(f"rollout_schedule[{epoch}] = {steps} is outside 1..posterior_steps")

        unknown = set(lo.terms) - set(KNOWN_LOSS_TERMS)
        if unknown:
            raise ValueError(f"Unknown loss terms {sorted(unknown)}; use {KNOWN_LOSS_TERMS}")
        if not any(_is_active(v) for v in lo.terms.values()):
            raise ValueError("All loss weights are zero -- at least one loss term must be active")
        for name, weight in lo.terms.items():
            if isinstance(weight, (list, tuple)) and len(weight) != w.posterior_steps:
                raise ValueError(f"loss.terms[{name!r}] has {len(weight)} entries, expected posterior_steps")
        if lo.step_weights is not None:
            if len(lo.step_weights) != w.posterior_steps:
                raise ValueError("loss.step_weights must have posterior_steps entries")
            if sum(lo.step_weights) == 0:
                raise ValueError("loss.step_weights must not sum to zero")
        if _is_active(lo.terms.get("global_closure", 0)):
            if d.ohc_variable not in d.prognostic:
                raise ValueError("global_closure needs data.ohc_variable among the prognostic variables")
            if d.heat_flux_variable not in d.forcing:
                raise ValueError("global_closure needs data.heat_flux_variable among the forcing variables")

        if m.arch not in KNOWN_ARCHITECTURES:
            raise ValueError(f"Unknown arch {m.arch!r}; use {KNOWN_ARCHITECTURES}")
        if m.output_head not in KNOWN_OUTPUT_HEADS:
            raise ValueError(f"Unknown output_head {m.output_head!r}; use {KNOWN_OUTPUT_HEADS}")
        if m.latent_processor not in KNOWN_LATENT_PROCESSORS:
            raise ValueError(f"Unknown latent_processor {m.latent_processor!r}; use {KNOWN_LATENT_PROCESSORS}")
        if m.padding_mode not in ("zeros", "replicate", "reflect", "circular"):
            raise ValueError(f"Unknown padding_mode {m.padding_mode!r}")

        if tr.batch_size < 1 or tr.accumulate_grad_batches < 1:
            raise ValueError("train.batch_size and train.accumulate_grad_batches must be >= 1")
        if tr.lr_scheduler not in (None, "cosine"):
            raise ValueError("train.lr_scheduler must be None or 'cosine'")

    # --------------------------------------------------------------- utilities
    def to_dict(self):
        return dataclasses.asdict(self)

    def save(self, path):
        """Write the config as JSON (e.g. next to a checkpoint)."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str))

    def data_contract(self):
        """
        The settings that determine the cached normalised fields.

        Changing any of these builds a new cache. Windows, splits, the model and
        the losses are NOT part of it: they are applied to the cached fields
        at run time, so changing them never needs a rebuild.
        """
        d = self.data
        return {
            "path": str(Path(d.path).resolve()),
            "prognostic": list(d.prognostic),
            "forcing": list(d.forcing),
            "ohc_variable": d.ohc_variable,
            "heat_flux_variable": d.heat_flux_variable,
            "mask_variable": d.mask_variable,
            "area_variable": d.area_variable,
            "normalisation": dataclasses.asdict(d.normalisation),
            "time_axis": list(self.time.time_axis),
        }

    def data_hash(self, file_stamp=""):
        payload = json.dumps(self.data_contract(), sort_keys=True, default=str) + file_stamp
        return hashlib.sha1(payload.encode()).hexdigest()[:12]


def _is_active(weight):
    """True unless a loss weight is 0 (or a per-step list of zeros)."""
    if isinstance(weight, (list, tuple)):
        return any(w != 0 for w in weight)
    return weight != 0
