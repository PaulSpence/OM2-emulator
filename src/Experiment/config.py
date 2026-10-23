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
# Fitting terms. Every budget in LossConfig.closures adds a "<budget>_closure" term.
KNOWN_LOSS_TERMS = ("local_mse", "spectral")
CLOSURE_SUFFIX = "_closure"


def _month(value):
    """Parse a "YYYY-MM" string to a pandas month period, with a clear error."""
    try:
        return pd.Period(value, freq="M")
    except (ValueError, TypeError) as err:
        raise ValueError(f"Expected a 'YYYY-MM' date, got {value!r}") from err


def rollout_spec(steps, name="rollout_steps"):
    """
    A rollout length as (n_free, n_trained): n -> (0, n); [n_free, n_trained]
    -> (n_free, n_trained). See WindowConfig.rollout_steps.
    """
    if isinstance(steps, (list, tuple)):
        if len(steps) != 2:
            raise ValueError(f"{name} = {steps}: use n or [n_free, n_trained]")
        n_free, n_trained = steps
    else:
        n_free, n_trained = 0, steps
    if not (isinstance(n_free, int) and isinstance(n_trained, int)) or n_free < 0 or n_trained < 1:
        raise ValueError(f"{name} = {steps}: need integers n_free >= 0 and n_trained >= 1")
    return n_free, n_trained


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
    # Land boundary condition per variable, applied to the model's input
    # channels (see "Land boundary conditions" in src/Emulator/om2_model_utils.py
    # for why). Variables not listed are Neumann: land takes the nearest ocean
    # value (no flux across the coast), right for tracers such as heat and
    # freshwater content and neutral for the rest. Set velocity components to
    # "dirichlet" (land held at z = 0, i.e. no normal flow), e.g.
    #   {"u": "dirichlet", "v": "dirichlet"}
    # or give a value in z-score units: {"u": ["dirichlet", 0.0]}.
    # Hidden layers always use the Neumann fill.
    boundary_conditions: dict = field(default_factory=dict)
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
    # Training windows must also END inside `train` (targets <= train[1]), so
    # training never sees the validation or test months; initial months too
    # late for a full posterior_steps window are dropped.
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
    # Autoregressive steps per training sample: n (all trained), or
    # [n_free, n_trained] ("pushforward", Brandstetter, Worrall & Welling 2022,
    # ICLR, "Message Passing Neural PDE Solvers"): n_free steps run WITHOUT
    # gradients from the true initial state, then the loss is taken over the
    # next n_trained steps. The model is trained on its own drifted states at
    # leads n_free+1 .. n_free+n_trained (where long-rollout damping and noise
    # show up), for the cost of n_free forward passes with no stored
    # activations and no backward pass. n_free + n_trained <= posterior_steps.
    rollout_steps: int | list = 12
    # Optional curriculum: {first_epoch: rollout_steps}, each entry as above,
    # e.g. {0: 1, 5: 4, 10: 12, 15: [24, 12]}. Training cost scales linearly
    # with the trained steps. None = always rollout_steps.
    rollout_schedule: dict | None = None
    # Steps used for validation (all scored). None = all steps of rollout_steps.
    # Keep it fixed so val_loss stays comparable across epochs when a schedule
    # is used.
    valid_rollout_steps: int | None = None


@dataclass
class ClosureConfig:
    """
    One global budget for a "<budget>_closure" loss term: the area-integrated
    change of a content (prognostic) must match the time-integrated surface
    flux (forcing). The flux must be in content units per second.
    """
    content_variable: str
    flux_variable: str
    surface_flux_sign: float = 1.0  # +1 if positive flux means INTO the ocean
    min_scale: float = 1.0          # floor on the residual's normalisation scale (content units x m^2)


def _default_closures():
    return {
        # OHC (J/m^2) vs surface heat flux incl. frazil (W/m^2); min_scale in J.
        "heat": ClosureConfig("ocean_heat_content_2d", "total_surface_heat_flx", 1.0, 1.0e20),
        # Freshwater content rel. to 35 psu (kg/m^2) vs P-E+R-ice salt flux (kg/m^2/s); min_scale in kg.
        "freshwater": ClosureConfig("ocean_freshwater_content_2d", "ocean_freshwater_flux", 1.0, 1.0e12),
    }


@dataclass
class LossConfig:
    # Loss terms and their weights, summed at every rollout step and averaged
    # over the steps. A weight of 0 removes the term entirely.
    #   local_mse      -- ocean-masked MSE in z-score space (primary fit)
    #   spectral       -- squared log10 difference in energy per band of physical zonal
    #                     wavenumber: penalises lost large-scale energy (blurring) and
    #                     excess small-scale energy (noise growth), phase-free
    #   <budget>_closure -- global budget closure on physical anomalies, one
    #                     per entry in `closures` (heat_closure, freshwater_closure);
    #                     see issue #43: the true data score ~0.5 on heat_closure
    # A weight may also be a list with one ABSOLUTE weight per rollout step.
    terms: dict = field(
        default_factory=lambda: {"local_mse": 1.0, "spectral": 0.0, "heat_closure": 0.1, "freshwater_closure": 0.0}
    )
    # Optional relative emphasis per rollout step, applied to every term and
    # rescaled to mean 1 (so it changes the balance across lead times, not the
    # overall size), e.g. [1] * 6 + [2] * 6. Length must be posterior_steps.
    # Per-step entries (here and in `terms`) index the SCORED steps: with a
    # [n_free, n_trained] rollout, entry 0 is the first trained step.
    step_weights: list | None = None
    # The budgets available as "<budget>_closure" terms: {budget: ClosureConfig}.
    closures: dict = field(default_factory=_default_closures)
    seconds_per_step: float = 30 * 24 * 60 * 60  # one month


@dataclass
class ModelConfig:
    arch: str = "unet"
    # North/south edge padding of every partial convolution (longitude is
    # always periodic; see PartialConv2d).
    padding_mode: str = "zeros"
    # UNet channel widths scale with the number of input channels
    # (n_prior * prognostic + forcing): width_multiplier x inputs at 1/2
    # resolution, doubling at each deeper level. >= 2 lets the first ReLU layer
    # carry every signed input; memory grows ~linearly with it.
    width_multiplier: float = 4
    # NHWC memory layout: faster fp16 convolutions on the V100's tensor cores
    # (use with train.precision="16-mixed"). Same results.
    channels_last: bool = False
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
    # Recompute each rollout step's activations in the backward pass instead
    # of storing all of them: peak memory ~1 step instead of rollout_steps
    # steps, for ~30% more compute. Lets wide models train with large batches.
    checkpoint_rollout_steps: bool = False
    # With checkpointing on: keep the activations of up to this many rollout
    # steps (no recompute for them) and checkpoint the rest. Peak memory ~
    # (this + 1) steps; each kept step saves ~1/4 of a step's compute. Raise it
    # until training runs out of memory, then step back. 0 = checkpoint all.
    rollout_steps_in_memory: int = 0
    # torch.compile the model (fuses the elementwise mask/ReLU operations).
    # Experimental: the first steps are slow while it compiles, and a smaller
    # last batch triggers one recompile.
    compile_model: bool = False
    # Lightning precision: "32-true", or "16-mixed" (fp16, ~1.5x faster on a
    # V100). Predictions are cast back to float32 before the losses.
    precision: str = "32-true"
    seed: int = 42
    # DataLoader workers. Batches are only initial-month indices (the windows
    # are cut on the GPU), so 0 is right here.
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
    # Colour limits of the anomaly / difference maps: a number for every
    # variable, {variable: number}, or None. Variables without a number get
    # the 99th percentile of |truth anomaly| / |difference|.
    anomaly_scale: object = None
    difference_scale: object = None


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
        for name, condition in d.boundary_conditions.items():
            if name not in (*d.prognostic, *d.forcing):
                raise ValueError(f"data.boundary_conditions: {name!r} is not a prognostic or forcing variable")
            _dirichlet_value(name, condition)  # raises on an invalid condition
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
        n_free, n_trained = rollout_spec(w.rollout_steps, "window.rollout_steps")
        if n_free + n_trained > w.posterior_steps:
            raise ValueError("window.rollout_steps must total at most posterior_steps")
        if w.valid_rollout_steps is not None and not 1 <= w.valid_rollout_steps <= w.posterior_steps:
            raise ValueError("window.valid_rollout_steps must be between 1 and posterior_steps")
        if w.rollout_schedule is not None:
            if 0 not in w.rollout_schedule:
                raise ValueError("window.rollout_schedule must define epoch 0")
            for epoch, steps in w.rollout_schedule.items():
                if sum(rollout_spec(steps, f"rollout_schedule[{epoch}]")) > w.posterior_steps:
                    raise ValueError(f"rollout_schedule[{epoch}] = {steps} totals more than posterior_steps")

        if "global_closure" in lo.terms:
            raise ValueError("loss.terms['global_closure'] is now 'heat_closure' (see loss.closures)")
        known = (*KNOWN_LOSS_TERMS, *(b + CLOSURE_SUFFIX for b in lo.closures))
        unknown = set(lo.terms) - set(known)
        if unknown:
            raise ValueError(f"Unknown loss terms {sorted(unknown)}; use {known}")
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
        for budget, closure in self.active_closures().items():
            if closure.content_variable not in d.prognostic:
                raise ValueError(
                    f"{budget}{CLOSURE_SUFFIX} needs {closure.content_variable!r} among the prognostic variables"
                )
            if closure.flux_variable not in d.forcing:
                raise ValueError(
                    f"{budget}{CLOSURE_SUFFIX} needs {closure.flux_variable!r} among the forcing variables"
                )

        if m.arch not in KNOWN_ARCHITECTURES:
            raise ValueError(f"Unknown arch {m.arch!r}; use {KNOWN_ARCHITECTURES}")
        if m.output_head not in KNOWN_OUTPUT_HEADS:
            raise ValueError(f"Unknown output_head {m.output_head!r}; use {KNOWN_OUTPUT_HEADS}")
        if m.latent_processor not in KNOWN_LATENT_PROCESSORS:
            raise ValueError(f"Unknown latent_processor {m.latent_processor!r}; use {KNOWN_LATENT_PROCESSORS}")
        if m.width_multiplier < 1:
            raise ValueError("model.width_multiplier must be >= 1")
        if m.padding_mode not in ("zeros", "replicate", "reflect", "circular"):
            raise ValueError(f"Unknown padding_mode {m.padding_mode!r}")

        if tr.batch_size < 1 or tr.accumulate_grad_batches < 1:
            raise ValueError("train.batch_size and train.accumulate_grad_batches must be >= 1")
        if tr.rollout_steps_in_memory < 0:
            raise ValueError("train.rollout_steps_in_memory must be >= 0")
        if tr.lr_scheduler not in (None, "cosine"):
            raise ValueError("train.lr_scheduler must be None or 'cosine'")

    # --------------------------------------------------------------- utilities
    def active_closures(self):
        """{budget: ClosureConfig} for every "<budget>_closure" term with a non-zero weight."""
        lo = self.loss
        return {
            budget: closure
            for budget, closure in lo.closures.items()
            if _is_active(lo.terms.get(budget + CLOSURE_SUFFIX, 0))
        }

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
            "mask_variable": d.mask_variable,
            "area_variable": d.area_variable,
            "normalisation": dataclasses.asdict(d.normalisation),
            "time_axis": list(self.time.time_axis),
        }

    def data_hash(self, file_stamp=""):
        payload = json.dumps(self.data_contract(), sort_keys=True, default=str) + file_stamp
        return hashlib.sha1(payload.encode()).hexdigest()[:12]


def _dirichlet_value(name, condition):
    """The Dirichlet value (z-score units) for a boundary condition, or None for Neumann."""
    if condition == "neumann":
        return None
    if condition == "dirichlet":
        return 0.0
    if isinstance(condition, (list, tuple)) and len(condition) == 2 and condition[0] == "dirichlet":
        return float(condition[1])
    raise ValueError(
        f"data.boundary_conditions[{name!r}] = {condition!r}: use 'neumann', 'dirichlet' "
        "or ['dirichlet', value]"
    )


def _is_active(weight):
    """True unless a loss weight is 0 (or a per-step list of zeros)."""
    if isinstance(weight, (list, tuple)):
        return any(w != 0 for w in weight)
    return weight != 0
