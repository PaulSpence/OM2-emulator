"""
Training: the Lightning module, the rollout-length schedule and the Trainer.

    module  = build_module(cfg, model, losses, data)
    trainer = build_trainer(cfg)
    trainer.fit(module, data.train_dl, data.valid_dl)
"""

import os
import subprocess
import warnings
from pathlib import Path

import lightning as L
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from Emulator import total_rollout_loss

from .config import rollout_spec
from .data import gather_windows
from .losses import closure_std_fields


def cuda_is_usable():
    """True if CUDA is available AND this PyTorch build supports the GPU."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(0)
    arches = torch.cuda.get_arch_list()
    return not arches or f"sm_{major}{minor}" in arches


def check_gpu():
    """
    Warn if other processes hold memory on the GPU, and return them.

    A second kernel left running on the same GPU (e.g. a finished notebook)
    keeps its memory reserved and competes for compute: it caused both slow
    training and CUDA out-of-memory errors before.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    others = [line.strip() for line in out.splitlines() if line.strip() and line.split(",")[0].strip() != str(os.getpid())]
    if others:
        warnings.warn(
            "Other processes are using this GPU (pid, memory): "
            f"{others}. Shut down other kernels before training, or expect "
            "slower training and possible CUDA out-of-memory errors."
        )
    return others


class PerVariableError:
    """
    A zero-valued "loss" that accumulates the area-weighted squared error of
    every prognostic variable (in normalised units) over a rollout, so the
    module can report a per-variable RMSE each epoch without changing the loss.
    """

    def __init__(self, n_prognostic):
        self.n_prognostic = n_prognostic
        self.reset()

    def reset(self):
        self.squared_error = None  # (P,) sum of weighted squared error
        self.weight = 0.0          # sum of the weights (one per sample and step)

    def __call__(self, *, pred_t, target_t, mask, area, **_):
        with torch.no_grad():
            weight = (area * mask).float()
            weight = weight / weight.sum()
            error = ((pred_t.float() - target_t.float()) ** 2 * weight).sum(dim=(0, -2, -1))
            self.squared_error = error if self.squared_error is None else self.squared_error + error
            self.weight += pred_t.shape[0]
        return pred_t.new_zeros(())

    def pop_rmse(self):
        """Per-variable RMSE since the last reset, or None; then reset."""
        if self.squared_error is None:
            return None
        rmse = torch.sqrt(self.squared_error / self.weight).cpu()
        self.reset()
        return rmse


class EmulatorModule(L.LightningModule):
    """
    Autoregressive rollout training for a ForwardEmulator.

    Each batch is a (B,) tensor of initial months from the dataloaders. The
    normalised fields live on the module's device (non-persistent buffers, so
    they stay out of checkpoints), and each batch's windows are cut there with
    gather_windows: only the rollout steps in use, with no CPU work and no
    host-to-device copy. A ready-made window dict (gather_windows' output) is
    accepted too. The loss is Emulator.total_rollout_loss over the configured
    loss terms, which feeds every prediction back in as the newest prior state.
    A [n_free, n_trained] rollout first runs n_free steps without gradients
    (push_forward) and scores only the n_trained steps after them.
    """

    def __init__(self, model, losses, data, cfg):
        super().__init__()
        self.model = model
        # The compiled wrapper shares self.model's parameters. It is kept out
        # of the module tree so state_dict / checkpoints keep plain key names.
        self.__dict__["_run_model"] = torch.compile(model) if cfg.train.compile_model else model
        self.checkpoint_steps = cfg.train.checkpoint_rollout_steps
        self.steps_in_memory = cfg.train.rollout_steps_in_memory
        self.losses = list(losses)
        self.n_prognostic = data.n_prognostic
        self.prognostic_names = list(data.fields["prognostic_names"])
        # Per-variable RMSE (normalised units) per epoch and stage; see rmse_history.
        self.errors = {stage: PerVariableError(self.n_prognostic) for stage in ("train", "val")}
        self.history = []
        w, tr = cfg.window, cfg.train
        self.n_prior = w.n_prior
        # (n_free, n_trained); train_steps is updated by RolloutSchedule, if used.
        self.train_steps = rollout_spec(w.rollout_steps)
        self.valid_steps = (0, w.valid_rollout_steps or sum(self.train_steps))
        self.dt_seconds = cfg.loss.seconds_per_step
        self.lr, self.weight_decay = tr.lr, tr.weight_decay
        self.lr_scheduler, self.max_epochs = tr.lr_scheduler, tr.max_epochs

        f = data.fields
        # The normalised fields, moved to the GPU with the module (~1 GB per
        # four variables at 1 degree). Not saved in checkpoints.
        self.register_buffer("prognostic_z", f["prognostic"].float(), persistent=False)
        self.register_buffer("forcing_z", f["forcing"].float(), persistent=False)
        self.register_buffer("mask", f["mask"].float())
        self.register_buffer("area", f["area"].float())
        # Normalisation std fields that the closure terms use to turn z-scores
        # into physical anomalies: (T, H, W) content and flux std per active
        # budget. Only active budgets are kept, as each pair costs GPU memory.
        self.budgets = []
        for budget, (content_std, flux_std) in closure_std_fields(cfg, data).items():
            self.register_buffer(f"{budget}_content_std", content_std.float())
            self.register_buffer(f"{budget}_flux_std", flux_std.float())
            self.budgets.append(budget)

    @property
    def closure_std(self):
        """{budget: (content_std, flux_std)} on the module's device."""
        return {b: (getattr(self, f"{b}_content_std"), getattr(self, f"{b}_flux_std")) for b in self.budgets}

    def windows(self, t0, n_steps):
        """Rollout windows for initial months t0, cut on the module's device."""
        return gather_windows(self.prognostic_z, self.forcing_z, t0, self.n_prior, n_steps)

    def push_forward(self, t0, n_free, n_steps):
        """
        Run n_free steps from the true states at initial months t0 WITHOUT
        gradients (see WindowConfig.rollout_steps), then return the windows of
        the next n_steps months (gather_windows layout) starting from the
        model's own state: prior = the last n_prior states of the free run
        (true states while n_free < n_prior).

        Each free step reads its forcing month straight from the fields, and
        windows are cut only for the n_steps scored months, so a [24, 12]
        rollout holds 12 months of targets and forcing per sample, not 36.
        """
        t0 = torch.as_tensor(t0, dtype=torch.long, device=self.prognostic_z.device)
        # (B, n_prior, P, H, W) -> (B, n_prior * P, H, W), oldest state first
        prior = self.windows(t0, 0)["prior"].flatten(1, 2)
        with torch.no_grad():
            for k in range(1, n_free + 1):
                pred = self._run_model(prior, self.forcing_z[t0 + k], self.mask)
                prior = torch.cat([prior[:, self.n_prognostic:], pred.to(prior.dtype)], dim=1)
        batch = self.windows(t0 + n_free, n_steps)
        batch["prior"] = prior.unflatten(1, (self.n_prior, self.n_prognostic))
        return batch

    def _step(self, batch, steps, stage):
        n_free, n_steps = rollout_spec(steps)  # n or (n_free, n_trained)
        if isinstance(batch, dict):
            if n_free:
                raise ValueError("A [n_free, n_trained] rollout needs initial months, not ready-made windows")
        elif n_free:
            batch = self.push_forward(batch, n_free, n_steps)
        else:
            batch = self.windows(batch, n_steps)
        loss = total_rollout_loss(
            model=self._run_model,
            # (B, n_prior, P, H, W) -> (B, n_prior * P, H, W), oldest state first
            initial_prior_states=batch["prior"].flatten(1, 2),
            forcing_sequence=batch["forcing"],
            target_sequence=batch["target"],
            mask=self.mask,
            losses=[*self.losses, self.errors[stage]],
            n_steps=n_steps,
            target_time_indices=batch["target_time_index"],
            area=self.area,
            closure_std=self.closure_std,
            dt_seconds=self.dt_seconds,
            initial_forcing=batch["initial_forcing"],
            n_prognostic=self.n_prognostic,
            checkpoint_steps=self.checkpoint_steps,
            steps_in_memory=self.steps_in_memory,
        )
        batch_size = batch["prior"].shape[0]
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)
        for term in self.losses:
            value = term.pop_running()
            if value is not None:
                self.log(f"{stage}_{term.name}", value / n_steps, on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

    def _log_rmse(self, stage):
        rmse = self.errors[stage].pop_rmse()
        if rmse is None or self.trainer.sanity_checking:
            return
        for name, value in zip(self.prognostic_names, rmse.tolist()):
            self.log(f"{stage}_rmse_{name}", value)
            self.history.append({"epoch": self.current_epoch, "stage": stage, "variable": name, "rmse": value})

    def on_train_epoch_end(self):
        self._log_rmse("train")

    def on_validation_epoch_end(self):
        self._log_rmse("val")

    @property
    def rmse_history(self):
        """Per-variable RMSE (normalised units): DataFrame of epoch, stage, variable, rmse."""
        return pd.DataFrame(self.history, columns=["epoch", "stage", "variable", "rmse"])

    def training_step(self, batch, batch_idx):
        return self._step(batch, self.train_steps, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, self.valid_steps, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer


class RolloutSchedule(Callback):
    """
    Set the training rollout from {first_epoch: steps} at the start of every
    epoch; steps is n or [n_free, n_trained] (WindowConfig.rollout_steps).
    """

    def __init__(self, schedule):
        self.schedule = {int(e): rollout_spec(s) for e, s in sorted(schedule.items(), key=lambda x: int(x[0]))}

    def steps_for(self, epoch):
        """(n_free, n_trained) for this epoch."""
        return self.schedule[max(e for e in self.schedule if e <= epoch)]

    def on_train_epoch_start(self, trainer, pl_module):
        steps = self.steps_for(trainer.current_epoch)
        if steps != pl_module.train_steps:
            n_free, n_trained = steps
            free = f"{n_free} free-running (no gradient) + " if n_free else ""
            print(f"Epoch {trainer.current_epoch}: training rollout -> {free}{n_trained} trained steps")
        pl_module.train_steps = steps


def build_module(cfg, model, losses, data):
    return EmulatorModule(model, losses, data, cfg)


def build_trainer(cfg):
    """A Lightning Trainer configured from cfg.train (plus the rollout schedule)."""
    tr = cfg.train
    use_gpu = cuda_is_usable()
    if torch.cuda.is_available() and not use_gpu:
        warnings.warn("CUDA is visible but this PyTorch build does not support the GPU; training on CPU.")
    if use_gpu and tr.gpu_check:
        check_gpu()

    callbacks = []
    if cfg.window.rollout_schedule is not None:
        callbacks.append(RolloutSchedule(cfg.window.rollout_schedule))

    logger, checkpointing = False, False
    if tr.run_dir is not None:
        run_dir = Path(tr.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg.save(run_dir / "config.json")
        logger = CSVLogger(save_dir=str(run_dir), name="", version="")
        callbacks.append(ModelCheckpoint(dirpath=run_dir / "checkpoints", monitor="val_loss", save_last=True))
        checkpointing = True

    if use_gpu:
        torch.cuda.empty_cache()

    # Batches are B integers (windows are cut on the GPU), so Lightning's
    # advice to add DataLoader workers does not apply.
    warnings.filterwarnings("ignore", message=".*does not have many workers.*")

    return L.Trainer(
        max_epochs=tr.max_epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        precision=tr.precision,
        accumulate_grad_batches=tr.accumulate_grad_batches,
        gradient_clip_val=tr.gradient_clip_val,
        check_val_every_n_epoch=tr.check_val_every_n_epoch,
        num_sanity_val_steps=0,
        # Input shapes are fixed, so let cuDNN pick the fastest kernels once.
        benchmark=True,
        logger=logger,
        enable_checkpointing=checkpointing,
        enable_model_summary=False,
        callbacks=callbacks,
    )
