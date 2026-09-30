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
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from Emulator import total_rollout_loss


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


class EmulatorModule(L.LightningModule):
    """
    Autoregressive rollout training for a ForwardEmulator.

    Each batch is a dict from RolloutWindowDataset. The loss is
    Emulator.total_rollout_loss over the configured loss terms, which feeds
    every prediction back in as the newest prior state.
    """

    def __init__(self, model, losses, data, cfg):
        super().__init__()
        self.model = model
        self.losses = list(losses)
        self.n_prognostic = data.n_prognostic
        w, tr = cfg.window, cfg.train
        self.train_steps = w.rollout_steps  # updated by RolloutSchedule, if used
        self.valid_steps = w.valid_rollout_steps or w.rollout_steps
        self.dt_seconds = cfg.loss.seconds_per_step
        self.lr, self.weight_decay = tr.lr, tr.weight_decay
        self.lr_scheduler, self.max_epochs = tr.lr_scheduler, tr.max_epochs

        f, d = data.fields, cfg.data
        self.register_buffer("mask", f["mask"].float())
        self.register_buffer("area", f["area"].float())
        # Normalisation std fields used by the closure term to turn z-scores
        # into physical anomalies. Placeholders of 1 if the closure is unused.
        ones = torch.ones_like(f["prognostic"][:, 0])
        ohc_std = f["prognostic_std"][:, d.prognostic.index(d.ohc_variable)] if d.ohc_variable in d.prognostic else ones
        heat_flux_std = f["heat_flux_std"] if f["heat_flux_std"] is not None else ones
        self.register_buffer("ohc_std", ohc_std.float())
        self.register_buffer("heat_flux_std", heat_flux_std.float())

    def _step(self, batch, n_steps, stage):
        loss = total_rollout_loss(
            model=self.model,
            # (B, n_prior, P, H, W) -> (B, n_prior * P, H, W), oldest state first
            initial_prior_states=batch["prior"].flatten(1, 2),
            forcing_sequence=batch["forcing"],
            target_sequence=batch["target"],
            mask=self.mask,
            losses=self.losses,
            n_steps=n_steps,
            target_time_indices=batch["target_time_index"],
            area=self.area,
            ohc_std=self.ohc_std,
            forcing_std=self.heat_flux_std,
            dt_seconds=self.dt_seconds,
            initial_forcing=batch["initial_forcing"],
            n_prognostic=self.n_prognostic,
        )
        batch_size = batch["prior"].shape[0]
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)
        for term in self.losses:
            value = term.pop_running()
            if value is not None:
                self.log(f"{stage}_{term.name}", value / n_steps, on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

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
    """Set the training rollout length from {first_epoch: steps} at the start of every epoch."""

    def __init__(self, schedule):
        self.schedule = dict(sorted(schedule.items()))

    def steps_for(self, epoch):
        return self.schedule[max(e for e in self.schedule if e <= epoch)]

    def on_train_epoch_start(self, trainer, pl_module):
        steps = self.steps_for(trainer.current_epoch)
        if steps != pl_module.train_steps:
            print(f"Epoch {trainer.current_epoch}: training rollout length -> {steps} steps")
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

    return L.Trainer(
        max_epochs=tr.max_epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        precision=tr.precision,
        accumulate_grad_batches=tr.accumulate_grad_batches,
        gradient_clip_val=tr.gradient_clip_val,
        check_val_every_n_epoch=tr.check_val_every_n_epoch,
        num_sanity_val_steps=0,
        logger=logger,
        enable_checkpointing=checkpointing,
        enable_model_summary=False,
        callbacks=callbacks,
    )
