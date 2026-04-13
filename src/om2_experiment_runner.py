"""Top-level experiment orchestration for OM2 emulator.

This module ties together config loading, data pipeline setup, model training,
inference diagnostics, and artifact persistence. It is the engine behind the
CLI scripts so each run has a standardized output structure.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from om2_experiment_config import dump_yaml_config, load_yaml_config, resolve_config
from om2_experiment_data import build_pipeline, load_mask_and_area, take_pipeline_sample
from om2_experiment_diagnostics import (
    append_summary_row,
    compute_area_weighted_rmse_by_channel,
    plot_latent_feature_maps,
    plot_reconstruction_maps,
    run_inference_sample,
    save_metrics_json,
)
from om2_experiment_model import build_model_from_config
from om2_experiment_training import train_model


def resolve_device(device_name: str) -> torch.device:
    """Resolve runtime torch device from config.

    Parameters
    ----------
    device_name
        Device string from config. Supports ``"auto"`` plus explicit torch
        device strings (for example ``"cpu"`` or ``"cuda:0"``).

    Returns
    -------
    torch.device
        Resolved torch device.
    """
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return torch.device(device_name)


def set_global_seed(seed: int) -> None:
    """Set random seed for deterministic torch and numpy behavior."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_directory(run_config: dict[str, Any]) -> Path:
    """Create a timestamped run directory.

    Parameters
    ----------
    run_config
        Resolved ``run`` config section.

    Returns
    -------
    Path
        Newly-created run directory path.
    """
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_name = run_config["name"]
    output_root = Path(run_config["output_root"]).expanduser().resolve()
    run_dir = output_root / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def run_experiment_from_config_dict(config: dict[str, Any]) -> dict[str, Any]:
    """Execute one experiment run from an already-resolved config.

    Parameters
    ----------
    config
        Resolved experiment configuration dictionary.

    Returns
    -------
    dict[str, Any]
        Summary object with run directory and scalar metrics.
    """
    run_dir = make_run_directory(config["run"])

    set_global_seed(int(config["run"]["seed"]))
    device = resolve_device(str(config["run"]["device"]))

    # Build data access once so train and diagnostics share identical preprocessing.
    pipeline = build_pipeline(config["data"])
    mask, area_weighted_mask = load_mask_and_area(config["data"])
    mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    model = build_model_from_config(config["model"]).to(device)

    training_summary = train_model(
        model=model,
        pipeline=pipeline,
        mask_tensor=mask_tensor,
        training_config=config["training"],
        normalization_config=config["normalization"],
        device=device,
    )

    sample = take_pipeline_sample(
        pipeline,
        sample_index=int(config["diagnostics"]["sample_index"]),
    )

    inference = run_inference_sample(
        model=model,
        sample=sample,
        mask_tensor=mask_tensor,
        normalization_config=config["normalization"],
        device=device,
    )

    rmse_metrics = compute_area_weighted_rmse_by_channel(
        inference["prediction_physical"],
        inference["target_physical"],
        area_weighted_mask,
        config["diagnostics"]["channel_names"],
    )

    diagnostics_dir = run_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    plot_reconstruction_maps(
        prediction_physical=inference["prediction_physical"],
        target_physical=inference["target_physical"],
        diagnostics_config=config["diagnostics"],
        rmse_metrics=rmse_metrics,
        output_path=diagnostics_dir / "reconstruction_maps.png",
    )

    plot_latent_feature_maps(
        latent_tensor=inference["latent"],
        diagnostics_config=config["diagnostics"],
        output_path=diagnostics_dir / "latent_feature_maps.png",
    )

    if bool(config["run"].get("save_arrays", True)):
        np.savez_compressed(
            run_dir / "arrays.npz",
            prediction_physical=inference["prediction_physical"],
            target_physical=inference["target_physical"],
            latent=inference["latent"],
            latent_mask=inference["latent_mask"],
            output_mask=inference["output_mask"],
            area_weighted_mask=area_weighted_mask,
            mask=mask,
        )

    torch.save(model.state_dict(), run_dir / "model_state.pt")
    dump_yaml_config(config, run_dir / "resolved_config.yaml")

    metrics = {
        "run_name": config["run"]["name"],
        "run_dir": str(run_dir),
        "device": str(device),
        "final_epoch_loss": float(training_summary["epoch_losses"][-1]),
        "total_steps": int(training_summary["total_steps"]),
        **rmse_metrics,
    }

    save_metrics_json(metrics, run_dir / "metrics.json")
    save_metrics_json(training_summary, run_dir / "training_summary.json")

    append_summary_row(
        Path(config["run"]["output_root"]).expanduser().resolve() / "summary.csv",
        metrics,
    )

    return metrics


def run_experiment_from_yaml(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config, resolve defaults, and execute one run.

    Parameters
    ----------
    config_path
        Path to user config YAML.

    Returns
    -------
    dict[str, Any]
        Run metrics summary.
    """
    user_config = load_yaml_config(config_path)
    resolved_config = resolve_config(user_config)
    return run_experiment_from_config_dict(resolved_config)
