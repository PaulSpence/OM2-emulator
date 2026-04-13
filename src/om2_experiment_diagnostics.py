"""Diagnostics and artifact writing for OM2 emulator experiments.

The functions in this module produce the exact comparison products we care
about for scientific model iteration: area-weighted RMSE, prediction/actual/
error maps, and latent feature-map visualizations.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from om2_experiment_normalization import denormalize_tensor, normalize_tensor


def run_inference_sample(
    model: torch.nn.Module,
    sample: np.ndarray,
    mask_tensor: torch.Tensor,
    normalization_config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Run inference for one sample and capture physical-space outputs.

    Parameters
    ----------
    model
        Trained model.
    sample
        Numpy sample with shape ``(1, channels, h, w)``.
    mask_tensor
        Binary support mask ``(1, 1, h, w)``.
    normalization_config
        Resolved normalization config section.
    device
        Active torch device.

    Returns
    -------
    dict[str, Any]
        Dictionary containing normalized and physical tensors plus features.
    """
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(sample)).float().to(device)
        x_normalized, stats = normalize_tensor(x, mask_tensor, normalization_config)

        prediction_normalized, features = model(
            x_normalized,
            mask_tensor,
            return_features=True,
        )

        prediction_physical = denormalize_tensor(prediction_normalized, stats)
        target_physical = denormalize_tensor(x_normalized, stats)

    return {
        "prediction_normalized": prediction_normalized.detach().cpu().numpy(),
        "target_normalized": x_normalized.detach().cpu().numpy(),
        "prediction_physical": prediction_physical.detach().cpu().numpy(),
        "target_physical": target_physical.detach().cpu().numpy(),
        "latent": features["latent"].detach().cpu().numpy(),
        "latent_mask": features["latent_mask"].detach().cpu().numpy(),
        "output_mask": features["output_mask"].detach().cpu().numpy(),
    }


def compute_area_weighted_rmse_by_channel(
    prediction_physical: np.ndarray,
    target_physical: np.ndarray,
    area_weighted_mask: np.ndarray,
    channel_names: list[str],
) -> dict[str, float]:
    """Compute area-weighted RMSE for each channel.

    Parameters
    ----------
    prediction_physical
        Predicted tensor in physical units with shape ``(1, c, h, w)``.
    target_physical
        Reference tensor in physical units with shape ``(1, c, h, w)``.
    area_weighted_mask
        Area weights multiplied by ocean mask, shape ``(h, w)``.
    channel_names
        Names for each channel index.

    Returns
    -------
    dict[str, float]
        Channel name -> RMSE value.
    """
    metrics: dict[str, float] = {}
    for channel_index, channel_name in enumerate(channel_names):
        error = prediction_physical[0, channel_index] - target_physical[0, channel_index]
        mse = np.nansum((error**2) * area_weighted_mask) / np.nansum(area_weighted_mask)
        metrics[f"rmse_{channel_name}"] = float(np.sqrt(mse))

    return metrics


def plot_reconstruction_maps(
    prediction_physical: np.ndarray,
    target_physical: np.ndarray,
    diagnostics_config: dict[str, Any],
    rmse_metrics: dict[str, float],
    output_path: str | Path,
) -> None:
    """Plot prediction/target/error maps for heat flux and OHC channels.

    Parameters
    ----------
    prediction_physical
        Predicted tensor in physical units.
    target_physical
        Target tensor in physical units.
    diagnostics_config
        Resolved diagnostics config section.
    rmse_metrics
        RMSE metrics dictionary for title annotations.
    output_path
        Path to save the PNG figure.
    """
    flux_index = int(diagnostics_config["flux_channel_index"])
    ohc_index = int(diagnostics_config["ohc_channel_index"])
    flux_scale = float(diagnostics_config.get("flux_scale", 1.0))

    fig = plt.figure(figsize=(15, 10))
    grid = fig.add_gridspec(2, 3, wspace=0.25, hspace=0.25)

    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(3)]
    ax1, ax2, ax3, ax4, ax5, ax6 = axes

    flux_prediction = prediction_physical[0, flux_index] * flux_scale
    flux_target = target_physical[0, flux_index] * flux_scale
    flux_error = flux_prediction - flux_target

    ohc_prediction = prediction_physical[0, ohc_index]
    ohc_target = target_physical[0, ohc_index]
    ohc_error = ohc_prediction - ohc_target

    im_flux = ax1.imshow(
        flux_prediction,
        vmin=float(diagnostics_config["flux_vmin"]),
        vmax=float(diagnostics_config["flux_vmax"]),
        cmap="bwr",
    )
    ax2.imshow(
        flux_target,
        vmin=float(diagnostics_config["flux_vmin"]),
        vmax=float(diagnostics_config["flux_vmax"]),
        cmap="bwr",
    )
    ax3.imshow(
        flux_error,
        vmin=float(diagnostics_config["flux_vmin"]),
        vmax=float(diagnostics_config["flux_vmax"]),
        cmap="bwr",
    )

    for axis in (ax1, ax2, ax3):
        axis.invert_yaxis()

    im_ohc = ax4.imshow(
        ohc_prediction,
        vmin=float(diagnostics_config["ohc_vmin"]),
        vmax=float(diagnostics_config["ohc_vmax"]),
        cmap="bwr",
    )
    ax5.imshow(
        ohc_target,
        vmin=float(diagnostics_config["ohc_vmin"]),
        vmax=float(diagnostics_config["ohc_vmax"]),
        cmap="bwr",
    )
    ax6.imshow(
        ohc_error,
        vmin=float(diagnostics_config["ohc_vmin"]),
        vmax=float(diagnostics_config["ohc_vmax"]),
        cmap="bwr",
    )

    for axis in (ax4, ax5, ax6):
        axis.invert_yaxis()

    channel_names = diagnostics_config["channel_names"]
    flux_name = channel_names[flux_index]
    ohc_name = channel_names[ohc_index]

    ax1.set_title("Prediction")
    ax2.set_title("Actual")
    ax3.set_title(f"Error RMSE: {rmse_metrics[f'rmse_{flux_name}']:.3e}")

    ax4.set_title("Prediction")
    ax5.set_title("Actual")
    ax6.set_title(f"Error RMSE: {rmse_metrics[f'rmse_{ohc_name}']:.3e}")

    ax1.set_ylabel(flux_name)
    ax4.set_ylabel(ohc_name)

    colorbar_flux = fig.colorbar(
        im_flux,
        ax=[ax1, ax2, ax3],
        orientation="vertical",
        fraction=0.025,
        pad=0.02,
    )
    colorbar_flux.set_label(flux_name)

    colorbar_ohc = fig.colorbar(
        im_ohc,
        ax=[ax4, ax5, ax6],
        orientation="vertical",
        fraction=0.025,
        pad=0.02,
    )
    colorbar_ohc.set_label(ohc_name)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_latent_feature_maps(
    latent_tensor: np.ndarray,
    diagnostics_config: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Plot selected latent channels as feature maps.

    Parameters
    ----------
    latent_tensor
        Latent tensor with shape ``(1, latent_channels, h_latent, w_latent)``.
    diagnostics_config
        Resolved diagnostics config section.
    output_path
        Path to save latent-map PNG.
    """
    max_channels = int(diagnostics_config.get("latent_channels", 8))
    latent_channels = min(max_channels, latent_tensor.shape[1])

    rows = 2
    cols = max(1, int(np.ceil(latent_channels / rows)))

    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.8 * rows))
    axes_array = np.array(axes).reshape(-1)

    for channel_index in range(latent_channels):
        axis = axes_array[channel_index]
        image = axis.pcolormesh(latent_tensor[0, channel_index], cmap="viridis")
        axis.set_title(f"Latent ch {channel_index}")
        axis.axis("off")
        plt.colorbar(image, ax=axis, fraction=0.046)

    for channel_index in range(latent_channels, len(axes_array)):
        axes_array[channel_index].axis("off")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_metrics_json(metrics: dict[str, Any], output_path: str | Path) -> None:
    """Serialize metrics dictionary to JSON."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)


def append_summary_row(
    summary_path: str | Path,
    row: dict[str, Any],
) -> None:
    """Append one run summary row to a CSV table.

    Parameters
    ----------
    summary_path
        Path to summary CSV file.
    row
        Flat dictionary of scalar values for one experiment run.

    Notes
    -----
    The header is created automatically on first write. Later writes append
    rows in the same column order.
    """
    output_file = Path(summary_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    write_header = not output_file.exists()
    fieldnames = list(row.keys())

    with output_file.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
