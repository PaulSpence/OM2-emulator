"""Normalization utilities for OM2 emulator experiments.

This module makes normalization a first-class, configurable component so we can
run controlled ablations over normalization strategy and quantify impacts on
error maps and latent features.
"""

from __future__ import annotations

from typing import Any

import torch


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor, eps: float) -> torch.Tensor:
    """Safely divide tensors with a small epsilon in the denominator."""
    return numerator / (denominator + eps)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float) -> torch.Tensor:
    """Compute masked mean per channel.

    Parameters
    ----------
    x
        Input tensor ``(batch, channels, height, width)``.
    mask
        Mask tensor broadcastable to ``x``.
    eps
        Numerical stability epsilon.
    """
    weighted = x * mask
    numerator = weighted.sum(dim=(0, 2, 3), keepdim=True)
    denominator = mask.sum(dim=(0, 2, 3), keepdim=True)
    return _safe_divide(numerator, denominator, eps)


def _masked_variance(x: torch.Tensor, mean: torch.Tensor, mask: torch.Tensor, eps: float) -> torch.Tensor:
    """Compute masked variance per channel."""
    weighted_sq = ((x - mean) ** 2) * mask
    numerator = weighted_sq.sum(dim=(0, 2, 3), keepdim=True)
    denominator = mask.sum(dim=(0, 2, 3), keepdim=True)
    return _safe_divide(numerator, denominator, eps)


def _masked_quantile(
    x: torch.Tensor,
    mask: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    """Compute masked per-channel quantile with simple flatten-and-filter logic.

    Notes
    -----
    This function intentionally favors clarity over peak speed. The number of
    channels here is small (OHC + heat flux), so channel-wise filtering remains
    practical and easy to reason about.
    """
    quantiles = []
    for channel_index in range(x.shape[1]):
        channel_data = x[:, channel_index, :, :]
        channel_mask = mask[:, 0, :, :] > 0

        valid = channel_data[channel_mask]
        if valid.numel() == 0:
            valid = channel_data.reshape(-1)

        quantiles.append(torch.quantile(valid, quantile))

    stacked = torch.stack(quantiles).view(1, x.shape[1], 1, 1)
    return stacked


def normalize_tensor(
    x: torch.Tensor,
    mask: torch.Tensor,
    normalization_config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Normalize tensor according to config strategy.

    Parameters
    ----------
    x
        Input tensor ``(batch, channels, height, width)``.
    mask
        Binary support mask ``(batch, 1, height, width)``.
    normalization_config
        Resolved ``normalization`` config section.

    Returns
    -------
    tuple[torch.Tensor, dict[str, torch.Tensor]]
        ``(x_normalized, stats)`` where ``stats`` is used for inverse transform.

    Raises
    ------
    ValueError
        Raised for unsupported normalization strategies.
    """
    strategy = normalization_config["strategy"]
    eps = float(normalization_config.get("eps", 1e-6))

    if strategy == "masked_zscore":
        mean = _masked_mean(x, mask, eps)
        var = _masked_variance(x, mean, mask, eps)
        scale = torch.sqrt(var + eps)

    elif strategy == "nan_zscore":
        mean = torch.nanmean(x, dim=(0, 2, 3), keepdim=True)
        var = torch.nanmean((x - mean) ** 2, dim=(0, 2, 3), keepdim=True)
        scale = torch.sqrt(var + eps)

    elif strategy == "robust_median_iqr":
        q_low, q_high = normalization_config.get("robust_quantiles", [0.25, 0.75])
        mean = _masked_quantile(x, mask, 0.5)
        low = _masked_quantile(x, mask, float(q_low))
        high = _masked_quantile(x, mask, float(q_high))
        scale = (high - low).clamp_min(eps)

    elif strategy == "minmax":
        mins = []
        maxs = []
        for channel_index in range(x.shape[1]):
            channel_data = x[:, channel_index, :, :]
            channel_mask = mask[:, 0, :, :] > 0
            valid = channel_data[channel_mask]
            if valid.numel() == 0:
                valid = channel_data.reshape(-1)

            mins.append(valid.min())
            maxs.append(valid.max())

        min_tensor = torch.stack(mins).view(1, x.shape[1], 1, 1)
        max_tensor = torch.stack(maxs).view(1, x.shape[1], 1, 1)
        mean = min_tensor
        scale = (max_tensor - min_tensor).clamp_min(eps)

    else:
        raise ValueError(f"Unsupported normalization strategy: {strategy!r}")

    x_normalized = _safe_divide((x - mean), scale, eps)

    # Ensure NaNs do not propagate through training when masked values are NaN.
    x_normalized = torch.nan_to_num(x_normalized, nan=0.0, posinf=0.0, neginf=0.0)

    stats = {
        "shift": mean,
        "scale": scale,
        "strategy": torch.tensor(0),  # placeholder tensor to keep stats torch-only if needed
    }
    return x_normalized, stats


def denormalize_tensor(
    x_normalized: torch.Tensor,
    stats: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Convert normalized predictions back to physical units.

    Parameters
    ----------
    x_normalized
        Normalized tensor.
    stats
        Normalization statistics dictionary from :func:`normalize_tensor`.

    Returns
    -------
    torch.Tensor
        Tensor in original physical units.
    """
    return x_normalized * stats["scale"] + stats["shift"]
