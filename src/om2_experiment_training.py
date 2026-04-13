"""Training utilities for OM2 emulator experiments.

This module owns training-loop behavior so all experiment variants share the
same optimization logic and differ only by explicit config changes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.optim as optim

from om2_experiment_normalization import normalize_tensor


def compute_masked_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute area-unweighted masked L1 loss.

    Parameters
    ----------
    prediction
        Model output tensor.
    target
        Reference tensor in the same normalized space.
    mask
        Binary support mask.

    Returns
    -------
    torch.Tensor
        Scalar masked L1 loss.

    Notes
    -----
    This preserves the notebook's masked-loss behavior where land points are
    ignored and only valid ocean points contribute.
    """
    absolute_error = torch.abs(prediction - target)
    return (absolute_error * mask).sum() / mask.sum().clamp_min(1.0)


def train_model(
    model: torch.nn.Module,
    pipeline: Any,
    mask_tensor: torch.Tensor,
    training_config: dict[str, Any],
    normalization_config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Train a model using samples from a PyEarthTools pipeline.

    Parameters
    ----------
    model
        Initialized model instance.
    pipeline
        Iterable pipeline that yields numpy arrays.
    mask_tensor
        Binary mask tensor with shape ``(1, 1, h, w)``.
    training_config
        Resolved ``training`` config section.
    normalization_config
        Resolved ``normalization`` config section.
    device
        Active torch device.

    Returns
    -------
    dict[str, Any]
        Training summary containing epoch losses and sample counts.

    Notes
    -----
    Missing samples in the iterator can optionally be skipped. This mirrors the
    notebook behavior and prevents sparse archive gaps from aborting long runs.
    """
    learning_rate = float(training_config["learning_rate"])
    num_epochs = int(training_config["num_epochs"])
    max_samples_per_epoch = int(training_config["max_samples_per_epoch"])
    print_every = int(training_config.get("print_every", 100))
    ignore_sample_errors = bool(training_config.get("ignore_sample_errors", True))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    epoch_losses: list[float] = []
    epoch_sample_counts: list[int] = []
    total_steps = 0

    model.train()
    for epoch_index in range(num_epochs):
        iterator = iter(pipeline)
        running_loss = 0.0
        processed_samples = 0

        while processed_samples < max_samples_per_epoch:
            try:
                sample = next(iterator)
            except StopIteration:
                break
            except Exception:
                if ignore_sample_errors:
                    continue
                raise

            x = torch.from_numpy(np.asarray(sample)).float().to(device)

            # Normalize each sample according to the strategy under test.
            x_normalized, _ = normalize_tensor(x, mask_tensor, normalization_config)

            optimizer.zero_grad()
            prediction, _ = model(x_normalized, mask_tensor, return_features=False)
            loss = compute_masked_l1_loss(prediction, x_normalized, mask_tensor)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            processed_samples += 1
            total_steps += 1

            if print_every > 0 and processed_samples % print_every == 0:
                print(
                    f"Epoch {epoch_index + 1}/{num_epochs}: "
                    f"processed {processed_samples} samples, current loss={loss.item():.6f}"
                )

        mean_epoch_loss = running_loss / max(processed_samples, 1)
        epoch_losses.append(mean_epoch_loss)
        epoch_sample_counts.append(processed_samples)

        print(
            f"Epoch [{epoch_index + 1}/{num_epochs}] "
            f"mean_loss={mean_epoch_loss:.6f}, samples={processed_samples}"
        )

    return {
        "epoch_losses": epoch_losses,
        "epoch_sample_counts": epoch_sample_counts,
        "total_steps": total_steps,
        "learning_rate": learning_rate,
    }
