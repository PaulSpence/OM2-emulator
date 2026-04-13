"""Configuration utilities for OM2 emulator experiments.

This module centralizes everything related to experiment configuration so
training runs are reproducible and easy to compare. The key design goal is
that every experiment should have a fully-resolved config written to disk,
which means we can always reconstruct exactly how a model was trained.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only if dependency missing
    raise ImportError(
        "PyYAML is required for OM2 experiment configs. On NCI, run `module load conda/analysis3` before launching the CLI, or install with `pip install pyyaml`."
    ) from exc


DEFAULT_CONFIG: dict[str, Any] = {
    "run": {
        "name": "baseline",
        "seed": 42,
        "device": "auto",
        "output_root": "runs",
        "save_arrays": True,
    },
    "data": {
        "root": "data",
        "dataset_file": "1deg_ocean_heat_emulator_data.nc",
        "variables": ["area_t", "ocean_heat_content_2d", "total_surface_heat_flx"],
        "drop_coordinates": ["geolat_t", "geolon_t"],
        "drop_variables": ["area_t"],
        "sort_order": ["time", "latitude", "longitude"],
        "time_start": "2000-06",
        "time_end": "2001-01",
        "time_interval": "1 month",
        "fill_nan": 0.0,
        "reshape_pattern": "c t h w -> t c h w",
    },
    "model": {
        "input_channels": 2,
        "output_channels": 2,
        "encoder": [
            {"out_channels": 16, "kernel_size": 4, "stride": 2, "padding": 1},
            {"out_channels": 32, "kernel_size": 3, "stride": 2, "padding": 1},
            {"out_channels": 64, "kernel_size": 7, "stride": 1, "padding": 3},
        ],
        "decoder": [
            {
                "out_channels": 32,
                "kernel_size": 7,
                "stride": 1,
                "padding": 3,
                "upsample_factor": 2,
            },
            {
                "out_channels": 16,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "upsample_factor": 2,
            },
            {
                "out_channels": 2,
                "kernel_size": 4,
                "stride": 1,
                "padding": 2,
                "upsample_factor": 1,
            },
        ],
        "activation": "relu",
        "final_activation": "sigmoid",
        "upsample_mode": "bilinear",
    },
    "normalization": {
        "strategy": "masked_zscore",
        "scope": "per_sample",
        "eps": 1e-6,
        "robust_quantiles": [0.25, 0.75],
    },
    "training": {
        "num_epochs": 30,
        "max_samples_per_epoch": 500,
        "learning_rate": 1e-4,
        "print_every": 100,
        "ignore_sample_errors": True,
    },
    "diagnostics": {
        "sample_index": 0,
        "channel_names": ["OHC", "HeatFlux"],
        "flux_channel_index": 1,
        "ohc_channel_index": 0,
        "flux_scale": 2592000.0,
        "flux_vmin": -1e8,
        "flux_vmax": 1e8,
        "ohc_vmin": -1e13,
        "ohc_vmax": 1e13,
        "latent_channels": 8,
    },
}


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment config from disk.

    Parameters
    ----------
    path
        Path to a YAML file.

    Returns
    -------
    dict[str, Any]
        Parsed YAML content.

    Notes
    -----
    The function keeps parsing intentionally simple and transparent.
    Validation is handled in :func:`validate_config` so parse errors and
    semantic errors remain clearly separated.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    return loaded or {}


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overrides`` into ``base``.

    Parameters
    ----------
    base
        Baseline dictionary, usually defaults.
    overrides
        User-provided values that should replace default values.

    Returns
    -------
    dict[str, Any]
        A new merged dictionary.

    Notes
    -----
    - Scalars and lists are replaced, not merged element-wise.
    - Nested dictionaries are merged recursively.
    - Inputs are never mutated.
    """
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)

    return merged


def resolve_config(user_config: dict[str, Any]) -> dict[str, Any]:
    """Resolve user config against defaults.

    Parameters
    ----------
    user_config
        Parsed YAML content from a user config file.

    Returns
    -------
    dict[str, Any]
        Fully-resolved config dictionary with defaults filled in.
    """
    resolved = deep_merge(DEFAULT_CONFIG, user_config)
    validate_config(resolved)
    return resolved


def validate_config(config: dict[str, Any]) -> None:
    """Validate required config sections and key types.

    Parameters
    ----------
    config
        Fully-resolved config dictionary.

    Raises
    ------
    ValueError
        Raised if any required section or key is invalid.

    Notes
    -----
    Validation is intentionally strict for fields that affect model shape
    and training logic. This catches errors early and avoids wasting GPU/CPU
    time on runs that would fail mid-training.
    """
    required_sections = ["run", "data", "model", "normalization", "training", "diagnostics"]
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise ValueError(f"Config missing required sections: {missing_sections}")

    model = config["model"]
    if not isinstance(model.get("encoder"), list) or not model["encoder"]:
        raise ValueError("model.encoder must be a non-empty list of block configs")

    if not isinstance(model.get("decoder"), list) or not model["decoder"]:
        raise ValueError("model.decoder must be a non-empty list of block configs")

    for block_name in ("encoder", "decoder"):
        for index, block in enumerate(model[block_name]):
            for key in ("out_channels", "kernel_size", "stride", "padding"):
                if key not in block:
                    raise ValueError(
                        f"model.{block_name}[{index}] missing required key '{key}'"
                    )

    strategy = config["normalization"].get("strategy")
    valid_strategies = {"masked_zscore", "nan_zscore", "robust_median_iqr", "minmax"}
    if strategy not in valid_strategies:
        raise ValueError(
            "normalization.strategy must be one of "
            f"{sorted(valid_strategies)}; got {strategy!r}"
        )


def dump_yaml_config(config: dict[str, Any], path: str | Path) -> None:
    """Write a config dictionary to YAML.

    Parameters
    ----------
    config
        Configuration dictionary.
    path
        Output YAML path.

    Notes
    -----
    The parent directory is created automatically so callers can write to new
    run folders without extra filesystem checks.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def slugify_variant_name(raw_name: str) -> str:
    """Create a filesystem-safe slug for variant names.

    Parameters
    ----------
    raw_name
        Human-readable variant label.

    Returns
    -------
    str
        Lowercase, underscore-separated slug.
    """
    cleaned = raw_name.strip().lower().replace(" ", "_")
    for disallowed in ("/", "\\", ":", "[", "]", "(", ")", ",", "="):
        cleaned = cleaned.replace(disallowed, "_")

    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")

    return cleaned.strip("_")
