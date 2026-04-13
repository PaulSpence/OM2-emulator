"""Ablation-matrix generation and execution helpers.

This module creates one-factor-at-a-time variants from a baseline config. The
resulting runs are directly comparable and well-suited for attribution-focused
analysis of architecture and normalization choices.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from om2_experiment_config import (
    dump_yaml_config,
    load_yaml_config,
    resolve_config,
    slugify_variant_name,
)


def _parse_path_token(token: str) -> tuple[str, int | None]:
    """Parse one dotted-path token with optional list index.

    Examples
    --------
    - ``"model" -> ("model", None)``
    - ``"encoder[0]" -> ("encoder", 0)``
    """
    if "[" not in token:
        return token, None

    key, indexed = token.split("[", 1)
    index_str = indexed.rstrip("]")
    return key, int(index_str)


def set_nested_path(config: dict[str, Any], path: str, value: Any) -> None:
    """Set a nested config value using dot-path notation.

    Parameters
    ----------
    config
        Mutable config dictionary to update.
    path
        Dot path with optional list indexes.
        Example: ``"model.encoder[0].kernel_size"``.
    value
        Replacement value.
    """
    tokens = path.split(".")
    cursor: Any = config

    for token in tokens[:-1]:
        key, index = _parse_path_token(token)
        cursor = cursor[key]
        if index is not None:
            cursor = cursor[index]

    last_key, last_index = _parse_path_token(tokens[-1])
    if last_index is None:
        cursor[last_key] = value
    else:
        cursor[last_key][last_index] = value


def get_nested_path(config: dict[str, Any], path: str) -> Any:
    """Read a nested value from dot-path notation."""
    tokens = path.split(".")
    cursor: Any = config
    for token in tokens:
        key, index = _parse_path_token(token)
        cursor = cursor[key]
        if index is not None:
            cursor = cursor[index]

    return cursor


def generate_ablation_variants(
    baseline_config: dict[str, Any],
    matrix_config: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Generate one-factor-at-a-time variant configs.

    Parameters
    ----------
    baseline_config
        Fully resolved baseline config.
    matrix_config
        Ablation matrix config with structure:
        ``{"factors": {"path.to.key": [candidate_values...]}}``.

    Returns
    -------
    list[tuple[str, dict[str, Any]]]
        Sequence of ``(variant_name, variant_config)`` pairs.

    Notes
    -----
    If a candidate value equals the baseline value, that candidate is skipped
    because it would duplicate the baseline run.
    """
    factors = matrix_config.get("factors", {})
    variants: list[tuple[str, dict[str, Any]]] = []

    for path, candidate_values in factors.items():
        baseline_value = get_nested_path(baseline_config, path)
        for candidate in candidate_values:
            if candidate == baseline_value:
                continue

            variant = deepcopy(baseline_config)
            set_nested_path(variant, path, candidate)

            candidate_slug = slugify_variant_name(str(candidate))
            path_slug = slugify_variant_name(path)
            variant_name = f"abl_{path_slug}_{candidate_slug}"
            variant["run"]["name"] = variant_name

            variants.append((variant_name, variant))

    return variants


def run_ablation_matrix(
    baseline_yaml: str | Path,
    matrix_yaml: str | Path,
    save_generated_configs: bool = True,
) -> list[dict[str, Any]]:
    """Run baseline + matrix-generated ablation variants.

    Parameters
    ----------
    baseline_yaml
        Baseline experiment YAML file.
    matrix_yaml
        Ablation matrix YAML file.
    save_generated_configs
        If ``True``, write generated variant configs next to run outputs.

    Returns
    -------
    list[dict[str, Any]]
        List of metrics dictionaries from each executed run.
    """
    baseline_user = load_yaml_config(baseline_yaml)
    baseline_resolved = resolve_config(baseline_user)

    matrix_config = load_yaml_config(matrix_yaml)
    variants = generate_ablation_variants(baseline_resolved, matrix_config)

    output_root = Path(baseline_resolved["run"]["output_root"]).expanduser().resolve()
    generated_dir = output_root / "generated_configs"

    from om2_experiment_runner import run_experiment_from_config_dict

    metrics: list[dict[str, Any]] = []

    # Always run baseline first so matrix comparisons have a reference row.
    baseline_metrics = run_experiment_from_config_dict(deepcopy(baseline_resolved))
    metrics.append(baseline_metrics)

    for variant_name, variant_config in variants:
        if save_generated_configs:
            generated_dir.mkdir(parents=True, exist_ok=True)
            dump_yaml_config(variant_config, generated_dir / f"{variant_name}.yaml")

        variant_metrics = run_experiment_from_config_dict(variant_config)
        metrics.append(variant_metrics)

    return metrics
