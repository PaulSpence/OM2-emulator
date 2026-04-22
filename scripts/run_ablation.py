#!/usr/bin/env python3
"""CLI entrypoint for ablation-matrix experiment batches."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path


def _bootstrap_modules_if_needed() -> None:
    """Relaunch script in an environment with required HPC modules loaded.

    Notes
    -----
    The relaunch occurs only once per invocation via the
    ``OM2_MODULE_BOOTSTRAPPED`` flag. We intentionally load PET only,
    with fallback to PET bin path if modulefiles are unavailable.
    """
    if os.environ.get("OM2_MODULE_BOOTSTRAPPED") == "1":
        return

    script_path = Path(__file__).resolve()
    args = " ".join(shlex.quote(arg) for arg in sys.argv[1:])

    pet_module = os.environ.get("OM2_PET_MODULE", "pet/2025.08")

    relaunch_command = (
        "module unload openmpi >/dev/null 2>&1 || true; "
        f"(module load {shlex.quote(pet_module)} || export PATH=/g/data/dk92/apps/pet/2025.08/bin:$PATH) && "
        f"OM2_MODULE_BOOTSTRAPPED=1 exec python {shlex.quote(str(script_path))} {args}"
    )

    os.execv("/bin/bash", ["bash", "-lc", relaunch_command])


def _add_local_paths() -> None:
    """Ensure local source and sibling PyEarthTools paths are importable."""
    repo_root = Path(__file__).resolve().parents[1]

    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    pet_packages_root = repo_root.parent / "PyEarthTools" / "packages"
    if pet_packages_root.exists():
        for package_src in sorted(pet_packages_root.glob("*/src")):
            package_src_str = str(package_src)
            if package_src_str not in sys.path:
                sys.path.insert(0, package_src_str)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for matrix execution."""
    parser = argparse.ArgumentParser(
        description=(
            "Run baseline + one-factor-at-a-time ablation variants from config "
            "matrix definition."
        )
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to baseline config file (JSON content; .json or .yaml).",
    )
    parser.add_argument(
        "--matrix",
        required=True,
        help="Path to ablation matrix config file (JSON content; .json or .yaml).",
    )
    parser.add_argument(
        "--no-save-generated-configs",
        action="store_true",
        help="Disable writing generated variant config files.",
    )
    return parser.parse_args()


def main() -> int:
    """Execute the matrix workflow and print consolidated metrics."""
    _bootstrap_modules_if_needed()

    args = parse_args()
    _add_local_paths()

    from om2_experiment_ablation import run_ablation_matrix

    metrics = run_ablation_matrix(
        baseline_yaml=args.baseline,
        matrix_yaml=args.matrix,
        save_generated_configs=not args.no_save_generated_configs,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
