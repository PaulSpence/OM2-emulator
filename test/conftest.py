"""Shared test setup."""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def loss_functions():
    """
    Load src/Emulator/om2_loss_functions.py on its own.

    Importing the Emulator package also imports the model utilities, which pull
    in Lightning. The loss module only needs torch, so it is loaded directly to
    keep the CI install to torch (CPU) + numpy + pytest.
    """
    path = REPO_ROOT / "src" / "Emulator" / "om2_loss_functions.py"
    spec = importlib.util.spec_from_file_location("om2_loss_functions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
