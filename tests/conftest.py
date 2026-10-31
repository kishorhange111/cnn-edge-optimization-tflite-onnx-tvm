"""Shared pytest fixtures."""

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"


@pytest.fixture
def fake_mnist(tmp_path: Path) -> Path:
    """Write a tiny ``mnist.npz`` with the same keys/dtypes as the real one.

    Lets data-loading code be tested without downloading MNIST.
    """
    rng = np.random.default_rng(0)
    path = tmp_path / "mnist.npz"
    np.savez(
        path,
        x_train=rng.integers(0, 256, size=(6, 28, 28), dtype=np.uint8),
        y_train=rng.integers(0, 10, size=6, dtype=np.uint8),
        x_test=rng.integers(0, 256, size=(4, 28, 28), dtype=np.uint8),
        y_test=np.array([3, 1, 4, 1], dtype=np.uint8),
    )
    return path
