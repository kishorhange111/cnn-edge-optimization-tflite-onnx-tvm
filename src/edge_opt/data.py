"""MNIST loading and preprocessing shared by training and evaluation.

Every original script repeated the same three lines (load MNIST, divide by
255, reshape to ``(N, 28, 28, 1)``). They live here once so that training and
all four evaluators feed the models exactly the same input.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

#: Height/width of an MNIST digit image.
IMAGE_SIZE = 28


def preprocess(images: np.ndarray) -> np.ndarray:
    """Scale raw MNIST pixels to ``[0, 1]`` and add a channel axis.

    Args:
        images: ``uint8`` array of shape ``(N, 28, 28)`` (or already
            ``(N, 28, 28, 1)``) with values in ``0..255``.

    Returns:
        ``float32`` array of shape ``(N, 28, 28, 1)``. ``float32`` is what the
        TFLite and ONNX models declare as their input type; Keras accepts it
        too, so one dtype works for every format.
    """
    images = np.asarray(images, dtype=np.float32) / 255.0
    return images.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)


def load_mnist(
    path: str | os.PathLike[str] | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Load raw (un-normalized) MNIST as ``((x_train, y_train), (x_test, y_test))``.

    Args:
        path: Optional path to an ``mnist.npz`` file (the same file Keras
            downloads, with keys ``x_train``, ``y_train``, ``x_test``,
            ``y_test``). Passing it lets you evaluate ONNX/TFLite models on a
            machine without TensorFlow. When ``None``, TensorFlow's Keras
            dataset loader is used, which downloads and caches the file.
    """
    if path is not None:
        with np.load(Path(path)) as data:
            return (data["x_train"], data["y_train"]), (data["x_test"], data["y_test"])

    import tensorflow as tf  # Lazy: only needed when no local file is given.

    return tf.keras.datasets.mnist.load_data()


def load_test_set(
    path: str | os.PathLike[str] | None = None, limit: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return a preprocessed slice of the MNIST *test* split.

    Args:
        path: Optional local ``mnist.npz`` (see :func:`load_mnist`).
        limit: Keep only the first ``limit`` images. Some formats only accept a
            batch of one image, so evaluating all 10,000 can be slow.

    Returns:
        ``(images, labels)`` with images shaped ``(N, 28, 28, 1)``.
    """
    _, (x_test, y_test) = load_mnist(path)
    if limit is not None:
        x_test, y_test = x_test[:limit], y_test[:limit]
    return preprocess(x_test), np.asarray(y_test)
