"""Run the shared accuracy / latency / size benchmark against model files.

This replaces the four near-identical evaluation scripts of the original
project (``keras_evaluation.py``, ``Normal_Pb_Evaluation.py``,
``Tflite_evaluation.py``, ``Onnx_Evaluation.py``). Each of those timed 100
single-image inferences; here the same loop is shared, and top-1 accuracy on
the MNIST test set and file size are reported alongside it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from edge_opt.benchmark import (
    BenchmarkResult,
    file_size_bytes,
    measure_latency,
    top1_accuracy,
)
from edge_opt.data import load_test_set
from edge_opt.paths import detect_format
from edge_opt.runners import Predictor, load_predictor


def predict_all(predict: Predictor, images: np.ndarray) -> np.ndarray:
    """Run ``predict`` one image at a time and stack the scores.

    One image per call because the converted models were exported with a
    fixed batch size of 1 (that is the edge-device use case).
    """
    return np.concatenate([predict(images[i : i + 1]) for i in range(len(images))])


def evaluate_model(
    model_path: str | os.PathLike[str],
    name: str | None = None,
    model_format: str | None = None,
    images: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    iterations: int = 100,
    warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark one model file.

    Args:
        model_path: The model to evaluate.
        name: Label for reports (defaults to the file name).
        model_format: Override the format detected from the extension.
        images: Preprocessed test images ``(N, 28, 28, 1)``. When omitted,
            accuracy is skipped and latency uses a blank image.
        labels: Labels for ``images``.
        iterations: Timed single-image inferences for the latency figure.
        warmup: Untimed inferences before timing starts.
    """
    model_path = Path(model_path)
    model_format = model_format or detect_format(model_path)
    predict = load_predictor(model_path, model_format)

    accuracy = None
    if images is not None and labels is not None:
        accuracy = top1_accuracy(predict_all(predict, images), labels)

    # Time a single image, like the original scripts (which used one fixed
    # training image). The content of the image does not affect runtime.
    sample = images[:1] if images is not None else np.zeros((1, 28, 28, 1), np.float32)
    latency = measure_latency(lambda: predict(sample), iterations=iterations, warmup=warmup)

    return BenchmarkResult(
        name=name or model_path.name,
        model_format=model_format,
        path=str(model_path),
        size_bytes=file_size_bytes(model_path),
        accuracy=accuracy,
        latency=latency,
    )


def evaluate_many(
    models: Mapping[str, str | os.PathLike[str]],
    mnist_path: str | os.PathLike[str] | None = None,
    num_samples: int | None = 1000,
    iterations: int = 100,
    skip_accuracy: bool = False,
) -> list[BenchmarkResult]:
    """Benchmark several models on the same test images.

    A model that cannot be loaded (framework not installed, or a format the
    installed version cannot read) is reported and skipped instead of
    aborting the whole run.

    Args:
        models: ``{display_name: path}``, in the order to report them.
        mnist_path: Optional local ``mnist.npz``.
        num_samples: How many test images to score (``None`` = all 10,000).
        iterations: Timed inferences per model.
        skip_accuracy: Only measure size and latency (no dataset needed).
    """
    images = labels = None
    if not skip_accuracy:
        images, labels = load_test_set(mnist_path, limit=num_samples)

    results = []
    for name, path in models.items():
        try:
            results.append(
                evaluate_model(path, name=name, images=images, labels=labels, iterations=iterations)
            )
        except Exception as exc:  # noqa: BLE001 - report and keep benchmarking the rest
            # Typical causes: the framework is not installed, or a TF1-only
            # graph cannot be loaded by the installed TensorFlow version.
            print(f"Skipping {name}: {type(exc).__name__}: {exc}")
    return results
