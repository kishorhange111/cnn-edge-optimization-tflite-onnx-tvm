"""Smoke tests that load the committed models (skipped if a runtime is missing)."""

import numpy as np
import pytest

from edge_opt.evaluate import evaluate_model, predict_all
from edge_opt.paths import ModelPaths

from .conftest import MODELS_DIR

PATHS = ModelPaths.from_dir(MODELS_DIR)


def _check_scores(scores):
    assert scores.shape == (1, 10)
    assert np.isfinite(scores).all()


def test_onnx_model_runs():
    pytest.importorskip("onnxruntime")
    from edge_opt.runners import load_onnx

    predict = load_onnx(PATHS.optimized_onnx)
    scores = predict(np.zeros((1, 28, 28, 1), np.float32))
    _check_scores(scores)
    # Softmax output: probabilities sum to 1.
    assert scores.sum() == pytest.approx(1.0, abs=1e-4)


def test_evaluate_model_onnx_with_labels():
    pytest.importorskip("onnxruntime")
    images = np.random.default_rng(0).random((3, 28, 28, 1), dtype=np.float32)
    result = evaluate_model(
        PATHS.optimized_onnx, images=images, labels=np.array([0, 1, 2]), iterations=3
    )
    assert result.model_format == "onnx"
    assert result.size_bytes == PATHS.optimized_onnx.stat().st_size
    assert 0.0 <= result.accuracy <= 1.0
    assert result.latency.iterations == 3


@pytest.mark.parametrize("path", [PATHS.keras_tflite, PATHS.frozen_tflite, PATHS.optimized_tflite])
def test_tflite_models_run(path):
    pytest.importorskip("tensorflow")
    from edge_opt.runners import load_tflite

    _check_scores(load_tflite(path)(np.zeros((1, 28, 28, 1), np.float32)))


def test_tflite_and_onnx_agree():
    """The ONNX model was converted from the optimized TFLite model, so both
    should pick the same class. Scores differ by up to ~4e-3 because TFLite
    runs int8-weight "hybrid" kernels while ONNX de-quantizes the weights and
    computes in float32."""
    pytest.importorskip("tensorflow")
    pytest.importorskip("onnxruntime")
    from edge_opt.runners import load_onnx, load_tflite

    images = np.random.default_rng(1).random((5, 28, 28, 1), dtype=np.float32)
    a = predict_all(load_tflite(PATHS.optimized_tflite), images)
    b = predict_all(load_onnx(PATHS.optimized_onnx), images)
    np.testing.assert_array_equal(a.argmax(axis=1), b.argmax(axis=1))
    np.testing.assert_allclose(a, b, atol=1e-2)


def test_build_model_matches_report_parameter_count():
    pytest.importorskip("tensorflow")
    from edge_opt.train import build_model

    # 119,652 parameters, as printed in the report's training screenshot.
    assert build_model().count_params() == 119_652
