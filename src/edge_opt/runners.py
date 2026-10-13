"""One small "predict" wrapper per model format.

Each loader returns a function ``predict(batch) -> scores`` where ``batch`` is
a ``float32`` array of shape ``(1, 28, 28, 1)`` and ``scores`` has shape
``(1, 10)``. Hiding the framework differences behind one signature is what
lets :mod:`edge_opt.evaluate` benchmark all formats with the same code.

The input/output tensor names are those of the project CNN; override them if
you export a model with different layer names.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import numpy as np

from edge_opt.paths import detect_format

Predictor = Callable[[np.ndarray], np.ndarray]


def load_keras(model_path: str | os.PathLike[str]) -> Predictor:
    """Return a predictor backed by ``tf.keras``.

    The committed ``.h5`` was saved by TensorFlow 1.15. Keras 3 (TensorFlow
    2.16+) cannot rebuild some of its saved layer configs, so if a plain
    ``load_model`` fails we rebuild the same architecture in code and load
    only the weights, which are stored by layer name and still compatible.
    """
    import tensorflow as tf

    try:
        model = tf.keras.models.load_model(str(model_path), compile=False)
    except (TypeError, ValueError):
        from edge_opt.train import build_model

        model = build_model(summary=False)
        model.load_weights(str(model_path))

    def predict(batch: np.ndarray) -> np.ndarray:
        # Calling the model directly avoids predict()'s per-call setup cost,
        # which would dominate the timing for a single 28x28 image.
        return np.asarray(model(batch, training=False))

    return predict


def load_frozen_pb(
    model_path: str | os.PathLike[str],
    input_tensor: str = "conv1_input:0",
    output_tensor: str = "dense_output/BiasAdd:0",
) -> Predictor:
    """Return a predictor that runs a frozen ``.pb`` graph in a TF1 session.

    The default output is the pre-softmax ``BiasAdd`` tensor, as in the
    original ``Normal_Pb_Evaluation.py``; its argmax equals the softmax argmax.
    """
    import tensorflow as tf

    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(str(model_path), "rb") as f:
        graph_def.ParseFromString(f.read())

    graph = tf.Graph()
    with graph.as_default():
        # name="" keeps the original node names (no "import/" prefix).
        tf.import_graph_def(graph_def, name="")
    session = tf.compat.v1.Session(graph=graph)
    inputs = graph.get_tensor_by_name(input_tensor)
    outputs = graph.get_tensor_by_name(output_tensor)

    def predict(batch: np.ndarray) -> np.ndarray:
        return session.run(outputs, feed_dict={inputs: batch})

    return predict


def _tflite_interpreter_class():
    """Find a TFLite interpreter, preferring the lightweight runtimes.

    Order: ``tflite_runtime`` (what the project used on-device), then
    ``ai_edge_litert`` (its successor), then full TensorFlow.
    """
    try:
        from tflite_runtime.interpreter import Interpreter

        return Interpreter
    except ImportError:
        pass
    try:
        from ai_edge_litert.interpreter import Interpreter

        return Interpreter
    except ImportError:
        pass
    import tensorflow as tf

    return tf.lite.Interpreter


def load_tflite(model_path: str | os.PathLike[str]) -> Predictor:
    """Return a predictor backed by the TensorFlow Lite interpreter."""
    interpreter = _tflite_interpreter_class()(model_path=str(model_path))
    # Reserve memory for all tensors once, up front.
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    def predict(batch: np.ndarray) -> np.ndarray:
        interpreter.set_tensor(input_index, batch)
        interpreter.invoke()
        # Copy, because get_tensor returns a view into interpreter memory that
        # the next invoke() would overwrite.
        return np.array(interpreter.get_tensor(output_index))

    return predict


def load_onnx(
    model_path: str | os.PathLike[str],
    input_name: str = "conv1_input",
    output_name: str = "dense_output/Softmax",
) -> Predictor:
    """Return a predictor backed by ONNX Runtime (CPU)."""
    import onnxruntime

    session = onnxruntime.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    def predict(batch: np.ndarray) -> np.ndarray:
        return session.run([output_name], {input_name: batch})[0]

    return predict


LOADERS: dict[str, Callable[..., Predictor]] = {
    "keras": load_keras,
    "pb": load_frozen_pb,
    "tflite": load_tflite,
    "onnx": load_onnx,
}


def load_predictor(
    model_path: str | os.PathLike[str], model_format: str | None = None
) -> Predictor:
    """Load any supported model file and return its predictor.

    Args:
        model_path: Path to a ``.h5``, ``.pb``, ``.tflite`` or ``.onnx`` file.
        model_format: Force a format instead of guessing from the extension.
    """
    model_format = model_format or detect_format(model_path)
    try:
        loader = LOADERS[model_format]
    except KeyError:
        raise ValueError(
            f"Unknown model format {model_format!r}; use one of {sorted(LOADERS)}"
        ) from None
    return loader(model_path)
