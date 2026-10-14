"""Every model-format conversion step in the optimization pipeline.

Pipeline (each function below is one arrow)::

    Keras .h5 --keras_to_frozen_pb--> frozen .pb --optimize_frozen_pb--> optimized .pb
        |                                 |                                   |
        |                          frozen_pb_to_tflite              frozen_pb_to_tflite
        |                                 v                                   v
        +--keras_to_tflite--> .tflite   frozen .tflite                optimized .tflite
        |                                                                     |
        +--keras_to_onnx--> .onnx                              tflite_to_onnx v
                                                                       optimized .onnx

TensorFlow version: the graph-freezing steps use TF1-style sessions and the
``graph_transforms`` tool, exactly as the project did with TensorFlow 1.15
(see ``requirements-tf1.txt``). They are written against ``tf.compat.v1`` so
the same API names work, but ``graph_transforms`` and Keras sessions do not
exist in TensorFlow 2.x eager mode. The ONNX steps rely on ``tf2onnx``.

All inputs and outputs are explicit paths; nothing is read from or written to
the current working directory implicitly.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

#: Graph input node of the CNN (Keras names it ``<first layer>_input``).
DEFAULT_INPUT_NODES = ("conv1_input",)
#: Graph output node of the CNN (the softmax of the ``dense_output`` layer).
DEFAULT_OUTPUT_NODES = ("dense_output/Softmax",)

#: Graph rewrites applied by :func:`optimize_frozen_pb`. They strip training-only
#: and debugging ops and fold batch-norm into the preceding weights. Re-sorting
#: after each rewrite keeps the node order valid for the next one.
GRAPH_TRANSFORMS = (
    "remove_nodes(op=Identity, op=CheckNumerics, op=StopGradient)",
    "sort_by_execution_order",
    "remove_attribute(attribute_name=_XlaSeparateCompiledGradients)",
    "remove_attribute(attribute_name=_XlaCompile)",
    "remove_attribute(attribute_name=_XlaScope)",
    "sort_by_execution_order",
    "remove_device",
    "sort_by_execution_order",
    "fold_batch_norms",
    "sort_by_execution_order",
    "fold_old_batch_norms",
    "sort_by_execution_order",
)


def _prepare_output(path: str | os.PathLike[str]) -> Path:
    """Make sure the parent folder of an output file exists and return it as a Path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_keras_for_export(h5_path: str | os.PathLike[str]):
    """Load a Keras model in inference mode inside a TF1-style session.

    ``set_learning_phase(0)`` tells layers such as Dropout/BatchNorm to use
    their inference behaviour, so the exported graph does not contain
    training branches. It must be called *before* the model is loaded.

    Returns:
        ``(tf_module, model)`` so callers can reuse the imported module.
    """
    import tensorflow as tf

    tf.keras.backend.set_learning_phase(0)
    model = tf.keras.models.load_model(str(h5_path))
    return tf, model


def keras_node_names(h5_path: str | os.PathLike[str]) -> tuple[list[str], list[str]]:
    """Read the graph input and output node names of a Keras model.

    For the project's CNN this returns ``(['conv1_input'], ['dense_output/Softmax'])``.
    """
    _, model = _load_keras_for_export(h5_path)
    inputs = [t.op.name for t in model.inputs]
    outputs = [t.op.name for t in model.outputs]
    return inputs, outputs


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """Turn every variable in a TF1 session into a constant ("freezing").

    A frozen graph is a single self-contained ``GraphDef`` with weights baked
    in, which is what TFLite's frozen-graph converter and ``graph_transforms``
    expect.

    Args:
        session: A ``tf.compat.v1.Session`` holding the loaded Keras model.
        keep_var_names: Variables to leave as variables (``None`` = freeze all).
        output_names: Output node names to keep.
        clear_devices: Drop device placements (``/GPU:0`` etc.) so the graph
            runs anywhere.

    Returns:
        The frozen ``GraphDef``.
    """
    import tensorflow as tf

    graph = session.graph
    with graph.as_default():
        global_vars = tf.compat.v1.global_variables()
        freeze_var_names = list({v.op.name for v in global_vars}.difference(keep_var_names or []))
        output_names = list(output_names or [])
        # The original project also listed every variable as an "output". That
        # stops them from being pruned, so the optimizer's Adam slots end up in
        # the frozen graph too - one reason this .pb is ~3x larger than the
        # optimized one produced by optimize_frozen_pb().
        output_names += [v.op.name for v in global_vars]

        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        return tf.compat.v1.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names
        )


def keras_to_frozen_pb(h5_path: str | os.PathLike[str], pb_path: str | os.PathLike[str]) -> Path:
    """Convert a Keras ``.h5`` model to a frozen TensorFlow graph (``.pb``).

    This is the "first line of optimization" in the report: the result has no
    Keras wrapper and can be loaded by any TensorFlow tool.
    """
    tf, model = _load_keras_for_export(h5_path)
    output_names = [t.op.name for t in model.outputs]
    frozen = freeze_session(tf.compat.v1.keras.backend.get_session(), output_names=output_names)

    pb_path = _prepare_output(pb_path)
    # as_text=False writes the compact binary protobuf format.
    tf.io.write_graph(frozen, str(pb_path.parent), pb_path.name, as_text=False)
    print(f"Wrote frozen graph to {pb_path}")
    return pb_path


def optimize_frozen_pb(
    h5_path: str | os.PathLike[str],
    optimized_pb_path: str | os.PathLike[str],
    input_nodes: Sequence[str] = DEFAULT_INPUT_NODES,
    output_nodes: Sequence[str] = DEFAULT_OUTPUT_NODES,
    transforms: Sequence[str] = GRAPH_TRANSFORMS,
) -> Path:
    """Freeze the Keras model and clean it up with TensorFlow's ``graph_transforms``.

    Unlike :func:`keras_to_frozen_pb`, only the path from ``input_nodes`` to
    ``output_nodes`` is kept, and the rewrites in :data:`GRAPH_TRANSFORMS`
    remove inference-irrelevant ops. This took the graph from 1.44 MB to
    484 KB in the project.

    Requires TensorFlow 1.x (``tensorflow.tools.graph_transforms``).
    """
    tf, _ = _load_keras_for_export(h5_path)
    from tensorflow.tools.graph_transforms import TransformGraph

    session = tf.compat.v1.keras.backend.get_session()
    # add_shapes=True records tensor shapes, which some transforms rely on.
    graph_def = session.graph.as_graph_def(add_shapes=True)
    graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
        session, graph_def, list(output_nodes)
    )
    graph_def = TransformGraph(graph_def, list(input_nodes), list(output_nodes), list(transforms))

    optimized_pb_path = _prepare_output(optimized_pb_path)
    with tf.io.gfile.GFile(str(optimized_pb_path), "wb") as f:
        f.write(graph_def.SerializeToString())
    print(f"Wrote optimized graph to {optimized_pb_path}")
    return optimized_pb_path


def frozen_pb_to_tflite(
    pb_path: str | os.PathLike[str],
    tflite_path: str | os.PathLike[str],
    input_nodes: Sequence[str] = DEFAULT_INPUT_NODES,
    output_nodes: Sequence[str] = DEFAULT_OUTPUT_NODES,
) -> Path:
    """Convert a frozen (plain or optimized) ``.pb`` graph to TensorFlow Lite.

    The TFLite converter fuses consecutive ops (e.g. conv + bias + ReLU) into
    single kernels, and the optimization flag below stores weights as int8,
    which is where most of the size reduction comes from.
    Used for both ``digit_recognition_frozen.tflite`` and
    ``digit_recognition_optimized.tflite``.
    """
    import tensorflow as tf

    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        str(pb_path), input_arrays=list(input_nodes), output_arrays=list(output_nodes)
    )
    # In TF 1.15, OPTIMIZE_FOR_LATENCY is an alias of Optimize.DEFAULT, which
    # applies post-training *dynamic-range quantization*: weights are stored as
    # int8 (about 4x smaller) and de-quantized on the fly, while inputs and
    # outputs stay float32. This is why these .tflite files are ~125 KB versus
    # ~482 KB for the un-optimized keras_to_tflite() output, and why their
    # scores can differ slightly (~1e-3) from the float Keras model.
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    converter.target_spec.supported_types = [tf.float32]
    tflite_model = converter.convert()

    tflite_path = _prepare_output(tflite_path)
    tflite_path.write_bytes(tflite_model)
    print(f"Wrote TFLite model to {tflite_path}")
    return tflite_path


def keras_to_tflite(h5_path: str | os.PathLike[str], tflite_path: str | os.PathLike[str]) -> Path:
    """Convert the Keras ``.h5`` straight to TFLite with default settings.

    This is the plain baseline: no optimization flag is set, so weights stay
    float32 and the result (482 KB) is about 4x larger than the TFLite files
    produced by :func:`frozen_pb_to_tflite` (125 KB).
    """
    import tensorflow as tf

    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(str(h5_path))
    tflite_model = converter.convert()

    tflite_path = _prepare_output(tflite_path)
    tflite_path.write_bytes(tflite_model)
    print(f"Wrote TFLite model to {tflite_path}")
    return tflite_path


def keras_to_onnx(
    h5_path: str | os.PathLike[str], onnx_path: str | os.PathLike[str], opset: int = 13
) -> Path:
    """Convert the Keras ``.h5`` model directly to ONNX using ``tf2onnx``.

    The original ``h5_to_onnx.py`` was a ``keras2onnx`` snippet that converted
    a stock VGG16 rather than the project model; ``keras2onnx`` is no longer
    maintained, so this uses ``tf2onnx`` (the tool the project used for the
    TFLite -> ONNX step) on the project's own model instead.
    """
    import tensorflow as tf
    import tf2onnx

    model = tf.keras.models.load_model(str(h5_path))
    onnx_path = _prepare_output(onnx_path)
    tf2onnx.convert.from_keras(model, opset=opset, output_path=str(onnx_path))
    print(f"Wrote ONNX model to {onnx_path}")
    return onnx_path


def tflite_to_onnx_command(
    tflite_path: str | os.PathLike[str], onnx_path: str | os.PathLike[str], opset: int | None = None
) -> list[str]:
    """Build the ``tf2onnx`` command line used to convert TFLite to ONNX.

    Kept separate from :func:`tflite_to_onnx` so it can be unit-tested and
    printed for users who prefer to run it by hand. This is the same command
    as the original ``Tflite_To_Onnx.txt``.
    """
    cmd = [
        sys.executable,
        "-m",
        "tf2onnx.convert",
        "--tflite",
        str(tflite_path),
        "--output",
        str(onnx_path),
    ]
    if opset is not None:
        cmd += ["--opset", str(opset)]
    return cmd


def tflite_to_onnx(
    tflite_path: str | os.PathLike[str], onnx_path: str | os.PathLike[str], opset: int | None = None
) -> Path:
    """Convert a TFLite model to ONNX by running the ``tf2onnx`` CLI.

    ``tf2onnx`` is invoked as a subprocess (rather than imported) because its
    TFLite path is only exposed through the command-line tool.
    """
    onnx_path = _prepare_output(onnx_path)
    subprocess.run(tflite_to_onnx_command(tflite_path, onnx_path, opset), check=True)
    print(f"Wrote ONNX model to {onnx_path}")
    return onnx_path
