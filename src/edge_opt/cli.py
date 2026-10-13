"""Command line entry point: ``edge-opt`` (or ``python -m edge_opt``).

Sub-commands mirror the pipeline stages::

    edge-opt train                       # MNIST -> models/digit_recognition.h5
    edge-opt convert keras-to-pb         # .h5 -> frozen .pb
    edge-opt convert optimize-pb         # .h5 -> optimized frozen .pb
    edge-opt convert pb-to-tflite        # frozen .pb -> .tflite
    edge-opt convert optimized-pb-to-tflite
    edge-opt convert keras-to-tflite
    edge-opt convert keras-to-onnx
    edge-opt convert tflite-to-onnx
    edge-opt convert all                 # the original pipeline, in order
    edge-opt evaluate PATH               # benchmark one model file
    edge-opt benchmark                   # benchmark every model in models/

Every path defaults to a file inside the models directory
(``--models-dir`` or ``$EDGE_OPT_MODELS_DIR``, default ``models/``), and every
step accepts ``--input`` / ``--output`` to override it.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

from edge_opt.paths import ModelPaths

# name -> (convert function name, default input field, default output field).
# Functions are referenced by name so edge_opt.convert (and TensorFlow) is only
# imported when a conversion actually runs.
CONVERSIONS: dict[str, tuple[str, str, str]] = {
    "keras-to-pb": ("keras_to_frozen_pb", "keras_h5", "frozen_pb"),
    "optimize-pb": ("optimize_frozen_pb", "keras_h5", "optimized_pb"),
    "pb-to-tflite": ("frozen_pb_to_tflite", "frozen_pb", "frozen_tflite"),
    "optimized-pb-to-tflite": ("frozen_pb_to_tflite", "optimized_pb", "optimized_tflite"),
    "keras-to-tflite": ("keras_to_tflite", "keras_h5", "keras_tflite"),
    "keras-to-onnx": ("keras_to_onnx", "keras_h5", "keras_onnx"),
    "tflite-to-onnx": ("tflite_to_onnx", "optimized_tflite", "optimized_onnx"),
}

#: Steps run by ``convert all``: the project's original pipeline. ``keras-to-onnx``
#: is left out because tf2onnx's Keras path needs TensorFlow 2, while the
#: graph-freezing steps need TensorFlow 1.x; run it separately in a TF2 env.
ALL_STEPS = (
    "keras-to-pb",
    "optimize-pb",
    "pb-to-tflite",
    "optimized-pb-to-tflite",
    "keras-to-tflite",
    "tflite-to-onnx",
)

#: Human-readable names used in benchmark tables, keyed by ModelPaths field.
DISPLAY_NAMES = {
    "keras_h5": "Keras",
    "frozen_pb": "Frozen graph",
    "optimized_pb": "Optimized graph",
    "keras_tflite": "TFLite (from Keras)",
    "frozen_tflite": "TFLite (from frozen graph)",
    "optimized_tflite": "TFLite (from optimized graph)",
    "optimized_onnx": "ONNX (from optimized TFLite)",
    "keras_onnx": "ONNX (from Keras)",
}


def resolve_conversion(
    step: str, paths: ModelPaths, input_path: str | None, output_path: str | None
) -> tuple[str, Path, Path]:
    """Work out which function to call and with which files for a convert step.

    Explicit ``--input`` / ``--output`` win over the models-directory defaults.

    Returns:
        ``(function_name, input_path, output_path)``.
    """
    func_name, in_field, out_field = CONVERSIONS[step]
    src = Path(input_path) if input_path else getattr(paths, in_field)
    dst = Path(output_path) if output_path else getattr(paths, out_field)
    return func_name, src, dst


def _run_conversion(step: str, paths: ModelPaths, input_path: str | None, output_path: str | None):
    """Execute one conversion step."""
    from edge_opt import convert  # Lazy: pulls in TensorFlow.

    func_name, src, dst = resolve_conversion(step, paths, input_path, output_path)
    func: Callable[[Path, Path], Path] = getattr(convert, func_name)
    print(f"[{step}] {src} -> {dst}")
    func(src, dst)


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser (separate from main() so tests can use it)."""
    parser = argparse.ArgumentParser(
        prog="edge-opt", description="Optimize a Keras CNN for edge devices."
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Folder holding model files (default: $EDGE_OPT_MODELS_DIR or ./models).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train the CNN on MNIST and save a .h5 model.")
    p_train.add_argument("--epochs", type=int, default=2)
    p_train.add_argument("--output", help="Where to save the .h5 (default: models dir).")
    p_train.add_argument("--mnist-path", help="Local mnist.npz instead of downloading.")

    p_conv = sub.add_parser("convert", help="Run one conversion step (or all).")
    p_conv.add_argument("step", choices=[*CONVERSIONS, "all"])
    p_conv.add_argument("--input", help="Override the input model path.")
    p_conv.add_argument("--output", help="Override the output model path.")

    # Options shared by evaluate and benchmark.
    eval_opts = argparse.ArgumentParser(add_help=False)
    eval_opts.add_argument("--mnist-path", help="Local mnist.npz instead of downloading.")
    eval_opts.add_argument(
        "--num-samples", type=int, default=1000, help="Test images to score (default 1000)."
    )
    eval_opts.add_argument(
        "--iterations", type=int, default=100, help="Timed inferences (default 100)."
    )
    eval_opts.add_argument(
        "--skip-accuracy", action="store_true", help="Only measure size and latency."
    )
    eval_opts.add_argument("--json", dest="json_path", help="Also write results to this file.")

    p_eval = sub.add_parser("evaluate", parents=[eval_opts], help="Benchmark one model file.")
    p_eval.add_argument("model", help="Path to a .h5, .pb, .tflite or .onnx model.")

    sub.add_parser(
        "benchmark", parents=[eval_opts], help="Benchmark every model found in the models dir."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch to the requested sub-command."""
    args = build_parser().parse_args(argv)
    paths = ModelPaths.from_dir(args.models_dir)

    if args.command == "train":
        from edge_opt.train import train

        train(args.output or paths.keras_h5, epochs=args.epochs, mnist_path=args.mnist_path)
        return 0

    if args.command == "convert":
        if args.step == "all":
            if args.input or args.output:
                raise SystemExit("--input/--output cannot be combined with 'all'")
            for step in ALL_STEPS:
                _run_conversion(step, paths, None, None)
        else:
            _run_conversion(args.step, paths, args.input, args.output)
        return 0

    # evaluate / benchmark
    from edge_opt.benchmark import format_results_table
    from edge_opt.evaluate import evaluate_many

    if args.command == "evaluate":
        models = {Path(args.model).name: args.model}
    else:
        models = {DISPLAY_NAMES[k]: v for k, v in paths.existing().items()}
        if not models:
            raise SystemExit(f"No model files found in {paths.keras_h5.parent}")

    results = evaluate_many(
        models,
        mnist_path=args.mnist_path,
        num_samples=args.num_samples,
        iterations=args.iterations,
        skip_accuracy=args.skip_accuracy,
    )
    print(format_results_table(results))
    if args.json_path:
        Path(args.json_path).write_text(json.dumps([r.to_dict() for r in results], indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
