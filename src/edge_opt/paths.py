"""Locations of every model artifact in the pipeline.

The original scripts hard-coded file names such as ``"Digit_Recognition.h5"``
and assumed they were run from the repository root. Here all paths are derived
from a single *models directory*, which can be set in three ways (highest
priority first):

1. an explicit argument (for example ``--models-dir`` on the CLI),
2. the ``EDGE_OPT_MODELS_DIR`` environment variable,
3. the ``models/`` folder at the repository root (the default).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

#: Environment variable that overrides the default models directory.
MODELS_DIR_ENV = "EDGE_OPT_MODELS_DIR"

# src/edge_opt/paths.py -> parents[2] is the repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = REPO_ROOT / "models"


def resolve_models_dir(models_dir: str | os.PathLike[str] | None = None) -> Path:
    """Return the directory that holds (or will hold) the model files.

    Args:
        models_dir: Explicit directory. When ``None`` the ``EDGE_OPT_MODELS_DIR``
            environment variable is used, and if that is unset too, the
            repository's ``models/`` folder.

    Returns:
        An absolute, user-expanded ``Path``. The directory is *not* created
        here; writers call ``mkdir`` themselves so read-only use has no side
        effects.
    """
    if models_dir is None:
        models_dir = os.environ.get(MODELS_DIR_ENV) or DEFAULT_MODELS_DIR
    return Path(models_dir).expanduser().resolve()


@dataclass(frozen=True)
class ModelPaths:
    """File path of each artifact produced by the pipeline.

    Each field is one stage. The default file names match the artifacts that
    are committed in ``models/``.
    """

    keras_h5: Path
    frozen_pb: Path
    optimized_pb: Path
    keras_tflite: Path
    frozen_tflite: Path
    optimized_tflite: Path
    optimized_onnx: Path
    keras_onnx: Path

    @classmethod
    def from_dir(cls, models_dir: str | os.PathLike[str] | None = None) -> ModelPaths:
        """Build the standard set of paths inside ``models_dir``.

        Args:
            models_dir: See :func:`resolve_models_dir`.
        """
        root = resolve_models_dir(models_dir)
        return cls(
            keras_h5=root / "digit_recognition.h5",
            frozen_pb=root / "digit_recognition_frozen.pb",
            optimized_pb=root / "digit_recognition_optimized.pb",
            # TFLite straight from Keras (no graph optimization first).
            keras_tflite=root / "digit_recognition_keras.tflite",
            # TFLite from the plain frozen graph.
            frozen_tflite=root / "digit_recognition_frozen.tflite",
            # TFLite from the transform_graph-optimized frozen graph.
            optimized_tflite=root / "digit_recognition_optimized.tflite",
            # ONNX converted from the optimized TFLite model.
            optimized_onnx=root / "digit_recognition_optimized.onnx",
            # ONNX converted directly from the Keras model (not committed).
            keras_onnx=root / "digit_recognition_keras.onnx",
        )

    def as_dict(self) -> dict[str, Path]:
        """Return ``{stage_name: path}`` for every stage, in pipeline order."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def existing(self) -> dict[str, Path]:
        """Return only the stages whose file is present on disk."""
        return {name: path for name, path in self.as_dict().items() if path.is_file()}


def detect_format(model_path: str | os.PathLike[str]) -> str:
    """Guess the model format from a file extension.

    Args:
        model_path: Path to a model file.

    Returns:
        One of ``"keras"``, ``"pb"``, ``"tflite"`` or ``"onnx"``.

    Raises:
        ValueError: If the extension is not one the pipeline knows about.
    """
    suffix = Path(model_path).suffix.lower()
    mapping = {
        ".h5": "keras",
        ".keras": "keras",
        ".pb": "pb",
        ".tflite": "tflite",
        ".onnx": "onnx",
    }
    try:
        return mapping[suffix]
    except KeyError:
        raise ValueError(
            f"Cannot tell the model format of {model_path!s}; expected one of {sorted(mapping)}"
        ) from None
