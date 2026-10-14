"""Tests for model path resolution."""

from pathlib import Path

import pytest

from edge_opt import paths
from edge_opt.paths import ModelPaths, detect_format, resolve_models_dir


def test_default_models_dir_is_repo_models(monkeypatch):
    monkeypatch.delenv(paths.MODELS_DIR_ENV, raising=False)
    assert resolve_models_dir() == paths.DEFAULT_MODELS_DIR
    assert paths.DEFAULT_MODELS_DIR.name == "models"


def test_env_var_overrides_default(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.MODELS_DIR_ENV, str(tmp_path))
    assert resolve_models_dir() == tmp_path.resolve()


def test_explicit_argument_beats_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.MODELS_DIR_ENV, "/somewhere/else")
    assert resolve_models_dir(tmp_path / "x") == (tmp_path / "x").resolve()


def test_home_is_expanded(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert resolve_models_dir("~/models") == (tmp_path / "models").resolve()


def test_model_paths_layout(tmp_path):
    mp = ModelPaths.from_dir(tmp_path)
    stages = mp.as_dict()
    assert list(stages)[0] == "keras_h5"
    assert all(p.parent == tmp_path.resolve() for p in stages.values())
    assert mp.optimized_onnx.name == "digit_recognition_optimized.onnx"


def test_existing_only_lists_present_files(tmp_path):
    mp = ModelPaths.from_dir(tmp_path)
    mp.optimized_tflite.write_bytes(b"x")
    assert mp.existing() == {"optimized_tflite": mp.optimized_tflite}


def test_committed_models_match_default_names():
    # Guards against renaming a file in models/ without updating paths.py.
    mp = ModelPaths.from_dir(paths.DEFAULT_MODELS_DIR)
    missing = set(mp.as_dict()) - set(mp.existing())
    assert missing == {"keras_onnx"}  # the only stage not committed


@pytest.mark.parametrize(
    "name, fmt",
    [
        ("m.h5", "keras"),
        ("m.keras", "keras"),
        ("m.pb", "pb"),
        ("M.TFLITE", "tflite"),
        (Path("a/b.onnx"), "onnx"),
    ],
)
def test_detect_format(name, fmt):
    assert detect_format(name) == fmt


def test_detect_format_unknown():
    with pytest.raises(ValueError, match="model format"):
        detect_format("model.bin")
