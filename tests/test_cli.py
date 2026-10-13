"""Tests for CLI argument parsing and path wiring (no ML framework needed)."""

import sys
from pathlib import Path

import pytest

from edge_opt import cli, convert
from edge_opt.paths import ModelPaths


def test_every_conversion_maps_to_a_real_function_and_path_field(tmp_path):
    mp = ModelPaths.from_dir(tmp_path)
    for func_name, src, dst in cli.CONVERSIONS.values():
        assert callable(getattr(convert, func_name))
        assert hasattr(mp, src) and hasattr(mp, dst)
    assert set(cli.ALL_STEPS) <= set(cli.CONVERSIONS)


def test_resolve_conversion_defaults(tmp_path):
    mp = ModelPaths.from_dir(tmp_path)
    func, src, dst = cli.resolve_conversion("optimized-pb-to-tflite", mp, None, None)
    assert func == "frozen_pb_to_tflite"
    assert src == mp.optimized_pb
    assert dst == mp.optimized_tflite


def test_resolve_conversion_overrides(tmp_path):
    mp = ModelPaths.from_dir(tmp_path)
    _, src, dst = cli.resolve_conversion("tflite-to-onnx", mp, "in.tflite", "out/x.onnx")
    assert src == Path("in.tflite")
    assert dst == Path("out/x.onnx")


def test_parser_accepts_documented_commands():
    parser = cli.build_parser()
    args = parser.parse_args(["--models-dir", "/m", "benchmark", "--skip-accuracy"])
    assert args.command == "benchmark" and args.skip_accuracy and args.models_dir == "/m"
    args = parser.parse_args(["evaluate", "model.onnx", "--iterations", "5"])
    assert args.model == "model.onnx" and args.iterations == 5
    args = parser.parse_args(["convert", "all"])
    assert args.step == "all"
    with pytest.raises(SystemExit):
        parser.parse_args(["convert", "not-a-step"])


def test_convert_all_rejects_overrides():
    with pytest.raises(SystemExit):
        cli.main(["convert", "all", "--output", "x"])


def test_benchmark_with_empty_models_dir(tmp_path):
    with pytest.raises(SystemExit, match="No model files"):
        cli.main(["--models-dir", str(tmp_path), "benchmark", "--skip-accuracy"])


def test_tflite_to_onnx_command():
    cmd = convert.tflite_to_onnx_command("a.tflite", "b.onnx", opset=13)
    assert cmd[:3] == [sys.executable, "-m", "tf2onnx.convert"]
    assert cmd[3:] == ["--tflite", "a.tflite", "--output", "b.onnx", "--opset", "13"]
    assert "--opset" not in convert.tflite_to_onnx_command("a", "b")
