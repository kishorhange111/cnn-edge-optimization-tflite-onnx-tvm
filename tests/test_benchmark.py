"""Tests for the framework-free benchmark helpers."""

import itertools
import math

import numpy as np
import pytest

from edge_opt.benchmark import (
    BenchmarkResult,
    LatencyStats,
    compression_ratio,
    format_results_table,
    human_size,
    measure_latency,
    percentile,
    summarize_latencies,
    top1_accuracy,
)


@pytest.mark.parametrize("pct", [0, 10, 50, 95, 100])
def test_percentile_matches_numpy(pct):
    values = [5.0, 1.0, 3.0, 2.0, 4.0, 10.0]
    assert percentile(values, pct) == pytest.approx(np.percentile(values, pct))


def test_percentile_rejects_bad_input():
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_summarize_latencies():
    stats = summarize_latencies([1.0, 2.0, 3.0, 4.0])
    assert stats.iterations == 4
    assert stats.mean_ms == pytest.approx(2.5)
    assert stats.median_ms == pytest.approx(2.5)
    assert stats.min_ms == 1.0
    assert stats.max_ms == 4.0
    # FPS comes from the mean latency, not from averaging 1/t per iteration.
    assert stats.fps == pytest.approx(400.0)


def test_summarize_latencies_empty():
    with pytest.raises(ValueError):
        summarize_latencies([])


def test_zero_latency_fps_is_infinite():
    stats = LatencyStats(1, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert math.isinf(stats.fps)


def test_measure_latency_uses_injected_timer_and_warmup():
    calls = []
    # Fake clock: each timed call appears to take exactly 2 ms.
    ticks = itertools.count(step=0.002)
    stats = measure_latency(
        lambda: calls.append(1), iterations=5, warmup=3, timer=lambda: next(ticks)
    )
    assert len(calls) == 8  # 3 warm-up + 5 timed
    assert stats.iterations == 5
    assert stats.mean_ms == pytest.approx(2.0)
    assert stats.p95_ms == pytest.approx(2.0)


def test_measure_latency_requires_iterations():
    with pytest.raises(ValueError):
        measure_latency(lambda: None, iterations=0)


def test_top1_accuracy_with_probabilities_and_logits():
    labels = np.array([0, 1, 2, 1])
    probs = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.7, 0.1], [0.3, 0.6, 0.1]])
    assert top1_accuracy(probs, labels) == pytest.approx(0.75)
    # Logits give the same argmax, so the same accuracy.
    assert top1_accuracy(np.log(probs) + 3.0, labels) == pytest.approx(0.75)


@pytest.mark.parametrize(
    "scores, labels",
    [
        (np.zeros((2, 3)), np.array([0])),  # length mismatch
        (np.zeros(3), np.array([0, 1, 2])),  # not 2-D
        (np.zeros((0, 3)), np.array([])),  # empty
    ],
)
def test_top1_accuracy_validates_input(scores, labels):
    with pytest.raises(ValueError):
        top1_accuracy(scores, labels)


@pytest.mark.parametrize(
    "num_bytes, expected",
    [
        (0, "0 B"),
        (999, "999 B"),
        (125112, "125 KB"),
        (483736, "484 KB"),
        (1498464, "1.50 MB"),
    ],
)
def test_human_size(num_bytes, expected):
    assert human_size(num_bytes) == expected


def test_human_size_negative():
    with pytest.raises(ValueError):
        human_size(-1)


def test_compression_ratio():
    assert compression_ratio(1498464, 125112) == pytest.approx(11.98, abs=0.01)
    with pytest.raises(ValueError):
        compression_ratio(10, 0)


def _result(name, size, acc=None, mean=None):
    latency = summarize_latencies([mean]) if mean is not None else None
    return BenchmarkResult(name, "onnx", f"/m/{name}", size, acc, latency)


def test_format_results_table_relative_to_first_row():
    table = format_results_table(
        [_result("Keras", 1000, 0.99, 4.0), _result("ONNX", 250, None, None)]
    )
    lines = table.splitlines()
    assert lines[0].startswith("| Model |")
    assert "| Keras | onnx | 1 KB | 1.0x | 99.00% | 4.000 | 4.000 |" in lines
    assert "| ONNX | onnx | 250 B | 4.0x | n/a | n/a | n/a |" in lines


def test_format_results_table_empty():
    assert format_results_table([]).count("\n") == 1


def test_benchmark_result_to_dict_flattens_latency():
    d = _result("Keras", 1000, 0.5, 4.0).to_dict()
    assert d["latency_mean_ms"] == 4.0
    assert d["fps"] == pytest.approx(250.0)
    assert "latency" not in d
    empty = _result("x", 1).to_dict()
    assert empty["latency_mean_ms"] is None and empty["fps"] is None
