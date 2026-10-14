"""Framework-free helpers for measuring accuracy, latency and model size.

Everything in this module is plain Python + NumPy so it can be unit-tested
without TensorFlow or ONNX Runtime. The evaluators in :mod:`edge_opt.evaluate`
pass in a ``predict`` callable and get comparable numbers for every format.

Note on timing: the original scripts used ``time.time()`` and averaged
``1 / t`` per iteration to get FPS. On Windows ``time.time()`` ticks in
steps of several milliseconds, so many iterations measured as ~0 ms and the
averaged FPS exploded into the millions (visible in the report's screenshots).
Here we use ``time.perf_counter()`` (a high-resolution clock) and derive FPS
from the mean latency instead.
"""

from __future__ import annotations

import math
import os
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LatencyStats:
    """Summary of repeated single-inference timings, all in milliseconds."""

    iterations: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float

    @property
    def fps(self) -> float:
        """Throughput implied by the mean latency (inferences per second)."""
        return 1000.0 / self.mean_ms if self.mean_ms > 0 else math.inf


def percentile(values: Sequence[float], pct: float) -> float:
    """Return the ``pct``-th percentile using linear interpolation.

    Matches ``numpy.percentile(values, pct)`` (default ``linear`` method) but
    is written out so the behaviour is obvious and dependency-free.

    Raises:
        ValueError: If ``values`` is empty or ``pct`` is outside ``[0, 100]``.
    """
    if not values:
        raise ValueError("percentile() needs at least one value")
    if not 0 <= pct <= 100:
        raise ValueError(f"pct must be in [0, 100], got {pct}")
    ordered = sorted(values)
    # Position of the percentile on a 0..n-1 index scale.
    rank = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def summarize_latencies(latencies_ms: Sequence[float]) -> LatencyStats:
    """Turn raw per-iteration timings into a :class:`LatencyStats`.

    Raises:
        ValueError: If no timings are given.
    """
    if not latencies_ms:
        raise ValueError("summarize_latencies() needs at least one timing")
    return LatencyStats(
        iterations=len(latencies_ms),
        mean_ms=statistics.fmean(latencies_ms),
        median_ms=statistics.median(latencies_ms),
        p95_ms=percentile(latencies_ms, 95),
        min_ms=min(latencies_ms),
        max_ms=max(latencies_ms),
    )


def measure_latency(
    fn: Callable[[], object],
    iterations: int = 100,
    warmup: int = 1,
    timer: Callable[[], float] = time.perf_counter,
) -> LatencyStats:
    """Call ``fn`` repeatedly and report how long each call took.

    Args:
        fn: Zero-argument callable that runs one inference.
        iterations: Number of timed calls. The original scripts used 100.
        warmup: Untimed calls made first. The first inference usually pays
            one-off costs (memory allocation, graph setup), which would skew the
            average; the original scripts also ran one inference before timing.
        timer: Clock returning seconds. Injectable so tests can use a fake.

    Returns:
        Latency summary in milliseconds.
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    for _ in range(warmup):
        fn()
    timings_ms = []
    for _ in range(iterations):
        start = timer()
        fn()
        timings_ms.append((timer() - start) * 1000.0)
    return summarize_latencies(timings_ms)


def top1_accuracy(scores: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of rows whose highest score matches the label.

    Works for both softmax probabilities and raw logits, because ``argmax`` is
    the same for either. (The frozen-graph evaluator reads the pre-softmax
    ``BiasAdd`` tensor, the others read softmax output.)

    Args:
        scores: Array of shape ``(N, num_classes)``.
        labels: Integer class ids of shape ``(N,)``.

    Raises:
        ValueError: On empty input or mismatched lengths.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels).reshape(-1)
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2-D (N, classes), got shape {scores.shape}")
    if len(scores) != len(labels):
        raise ValueError(f"{len(scores)} score rows but {len(labels)} labels")
    if len(labels) == 0:
        raise ValueError("cannot compute accuracy of an empty set")
    return float(np.mean(np.argmax(scores, axis=1) == labels))


def file_size_bytes(path: str | os.PathLike[str]) -> int:
    """Size of a model file on disk, in bytes."""
    return Path(path).stat().st_size


def human_size(num_bytes: int) -> str:
    """Format a byte count with decimal units, e.g. ``1498464 -> '1.50 MB'``.

    Decimal (1 KB = 1000 B) units are used because that is what file browsers
    and the project report show.
    """
    if num_bytes < 0:
        raise ValueError("size cannot be negative")
    if num_bytes < 1000:
        return f"{num_bytes} B"
    if num_bytes < 1000**2:
        return f"{num_bytes / 1000:.0f} KB"
    return f"{num_bytes / 1000**2:.2f} MB"


def compression_ratio(baseline_bytes: int, candidate_bytes: int) -> float:
    """How many times smaller ``candidate`` is than ``baseline`` (e.g. 12.0)."""
    if candidate_bytes <= 0:
        raise ValueError("candidate size must be positive")
    return baseline_bytes / candidate_bytes


@dataclass(frozen=True)
class BenchmarkResult:
    """Everything measured for one model file."""

    name: str
    model_format: str
    path: str
    size_bytes: int
    accuracy: float | None
    latency: LatencyStats | None

    def to_dict(self) -> dict[str, object]:
        """Flatten into a JSON-friendly dict.

        Latency fields are inlined as ``latency_<field>`` (``None`` when latency
        was not measured) so the output loads straight into a DataFrame.
        """
        out = asdict(self)
        latency = out.pop("latency") or {}
        for f in fields(LatencyStats):
            out[f"latency_{f.name}"] = latency.get(f.name)
        out["fps"] = self.latency.fps if self.latency else None
        return out


def format_results_table(results: Sequence[BenchmarkResult]) -> str:
    """Render results as a Markdown table, sizes relative to the first row.

    The first result is treated as the baseline (normally the Keras model),
    so the "Smaller" column reads as "x times smaller than Keras".
    """
    header = (
        "| Model | Format | Size | Smaller | Top-1 acc | Mean latency (ms) | p95 (ms) |\n"
        "|---|---|---:|---:|---:|---:|---:|"
    )
    if not results:
        return header
    baseline = results[0].size_bytes
    rows = [header]
    for r in results:
        acc = f"{r.accuracy * 100:.2f}%" if r.accuracy is not None else "n/a"
        mean = f"{r.latency.mean_ms:.3f}" if r.latency else "n/a"
        p95 = f"{r.latency.p95_ms:.3f}" if r.latency else "n/a"
        ratio = compression_ratio(baseline, r.size_bytes)
        rows.append(
            f"| {r.name} | {r.model_format} | {human_size(r.size_bytes)} | "
            f"{ratio:.1f}x | {acc} | {mean} | {p95} |"
        )
    return "\n".join(rows)
