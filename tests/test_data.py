"""Tests for MNIST preprocessing and local loading."""

import numpy as np

from edge_opt.data import load_mnist, load_test_set, preprocess


def test_preprocess_scales_and_reshapes():
    raw = np.full((2, 28, 28), 255, dtype=np.uint8)
    out = preprocess(raw)
    assert out.shape == (2, 28, 28, 1)
    assert out.dtype == np.float32
    assert out.max() == 1.0


def test_preprocess_accepts_channel_axis():
    assert preprocess(np.zeros((3, 28, 28, 1), np.uint8)).shape == (3, 28, 28, 1)


def test_load_mnist_from_local_file(fake_mnist):
    (x_train, y_train), (x_test, y_test) = load_mnist(fake_mnist)
    assert x_train.shape == (6, 28, 28)
    assert y_test.tolist() == [3, 1, 4, 1]


def test_load_test_set_limit(fake_mnist):
    images, labels = load_test_set(fake_mnist, limit=2)
    assert images.shape == (2, 28, 28, 1)
    assert labels.tolist() == [3, 1]
