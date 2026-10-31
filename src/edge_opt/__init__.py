"""edge_opt: shrink and speed up a Keras CNN for edge devices.

The package turns a trained Keras digit-recognition CNN into smaller, faster
formats (frozen TensorFlow graph, optimized graph, TensorFlow Lite, ONNX) and
measures every version with the same accuracy / latency / size benchmark.

Modules:

* ``paths``      - where model files live (configurable, no hard-coded paths)
* ``data``       - MNIST loading and preprocessing
* ``train``      - build and train the baseline CNN
* ``convert``    - every format conversion step in the pipeline
* ``runners``    - a tiny "predict one image" wrapper per model format
* ``benchmark``  - pure-Python accuracy / latency / size helpers
* ``evaluate``   - glue that runs the benchmark against a model file
* ``cli``        - the ``edge-opt`` command line entry point

Heavy frameworks (TensorFlow, ONNX Runtime, TFLite) are imported lazily inside
the functions that need them, so ``import edge_opt`` stays cheap and the pure
helpers can be tested without any ML framework installed.
"""

__version__ = "0.1.0"
