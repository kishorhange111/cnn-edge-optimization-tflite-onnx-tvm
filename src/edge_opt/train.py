"""Build and train the baseline digit-recognition CNN on MNIST.

This is the starting point of the pipeline: the Keras ``.h5`` file written
here is what every later step converts and optimizes.

Architecture (119,652 parameters, as printed in the project report)::

    conv1 (25 filters) -> maxpool1 -> conv2 (64) -> maxpool2 -> conv3 (64)
    -> maxpool3 -> flatten -> dense1 (64) -> dense2 (32) -> dense_output (10)

The layer names matter: the conversion and evaluation steps look tensors up by
name (``conv1_input``, ``dense_output/Softmax``), so do not rename layers
without updating :mod:`edge_opt.convert` and :mod:`edge_opt.runners`.
"""

from __future__ import annotations

import os
from pathlib import Path

from edge_opt.data import IMAGE_SIZE, load_mnist, preprocess

#: Number of epochs the original ``train.py`` used.
DEFAULT_EPOCHS = 2


def build_model(summary: bool = True):
    """Create and compile the CNN (unchanged from the original project).

    Args:
        summary: Print the Keras layer table, as the original script did.

    Returns:
        A compiled ``tf.keras.Sequential`` model.
    """
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

    model = Sequential()
    # 'same' padding keeps the 28x28 spatial size through each conv; pooling
    # halves it (28 -> 14 -> 7 -> 4) so the flattened vector stays small.
    model.add(
        Conv2D(
            25,
            (3, 3),
            strides=1,
            name="conv1",
            padding="same",
            activation="relu",
            kernel_initializer="he_normal",  # good default for ReLU layers
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
        )
    )
    model.add(MaxPooling2D((2, 2), strides=2, padding="same", name="maxpool1"))
    model.add(
        Conv2D(
            64,
            (3, 3),
            strides=1,
            name="conv2",
            padding="same",
            activation="relu",
            kernel_initializer="he_normal",
        )
    )
    model.add(MaxPooling2D((2, 2), strides=2, name="maxpool2", padding="same"))
    model.add(
        Conv2D(
            64,
            (3, 3),
            strides=1,
            name="conv3",
            padding="same",
            activation="relu",
            kernel_initializer="he_normal",
        )
    )
    model.add(MaxPooling2D((2, 2), strides=2, name="maxpool3", padding="same"))
    model.add(Flatten(name="flatten"))
    model.add(Dense(64, name="dense1", activation="relu"))
    model.add(Dense(32, name="dense2", activation="relu"))
    # Softmax output: one probability per digit 0-9.
    model.add(Dense(10, name="dense_output", activation="softmax"))
    # Sparse loss because labels are plain integers (not one-hot vectors).
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    if summary:
        model.summary()
    return model


def train(
    output_path: str | os.PathLike[str],
    epochs: int = DEFAULT_EPOCHS,
    mnist_path: str | os.PathLike[str] | None = None,
) -> Path:
    """Train the CNN on MNIST and save it as a Keras ``.h5`` file.

    Args:
        output_path: Where to write the ``.h5`` model.
        epochs: Passes over the 60,000 training images.
        mnist_path: Optional local ``mnist.npz`` instead of downloading.

    Returns:
        The path the model was saved to.
    """
    import tensorflow as tf

    print(f"TensorFlow {tf.__version__}")
    (x_train, y_train), (x_test, y_test) = load_mnist(mnist_path)
    x_train, x_test = preprocess(x_train), preprocess(x_test)

    model = build_model()
    model.fit(x_train, y_train, epochs=epochs)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    print(f"Saved Keras model to {output_path}")
    return output_path
