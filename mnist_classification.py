"""Trains a simple deep NN on MNIST (TensorFlow 2.x / Keras 3 compatible).

Original Keras example: ~98% test accuracy after a few epochs on CPU/GPU.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

batch_size = 512
num_classes = 10
epochs = 10

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        layers.Dense(512, activation="relu", input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
model.summary()

model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test),
)
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
