from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = SCRIPT_DIR if (SCRIPT_DIR / "chest_xray").is_dir() else SCRIPT_DIR.parent.parent
DEFAULT_TRAIN = _ROOT / "chest_xray" / "train"
DEFAULT_TEST = _ROOT / "chest_xray" / "test"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

BATCH_SIZE = 12
EPOCHS = 5
IMG_SIZE = 128


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_dir, test_dir = DEFAULT_TRAIN, DEFAULT_TEST
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise SystemExit(f"Missing data:\n  {train_dir}\n  {test_dir}")

    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="both",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        labels="inferred",
        shuffle=True,
    )
    test_ds = keras.utils.image_dataset_from_directory(
        test_dir,
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        labels="inferred",
        shuffle=False,
    )
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print("Classes:", class_names)

    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

    model = keras.Sequential(
        [
            layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            layers.Rescaling(1.0 / 255.0),
            layers.Conv2D(16, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(2),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(2),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(2),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    ckpt = OUTPUT_DIR / "pneumonia.keras"
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                filepath=str(ckpt),
                save_best_only=True,
                monitor="val_accuracy",
                mode="max",
            )
        ],
        verbose=1,
    )

    loss, acc = model.evaluate(test_ds, verbose=1)
    print(f"Test loss={loss:.4f} acc={acc:.4f}")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(history.history["accuracy"], label="train")
    ax.plot(history.history["val_accuracy"], label="val")
    ax.set_title("Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "history_accuracy.png", dpi=120)
    plt.close(fig)
    print(f"Saved plots under {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
