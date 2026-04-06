from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = SCRIPT_DIR if (SCRIPT_DIR / "chest_xray").is_dir() else SCRIPT_DIR.parent.parent
DEFAULT_TRAIN = _ROOT / "chest_xray" / "train"
DEFAULT_TEST = _ROOT / "chest_xray" / "test"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

BATCH_SIZE = 16
EPOCHS = 10
IMG_SIZE = 128
SEED = 123


def main() -> None:
    tf.keras.utils.set_random_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_dir, test_dir = DEFAULT_TRAIN, DEFAULT_TEST
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise SystemExit(f"Missing data:\n  {train_dir}\n  {test_dir}")

    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="both",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        labels="inferred",
        shuffle=True,
    )
    test_ds = keras.utils.image_dataset_from_directory(
        test_dir,
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        labels="inferred",
        shuffle=False,
    )
    num_classes = len(train_ds.class_names)

    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

    model = keras.Sequential(
        [
            layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            layers.Rescaling(1.0 / 255.0),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    best_path = OUTPUT_DIR / "best_gap_scratch.keras"
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                filepath=str(best_path),
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
                verbose=1,
            ),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ],
        verbose=1,
    )

    model.evaluate(test_ds, verbose=1)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(history.history["val_accuracy"], label="val acc")
    ax.set_title("Val accuracy (GAP scratch)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "val_accuracy_gap.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
