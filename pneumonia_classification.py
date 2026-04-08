from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model

SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = SCRIPT_DIR if (SCRIPT_DIR / "chest_xray").is_dir() else SCRIPT_DIR.parent.parent
DEFAULT_TRAIN = _ROOT / "chest_xray" / "train"
DEFAULT_TEST = _ROOT / "chest_xray" / "test"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 6
SEED = 123


def load_ds(train_dir: Path, test_dir: Path):
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
    names = train_ds.class_names
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, test_ds, names


def build_model(num_classes: int) -> Model:
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    # float rgb in 0 to 255 from image_dataset
    x = layers.Lambda(lambda t: preprocess_input(t))(inputs)
    base = EfficientNetB0(include_top=False, weights="imagenet", name="efficientnetb0")
    base.trainable = False
    x = base(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation="softmax")(x)
    return Model(inputs, out)


def main() -> None:
    tf.keras.utils.set_random_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_dir, test_dir = DEFAULT_TRAIN, DEFAULT_TEST
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise SystemExit(f"Missing data:\n  {train_dir}\n  {test_dir}")

    train_ds, val_ds, test_ds, class_names = load_ds(train_dir, test_dir)
    num_classes = len(class_names)
    model = build_model(num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                str(OUTPUT_DIR / "eff_nested.keras"),
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
            )
        ],
        verbose=1,
    )
    model.evaluate(test_ds, verbose=1)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(hist.history["val_accuracy"], label="val")
    ax.set_title("Nested EfficientNet + preprocess_input")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "val_acc_nested.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
