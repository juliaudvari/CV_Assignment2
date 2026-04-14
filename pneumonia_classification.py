from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model

SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = SCRIPT_DIR if (SCRIPT_DIR / "chest_xray").is_dir() else SCRIPT_DIR.parent.parent
DEFAULT_TRAIN = _ROOT / "chest_xray" / "train"
DEFAULT_TEST = _ROOT / "chest_xray" / "test"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
MODEL_PATH = OUTPUT_DIR / "best_chest_xray.keras"

IMG_SIZE = 224
BATCH_SIZE = 16
SEED = 123


@keras.utils.register_keras_serializable(package="cv_ca2")
class EfficientNetPreprocess(layers.Layer):
    def call(self, x):
        return preprocess_input(x)


def make_augmentation() -> keras.Sequential:
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.12),
        ],
        name="data_augmentation",
    )


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


def class_weights_for_dataset(train_ds: tf.data.Dataset, num_classes: int) -> dict[int, float]:
    labels = np.concatenate([y.numpy() for _, y in train_ds], axis=0)
    classes = np.arange(num_classes)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {int(i): float(w) for i, w in enumerate(cw)}


def get_inner_backbone(model: Model) -> keras.Model:
    for layer in model.layers:
        if isinstance(layer, keras.Model) and "efficientnet" in layer.name.lower():
            return layer
    raise ValueError("No EfficientNet backbone found.")


def build_model(num_classes: int, augment: keras.Sequential) -> Model:
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = augment(inputs)
    x = EfficientNetPreprocess()(x)
    base = EfficientNetB0(include_top=False, weights="imagenet", name="efficientnetb0")
    base.trainable = False
    x = base(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.45)(x)
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
    cw = class_weights_for_dataset(train_ds, num_classes)

    aug = make_augmentation()
    model = build_model(num_classes, aug)
    model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    ckpt = keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    print("--- Phase 1 ---")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=8,
        class_weight=cw,
        callbacks=[ckpt, es],
        verbose=1,
    )

    backbone = get_inner_backbone(model)
    backbone.trainable = True
    freeze_until = max(0, len(backbone.layers) - 40)
    for layer in backbone.layers[:freeze_until]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(8e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("--- Phase 2 ---")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        class_weight=cw,
        callbacks=[ckpt, es],
        verbose=1,
    )

    if MODEL_PATH.is_file():
        model = keras.models.load_model(
            MODEL_PATH,
            custom_objects={"EfficientNetPreprocess": EfficientNetPreprocess},
        )

    model.evaluate(test_ds, verbose=1)
    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix_test.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
