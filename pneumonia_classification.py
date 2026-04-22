from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score
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



@keras.utils.register_keras_serializable(package="cv_ca2")
class EfficientNetPreprocess(layers.Layer):
    def call(self, x):
        return preprocess_input(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chest X-ray classification with transfer learning.")
    p.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--test-dir", type=Path, default=DEFAULT_TEST)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--quick", action="store_true", help="Fewer epochs for a fast sanity check.")
    p.add_argument("--no-train", action="store_true", help="Load saved model and only evaluate / visualize.")
    p.add_argument("--weights", type=Path, default=None, help="Optional .keras path (default: outputs/best_chest_xray.keras).")
    p.add_argument("--no-gradcam", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)


def count_files_per_class(data_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sub in sorted(data_dir.iterdir()):
        if not sub.is_dir():
            continue
        n = sum(1 for f in sub.iterdir() if f.suffix.lower() in {".jpeg", ".jpg", ".png"})
        counts[sub.name] = n
    return counts


def make_augmentation() -> keras.Sequential:
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.12),
            layers.RandomContrast(0.12),
            layers.RandomTranslation(0.05, 0.05),
        ],
        name="data_augmentation",
    )


def load_datasets(
    train_dir: Path,
    test_dir: Path,
    img_size: int,
    batch_size: int,
    seed: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="both",
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        labels="inferred",
        shuffle=True,
    )
    test_ds = keras.utils.image_dataset_from_directory(
        test_dir,
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        labels="inferred",
        shuffle=False,
    )
    class_names = list(train_ds.class_names)
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, test_ds, class_names


def class_weights_for_dataset(train_ds: tf.data.Dataset, num_classes: int) -> dict[int, float]:
    labels = np.concatenate([y.numpy() for _, y in train_ds], axis=0)
    classes = np.arange(num_classes)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {int(i): float(w) for i, w in enumerate(cw)}


def build_model(img_size: int, num_classes: int, augment: keras.Sequential) -> Model:
    inputs = Input(shape=(img_size, img_size, 3), name="image")
    # augmentation only runs during training not validation or test
    x = augment(inputs)
    x = EfficientNetPreprocess(name="efficientnet_preprocess")(x)
    # use backbone as a layer so it stays as a nested model for fine tuning and grad cam
    base = EfficientNetB0(include_top=False, weights="imagenet")
    base.trainable = False
    x = base(x)
    x = GlobalAveragePooling2D(name="gap")(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.45)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs, name="chest_xray_classifier")


def get_inner_backbone(model: Model) -> keras.Model:
    for layer in model.layers:
        if isinstance(layer, keras.Model) and "efficientnet" in layer.name.lower():
            return layer
    raise ValueError("Could not locate EfficientNet backbone in model.layers.")


def get_last_conv_layer_name(backbone: keras.Model) -> str:
    for layer in backbone.layers:
        if layer.name == "top_conv":
            return "top_conv"
    for layer in reversed(backbone.layers):
        if isinstance(layer, (layers.Conv2D, layers.DepthwiseConv2D, layers.SeparableConv2D)):
            return layer.name
    raise RuntimeError("Could not find a convolutional layer in the backbone.")


def _head_layers_after_backbone(model: Model) -> list:
    idx = next(
        i
        for i, layer in enumerate(model.layers)
        if isinstance(layer, keras.Model) and "efficientnet" in layer.name.lower()
    )
    return model.layers[idx + 1 :]


def _gradcam_inner_and_tail(backbone: keras.Model, last_conv_layer_name: str) -> tuple[Model, Model]:
    top_conv = backbone.get_layer(last_conv_layer_name)
    inner = Model(backbone.input, top_conv.output, name="gradcam_to_top_conv")
    tail = Model(top_conv.output, backbone.output, name="gradcam_after_top_conv")
    return inner, tail


def make_gradcam_heatmap(
    img_array: np.ndarray | tf.Tensor,
    model: Model,
    inner: Model,
    bb_tail: Model,
    pred_index: int | None = None,
) -> np.ndarray:
    img_array = tf.cast(img_array, tf.float32)
    aug = model.get_layer("data_augmentation")
    prep = model.get_layer("efficientnet_preprocess")
    head = _head_layers_after_backbone(model)

    with tf.GradientTape() as tape:
        x = aug(img_array, training=False)
        x = prep(x)
        conv_outputs = inner(x, training=False)
        x = bb_tail(conv_outputs, training=False)
        for layer in head:
            if isinstance(layer, (layers.BatchNormalization, layers.Dropout)):
                x = layer(x, training=False)
            else:
                x = layer(x)
        predictions = x
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise RuntimeError("Grad-CAM gradient is None (graph not differentiable on conv features).")
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()


def save_gradcam_examples(
    model: Model,
    test_ds: tf.data.Dataset,
    class_names: list[str],
    out_dir: Path,
    img_size: int,
    n: int = 3,
) -> None:
    backbone = get_inner_backbone(model)
    last_conv = get_last_conv_layer_name(backbone)
    inner, bb_tail = _gradcam_inner_and_tail(backbone, last_conv)
    out_dir.mkdir(parents=True, exist_ok=True)
    taken = 0
    for images, labels in test_ds:
        for i in range(images.shape[0]):
            if taken >= n:
                return
            img = images[i : i + 1]
            label = int(labels[i].numpy())
            heatmap = make_gradcam_heatmap(img, model, inner, bb_tail)
            heatmap = np.uint8(255 * heatmap)
            heatmap = np.expand_dims(heatmap, axis=-1)
            heatmap = tf.image.resize(heatmap, (img_size, img_size)).numpy().squeeze()
            heatmap_color = plt.cm.jet(heatmap / 255.0)[:, :, :3]
            raw = img[0].numpy()
            raw = np.clip(raw, 0, 255).astype(np.float32)
            superimposed = 0.55 * (raw / 255.0) + 0.45 * heatmap_color
            superimposed = np.clip(superimposed, 0, 1)
            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            ax[0].imshow(raw.astype(np.uint8))
            ax[0].set_title(f"True: {class_names[label]}")
            ax[0].axis("off")
            ax[1].imshow(superimposed)
            ax[1].set_title("Grad-CAM overlay")
            ax[1].axis("off")
            fig.tight_layout()
            fig.savefig(out_dir / f"gradcam_sample_{taken}.png", dpi=150)
            plt.close(fig)
            taken += 1


def collect_predictions(model: Model, ds: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.concatenate([y.numpy() for _, y in ds], axis=0)
    probs = np.asarray(model.predict(ds, verbose=0))
    y_pred = np.argmax(probs, axis=1)
    return y_true, y_pred, probs


def write_report_metrics(
    out_dir: Path,
    *,
    test_loss: float,
    test_acc: float,
    class_names: list[str],
    report_dict: dict,
    macro_f1: float,
    weighted_f1: float,
    binary_metrics: dict[str, float],
) -> None:
    per_class = {}
    for name in class_names:
        block = report_dict.get(name, {})
        per_class[name] = {
            "precision": float(block.get("precision", 0.0)),
            "recall": float(block.get("recall", 0.0)),
            "f1-score": float(block.get("f1-score", 0.0)),
            "support": int(block.get("support", 0)),
        }
    payload = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "binary_sick_at_0.5": {k: float(v) for k, v in binary_metrics.items()},
    }
    path = out_dir / "report_metrics.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote metrics snapshot for report generator: {path}")


def sick_binary_masks(y: np.ndarray, normal_index: int) -> tuple[np.ndarray, np.ndarray]:
    true_sick = (y != normal_index).astype(int)
    return true_sick, np.array([normal_index])


def binary_metrics_from_scores(
    y_true: np.ndarray,
    y_prob_sick: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_hat = (y_prob_sick >= threshold).astype(int)
    tp = np.sum((y_hat == 1) & (y_true == 1))
    fp = np.sum((y_hat == 1) & (y_true == 0))
    fn = np.sum((y_hat == 0) & (y_true == 1))
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return {"precision_sick": prec, "recall_sick": rec, "f1_sick": f1}


def run_pipeline(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    train_dir = args.train_dir
    test_dir = args.test_dir
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise SystemExit(f"Train or test directory missing. Expected:\n  {train_dir}\n  {test_dir}")

    print("--- Files per class (train) ---")
    train_counts = count_files_per_class(train_dir)
    for k, v in train_counts.items():
        print(f"  {k}: {v}")
    print("--- Files per class (test) ---")
    test_counts = count_files_per_class(test_dir)
    for k, v in test_counts.items():
        print(f"  {k}: {v}")

    train_ds, val_ds, test_ds, class_names = load_datasets(
        train_dir, test_dir, args.img_size, args.batch_size, args.seed
    )
    num_classes = len(class_names)
    print("Class order:", class_names)

    augment = make_augmentation()
    class_weight = class_weights_for_dataset(train_ds, num_classes)
    print("Class weights (balanced):", class_weight)

    normal_index = class_names.index("NORMAL") if "NORMAL" in class_names else 0

    model = build_model(args.img_size, num_classes, augment)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.SparseCategoricalAccuracy(name="sparse_cat_acc"),
        ],
    )

    weights_path = args.weights or MODEL_PATH
    if args.no_train:
        if not weights_path.is_file():
            raise SystemExit(f"No weights at {weights_path}. Train first or pass --weights.")
        model = keras.models.load_model(
            weights_path,
            compile=True,
            custom_objects={"EfficientNetPreprocess": EfficientNetPreprocess},
        )
    else:
        if args.quick:
            e1, e2, patience = 4, 3, 2
        else:
            e1, e2, patience = 14, 10, 4

        ckpt = keras.callbacks.ModelCheckpoint(
            filepath=str(weights_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        )
        es1 = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
        rlr1 = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.4, patience=max(1, patience // 2))

        print("--- Phase 1: train classification head (backbone frozen) ---")
        t0 = time.perf_counter()
        hist1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=e1,
            class_weight=class_weight,
            callbacks=[ckpt, es1, rlr1],
            verbose=1,
        )
        t1 = time.perf_counter()

        backbone = get_inner_backbone(model)
        backbone.trainable = True
        freeze_until = max(0, len(backbone.layers) - 40)
        for layer in backbone.layers[:freeze_until]:
            layer.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=8e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        es2 = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
        rlr2 = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.4, patience=max(1, patience // 2))

        print("--- Phase 2: fine-tune last blocks of EfficientNetB0 ---")
        hist2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=e2,
            class_weight=class_weight,
            callbacks=[ckpt, es2, rlr2],
            verbose=1,
        )
        t2 = time.perf_counter()
        train_seconds = (t1 - t0) + (t2 - t1)
        with open(out / "training_time_seconds.txt", "w", encoding="utf-8") as f:
            f.write(f"phase1_seconds: {t1 - t0:.2f}\n")
            f.write(f"phase2_seconds: {t2 - t1:.2f}\n")
            f.write(f"total_fit_seconds: {train_seconds:.2f}\n")
        print(f"Total training time (fit): {train_seconds:.1f} s ({train_seconds / 60:.2f} min)")

        for h, name in [(hist1, "history_phase1"), (hist2, "history_phase2")]:
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].plot(h.history["accuracy"], label="train")
            ax[0].plot(h.history["val_accuracy"], label="val")
            ax[0].set_title("Accuracy")
            ax[0].legend()
            ax[1].plot(h.history["loss"], label="train")
            ax[1].plot(h.history["val_loss"], label="val")
            ax[1].set_title("Loss")
            ax[1].legend()
            fig.tight_layout()
            fig.savefig(out / f"{name}.png", dpi=150)
            plt.close(fig)

        if weights_path.is_file():
            model = keras.models.load_model(
                weights_path,
                compile=True,
                custom_objects={"EfficientNetPreprocess": EfficientNetPreprocess},
            )

    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"Test loss: {test_loss:.4f}  Test accuracy: {test_acc:.4f}")

    y_true, y_pred, probs = collect_predictions(model, test_ds)
    print("\n--- Classification report (test) ---")
    print(
        classification_report(
            y_true, y_pred, target_names=class_names, digits=4, zero_division=0
        )
    )

    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0, output_dict=True
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"Macro F1: {macro_f1:.4f}  Weighted F1: {weighted_f1:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix_test.png", dpi=150)
    plt.close(fig)

    p_sick = 1.0 - probs[:, normal_index]
    true_sick, _ = sick_binary_masks(y_true, normal_index)
    bm = binary_metrics_from_scores(true_sick, p_sick, threshold=0.5)
    print("\n--- Binary view: sick (BACTERIAL+VIRAL) vs NORMAL @0.5 on P(sick) ---")
    print({k: float(v) for k, v in bm.items()})

    write_report_metrics(
        out,
        test_loss=float(test_loss),
        test_acc=float(test_acc),
        class_names=class_names,
        report_dict=report_dict,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        binary_metrics=bm,
    )

    if not args.no_gradcam:
        save_gradcam_examples(model, test_ds, class_names, out / "gradcam", args.img_size, n=4)

    print(f"\nArtifacts written under: {out.resolve()}")


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
