import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_ml_service_dir = os.path.abspath(os.path.join(_current_dir, ".."))
if _ml_service_dir not in sys.path:
    sys.path.insert(0, _ml_service_dir)

import json
import argparse
import tensorflow as tf
import config
from training.base_trainer import (
    get_preprocessing_function,
    get_data_augmentation,
    load_dataset,
    calculate_class_weights,
    build_transfer_model,
    plot_and_save_training_curves
)

def train_efficientnetb0(
    train_dir: str = None,
    valid_dir: str = None,
    epochs: int = 5,
    fine_tune_epochs: int = 3,
    batch_size: int = 16,
    lr: float = 1e-3,
    fine_tune_lr: float = 1e-5,
    fine_tune_layers: int = 25,
    max_samples_per_class: int = 60,
    save_dir: str = None
):
    train_dir = train_dir or config.TRAIN_DIR
    valid_dir = valid_dir or config.VALID_DIR
    save_dir = save_dir or os.path.join(config.MODELS_DIR, "efficientnetb0")
    os.makedirs(save_dir, exist_ok=True)

    print("=================================================================")
    print("      Training EfficientNetB0 for Plant Pathology Detection")
    print("=================================================================")
    print(f"Train Directory:       {train_dir}")
    print(f"Valid Directory:       {valid_dir}")
    print(f"Max Samples/Class:     {max_samples_per_class} (0 = full dataset)")
    print(f"Batch Size:            {batch_size}")
    print(f"Frozen Epochs:         {epochs} (lr={lr})")
    print(f"Fine-tune Epochs:      {fine_tune_epochs} (lr={fine_tune_lr}, unfreezing {fine_tune_layers} layers)")
    print("-----------------------------------------------------------------")

    # 1. Load Dataset
    train_ds, class_names = load_dataset(
        train_dir,
        batch_size=batch_size,
        max_samples_per_class=max_samples_per_class,
        shuffle=True
    )
    val_max = max(10, int(max_samples_per_class * 0.25)) if max_samples_per_class > 0 else 0
    valid_ds, _ = load_dataset(
        valid_dir,
        batch_size=batch_size,
        max_samples_per_class=val_max,
        shuffle=False
    )
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")

    # 2. Compute Class Weights
    class_weights = calculate_class_weights(train_dir, num_classes, max_samples_per_class=max_samples_per_class)

    # 3. Preprocessing (EfficientNetB0: [0, 255] float pass-through) + Augmentation (Train only)
    preprocess_fn = get_preprocessing_function("efficientnetb0")
    augmentation = get_data_augmentation()

    def prep_train(img, lbl):
        img = augmentation(img, training=True)
        img = preprocess_fn(img)
        return img, lbl

    def prep_val(img, lbl):
        img = preprocess_fn(img)
        return img, lbl

    train_ds = train_ds.map(prep_train, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.map(prep_val, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    # 4. Construct Architecture
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=config.INPUT_SHAPE,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    model = build_transfer_model(
        base_model=base_model,
        num_classes=num_classes,
        model_name="leafscan_efficientnetb0",
        dropout_rate=0.3,
        dense_units=512
    )

    # 5. Phase 1: Train Classification Head (Frozen Backbone)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )

    checkpoint_path = os.path.join(save_dir, "efficientnetb0_model.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=False, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1)
    ]

    print("\n[Phase 1] Training classification head with frozen EfficientNetB0 backbone...")
    history1 = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # 6. Phase 2: Fine-Tuning Top Layers
    if fine_tune_epochs > 0:
        print(f"\n[Phase 2] Unfreezing top {fine_tune_layers} layers of EfficientNetB0...")
        base_model.trainable = True
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False
        # Preserve BatchNorm in non-training mode
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
            loss="categorical_crossentropy",
            metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
        )

        history2 = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=fine_tune_epochs,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
    else:
        history2 = None

    # 7. Merge Histories & Save Artifacts
    combined_history = {}
    for k in history1.history:
        combined_history[k] = [float(v) for v in history1.history[k]]
        if history2 and k in history2.history:
            combined_history[k].extend([float(v) for v in history2.history[k]])

    history_json_path = os.path.join(config.REPORTS_DIR, "training_histories", "efficientnetb0_history.json")
    plot_png_path = os.path.join(config.REPORTS_DIR, "training_histories", "efficientnetb0_curves.png")
    os.makedirs(os.path.dirname(history_json_path), exist_ok=True)

    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(combined_history, f, indent=2)

    plot_and_save_training_curves(combined_history, plot_png_path, "EfficientNetB0")

    if not os.path.exists(checkpoint_path):
        model.save(checkpoint_path)

    # Save model metadata
    meta = {
        "architecture": "EfficientNetB0",
        "target_conv_layer": "top_activation",
        "preprocessing": "efficientnet",
        "model_file": checkpoint_path,
        "input_shape": list(config.INPUT_SHAPE),
        "num_classes": num_classes,
        "class_names": class_names,
        "final_val_accuracy": combined_history.get("val_accuracy", [0])[-1],
        "final_val_loss": combined_history.get("val_loss", [0])[-1]
    }
    with open(os.path.join(save_dir, "architecture_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[EfficientNetB0 Complete] Model saved: {checkpoint_path}")
    print(f"Final Val Accuracy: {meta['final_val_accuracy']:.4f}")
    return model, meta

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples-per-class", type=int, default=60)
    args = parser.parse_args()

    train_efficientnetb0(
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        batch_size=args.batch_size,
        max_samples_per_class=args.max_samples_per_class
    )
