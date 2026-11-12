import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
import gc
import json
import time
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# Ensure ml-service root on sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_ml_service_dir = os.path.abspath(os.path.join(_current_dir, ".."))
if _ml_service_dir not in sys.path:
    sys.path.insert(0, _ml_service_dir)

import config
from training.base_trainer import (
    get_preprocessing_function,
    calculate_class_weights,
    build_transfer_model,
    plot_and_save_training_curves
)

ADV_MODELS_DIR = os.path.join(config.MODELS_DIR, "advanced_experiments")
ADV_REPORTS_DIR = os.path.join(config.REPORTS_DIR, "advanced_experiments")
os.makedirs(ADV_MODELS_DIR, exist_ok=True)
os.makedirs(ADV_REPORTS_DIR, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def get_advanced_augmentation(img_size=(224, 224)):
    """
    Advanced data augmentation designed to resolve early blight and target spot confusions.
    Provides rotation, zoom, horizontal/vertical flips, and contrast/brightness variations.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", name="aug_flips"),
        tf.keras.layers.RandomRotation(0.12, fill_mode="nearest", name="aug_rotation"),
        tf.keras.layers.RandomZoom(height_factor=(-0.10, 0.10), fill_mode="nearest", name="aug_zoom"),
        tf.keras.layers.RandomTranslation(height_factor=0.06, width_factor=0.06, fill_mode="nearest", name="aug_trans"),
        tf.keras.layers.RandomContrast(0.12, name="aug_contrast"),
        tf.keras.layers.RandomBrightness(0.10, name="aug_brightness"),
    ], name=f"adv_aug_{img_size[0]}x{img_size[1]}")

def build_datasets(
    train_dir: str,
    valid_dir: str,
    arch_type: str = "mobilenetv2",
    img_size: tuple = (224, 224),
    batch_size: int = 16,
    train_batches: int = 60,
    val_batches: int = 25,
    seed: int = 42
):
    """
    Builds robust, high-performance streaming datasets using native image_dataset_from_directory.
    """
    raw_train = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
        seed=seed
    )
    raw_val = tf.keras.utils.image_dataset_from_directory(
        valid_dir,
        labels="inferred",
        label_mode="categorical",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False
    )
    class_names = raw_train.class_names
    preprocess_fn = get_preprocessing_function(arch_type)

    def prep_train(imgs, lbls):
        imgs = tf.image.random_flip_left_right(imgs)
        imgs = tf.image.random_flip_up_down(imgs)
        imgs = tf.image.random_brightness(imgs, max_delta=0.08)
        imgs = tf.image.random_contrast(imgs, lower=0.9, upper=1.1)
        imgs = preprocess_fn(imgs)
        return imgs, lbls

    def prep_val(imgs, lbls):
        imgs = preprocess_fn(imgs)
        return imgs, lbls

    train_ds = raw_train.take(train_batches).map(prep_train).prefetch(tf.data.AUTOTUNE)
    val_ds = raw_val.take(val_batches).map(prep_val).prefetch(tf.data.AUTOTUNE)

    n_train = train_batches * batch_size
    n_val = val_batches * batch_size

    return train_ds, val_ds, class_names, n_train, n_val

def evaluate_on_dataset(model, ds, class_names):
    """
    Evaluates model predictions and returns comprehensive metrics.
    """
    y_true, y_pred_probs = [], []
    start_t = time.perf_counter()
    sample_count = 0

    for imgs, lbls in ds:
        preds = model.predict_on_batch(imgs)
        y_pred_probs.append(preds.numpy() if hasattr(preds, 'numpy') else np.array(preds))
        y_true.append(np.argmax(lbls.numpy(), axis=1))
        sample_count += len(lbls)

    lat_ms = (time.perf_counter() - start_t) / max(1, sample_count) * 1000
    y_true = np.concatenate(y_true, axis=0)
    y_pred_probs = np.concatenate(y_pred_probs, axis=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(class_names))), average=None, zero_division=0)

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": round(float(p_cls[i]), 4),
            "recall": round(float(r_cls[i]), 4),
            "f1_score": round(float(f1_cls[i]), 4)
        }

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_wt), 4),
        "avg_latency_ms": round(float(lat_ms), 2),
        "samples": sample_count,
        "per_class": per_class,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_probs": y_pred_probs
    }

def run_experiment_A_finetuning(champion_path, train_ds, val_ds, class_names):
    """
    Experiment A: Fine-Tuning Optimization from Champion.
    Freezes BatchNorm, unfreezes top 30 conv layers, trains at LR 1e-5.
    """
    print("\n" + "="*70)
    print("RUNNING EXPERIMENT A: MobileNetV2 Fine-Tuning (Frozen BN, LR=1e-5)")
    print("="*70)
    set_seed(42)

    base = tf.keras.applications.MobileNetV2(input_shape=config.INPUT_SHAPE, include_top=False, weights=None)
    model = build_transfer_model(base, len(class_names), "MobileNetV2_ExpA", dropout_rate=0.3, dense_units=512)
    model.load_weights(champion_path)

    base.trainable = True
    for layer in base.layers[:-15]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    trainable_count = sum(len(l.trainable_weights) for l in model.layers if l.trainable)
    print(f"Model layers: {len(model.layers)}, Trainable weight tensors: {trainable_count}")

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    exp_dir = os.path.join(ADV_MODELS_DIR, "expA_finetune")
    os.makedirs(exp_dir, exist_ok=True)
    best_ckpt = os.path.join(exp_dir, "expA_finetune_best.keras")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(best_ckpt, monitor="val_loss", save_best_only=True, verbose=1)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3,
        callbacks=callbacks,
        verbose=2
    )

    # Load best checkpoint and evaluate
    best_model = tf.keras.models.load_model(best_ckpt, compile=False)
    metrics = evaluate_on_dataset(best_model, val_ds, class_names)

    # Plot curves
    plot_path = os.path.join(ADV_REPORTS_DIR, "expA_finetune_curves.png")
    plot_and_save_training_curves(history.history, plot_path, "Exp A: Fine-Tuned MobileNetV2")

    metrics["model_path"] = best_ckpt
    metrics["plot_path"] = plot_path
    metrics["history"] = {k: [float(v) for v in vals] for k, vals in history.history.items()}

    with open(os.path.join(ADV_REPORTS_DIR, "expA_metrics.json"), "w") as f:
        json_save = {k: v for k, v in metrics.items() if k not in ["y_true", "y_pred", "y_pred_probs"]}
        json.dump(json_save, f, indent=2)

    print(f"Exp A Val Accuracy: {metrics['accuracy']*100:.2f}%, Val Macro F1: {metrics['macro_f1']*100:.2f}%")
    return best_model, metrics

def run_experiment_B_native_resolution(train_dir, valid_dir, class_names):
    """
    Experiment B: Native Resolution (256x256) Training.
    Trains MobileNetV2 directly at native 256x256 resolution to preserve small lesion details.
    """
    print("\n" + "="*70)
    print("RUNNING EXPERIMENT B: Native 256x256 Resolution Training")
    print("="*70)
    set_seed(42)

    # Build 256x256 datasets
    train_ds_256, val_ds_256, _, _, _ = build_datasets(
        train_dir, valid_dir,
        arch_type="mobilenetv2",
        img_size=(256, 256),
        batch_size=16,
        train_batches=60,
        val_batches=25
    )

    base = tf.keras.applications.MobileNetV2(
        input_shape=(256, 256, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    model = build_transfer_model(base, len(class_names), "MobileNetV2_256x256", dropout_rate=0.3, dense_units=512)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    exp_dir = os.path.join(ADV_MODELS_DIR, "expB_highres")
    os.makedirs(exp_dir, exist_ok=True)
    best_ckpt = os.path.join(exp_dir, "expB_highres_best.keras")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(best_ckpt, monitor="val_loss", save_best_only=True, verbose=1)
    ]

    history = model.fit(
        train_ds_256,
        validation_data=val_ds_256,
        epochs=3,
        callbacks=callbacks,
        verbose=2
    )

    best_model = tf.keras.models.load_model(best_ckpt, compile=False)
    metrics = evaluate_on_dataset(best_model, val_ds_256, class_names)

    plot_path = os.path.join(ADV_REPORTS_DIR, "expB_highres_curves.png")
    plot_and_save_training_curves(history.history, plot_path, "Exp B: Native 256x256 MobileNetV2")

    metrics["model_path"] = best_ckpt
    metrics["plot_path"] = plot_path
    metrics["history"] = {k: [float(v) for v in vals] for k, vals in history.history.items()}

    with open(os.path.join(ADV_REPORTS_DIR, "expB_metrics.json"), "w") as f:
        json_save = {k: v for k, v in metrics.items() if k not in ["y_true", "y_pred", "y_pred_probs"]}
        json.dump(json_save, f, indent=2)

    print(f"Exp B Val Accuracy: {metrics['accuracy']*100:.2f}%, Val Macro F1: {metrics['macro_f1']*100:.2f}%")
    return best_model, metrics, val_ds_256

def run_experiment_C_hard_class_focus(champion_path, train_ds, val_ds, class_names, train_dir):
    """
    Experiment C: Hard-Class Focus with Dampened Class Weighting and Focal Loss.
    Directly tackles Tomato Early Blight, Target Spot, and Potato healthy.
    """
    print("\n" + "="*70)
    print("RUNNING EXPERIMENT C: Hard-Class Focus (Dampened Class Weighting)")
    print("="*70)
    set_seed(42)

    raw_weights = calculate_class_weights(train_dir, num_classes=len(class_names))
    # Dampen weights (square root) and clip between 0.6 and 2.5 to stabilize training
    dampened_weights = {
        i: float(np.clip(np.sqrt(w), 0.6, 2.5))
        for i, w in raw_weights.items()
    }
    print("Dampened Class Weights:")
    for i, name in enumerate(class_names):
        print(f"  {name:45s}: {dampened_weights[i]:.2f}")

    base = tf.keras.applications.MobileNetV2(input_shape=config.INPUT_SHAPE, include_top=False, weights=None)
    model = build_transfer_model(base, len(class_names), "MobileNetV2_ExpC", dropout_rate=0.3, dense_units=512)
    model.load_weights(champion_path)

    base.trainable = True
    for layer in base.layers[:-15]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    exp_dir = os.path.join(ADV_MODELS_DIR, "expC_hard_class_focus")
    os.makedirs(exp_dir, exist_ok=True)
    best_ckpt = os.path.join(exp_dir, "expC_hard_class_best.keras")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(best_ckpt, monitor="val_loss", save_best_only=True, verbose=1)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3,
        class_weight=dampened_weights,
        callbacks=callbacks,
        verbose=2
    )

    best_model = tf.keras.models.load_model(best_ckpt, compile=False)
    metrics = evaluate_on_dataset(best_model, val_ds, class_names)

    plot_path = os.path.join(ADV_REPORTS_DIR, "expC_curves.png")
    plot_and_save_training_curves(history.history, plot_path, "Exp C: Hard-Class Focus MobileNetV2")

    metrics["model_path"] = best_ckpt
    metrics["plot_path"] = plot_path
    metrics["history"] = {k: [float(v) for v in vals] for k, vals in history.history.items()}

    with open(os.path.join(ADV_REPORTS_DIR, "expC_metrics.json"), "w") as f:
        json_save = {k: v for k, v in metrics.items() if k not in ["y_true", "y_pred", "y_pred_probs"]}
        json.dump(json_save, f, indent=2)

    print(f"Exp C Val Accuracy: {metrics['accuracy']*100:.2f}%, Val Macro F1: {metrics['macro_f1']*100:.2f}%")
    return best_model, metrics

def run_experiment_E_ensemble(models, weights, val_ds, class_names):
    """
    Experiment E: Soft Probability Averaging Ensemble.
    """
    print("\n" + "="*70)
    print("RUNNING EXPERIMENT E: Model Ensemble (Soft Averaging)")
    print("="*70)

    all_model_preds = []
    y_true = []
    sample_count = 0
    start_t = time.perf_counter()

    for imgs, lbls in val_ds:
        batch_preds = [m.predict_on_batch(imgs) for m in models]
        weighted_batch = np.zeros_like(batch_preds[0])
        for w, p in zip(weights, batch_preds):
            weighted_batch += w * (p.numpy() if hasattr(p, 'numpy') else np.array(p))
        all_model_preds.append(weighted_batch)
        y_true.append(np.argmax(lbls.numpy(), axis=1))
        sample_count += len(lbls)

    lat_ms = (time.perf_counter() - start_t) / max(1, sample_count) * 1000
    y_true = np.concatenate(y_true, axis=0)
    y_pred_probs = np.concatenate(all_model_preds, axis=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": round(float(p_cls[i]), 4),
            "recall": round(float(r_cls[i]), 4),
            "f1_score": round(float(f1_cls[i]), 4)
        }

    metrics = {
        "model_name": "Ensemble (Champion + Exp A + Exp C)",
        "accuracy": round(acc, 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_wt), 4),
        "avg_latency_ms": round(float(lat_ms), 2),
        "samples": sample_count,
        "per_class": per_class,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_probs": y_pred_probs
    }

    with open(os.path.join(ADV_REPORTS_DIR, "expE_ensemble_metrics.json"), "w") as f:
        json_save = {k: v for k, v in metrics.items() if k not in ["y_true", "y_pred", "y_pred_probs"]}
        json.dump(json_save, f, indent=2)

    print(f"Exp E Val Accuracy: {metrics['accuracy']*100:.2f}%, Val Macro F1: {metrics['macro_f1']*100:.2f}%")
    return metrics
