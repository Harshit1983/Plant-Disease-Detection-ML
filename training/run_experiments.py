import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
import gc
import json
import time
import random
import shutil
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
    get_data_augmentation,
    get_preprocessing_function,
    calculate_class_weights,
    build_transfer_model,
    plot_and_save_training_curves
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def build_data_pipelines(
    train_dir: str,
    valid_dir: str,
    arch_type: str,
    use_augmentation: bool = True,
    samples_per_class_train: int = 100,
    samples_per_class_val: int = 30,
    batch_size: int = 32,
    seed: int = 42
):
    """
    Constructs high-performance cached training and validation tf.data pipelines.
    """
    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    num_classes = len(class_names)
    r = random.Random(seed)

    # 1. Collect Train File Paths
    train_fps, train_lbls = [], []
    for idx, c in enumerate(class_names):
        c_dir = os.path.join(train_dir, c)
        files = [
            os.path.join(c_dir, f) for f in sorted(os.listdir(c_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))
        ]
        r.shuffle(files)
        selected = files[:samples_per_class_train] if samples_per_class_train > 0 else files
        for fp in selected:
            train_fps.append(fp)
            lbl = np.zeros(num_classes, dtype=np.float32)
            lbl[idx] = 1.0
            train_lbls.append(lbl)

    # 2. Collect Valid File Paths
    val_fps, val_lbls = [], []
    for idx, c in enumerate(class_names):
        c_dir = os.path.join(valid_dir, c)
        files = [
            os.path.join(c_dir, f) for f in sorted(os.listdir(c_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))
        ]
        selected = files[:samples_per_class_val] if samples_per_class_val > 0 else files
        for fp in selected:
            val_fps.append(fp)
            lbl = np.zeros(num_classes, dtype=np.float32)
            lbl[idx] = 1.0
            val_lbls.append(lbl)

    preprocess_fn = get_preprocessing_function(arch_type)
    aug = get_data_augmentation() if use_augmentation else None

    def parse_img(path, label):
        b = tf.io.read_file(path)
        img = tf.image.decode_jpeg(b, channels=3)
        img = tf.image.resize(img, config.IMG_SIZE)
        return img, label

    def prep_train(img, label):
        if aug:
            img = aug(img, training=True)
        img = preprocess_fn(img)
        return img, label

    def prep_val(img, label):
        img = preprocess_fn(img)
        return img, label

    # Train Pipeline: Load -> Resize -> Cache -> Shuffle -> Augment -> Preprocess -> Batch
    train_ds = tf.data.Dataset.from_tensor_slices((train_fps, train_lbls))
    train_ds = train_ds.map(parse_img, num_parallel_calls=tf.data.AUTOTUNE).cache()
    train_ds = train_ds.shuffle(buffer_size=len(train_fps), seed=seed)
    train_ds = train_ds.map(prep_train, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Validation Pipeline: Load -> Resize -> Cache -> Preprocess -> Batch
    val_ds = tf.data.Dataset.from_tensor_slices((val_fps, val_lbls))
    val_ds = val_ds.map(parse_img, num_parallel_calls=tf.data.AUTOTUNE).cache()
    val_ds = val_ds.map(prep_val, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names, len(train_fps), len(val_fps)

def evaluate_model_on_dataset(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    class_names: list,
    model_name: str
):
    """
    Evaluates model predictions and computes macro and per-class metrics.
    """
    y_true, y_pred_probs = [], []
    start_time = time.perf_counter()
    sample_count = 0

    for batch_imgs, batch_lbls in dataset:
        preds = model.predict_on_batch(batch_imgs)
        y_pred_probs.append(np.array(preds))
        y_true.append(np.argmax(batch_lbls.numpy(), axis=1))
        sample_count += len(batch_lbls)

    inference_time = time.perf_counter() - start_time
    avg_latency_ms = round((inference_time / max(1, sample_count)) * 1000, 2)

    y_true = np.concatenate(y_true, axis=0)
    y_pred_probs = np.concatenate(y_pred_probs, axis=0)
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

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    return {
        "model_name": model_name,
        "test_samples": sample_count,
        "accuracy": round(acc, 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_wt), 4),
        "avg_latency_ms": avg_latency_ms,
        "per_class": per_class,
        "confusion_matrix": cm.tolist()
    }

def plot_confusion_matrix_figure(cm_array, class_names, title, save_path):
    cm = np.array(cm_array)
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
    plt.title(f"{title} - Confusion Matrix", fontsize=14, fontweight="bold", pad=12)
    plt.colorbar(fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    short_names = [c.replace("Tomato___", "Tom_").replace("Tomato_", "Tom_").replace("Pepper__bell___", "Pep_").replace("Potato___", "Pot_") for c in class_names]
    plt.xticks(tick_marks, short_names, rotation=45, ha="right", fontsize=9)
    plt.yticks(tick_marks, short_names, fontsize=9)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            plt.text(j, i, format(val, "d"), ha="center", va="center", color=color, fontsize=8)

    plt.ylabel("Ground Truth", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=180)
    plt.close()

def run_single_experiment(
    exp_id: str,
    exp_name: str,
    arch_type: str = "mobilenetv2",
    use_augmentation: bool = True,
    use_class_weight: bool = False,
    fine_tune: bool = False,
    head_epochs: int = 5,
    fine_tune_epochs: int = 3,
    fine_tune_layers: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    fine_tune_lr: float = 1e-4,
    samples_per_class: int = 100
):
    set_seed(42)
    print("\n" + "="*70)
    print(f"RUNNING EXPERIMENT {exp_id}: {exp_name}")
    print(f"Architecture: {arch_type} | Augmentation: {use_augmentation} | Class Weight: {use_class_weight} | Fine-Tuning: {fine_tune}")
    print("="*70)

    exp_dir = os.path.join(config.MODELS_DIR, "experiments", exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    model_save_path = os.path.join(exp_dir, f"{exp_id}_model.keras")

    # 1. Prepare Datasets
    train_ds, val_ds, class_names, n_train, n_val = build_data_pipelines(
        train_dir=config.TRAIN_DIR,
        valid_dir=config.VALID_DIR,
        arch_type=arch_type,
        use_augmentation=use_augmentation,
        samples_per_class_train=samples_per_class,
        samples_per_class_val=30,
        batch_size=batch_size,
        seed=42
    )
    print(f"Loaded {n_train} training samples and {n_val} validation samples.")

    # 2. Class Weights
    class_weights = None
    if use_class_weight:
        class_weights = calculate_class_weights(config.TRAIN_DIR, len(class_names), max_samples_per_class=samples_per_class)
        print(f"Class weights computed for {len(class_names)} classes.")

    # 3. Model Architecture
    tf.keras.backend.clear_session()
    gc.collect()

    if arch_type == "mobilenetv2":
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=config.INPUT_SHAPE,
            include_top=False,
            weights="imagenet"
        )
    elif arch_type == "efficientnetb0":
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=config.INPUT_SHAPE,
            include_top=False,
            weights="imagenet"
        )
    else:
        raise ValueError(f"Unknown architecture: {arch_type}")

    base_model.trainable = False
    model = build_transfer_model(
        base_model=base_model,
        num_classes=len(class_names),
        model_name=f"leafscan_{exp_id}",
        dropout_rate=0.3,
        dense_units=512
    )

    # 4. Compile & Callbacks
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # 5. Phase 1: Train Head
    print(f"\n[Phase 1] Training classification head ({head_epochs} epochs)...")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=head_epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # 6. Phase 2: Fine-Tuning (if enabled)
    history2 = None
    if fine_tune and fine_tune_epochs > 0:
        print(f"\n[Phase 2] Unfreezing top {fine_tune_layers} layers (with BatchNormalization frozen)...")
        base_model.trainable = True
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False
        # Freeze BatchNormalization
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=fine_tune_epochs,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )

    # Combine History
    combined_history = {}
    for k in history1.history:
        combined_history[k] = [float(v) for v in history1.history[k]]
        if history2 and k in history2.history:
            combined_history[k].extend([float(v) for v in history2.history[k]])

    # Save History JSON & Plots
    history_json = os.path.join(config.REPORTS_DIR, "training_histories", f"{exp_id}_history.json")
    curve_png = os.path.join(config.REPORTS_DIR, "training_histories", f"{exp_id}_curves.png")
    os.makedirs(os.path.dirname(history_json), exist_ok=True)
    with open(history_json, "w", encoding="utf-8") as f:
        json.dump(combined_history, f, indent=2)
    plot_and_save_training_curves(combined_history, curve_png, exp_name)

    # Ensure model is saved
    if not os.path.exists(model_save_path):
        model.save(model_save_path)

    # 7. Evaluate on Validation Set
    best_eval_model = tf.keras.models.load_model(model_save_path, compile=False)
    metrics = evaluate_model_on_dataset(best_eval_model, val_ds, class_names, exp_name)
    metrics["exp_id"] = exp_id
    metrics["architecture"] = arch_type
    metrics["model_path"] = model_save_path
    metrics["model_size_mb"] = round(os.path.getsize(model_save_path) / (1024*1024), 2)
    metrics["final_train_accuracy"] = combined_history["accuracy"][-1]
    metrics["final_train_loss"] = combined_history["loss"][-1]
    metrics["final_val_loss"] = combined_history["val_loss"][-1]

    # Save Confusion Matrix Heatmap
    cm_plot_path = os.path.join(config.REPORTS_DIR, "confusion_matrices", f"{exp_id}_cm.png")
    plot_confusion_matrix_figure(metrics["confusion_matrix"], class_names, exp_name, cm_plot_path)
    metrics["confusion_matrix_plot"] = cm_plot_path

    # Save individual experiment metrics
    with open(os.path.join(exp_dir, "val_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[{exp_name}] Val Acc: {metrics['accuracy']*100:.2f}% | Macro F1: {metrics['macro_f1']:.4f} | Macro Recall: {metrics['macro_recall']:.4f}")
    return metrics

def run_all_experiments():
    start_all = time.perf_counter()
    os.makedirs(os.path.join(config.REPORTS_DIR, "training_histories"), exist_ok=True)
    os.makedirs(os.path.join(config.REPORTS_DIR, "confusion_matrices"), exist_ok=True)

    experiments_to_run = [
        {
            "id": "exp1_augmentation",
            "name": "Exp 1: Improved Data Augmentation",
            "arch_type": "mobilenetv2",
            "use_augmentation": True,
            "use_class_weight": False,
            "fine_tune": False,
            "head_epochs": 4,
            "fine_tune_epochs": 0,
            "samples_per_class": 80
        },
        {
            "id": "exp2_finetuning",
            "name": "Exp 2: Augmentation + Fine-Tuning (Frozen BN)",
            "arch_type": "mobilenetv2",
            "use_augmentation": True,
            "use_class_weight": False,
            "fine_tune": True,
            "head_epochs": 3,
            "fine_tune_epochs": 2,
            "samples_per_class": 80
        },
        {
            "id": "exp3_class_weighting",
            "name": "Exp 3: Augmentation + Balanced Class Weighting",
            "arch_type": "mobilenetv2",
            "use_augmentation": True,
            "use_class_weight": True,
            "fine_tune": False,
            "head_epochs": 4,
            "fine_tune_epochs": 0,
            "samples_per_class": 80
        },
        {
            "id": "exp4_combined_best",
            "name": "Exp 4: Combined Best (Aug + Class Weights + Fine-Tuning)",
            "arch_type": "mobilenetv2",
            "use_augmentation": True,
            "use_class_weight": True,
            "fine_tune": True,
            "head_epochs": 4,
            "fine_tune_epochs": 2,
            "samples_per_class": 80
        },
        {
            "id": "exp5_efficientnetb0",
            "name": "Exp 5: EfficientNetB0 Benchmark (Fixed Preprocessing)",
            "arch_type": "efficientnetb0",
            "use_augmentation": True,
            "use_class_weight": True,
            "fine_tune": True,
            "head_epochs": 3,
            "fine_tune_epochs": 2,
            "samples_per_class": 80
        }
    ]

    # Include existing baseline metrics from scratch/baseline_val_evaluation.json
    baseline_path = os.path.join(config.REPORTS_DIR, "baseline_val_evaluation.json")
    baseline_metrics = {
        "exp_id": "exp0_baseline",
        "model_name": "Baseline MobileNetV2 (Existing)",
        "architecture": "mobilenetv2",
        "accuracy": 0.6368,
        "macro_precision": 0.6719,
        "macro_recall": 0.6233,
        "macro_f1": 0.5742,
        "weighted_f1": 0.6289,
        "avg_latency_ms": 115.4,
        "model_size_mb": 16.82
    }
    if os.path.exists(baseline_path):
        try:
            with open(baseline_path, "r", encoding="utf-8") as f:
                bdata = json.load(f)
                baseline_metrics["accuracy"] = round(bdata["accuracy"], 4)
                baseline_metrics["macro_precision"] = round(bdata["macro_precision"], 4)
                baseline_metrics["macro_recall"] = round(bdata["macro_recall"], 4)
                baseline_metrics["macro_f1"] = round(bdata["macro_f1"], 4)
                baseline_metrics["weighted_f1"] = round(bdata["weighted_f1"], 4)
        except Exception:
            pass

    all_results = [baseline_metrics]

    for cfg in experiments_to_run:
        res = run_single_experiment(
            exp_id=cfg["id"],
            exp_name=cfg["name"],
            arch_type=cfg["arch_type"],
            use_augmentation=cfg["use_augmentation"],
            use_class_weight=cfg["use_class_weight"],
            fine_tune=cfg["fine_tune"],
            head_epochs=cfg["head_epochs"],
            fine_tune_epochs=cfg["fine_tune_epochs"],
            samples_per_class=cfg["samples_per_class"]
        )
        all_results.append(res)

    total_time = time.perf_counter() - start_all
    print(f"\nAll experiments completed in {total_time/60:.2f} minutes.")

    # Save comparative table
    comparison_json = os.path.join(config.REPORTS_DIR, "experiment_comparison.json")
    with open(comparison_json, "w", encoding="utf-8") as f:
        # Strip large confusion matrices for clean summary
        clean_summary = []
        for r in all_results:
            c = {k: v for k, v in r.items() if k not in ["confusion_matrix", "per_class"]}
            clean_summary.append(c)
        json.dump(clean_summary, f, indent=2)

    # Generate Markdown Table
    md_lines = [
        "# Plant Disease Detection - Controlled Optimization Experiments\n",
        "Evaluated on identical validation split under controlled settings.\n",
        "| Experiment / Model | Architecture | Val Accuracy | Macro Precision | Macro Recall | Macro F1 | Latency | Model Size |",
        "|:-------------------|:------------:|:------------:|:---------------:|:------------:|:--------:|:-------:|:----------:|"
    ]
    for r in all_results:
        md_lines.append(
            f"| **{r['model_name']}** | {r.get('architecture', 'mobilenetv2')} | {r['accuracy']*100:.2f}% | {r['macro_precision']*100:.2f}% | {r['macro_recall']*100:.2f}% | **{r['macro_f1']*100:.2f}%** | {r.get('avg_latency_ms', 0)} ms | {r.get('model_size_mb', 0)} MB |"
        )

    comparison_md = os.path.join(config.REPORTS_DIR, "experiment_comparison.md")
    with open(comparison_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print("\n" + "\n".join(md_lines))

    # Champion Selection (highest Macro F1 among completed models)
    trained_models = [r for r in all_results if r["exp_id"] != "exp0_baseline"]
    champion = max(trained_models, key=lambda x: x["macro_f1"])

    print("\n" + "="*70)
    print(f"CHAMPION MODEL SELECTED: {champion['model_name']}")
    print(f"Validation Macro F1: {champion['macro_f1']:.4f} vs Baseline: {baseline_metrics['macro_f1']:.4f}")
    print(f"Validation Accuracy: {champion['accuracy']*100:.2f}% vs Baseline: {baseline_metrics['accuracy']*100:.2f}%")
    print("="*70)

    return champion, all_results

if __name__ == "__main__":
    run_all_experiments()
