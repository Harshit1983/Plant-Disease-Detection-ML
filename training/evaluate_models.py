import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_ml_service_dir = os.path.abspath(os.path.join(_current_dir, ".."))
if _ml_service_dir not in sys.path:
    sys.path.insert(0, _ml_service_dir)

import gc
import json
import time
import shutil
import argparse
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
import config
from training.base_trainer import get_preprocessing_function, load_dataset

def plot_confusion_matrix(cm: np.ndarray, class_names: list, output_path: str, title: str):
    """
    Renders and saves high-resolution confusion matrix heatmap.
    """
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
    plt.title(f"{title} - Confusion Matrix", fontsize=14, fontweight="bold", pad=12)
    plt.colorbar(fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    short_names = [c.replace("Tomato___", "Tom_").replace("Tomato_", "Tom_").replace("Pepper__bell___", "Pep_").replace("Potato___", "Pot_") for c in class_names]
    plt.xticks(tick_marks, short_names, rotation=45, ha="right", fontsize=9)
    plt.yticks(tick_marks, short_names, fontsize=9)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            plt.text(j, i, format(val, "d"), ha="center", va="center", color=color, fontsize=8)

    plt.ylabel("Ground Truth Label", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()

def evaluate_single_model(
    model_name: str,
    model_path: str,
    arch_type: str,
    target_conv_layer: str,
    test_dir: str,
    class_names: list,
    batch_size: int = 1,
    max_samples_per_class: int = 0
):
    print(f"\n---> Evaluating {model_name} from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}. Skipping.")
        return None

    # Load trained model
    tf.keras.backend.clear_session()
    gc.collect()
    model = tf.keras.models.load_model(model_path, compile=False)

    # Model file size
    model_size_mb = round(os.path.getsize(model_path) / (1024 * 1024), 2)

    # Load raw test images
    raw_ds, _ = load_dataset(
        test_dir,
        batch_size=batch_size,
        max_samples_per_class=max_samples_per_class,
        shuffle=False
    )

    # Apply strictly architecture-specific preprocessing
    preprocess_fn = get_preprocessing_function(arch_type)
    def prep_test(img, lbl):
        return preprocess_fn(img), lbl
    test_ds = raw_ds.map(prep_test)

    # Ground truth labels and predictions
    y_true = []
    y_pred_probs = []

    # Measure inference latency over batches
    start_time = time.perf_counter()
    sample_count = 0

    for batch_images, batch_labels in test_ds:
        preds = model(batch_images, training=False).numpy()
        y_pred_probs.append(preds)
        y_true.append(np.argmax(batch_labels.numpy(), axis=-1))
        sample_count += len(batch_labels)

    total_time = time.perf_counter() - start_time
    avg_latency_ms = round((total_time / max(1, sample_count)) * 1000, 2)

    y_true = np.concatenate(y_true, axis=0)
    y_pred_probs = np.concatenate(y_pred_probs, axis=0)
    y_pred = np.argmax(y_pred_probs, axis=-1)

    # Metrics calculation
    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    # Per-class metrics dictionary
    per_class_metrics = {}
    for i, c_name in enumerate(class_names):
        per_class_metrics[c_name] = {
            "precision": round(float(p_cls[i]), 4),
            "recall": round(float(r_cls[i]), 4),
            "f1_score": round(float(f1_cls[i]), 4)
        }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_path = os.path.join(config.REPORTS_DIR, "confusion_matrices", f"{arch_type}_cm.png")
    plot_confusion_matrix(cm, class_names, cm_path, model_name)

    metrics_result = {
        "model_name": model_name,
        "architecture": arch_type,
        "target_conv_layer": target_conv_layer,
        "model_path": model_path,
        "model_size_mb": model_size_mb,
        "avg_latency_ms": avg_latency_ms,
        "test_samples": sample_count,
        "accuracy": round(acc, 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_wt), 4),
        "per_class": per_class_metrics,
        "confusion_matrix_plot": cm_path,
        "y_pred_probs": y_pred_probs,
        "y_true": y_true
    }

    print(f"[{model_name}] Test Acc: {acc:.4f} | Macro F1: {f1_macro:.4f} | Latency: {avg_latency_ms}ms | Size: {model_size_mb}MB")
    return metrics_result

def evaluate_ensemble(evaluated_models: list, class_names: list):
    """
    Combines softmax probabilities from available models via soft voting.
    """
    valid_models = [m for m in evaluated_models if m is not None and "y_pred_probs" in m]
    if len(valid_models) < 2:
        return None

    print("\n---> Evaluating Ensemble (Soft Probability Averaging)...")
    y_true = valid_models[0]["y_true"]
    combined_probs = np.mean([m["y_pred_probs"] for m in valid_models], axis=0)
    y_pred = np.argmax(combined_probs, axis=-1)

    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    sum_latency = round(sum(m["avg_latency_ms"] for m in valid_models), 2)
    sum_size = round(sum(m["model_size_mb"] for m in valid_models), 2)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_path = os.path.join(config.REPORTS_DIR, "confusion_matrices", "ensemble_cm.png")
    plot_confusion_matrix(cm, class_names, cm_path, "Ensemble (MobileNetV2 + ResNet50 + EfficientNetB0)")

    ensemble_res = {
        "model_name": "Ensemble (3-Model Soft Average)",
        "architecture": "Ensemble",
        "model_size_mb": sum_size,
        "avg_latency_ms": sum_latency,
        "test_samples": len(y_true),
        "accuracy": round(acc, 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_wt), 4),
        "confusion_matrix_plot": cm_path
    }
    print(f"[Ensemble] Test Acc: {acc:.4f} | Macro F1: {f1_macro:.4f} | Sum Latency: {sum_latency}ms | Total Size: {sum_size}MB")
    return ensemble_res

def run_evaluation_and_deployment(test_dir: str = None, max_samples_per_class: int = 0):
    test_dir = test_dir or config.TEST_DIR
    class_names = list(config.CLASS_NAMES)

    models_to_evaluate = [
        {
            "name": "MobileNetV2",
            "path": os.path.join(config.MODELS_DIR, "mobilenetv2", "mobilenetv2_model.keras"),
            "arch": "mobilenetv2",
            "layer": "out_relu"
        },
        {
            "name": "ResNet50",
            "path": os.path.join(config.MODELS_DIR, "resnet50", "resnet50_model.keras"),
            "arch": "resnet50",
            "layer": "conv5_block3_out"
        },
        {
            "name": "EfficientNetB0",
            "path": os.path.join(config.MODELS_DIR, "efficientnetb0", "efficientnetb0_model.keras"),
            "arch": "efficientnetb0",
            "layer": "top_activation"
        }
    ]

    results = []
    for m in models_to_evaluate:
        res = evaluate_single_model(
            model_name=m["name"],
            model_path=m["path"],
            arch_type=m["arch"],
            target_conv_layer=m["layer"],
            test_dir=test_dir,
            class_names=class_names,
            batch_size=1,
            max_samples_per_class=max_samples_per_class
        )
        if res:
            results.append(res)

    if not results:
        print("No models were evaluated.")
        return

    # Evaluate optional ensemble
    ensemble_res = evaluate_ensemble(results, class_names)

    # Strip numpy arrays for clean JSON serialization
    serialized_results = []
    for r in results:
        clean = {k: v for k, v in r.items() if k not in ["y_pred_probs", "y_true"]}
        serialized_results.append(clean)
    if ensemble_res:
        serialized_results.append(ensemble_res)

    # Save comparative report JSON
    comparison_json_path = os.path.join(config.REPORTS_DIR, "model_comparison.json")
    with open(comparison_json_path, "w", encoding="utf-8") as f:
        json.dump(serialized_results, f, indent=2)

    # Generate Markdown Table
    md_lines = [
        "# Plant Disease Detection - Multi-Model Architecture Comparison\n",
        f"Evaluated on untouched held-out test split ({test_dir}).\n",
        "| Architecture | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Latency (ms/img) | Model Size (MB) |",
        "|:-------------|:-------------:|:---------------:|:------------:|:--------------:|:----------------:|:---------------:|"
    ]
    for r in serialized_results:
        md_lines.append(
            f"| **{r['model_name']}** | {r['accuracy']*100:.2f}% | {r['macro_precision']*100:.2f}% | {r['macro_recall']*100:.2f}% | {r['macro_f1']*100:.2f}% | {r['avg_latency_ms']} ms | {r['model_size_mb']} MB |"
        )

    comparison_md_path = os.path.join(config.REPORTS_DIR, "model_comparison.md")
    with open(comparison_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\nComparative evaluation report saved to {comparison_json_path} and {comparison_md_path}")

    # Model Selection: Pick model with highest Macro F1
    individual_models = [r for r in results if r["architecture"] != "Ensemble"]
    best_model_meta = max(individual_models, key=lambda x: x["macro_f1"])

    print(f"\n=================================================================")
    print(f"  SELECTED CHAMPION MODEL: {best_model_meta['model_name']}")
    print(f"  Macro F1: {best_model_meta['macro_f1']:.4f} | Accuracy: {best_model_meta['accuracy']:.4f}")
    print(f"=================================================================")

    # Copy best model to models/best_model/ and models/plant_disease_model.keras
    best_dir = os.path.join(config.MODELS_DIR, "best_model")
    os.makedirs(best_dir, exist_ok=True)
    best_target_path = os.path.join(best_dir, "best_plant_disease_model.keras")
    shutil.copyfile(best_model_meta["model_path"], best_target_path)

    # Also update config.MODEL_PATH so standard production endpoints use it
    shutil.copyfile(best_model_meta["model_path"], config.MODEL_PATH)

    # Create model manifest
    manifest = {
        "selected_model_name": best_model_meta["model_name"],
        "architecture": best_model_meta["architecture"],
        "model_file": best_target_path,
        "production_active_path": config.MODEL_PATH,
        "target_conv_layer": best_model_meta["target_conv_layer"],
        "preprocessing": best_model_meta["architecture"],
        "metrics": {
            "test_accuracy": best_model_meta["accuracy"],
            "macro_f1": best_model_meta["macro_f1"],
            "macro_precision": best_model_meta["macro_precision"],
            "macro_recall": best_model_meta["macro_recall"],
            "avg_latency_ms": best_model_meta["avg_latency_ms"],
            "model_size_mb": best_model_meta["model_size_mb"]
        },
        "all_classes": class_names,
        "selection_rationale": f"Superior macro F1-score ({best_model_meta['macro_f1']:.4f}) and balanced per-class pathology discrimination."
    }

    manifest_path = os.path.join(config.MODELS_DIR, "model_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated production model manifest: {manifest_path}")
    return manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=str, default=None)
    parser.add_argument("--max-samples-per-class", type=int, default=0)
    args = parser.parse_args()

    run_evaluation_and_deployment(test_dir=args.test_dir, max_samples_per_class=args.max_samples_per_class)
