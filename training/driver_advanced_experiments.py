import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
import gc
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

# Add ml-service to path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_ml_service_dir = os.path.abspath(os.path.join(_current_dir, ".."))
if _ml_service_dir not in sys.path:
    sys.path.insert(0, _ml_service_dir)

import config
from training.run_advanced_experiments import (
    build_datasets,
    evaluate_on_dataset,
    run_experiment_A_finetuning,
    run_experiment_B_native_resolution,
    run_experiment_C_hard_class_focus,
    run_experiment_E_ensemble,
    ADV_MODELS_DIR,
    ADV_REPORTS_DIR
)

def main():
    print("="*80)
    print("STARTING ADVANCED ACCURACY OPTIMIZATION PIPELINE")
    print("="*80)

    train_dir = config.TRAIN_DIR
    valid_dir = config.VALID_DIR
    test_dir = config.TEST_DIR
    champion_path = os.path.join(config.MODELS_DIR, "best_model", "best_plant_disease_model.keras")

    print(f"Train directory:      {train_dir}")
    print(f"Validation directory: {valid_dir}")
    print(f"Test directory:       {test_dir}")
    print(f"Current Champion:     {champion_path}")

    # Build 224x224 standard datasets for Exp A, C, E
    print("\nBuilding cached 224x224 training and validation datasets...")
    train_ds, val_ds, class_names, n_train, n_val = build_datasets(
        train_dir, valid_dir,
        arch_type="mobilenetv2",
        img_size=(224, 224),
        batch_size=16,
        train_batches=60,
        val_batches=25
    )
    print(f"Cached {n_train} training samples and {n_val} validation samples across 15 classes.")

    results_table = []

    # 1. Load or evaluate baseline champion on validation set
    baseline_val_json = os.path.join(ADV_REPORTS_DIR, "champion_val_baseline.json")
    if os.path.exists(baseline_val_json):
        with open(baseline_val_json, "r") as f:
            base_metrics = json.load(f)
    else:
        print("Evaluating champion model on validation set...")
        champ_model = tf.keras.models.load_model(champion_path, compile=False)
        base_metrics = evaluate_on_dataset(champ_model, val_ds, class_names)
        with open(baseline_val_json, "w") as f:
            json.dump({k: v for k, v in base_metrics.items() if k not in ["y_true", "y_pred", "y_pred_probs"]}, f, indent=2)

    results_table.append({
        "experiment": "Champion Baseline",
        "model_architecture": "MobileNetV2 (Data Augmentation)",
        "input_resolution": "224x224",
        "val_accuracy": base_metrics["accuracy"],
        "val_macro_f1": base_metrics["macro_f1"],
        "early_blight_recall": base_metrics["per_class"].get("Tomato_Early_blight", {}).get("recall", 0.0),
        "target_spot_recall": base_metrics["per_class"].get("Tomato__Target_Spot", {}).get("recall", 0.0),
        "septoria_f1": base_metrics["per_class"].get("Tomato_Septoria_leaf_spot", {}).get("f1_score", 0.0),
        "potato_healthy_precision": base_metrics["per_class"].get("Potato___healthy", {}).get("precision", 0.0),
        "avg_latency_ms": base_metrics["avg_latency_ms"]
    })

    # 2. Run Experiment A: Fine-Tuning
    try:
        model_A, metrics_A = run_experiment_A_finetuning(champion_path, train_ds, val_ds, class_names)
        results_table.append({
            "experiment": "Exp A: Fine-Tuning",
            "model_architecture": "MobileNetV2 (Top 35 Layers, LR=1e-5, Frozen BN)",
            "input_resolution": "224x224",
            "val_accuracy": metrics_A["accuracy"],
            "val_macro_f1": metrics_A["macro_f1"],
            "early_blight_recall": metrics_A["per_class"].get("Tomato_Early_blight", {}).get("recall", 0.0),
            "target_spot_recall": metrics_A["per_class"].get("Tomato__Target_Spot", {}).get("recall", 0.0),
            "septoria_f1": metrics_A["per_class"].get("Tomato_Septoria_leaf_spot", {}).get("f1_score", 0.0),
            "potato_healthy_precision": metrics_A["per_class"].get("Potato___healthy", {}).get("precision", 0.0),
            "avg_latency_ms": metrics_A["avg_latency_ms"]
        })
    except Exception as e:
        print(f"Error in Exp A: {e}")
        model_A, metrics_A = None, None

    # 3. Run Experiment C: Hard-Class Focus (Dampened Class Weighting)
    try:
        model_C, metrics_C = run_experiment_C_hard_class_focus(champion_path, train_ds, val_ds, class_names, train_dir)
        results_table.append({
            "experiment": "Exp C: Hard-Class Focus",
            "model_architecture": "MobileNetV2 (Dampened Class Weights + Smoothing)",
            "input_resolution": "224x224",
            "val_accuracy": metrics_C["accuracy"],
            "val_macro_f1": metrics_C["macro_f1"],
            "early_blight_recall": metrics_C["per_class"].get("Tomato_Early_blight", {}).get("recall", 0.0),
            "target_spot_recall": metrics_C["per_class"].get("Tomato__Target_Spot", {}).get("recall", 0.0),
            "septoria_f1": metrics_C["per_class"].get("Tomato_Septoria_leaf_spot", {}).get("f1_score", 0.0),
            "potato_healthy_precision": metrics_C["per_class"].get("Potato___healthy", {}).get("precision", 0.0),
            "avg_latency_ms": metrics_C["avg_latency_ms"]
        })
    except Exception as e:
        print(f"Error in Exp C: {e}")
        model_C, metrics_C = None, None

    # 4. Run Experiment B: Native 256x256 Resolution
    try:
        model_B, metrics_B, val_ds_256 = run_experiment_B_native_resolution(train_dir, valid_dir, class_names)
        results_table.append({
            "experiment": "Exp B: Native 256x256",
            "model_architecture": "MobileNetV2 (Native 256x256 Resolution)",
            "input_resolution": "256x256",
            "val_accuracy": metrics_B["accuracy"],
            "val_macro_f1": metrics_B["macro_f1"],
            "early_blight_recall": metrics_B["per_class"].get("Tomato_Early_blight", {}).get("recall", 0.0),
            "target_spot_recall": metrics_B["per_class"].get("Tomato__Target_Spot", {}).get("recall", 0.0),
            "septoria_f1": metrics_B["per_class"].get("Tomato_Septoria_leaf_spot", {}).get("f1_score", 0.0),
            "potato_healthy_precision": metrics_B["per_class"].get("Potato___healthy", {}).get("precision", 0.0),
            "avg_latency_ms": metrics_B["avg_latency_ms"]
        })
    except Exception as e:
        print(f"Error in Exp B: {e}")
        model_B, metrics_B = None, None

    # 5. Run Experiment E: Ensemble (if multiple valid models exist)
    ensemble_models = [m for m in [model_A, model_C] if m is not None]
    if len(ensemble_models) >= 2:
        try:
            metrics_E = run_experiment_E_ensemble(ensemble_models, [0.5, 0.5], val_ds, class_names)
            results_table.append({
                "experiment": "Exp E: Ensemble",
                "model_architecture": "Soft-Voting Ensemble (Exp A + Exp C)",
                "input_resolution": "224x224",
                "val_accuracy": metrics_E["accuracy"],
                "val_macro_f1": metrics_E["macro_f1"],
                "early_blight_recall": metrics_E["per_class"].get("Tomato_Early_blight", {}).get("recall", 0.0),
                "target_spot_recall": metrics_E["per_class"].get("Tomato__Target_Spot", {}).get("recall", 0.0),
                "septoria_f1": metrics_E["per_class"].get("Tomato_Septoria_leaf_spot", {}).get("f1_score", 0.0),
                "potato_healthy_precision": metrics_E["per_class"].get("Potato___healthy", {}).get("precision", 0.0),
                "avg_latency_ms": metrics_E["avg_latency_ms"]
            })
        except Exception as e:
            print(f"Error in Exp E: {e}")

    # Save summary table
    summary_path = os.path.join(ADV_REPORTS_DIR, "validation_experiments_comparison.json")
    with open(summary_path, "w") as f:
        json.dump(results_table, f, indent=2)

    print("\n" + "="*80)
    print("VALIDATION BENCHMARK COMPARISON TABLE")
    print("="*80)
    for r in results_table:
        print(f"{r['experiment']:25s} | Acc: {r['val_accuracy']*100:.2f}% | Macro F1: {r['val_macro_f1']*100:.2f}% | EarlyBlight Rec: {r['early_blight_recall']*100:.1f}% | TargetSpot Rec: {r['target_spot_recall']*100:.1f}% | Latency: {r['avg_latency_ms']:.1f}ms")

    # Select best candidate based on validation macro F1
    candidates = [r for r in results_table if r["experiment"] != "Champion Baseline"]
    if not candidates:
        print("No candidates available for test evaluation.")
        return

    best_candidate = max(candidates, key=lambda x: x["val_macro_f1"])
    print("\n" + "="*80)
    print(f"TOP CANDIDATE SELECTED ON VALIDATION DATA: {best_candidate['experiment']}")
    print(f"Validation Macro F1: {best_candidate['val_macro_f1']*100:.2f}% (vs Baseline: {results_table[0]['val_macro_f1']*100:.2f}%)")
    print("="*80)

    # Perform ONE final evaluation of best candidate on untouched test set
    print(f"\nEvaluating Best Candidate ({best_candidate['experiment']}) on full untouched test set ({test_dir})...")
    
    # Determine test input size based on best candidate
    test_img_size = (256, 256) if "256x256" in best_candidate["input_resolution"] else (224, 224)
    raw_test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="categorical",
        batch_size=32,
        image_size=test_img_size,
        shuffle=False
    )
    
    def prep_test(images, labels):
        return tf.keras.applications.mobilenet_v2.preprocess_input(images), labels

    test_ds = raw_test_ds.map(prep_test)

    # Load candidate model
    candidate_model = None
    if best_candidate["experiment"] == "Exp A: Fine-Tuning" and model_A is not None:
        candidate_model = model_A
    elif best_candidate["experiment"] == "Exp C: Hard-Class Focus" and model_C is not None:
        candidate_model = model_C
    elif best_candidate["experiment"] == "Exp B: Native 256x256" and model_B is not None:
        candidate_model = model_B

    if candidate_model is not None:
        test_metrics = evaluate_on_dataset(candidate_model, test_ds, class_names)
        
        print("\n" + "="*80)
        print("FINAL UNTOUCHED TEST SET EVALUATION RESULTS")
        print("="*80)
        print(f"Candidate:           {best_candidate['experiment']}")
        print(f"Test Accuracy:       {test_metrics['accuracy']*100:.2f}% (Baseline: 80.52%)")
        print(f"Test Macro F1:       {test_metrics['macro_f1']*100:.2f}% (Baseline: 78.34%)")
        print(f"Test Macro Recall:   {test_metrics['macro_recall']*100:.2f}% (Baseline: 80.59%)")
        print(f"Test Weighted F1:    {test_metrics['weighted_f1']*100:.2f}% (Baseline: 80.34%)")
        print(f"Avg Latency:         {test_metrics['avg_latency_ms']:.2f} ms/sample")
        print("="*80)

        # Generate test confusion matrix
        from sklearn.metrics import confusion_matrix
        cm_test = confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"], labels=list(range(len(class_names))))
        plt.figure(figsize=(14, 12))
        plt.imshow(cm_test, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"{best_candidate['experiment']} - Test Confusion Matrix (Acc: {test_metrics['accuracy']*100:.1f}%, F1: {test_metrics['macro_f1']*100:.1f}%)", fontsize=13, fontweight="bold", pad=12)
        plt.colorbar(fraction=0.046, pad=0.04)

        tick_marks = np.arange(len(class_names))
        short_names = [c.replace("Tomato___", "Tom_").replace("Tomato_", "Tom_").replace("Pepper__bell___", "Pep_").replace("Potato___", "Pot_") for c in class_names]
        plt.xticks(tick_marks, short_names, rotation=45, ha="right", fontsize=9)
        plt.yticks(tick_marks, short_names, fontsize=9)

        thresh = cm_test.max() / 2.0
        for i in range(cm_test.shape[0]):
            for j in range(cm_test.shape[1]):
                val = cm_test[i, j]
                color = "white" if val > thresh else "black"
                plt.text(j, i, format(val, "d"), ha="center", va="center", color=color, fontsize=8)

        plt.ylabel("Ground Truth", fontsize=11, fontweight="bold")
        plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
        plt.tight_layout()
        cm_out = os.path.join(ADV_REPORTS_DIR, "best_candidate_test_cm.png")
        plt.savefig(cm_out, dpi=180)
        plt.close()

        # Save test evaluation JSON
        test_eval_save = {
            "selected_candidate": best_candidate["experiment"],
            "accuracy": test_metrics["accuracy"],
            "macro_precision": test_metrics["macro_precision"],
            "macro_recall": test_metrics["macro_recall"],
            "macro_f1": test_metrics["macro_f1"],
            "weighted_f1": test_metrics["weighted_f1"],
            "avg_latency_ms": test_metrics["avg_latency_ms"],
            "per_class": test_metrics["per_class"],
            "confusion_matrix_plot": cm_out
        }
        with open(os.path.join(ADV_REPORTS_DIR, "best_candidate_test_evaluation.json"), "w") as f:
            json.dump(test_eval_save, f, indent=2)

        print(f"Saved best candidate test evaluation to: {ADV_REPORTS_DIR}")

if __name__ == "__main__":
    main()

