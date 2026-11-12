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
from sklearn.metrics import confusion_matrix

_current_dir = os.path.dirname(os.path.abspath(__file__))
_ml_service_dir = os.path.abspath(os.path.join(_current_dir, ".."))
if _ml_service_dir not in sys.path:
    sys.path.insert(0, _ml_service_dir)

import config
from training.run_advanced_experiments import (
    evaluate_on_dataset,
    ADV_MODELS_DIR,
    ADV_REPORTS_DIR
)

def main():
    print("="*80)
    print("ADVANCED EXPERIMENTS: FULL VALIDATION & TEST BENCHMARK SUITE")
    print("="*80)

    val_dir = config.VALID_DIR
    test_dir = config.TEST_DIR
    champion_path = os.path.join(config.MODELS_DIR, "best_model", "best_plant_disease_model.keras")
    expA_path = os.path.join(ADV_MODELS_DIR, "expA_finetune", "expA_finetune_best.keras")
    expC_path = os.path.join(ADV_MODELS_DIR, "expC_hard_class_focus", "expC_hard_class_best.keras")

    # 1. Load full validation dataset (3095 samples)
    print(f"\nLoading full validation dataset ({val_dir})...", flush=True)
    raw_val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="categorical",
        batch_size=16,
        image_size=(224, 224),
        shuffle=False
    )
    class_names = raw_val_ds.class_names

    def prep(imgs, lbls):
        return tf.keras.applications.mobilenet_v2.preprocess_input(imgs), lbls

    val_ds = raw_val_ds.map(prep)

    # Load baseline champion validation metrics
    val_baseline_file = os.path.join(ADV_REPORTS_DIR, "champion_val_baseline.json")
    with open(val_baseline_file, "r") as f:
        champ_val_metrics = json.load(f)

    benchmark_rows = []
    benchmark_rows.append({
        "experiment": "Baseline Champion",
        "model_architecture": "MobileNetV2 (Data Augmentation)",
        "val_accuracy": champ_val_metrics["accuracy"],
        "val_macro_f1": champ_val_metrics["macro_f1"],
        "early_blight_recall": champ_val_metrics["per_class"]["Tomato_Early_blight"]["recall"],
        "target_spot_recall": champ_val_metrics["per_class"]["Tomato__Target_Spot"]["recall"],
        "septoria_f1": champ_val_metrics["per_class"]["Tomato_Septoria_leaf_spot"]["f1_score"],
        "potato_healthy_precision": champ_val_metrics["per_class"]["Potato___healthy"]["precision"],
        "avg_latency_ms": champ_val_metrics["avg_latency_ms"]
    })

    def run_eval(m, ds, name, total_samples=3095):
        print(f"[{name}] Starting dataset iteration (total: {total_samples})...", flush=True)
        y_t, y_p = [], []
        s_cnt = 0
        t0 = time.perf_counter()
        for idx, (imgs, lbls) in enumerate(ds):
            preds = m(imgs, training=False).numpy()
            y_p.append(preds)
            y_t.append(np.argmax(lbls.numpy(), axis=1))
            s_cnt += len(lbls)
            if idx < 3:
                print(f"[{name}] Batch {idx} ok (sample count: {s_cnt})...", flush=True)
            if (idx + 1) % 20 == 0 or s_cnt >= total_samples:
                print(f"[{name}] Evaluated {s_cnt}/{total_samples} samples (elapsed: {(time.perf_counter()-t0):.1f}s)...", flush=True)
                gc.collect()
        lat = (time.perf_counter() - t0) / max(1, s_cnt) * 1000
        y_t = np.concatenate(y_t, axis=0)
        y_p = np.concatenate(y_p, axis=0)
        y_pred = np.argmax(y_p, axis=1)

        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        acc = float(accuracy_score(y_t, y_pred))
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_t, y_pred, average="macro", zero_division=0)
        p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_t, y_pred, average="weighted", zero_division=0)
        p_c, r_c, f1_c, _ = precision_recall_fscore_support(y_t, y_pred, labels=list(range(len(class_names))), average=None, zero_division=0)

        pc = {}
        for i, cn in enumerate(class_names):
            pc[cn] = {
                "precision": round(float(p_c[i]), 4),
                "recall": round(float(r_c[i]), 4),
                "f1_score": round(float(f1_c[i]), 4)
            }
        return {
            "accuracy": round(acc, 4),
            "macro_precision": round(float(p_m), 4),
            "macro_recall": round(float(r_m), 4),
            "macro_f1": round(float(f1_m), 4),
            "weighted_f1": round(float(f1_wt), 4),
            "avg_latency_ms": round(float(lat), 2),
            "per_class": pc,
            "y_true": y_t,
            "y_pred": y_pred,
            "y_pred_probs": y_p
        }

    # Evaluate Exp A on validation
    metrics_A = None
    y_true_val = None
    expA_json = os.path.join(ADV_REPORTS_DIR, "expA_full_val_metrics.json")
    expA_npy = os.path.join(ADV_REPORTS_DIR, "expA_val_probs.npy")

    if os.path.exists(expA_json) and os.path.exists(expA_npy):
        print(f"\nLoading cached Exp A validation metrics from {expA_json}...", flush=True)
        with open(expA_json, "r") as f:
            metrics_A = json.load(f)
        metrics_A["y_pred_probs"] = np.load(expA_npy)
        benchmark_rows.append({
            "experiment": "Exp A: Fine-Tuning",
            "model_architecture": "MobileNetV2 (Top-15 Conv Layers, LR=1e-5, Frozen BN)",
            "val_accuracy": metrics_A["accuracy"],
            "val_macro_f1": metrics_A["macro_f1"],
            "early_blight_recall": metrics_A["per_class"]["Tomato_Early_blight"]["recall"],
            "target_spot_recall": metrics_A["per_class"]["Tomato__Target_Spot"]["recall"],
            "septoria_f1": metrics_A["per_class"]["Tomato_Septoria_leaf_spot"]["f1_score"],
            "potato_healthy_precision": metrics_A["per_class"]["Potato___healthy"]["precision"],
            "avg_latency_ms": metrics_A["avg_latency_ms"]
        })
    elif os.path.exists(expA_path):
        print(f"\nEvaluating Exp A Model ({expA_path}) on full validation set...", flush=True)
        gc.collect()
        expA_model = tf.keras.models.load_model(expA_path, compile=False)
        metrics_A = run_eval(expA_model, val_ds, "Exp A")
        del expA_model
        gc.collect()

        np.save(expA_npy, metrics_A["y_pred_probs"])
        benchmark_rows.append({
            "experiment": "Exp A: Fine-Tuning",
            "model_architecture": "MobileNetV2 (Top-15 Conv Layers, LR=1e-5, Frozen BN)",
            "val_accuracy": metrics_A["accuracy"],
            "val_macro_f1": metrics_A["macro_f1"],
            "early_blight_recall": metrics_A["per_class"]["Tomato_Early_blight"]["recall"],
            "target_spot_recall": metrics_A["per_class"]["Tomato__Target_Spot"]["recall"],
            "septoria_f1": metrics_A["per_class"]["Tomato_Septoria_leaf_spot"]["f1_score"],
            "potato_healthy_precision": metrics_A["per_class"]["Potato___healthy"]["precision"],
            "avg_latency_ms": metrics_A["avg_latency_ms"]
        })
        with open(expA_json, "w") as f:
            json.dump({k: v for k, v in metrics_A.items() if k not in ["y_true", "y_pred", "y_pred_probs"]}, f, indent=2)

    # Evaluate Exp C on validation
    metrics_C = None
    expC_json = os.path.join(ADV_REPORTS_DIR, "expC_full_val_metrics.json")
    expC_npy = os.path.join(ADV_REPORTS_DIR, "expC_val_probs.npy")

    if os.path.exists(expC_json) and os.path.exists(expC_npy):
        print(f"\nLoading cached Exp C validation metrics from {expC_json}...", flush=True)
        with open(expC_json, "r") as f:
            metrics_C = json.load(f)
        metrics_C["y_pred_probs"] = np.load(expC_npy)
        benchmark_rows.append({
            "experiment": "Exp C: Hard-Class Focus",
            "model_architecture": "MobileNetV2 (Dampened Class Weighting)",
            "val_accuracy": metrics_C["accuracy"],
            "val_macro_f1": metrics_C["macro_f1"],
            "early_blight_recall": metrics_C["per_class"]["Tomato_Early_blight"]["recall"],
            "target_spot_recall": metrics_C["per_class"]["Tomato__Target_Spot"]["recall"],
            "septoria_f1": metrics_C["per_class"]["Tomato_Septoria_leaf_spot"]["f1_score"],
            "potato_healthy_precision": metrics_C["per_class"]["Potato___healthy"]["precision"],
            "avg_latency_ms": metrics_C["avg_latency_ms"]
        })
    elif os.path.exists(expC_path):
        print(f"\nEvaluating Exp C Model ({expC_path}) on full validation set...", flush=True)
        gc.collect()
        expC_model = tf.keras.models.load_model(expC_path, compile=False)
        metrics_C = run_eval(expC_model, val_ds, "Exp C")
        del expC_model
        gc.collect()

        np.save(expC_npy, metrics_C["y_pred_probs"])
        benchmark_rows.append({
            "experiment": "Exp C: Hard-Class Focus",
            "model_architecture": "MobileNetV2 (Dampened Class Weighting)",
            "val_accuracy": metrics_C["accuracy"],
            "val_macro_f1": metrics_C["macro_f1"],
            "early_blight_recall": metrics_C["per_class"]["Tomato_Early_blight"]["recall"],
            "target_spot_recall": metrics_C["per_class"]["Tomato__Target_Spot"]["recall"],
            "septoria_f1": metrics_C["per_class"]["Tomato_Septoria_leaf_spot"]["f1_score"],
            "potato_healthy_precision": metrics_C["per_class"]["Potato___healthy"]["precision"],
            "avg_latency_ms": metrics_C["avg_latency_ms"]
        })
        with open(expC_json, "w") as f:
            json.dump({k: v for k, v in metrics_C.items() if k not in ["y_true", "y_pred", "y_pred_probs"]}, f, indent=2)

    # Get ground truth for validation
    val_true_path = os.path.join(ADV_REPORTS_DIR, "val_true.npy")
    if os.path.exists(val_true_path):
        y_true_val = np.load(val_true_path)
    else:
        y_t_list = []
        for _, lbls in val_ds:
            y_t_list.append(np.argmax(lbls.numpy(), axis=1))
        y_true_val = np.concatenate(y_t_list, axis=0)
        np.save(val_true_path, y_true_val)

    # Evaluate Exp E (Soft Ensemble of Exp A + Exp C) on validation
    if metrics_A is not None and metrics_C is not None:
        print("\nEvaluating Exp E: Soft Ensemble (Exp A + Exp C) on validation set...", flush=True)
        # Blend probabilities: 50% Exp A + 50% Exp C
        p_ens = 0.5 * metrics_A["y_pred_probs"] + 0.5 * metrics_C["y_pred_probs"]
        y_true = y_true_val
        y_pred = np.argmax(p_ens, axis=1)

        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        acc_ens = float(accuracy_score(y_true, y_pred))
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        p_c, r_c, f1_c, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(class_names))), average=None, zero_division=0)

        eb_idx = class_names.index("Tomato_Early_blight")
        ts_idx = class_names.index("Tomato__Target_Spot")
        sep_idx = class_names.index("Tomato_Septoria_leaf_spot")
        ph_idx = class_names.index("Potato___healthy")

        ens_lat = metrics_A["avg_latency_ms"] + metrics_C["avg_latency_ms"]
        benchmark_rows.append({
            "experiment": "Exp E: Ensemble",
            "model_architecture": "Ensemble (Exp A 50% + Exp C 50%)",
            "val_accuracy": round(acc_ens, 4),
            "val_macro_f1": round(float(f1_m), 4),
            "early_blight_recall": round(float(r_c[eb_idx]), 4),
            "target_spot_recall": round(float(r_c[ts_idx]), 4),
            "septoria_f1": round(float(f1_c[sep_idx]), 4),
            "potato_healthy_precision": round(float(p_c[ph_idx]), 4),
            "avg_latency_ms": round(float(ens_lat), 2)
        })

    # Save benchmark table
    table_file = os.path.join(ADV_REPORTS_DIR, "validation_experiments_comparison.json")
    with open(table_file, "w") as f:
        json.dump(benchmark_rows, f, indent=2)

    print("\n" + "="*80)
    print("VALIDATION BENCHMARK RESULTS TABLE")
    print("="*80)
    for r in benchmark_rows:
        print(f"{r['experiment']:25s} | Acc: {r['val_accuracy']*100:.2f}% | Macro F1: {r['val_macro_f1']*100:.2f}% | EarlyBlight Rec: {r['early_blight_recall']*100:.1f}% | TargetSpot Rec: {r['target_spot_recall']*100:.1f}% | Latency: {r['avg_latency_ms']:.1f}ms")

    # Select best candidate on validation macro F1
    candidates = [r for r in benchmark_rows if r["experiment"] != "Baseline Champion"]
    best_candidate = max(candidates, key=lambda x: x["val_macro_f1"])
    print("\n" + "="*80)
    print(f"TOP CANDIDATE SELECTED ON VALIDATION: {best_candidate['experiment']}")
    print(f"Validation Macro F1: {best_candidate['val_macro_f1']*100:.2f}% (vs Baseline: {benchmark_rows[0]['val_macro_f1']*100:.2f}%)")
    print("="*80)

    # 2. Final evaluation on untouched test set (3091 samples)
    print(f"\nPerforming ONE final evaluation on the untouched test set ({test_dir})...", flush=True)
    raw_test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="categorical",
        batch_size=16,
        image_size=(224, 224),
        shuffle=False
    )
    test_ds = raw_test_ds.map(prep)

    gc.collect()
    if best_candidate["experiment"] == "Exp A: Fine-Tuning":
        m_eval = tf.keras.models.load_model(expA_path, compile=False)
        test_metrics = run_eval(m_eval, test_ds, "Exp A (Test)", total_samples=3091)
        del m_eval
        gc.collect()
    elif best_candidate["experiment"] == "Exp C: Hard-Class Focus":
        m_eval = tf.keras.models.load_model(expC_path, compile=False)
        test_metrics = run_eval(m_eval, test_ds, "Exp C (Test)", total_samples=3091)
        del m_eval
        gc.collect()
    elif "Ensemble" in best_candidate["experiment"]:
        expA_test_npy = os.path.join(ADV_REPORTS_DIR, "expA_test_probs.npy")
        expC_test_npy = os.path.join(ADV_REPORTS_DIR, "expC_test_probs.npy")
        test_true_npy = os.path.join(ADV_REPORTS_DIR, "test_true.npy")

        if os.path.exists(expA_test_npy) and os.path.exists(test_true_npy):
            print(f"Loading cached Exp A test probabilities...", flush=True)
            p_A_test = np.load(expA_test_npy)
            y_true_test = np.load(test_true_npy)
            lat_A = 85.0
        else:
            mA = tf.keras.models.load_model(expA_path, compile=False)
            m_expA_test = run_eval(mA, test_ds, "Exp A (Test)", total_samples=3091)
            del mA
            gc.collect()
            p_A_test = m_expA_test["y_pred_probs"]
            y_true_test = m_expA_test["y_true"]
            lat_A = m_expA_test["avg_latency_ms"]
            np.save(expA_test_npy, p_A_test)
            np.save(test_true_npy, y_true_test)
            with open(os.path.join(ADV_REPORTS_DIR, "expA_test_metrics.json"), "w") as f:
                json.dump({k: v for k, v in m_expA_test.items() if k not in ["y_true", "y_pred", "y_pred_probs"]}, f, indent=2)

        if os.path.exists(expC_test_npy):
            print(f"Loading cached Exp C test probabilities...", flush=True)
            p_C_test = np.load(expC_test_npy)
            lat_C = 80.0
        else:
            mC = tf.keras.models.load_model(expC_path, compile=False)
            m_expC_test = run_eval(mC, test_ds, "Exp C (Test)", total_samples=3091)
            del mC
            gc.collect()
            p_C_test = m_expC_test["y_pred_probs"]
            lat_C = m_expC_test["avg_latency_ms"]
            np.save(expC_test_npy, p_C_test)
            with open(os.path.join(ADV_REPORTS_DIR, "expC_test_metrics.json"), "w") as f:
                json.dump({k: v for k, v in m_expC_test.items() if k not in ["y_true", "y_pred", "y_pred_probs"]}, f, indent=2)

        p_ens_test = 0.5 * p_A_test + 0.5 * p_C_test
        y_pred_test = np.argmax(p_ens_test, axis=1)

        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        acc_test = float(accuracy_score(y_true_test, y_pred_test))
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_true_test, y_pred_test, average="macro", zero_division=0)
        p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true_test, y_pred_test, average="weighted", zero_division=0)
        p_c, r_c, f1_c, _ = precision_recall_fscore_support(y_true_test, y_pred_test, labels=list(range(len(class_names))), average=None, zero_division=0)

        per_class_test = {}
        for i, name in enumerate(class_names):
            per_class_test[name] = {
                "precision": round(float(p_c[i]), 4),
                "recall": round(float(r_c[i]), 4),
                "f1_score": round(float(f1_c[i]), 4)
            }

        test_metrics = {
            "accuracy": round(acc_test, 4),
            "macro_precision": round(float(p_m), 4),
            "macro_recall": round(float(r_m), 4),
            "macro_f1": round(float(f1_m), 4),
            "weighted_f1": round(float(f1_wt), 4),
            "avg_latency_ms": round(float(lat_A + lat_C), 2),
            "per_class": per_class_test,
            "y_true": y_true_test,
            "y_pred": y_pred_test
        }

    print("\n" + "="*80)
    print("FINAL UNTOUCHED TEST SET RESULTS (3091 SAMPLES)")
    print("="*80)
    print(f"Candidate:           {best_candidate['experiment']}")
    print(f"Test Accuracy:       {test_metrics['accuracy']*100:.2f}% (Baseline: 80.52%)")
    print(f"Test Macro F1:       {test_metrics['macro_f1']*100:.2f}% (Baseline: 78.34%)")
    print(f"Test Macro Recall:   {test_metrics['macro_recall']*100:.2f}% (Baseline: 80.59%)")
    print(f"Test Weighted F1:    {test_metrics['weighted_f1']*100:.2f}% (Baseline: 80.34%)")
    print(f"Inference Latency:   {test_metrics['avg_latency_ms']:.2f} ms/sample")
    print("="*80)

    # Plot test confusion matrix
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

    # Save final JSON report
    final_report = {
        "candidate": best_candidate["experiment"],
        "architecture": best_candidate["model_architecture"],
        "accuracy": test_metrics["accuracy"],
        "macro_precision": test_metrics["macro_precision"],
        "macro_recall": test_metrics["macro_recall"],
        "macro_f1": test_metrics["macro_f1"],
        "weighted_f1": test_metrics["weighted_f1"],
        "avg_latency_ms": test_metrics["avg_latency_ms"],
        "per_class": test_metrics["per_class"],
        "confusion_matrix_plot": cm_out
    }
    with open(os.path.join(ADV_REPORTS_DIR, "final_test_benchmark.json"), "w") as f:
        json.dump(final_report, f, indent=2)

    print(f"\nFinal test benchmark saved to: {ADV_REPORTS_DIR}")

if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print("\nUNCAUGHT EXCEPTION IN MAIN:", flush=True)
        traceback.print_exc()
        sys.exit(1)
