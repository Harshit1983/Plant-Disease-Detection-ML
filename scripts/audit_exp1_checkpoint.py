import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
import gc
import json
import time
import psutil
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# Ensure ml-service root on sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_ml_service_dir = os.path.abspath(os.path.join(_current_dir, ".."))
if _ml_service_dir not in sys.path:
    sys.path.insert(0, _ml_service_dir)

import config

def audit_exp1():
    print("="*80)
    print(" AUDITING EXPERIMENT 1 CHECKPOINT & ARTIFACTS")
    print("="*80)

    best_ckpt_path = os.path.join(config.MODELS_DIR, "experiments", "exp1_full_streaming", "exp1_mobilenetv2_best.keras")
    latest_ckpt_path = os.path.join(config.MODELS_DIR, "experiments", "exp1_full_streaming", "exp1_mobilenetv2_latest.keras")
    prod_ckpt_path = config.MODEL_PATH

    print(f"Best Checkpoint:   {best_ckpt_path} (Exists: {os.path.exists(best_ckpt_path)})")
    print(f"Latest Checkpoint: {latest_ckpt_path} (Exists: {os.path.exists(latest_ckpt_path)})")
    print(f"Prod Champion:     {prod_ckpt_path} (Exists: {os.path.exists(prod_ckpt_path)})")

    # 1. Inspect checkpoint files on disk
    best_size = os.path.getsize(best_ckpt_path) if os.path.exists(best_ckpt_path) else 0
    latest_size = os.path.getsize(latest_ckpt_path) if os.path.exists(latest_ckpt_path) else 0
    prod_size = os.path.getsize(prod_ckpt_path) if os.path.exists(prod_ckpt_path) else 0

    print(f"\nFile Sizes:")
    print(f"  exp1_mobilenetv2_best.keras:   {best_size} bytes ({best_size / (1024*1024):.2f} MB)")
    print(f"  exp1_mobilenetv2_latest.keras: {latest_size} bytes ({latest_size / (1024*1024):.2f} MB)")
    print(f"  plant_disease_model.keras:     {prod_size} bytes ({prod_size / (1024*1024):.2f} MB)")

    # 2. Load exp1_mobilenetv2_best.keras and inspect layer trainability
    print("\nInspecting architecture and trainable layers in exp1_mobilenetv2_best.keras...")
    best_model = tf.keras.models.load_model(best_ckpt_path, compile=False)
    
    # Check if base_model layers are frozen or unfrozen
    base_model = None
    for layer in best_model.layers:
        if isinstance(layer, tf.keras.Model) or "mobilenetv2" in layer.name.lower():
            base_model = layer
            break

    total_layers = len(best_model.layers)
    trainable_weights = len(best_model.trainable_weights)
    base_trainable_layers = [l.name for l in base_model.layers if l.trainable] if base_model else []
    
    print(f"  Model total layers: {total_layers}")
    print(f"  Trainable weight tensors: {trainable_weights}")
    if base_model:
        print(f"  Base model trainable layers count: {len(base_trainable_layers)}")
        print(f"  Base model trainable layer names: {base_trainable_layers[-5:] if base_trainable_layers else 'NONE (All frozen)'}")
    
    # 3. Load Validation Dataset strictly (3,095 samples)
    print("\nLoading Validation Dataset (dataset/valid, 3,095 images)...")
    raw_val = tf.keras.utils.image_dataset_from_directory(
        config.VALID_DIR,
        labels="inferred",
        label_mode="categorical",
        batch_size=16,
        image_size=config.IMG_SIZE,
        shuffle=False
    )
    class_names = raw_val.class_names
    num_classes = len(class_names)
    
    preprocess_fn = tf.keras.applications.mobilenet_v2.preprocess_input
    val_ds = raw_val.map(lambda x, y: (preprocess_fn(x), y)).prefetch(1)

    # 4. Compile model to compute exact categorical cross-entropy loss and accuracy
    best_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("Running model.evaluate on validation dataset...")
    val_eval = best_model.evaluate(val_ds, verbose=1)
    recalculated_loss = float(val_eval[0])
    recalculated_acc = float(val_eval[1])

    # 5. Extract predictions and recalculate all macro and per-class metrics
    print("Running forward inference across all 3,095 validation samples...")
    y_true, y_probs = [], []
    t0 = time.perf_counter()
    sample_count = 0

    for imgs, lbls in val_ds:
        preds = best_model(imgs, training=False).numpy()
        y_probs.append(preds)
        y_true.append(np.argmax(lbls.numpy(), axis=-1))
        sample_count += len(lbls)

    eval_latency_ms = (time.perf_counter() - t0) / max(1, sample_count) * 1000.0
    y_true = np.concatenate(y_true, axis=0)
    y_probs = np.concatenate(y_probs, axis=0)
    y_pred = np.argmax(y_probs, axis=-1)

    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0)

    per_class = {}
    for i, c_name in enumerate(class_names):
        per_class[c_name] = {
            "precision": round(float(p_cls[i]), 4),
            "recall": round(float(r_cls[i]), 4),
            "f1_score": round(float(f1_cls[i]), 4),
            "support": int(np.sum(y_true == i))
        }

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    print("\n" + "="*80)
    print("AUDIT VERIFICATION SUMMARY")
    print("="*80)
    print(f"Recalculated Validation Loss:     {recalculated_loss:.4f}")
    print(f"Recalculated Validation Accuracy: {acc*100:.2f}% (evaluate: {recalculated_acc*100:.2f}%)")
    print(f"Macro Precision:                  {p_macro*100:.2f}%")
    print(f"Macro Recall:                     {r_macro*100:.2f}%")
    print(f"Macro F1-Score:                   {f1_macro*100:.2f}%")
    print(f"Weighted F1-Score:                {f1_wt*100:.2f}%")
    print(f"Avg CPU Latency:                  {eval_latency_ms:.2f} ms/sample")
    print(f"Total Validation Samples:         {len(y_true)}")
    print("="*80)

    audit_data = {
        "checkpoint_file": best_ckpt_path,
        "checkpoint_size_bytes": best_size,
        "is_stage2_finetune_weights": len(base_trainable_layers) > 0,
        "base_trainable_layers_count": len(base_trainable_layers),
        "validation_samples": int(len(y_true)),
        "recalculated_loss": round(recalculated_loss, 4),
        "recalculated_accuracy": round(acc, 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_wt), 4),
        "avg_latency_ms": round(float(eval_latency_ms), 2),
        "per_class": per_class,
        "confusion_matrix": cm.tolist()
    }

    audit_out_path = os.path.join(config.REPORTS_DIR, "experiments", "exp1_full_streaming", "exp1_audit_data.json")
    with open(audit_out_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
    print(f"Saved audit data to: {audit_out_path}")

    return audit_data

if __name__ == "__main__":
    audit_exp1()

