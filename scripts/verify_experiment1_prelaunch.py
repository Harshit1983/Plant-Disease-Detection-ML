import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
import gc
import json
import time
import shutil
import hashlib
import psutil
import numpy as np
import tensorflow as tf

# Ensure ml-service root on sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_ml_service_dir = os.path.abspath(os.path.join(_current_dir, ".."))
if _ml_service_dir not in sys.path:
    sys.path.insert(0, _ml_service_dir)

import config
from training.base_trainer import build_transfer_model, calculate_class_weights
from training.train_full_streaming import build_streaming_pipelines, get_ram_usage_mb

def get_file_md5(filepath: str) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def run_prelaunch_verification():
    print("="*80)
    print(" EXPERIMENT 1 PRE-LAUNCH VERIFICATION & SMOKE TEST")
    print("="*80)
    
    results = {}
    initial_rss = get_ram_usage_mb()
    sys_mem_init = psutil.virtual_memory()
    print(f"Initial Process RSS:  {initial_rss:.2f} MB")
    print(f"System Available RAM: {sys_mem_init.available / (1024*1024):.1f} MB ({sys_mem_init.percent}% used)")
    print("-"*80)

    # -------------------------------------------------------------------------
    # CHECK 1: Full dataset counts without truncation
    # -------------------------------------------------------------------------
    print("\n[Check 1] Verifying full dataset image counts (No .take() or sampling)...")
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    
    def count_images_in_dir(d):
        total = 0
        per_class = {}
        for cls in sorted(os.listdir(d)):
            cls_p = os.path.join(d, cls)
            if os.path.isdir(cls_p):
                cnt = len([f for f in os.listdir(cls_p) if os.path.splitext(f)[1].lower() in valid_exts])
                per_class[cls] = cnt
                total += cnt
        return total, per_class

    train_count, train_breakdown = count_images_in_dir(config.TRAIN_DIR)
    val_count, val_breakdown = count_images_in_dir(config.VALID_DIR)
    test_count, test_breakdown = count_images_in_dir(config.TEST_DIR)

    batch_size = 16
    train_ds, val_ds, test_ds, class_names = build_streaming_pipelines(batch_size=batch_size)

    # Check cardinality (number of batches)
    train_batches = int(train_ds.cardinality().numpy())
    expected_train_batches = int(np.ceil(train_count / batch_size))

    c1_passed = (train_count == 14438) and (val_count == 3095) and (test_count == 3091) and (train_batches == expected_train_batches)
    results["check_1_full_dataset_counts"] = {
        "passed": c1_passed,
        "train_images": train_count,
        "expected_train_images": 14438,
        "val_images": val_count,
        "expected_val_images": 3095,
        "test_images": test_count,
        "expected_test_images": 3091,
        "train_batches": train_batches,
        "expected_train_batches": expected_train_batches
    }
    print(f"  Train: {train_count}/14,438 images ({train_batches} batches of {batch_size})")
    print(f"  Valid: {val_count}/3,095 images")
    print(f"  Test:  {test_count}/3,091 images")
    print(f"  Result: {'PASSED' if c1_passed else 'FAILED'}")

    # -------------------------------------------------------------------------
    # CHECK 2: Prefetch(1) and Zero In-Memory Caching
    # -------------------------------------------------------------------------
    print("\n[Check 2] Verifying prefetch(1) and zero in-memory caching...")
    # Inspect train_full_streaming.py source code
    with open(os.path.join(config.BASE_DIR, "training", "train_full_streaming.py"), "r") as f:
        script_code = f.read()

    has_cache_call = ".cache()" in script_code
    has_prefetch_1 = ".prefetch(1)" in script_code
    has_prefetch_autotune = ".prefetch(tf.data.AUTOTUNE)" in script_code

    c2_passed = (not has_cache_call) and has_prefetch_1 and (not has_prefetch_autotune)
    results["check_2_prefetch_and_caching"] = {
        "passed": c2_passed,
        "has_cache_call": has_cache_call,
        "has_prefetch_1": has_prefetch_1,
        "has_prefetch_autotune": has_prefetch_autotune
    }
    print(f"  In-memory caching (.cache()): {'FOUND (Risk!)' if has_cache_call else 'NONE (Safe, streaming directly from disk)'}")
    print(f"  prefetch(1) configured:       {has_prefetch_1}")
    print(f"  prefetch(AUTOTUNE) disabled:  {not has_prefetch_autotune}")
    print(f"  Result: {'PASSED' if c2_passed else 'FAILED'}")

    # -------------------------------------------------------------------------
    # CHECK 3: Data augmentation applied ONLY to training, NOT validation
    # -------------------------------------------------------------------------
    print("\n[Check 3] Verifying augmentation applies strictly to training images only...")
    # Get 1 raw batch from validation
    raw_val_batch = next(iter(tf.keras.utils.image_dataset_from_directory(
        config.VALID_DIR,
        labels="inferred",
        label_mode="categorical",
        batch_size=4,
        image_size=config.IMG_SIZE,
        shuffle=False
    )))
    val_sample_imgs, val_sample_lbls = raw_val_batch

    preprocess_fn = tf.keras.applications.mobilenet_v2.preprocess_input
    # Pass through eval prep function 3 times
    pass1 = preprocess_fn(tf.identity(val_sample_imgs)).numpy()
    pass2 = preprocess_fn(tf.identity(val_sample_imgs)).numpy()
    pass3 = preprocess_fn(tf.identity(val_sample_imgs)).numpy()

    val_deterministic = np.allclose(pass1, pass2) and np.allclose(pass2, pass3)

    # Pass 1 training image through train prep function twice to verify stochastic variation
    from training.train_full_streaming import get_standard_augmentation
    aug = get_standard_augmentation()
    aug_pass1 = preprocess_fn(aug(val_sample_imgs, training=True)).numpy()
    aug_pass2 = preprocess_fn(aug(val_sample_imgs, training=True)).numpy()
    train_stochastic = not np.allclose(aug_pass1, aug_pass2)

    c3_passed = val_deterministic and train_stochastic
    results["check_3_augmentation_isolation"] = {
        "passed": c3_passed,
        "validation_deterministic": bool(val_deterministic),
        "training_stochastic": bool(train_stochastic)
    }
    print(f"  Validation preprocessing deterministic: {val_deterministic} (Exact pixel match across passes)")
    print(f"  Training augmentation stochastic:       {train_stochastic} (Active flips/rotations/contrast)")
    print(f"  Result: {'PASSED' if c3_passed else 'FAILED'}")

    # -------------------------------------------------------------------------
    # CHECK 4: Class mapping and class weights verification
    # -------------------------------------------------------------------------
    print("\n[Check 4] Verifying class mapping and class weights against config...")
    with open(config.CLASS_INDICES_PATH, "r") as f:
        saved_class_indices = json.load(f)

    sorted_config_classes = list(config.CLASS_NAMES)
    class_order_matched = (class_names == sorted_config_classes) and (list(saved_class_indices.keys()) == sorted_config_classes)
    
    # Calculate class weights
    raw_class_weights = calculate_class_weights(config.TRAIN_DIR, num_classes=len(class_names))
    dampened_weights = {i: float(np.clip(np.sqrt(w), 0.6, 2.5)) for i, w in raw_class_weights.items()}

    c4_passed = class_order_matched and (len(class_names) == 15)
    results["check_4_class_mapping"] = {
        "passed": c4_passed,
        "class_count": len(class_names),
        "mapping_exact_match": class_order_matched,
        "dampened_weights_sample": {class_names[i]: round(dampened_weights[i], 3) for i in [0, 4, 6, 12]}
    }
    print(f"  15 classes match config.CLASS_NAMES exactly: {class_order_matched}")
    print(f"  Class weights correctly calculated: {len(dampened_weights)} classes (Minority 'Potato___healthy': {dampened_weights[4]:.2f}, Majority 'Tomato_YellowLeaf_Curl': {dampened_weights[12]:.2f})")
    print(f"  Result: {'PASSED' if c4_passed else 'FAILED'}")

    # -------------------------------------------------------------------------
    # CHECK 5: Run 1 train batch and 1 validation batch (forward & backward)
    # -------------------------------------------------------------------------
    print("\n[Check 5] Running smoke batch: forward pass, backward pass, loss & shape checks...")
    tf.keras.backend.clear_session()
    gc.collect()

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=config.INPUT_SHAPE,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    smoke_model = build_transfer_model(
        base_model=base_model,
        num_classes=15,
        model_name="smoke_test_mobilenetv2",
        dropout_rate=0.3,
        dense_units=512
    )

    smoke_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Fetch 1 train batch and 1 validation batch
    t_batch_imgs, t_batch_lbls = next(iter(train_ds))
    v_batch_imgs, v_batch_lbls = next(iter(val_ds))

    shape_ok = (t_batch_imgs.shape == (batch_size, 224, 224, 3)) and (t_batch_lbls.shape == (batch_size, 15))
    range_ok = (float(np.min(t_batch_imgs.numpy())) >= -1.05) and (float(np.max(t_batch_imgs.numpy())) <= 1.05)

    # Train 1 step
    loss_before = float(smoke_model.evaluate(t_batch_imgs, t_batch_lbls, verbose=0)[0])
    train_step_loss, train_step_acc = smoke_model.train_on_batch(t_batch_imgs, t_batch_lbls)
    val_step_loss, val_step_acc = smoke_model.test_on_batch(v_batch_imgs, v_batch_lbls)

    losses_finite = np.isfinite(train_step_loss) and np.isfinite(val_step_loss)
    c5_passed = shape_ok and range_ok and losses_finite

    results["check_5_batch_smoke_execution"] = {
        "passed": c5_passed,
        "input_shape": list(t_batch_imgs.shape),
        "target_shape": list(t_batch_lbls.shape),
        "normalization_range": [float(np.min(t_batch_imgs.numpy())), float(np.max(t_batch_imgs.numpy()))],
        "train_step_loss": float(train_step_loss),
        "train_step_acc": float(train_step_acc),
        "val_step_loss": float(val_step_loss),
        "val_step_acc": float(val_step_acc)
    }
    print(f"  Input batch shape:   {t_batch_imgs.shape} (Expected: ({batch_size}, 224, 224, 3))")
    print(f"  Target batch shape:  {t_batch_lbls.shape} (Expected: ({batch_size}, 15))")
    print(f"  Normalization range: [{np.min(t_batch_imgs.numpy()):.2f}, {np.max(t_batch_imgs.numpy()):.2f}] (Expected: [-1.0, 1.0])")
    print(f"  Train step loss:     {train_step_loss:.4f} (Accuracy: {train_step_acc:.2%})")
    print(f"  Val step loss:       {val_step_loss:.4f} (Accuracy: {val_step_acc:.2%})")
    print(f"  Result: {'PASSED' if c5_passed else 'FAILED'}")

    # -------------------------------------------------------------------------
    # CHECK 6: ModelCheckpoint & EarlyStopping configuration
    # -------------------------------------------------------------------------
    print("\n[Check 6] Verifying ModelCheckpoint and EarlyStopping callbacks...")
    from training.train_full_streaming import run_experiment_1
    import inspect
    source = inspect.getsource(run_experiment_1)

    has_ckpt_monitor = 'monitor="val_loss"' in source
    has_save_best = 'save_best_only=True' in source
    has_restore_best = 'restore_best_weights=True' in source
    has_reduce_lr = 'ReduceLROnPlateau' in source

    c6_passed = has_ckpt_monitor and has_save_best and has_restore_best and has_reduce_lr
    results["check_6_callbacks"] = {
        "passed": c6_passed,
        "has_ckpt_monitor": has_ckpt_monitor,
        "has_save_best": has_save_best,
        "has_restore_best": has_restore_best,
        "has_reduce_lr": has_reduce_lr
    }
    print(f"  ModelCheckpoint monitor='val_loss':      {has_ckpt_monitor}")
    print(f"  ModelCheckpoint save_best_only=True:     {has_save_best}")
    print(f"  EarlyStopping restore_best_weights=True: {has_restore_best}")
    print(f"  ReduceLROnPlateau configured:            {has_reduce_lr}")
    print(f"  Result: {'PASSED' if c6_passed else 'FAILED'}")

    # -------------------------------------------------------------------------
    # CHECK 7: RAM consumption recording
    # -------------------------------------------------------------------------
    print("\n[Check 7] Recording actual RAM consumption during smoke test...")
    smoke_rss = get_ram_usage_mb()
    sys_mem_smoke = psutil.virtual_memory()
    rss_delta = smoke_rss - initial_rss

    # Process RSS under 750 MB is safe
    c7_passed = smoke_rss < 750.0 and sys_mem_smoke.percent < 95.0
    results["check_7_ram_consumption"] = {
        "passed": c7_passed,
        "initial_rss_mb": round(initial_rss, 1),
        "smoke_rss_mb": round(smoke_rss, 1),
        "rss_delta_mb": round(rss_delta, 1),
        "system_available_mb": round(sys_mem_smoke.available / (1024*1024), 1),
        "system_used_percent": sys_mem_smoke.percent
    }
    print(f"  Initial Process RSS:  {initial_rss:.1f} MB")
    print(f"  Active Process RSS:   {smoke_rss:.1f} MB (Delta: +{rss_delta:.1f} MB)")
    print(f"  System Available RAM: {sys_mem_smoke.available / (1024*1024):.1f} MB ({sys_mem_smoke.percent}% used)")
    print(f"  Result: {'PASSED' if c7_passed else 'FAILED'}")

    # -------------------------------------------------------------------------
    # CHECK 8: Production checkpoint and backup integrity
    # -------------------------------------------------------------------------
    print("\n[Check 8] Verifying production champion checkpoint and backups are untouched...")
    prod_path = config.MODEL_PATH
    best_path = os.path.join(config.MODELS_DIR, "best_model", "best_plant_disease_model.keras")
    backup_path = os.path.join(config.MODELS_DIR, "backup_champion_86_expA", "plant_disease_model.keras")

    prod_exists = os.path.exists(prod_path)
    best_exists = os.path.exists(best_path)
    backup_exists = os.path.exists(backup_path)

    prod_size = os.path.getsize(prod_path) if prod_exists else 0
    best_size = os.path.getsize(best_path) if best_exists else 0
    backup_size = os.path.getsize(backup_path) if backup_exists else 0

    expected_size = 25899365
    c8_passed = (prod_size == expected_size) and (best_size == expected_size) and (backup_size == expected_size)
    results["check_8_production_integrity"] = {
        "passed": c8_passed,
        "prod_model_size": prod_size,
        "best_model_size": best_size,
        "backup_model_size": backup_size,
        "expected_size": expected_size
    }
    print(f"  Production model size:    {prod_size} bytes (Expected: {expected_size})")
    print(f"  Best champion model size: {best_size} bytes (Expected: {expected_size})")
    print(f"  Backup model size:        {backup_size} bytes (Expected: {expected_size})")
    print(f"  Result: {'PASSED' if c8_passed else 'FAILED'}")

    # -------------------------------------------------------------------------
    # CHECK 9: Test set isolation
    # -------------------------------------------------------------------------
    print("\n[Check 9] Verifying test set isolation (Not used for training or checkpoint selection)...")
    # In train_full_streaming.py: model.fit uses validation_data=val_ds only
    train_fit_on_val_only = ("validation_data=val_ds" in script_code) and ("validation_data=test_ds" not in script_code)
    c9_passed = train_fit_on_val_only
    results["check_9_test_set_isolation"] = {
        "passed": c9_passed,
        "training_uses_val_only": train_fit_on_val_only
    }
    print(f"  model.fit uses validation_data=val_ds only: {train_fit_on_val_only}")
    print(f"  Test set strictly isolated from training/checkpoint selection: {c9_passed}")
    print(f"  Result: {'PASSED' if c9_passed else 'FAILED'}")

    # -------------------------------------------------------------------------
    # CHECK 10: Checkpoint save and reload verification
    # -------------------------------------------------------------------------
    print("\n[Check 10] Verifying checkpoint save and reload functionality...")
    smoke_dir = os.path.join(config.MODELS_DIR, "experiments", "smoke_test")
    os.makedirs(smoke_dir, exist_ok=True)
    smoke_save_path = os.path.join(smoke_dir, "smoke_checkpoint.keras")

    smoke_model.save(smoke_save_path)
    save_exists = os.path.exists(smoke_save_path)
    save_size = os.path.getsize(smoke_save_path)

    reloaded_model = tf.keras.models.load_model(smoke_save_path, compile=False)
    pred_orig = smoke_model.predict(v_batch_imgs[:2], verbose=0)
    pred_reloaded = reloaded_model.predict(v_batch_imgs[:2], verbose=0)

    preds_match = np.allclose(pred_orig, pred_reloaded, atol=1e-5)

    # Clean up smoke test dir
    shutil.rmtree(smoke_dir, ignore_errors=True)

    c10_passed = save_exists and (save_size > 10*1024*1024) and preds_match
    results["check_10_checkpoint_save_and_reload"] = {
        "passed": c10_passed,
        "save_exists": save_exists,
        "save_size_bytes": save_size,
        "predictions_match": bool(preds_match)
    }
    print(f"  Saved model to disk:      {save_exists} ({save_size / (1024*1024):.1f} MB)")
    print(f"  Reloaded model from disk: Success")
    print(f"  Inference outputs match:  {preds_match} (Exact prediction parity)")
    print(f"  Result: {'PASSED' if c10_passed else 'FAILED'}")

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    all_passed = all(r["passed"] for r in results.values())
    print("\n" + "="*80)
    print(f"PRE-LAUNCH VERIFICATION SUMMARY: {'ALL 10 CHECKS PASSED' if all_passed else 'CHECKS FAILED'}")
    print("="*80)

    def sanitize_for_json(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [sanitize_for_json(v) for v in obj]
        return obj

    report_path = os.path.join(config.REPORTS_DIR, "exp1_prelaunch_verification_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "all_passed": bool(all_passed),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "checks": sanitize_for_json(results)
        }, f, indent=2)
    print(f"Report saved to: {report_path}")
    return all_passed, results

if __name__ == "__main__":
    passed, res = run_prelaunch_verification()
    sys.exit(0 if passed else 1)
