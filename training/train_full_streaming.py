import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
import gc
import json
import time
import uuid
import math
import hashlib
import psutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

# Ensure ml-service root on sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_ml_service_dir = os.path.abspath(os.path.join(_current_dir, ".."))
if _ml_service_dir not in sys.path:
    sys.path.insert(0, _ml_service_dir)

import config
from training.base_trainer import build_transfer_model, plot_and_save_training_curves

def get_ram_usage_mb() -> float:
    """Returns current process RAM usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def calculate_file_sha256(file_path: str, chunk_size: int = 1024 * 1024) -> str:
    """Calculates SHA256 checksum of a file in streaming chunks to keep RAM usage minimal."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()

def verify_checkpoint_metadata_sha256(ckpt_path: str, meta_path: str, ckpt_name: str) -> tuple:
    """
    Verifies that a checkpoint file and its metadata JSON exist, are well-formed,
    have matching SHA256 checksums, and contain valid required metadata fields:
      - stage: non-empty string
      - stage_epoch: positive integer
      - global_epoch: positive integer
      - val_loss: finite, non-negative number
    Returns: (is_valid: bool, metadata: dict | None, rejection_reason: str | None)
    """
    if not os.path.exists(ckpt_path):
        return False, None, f"{ckpt_name} file does not exist at '{ckpt_path}'."

    if not os.path.exists(meta_path):
        return False, None, f"Metadata file missing for {ckpt_name} at '{meta_path}'."

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        return False, None, f"Metadata file for {ckpt_name} is malformed or unreadable: {e}"

    if not isinstance(metadata, dict):
        return False, None, f"Metadata for {ckpt_name} is not a valid JSON dictionary."

    recorded_sha256 = metadata.get("sha256")
    if not recorded_sha256 or not isinstance(recorded_sha256, str):
        return False, None, f"Metadata for {ckpt_name} lacks a valid 'sha256' field."

    try:
        actual_sha256 = calculate_file_sha256(ckpt_path)
    except Exception as e:
        return False, None, f"Failed calculating SHA256 for {ckpt_name} ('{ckpt_path}'): {e}"

    if actual_sha256.lower() != recorded_sha256.lower():
        return False, None, (
            f"SHA256 mismatch for {ckpt_name}! "
            f"Recorded in metadata: {recorded_sha256}, Actual file hash: {actual_sha256}."
        )

    # Validate required metadata fields
    stage = metadata.get("stage")
    if not isinstance(stage, str) or not stage.strip():
        return False, None, f"Metadata for {ckpt_name} has invalid 'stage': {stage!r} (must be non-empty string)."

    stage_epoch = metadata.get("stage_epoch")
    if type(stage_epoch) is not int or stage_epoch <= 0:
        return False, None, f"Metadata for {ckpt_name} has invalid 'stage_epoch': {stage_epoch!r} (must be positive integer)."

    global_epoch = metadata.get("global_epoch")
    if type(global_epoch) is not int or global_epoch <= 0:
        return False, None, f"Metadata for {ckpt_name} has invalid 'global_epoch': {global_epoch!r} (must be positive integer)."

    val_loss = metadata.get("val_loss")
    if val_loss is None or isinstance(val_loss, bool) or not isinstance(val_loss, (int, float)) or not math.isfinite(val_loss) or val_loss < 0.0:
        return False, None, f"Metadata for {ckpt_name} has invalid 'val_loss': {val_loss!r} (must be finite non-negative number)."

    val_accuracy = metadata.get("val_accuracy")
    if val_accuracy is not None:
        if isinstance(val_accuracy, bool) or not isinstance(val_accuracy, (int, float)) or not math.isfinite(val_accuracy) or val_accuracy < 0.0 or val_accuracy > 1.0:
            return False, None, f"Metadata for {ckpt_name} has invalid 'val_accuracy': {val_accuracy!r} (must be a finite number between 0.0 and 1.0)."

    return True, metadata, None

def atomic_save_model(model: tf.keras.Model, target_path: str):
    """
    Saves a Keras model using a temporary file followed by an atomic replacement (os.replace).
    Preserves the file extension (e.g. .keras) during saving to satisfy Keras 3 path validation.
    If saving fails midway, any existing checkpoint at target_path remains 100% uncorrupted.
    Cleans up the temporary file if saving or replacing fails.
    """
    target_dir = os.path.dirname(os.path.abspath(target_path))
    os.makedirs(target_dir, exist_ok=True)

    base, ext = os.path.splitext(target_path)
    ext = ext if ext else ".keras"
    temp_path = f"{base}.tmp_{uuid.uuid4().hex[:8]}{ext}"
    try:
        model.save(temp_path)
        os.replace(temp_path, target_path)
    except Exception:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise

def atomic_save_json(data: dict, target_path: str):
    """
    Saves a JSON file using a temporary file followed by an atomic replacement (os.replace).
    If writing fails midway, any existing file at target_path remains 100% uncorrupted.
    Cleans up the temporary file if writing or replacing fails.
    """
    target_dir = os.path.dirname(os.path.abspath(target_path))
    os.makedirs(target_dir, exist_ok=True)

    base, ext = os.path.splitext(target_path)
    ext = ext if ext else ".json"
    temp_path = f"{base}.tmp_{uuid.uuid4().hex[:8]}{ext}"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(temp_path, target_path)
    except Exception:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise

class SafetyMemoryMonitorCallback(tf.keras.callbacks.Callback):
    """
    Monitors process RSS and system available RAM continuously at batch and epoch boundaries.
    Enforces user safeguards:
      - Warning / gc trigger if RSS > 650 MB or system available < 350 MB.
      - Safe graceful termination if sustained RSS > 750 MB or sustained system available < 300 MB.
      - Includes a 7.0-second confirmation window before stopping to filter out transient OS paging spikes.
    Tracks peak RSS, minimum available RAM, latest completed epoch, and detailed interruption telemetry.
    """
    def __init__(self, history_save_path: str = None, latest_ckpt_path: str = None, latest_meta_json_path: str = None):
        super().__init__()
        self.history_save_path = history_save_path
        self.latest_ckpt_path = latest_ckpt_path
        self.latest_meta_json_path = latest_meta_json_path
        self.peak_rss_mb = get_ram_usage_mb()
        self.min_avail_mb = psutil.virtual_memory().available / (1024 * 1024)
        self.epoch_metrics = []

        # Stage and epoch tracking
        self.current_stage = "warmup"
        self.epoch_offset = 0
        self.current_stage_epoch_idx = 0
        self.current_batch = 0

        # Completion and interruption tracking
        self.interrupted = False
        self.termination_reason = None
        self.interruption_stage = None
        self.interruption_global_epoch = None
        self.interruption_stage_epoch = None
        self.interruption_batch = None
        self.latest_completed_stage = None
        self.latest_completed_stage_epoch = 0
        self.latest_completed_global_epoch = 0

    def set_stage(self, stage_name: str, epoch_offset: int = 0):
        """Updates active stage and epoch offset for logging and telemetry."""
        self.current_stage = stage_name
        self.epoch_offset = epoch_offset

    def _check_memory(self, context=""):
        rss = get_ram_usage_mb()
        avail = psutil.virtual_memory().available / (1024 * 1024)

        if rss > self.peak_rss_mb:
            self.peak_rss_mb = rss
        if avail < self.min_avail_mb:
            self.min_avail_mb = avail

        CRITICAL_RSS = 750.0
        CRITICAL_AVAIL = 300.0
        CONFIRM_WINDOW_SECS = 7.0

        # Critical threshold check
        if rss > CRITICAL_RSS or avail < CRITICAL_AVAIL:
            print(f"\n[CRITICAL RAM ALERT {context}] Triggered threshold breach:")
            if rss > CRITICAL_RSS:
                print(f"  - Process RSS: {rss:.1f} MB exceeds {CRITICAL_RSS:.1f} MB limit")
            if avail < CRITICAL_AVAIL:
                print(f"  - System Available RAM: {avail:.1f} MB is below {CRITICAL_AVAIL:.1f} MB limit")

            print(f"  [ACTION] Running gc.collect() and waiting {CONFIRM_WINDOW_SECS:.1f}s confirmation window...")
            gc.collect()
            time.sleep(CONFIRM_WINDOW_SECS)
            gc.collect()

            rss_recheck = get_ram_usage_mb()
            avail_recheck = psutil.virtual_memory().available / (1024 * 1024)

            print(f"  [RECHECK] After {CONFIRM_WINDOW_SECS:.1f}s delay: Process RSS: {rss_recheck:.1f} MB | System Avail: {avail_recheck:.1f} MB")

            if rss_recheck > CRITICAL_RSS or avail_recheck < CRITICAL_AVAIL:
                reason = (
                    f"Sustained RAM safeguard breach at {context}: "
                    f"Process RSS={rss_recheck:.1f} MB (limit {CRITICAL_RSS:.1f} MB), "
                    f"System Available={avail_recheck:.1f} MB (limit {CRITICAL_AVAIL:.1f} MB)"
                )
                print(f"  [SAFE TERMINATION] {reason}")
                print("  Safely stopping training to protect host system stability.")

                self.interrupted = True
                self.termination_reason = reason
                self.interruption_stage = self.current_stage
                self.interruption_stage_epoch = self.current_stage_epoch_idx + 1
                self.interruption_global_epoch = self.epoch_offset + self.current_stage_epoch_idx + 1
                self.interruption_batch = self.current_batch
                self.model.stop_training = True
            else:
                print(f"  [RECOVERY CONFIRMED] Memory recovered to safe operating bounds. Resuming training.")
        elif rss > 650.0 or avail < 350.0:
            print(f"\n[RAM Caution {context}] Process RSS: {rss:.1f} MB, Available: {avail:.1f} MB. Triggering gc.collect().")
            gc.collect()

    def on_train_batch_begin(self, batch, logs=None):
        self.current_batch = batch + 1

    def on_train_batch_end(self, batch, logs=None):
        # Sample memory every 100 batches to keep overhead near zero
        if (batch + 1) % 100 == 0:
            self._check_memory(context=f"{self.current_stage.capitalize()} Batch {batch+1}")

    def on_epoch_begin(self, epoch, logs=None):
        self.current_stage_epoch_idx = epoch
        self.current_batch = 0

    def on_epoch_end(self, epoch, logs=None):
        self._check_memory(context=f"{self.current_stage.capitalize()} Epoch {epoch+1} End")
        ram_mb = get_ram_usage_mb()
        sys_mem = psutil.virtual_memory()
        print(f"  [RAM Monitor] Process RSS: {ram_mb:.1f} MB (Peak: {self.peak_rss_mb:.1f} MB) | System Avail: {sys_mem.available/(1024*1024):.1f} MB ({sys_mem.percent}% used, Min: {self.min_avail_mb:.1f} MB)")

        stage_epoch = epoch + 1
        global_epoch = self.epoch_offset + stage_epoch

        # If interrupted during this epoch, do not mark as completed
        if self.interrupted:
            print(f"  [Notice] {self.current_stage.capitalize()} Epoch {stage_epoch} was interrupted and is NOT marked as cleanly completed.")
            return

        self.latest_completed_stage = self.current_stage
        self.latest_completed_stage_epoch = stage_epoch
        self.latest_completed_global_epoch = global_epoch

        # Record metrics for every cleanly completed epoch
        if logs:
            epoch_data = {
                "stage": self.current_stage,
                "stage_epoch": stage_epoch,
                "global_epoch": global_epoch
            }
            for k, v in logs.items():
                epoch_data[k] = float(v)
            epoch_data["process_rss_mb"] = float(round(ram_mb, 1))
            epoch_data["system_avail_mb"] = float(round(sys_mem.available / (1024 * 1024), 1))
            self.epoch_metrics.append(epoch_data)

            # Save running history incrementally so interrupted run never loses progress
            if self.history_save_path:
                try:
                    atomic_save_json(self.epoch_metrics, self.history_save_path)
                except Exception as e:
                    print(f"  [Warning saving incremental history]: {e}")

        # Save latest checkpoint after every completed epoch (distinct from best checkpoint)
        if self.latest_ckpt_path:
            try:
                atomic_save_model(self.model, self.latest_ckpt_path)
                if self.latest_meta_json_path:
                    try:
                        latest_sha256 = calculate_file_sha256(self.latest_ckpt_path)
                        l_meta = {
                            "checkpoint_path": self.latest_ckpt_path,
                            "sha256": latest_sha256,
                            "stage": self.current_stage,
                            "stage_epoch": stage_epoch,
                            "global_epoch": global_epoch,
                            "val_loss": round(float(logs.get("val_loss", 0.0)), 4) if logs and logs.get("val_loss") is not None else None,
                            "val_accuracy": round(float(logs.get("val_accuracy", 0.0)), 4) if logs and logs.get("val_accuracy") is not None else None,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        atomic_save_json(l_meta, self.latest_meta_json_path)
                    except Exception as lm_err:
                        print(f"  [Warning saving latest metadata JSON]: {lm_err}")
            except Exception as e:
                print(f"  [Warning saving latest checkpoint]: {e}")

        gc.collect()

class GlobalModelCheckpointTracker(tf.keras.callbacks.Callback):
    """
    Tracks global best validation loss and accuracy across multiple training stages
    (Warmup and Fine-Tuning) without state resets.

    Key guarantees:
    - Retains best validation loss across stage boundaries.
    - NEVER overwrites a superior warmup checkpoint with a worse fine-tuning checkpoint.
    - Saves 'exp1_mobilenetv2_warmup_best.keras' whenever warmup validation loss improves.
    - Saves 'exp1_mobilenetv2_best.keras' whenever global validation loss improves.
    - Persists sidecar JSON metadata recording the exact stage, epoch, val_loss, and val_accuracy.
    - Does NOT reset state in on_train_begin().
    """
    def __init__(
        self,
        best_ckpt_path: str,
        warmup_best_ckpt_path: str,
        meta_json_path: str,
        warmup_meta_json_path: str = None,
        memory_monitor: SafetyMemoryMonitorCallback = None
    ):
        super().__init__()
        self.best_ckpt_path = best_ckpt_path
        self.warmup_best_ckpt_path = warmup_best_ckpt_path
        self.meta_json_path = meta_json_path
        self.warmup_meta_json_path = warmup_meta_json_path
        self.memory_monitor = memory_monitor

        # Global best state (persists across stages)
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.best_stage = None
        self.best_global_epoch = None
        self.best_stage_epoch = None

        # Warmup specific best state
        self.warmup_best_val_loss = float("inf")
        self.warmup_best_val_acc = 0.0
        self.warmup_best_epoch = None

        # Active stage configuration
        self.current_stage = "warmup"
        self.epoch_offset = 0

    def set_stage(self, stage_name: str, epoch_offset: int = 0):
        """Transitions tracker to a new stage while preserving all historical best state."""
        self.current_stage = stage_name
        self.epoch_offset = epoch_offset
        print(f"[GlobalCheckpointTracker] Stage transitioned to: '{stage_name}' (Epoch offset: {epoch_offset})")

    def on_train_begin(self, logs=None):
        # Explicitly do NOT reset best_val_loss, best_metadata, or epoch_offset here!
        pass

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Safeguard Check: Prevent checkpointing an interrupted epoch
        if self.memory_monitor and self.memory_monitor.interrupted:
            print(f"\n[GlobalCheckpointTracker] Skipping checkpointing for {self.current_stage.capitalize()} Epoch {epoch+1}: "
                  f"Training was interrupted by RAM safeguard ({self.memory_monitor.termination_reason}).")
            return

        val_loss = logs.get("val_loss")
        val_acc = logs.get("val_accuracy")

        if val_loss is None:
            return

        stage_epoch = epoch + 1
        global_epoch = self.epoch_offset + stage_epoch

        # 1. Warmup-specific checkpoint preservation (save-safe: update state only AFTER save succeeds)
        if self.current_stage == "warmup" and val_loss < self.warmup_best_val_loss:
            try:
                atomic_save_model(self.model, self.warmup_best_ckpt_path)
                # Update warmup-best state ONLY after successful model save
                self.warmup_best_val_loss = float(val_loss)
                self.warmup_best_val_acc = float(val_acc) if val_acc is not None else 0.0
                self.warmup_best_epoch = stage_epoch
                print(f"\n[GlobalCheckpointTracker] Saved Stage 1 Warmup Best -> {self.warmup_best_ckpt_path} "
                      f"(Val Loss: {self.warmup_best_val_loss:.4f}, Val Acc: {self.warmup_best_val_acc*100:.2f}%)")

                # Handle warmup metadata JSON saving separately
                if self.warmup_meta_json_path:
                    try:
                        warmup_sha256 = calculate_file_sha256(self.warmup_best_ckpt_path)
                        w_meta = {
                            "checkpoint_path": self.warmup_best_ckpt_path,
                            "sha256": warmup_sha256,
                            "stage": "warmup",
                            "stage_epoch": self.warmup_best_epoch,
                            "global_epoch": self.warmup_best_epoch,
                            "val_loss": round(self.warmup_best_val_loss, 4),
                            "val_accuracy": round(self.warmup_best_val_acc, 4),
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        atomic_save_json(w_meta, self.warmup_meta_json_path)
                    except Exception as meta_err:
                        print(f"  [GlobalCheckpointTracker Warning] Warmup checkpoint saved successfully, but metadata JSON write failed: {meta_err}")

            except Exception as e:
                print(f"\n[GlobalCheckpointTracker Warning] Failed to save warmup best checkpoint to {self.warmup_best_ckpt_path}: {e}")
                print(f"  Warmup best state NOT updated. Previous best val_loss ({self.warmup_best_val_loss:.4f}) retained.")

        # 2. Global best checkpoint preservation across all stages (save-safe: update state only AFTER save succeeds)
        if val_loss < self.best_val_loss:
            prev_best = self.best_val_loss
            try:
                atomic_save_model(self.model, self.best_ckpt_path)
                # Update global-best state ONLY after successful model save
                self.best_val_loss = float(val_loss)
                self.best_val_acc = float(val_acc) if val_acc is not None else 0.0
                self.best_stage = self.current_stage
                self.best_global_epoch = global_epoch
                self.best_stage_epoch = stage_epoch

                print(f"\n[GlobalCheckpointTracker] *** NEW GLOBAL BEST CHECKPOINT ***")
                print(f"  Stage: {self.best_stage} | Stage Epoch: {self.best_stage_epoch} | Global Epoch: {self.best_global_epoch}")
                print(f"  Validation Loss: {self.best_val_loss:.4f} (improved from {prev_best:.4f}) | Val Acc: {self.best_val_acc*100:.2f}%")
                print(f"  Saved to: {self.best_ckpt_path}")

                # Handle metadata JSON saving separately without claiming model save failed if metadata write fails
                if self.meta_json_path:
                    try:
                        best_sha256 = calculate_file_sha256(self.best_ckpt_path)
                        meta = {
                            "checkpoint_path": self.best_ckpt_path,
                            "sha256": best_sha256,
                            "stage": self.best_stage,
                            "stage_epoch": self.best_stage_epoch,
                            "global_epoch": self.best_global_epoch,
                            "val_loss": round(self.best_val_loss, 4),
                            "val_accuracy": round(self.best_val_acc, 4),
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        atomic_save_json(meta, self.meta_json_path)
                    except Exception as meta_err:
                        print(f"  [GlobalCheckpointTracker Warning] Model checkpoint saved successfully, but metadata JSON write failed: {meta_err}")

            except Exception as e:
                print(f"\n[GlobalCheckpointTracker Warning] Failed to save global best checkpoint to {self.best_ckpt_path}: {e}")
                print(f"  Global best state NOT updated. Previous best val_loss ({self.best_val_loss:.4f}) retained.")
        else:
            print(f"  [GlobalCheckpointTracker] Val Loss {val_loss:.4f} did not beat global best {self.best_val_loss:.4f} "
                  f"(held by {self.best_stage} Epoch {self.best_stage_epoch}). Best checkpoint preserved.")

def get_standard_augmentation():
    """Biologically sound augmentation preserving lesion diagnostic structure."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", name="aug_hflip"),
        tf.keras.layers.RandomRotation(0.08, fill_mode="nearest", name="aug_rotation"),
        tf.keras.layers.RandomZoom(height_factor=(-0.08, 0.08), fill_mode="nearest", name="aug_zoom"),
        tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05, fill_mode="nearest", name="aug_trans"),
        tf.keras.layers.RandomContrast(0.08, name="aug_contrast"),
        tf.keras.layers.RandomBrightness(0.08, name="aug_brightness"),
    ], name="exp1_data_augmentation")

def build_streaming_pipelines(batch_size: int = 16):
    """
    Constructs memory-efficient streaming tf.data pipelines for all 14,438 training images.
    Strictly uses prefetch(1) and no in-memory caching.
    Ensures that the historical test directory is NEVER accessed.
    """
    train_dir = config.TRAIN_DIR
    valid_dir = config.VALID_DIR
    test_dir = config.TEST_DIR

    # Explicit dataset isolation assertions
    assert os.path.abspath(train_dir) != os.path.abspath(test_dir), "CRITICAL: train_dir must not point to TEST_DIR!"
    assert os.path.abspath(valid_dir) != os.path.abspath(test_dir), "CRITICAL: valid_dir must not point to TEST_DIR!"
    assert "test" not in os.path.normpath(train_dir).split(os.sep), "CRITICAL: train_dir path must not contain 'test'!"
    assert "test" not in os.path.normpath(valid_dir).split(os.sep), "CRITICAL: valid_dir path must not contain 'test'!"

    print(f"[Data Pipeline Policy] Historical test set ({test_dir}) is STRICTLY EXCLUDED and NEVER ACCESSED.")
    print(f"[Data Pipeline] Loading streaming datasets from:")
    print(f"  Train: {train_dir}")
    print(f"  Valid: {valid_dir}")

    raw_train = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        batch_size=batch_size,
        image_size=config.IMG_SIZE,
        shuffle=True,
        seed=config.RANDOM_SEED
    )
    raw_val = tf.keras.utils.image_dataset_from_directory(
        valid_dir,
        labels="inferred",
        label_mode="categorical",
        batch_size=batch_size,
        image_size=config.IMG_SIZE,
        shuffle=False
    )

    class_names = raw_train.class_names
    num_classes = len(class_names)
    print(f"[Data Pipeline] Discovered {num_classes} classes. Batch size: {batch_size}")

    aug = get_standard_augmentation()
    preprocess_fn = tf.keras.applications.mobilenet_v2.preprocess_input

    def prep_train(imgs, lbls):
        imgs = aug(imgs, training=True)
        imgs = preprocess_fn(imgs)
        return imgs, lbls

    def prep_eval(imgs, lbls):
        imgs = preprocess_fn(imgs)
        return imgs, lbls

    train_ds = raw_train.map(prep_train, num_parallel_calls=tf.data.AUTOTUNE).prefetch(1)
    val_ds = raw_val.map(prep_eval, num_parallel_calls=tf.data.AUTOTUNE).prefetch(1)

    return train_ds, val_ds, class_names

def run_experiment_1(
    batch_size: int = 16,
    warmup_epochs: int = 3,
    finetune_epochs: int = 8,
    warmup_lr: float = 1e-3,
    finetune_lr: float = 1e-5,
    unfreeze_layers: int = 15,
    run_id: str = None
):
    if run_id is None:
        run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    print("="*80)
    print(f" EXPERIMENT 1: FULL-DATASET STREAMING MOBILENETV2 (RUN ID: {run_id})")
    print("="*80)
    print(f"Run Identifier:     {run_id}")
    print(f"Training Set:       14,438 images (100% full dataset)")
    print(f"Validation Set:     3,095 images (for checkpoint selection and metric evaluation)")
    print(f"Test Set Policy:    STRICTLY EXCLUDED (Reserved for final verification)")
    print(f"Batch Size:         {batch_size}")
    print(f"Warmup Epochs:      {warmup_epochs} (LR={warmup_lr})")
    print(f"Fine-tune Epochs:   {finetune_epochs} (LR={finetune_lr}, top {unfreeze_layers} conv layers unfrozen)")
    print(f"Prefetch Buffer:    prefetch(1)")
    print(f"In-Memory Cache:    DISABLED (.cache() omitted)")
    print(f"RAM Safeguards:     RSS Limit: 750 MB | System Avail Limit: 300 MB (with 7s confirmation window)")
    print(f"Initial Process RSS:{get_ram_usage_mb():.1f} MB")
    print("-"*80)

    # 1. Output directories (Isolated per unique run_id; prevents stale checkpoint fallback)
    exp_dir = os.path.join(config.MODELS_DIR, "experiments", "exp1_full_streaming", run_id)
    reports_exp_dir = os.path.join(config.REPORTS_DIR, "experiments", "exp1_full_streaming", run_id)

    # Collision rejection: do not reuse or overwrite an existing run directory
    if os.path.exists(exp_dir) or os.path.exists(reports_exp_dir):
        raise FileExistsError(
            f"Execution rejected: Run directory already exists for run_id '{run_id}'. "
            f"Overwriting or reusing existing run directories is strictly prohibited. "
            f"Provide a unique run_id or omit to auto-generate a collision-resistant run ID."
        )

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(reports_exp_dir, exist_ok=True)

    ckpt_best = os.path.join(exp_dir, "exp1_mobilenetv2_best.keras")
    ckpt_warmup_best = os.path.join(exp_dir, "exp1_mobilenetv2_warmup_best.keras")
    ckpt_latest = os.path.join(exp_dir, "exp1_mobilenetv2_latest.keras")
    ckpt_meta_json = os.path.join(reports_exp_dir, "exp1_best_checkpoint_meta.json")
    ckpt_warmup_meta_json = os.path.join(reports_exp_dir, "exp1_warmup_best_checkpoint_meta.json")
    ckpt_latest_meta_json = os.path.join(reports_exp_dir, "exp1_latest_checkpoint_meta.json")
    incremental_history_json = os.path.join(reports_exp_dir, "exp1_incremental_history.json")

    # 2. Build streaming datasets (Train: 14,438 images, Val: 3,095 images)
    train_ds, val_ds, class_names = build_streaming_pipelines(batch_size=batch_size)
    num_classes = len(class_names)

    # 3. Model construction: MobileNetV2 with ImageNet weights
    tf.keras.backend.clear_session()
    gc.collect()

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=config.INPUT_SHAPE,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    model = build_transfer_model(
        base_model=base_model,
        num_classes=num_classes,
        model_name="leafscan_mobilenetv2_exp1",
        dropout_rate=0.3,
        dense_units=512
    )

    # Callbacks instantiated ONCE across the entire training lifecycle
    mem_monitor = SafetyMemoryMonitorCallback(
        history_save_path=incremental_history_json,
        latest_ckpt_path=ckpt_latest,
        latest_meta_json_path=ckpt_latest_meta_json
    )
    checkpoint_tracker = GlobalModelCheckpointTracker(
        best_ckpt_path=ckpt_best,
        warmup_best_ckpt_path=ckpt_warmup_best,
        meta_json_path=ckpt_meta_json,
        warmup_meta_json_path=ckpt_warmup_meta_json,
        memory_monitor=mem_monitor
    )

    # =========================================================================
    # STAGE 1: CLASSIFIER HEAD WARMUP (3 Epochs)
    # =========================================================================
    print("\n" + "="*70)
    print(f"STAGE 1: Training Classification Head ({warmup_epochs} Epochs, Backbone Frozen, LR={warmup_lr})")
    print("="*70)
    checkpoint_tracker.set_stage("warmup", epoch_offset=0)
    mem_monitor.set_stage("warmup", epoch_offset=0)

    optimizer_warmup = tf.keras.optimizers.Adam(learning_rate=warmup_lr)
    model.compile(
        optimizer=optimizer_warmup,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    warmup_callbacks = [
        mem_monitor,
        checkpoint_tracker,
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True, verbose=1)
    ]

    t_start = time.perf_counter()
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=warmup_epochs,
        callbacks=warmup_callbacks,
        verbose=1
    )
    warmup_duration_mins = (time.perf_counter() - t_start) / 60.0
    # Derive confirmed completed epoch counts strictly from mem_monitor.epoch_metrics
    completed_warmup_epochs = sum(1 for m in mem_monitor.epoch_metrics if m.get("stage") == "warmup")
    print(f"[Stage 1 Complete] Elapsed time: {warmup_duration_mins:.2f} mins (Confirmed completed {completed_warmup_epochs}/{warmup_epochs} epochs)")

    # =========================================================================
    # STAGE 2: DEEP CONVOLUTIONAL FINE-TUNING (Up to 8 Epochs)
    # =========================================================================
    h2 = None
    finetune_duration_mins = 0.0

    if mem_monitor.interrupted:
        print("\n[Safe Termination Triggered] Warmup was interrupted by RAM safeguard. Skipping Stage 2 fine-tuning.")
    else:
        print("\n" + "="*70)
        print(f"STAGE 2: Fine-Tuning Top {unfreeze_layers} Conv Layers (Up to {finetune_epochs} Epochs, BN Frozen, LR={finetune_lr})")
        print("="*70)
        # Use confirmed completed warmup count as fine-tuning global epoch offset
        checkpoint_tracker.set_stage("finetune", epoch_offset=completed_warmup_epochs)
        mem_monitor.set_stage("finetune", epoch_offset=completed_warmup_epochs)

        base_model.trainable = True

        # Freeze all layers before the top N layers
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False

        # Freeze all BatchNormalization layers throughout backbone
        bn_count = 0
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
                bn_count += 1

        trainable_weights_count = sum(len(l.trainable_weights) for l in model.layers if l.trainable)
        print(f"Total Base Layers: {len(base_model.layers)} | Frozen BN Layers: {bn_count} | Trainable Weights: {trainable_weights_count}")

        optimizer_ft = tf.keras.optimizers.Adam(learning_rate=finetune_lr)
        model.compile(
            optimizer=optimizer_ft,
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        ft_callbacks = [
            mem_monitor,
            checkpoint_tracker,  # Replaces isolated ModelCheckpoint, preserving warmup best!
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1)
        ]

        t_ft = time.perf_counter()
        h2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=finetune_epochs,
            callbacks=ft_callbacks,
            verbose=1
        )
        finetune_duration_mins = (time.perf_counter() - t_ft) / 60.0
        completed_finetune_epochs = sum(1 for m in mem_monitor.epoch_metrics if m.get("stage") == "finetune")
        print(f"[Stage 2 Complete] Elapsed time: {finetune_duration_mins:.2f} mins (Confirmed completed {completed_finetune_epochs}/{finetune_epochs} epochs)")

    total_training_duration_mins = warmup_duration_mins + finetune_duration_mins
    completed_finetune_epochs = sum(1 for m in mem_monitor.epoch_metrics if m.get("stage") == "finetune") if h2 else 0
    total_completed_epochs = len(mem_monitor.epoch_metrics)
    print(f"\n[Total Training Duration] {total_training_duration_mins:.2f} mins (Total Confirmed Completed Epochs: {total_completed_epochs})")

    # Combine training histories using confirmed completed epochs only (excludes incomplete interrupted epochs)
    combined_history = {}
    for k in h1.history.keys():
        w_vals = [float(v) for v in h1.history[k][:completed_warmup_epochs]]
        ft_vals = [float(v) for v in h2.history.get(k, [])[:completed_finetune_epochs]] if h2 else []
        combined_history[k] = w_vals + ft_vals

    # Plot learning curves if any completed epochs exist
    if total_completed_epochs > 0:
        curves_path = os.path.join(reports_exp_dir, "exp1_learning_curves.png")
        plot_and_save_training_curves(combined_history, curves_path, f"Exp 1 MobileNetV2 ({run_id})")

    print(f"\n[Checkpoints Preserved in Run Directory: {exp_dir}]")
    print(f"  Global Best Checkpoint: {ckpt_best} (Exists: {os.path.exists(ckpt_best)})")
    print(f"  Warmup Best Checkpoint: {ckpt_warmup_best} (Exists: {os.path.exists(ckpt_warmup_best)})")
    print(f"  Latest Run Checkpoint:  {ckpt_latest} (Exists: {os.path.exists(ckpt_latest)})")

    # =========================================================================
    # STAGE 3: RIGOROUS EVALUATION (ON VALIDATION SET ONLY)
    # =========================================================================
    print("\n" + "="*70)
    print(f"STAGE 3: Rigorous Evaluation on Full Validation Set (Run: {run_id})")
    print("="*70)

    # Strictly evaluate checkpoints produced by the CURRENT run in exp_dir
    # Checkpoint provenance selection hierarchy with strict SHA256 & model load verification:
    # Priority 1: Global best checkpoint (exp1_mobilenetv2_best.keras)
    # Priority 2: Warmup best checkpoint (exp1_mobilenetv2_warmup_best.keras)
    # Priority 3: Latest checkpoint fallback (exp1_mobilenetv2_latest.keras)
    # In-memory tracker state alone cannot bypass this verification.
    eval_model = None
    selected_eval_model_path = None
    selected_ckpt_type = None
    selected_eval_metadata = None
    rejection_logs = []

    candidates = [
        ("global_best", ckpt_best, ckpt_meta_json, "Global best checkpoint"),
        ("warmup_best", ckpt_warmup_best, ckpt_warmup_meta_json, "Warmup best checkpoint"),
        ("latest_fallback", ckpt_latest, ckpt_latest_meta_json, "Latest checkpoint")
    ]

    for c_type, c_path, c_meta_path, c_name in candidates:
        if not os.path.exists(c_path) and not os.path.exists(c_meta_path):
            continue

        is_valid, meta_loaded, reason = verify_checkpoint_metadata_sha256(c_path, c_meta_path, c_name)
        if not is_valid:
            print(f"[Checkpoint Verification Rejected] {c_name} rejected: {reason}")
            rejection_logs.append({"checkpoint": c_type, "reason": reason})
            continue

        print(f"[Checkpoint Verified] {c_name} metadata and SHA256 verified successfully (SHA256: {meta_loaded['sha256'][:16]}...). Attempting model load...")
        try:
            loaded_model = tf.keras.models.load_model(c_path, compile=False)
            eval_model = loaded_model
            selected_eval_model_path = c_path
            selected_ckpt_type = c_type
            selected_eval_metadata = {
                "checkpoint_path": c_path,
                "sha256": meta_loaded["sha256"],
                "stage": meta_loaded["stage"],
                "stage_epoch": meta_loaded["stage_epoch"],
                "global_epoch": meta_loaded["global_epoch"],
                "val_loss": round(float(meta_loaded["val_loss"]), 4),
                "val_accuracy": round(float(meta_loaded["val_accuracy"]), 4) if meta_loaded.get("val_accuracy") is not None else None
            }
            print(f"[Checkpoint Loaded] Successfully loaded {c_name} from '{c_path}'. Selected for Stage 3 validation.")
            break
        except Exception as e:
            load_err_msg = f"Failed to load model from '{c_path}': {type(e).__name__}: {e}"
            print(f"[Checkpoint Model Load Failed] {c_name} rejected: {load_err_msg}")
            rejection_logs.append({"checkpoint": c_type, "reason": load_err_msg})

    if selected_eval_model_path is None or eval_model is None:
        print(f"\n[STAGE 3 SKIPPED] NO_VALID_CHECKPOINT: No valid checkpoint with verified metadata, matching SHA256, and loadable weights was available for current run '{run_id}'.")
        print("Stale checkpoint fallback is strictly prevented. Model evaluation skipped.")
        val_metrics = {
            "status": "NO_VALID_CHECKPOINT",
            "run_id": run_id,
            "message": f"No valid, loadable checkpoint was produced during current run '{run_id}'.",
            "rejection_reasons": rejection_logs
        }
    else:
        print(f"\nProceeding to evaluation with already-loaded model: {selected_eval_model_path} (Type: {selected_ckpt_type}, SHA256: {selected_eval_metadata['sha256'][:16]}...)")
        print(f"Evaluating across full validation set...")
        y_true, y_probs = [], []
        t_eval = time.perf_counter()
        sample_count = 0

        for imgs, lbls in val_ds:
            preds = eval_model(imgs, training=False).numpy()
            y_probs.append(preds)
            y_true.append(np.argmax(lbls.numpy(), axis=-1))
            sample_count += len(lbls)

        lat_ms = (time.perf_counter() - t_eval) / max(1, sample_count) * 1000.0
        y_true = np.concatenate(y_true, axis=0)
        y_probs = np.concatenate(y_probs, axis=0)
        y_pred = np.argmax(y_probs, axis=-1)

        acc = float(accuracy_score(y_true, y_pred))
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        p_c, r_c, f1_c, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0)

        per_class = {}
        for i, c_name in enumerate(class_names):
            per_class[c_name] = {
                "precision": round(float(p_c[i]), 4),
                "recall": round(float(r_c[i]), 4),
                "f1_score": round(float(f1_c[i]), 4),
                "support": int(np.sum(y_true == i))
            }

        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

        val_metrics = {
            "status": "EVALUATED",
            "run_id": run_id,
            "accuracy": round(acc, 4),
            "macro_precision": round(float(p_m), 4),
            "macro_recall": round(float(r_m), 4),
            "macro_f1": round(float(f1_m), 4),
            "weighted_f1": round(float(f1_wt), 4),
            "avg_latency_ms": round(float(lat_ms), 2),
            "total_samples": int(len(y_true)),
            "checkpoint_evaluated": selected_eval_model_path,
            "checkpoint_evaluated_sha256": selected_eval_metadata["sha256"],
            "checkpoint_type": selected_ckpt_type,
            "best_stage": selected_eval_metadata["stage"],
            "best_stage_epoch": selected_eval_metadata["stage_epoch"],
            "best_global_epoch": selected_eval_metadata["global_epoch"],
            "best_val_loss": round(float(selected_eval_metadata["val_loss"]), 4) if selected_eval_metadata["val_loss"] is not None else None,
            "best_val_accuracy": round(float(selected_eval_metadata["val_accuracy"]), 4) if selected_eval_metadata["val_accuracy"] is not None else None,
            "rejection_reasons": rejection_logs,
            "per_class": per_class,
            "confusion_matrix": cm.tolist()
        }

        # Save validation confusion matrix plot
        plt.figure(figsize=(14, 12))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
        plt.title(f"Exp 1 MobileNetV2 ({run_id}) - Validation CM (Acc: {acc*100:.2f}%, Macro F1: {f1_m*100:.2f}%)", fontsize=13, fontweight="bold", pad=12)
        plt.colorbar(fraction=0.046, pad=0.04)

        short_names = [
            c.replace("Tomato___", "Tom_")
             .replace("Tomato_", "Tom_")
             .replace("Pepper__bell___", "Pep_")
             .replace("Potato___", "Pot_")
            for c in class_names
        ]
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, short_names, rotation=45, ha="right", fontsize=9)
        plt.yticks(tick_marks, short_names, fontsize=9)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                color = "white" if val > thresh else "black"
                plt.text(j, i, format(val, "d"), ha="center", va="center", color=color, fontsize=8)

        plt.ylabel("Ground Truth", fontsize=11, fontweight="bold")
        plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
        plt.tight_layout()
        val_cm_path = os.path.join(reports_exp_dir, "exp1_validation_cm.png")
        plt.savefig(val_cm_path, dpi=180)
        plt.close()

    # Print summary
    print("\n" + "="*80)
    print(f"EXPERIMENT 1 VALIDATION BENCHMARK RESULTS (RUN: {run_id})")
    print("="*80)
    if val_metrics.get("status") == "NO_VALID_CHECKPOINT":
        print(f"Validation Status:         NO_VALID_CHECKPOINT")
        print(f"Message:                   {val_metrics.get('message')}")
        if rejection_logs:
            print("Checkpoint Rejections:")
            for r in rejection_logs:
                print(f"  - [{r['checkpoint']}] {r['reason']}")
    else:
        print(f"Evaluated Checkpoint:      {selected_eval_model_path} ({selected_ckpt_type})")
        print(f"Checkpoint SHA256:         {selected_eval_metadata['sha256']}")
        print(f"Validation Accuracy:       {val_metrics['accuracy']*100:.2f}% (Baseline Champion Val: 83.88%)")
        print(f"Validation Macro F1:       {val_metrics['macro_f1']*100:.2f}% (Baseline Champion Val: 82.00%)")
        print(f"Validation Macro Recall:   {val_metrics['macro_recall']*100:.2f}%")
        print(f"Validation Weighted F1:    {val_metrics['weighted_f1']*100:.2f}%")
        stage_desc = selected_eval_metadata['stage'].capitalize() if selected_eval_metadata['stage'] else "Unknown"
        val_loss_str = f"{selected_eval_metadata['val_loss']:.4f}" if selected_eval_metadata['val_loss'] is not None else "N/A"
        print(f"Best Validation Epoch:     {stage_desc} Epoch {selected_eval_metadata['stage_epoch']} (Global Epoch {selected_eval_metadata['global_epoch']}, Loss: {val_loss_str})")
        print(f"Avg CPU Inference Latency: {val_metrics['avg_latency_ms']:.2f} ms/sample")
    print(f"Confirmed Completed Epochs:{total_completed_epochs} (Warmup: {completed_warmup_epochs}, Fine-tuning: {completed_finetune_epochs})")
    print(f"Peak Process RSS:          {mem_monitor.peak_rss_mb:.1f} MB")
    print(f"Minimum Available RAM:     {mem_monitor.min_avail_mb:.1f} MB")
    print(f"Total Training Duration:   {total_training_duration_mins:.2f} mins")
    print("="*80)

    # Determine execution status
    if mem_monitor.interrupted:
        train_exec_status = "INTERRUPTED_RAM_SAFEGUARD"
    elif (completed_warmup_epochs < warmup_epochs) or (h2 and completed_finetune_epochs < finetune_epochs):
        train_exec_status = "STOPPED_EARLY_STOPPING"
    else:
        train_exec_status = "COMPLETED"

    has_valid_eval = (val_metrics.get("status") == "EVALUATED")
    if has_valid_eval:
        overall_status = train_exec_status
    else:
        overall_status = f"{train_exec_status}_NO_VALID_CHECKPOINT"

    # Save comprehensive results JSON
    result_payload = {
        "experiment_name": "Exp 1: Full-Dataset Streaming MobileNetV2",
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "execution_status": {
            "status": overall_status,
            "training_status": train_exec_status,
            "evaluation_completed": has_valid_eval,
            "checkpoint_evaluated_type": selected_ckpt_type if has_valid_eval else None,
            "interrupted": mem_monitor.interrupted,
            "interruption_details": {
                "stage": mem_monitor.interruption_stage,
                "global_epoch": mem_monitor.interruption_global_epoch,
                "stage_epoch": mem_monitor.interruption_stage_epoch,
                "batch": mem_monitor.interruption_batch,
                "termination_reason": mem_monitor.termination_reason
            } if mem_monitor.interrupted else None,
            "latest_completed_epoch": {
                "stage": mem_monitor.latest_completed_stage,
                "stage_epoch": mem_monitor.latest_completed_stage_epoch,
                "global_epoch": mem_monitor.latest_completed_global_epoch
            }
        },
        "training_config": {
            "run_id": run_id,
            "train_samples": 14438,
            "val_samples": 3095,
            "test_set_status": "STRICTLY_EXCLUDED",
            "batch_size": batch_size,
            "warmup_epochs_planned": warmup_epochs,
            "warmup_epochs_completed": completed_warmup_epochs,
            "finetune_epochs_planned": finetune_epochs,
            "finetune_epochs_completed": completed_finetune_epochs,
            "total_epochs_completed": total_completed_epochs,
            "warmup_lr": warmup_lr,
            "finetune_lr": finetune_lr,
            "unfreeze_layers": unfreeze_layers,
            "prefetch": 1,
            "in_memory_cache": False
        },
        "hardware_metrics": {
            "peak_process_rss_mb": round(float(mem_monitor.peak_rss_mb), 1),
            "min_system_avail_mb": round(float(mem_monitor.min_avail_mb), 1),
            "total_training_duration_mins": round(float(total_training_duration_mins), 2),
            "warmup_duration_mins": round(float(warmup_duration_mins), 2),
            "finetune_duration_mins": round(float(finetune_duration_mins), 2)
        },
        "epoch_metrics": mem_monitor.epoch_metrics,
        "validation_metrics": val_metrics,
        "combined_history": combined_history,
        "checkpoints": {
            "run_id": run_id,
            "checkpoint_evaluated": selected_eval_model_path if has_valid_eval else None,
            "checkpoint_evaluated_sha256": selected_eval_metadata["sha256"] if has_valid_eval else None,
            "evaluated_checkpoint_type": selected_ckpt_type if has_valid_eval else None,
            "best": ckpt_best if os.path.exists(ckpt_best) else None,
            "warmup_best": ckpt_warmup_best if os.path.exists(ckpt_warmup_best) else None,
            "latest": ckpt_latest if os.path.exists(ckpt_latest) else None,
            "best_stage": selected_eval_metadata["stage"] if has_valid_eval else None,
            "best_stage_epoch": selected_eval_metadata["stage_epoch"] if has_valid_eval else None,
            "best_global_epoch": selected_eval_metadata["global_epoch"] if has_valid_eval else None,
            "best_val_loss": selected_eval_metadata["val_loss"] if has_valid_eval else None,
            "best_val_accuracy": selected_eval_metadata["val_accuracy"] if has_valid_eval else None,
            "rejection_reasons": rejection_logs
        }
    }

    result_json_path = os.path.join(reports_exp_dir, "exp1_full_streaming_results.json")
    atomic_save_json(result_payload, result_json_path)
    print(f"[Done] Saved full results to: {result_json_path}")

    # Verify production champion remained untouched via cryptographic SHA256
    prod_path = config.MODEL_PATH
    EXPECTED_PROD_CHAMPION_SHA256 = "ac1b1d8428dfe2ba1695d6a447c7384362fcf310d445f9ad51a2877929c8fee2"
    if os.path.exists(prod_path):
        actual_prod_sha256 = calculate_file_sha256(prod_path)
        if actual_prod_sha256.lower() == EXPECTED_PROD_CHAMPION_SHA256.lower():
            print(f"\n[Safety Audit PASS] Production champion SHA256 verified untouched: {actual_prod_sha256}")
        else:
            print(f"\n[CRITICAL WARNING] Production champion SHA256 integrity breach!")
            print(f"  Expected: {EXPECTED_PROD_CHAMPION_SHA256}")
            print(f"  Actual:   {actual_prod_sha256}")
    else:
        print(f"\n[CRITICAL WARNING] Production champion model missing at '{prod_path}'!")

    return result_payload

if __name__ == "__main__":
    run_experiment_1()
