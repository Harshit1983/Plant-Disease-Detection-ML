import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
import json
import random
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg") # Non-interactive headless backend
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

# Add ml-service root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

def build_model(num_classes: int, input_shape: tuple = (224, 224, 3)) -> tuple:
    """
    Construct Transfer Learning CNN architecture using MobileNetV2 with an agronomic classification head.
    """
    print(f"[Model Builder] Initializing MobileNetV2 backbone (input_shape={input_shape})...")
    
    # Load pretrained MobileNetV2 without top classifier
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    # Freeze base model weights initially
    base_model.trainable = False

    # Build classification head directly on base_model.output
    inputs = base_model.input
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.BatchNormalization(name="head_batch_norm")(x)
    x = tf.keras.layers.Dense(512, activation="relu", name="head_dense_512")(x)
    x = tf.keras.layers.Dropout(0.3, name="head_dropout_0.3")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="disease_predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="leafscan_mobilenetv2")
    return model, base_model

def get_data_augmentation():
    """Create training-only data augmentation pipeline."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ], name="data_augmentation")

def load_dataset(directory: str, batch_size: int, max_samples_per_class: int = 0, shuffle: bool = True):
    """
    Load dataset from split folder. Supports full dataset or balanced subset for fast CPU training.
    """
    if max_samples_per_class <= 0:
        ds = tf.keras.utils.image_dataset_from_directory(
            directory,
            labels="inferred",
            label_mode="categorical",
            batch_size=batch_size,
            image_size=config.IMG_SIZE,
            shuffle=shuffle,
            seed=config.RANDOM_SEED
        )
        return ds, ds.class_names

    # Collect class folders
    class_names = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
    num_classes = len(class_names)
    file_paths = []
    labels_one_hot = []

    for cls_idx, cls in enumerate(class_names):
        cls_dir = os.path.join(directory, cls)
        files = [
            os.path.join(cls_dir, f) for f in sorted(os.listdir(cls_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))
        ]
        if shuffle:
            r = random.Random(config.RANDOM_SEED)
            r.shuffle(files)
        selected = files[:max_samples_per_class]
        for fp in selected:
            file_paths.append(fp)
            one_hot = np.zeros(num_classes, dtype=np.float32)
            one_hot[cls_idx] = 1.0
            labels_one_hot.append(one_hot)

    print(f"Sampled {len(file_paths)} images ({max_samples_per_class} max per class) from {directory} for optimized CPU training.")

    def parse_image(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, config.IMG_SIZE)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels_one_hot))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(file_paths), seed=config.RANDOM_SEED)
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    return ds, class_names

def train_pipeline(
    train_dir: str = None,
    valid_dir: str = None,
    epochs: int = 10,
    fine_tune_epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    fine_tune_lr: float = 1e-4,
    fine_tune_layers: int = 25,
    max_samples_per_class: int = 0,
    model_save_path: str = None
):
    train_dir = train_dir or config.TRAIN_DIR
    valid_dir = valid_dir or config.VALID_DIR
    model_save_path = model_save_path or config.MODEL_PATH

    print("=================================================================")
    print("      LeafScan TensorFlow/Keras CNN Training Pipeline")
    print("=================================================================")
    print(f"Train Dataset:         {train_dir}")
    print(f"Valid Dataset:         {valid_dir}")
    print(f"Model Save Path:       {model_save_path}")
    print(f"Batch Size:            {batch_size}")
    print(f"Frozen Epochs:         {epochs} (lr={lr})")
    print(f"Fine-tune Epochs:      {fine_tune_epochs} (lr={fine_tune_lr}, unfreezing top {fine_tune_layers} layers)")
    if max_samples_per_class > 0:
        print(f"Max Samples Per Class: {max_samples_per_class} (CPU optimization mode)")
    print("-----------------------------------------------------------------")

    if not os.path.exists(train_dir) or not os.path.exists(valid_dir):
        raise FileNotFoundError(
            f"Dataset split not found. Please run 'python scripts/prepare_dataset.py' first."
        )

    # Load datasets
    train_ds, class_names = load_dataset(
        train_dir,
        batch_size=batch_size,
        max_samples_per_class=max_samples_per_class,
        shuffle=True
    )
    # Scale validation samples proportionally if subset mode is used
    val_max = max(10, int(max_samples_per_class * 0.25)) if max_samples_per_class > 0 else 0
    valid_ds, _ = load_dataset(
        valid_dir,
        batch_size=batch_size,
        max_samples_per_class=val_max,
        shuffle=False
    )

    num_classes = len(class_names)
    print(f"\nDiscovered {num_classes} classes: {class_names}")

    # Save class indices
    os.makedirs(os.path.dirname(config.CLASS_INDICES_PATH), exist_ok=True)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    with open(config.CLASS_INDICES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)

    # Calculate class weights to handle imbalance
    print("Calculating class weights...")
    train_labels = []
    for _, labels in train_ds.unbatch():
        train_labels.append(np.argmax(labels.numpy()))
    train_labels = np.array(train_labels)

    classes_unique = np.unique(train_labels)
    class_weights_arr = compute_class_weight("balanced", classes=classes_unique, y=train_labels)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes_unique, class_weights_arr)}
    print(f"Computed class weights: {class_weight_dict}")

    # Preprocessing & Data Augmentation Pipeline
    data_augmentation = get_data_augmentation()

    def preprocess_train(images, labels):
        images = data_augmentation(images, training=True)
        images = tf.keras.applications.mobilenet_v2.preprocess_input(images)
        return images, labels

    def preprocess_val(images, labels):
        images = tf.keras.applications.mobilenet_v2.preprocess_input(images)
        return images, labels

    train_ds = train_ds.map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    valid_ds = valid_ds.map(preprocess_val, num_parallel_calls=tf.data.AUTOTUNE)
    valid_ds = valid_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Build Model
    model, base_model = build_model(num_classes=num_classes, input_shape=config.INPUT_SHAPE)

    # Compile with Adam and CrossEntropy
    metrics = [
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=metrics
    )

    # Callbacks
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print("\n--- Phase 1: Training Classification Head (Backbone Frozen) ---")
    history_phase1 = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )

    combined_history = {k: [float(val) for val in v] for k, v in history_phase1.history.items()}

    # Phase 2: Fine-Tuning
    if fine_tune_epochs > 0:
        print(f"\n--- Phase 2: Fine-Tuning (Unfreezing top {fine_tune_layers} layers of backbone) ---")
        base_model.trainable = True
        # Freeze all layers before the top N layers
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
            loss="categorical_crossentropy",
            metrics=metrics
        )

        history_phase2 = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=fine_tune_epochs,
            class_weight=class_weight_dict,
            callbacks=callbacks
        )

        for k, v in history_phase2.history.items():
            combined_history[k].extend([float(val) for val in v])

    # Save final best model (if not already saved by checkpoint)
    if not os.path.exists(model_save_path):
        model.save(model_save_path)
    print(f"\n[Saved] Best model checkpoint saved to {model_save_path}")

    # Save history JSON
    with open(config.TRAINING_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(combined_history, f, indent=2)
    print(f"[Saved] Training history saved to {config.TRAINING_HISTORY_PATH}")

    # Plot & Save Learning Curves
    plot_learning_curves(combined_history, config.REPORTS_DIR)

    # Save Training Config Summary
    config_summary = {
        "num_classes": num_classes,
        "class_names": class_names,
        "initial_epochs": epochs,
        "fine_tune_epochs": fine_tune_epochs,
        "batch_size": batch_size,
        "initial_learning_rate": lr,
        "fine_tune_learning_rate": fine_tune_lr,
        "fine_tune_unfrozen_layers": fine_tune_layers,
        "max_samples_per_class": max_samples_per_class,
        "model_architecture": "MobileNetV2 + Custom Dense Head",
        "model_save_path": model_save_path,
        "final_train_accuracy": combined_history["accuracy"][-1],
        "final_val_accuracy": combined_history["val_accuracy"][-1],
        "final_train_loss": combined_history["loss"][-1],
        "final_val_loss": combined_history["val_loss"][-1]
    }
    with open(os.path.join(config.REPORTS_DIR, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_summary, f, indent=2)

    print("\nTraining completed successfully!")
    return model, combined_history

def plot_learning_curves(history: dict, reports_dir: str):
    """Plot Accuracy and Loss learning curves and save to reports/."""
    epochs_range = range(1, len(history["accuracy"]) + 1)

    plt.figure(figsize=(14, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["accuracy"], label="Training Accuracy", color="#2563EB", lw=2)
    plt.plot(epochs_range, history["val_accuracy"], label="Validation Accuracy", color="#059669", lw=2, linestyle="--")
    plt.title("Model Accuracy Across Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["loss"], label="Training Loss", color="#DC2626", lw=2)
    plt.plot(epochs_range, history["val_loss"], label="Validation Loss", color="#D97706", lw=2, linestyle="--")
    plt.title("Model Loss Across Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")

    plt.tight_layout()
    chart_path = os.path.join(reports_dir, "learning_curves.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"[Saved] Learning curves plot saved to {chart_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV2 for Plant Disease Detection")
    parser.add_argument("--epochs", type=int, default=5, help="Number of head training epochs")
    parser.add_argument("--fine-tune-epochs", type=int, default=2, help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Initial learning rate")
    parser.add_argument("--fine-tune-lr", type=float, default=config.FINE_TUNE_LEARNING_RATE, help="Fine tuning learning rate")
    parser.add_argument("--max-samples-per-class", type=int, default=0, help="Max samples per class (0 for all)")
    parser.add_argument("--model-path", type=str, default=config.MODEL_PATH, help="Destination model path")
    args = parser.parse_args()

    train_pipeline(
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        fine_tune_lr=args.fine_tune_lr,
        max_samples_per_class=args.max_samples_per_class,
        model_save_path=args.model_path
    )
