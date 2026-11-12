import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
import json
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

# Ensure ml-service root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

def get_preprocessing_function(architecture: str):
    """
    Returns the exact input preprocessing function for a specified architecture.
    """
    arch = architecture.lower().strip()
    if "mobilenet" in arch:
        return tf.keras.applications.mobilenet_v2.preprocess_input
    elif "resnet" in arch:
        return tf.keras.applications.resnet50.preprocess_input
    elif "efficientnet" in arch:
        return tf.keras.applications.efficientnet.preprocess_input
    else:
        raise ValueError(f"Unsupported architecture for preprocessing: {architecture}")

def get_data_augmentation():
    """
    Returns realistic, biologically appropriate leaf data augmentation pipeline.
    Preserves lesion geometry while improving invariance to field capture conditions.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", name="aug_hflip"),
        tf.keras.layers.RandomRotation(0.08, fill_mode="nearest", name="aug_rotation"),
        tf.keras.layers.RandomZoom(height_factor=(-0.08, 0.08), fill_mode="nearest", name="aug_zoom"),
        tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05, fill_mode="nearest", name="aug_trans"),
        tf.keras.layers.RandomContrast(0.10, name="aug_contrast"),
        tf.keras.layers.RandomBrightness(0.08, name="aug_brightness"),
    ], name="training_augmentation")

def load_dataset(
    directory: str,
    batch_size: int = 16,
    max_samples_per_class: int = 0,
    shuffle: bool = True,
    seed: int = 42
):
    """
    Loads dataset from directory. Supports full dataset or balanced sampling per class.
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Dataset directory not found: {directory}")

    class_names = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
    num_classes = len(class_names)

    if max_samples_per_class <= 0:
        ds = tf.keras.utils.image_dataset_from_directory(
            directory,
            labels="inferred",
            label_mode="categorical",
            batch_size=batch_size,
            image_size=config.IMG_SIZE,
            shuffle=shuffle,
            seed=seed
        )
        return ds, class_names

    # Balanced per-class sampling
    file_paths = []
    labels_one_hot = []
    r = random.Random(seed)

    for cls_idx, cls in enumerate(class_names):
        cls_dir = os.path.join(directory, cls)
        files = [
            os.path.join(cls_dir, f) for f in sorted(os.listdir(cls_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))
        ]
        if shuffle:
            r.shuffle(files)
        selected = files[:max_samples_per_class]
        for fp in selected:
            file_paths.append(fp)
            one_hot = np.zeros(num_classes, dtype=np.float32)
            one_hot[cls_idx] = 1.0
            labels_one_hot.append(one_hot)

    def parse_image(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, config.IMG_SIZE)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels_one_hot))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(file_paths), seed=seed)
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    return ds, class_names

def calculate_class_weights(train_source, num_classes: int = 15, max_samples_per_class: int = 0):
    """
    Computes balanced class weights with zero unnecessary memory allocation.
    Accepts directory path (str), or dataset/fallback.
    Formula: weight = total_samples / (num_classes * class_count)
    """
    if isinstance(train_source, str) and os.path.isdir(train_source):
        class_names = sorted([d for d in os.listdir(train_source) if os.path.isdir(os.path.join(train_source, d))])
        counts = []
        for cls in class_names:
            cls_dir = os.path.join(train_source, cls)
            files = [f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))]
            c = len(files)
            if max_samples_per_class > 0:
                c = min(c, max_samples_per_class)
            counts.append(max(1, c))
        total_samples = sum(counts)
        n_classes = len(class_names)
        return {int(idx): float(total_samples / (n_classes * c)) for idx, c in enumerate(counts)}

    return {int(i): 1.0 for i in range(num_classes)}

def build_transfer_model(
    base_model: tf.keras.Model,
    num_classes: int,
    model_name: str,
    dropout_rate: float = 0.3,
    dense_units: int = 512
) -> tf.keras.Model:
    """
    Attaches GlobalAveragePooling2D, BatchNorm, Dense, Dropout, and Softmax classifier.
    """
    inputs = base_model.input
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D(name="head_gap")(x)
    x = tf.keras.layers.BatchNormalization(name="head_bn")(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu", name="head_dense")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model

def plot_and_save_training_curves(history_data: dict, output_png_path: str, title: str):
    """
    Generates training/validation loss and accuracy curves plot.
    """
    epochs = range(1, len(history_data.get("loss", [])) + 1)
    if not epochs:
        return

    plt.figure(figsize=(12, 4.5))

    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_data.get("loss", []), "o-", label="Train Loss", color="#1e40af")
    if "val_loss" in history_data:
        plt.plot(epochs, history_data.get("val_loss", []), "s-", label="Val Loss", color="#dc2626")
    plt.title(f"{title} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Categorical Cross-Entropy")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    # Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_data.get("accuracy", []), "o-", label="Train Acc", color="#16a34a")
    if "val_accuracy" in history_data:
        plt.plot(epochs, history_data.get("val_accuracy", []), "s-", label="Val Acc", color="#d97706")
    plt.title(f"{title} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    plt.savefig(output_png_path, dpi=150)
    plt.close()
