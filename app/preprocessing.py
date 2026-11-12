import io
from PIL import Image
import numpy as np
import tensorflow as tf
from typing import Tuple

# Ensure consistent image dimensions with config
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

def validate_and_load_image(image_bytes: bytes) -> Image.Image:
    """
    Validate that raw bytes represent a readable, supported image.
    Converts image to RGB mode.
    """
    if not image_bytes or len(image_bytes) == 0:
        raise ValueError("Empty image data received.")

    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img.verify()
    except Exception as e:
        raise ValueError(f"Corrupt or unsupported image format: {str(e)}")

    # Re-open after verify() to load pixel data
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return pil_img

def get_active_preprocessing_function(arch_hint: str = None):
    """Dynamically resolve architecture preprocessing function."""
    if arch_hint is None:
        import os, json
        import config
        manifest_path = os.path.join(config.MODELS_DIR, "model_manifest.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                    arch_hint = manifest.get("architecture", "mobilenetv2")
            except Exception:
                arch_hint = "mobilenetv2"
        else:
            arch_hint = "mobilenetv2"

    arch = (arch_hint or "mobilenetv2").lower()
    if "resnet" in arch:
        return tf.keras.applications.resnet50.preprocess_input
    elif "efficientnet" in arch:
        return tf.keras.applications.efficientnet.preprocess_input
    else:
        return tf.keras.applications.mobilenet_v2.preprocess_input

def preprocess_image_for_model(pil_img: Image.Image, arch_hint: str = None) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Resizes image to 224x224 and applies architecture-specific preprocessing.
    Returns:
        (preprocessed_batch, (orig_width, orig_height))
    """
    orig_dims = pil_img.size # (width, height)
    resized_img = pil_img.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    img_array = np.array(resized_img, dtype=np.float32)

    # Expand batch dimension: (1, 224, 224, 3)
    batch_array = np.expand_dims(img_array, axis=0)

    # Apply architecture-specific preprocess_input
    preprocess_fn = get_active_preprocessing_function(arch_hint)
    preprocessed_batch = preprocess_fn(batch_array)
    return preprocessed_batch, orig_dims

def preprocess_image_from_path(filepath: str) -> Tuple[np.ndarray, Image.Image]:
    """Load image from filesystem path and preprocess for inference."""
    with open(filepath, "rb") as f:
        bytes_data = f.read()
    pil_img = validate_and_load_image(bytes_data)
    preprocessed_batch, _ = preprocess_image_for_model(pil_img)
    return preprocessed_batch, pil_img
