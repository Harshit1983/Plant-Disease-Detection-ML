import io
import base64
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from typing import Tuple, Dict, Any, Optional

class GradCAMExplainer:
    """
    Explainable AI (XAI) using Gradient-weighted Class Activation Mapping (Grad-CAM)
    implemented natively in TensorFlow 2.x and Keras.
    """
    def __init__(self, model: tf.keras.Model):
        self.model = model
        self.target_layer_name = self._find_target_conv_layer()
        self.grad_model = self._build_grad_model()

    def _find_target_conv_layer(self) -> str:
        """Find the final convolutional/activation layer in the model."""
        import os, json
        import config

        # 1. Check if model manifest defines target layer
        manifest_path = os.path.join(config.MODELS_DIR, "model_manifest.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                    target = manifest.get("target_conv_layer")
                    if target:
                        self.model.get_layer(target)
                        return target
            except Exception:
                pass

        # 2. Check architecture-specific candidate layers
        candidate_names = [
            "out_relu",         # MobileNetV2
            "top_activation",   # EfficientNetB0
            "top_conv",         # EfficientNet variant
            "conv5_block3_out", # ResNet50
            "Conv_1",           # MobileNet fallback
            "Conv_1_bn"
        ]
        for name in candidate_names:
            try:
                layer = self.model.get_layer(name)
                return layer.name
            except (ValueError, AttributeError):
                pass

        # Otherwise, find the last Conv2D or 4D layer before GlobalAveragePooling
        for layer in reversed(self.model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                return layer.name
            # If layer has 4D output shape (batch, H, W, C)
            try:
                if len(layer.output.shape) == 4:
                    return layer.name
            except Exception:
                pass

        raise ValueError("Could not find a suitable convolutional layer for Grad-CAM in model.")

    def _build_grad_model(self) -> tf.keras.Model:
        """Construct multi-output model returning target conv activations and final predictions."""
        target_conv_layer = self.model.get_layer(self.target_layer_name)
        return tf.keras.Model(
            inputs=self.model.inputs,
            outputs=[target_conv_layer.output, self.model.output]
        )

    def generate_heatmap(self, input_tensor: np.ndarray, class_index: int) -> np.ndarray:
        """
        Compute Grad-CAM activation heatmap for a given input tensor and target class.
        Returns: 2D numpy array with values in [0, 1].
        """
        tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(tensor)
            # Score for target class
            class_score = predictions[:, class_index]

        # Compute gradients of class score w.r.t. feature map outputs
        grads = tape.gradient(class_score, conv_outputs)
        if grads is None:
            # Fallback if gradient is disconnected
            return np.ones((7, 7), dtype=np.float32) * 0.5

        # Global average pooling of gradients: weights across feature maps
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weighted combination of feature channels
        cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)

        # Apply ReLU to keep features positively contributing to class
        cam = tf.maximum(cam, 0.0)

        cam_np = cam.numpy()
        max_val = np.max(cam_np)
        if max_val > 0:
            cam_np = cam_np / max_val
        else:
            cam_np = np.ones_like(cam_np) * 0.5

        return cam_np

    def process_and_visualize(
        self,
        input_tensor: np.ndarray,
        pil_img: Image.Image,
        class_index: int
    ) -> Dict[str, Any]:
        """
        Generate Grad-CAM heatmap, apply JET colormap, blend with original leaf image,
        and encode to base64 Data URLs.
        """
        orig_width, orig_height = pil_img.size
        img_np = np.array(pil_img)

        # 1. Generate normalized CAM map
        cam_map = self.generate_heatmap(input_tensor, class_index)

        # 2. Resize CAM map to original image resolution
        cam_resized = cv2.resize(cam_map, (orig_width, orig_height))

        # 3. Apply OpenCV JET colormap
        cam_uint8 = np.uint8(255 * cam_resized)
        heatmap_colored = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # 4. Create overlay: 55% original + 45% heatmap
        overlay_np = cv2.addWeighted(img_np, 0.55, heatmap_colored, 0.45, 0)

        # 5. Base64 encoding helper
        def array_to_base64_jpeg(array: np.ndarray) -> str:
            buf = io.BytesIO()
            Image.fromarray(array).save(buf, format="JPEG", quality=90)
            encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded}"

        original_b64 = array_to_base64_jpeg(img_np)
        heatmap_b64 = array_to_base64_jpeg(heatmap_colored)
        overlay_b64 = array_to_base64_jpeg(overlay_np)

        # 6. Affected area percentage (hotspot activation > 0.45)
        affected_area_pct = float(np.mean(cam_map > 0.45) * 100)

        return {
            "original_image": original_b64,
            "heatmap_image": heatmap_b64,
            "overlay_image": overlay_b64,
            "affected_area_pct": round(affected_area_pct, 1)
        }
