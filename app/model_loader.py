import os
import json
import threading
import tensorflow as tf
from typing import Optional, List, Dict, Tuple
import config

class ModelLoader:
    """
    Thread-safe Singleton Model Loader for TensorFlow/Keras Plant Disease Classifier.
    Loads and caches the model and class mapping at service startup.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelLoader, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self.model: Optional[tf.keras.Model] = None
        self.class_indices: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.class_names: List[str] = list(config.CLASS_NAMES)
        self.is_loaded: bool = False
        self.load_error: Optional[str] = None
        self.model_path: str = config.MODEL_PATH

        self.load_class_indices()
        self.load_model()
        self._initialized = True

    def load_class_indices(self):
        """Load class index mapping from JSON file or fall back to config."""
        if os.path.exists(config.CLASS_INDICES_PATH):
            try:
                with open(config.CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
                    self.class_indices = json.load(f)
                # Invert mapping: index -> class_name
                self.idx_to_class = {v: k for k, v in self.class_indices.items()}
                # Ensure class_names list matches indexed order
                self.class_names = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]
                print(f"[ModelLoader] Loaded class mapping ({len(self.class_names)} classes) from {config.CLASS_INDICES_PATH}")
                return
            except Exception as e:
                print(f"[ModelLoader] Warning reading {config.CLASS_INDICES_PATH}: {e}")

        # Fallback to config.CLASS_NAMES
        self.class_names = list(config.CLASS_NAMES)
        self.class_indices = {name: i for i, name in enumerate(self.class_names)}
        self.idx_to_class = {i: name for i, name in enumerate(self.class_names)}

    def load_model(self, custom_path: Optional[str] = None):
        """Loads Keras model checkpoint, prioritizing production champion model in model_manifest.json."""
        target_path = custom_path
        if not target_path:
            manifest_path = os.path.join(config.MODELS_DIR, "model_manifest.json")
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                        cand = manifest.get("model_file") or manifest.get("production_active_path")
                        if cand and os.path.exists(cand):
                            target_path = cand
                except Exception:
                    pass
        target_path = target_path or self.model_path
        if not os.path.exists(target_path):
            self.model = None
            self.is_loaded = False
            self.load_error = f"Model artifact not found at {target_path}. Please train a model with 'python scripts/train.py'."
            print(f"[ModelLoader] {self.load_error}")
            return False

        try:
            print(f"[ModelLoader] Loading trained Keras model from {target_path}...")
            # Load with compile=False for fast, clean inference
            self.model = tf.keras.models.load_model(target_path, compile=False)
            self.model_path = target_path
            self.is_loaded = True
            self.load_error = None
            print(f"[ModelLoader] Successfully loaded model from {target_path}")
            return True
        except Exception as e:
            self.model = None
            self.is_loaded = False
            self.load_error = f"Failed to load Keras model from {target_path}: {str(e)}"
            print(f"[ModelLoader] Error: {self.load_error}")
            return False

    def reload(self, model_path: Optional[str] = None) -> bool:
        """Reload the model from disk (useful after completing a training job)."""
        self.load_class_indices()
        return self.load_model(model_path)

    def get_model(self) -> tf.keras.Model:
        """Return cached model or raise RuntimeError if unavailable."""
        if not self.is_loaded or self.model is None:
            raise RuntimeError(self.load_error or "Trained model is not available.")
        return self.model

# Singleton instance
model_loader_instance = ModelLoader()
