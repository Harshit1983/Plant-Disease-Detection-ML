import numpy as np
from typing import Optional, Dict, Any, List
import config
from app.model_loader import model_loader_instance
from app.preprocessing import validate_and_load_image, preprocess_image_for_model
from app.gradcam import GradCAMExplainer

class InferenceService:
    """
    Handles end-to-end inference request lifecycle:
    Validation -> Preprocessing -> Model Prediction -> Crop Filtering -> Grad-CAM XAI -> Diagnostics Report
    """
    def __init__(self):
        self._explainer: Optional[GradCAMExplainer] = None

    def _get_explainer(self) -> GradCAMExplainer:
        model = model_loader_instance.get_model()
        if self._explainer is None or self._explainer.model is not model:
            self._explainer = GradCAMExplainer(model)
        return self._explainer

    def predict(self, image_bytes: bytes, crop_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute prediction pipeline for an uploaded leaf image.
        """
        # 1. Image validation & loading
        pil_img = validate_and_load_image(image_bytes)

        # 2. Consistent preprocessing (Resizing to 224x224 & MobileNetV2 scaling)
        preprocessed_batch, _ = preprocess_image_for_model(pil_img)

        # 3. Model retrieval
        model = model_loader_instance.get_model()
        class_names = model_loader_instance.class_names

        # 4. Forward inference
        predictions = model.predict(preprocessed_batch, verbose=0)[0]
        probs = np.array(predictions, dtype=np.float32)

        # 5. Crop hint filtering (Pepper, Potato, Tomato)
        filtered_indices = list(range(len(class_names)))
        if crop_hint:
            hint = crop_hint.lower().strip()
            if "pepper" in hint:
                matching = [i for i, name in enumerate(class_names) if "pepper" in name.lower()]
                if matching:
                    filtered_indices = matching
            elif "potato" in hint:
                matching = [i for i, name in enumerate(class_names) if "potato" in name.lower()]
                if matching:
                    filtered_indices = matching
            elif "tomato" in hint:
                matching = [i for i, name in enumerate(class_names) if "tomato" in name.lower()]
                if matching:
                    filtered_indices = matching

        # 6. Find best class among candidates
        best_idx = filtered_indices[0]
        max_prob = probs[best_idx]
        for idx in filtered_indices:
            if probs[idx] > max_prob:
                max_prob = probs[idx]
                best_idx = idx

        predicted_class = class_names[best_idx]

        # 7. Generate Grad-CAM explainability artifacts
        explainer = self._get_explainer()
        xai_artifacts = explainer.process_and_visualize(preprocessed_batch, pil_img, best_idx)

        # 8. Top 3 probability distribution
        top_k_indices = np.argsort(probs)[::-1][:3]
        top_predictions: List[Dict[str, Any]] = []
        for tk in top_k_indices:
            cls_name = class_names[tk]
            pathology = config.DISEASE_PATHOLOGY_INFO.get(cls_name, {})
            top_predictions.append({
                "class_name": cls_name,
                "display_name": pathology.get("disease_name", cls_name),
                "crop": pathology.get("crop", "Unknown"),
                "confidence": float(probs[tk])
            })

        # 9. Pathology & Agronomy details
        pathology_details = config.DISEASE_PATHOLOGY_INFO.get(predicted_class, {
            "crop": "Crop Specimen",
            "disease_name": predicted_class.replace("_", " "),
            "scientific_name": "Pathogen diagnostic pending",
            "severity": "Moderate",
            "symptoms": "Leaf tissue abnormality detected.",
            "environmental_factors": "Variable humidity and foliage moisture.",
            "organic_treatment": "Apply bio-fungicide and ensure proper ventilation.",
            "chemical_treatment": "Consult local agricultural extension office.",
            "prevention": "Ensure good sanitation and crop rotation."
        })

        return {
            "predicted_class": predicted_class,
            "crop": pathology_details["crop"],
            "disease_name": pathology_details["disease_name"],
            "scientific_name": pathology_details["scientific_name"],
            "confidence": float(max_prob),
            "severity": pathology_details["severity"],
            "symptoms": pathology_details["symptoms"],
            "environmental_factors": pathology_details.get("environmental_factors", ""),
            "organic_treatment": pathology_details.get("organic_treatment", ""),
            "chemical_treatment": pathology_details.get("chemical_treatment", ""),
            "prevention": pathology_details.get("prevention", ""),
            "affected_area_pct": xai_artifacts["affected_area_pct"],
            "original_image": xai_artifacts["original_image"],
            "heatmap_image": xai_artifacts["heatmap_image"],
            "overlay_image": xai_artifacts["overlay_image"],
            "top_predictions": top_predictions
        }

# Global inference service instance
inference_service = InferenceService()
