import os
import sys
import argparse
import base64
from PIL import Image
import io

# Add ml-service root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from app.inference import inference_service
from app.model_loader import model_loader_instance

def run_prediction(image_path: str, crop_hint: str = None, output_overlay: str = None):
    print("=================================================================")
    print("      LeafScan Standalone Command-Line Inference")
    print("=================================================================")
    print(f"Target Image:  {image_path}")
    print(f"Crop Hint:     {crop_hint or 'None'}")
    print("-----------------------------------------------------------------")

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)

    if not model_loader_instance.is_loaded:
        print(f"Error: Model not loaded! {model_loader_instance.load_error}")
        sys.exit(1)

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    try:
        result = inference_service.predict(image_bytes, crop_hint=crop_hint)
    except Exception as e:
        print(f"Prediction failed: {e}")
        sys.exit(1)

    print("\n----------------- DIAGNOSTIC REPORT -----------------")
    print(f"Crop Specimen:          {result['crop']}")
    print(f"Pathology Diagnosis:    {result['disease_name']} ({result['scientific_name']})")
    print(f"Class Key:              {result['predicted_class']}")
    print(f"Confidence:             {result['confidence'] * 100:.2f}%")
    print(f"Severity:               {result['severity']}")
    print(f"Affected Leaf Surface:  {result['affected_area_pct']}%")
    print(f"\nSymptoms:\n  {result['symptoms']}")
    print(f"\nOrganic Intervention:\n  {result['organic_treatment']}")
    print(f"\nChemical Regimen:\n  {result['chemical_treatment']}")
    print(f"\nCultural Prevention:\n  {result['prevention']}")

    print("\nTop 3 Predictions:")
    for i, p in enumerate(result["top_predictions"], 1):
        print(f"  {i}. {p['display_name']} ({p['crop']}) — {p['confidence'] * 100:.2f}%")

    if output_overlay:
        overlay_b64 = result["overlay_image"]
        # Strip header if present
        if "base64," in overlay_b64:
            overlay_b64 = overlay_b64.split("base64,")[1]
        img_bytes = base64.b64decode(overlay_b64)
        with open(output_overlay, "wb") as f:
            f.write(img_bytes)
        print(f"\n[Saved] Grad-CAM overlay heatmap saved to: {output_overlay}")

    print("=================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run standalone prediction on a leaf image")
    parser.add_argument("--image", type=str, required=True, help="Path to leaf image file")
    parser.add_argument("--crop-hint", type=str, default=None, help="Crop filter (pepper, potato, tomato)")
    parser.add_argument("--output-overlay", type=str, default=None, help="Optional output path to save Grad-CAM overlay")
    args = parser.parse_args()

    run_prediction(args.image, args.crop_hint, args.output_overlay)
