import os
import sys
import json
import base64
import numpy as np
from PIL import Image
import io

# Add ml-service root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

def run_all_tests():
    print("=================================================================")
    print("      LeafScan ML Microservice Full Verification Suite")
    print("=================================================================")
    passed = 0
    failed = 0

    def assert_test(condition, test_name, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  [PASS] {test_name}")
            passed += 1
        else:
            print(f"  [FAIL] {test_name} - {detail}")
            failed += 1

    # Test 1: Dataset Directory Integrity & Splits
    print("\n--- Test 1: Dataset Split Integrity ---")
    splits = ["train", "valid", "test"]
    all_splits_exist = all(os.path.exists(os.path.join(config.DATASET_DIR, s)) for s in splits)
    assert_test(all_splits_exist, "All split folders exist (train, valid, test)")

    train_classes = os.listdir(config.TRAIN_DIR) if os.path.exists(config.TRAIN_DIR) else []
    assert_test(len(train_classes) == 15, f"Train set has 15 classes (found {len(train_classes)})")

    # Test 2: Check for Overlapping Images across Splits (Zero Data Leakage)
    print("\n--- Test 2: Zero Data Leakage Across Splits ---")
    train_files = set()
    for root, _, files in os.walk(config.TRAIN_DIR):
        for f in files:
            train_files.add(f)

    valid_files = set()
    for root, _, files in os.walk(config.VALID_DIR):
        for f in files:
            valid_files.add(f)

    test_files = set()
    for root, _, files in os.walk(config.TEST_DIR):
        for f in files:
            test_files.add(f)

    overlap_train_valid = train_files.intersection(valid_files)
    overlap_train_test = train_files.intersection(test_files)
    overlap_valid_test = valid_files.intersection(test_files)

    assert_test(len(overlap_train_valid) == 0, f"No overlap between Train and Valid (overlap: {len(overlap_train_valid)})")
    assert_test(len(overlap_train_test) == 0, f"No overlap between Train and Test (overlap: {len(overlap_train_test)})")
    assert_test(len(overlap_valid_test) == 0, f"No overlap between Valid and Test (overlap: {len(overlap_valid_test)})")

    # Test 3: Class to Index Mapping
    print("\n--- Test 3: Class-to-Index Mapping ---")
    mapping_exists = os.path.exists(config.CLASS_INDICES_PATH)
    assert_test(mapping_exists, "class_indices.json exists")
    if mapping_exists:
        with open(config.CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
            c2i = json.load(f)
        assert_test(len(c2i) == 15, f"15 classes mapped in class_indices.json (found {len(c2i)})")
        assert_test("Tomato_healthy" in c2i, "'Tomato_healthy' present in class mapping")

    # Test 4: Preprocessing Verification
    print("\n--- Test 4: Image Preprocessing ---")
    from app.preprocessing import validate_and_load_image, preprocess_image_for_model
    # Generate test RGB synthetic leaf image
    test_img = Image.new("RGB", (300, 300), color=(34, 139, 34))
    buf = io.BytesIO()
    test_img.save(buf, format="JPEG")
    valid_bytes = buf.getvalue()

    loaded_pil = validate_and_load_image(valid_bytes)
    assert_test(loaded_pil.mode == "RGB", "Image successfully loaded in RGB mode")

    prep_batch, dims = preprocess_image_for_model(loaded_pil)
    assert_test(prep_batch.shape == (1, 224, 224, 3), f"Preprocessed shape is (1, 224, 224, 3) (got {prep_batch.shape})")
    assert_test(prep_batch.min() >= -1.0 and prep_batch.max() <= 1.0, "MobileNetV2 normalized values in [-1, 1]")

    # Test 5: Error Handling for Invalid / Corrupt Images
    print("\n--- Test 5: Error Handling for Corrupt Images ---")
    corrupt_bytes = b"NOT_AN_IMAGE_RANDOM_CORRUPT_BYTES_XYZ"
    caught_ve = False
    try:
        validate_and_load_image(corrupt_bytes)
    except ValueError:
        caught_ve = True
    assert_test(caught_ve, "Corrupt image bytes correctly caught with ValueError")

    # Test 6: Model Loader & Inference Verification (if model exists)
    print("\n--- Test 6: Model & Inference Verification ---")
    from app.model_loader import model_loader_instance
    print(f"Model loaded status: {model_loader_instance.is_loaded}")

    if model_loader_instance.is_loaded:
        assert_test(model_loader_instance.model is not None, "Keras model object successfully loaded")
        from app.inference import inference_service

        # Pick an actual image from test set if available
        sample_path = None
        for root, _, files in os.walk(config.TEST_DIR):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    sample_path = os.path.join(root, f)
                    break
            if sample_path:
                break

        if sample_path:
            with open(sample_path, "rb") as f:
                img_bytes = f.read()
            print(f"Testing inference on test image: {os.path.basename(sample_path)}")
        else:
            img_bytes = valid_bytes

        result = inference_service.predict(img_bytes)

        # Check API response compatibility
        expected_keys = [
            "predicted_class", "crop", "disease_name", "scientific_name",
            "confidence", "severity", "symptoms", "environmental_factors",
            "organic_treatment", "chemical_treatment", "prevention",
            "affected_area_pct", "original_image", "heatmap_image",
            "overlay_image", "top_predictions"
        ]
        all_keys_present = all(k in result for k in expected_keys)
        assert_test(all_keys_present, "Inference response contains all 16 required backend contract fields")
        assert_test(0.0 <= result["confidence"] <= 1.0, f"Confidence score is in [0, 1] ({result['confidence']})")
        assert_test(len(result["top_predictions"]) == 3, "Top 3 predictions list is populated")
        assert_test(result["overlay_image"].startswith("data:image/jpeg;base64,"), "Overlay image formatted as base64 Data URL")
        assert_test(result["heatmap_image"].startswith("data:image/jpeg;base64,"), "Heatmap image formatted as base64 Data URL")
    else:
        print("  [INFO] Model is not yet trained or saved. Skipping inference assertion until training completes.")

    # Test 7: FastAPI Health Check Endpoint
    print("\n--- Test 7: FastAPI App Endpoints ---")
    from app.main import app
    from fastapi.testclient import TestClient
    client = TestClient(app)

    res_health = client.get("/health")
    assert_test(res_health.status_code == 200, f"/health endpoint returns HTTP 200 (got {res_health.status_code})")
    health_data = res_health.json()
    assert_test(health_data["status"] == "healthy", "Health status is 'healthy'")
    assert_test(health_data["crops"] == ["Pepper", "Potato", "Tomato"], "Crops listed: Pepper, Potato, Tomato")

    res_classes = client.get("/classes")
    assert_test(res_classes.status_code == 200, f"/classes endpoint returns HTTP 200 (got {res_classes.status_code})")
    assert_test(res_classes.json()["count"] == 15, "15 classes returned by /classes")

    res_pathology = client.get("/pathology/Tomato_healthy")
    assert_test(res_pathology.status_code == 200, "/pathology/Tomato_healthy returns HTTP 200")
    assert_test(res_pathology.json()["crop"] == "Tomato", "Pathology crop matches 'Tomato'")

    # Test bad image on /predict
    res_bad = client.post("/predict", files={"image": ("bad.jpg", b"", "image/jpeg")})
    assert_test(res_bad.status_code in [400, 503], f"Empty image rejected with HTTP 400 or 503 (got {res_bad.status_code})")

    print("\n=================================================================")
    print(f" Verification Summary: {passed} PASSED | {failed} FAILED")
    print("=================================================================")
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
