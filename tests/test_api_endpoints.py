import requests
import json
import os
import sys

BASE_URL = "http://127.0.0.1:8000"

def test_api():
    print("=================================================================")
    print("    LeafScan ML Service HTTP Network API Integration Test")
    print("=================================================================")
    
    passed = 0
    failed = 0

    def record(name, success, detail=""):
        nonlocal passed, failed
        if success:
            print(f"  [PASS] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name} - {detail}")
            failed += 1

    # 1. GET /health
    print("\n--- 1. Testing Health Endpoints ---")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        record("GET /health status 200", r.status_code == 200, f"Got {r.status_code}")
        data = r.json()
        record("Health response has status: 'healthy'", data.get("status") == "healthy")
        record("Health response confirms model_loaded == True", data.get("model_loaded") is True)
        record("Health response lists 15 classes count", data.get("supported_classes_count") == 15)
        record("Health response lists crops", "Pepper" in data.get("crops", []))
    except Exception as e:
        record("GET /health request", False, str(e))

    # 2. GET /classes
    print("\n--- 2. Testing /classes Endpoint ---")
    try:
        r = requests.get(f"{BASE_URL}/classes", timeout=5)
        record("GET /classes status 200", r.status_code == 200, f"Got {r.status_code}")
        data = r.json()
        record("/classes returns 15 classes", data.get("count") == 15 and len(data.get("classes", [])) == 15)
    except Exception as e:
        record("GET /classes request", False, str(e))

    # 3. GET /pathology
    print("\n--- 3. Testing /pathology Endpoints ---")
    try:
        r = requests.get(f"{BASE_URL}/pathology/Tomato_healthy", timeout=5)
        record("GET /pathology/Tomato_healthy status 200", r.status_code == 200)
        data = r.json()
        record("Pathology contains disease_name and crop", "disease_name" in data and data.get("crop") == "Tomato")
        
        # Test 404 for non-existent class
        r_404 = requests.get(f"{BASE_URL}/pathology/Non_Existent_Crop_Class_999", timeout=5)
        record("GET invalid class returns 404", r_404.status_code == 404, f"Got {r_404.status_code}")
    except Exception as e:
        record("GET /pathology request", False, str(e))

    # 4. POST /predict with actual leaf image
    print("\n--- 4. Testing POST /predict with Valid Test Image ---")
    test_img_path = None
    test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "test"))
    for root, _, files in os.walk(test_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                test_img_path = os.path.join(root, f)
                break
        if test_img_path:
            break

    if test_img_path:
        try:
            with open(test_img_path, "rb") as f:
                files = {"image": (os.path.basename(test_img_path), f, "image/jpeg")}
                data = {"crop_type": "potato"}
                r = requests.post(f"{BASE_URL}/predict", files=files, data=data, timeout=15)
            
            record("POST /predict status 200", r.status_code == 200, f"Got {r.status_code}")
            resp = r.json()
            
            expected_fields = [
                "predicted_class", "crop", "disease_name", "scientific_name",
                "confidence", "severity", "symptoms", "environmental_factors",
                "organic_treatment", "chemical_treatment", "prevention",
                "affected_area_pct", "original_image", "heatmap_image",
                "overlay_image", "top_predictions"
            ]
            all_fields = all(f in resp for f in expected_fields)
            record("Prediction payload contains all 16 required response keys", all_fields)
            record("Confidence score is valid float between 0 and 1", 0.0 <= resp.get("confidence", -1) <= 1.0)
            record("Top 3 predictions list length is 3", len(resp.get("top_predictions", [])) == 3)
            record("Original image is base64 data URL", resp.get("original_image", "").startswith("data:image/jpeg;base64,"))
            record("Heatmap image is base64 data URL", resp.get("heatmap_image", "").startswith("data:image/jpeg;base64,"))
            record("Overlay image is base64 data URL", resp.get("overlay_image", "").startswith("data:image/jpeg;base64,"))
            record("Affected area percentage is numeric", isinstance(resp.get("affected_area_pct"), (int, float)))
        except Exception as e:
            record("POST /predict valid image request", False, str(e))
    else:
        record("Test image found", False, "No test image located")

    # 5. POST /predict Negative & Error Handling Tests
    print("\n--- 5. Testing Negative & Edge Case Error Handling ---")
    try:
        # Empty upload
        r_empty = requests.post(f"{BASE_URL}/predict", files={"image": ("empty.jpg", b"", "image/jpeg")}, timeout=5)
        record("Empty image upload returns HTTP 400", r_empty.status_code == 400, f"Got {r_empty.status_code}")

        # Corrupt file bytes
        r_corrupt = requests.post(
            f"{BASE_URL}/predict",
            files={"image": ("corrupt.jpg", b"THIS_IS_NOT_A_VALID_IMAGE_DATA_12345", "image/jpeg")},
            timeout=5
        )
        record("Corrupt file bytes returns HTTP 400", r_corrupt.status_code == 400, f"Got {r_corrupt.status_code}")

        # Non-image plain text file
        r_txt = requests.post(
            f"{BASE_URL}/predict",
            files={"image": ("document.txt", b"Hello world plant leaf diagnostic", "text/plain")},
            timeout=5
        )
        record("Plain text file rejected with HTTP 400", r_txt.status_code == 400, f"Got {r_txt.status_code}")
    except Exception as e:
        record("Negative tests request", False, str(e))

    print("\n=================================================================")
    print(f" HTTP Network Test Summary: {passed} PASSED | {failed} FAILED")
    print("=================================================================")
    return failed == 0

if __name__ == "__main__":
    success = test_api()
    sys.exit(0 if success else 1)
