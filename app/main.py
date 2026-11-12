import os
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import config
from app.model_loader import model_loader_instance
from app.inference import inference_service

app = FastAPI(
    title="LeafScan ML Microservice",
    description="TensorFlow/Keras MobileNetV2 + Grad-CAM Explainable Plant Pathology API",
    version="2.0.0"
)

# Enable CORS for Express server and React client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
@app.get("/health")
def health_check():
    """Service health and model readiness check."""
    model_name = "MobileNetV2 (TensorFlow/Keras) + Grad-CAM"
    manifest_path = os.path.join(config.MODELS_DIR, "model_manifest.json")
    if os.path.exists(manifest_path):
        try:
            import json
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
                arch = manifest.get("architecture", "")
                m_name = manifest.get("selected_model_name", "")
                model_name = f"{m_name or arch} (TensorFlow/Keras) + Grad-CAM"
        except Exception:
            pass

    return {
        "status": "healthy",
        "service": "LeafScan ML Engine",
        "model": model_name,
        "model_loaded": model_loader_instance.is_loaded,
        "model_path": model_loader_instance.model_path,
        "load_error": model_loader_instance.load_error,
        "supported_classes_count": len(model_loader_instance.class_names),
        "crops": ["Pepper", "Potato", "Tomato"]
    }

@app.get("/classes")
def get_classes():
    """Retrieve all supported crop disease classes."""
    return {
        "classes": model_loader_instance.class_names,
        "count": len(model_loader_instance.class_names)
    }

@app.get("/pathology/{class_name}")
def get_pathology_info(class_name: str):
    """Retrieve detailed agronomic pathology and management guidance for a given class."""
    if class_name not in config.DISEASE_PATHOLOGY_INFO:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Class '{class_name}' not found in pathology database."
        )
    return config.DISEASE_PATHOLOGY_INFO[class_name]

@app.post("/predict")
async def predict_leaf_disease(
    image: UploadFile = File(...),
    crop_type: Optional[str] = Form(None)
):
    """
    Diagnose plant leaf disease from image upload with Grad-CAM visual explainability.
    """
    if not model_loader_instance.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Trained model is not loaded: {model_loader_instance.load_error or 'Model not found'}"
        )

    try:
        contents = await image.read()
        if not contents or len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty image file received. Please provide a valid leaf image."
            )

        result = inference_service.predict(contents, crop_hint=crop_type)
        return result

    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image format or corrupted file: {str(ve)}"
        )
    except Exception as e:
        print(f"[FastAPI Predict Error] {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=config.SERVICE_HOST, port=config.SERVICE_PORT, reload=False)
