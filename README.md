# LeafScan ML Service

MobileNetV2 Deep Learning Model with Grad-CAM Explainability for Pepper, Potato, and Tomato Leaf Pathology.

## Dataset Structure

Place your crop leaf dataset images inside `data/` according to the following directory layout:

```text
data/
├── controlled/
│   ├── train/
│   │   ├── Pepper__bell___Bacterial_spot/
│   │   ├── Pepper__bell___healthy/
│   │   ├── Potato___Early_blight/
│   │   ├── Potato___Late_blight/
│   │   ├── Potato___healthy/
│   │   ├── Tomato_Bacterial_spot/
│   │   ├── Tomato_Early_blight/
│   │   ├── Tomato_Late_blight/
│   │   ├── Tomato_Leaf_Mold/
│   │   ├── Tomato_Septoria_leaf_spot/
│   │   ├── Tomato_Spider_mites_Two_spotted_spider_mite/
│   │   ├── Tomato__Target_Spot/
│   │   ├── Tomato__Tomato_YellowLeaf__Curl_Virus/
│   │   ├── Tomato__Tomato_mosaic_virus/
│   │   └── Tomato_healthy/
│   ├── val/
│   │   └── [same class subfolders]
│   └── test/
│       └── [same class subfolders]
└── real_world/
    └── test/
        └── [same class subfolders]
```

Note: If you have dataset images in `c:\Users\karan\OneDrive\Desktop\Plant Disease Detection\PlantVillage`, you can run `python train.py` which will automatically detect images either in `data/controlled/train` or fallback to `../PlantVillage`!

## Setup & Running FastAPI ML Server

1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Model Training (Optional)**:
   ```bash
   python train.py --epochs 10 --batch-size 32
   ```
   *Note: If no custom trained `models/best_model.pth` exists yet, the service uses MobileNetV2 with pre-trained PyTorch weights + Grad-CAM layer feature extraction automatically.*

3. **Start FastAPI Service**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Endpoints

- `GET /health` : Service health status check
- `POST /predict` : Upload image file (`image`), optional `crop_type`. Returns prediction, confidence, Grad-CAM heatmap base64 overlay, and disease treatment advice.
- `GET /classes` : Returns all supported crop disease classes.
