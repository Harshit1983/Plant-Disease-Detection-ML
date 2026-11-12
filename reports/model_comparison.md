# Plant Disease Detection - Multi-Model Architecture Comparison

Evaluated on untouched held-out test split (C:\Users\karan\OneDrive\Desktop\plant ml\ml-service\dataset\test).

| Architecture | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Latency (ms/img) | Model Size (MB) |
|:-------------|:-------------:|:---------------:|:------------:|:--------------:|:----------------:|:---------------:|
| **MobileNetV2** | 60.00% | 66.38% | 60.00% | 58.27% | 160.52 ms | 16.82 MB |
| **EfficientNetB0** | 10.67% | 9.70% | 10.67% | 6.27% | 224.66 ms | 23.9 MB |
| **Ensemble (3-Model Soft Average)** | 60.00% | 66.48% | 60.00% | 58.38% | 385.18 ms | 40.72 MB |
