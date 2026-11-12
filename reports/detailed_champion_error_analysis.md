# Detailed Champion Model Diagnostic Error Analysis (Phase 2)

**Evaluated Checkpoint**: `MobileNetV2 (Champion Checkpoint)`  
**Evaluation Set**: Independent Untouched Test Split (`dataset/test`)  
**Total Test Samples**: 3091  
**Overall Test Accuracy**: 86.28% (2667/3091 correct, 424 misclassifications)  
**Macro Metrics**: Precision: 84.29% | Recall: 86.69% | F1-Score: 85.16%  

---

## 1. Targeted Difficult Categories Deep-Dive

| Category | Test Support | Correct (TP) | False Neg (FN) | False Pos (FP) | Precision | Recall | F1-Score | Primary Confused Class |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **`Tomato_Early_blight`** | 150 | 92 | 58 | 46 | 66.7% | 61.3% | 63.9% | Tomato_Late_blight (22 images) |
| **`Tomato__Target_Spot`** | 210 | 158 | 52 | 67 | 70.2% | 75.2% | 72.6% | Tomato_Spider_mites_Two_spotted_spider_mite (24 images) |
| **`Tomato_Septoria_leaf_spot`** | 265 | 197 | 68 | 32 | 86.0% | 74.3% | 79.8% | Tomato_Leaf_Mold (16 images) |
| **`Potato___Late_blight`** | 150 | 124 | 26 | 7 | 94.7% | 82.7% | 88.3% | Tomato_Late_blight (11 images) |

---

## 2. Top 10 Major Confusion Pairs (Accounting for 36.1% of All Test Errors)

| Rank | Ground Truth | Misclassified As | Misclassified Samples | % of All Test Errors | Biological & Visual Cause |
|:---:|:---|:---|:---:|:---:|:---|
| 1 | `Tomato__Target_Spot` | `Tomato_Spider_mites_Two_spotted_spider_mite` | **24** | 5.66% | Both exhibit dispersed yellowish stippling and localized chlorosis across the lamina. |
| 2 | `Tomato_Early_blight` | `Tomato_Late_blight` | **22** | 5.19% | Visual similarity in lesion shape and leaf texture under field lighting. |
| 3 | `Tomato_Bacterial_spot` | `Tomato__Target_Spot` | **16** | 3.77% | Visual similarity in lesion shape and leaf texture under field lighting. |
| 4 | `Tomato_Septoria_leaf_spot` | `Tomato_Leaf_Mold` | **16** | 3.77% | Visual similarity in lesion shape and leaf texture under field lighting. |
| 5 | `Tomato_Spider_mites_Two_spotted_spider_mite` | `Tomato__Target_Spot` | **15** | 3.54% | Visual similarity in lesion shape and leaf texture under field lighting. |
| 6 | `Tomato_Septoria_leaf_spot` | `Tomato_Early_blight` | **14** | 3.3% | Visual similarity in lesion shape and leaf texture under field lighting. |
| 7 | `Tomato_Bacterial_spot` | `Tomato_Septoria_leaf_spot` | **13** | 3.07% | Both present as small dark circular lesions with chlorotic halos; nearly identical at small spatial receptive fields. |
| 8 | `Potato___Late_blight` | `Tomato_Late_blight` | **11** | 2.59% | Visual similarity in lesion shape and leaf texture under field lighting. |
| 9 | `Tomato_Late_blight` | `Tomato_Leaf_Mold` | **11** | 2.59% | Diffuse velvet-like mold lesions mistaken for sporulating late blight margin. |
| 10 | `Tomato_Septoria_leaf_spot` | `Tomato_Spider_mites_Two_spotted_spider_mite` | **11** | 2.59% | Visual similarity in lesion shape and leaf texture under field lighting. |

---

## 3. Complete Per-Class Diagnostic Profile

| Class Name | Support | Precision | Recall | F1-Score | Specificity |
|:---|:---:|:---:|:---:|:---:|:---:|
| `Pepper__bell___Bacterial_spot` | 149 | 89.81% | 94.63% | **92.16%** | 99.46% |
| `Pepper__bell___healthy` | 221 | 98.10% | 93.67% | **95.83%** | 99.86% |
| `Potato___Early_blight` | 150 | 96.58% | 94.00% | **95.27%** | 99.83% |
| `Potato___Late_blight` | 150 | 94.66% | 82.67% | **88.26%** | 99.76% |
| `Potato___healthy` | 23 | 74.19% | 100.00% | **85.19%** | 99.74% |
| `Tomato_Bacterial_spot` | 319 | 90.24% | 84.01% | **87.01%** | 98.95% |
| `Tomato_Early_blight` | 150 | 66.67% | 61.33% | **63.89%** | 98.44% |
| `Tomato_Late_blight` | 285 | 83.45% | 83.16% | **83.30%** | 98.33% |
| `Tomato_Leaf_Mold` | 143 | 76.25% | 85.31% | **80.53%** | 98.71% |
| `Tomato_Septoria_leaf_spot` | 265 | 86.03% | 74.34% | **79.76%** | 98.87% |
| `Tomato_Spider_mites_Two_spotted_spider_mite` | 252 | 77.35% | 88.10% | **82.37%** | 97.71% |
| `Tomato__Target_Spot` | 210 | 70.22% | 75.24% | **72.64%** | 97.67% |
| `Tomato__Tomato_YellowLeaf__Curl_Virus` | 481 | 96.64% | 95.63% | **96.13%** | 99.39% |
| `Tomato__Tomato_mosaic_virus` | 56 | 74.65% | 94.64% | **83.46%** | 99.41% |
| `Tomato_healthy` | 237 | 89.52% | 93.67% | **91.55%** | 99.09% |
