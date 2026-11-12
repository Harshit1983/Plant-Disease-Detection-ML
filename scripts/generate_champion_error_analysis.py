import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# Ensure ml-service root on sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_ml_service_dir = os.path.abspath(os.path.join(_current_dir, ".."))
if _ml_service_dir not in sys.path:
    sys.path.insert(0, _ml_service_dir)

import config

def analyze_errors():
    print("="*75)
    print("Phase 2: Deep Diagnostic Error Analysis for Champion MobileNetV2")
    print("="*75)

    with open(config.CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    reports_adv_dir = os.path.join(config.REPORTS_DIR, "advanced_experiments")
    test_true_path = os.path.join(reports_adv_dir, "test_true.npy")
    test_probs_path = os.path.join(reports_adv_dir, "expA_test_probs.npy")

    if not os.path.exists(test_true_path) or not os.path.exists(test_probs_path):
        raise FileNotFoundError(f"Champion test predictions not found in {reports_adv_dir}")

    y_true = np.load(test_true_path)
    y_probs = np.load(test_probs_path)
    y_pred = np.argmax(y_probs, axis=-1)

    total_samples = len(y_true)
    correct_samples = int(np.sum(y_true == y_pred))
    error_samples = total_samples - correct_samples
    overall_acc = correct_samples / total_samples

    # Metrics
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    # 15x15 Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    # Detailed per-class breakdowns
    per_class_analysis = {}
    for i, name in enumerate(class_names):
        support = int(np.sum(y_true == i))
        tp = int(cm[i, i])
        fn = support - tp
        fp = int(np.sum(cm[:, i]) - tp)
        tn = int(total_samples - tp - fp - fn)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Find which classes false negatives were predicted as
        fn_breakdown = []
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                fn_breakdown.append({
                    "confused_as": class_names[j],
                    "count": int(cm[i, j]),
                    "pct_of_class_errors": round(float(cm[i, j] / max(1, fn) * 100), 2)
                })
        fn_breakdown.sort(key=lambda x: x["count"], reverse=True)

        # Find which classes false positives originated from
        fp_breakdown = []
        for j in range(len(class_names)):
            if i != j and cm[j, i] > 0:
                fp_breakdown.append({
                    "intruding_ground_truth": class_names[j],
                    "count": int(cm[j, i]),
                    "pct_of_false_alarms": round(float(cm[j, i] / max(1, fp) * 100), 2)
                })
        fp_breakdown.sort(key=lambda x: x["count"], reverse=True)

        per_class_analysis[name] = {
            "class_index": i,
            "support": support,
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "specificity": round(float(specificity), 4),
            "fn_breakdown": fn_breakdown,
            "fp_breakdown": fp_breakdown
        }

    # Top confused class pairs across whole dataset
    confusion_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append({
                    "ground_truth": class_names[i],
                    "predicted_as": class_names[j],
                    "count": int(cm[i, j]),
                    "pct_of_all_errors": round(float(cm[i, j] / error_samples * 100), 2)
                })
    confusion_pairs.sort(key=lambda x: x["count"], reverse=True)

    # Focus deeply on the 4 difficult categories specified by the user
    focal_classes = [
        "Tomato_Early_blight",
        "Tomato__Target_Spot",
        "Tomato_Septoria_leaf_spot",
        "Potato___Late_blight"
    ]
    focal_analysis = {cls: per_class_analysis[cls] for cls in focal_classes if cls in per_class_analysis}

    # Summary payload
    summary = {
        "model_name": "MobileNetV2 (Champion Checkpoint)",
        "test_samples": total_samples,
        "correct_samples": correct_samples,
        "error_samples": error_samples,
        "accuracy": round(float(overall_acc), 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_wt), 4),
        "focal_classes_overview": {
            cls: {
                "support": per_class_analysis[cls]["support"],
                "recall": per_class_analysis[cls]["recall"],
                "precision": per_class_analysis[cls]["precision"],
                "f1_score": per_class_analysis[cls]["f1_score"],
                "false_negatives": per_class_analysis[cls]["false_negatives"],
                "top_confusion": per_class_analysis[cls]["fn_breakdown"][:2] if per_class_analysis[cls]["fn_breakdown"] else []
            }
            for cls in focal_classes
        },
        "top_10_pairwise_confusions": confusion_pairs[:10],
        "per_class_breakdown": per_class_analysis
    }

    # Save detailed JSON
    out_json_path = os.path.join(config.REPORTS_DIR, "detailed_champion_error_analysis.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[Done] Saved detailed JSON report to: {out_json_path}")

    # Generate Confusion Matrix Figure with Annotations
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Champion MobileNetV2 - Test Error Matrix (N={total_samples}, Acc={overall_acc*100:.2f}%)", fontsize=14, fontweight="bold", pad=12)
    plt.colorbar(fraction=0.046, pad=0.04)

    short_names = [
        c.replace("Tomato___", "Tom_")
         .replace("Tomato_", "Tom_")
         .replace("Pepper__bell___", "Pep_")
         .replace("Potato___", "Pot_")
        for c in class_names
    ]
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, short_names, rotation=45, ha="right", fontsize=9)
    plt.yticks(tick_marks, short_names, fontsize=9)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            # Highlight non-diagonal errors in bold red/dark
            if i != j and val > 5:
                plt.text(j, i, format(val, "d"), ha="center", va="center", color="red", fontweight="bold", fontsize=8)
            else:
                plt.text(j, i, format(val, "d"), ha="center", va="center", color=color, fontsize=8)

    plt.ylabel("Ground Truth Label", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Class Label", fontsize=11, fontweight="bold")
    plt.tight_layout()

    out_cm_path = os.path.join(config.REPORTS_DIR, "detailed_champion_confusion_matrix.png")
    plt.savefig(out_cm_path, dpi=200)
    plt.close()
    print(f"[Done] Saved annotated confusion matrix plot to: {out_cm_path}")

    # Generate Markdown Summary Report
    md_content = f"""# Detailed Champion Model Diagnostic Error Analysis (Phase 2)

**Evaluated Checkpoint**: `MobileNetV2 (Champion Checkpoint)`  
**Evaluation Set**: Independent Untouched Test Split (`dataset/test`)  
**Total Test Samples**: {total_samples}  
**Overall Test Accuracy**: {overall_acc*100:.2f}% ({correct_samples}/{total_samples} correct, {error_samples} misclassifications)  
**Macro Metrics**: Precision: {p_macro*100:.2f}% | Recall: {r_macro*100:.2f}% | F1-Score: {f1_macro*100:.2f}%  

---

## 1. Targeted Difficult Categories Deep-Dive

| Category | Test Support | Correct (TP) | False Neg (FN) | False Pos (FP) | Precision | Recall | F1-Score | Primary Confused Class |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **`Tomato_Early_blight`** | {per_class_analysis['Tomato_Early_blight']['support']} | {per_class_analysis['Tomato_Early_blight']['true_positives']} | {per_class_analysis['Tomato_Early_blight']['false_negatives']} | {per_class_analysis['Tomato_Early_blight']['false_positives']} | {per_class_analysis['Tomato_Early_blight']['precision']*100:.1f}% | {per_class_analysis['Tomato_Early_blight']['recall']*100:.1f}% | {per_class_analysis['Tomato_Early_blight']['f1_score']*100:.1f}% | {per_class_analysis['Tomato_Early_blight']['fn_breakdown'][0]['confused_as']} ({per_class_analysis['Tomato_Early_blight']['fn_breakdown'][0]['count']} images) |
| **`Tomato__Target_Spot`** | {per_class_analysis['Tomato__Target_Spot']['support']} | {per_class_analysis['Tomato__Target_Spot']['true_positives']} | {per_class_analysis['Tomato__Target_Spot']['false_negatives']} | {per_class_analysis['Tomato__Target_Spot']['false_positives']} | {per_class_analysis['Tomato__Target_Spot']['precision']*100:.1f}% | {per_class_analysis['Tomato__Target_Spot']['recall']*100:.1f}% | {per_class_analysis['Tomato__Target_Spot']['f1_score']*100:.1f}% | {per_class_analysis['Tomato__Target_Spot']['fn_breakdown'][0]['confused_as']} ({per_class_analysis['Tomato__Target_Spot']['fn_breakdown'][0]['count']} images) |
| **`Tomato_Septoria_leaf_spot`** | {per_class_analysis['Tomato_Septoria_leaf_spot']['support']} | {per_class_analysis['Tomato_Septoria_leaf_spot']['true_positives']} | {per_class_analysis['Tomato_Septoria_leaf_spot']['false_negatives']} | {per_class_analysis['Tomato_Septoria_leaf_spot']['false_positives']} | {per_class_analysis['Tomato_Septoria_leaf_spot']['precision']*100:.1f}% | {per_class_analysis['Tomato_Septoria_leaf_spot']['recall']*100:.1f}% | {per_class_analysis['Tomato_Septoria_leaf_spot']['f1_score']*100:.1f}% | {per_class_analysis['Tomato_Septoria_leaf_spot']['fn_breakdown'][0]['confused_as']} ({per_class_analysis['Tomato_Septoria_leaf_spot']['fn_breakdown'][0]['count']} images) |
| **`Potato___Late_blight`** | {per_class_analysis['Potato___Late_blight']['support']} | {per_class_analysis['Potato___Late_blight']['true_positives']} | {per_class_analysis['Potato___Late_blight']['false_negatives']} | {per_class_analysis['Potato___Late_blight']['false_positives']} | {per_class_analysis['Potato___Late_blight']['precision']*100:.1f}% | {per_class_analysis['Potato___Late_blight']['recall']*100:.1f}% | {per_class_analysis['Potato___Late_blight']['f1_score']*100:.1f}% | {per_class_analysis['Potato___Late_blight']['fn_breakdown'][0]['confused_as']} ({per_class_analysis['Potato___Late_blight']['fn_breakdown'][0]['count']} images) |

---

## 2. Top 10 Major Confusion Pairs (Accounting for {sum(p['pct_of_all_errors'] for p in confusion_pairs[:10]):.1f}% of All Test Errors)

| Rank | Ground Truth | Misclassified As | Misclassified Samples | % of All Test Errors | Biological & Visual Cause |
|:---:|:---|:---|:---:|:---:|:---|
"""
        # Annotate top confusions with biological causes
    bio_causes = {
        ("Tomato_Bacterial_spot", "Tomato_Septoria_leaf_spot"): "Both present as small dark circular lesions with chlorotic halos; nearly identical at small spatial receptive fields.",
        ("Tomato__Target_Spot", "Tomato_Spider_mites_Two_spotted_spider_mite"): "Both exhibit dispersed yellowish stippling and localized chlorosis across the lamina.",
        ("Tomato_Late_blight", "Potato___Late_blight"): "Identical pathogen (Phytophthora infestans) creating identical water-soaked necrotic foliar lesions across both solanaceous species.",
        ("Tomato_Early_blight", "Tomato_Septoria_leaf_spot"): "Early lesions of Alternaria solani resemble Septoria spots before characteristic concentric rings fully develop.",
        ("Tomato__Target_Spot", "Tomato_Septoria_leaf_spot"): "Overlapping lesion boundaries and necrotic centres under varying lighting.",
        ("Tomato_Late_blight", "Tomato_Leaf_Mold"): "Diffuse velvet-like mold lesions mistaken for sporulating late blight margin.",
        ("Tomato__Target_Spot", "Tomato_healthy"): "Faint early-stage spot infections mistaken for specular reflection on healthy leaves.",
        ("Tomato_Early_blight", "Tomato_Bacterial_spot"): "Spot-like early lesions mistaken for bacterial flecks.",
        ("Tomato__Target_Spot", "Tomato_Bacterial_spot"): "Similar dark necrotic spots on tomato leaf surfaces.",
        ("Potato___Early_blight", "Potato___Late_blight"): "Necrotic patches on potato leaves differentiated primarily by lesion edge crispness.",
    }

    for idx, p in enumerate(confusion_pairs[:10], start=1):
        pair = (p["ground_truth"], p["predicted_as"])
        cause = bio_causes.get(pair, "Visual similarity in lesion shape and leaf texture under field lighting.")
        md_content += f"| {idx} | `{p['ground_truth']}` | `{p['predicted_as']}` | **{p['count']}** | {p['pct_of_all_errors']}% | {cause} |\n"

    md_content += "\n---\n\n## 3. Complete Per-Class Diagnostic Profile\n\n"
    md_content += "| Class Name | Support | Precision | Recall | F1-Score | Specificity |\n"
    md_content += "|:---|:---:|:---:|:---:|:---:|:---:|\n"
    for name, m in per_class_analysis.items():
        md_content += f"| `{name}` | {m['support']} | {m['precision']*100:.2f}% | {m['recall']*100:.2f}% | **{m['f1_score']*100:.2f}%** | {m['specificity']*100:.2f}% |\n"

    out_md_path = os.path.join(config.REPORTS_DIR, "detailed_champion_error_analysis.md")
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[Done] Saved Markdown analysis to: {out_md_path}")

if __name__ == "__main__":
    analyze_errors()

