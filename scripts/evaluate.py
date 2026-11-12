import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Add ml-service root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

def evaluate_pipeline(
    model_path: str = None,
    test_dir: str = None,
    batch_size: int = 16,
    max_samples: int = 0,
    reports_dir: str = None
):
    model_path = model_path or config.MODEL_PATH
    test_dir = test_dir or config.TEST_DIR
    reports_dir = reports_dir or config.REPORTS_DIR

    print("=================================================================")
    print("      LeafScan Model Evaluation on Untouched Test Split")
    print("=================================================================")
    print(f"Model Path:     {model_path}")
    print(f"Test Directory: {test_dir}")
    print(f"Reports Output: {reports_dir}")
    print(f"Batch Size:     {batch_size}")
    if max_samples > 0:
        print(f"Max Samples:    {max_samples} (fast test evaluation mode)")
    print("-----------------------------------------------------------------")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at: {model_path}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test split directory not found at: {test_dir}")

    # Load class index mapping
    with open(config.CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # Load test dataset
    if max_samples > 0:
        per_class = max(1, max_samples // len(class_names))
        file_paths = []
        labels_one_hot = []
        for cls_idx, cls in enumerate(class_names):
            cls_dir = os.path.join(test_dir, cls)
            if os.path.exists(cls_dir):
                files = [
                    os.path.join(cls_dir, f) for f in sorted(os.listdir(cls_dir))
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))
                ]
                for fp in files[:per_class]:
                    file_paths.append(fp)
                    one_hot = np.zeros(len(class_names), dtype=np.float32)
                    one_hot[cls_idx] = 1.0
                    labels_one_hot.append(one_hot)

        def parse_test_image(path, label):
            b = tf.io.read_file(path)
            img = tf.image.decode_jpeg(b, channels=3)
            img = tf.image.resize(img, config.IMG_SIZE)
            return img, label

        raw_test_ds = tf.data.Dataset.from_tensor_slices((file_paths, labels_one_hot))
        raw_test_ds = raw_test_ds.map(parse_test_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
        print(f"Sampled {len(file_paths)} test images ({per_class} per class across all {len(class_names)} classes).")
    else:
        raw_test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            labels="inferred",
            label_mode="categorical",
            batch_size=batch_size,
            image_size=config.IMG_SIZE,
            shuffle=False
        )

    # Preprocessing for test data (strictly NO augmentation)
    def preprocess_test(images, labels):
        images = tf.keras.applications.mobilenet_v2.preprocess_input(images)
        return images, labels

    test_ds = raw_test_ds.map(preprocess_test, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Load trained model
    print(f"\nLoading trained model from {model_path}...")
    model = tf.keras.models.load_model(model_path, compile=False)

    # Collect true labels and predictions
    print("Running inference across test split...")
    y_true = []
    y_pred_probs = []
    test_images_raw = [] # Keep small buffer for misclassified sample plotting

    for batch_images, batch_labels in raw_test_ds:
        # Preprocess
        proc_images = tf.keras.applications.mobilenet_v2.preprocess_input(batch_images)
        probs = model.predict(proc_images, verbose=0)
        
        for i in range(len(batch_labels)):
            y_true.append(np.argmax(batch_labels[i].numpy()))
            y_pred_probs.append(probs[i])
            if len(test_images_raw) < 500:
                test_images_raw.append(batch_images[i].numpy().astype(np.uint8))

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 1. Overall Metrics
    test_acc = accuracy_score(y_true, y_pred)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    print("\n-----------------------------------------------------------------")
    print(f" Test Accuracy:       {test_acc * 100:.2f}%")
    print(f" Macro Precision:     {macro_precision:.4f}")
    print(f" Macro Recall:        {macro_recall:.4f}")
    print(f" Macro F1-Score:      {macro_f1:.4f}")
    print(f" Weighted F1-Score:   {weighted_f1:.4f}")
    print("-----------------------------------------------------------------")

    # 2. Classification Report
    labels_all = list(range(len(class_names)))
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels_all,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    report_str = classification_report(
        y_true,
        y_pred,
        labels=labels_all,
        target_names=class_names,
        zero_division=0
    )
    print("\nClassification Report:\n")
    print(report_str)

    os.makedirs(reports_dir, exist_ok=True)

    # Save classification reports
    report_json_path = os.path.join(reports_dir, "classification_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    report_txt_path = os.path.join(reports_dir, "classification_report.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("=========================================================================\n")
        f.write("             LEAFSCAN CNN TEST SET EVALUATION REPORT\n")
        f.write("=========================================================================\n")
        f.write(f"Model Checkpoint:     {model_path}\n")
        f.write(f"Total Test Samples:   {len(y_true)}\n")
        f.write(f"Test Accuracy:        {test_acc * 100:.2f}%\n")
        f.write(f"Macro Precision:      {macro_precision:.4f}\n")
        f.write(f"Macro Recall:         {macro_recall:.4f}\n")
        f.write(f"Macro F1-Score:       {macro_f1:.4f}\n")
        f.write(f"Weighted F1-Score:    {weighted_f1:.4f}\n")
        f.write("-------------------------------------------------------------------------\n\n")
        f.write(report_str)
        f.write("\n=========================================================================\n")

    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels_all)
    plt.figure(figsize=(14, 12))
    short_labels = [c.replace("Tomato_", "Tom_").replace("Pepper__bell___", "Pep_") for c in class_names]

    if HAS_SEABORN:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=short_labels,
            yticklabels=short_labels
        )
    else:
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar()
        tick_marks = np.arange(len(short_labels))
        plt.xticks(tick_marks, short_labels, rotation=45, ha="right", fontsize=9)
        plt.yticks(tick_marks, short_labels, fontsize=9)
        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], "d"),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
    plt.title(f"Confusion Matrix (Test Accuracy: {test_acc*100:.1f}%)", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    cm_path = os.path.join(reports_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"[Saved] Confusion matrix plot saved to: {cm_path}")

    # 4. Misclassified Samples Analysis
    misclassified_indices = np.where(y_true != y_pred)[0]
    print(f"Total misclassified test samples: {len(misclassified_indices)} out of {len(y_true)}")

    if len(misclassified_indices) > 0 and len(test_images_raw) > 0:
        num_samples_to_plot = min(8, len(misclassified_indices))
        cols = 4
        rows = int(np.ceil(num_samples_to_plot / cols))

        plt.figure(figsize=(16, rows * 4))
        for i, idx in enumerate(misclassified_indices[:num_samples_to_plot]):
            if idx < len(test_images_raw):
                img = test_images_raw[idx]
                true_name = class_names[y_true[idx]]
                pred_name = class_names[y_pred[idx]]
                conf = float(y_pred_probs[idx][y_pred[idx]])

                plt.subplot(rows, cols, i + 1)
                plt.imshow(img)
                plt.title(
                    f"True: {true_name[:18]}...\nPred: {pred_name[:18]}...\nConf: {conf*100:.1f}%",
                    fontsize=9,
                    color="#DC2626"
                )
                plt.axis("off")

        plt.tight_layout()
        misclass_path = os.path.join(reports_dir, "misclassified_samples.png")
        plt.savefig(misclass_path, dpi=200)
        plt.close()
        print(f"[Saved] Misclassified samples visualization saved to: {misclass_path}")

    summary_metrics = {
        "test_samples": int(len(y_true)),
        "accuracy": float(test_acc),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "misclassified_count": int(len(misclassified_indices))
    }
    with open(os.path.join(reports_dir, "evaluation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    print("\nEvaluation completed successfully!")
    return summary_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LeafScan Model on Test Set")
    parser.add_argument("--model-path", type=str, default=config.MODEL_PATH, help="Path to .keras model")
    parser.add_argument("--test-dir", type=str, default=config.TEST_DIR, help="Path to test split")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--max-samples", type=int, default=0, help="Max test samples to evaluate (0 for full set)")
    args = parser.parse_args()

    evaluate_pipeline(
        model_path=args.model_path,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
