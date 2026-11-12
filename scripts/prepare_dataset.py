import os
import sys
import shutil
import hashlib
import json
import random
import argparse
from pathlib import Path
from PIL import Image

# Ensure ml-service root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def calculate_file_hash(filepath: str) -> str:
    """Calculate MD5 hash of file to detect duplicate images."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def is_valid_image(filepath: str) -> bool:
    """Validate that the image can be opened and parsed without error."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        # Verify again by actually loading pixel data
        with Image.open(filepath) as img:
            img.convert("RGB")
        return True
    except Exception:
        return False

def prepare_dataset(
    source_dir: str = None,
    output_dir: str = None,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    max_samples_per_class: int = 0
):
    source_dir = source_dir or config.RAW_DATASET_DIR
    output_dir = output_dir or config.DATASET_DIR

    print(f"============================================================")
    print(f" LeafScan Dataset Preparation & Stratified 70/15/15 Split")
    print(f"============================================================")
    print(f"Source Directory: {source_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Ratios: Train={train_ratio:.0%}, Valid={valid_ratio:.0%}, Test={test_ratio:.0%}")
    print(f"Random Seed: {seed}")
    if max_samples_per_class > 0:
        print(f"Max samples per class limit: {max_samples_per_class}")
    print(f"------------------------------------------------------------")

    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source dataset directory not found at: {source_dir}")

    # Discover class directories
    source_classes = sorted([
        d for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d)) and not d.startswith(".")
    ])

    if not source_classes:
        raise ValueError(f"No class subfolders found inside {source_dir}")

    print(f"Found {len(source_classes)} class directories:")
    for c in source_classes:
        print(f"  - {c}")

    random.seed(seed)

    # Initialize report structure
    report = {
        "source_directory": source_dir,
        "output_directory": output_dir,
        "random_seed": seed,
        "ratios": {
            "train": train_ratio,
            "valid": valid_ratio,
            "test": test_ratio
        },
        "max_samples_per_class_limit": max_samples_per_class,
        "classes": {},
        "summary": {
            "total_source_images": 0,
            "total_valid_unique_images": 0,
            "total_corrupt_skipped": 0,
            "total_duplicate_skipped": 0,
            "train_count": 0,
            "valid_count": 0,
            "test_count": 0
        }
    }

    # Hash set to prevent exact duplicate images across entire dataset
    seen_hashes = {}
    class_images_clean = {}

    for cls in source_classes:
        cls_path = os.path.join(source_dir, cls)
        all_files = sorted(os.listdir(cls_path))
        
        valid_images = []
        corrupt_count = 0
        duplicate_count = 0

        for f in all_files:
            ext = os.path.splitext(f)[1].lower()
            if ext not in VALID_EXTENSIONS:
                continue

            report["summary"]["total_source_images"] += 1
            f_path = os.path.join(cls_path, f)

            # Check image integrity
            if not is_valid_image(f_path):
                corrupt_count += 1
                continue

            # Check duplicate hash
            file_hash = calculate_file_hash(f_path)
            if file_hash in seen_hashes:
                duplicate_count += 1
                continue

            seen_hashes[file_hash] = f_path
            valid_images.append((f, f_path))

        report["summary"]["total_corrupt_skipped"] += corrupt_count
        report["summary"]["total_duplicate_skipped"] += duplicate_count

        # Shuffle deterministically
        random.shuffle(valid_images)

        # Apply max_samples_per_class if requested
        if max_samples_per_class > 0 and len(valid_images) > max_samples_per_class:
            valid_images = valid_images[:max_samples_per_class]

        class_images_clean[cls] = valid_images
        report["summary"]["total_valid_unique_images"] += len(valid_images)

    # Create target directories: train, valid, test
    train_dir = os.path.join(output_dir, "train")
    valid_dir = os.path.join(output_dir, "valid")
    test_dir = os.path.join(output_dir, "test")

    for split_dir in [train_dir, valid_dir, test_dir]:
        os.makedirs(split_dir, exist_ok=True)
        for cls in source_classes:
            os.makedirs(os.path.join(split_dir, cls), exist_ok=True)

    # Class-to-index mapping (ordered alphabetically for consistency)
    class_to_idx = {cls: idx for idx, cls in enumerate(source_classes)}

    # Perform stratified split and copy files
    print("\nSplitting and copying images...")
    for cls, images in class_images_clean.items():
        n = len(images)
        n_train = int(round(train_ratio * n))
        n_valid = int(round(valid_ratio * n))
        n_test = n - n_train - n_valid

        train_imgs = images[:n_train]
        valid_imgs = images[n_train:n_train + n_valid]
        test_imgs = images[n_train + n_valid:]

        for fname, fpath in train_imgs:
            dst = os.path.join(train_dir, cls, fname)
            shutil.copy2(fpath, dst)

        for fname, fpath in valid_imgs:
            dst = os.path.join(valid_dir, cls, fname)
            shutil.copy2(fpath, dst)

        for fname, fpath in test_imgs:
            dst = os.path.join(test_dir, cls, fname)
            shutil.copy2(fpath, dst)

        report["classes"][cls] = {
            "class_index": class_to_idx[cls],
            "total": n,
            "train": len(train_imgs),
            "valid": len(valid_imgs),
            "test": len(test_imgs)
        }

        report["summary"]["train_count"] += len(train_imgs)
        report["summary"]["valid_count"] += len(valid_imgs)
        report["summary"]["test_count"] += len(test_imgs)

        print(f"  [{cls}] Total: {n} -> Train: {len(train_imgs)}, Valid: {len(valid_imgs)}, Test: {len(test_imgs)}")

    # Save class_indices.json
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    with open(config.CLASS_INDICES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"\n[Saved] Class index mapping written to: {config.CLASS_INDICES_PATH}")

    # Save split reports
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    with open(config.SPLIT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[Saved] Detailed JSON split report saved to: {config.SPLIT_REPORT_PATH}")

    # Generate readable text report
    text_report_path = os.path.join(config.REPORTS_DIR, "dataset_split_report.txt")
    with open(text_report_path, "w", encoding="utf-8") as f:
        f.write("========================================================================\n")
        f.write("           PLANT DISEASE DETECTION DATASET SPLIT REPORT\n")
        f.write("========================================================================\n")
        f.write(f"Source Directory:      {source_dir}\n")
        f.write(f"Target Directory:      {output_dir}\n")
        f.write(f"Random Seed:           {seed}\n")
        f.write(f"Split Ratios:          Train: {train_ratio:.0%} | Valid: {valid_ratio:.0%} | Test: {test_ratio:.0%}\n")
        f.write(f"Total Source Images:   {report['summary']['total_source_images']}\n")
        f.write(f"Corrupt Skipped:       {report['summary']['total_corrupt_skipped']}\n")
        f.write(f"Duplicates Skipped:    {report['summary']['total_duplicate_skipped']}\n")
        f.write(f"Total Clean Images:    {report['summary']['total_valid_unique_images']}\n")
        f.write(f"Train Set Images:      {report['summary']['train_count']}\n")
        f.write(f"Valid Set Images:      {report['summary']['valid_count']}\n")
        f.write(f"Test Set Images:       {report['summary']['test_count']}\n")
        f.write("------------------------------------------------------------------------\n")
        f.write(f"{'Class Name':<45} | {'Idx':<4} | {'Total':<6} | {'Train':<6} | {'Valid':<6} | {'Test':<6}\n")
        f.write("------------------------------------------------------------------------\n")
        for cls, d in report["classes"].items():
            f.write(f"{cls:<45} | {d['class_index']:<4} | {d['total']:<6} | {d['train']:<6} | {d['valid']:<6} | {d['test']:<6}\n")
        f.write("========================================================================\n")

    print(f"[Saved] Text report summary saved to: {text_report_path}")
    print("\nDataset preparation completed successfully!")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stratified dataset splitter for Plant Disease Detection")
    parser.add_argument("--source", type=str, default=None, help="Path to raw PlantVillage dataset")
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: ml-service/dataset)")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train split ratio (default: 0.70)")
    parser.add_argument("--valid-ratio", type=float, default=0.15, help="Validation split ratio (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--max-samples-per-class", type=int, default=0, help="Max clean samples per class (0 for all)")
    args = parser.parse_args()

    prepare_dataset(
        source_dir=args.source,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        max_samples_per_class=args.max_samples_per_class
    )
