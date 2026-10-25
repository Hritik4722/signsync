"""
Process Kaggle ASL Alphabet images to extract MediaPipe 3D hand landmarks and save to CSV.
"""

import os
import sys
import argparse
from typing import List
import cv2
import pandas as pd

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.hand_detector import HandDetector


def process_dataset(
    dataset_dir: str,
    output_csv: str,
    max_samples_per_class: int = 500,
):
    """
    Scans class folders in dataset_dir, extracts MediaPipe landmarks from images,
    and saves the features to a CSV file.
    """
    if not os.path.exists(dataset_dir):
        print(f"[Error] Dataset directory not found: {dataset_dir}")
        print("Please download the Kaggle ASL Alphabet dataset and extract it to:")
        print(f"  {dataset_dir}")
        print("Or pass custom path via: python data/process_dataset.py --dataset-dir <path>")
        return False

    print(f"Initializing MediaPipe HandDetector in static image mode...")
    detector = HandDetector(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

    subdirs = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])

    # Filter to standard 26 letters if present (or keep all valid folders)
    valid_classes = [c for c in subdirs if c.isalpha() and len(c) == 1]
    if not valid_classes:
        valid_classes = subdirs  # Fallback if names are different

    print(f"Found {len(valid_classes)} gesture classes to process: {', '.join(valid_classes)}")

    records: List[List] = []
    total_processed = 0

    for class_name in valid_classes:
        class_folder = os.path.join(dataset_dir, class_name)
        image_files = [
            f for f in os.listdir(class_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        # Limit to max_samples_per_class for speed and balance
        selected_files = image_files[:max_samples_per_class]
        detected_count = 0

        for img_name in selected_files:
            img_path = os.path.join(class_folder, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue

            detection = detector.detect(image)
            if detection["hand_detected"]:
                row = detection["landmark_list"] + [class_name.upper()]
                records.append(row)
                detected_count += 1

        print(f"Class [{class_name.upper()}]: detected hand landmarks in {detected_count}/{len(selected_files)} images.")
        total_processed += detected_count

    detector.close()

    if not records:
        print("[Error] No hand landmarks were extracted. Please check dataset images.")
        return False

    # Build column names: x0, y0, z0, ..., x20, y20, z20, label
    columns = []
    for i in range(21):
        columns.extend([f"x{i}", f"y{i}", f"z{i}"])
    columns.append("label")

    df = pd.DataFrame(records, columns=columns)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"\n==========================================")
    print(f"Successfully processed {total_processed} samples across {len(valid_classes)} classes.")
    print(f"Saved dataset features to: {output_csv}")
    print(f"==========================================")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract MediaPipe landmarks from ASL images.")
    parser.add_argument(
        "--dataset-dir",
        default=os.path.join("data", "asl_alphabet", "asl_alphabet_train"),
        help="Path to folder containing class subdirectories (A, B, C, ...)",
    )
    parser.add_argument(
        "--output-csv",
        default=os.path.join("data", "landmarks.csv"),
        help="Output CSV path for extracted landmarks",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Maximum images to process per letter class (default: 500)",
    )

    args = parser.parse_args()
    process_dataset(args.dataset_dir, args.output_csv, args.max_samples)
