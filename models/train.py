"""
Model training script for SignSync.
Trains a Random Forest classifier on wrist-normalized, scale-invariant 3D hand landmarks
with left/right hand symmetry augmentation.
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import normalize_landmarks


def train_model(
    csv_path: str = "data/landmarks.csv",
    model_output: str = "models/gesture_model.pkl",
    classes_output: str = "models/gesture_classes.pkl",
    n_estimators: int = 150,
):
    if not os.path.exists(csv_path):
        print(f"[Error] Dataset file not found: {csv_path}")
        print("Please run data/process_dataset.py first.")
        return False

    print(f"Loading dataset from: {csv_path}...")
    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        print("[Error] CSV must contain a 'label' column.")
        return False

    feature_cols = [c for c in df.columns if c != "label"]
    if len(feature_cols) != 63:
        print(f"[Warning] Expected 63 landmark coordinates (x0..z20), found {len(feature_cols)}.")

    raw_X = df[feature_cols].values
    y = df["label"].astype(str).values

    classes = np.unique(y)
    print(f"Raw samples loaded: {len(raw_X)}")
    print(f"Classes ({len(classes)}): {', '.join(classes)}")

    if len(classes) < 2:
        print("[Error] Need at least 2 distinct classes to train a classifier.")
        return False

    # 1. Normalize all samples (wrist-centered + scale invariant)
    print("Applying wrist-centered scale-invariant normalization...")
    X_norm = np.array([normalize_landmarks(row.tolist()) for row in raw_X], dtype=np.float32)

    # 2. Add left/right hand augmentation (flip relative x coordinates)
    # Relative x coords are at indices 0, 3, 6, 9, ... 60
    print("Applying left/right hand symmetry augmentation...")
    X_flipped = X_norm.copy()
    X_flipped[:, 0::3] = -X_flipped[:, 0::3]

    X_augmented = np.vstack([X_norm, X_flipped])
    y_augmented = np.concatenate([y, y])

    print(f"Total training samples after left/right augmentation: {len(X_augmented)}")

    min_class_count = min(np.sum(y_augmented == c) for c in classes)
    stratify_option = y_augmented if min_class_count >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X_augmented, y_augmented, test_size=0.2, random_state=42, stratify=stratify_option
    )

    print(f"Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")
    print(f"Training Random Forest classifier (n_estimators={n_estimators})...")

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n==========================================")
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print("==========================================\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    os.makedirs(os.path.dirname(os.path.abspath(model_output)), exist_ok=True)
    joblib.dump(clf, model_output)
    print(f"Model saved to: {model_output}")

    joblib.dump(clf.classes_, classes_output)
    print(f"Classes saved to: {classes_output}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Random Forest model on hand landmarks.")
    parser.add_argument("--csv", default="data/landmarks.csv", help="Path to landmark CSV")
    parser.add_argument("--model-out", default="models/gesture_model.pkl", help="Output path for model")
    parser.add_argument("--classes-out", default="models/gesture_classes.pkl", help="Output path for class labels")
    parser.add_argument("--n-estimators", type=int, default=150, help="Number of trees (default: 150)")

    args = parser.parse_args()
    train_model(args.csv, args.model_out, args.classes_out, args.n_estimators)
