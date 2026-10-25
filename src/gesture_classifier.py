"""
Model inference wrapper for classifying hand landmark sequences into ASL letters.
Uses wrist-relative scale-invariant normalization for high accuracy across camera positions.
"""

from typing import Dict, Any, List, Optional
import os
import numpy as np
import joblib

from src.utils import normalize_landmarks


class GestureClassifier:
    """
    Classifies 63 flattened 3D hand landmarks into sign language letters.
    """

    def __init__(
        self,
        model_path: str = "models/gesture_model.pkl",
        classes_path: str = "models/gesture_classes.pkl",
    ):
        self.model_path = model_path
        self.classes_path = classes_path
        self.model = None
        self.classes = None
        self.is_loaded = False

        self.load_model()

    def load_model(self) -> bool:
        """
        Attempts to load the trained model and class labels from disk.
        """
        if not os.path.exists(self.model_path):
            self.is_loaded = False
            return False

        try:
            self.model = joblib.load(self.model_path)
            if os.path.exists(self.classes_path):
                self.classes = joblib.load(self.classes_path)
            elif hasattr(self.model, "classes_"):
                self.classes = self.model.classes_

            self.is_loaded = True
            return True
        except Exception as e:
            print(f"[GestureClassifier] Error loading model: {e}")
            self.is_loaded = False
            return False

    def is_ready(self) -> bool:
        """Return True if model is loaded and ready for predictions."""
        return self.is_loaded and self.model is not None

    def predict(self, landmark_list: List[float]) -> Dict[str, Any]:
        """
        Predict the sign gesture corresponding to a 63-element landmark vector.
        Applies wrist-centered scale normalization.
        """
        if not self.is_ready():
            return {
                "letter": None,
                "confidence": 0.0,
                "error": "Model not loaded. Train a model first via models/train.py",
            }

        if len(landmark_list) != 63:
            return {
                "letter": None,
                "confidence": 0.0,
                "error": f"Expected 63 landmark coordinates, got {len(landmark_list)}",
            }

        try:
            # Normalize coordinates relative to wrist and scaled by hand size
            norm_landmarks = normalize_landmarks(landmark_list)
            features = np.array(norm_landmarks, dtype=np.float32).reshape(1, -1)

            pred = self.model.predict(features)[0]

            confidence = 1.0
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(features)[0]
                confidence = float(np.max(probabilities))

            return {
                "letter": str(pred),
                "confidence": round(confidence, 3),
            }
        except Exception as e:
            return {
                "letter": None,
                "confidence": 0.0,
                "error": f"Inference error: {str(e)}",
            }
