"""
Utility functions, landmark normalization, and prediction smoothing for SignSync.
"""

from typing import Optional, List
from collections import deque, Counter
import base64
import numpy as np
import cv2


def normalize_landmarks(landmark_list: List[float]) -> List[float]:
    """
    Normalizes 63-element landmark coordinates (21 joints x [x, y, z]):
    1. Translates all landmarks relative to the wrist (landmark 0 becomes [0, 0, 0]).
    2. Scales by the maximum Euclidean distance from the wrist to any joint.
    This makes classification completely invariant to:
       - Hand position in the camera frame (left, right, top, bottom, center)
       - Distance from the camera (hand size scaling)
    """
    if len(landmark_list) != 63:
        return landmark_list

    pts = np.array(landmark_list, dtype=np.float32).reshape(21, 3)
    # Shift relative to wrist
    rel = pts - pts[0]
    # Max distance across all points
    distances = np.linalg.norm(rel, axis=1)
    scale = float(np.max(distances))
    if scale <= 1e-6:
        scale = 1.0

    normalized = rel / scale
    return normalized.flatten().tolist()


def decode_base64_image(base64_string: str) -> Optional[np.ndarray]:
    """
    Decode a base64 encoded image string (JPEG/PNG) to an OpenCV BGR numpy array.
    """
    if not base64_string:
        return None

    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(base64_string)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return image
    except Exception:
        return None


class PredictionSmoother:
    """
    Smooths real-time gesture predictions across consecutive frames to prevent rapid flickering.
    """

    def __init__(self, buffer_size: int = 8, min_ratio: float = 0.5):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.min_ratio = min_ratio

    def smooth(self, letter: Optional[str]) -> Optional[str]:
        """
        Add the latest raw prediction to the rolling buffer and return the smoothed prediction.
        """
        self.buffer.append(letter)

        valid_predictions = [p for p in self.buffer if p is not None]

        if len(valid_predictions) < min(3, self.buffer_size // 2):
            return None

        counts = Counter(valid_predictions)
        most_common_letter, count = counts.most_common(1)[0]

        if (count / len(valid_predictions)) >= self.min_ratio:
            return most_common_letter

        return None

    def reset(self):
        """Clear the prediction history."""
        self.buffer.clear()
