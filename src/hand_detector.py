"""
Hand detection and landmark extraction module.
Supports both modern MediaPipe Tasks API and legacy solutions API.
"""

from typing import Dict, Any, Tuple, Optional, List
import os
import base64
import urllib.request
import numpy as np
import cv2
import mediapipe as mp

TASK_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


class HandDetector:
    """
    Extracts 21 3D hand landmarks from images using MediaPipe.
    Configured with sensitive detection thresholds for domestic/webcam environments.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.35,
        min_tracking_confidence: float = 0.35,
        model_path: Optional[str] = None,
    ):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.use_tasks_api = not hasattr(mp, "solutions")

        if self.use_tasks_api:
            # Modern MediaPipe Tasks API
            if model_path is None:
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                model_path = os.path.join(base_dir, "models", "hand_landmarker.task")

            self._ensure_model_exists(model_path)
            self._init_tasks_api(model_path)
        else:
            # Legacy MediaPipe Solutions API
            self._init_legacy_api()

    def _ensure_model_exists(self, model_path: str):
        if not os.path.exists(model_path):
            print(f"[HandDetector] Downloading hand_landmarker.task to {model_path}...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            try:
                urllib.request.urlretrieve(TASK_MODEL_URL, model_path)
                print("[HandDetector] Model download complete.")
            except Exception as e:
                print(f"[HandDetector] Failed to download model: {e}")

    def _init_tasks_api(self, model_path: str):
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            running_mode=RunningMode.IMAGE,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def _init_legacy_api(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_image_mode,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process an OpenCV BGR image and return normalized landmarks.
        """
        if image is None or image.size == 0:
            return {"hand_detected": False, "landmarks": [], "landmark_list": []}

        rgb_image = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if self.use_tasks_api:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            result = self.landmarker.detect(mp_image)

            if not result.hand_landmarks:
                return {"hand_detected": False, "landmarks": [], "landmark_list": []}

            primary_hand = result.hand_landmarks[0]
            landmarks: List[List[float]] = []
            landmark_list: List[float] = []

            for lm in primary_hand:
                landmarks.append([float(lm.x), float(lm.y), float(lm.z)])
                landmark_list.extend([float(lm.x), float(lm.y), float(lm.z)])

            return {
                "hand_detected": True,
                "landmarks": landmarks,
                "landmark_list": landmark_list,
            }
        else:
            results = self.hands.process(rgb_image)
            if not results.multi_hand_landmarks:
                return {"hand_detected": False, "landmarks": [], "landmark_list": []}

            primary_hand = results.multi_hand_landmarks[0]
            landmarks: List[List[float]] = []
            landmark_list: List[float] = []

            for lm in primary_hand.landmark:
                landmarks.append([float(lm.x), float(lm.y), float(lm.z)])
                landmark_list.extend([float(lm.x), float(lm.y), float(lm.z)])

            return {
                "hand_detected": True,
                "landmarks": landmarks,
                "landmark_list": landmark_list,
            }

    def detect_from_base64(self, base64_string: str) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
        """
        Decode a base64 encoded image string and run landmark detection.
        """
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]

        try:
            image_bytes = base64.b64decode(base64_string)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception:
            return {"hand_detected": False, "landmarks": [], "landmark_list": []}, None

        if image is None:
            return {"hand_detected": False, "landmarks": [], "landmark_list": []}, None

        detection = self.detect(image)
        return detection, image

    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, "landmarker") and self.landmarker:
            self.landmarker.close()
        elif hasattr(self, "hands") and self.hands:
            self.hands.close()
