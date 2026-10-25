"""
FastAPI application entry point for SignSync.
Provides WebSocket for real-time video stream processing and serves the web UI.
"""

import os
import sys
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.hand_detector import HandDetector
from src.gesture_classifier import GestureClassifier
from src.utils import decode_base64_image, PredictionSmoother

# Global instances initialized on lifespan startup
hand_detector: HandDetector = None
gesture_classifier: GestureClassifier = None
smoother: PredictionSmoother = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global hand_detector, gesture_classifier, smoother
    print("[SignSync] Initializing MediaPipe Hand Detector & Gesture Classifier...")
    # Sensitive detection threshold for home webcam lighting
    hand_detector = HandDetector(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    gesture_classifier = GestureClassifier(
        model_path=os.path.join(BASE_DIR, "models", "gesture_model.pkl"),
        classes_path=os.path.join(BASE_DIR, "models", "gesture_classes.pkl"),
    )
    smoother = PredictionSmoother(buffer_size=8, min_ratio=0.5)

    if gesture_classifier.is_ready():
        print(f"[SignSync] Model loaded with {len(gesture_classifier.classes)} classes: {gesture_classifier.classes}")
    else:
        print("[SignSync] Note: gesture_model.pkl not found. App will run in detection-only mode until trained.")

    yield

    print("[SignSync] Cleaning up resources...")
    if hand_detector:
        hand_detector.close()


app = FastAPI(title="SignSync", lifespan=lifespan)

# Mount static and template directories
static_dir = os.path.join(BASE_DIR, "static")
templates_dir = os.path.join(BASE_DIR, "templates")

os.makedirs(static_dir, exist_ok=True)
os.makedirs(templates_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)


@app.get("/")
async def index(request: Request):
    """Serve the main application user interface."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "model_ready": gesture_classifier.is_ready() if gesture_classifier else False,
        },
    )


@app.get("/health")
async def health():
    """Health check endpoint indicating model readiness and service status."""
    is_ready = gesture_classifier.is_ready() if gesture_classifier else False
    classes = [str(c) for c in gesture_classifier.classes] if (gesture_classifier and gesture_classifier.classes is not None) else []
    return JSONResponse(
        {
            "status": "ok",
            "model_loaded": is_ready,
            "vocabulary": classes,
        }
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint receiving webcam frames and returning detected landmarks & predictions.
    """
    await websocket.accept()

    conn_smoother = PredictionSmoother(buffer_size=8, min_ratio=0.5)

    try:
        while True:
            text_data = await websocket.receive_text()
            if not text_data:
                continue

            try:
                payload = json.loads(text_data)
            except Exception:
                continue

            base64_frame = payload.get("frame", "")
            if not base64_frame:
                continue

            image = decode_base64_image(base64_frame)
            if image is None:
                await websocket.send_json({"success": False, "error": "Invalid frame"})
                continue

            detection = hand_detector.detect(image)

            if not detection["hand_detected"]:
                conn_smoother.smooth(None)
                await websocket.send_json(
                    {
                        "success": True,
                        "hand_detected": False,
                        "letter": None,
                        "confidence": 0.0,
                        "landmarks": [],
                        "smoothed_letter": None,
                    }
                )
                continue

            pred_letter = None
            pred_conf = 0.0

            if gesture_classifier.is_ready():
                pred = gesture_classifier.predict(detection["landmark_list"])
                pred_letter = pred.get("letter")
                pred_conf = pred.get("confidence", 0.0)

            smoothed = conn_smoother.smooth(pred_letter)

            await websocket.send_json(
                {
                    "success": True,
                    "hand_detected": True,
                    "letter": pred_letter,
                    "confidence": pred_conf,
                    "landmarks": detection["landmarks"],
                    "smoothed_letter": smoothed,
                }
            )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket Error] {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
