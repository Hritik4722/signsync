<div align="center">

# 🤟 SignSync
### AI-Based Real-Time Sign Language Recognition System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange.svg)](https://google.github.io/mediapipe/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*An AI-powered computer vision web application that translates American Sign Language (ASL) hand gestures into text in real time using webcam video streaming.*

</div>

---

## 📖 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [How It Works](#-how-it-works)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [Model Training & Customization](#-model-training--customization)
  - [Training on the Full ASL Dataset](#option-1-kaggle-asl-alphabet-dataset)
  - [Recording Custom Gestures](#option-2-record-custom-signs-via-webcam)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)

---

## 🌟 Overview

Sign language is the primary mode of communication for deaf and hard-of-hearing communities, yet most hearing individuals cannot understand sign language. **SignSync** bridges this communication divide by providing an intuitive, browser-based real-time translation tool.

Unlike traditional heavy deep learning systems requiring expensive GPUs, SignSync extracts **21 3D hand landmarks** using Google MediaPipe and uses **wrist-relative scale-invariant normalization** with a **Random Forest classifier**. This achieves **>99% accuracy** while running smoothly at high frame rates on standard laptops (even Intel Core i3 with integrated graphics).

---

## ✨ Key Features

- **⚡ Real-Time 3D Landmark Tracking**: 21 3D hand joint coordinates tracked via MediaPipe with sub-millisecond CPU inference.
- **🎯 Translation & Scale Invariance**: Custom wrist-centered normalization ensures gestures are recognized regardless of where your hand is positioned in the camera frame or how close/far you are from the camera.
- **🤲 Left & Right Hand Support**: Symmetrical data augmentation allows signing with either hand seamlessly.
- **🛡️ Anti-Flicker Consensus Stabilization**: Rolling buffer smoothing algorithm eliminates frame jitter and stabilizes predictions.
- **💻 Interactive Web Interface**:
  - Live webcam feed with mirroring and real-time skeleton overlay.
  - Large sign prediction display with confidence indicators.
  - Automatic sentence construction with hold cooldowns.
  - Quick action controls: **Space**, **Backspace**, **Clear**, **Copy to Clipboard**, and **Text-to-Speech (TTS)**.

---

## 🔄 How It Works

```
[Webcam Stream]
      │
      ▼  (WebSocket JPEG Stream @ ~14 FPS)
[FastAPI Server]
      │
      ▼
[MediaPipe HandLandmarker] ──▶ Extracts 21 3D Coordinates (x, y, z)
      │
      ▼
[Wrist Normalization] ──▶ Shifts wrist to (0,0,0) & normalizes by hand span
      │
      ▼
[Random Forest Model] ──▶ Predicts ASL Letter + Confidence Score
      │
      ▼
[Prediction Smoother] ──▶ Consensus filtering across recent frames
      │
      ▼  (JSON Response)
[Web UI Display] ──▶ Updates Skeleton, Prediction Badge & Text Buffer
```

---

## 📁 Project Structure

```
signsync/
├── app.py                      # FastAPI web server & WebSocket streaming hub
├── requirements.txt            # Python package dependencies
├── README.md                   # Project documentation
├── agent.md                    # Architecture decisions & design notes
│
├── src/                        # Core application modules
│   ├── __init__.py
│   ├── hand_detector.py        # MediaPipe wrapper for 3D landmark extraction
│   ├── gesture_classifier.py   # Landmark normalization & model inference
│   └── utils.py                # Base64 decoding, normalization & smoother
│
├── data/                       # Dataset processing scripts
│   ├── process_dataset.py      # Extracts landmarks from image folders to CSV
│   ├── collect_data.py         # Live webcam recorder to record custom signs
│   ├── generate_sample_data.py # Synthetic sample generator for pipeline testing
│   └── landmarks.csv           # Pre-extracted 26-letter dataset (63 coords + label)
│
├── models/                     # Trained models & weights
│   ├── train.py                # Random Forest training with data augmentation
│   ├── gesture_model.pkl       # Serialized trained model (A–Z)
│   ├── gesture_classes.pkl     # Class label dictionary (26 alphabet signs)
│   └── hand_landmarker.task    # Google MediaPipe HandLandmarker binary model
│
├── static/                     # Frontend static assets
│   ├── css/
│   │   └── style.css           # Modern Navy/Teal responsive styling
│   └── js/
│       └── app.js              # WebSocket client, video feed & canvas overlay
│
└── templates/
    └── index.html              # Main single-page web interface
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+** (Tested on Python 3.11)
- **Webcam** (built-in or USB)
- Modern web browser (Chrome, Edge, Firefox, or Safari)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Hritik4722/signsync.git
   cd signsync
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the FastAPI server:**
   ```bash
   python app.py
   ```

2. **Open your browser and navigate to:**
   ```
   http://localhost:8000
   ```

3. **Allow camera access** when prompted and click **"Start Camera"**.
4. Hold your hand in front of the camera and form sign gestures (e.g., **A**, **B**, **C**, **L**, **V**, **Y**). The system will detect your hand joints and translate the gesture in real time!

---

## 📊 Model Training & Customization

The repository comes **pre-trained** with all 26 letters (A–Z) in `models/gesture_model.pkl` using the pre-extracted `data/landmarks.csv`. If you want to retrain or customize:

### Retraining the Model

To retrain the model with wrist-normalization and left/right hand augmentation:
```bash
python models/train.py
```
*(Takes under 15 seconds on a modern CPU!)*

### Option 1: Kaggle ASL Alphabet Dataset

To extract fresh landmarks from the raw Kaggle dataset:
1. Download the [ASL Alphabet Dataset from Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).
2. Extract the folders into `data/asl_alphabet/asl_alphabet_train/` (folders named `A/`, `B/`, ..., `Z/`).
3. Run landmark extraction:
   ```bash
   python data/process_dataset.py --max-samples 250
   ```
4. Train the model:
   ```bash
   python models/train.py
   ```

### Option 2: Record Custom Signs via Webcam

Want to add new words or custom signs? Use the interactive live collector:
```bash
# Capture 150 frames of gesture 'HELLO'
python data/collect_data.py --letter HELLO --samples 150

# Retrain the model to include the new gesture
python models/train.py
```

---

## ⚙️ Configuration

Key operational settings can be tuned in these files:

| Setting | File | Default | Description |
|---|---|---|---|
| Frame Interval | `static/js/app.js` | `70ms (~14 FPS)` | Frame capture interval sent to the backend |
| JPEG Quality | `static/js/app.js` | `0.82` | Frame compression quality |
| Capture Resolution | `static/js/app.js` | `640x480` | Resolution sent to MediaPipe |
| Detection Confidence | `src/hand_detector.py` | `0.35` | Minimum palm detection threshold |
| Tracking Confidence | `src/hand_detector.py` | `0.35` | Landmark tracking confidence threshold |
| Consensus Buffer | `app.py` | `8 frames` | Rolling consensus length to eliminate flickering |
| Letter Hold Delay | `static/js/app.js` | `1200ms` | Time before repeating an identical held sign |

---

## ❓ Troubleshooting

- **Camera not starting:** Ensure your browser has permission to access your webcam (`chrome://settings/content/camera`). Make sure no other application (like Zoom or Teams) is using the camera.
- **Model not loaded:** Ensure `models/gesture_model.pkl` exists. If missing, simply run `python models/train.py`.
- **Low accuracy in dark rooms:** MediaPipe benefits from adequate lighting. Ensure your hand has reasonable contrast against the background.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
