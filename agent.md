# SignSync — Agent Context Document

> **Last Updated:** 2026-10-25
> **Status:** Planning Phase (Pre-Development)

---

## 1. Project Overview

**SignSync** is an AI-based sign language recognition system that captures hand gestures via a webcam and converts them into text in real time. It is a local web application built as a college/research project prototype.

### What SignSync Does
1. User makes ASL (American Sign Language) hand signs in front of their webcam
2. System detects the hand and extracts 21 landmark points using MediaPipe
3. A trained Random Forest classifier predicts which ASL letter the gesture represents
4. The predicted letter/word is displayed on the web UI in real time
5. Letters accumulate to form words/sentences

---

## 2. Key Decisions Made

| Decision | Choice | Reason |
|----------|--------|--------|
| Application type | **Local web app** (localhost) | Good for demos/submissions, runs in browser |
| Backend | **FastAPI** | User preference, modern async Python framework |
| Frontend | **HTML / CSS / JS** (vanilla) | User preference, no framework overhead |
| Hand detection | **MediaPipe Hands** | Pre-trained by Google, 21 landmarks, runs on CPU, no training needed |
| Classification | **scikit-learn Random Forest** | Fast training (~minutes), >95% accuracy on landmarks, no GPU needed |
| Dataset | **Kaggle ASL Alphabet** (87K images) | Existing dataset, user downloads manually |
| Communication | **WebSocket** | Real-time bidirectional, low latency for live video |
| Language | **Python 3.11** | Available on user's machine |

---

## 3. Tech Stack

### Backend
- **Python 3.11** — runtime
- **FastAPI** — web framework (async, WebSocket support)
- **Uvicorn** — ASGI server to run FastAPI
- **Jinja2** — HTML template rendering

### AI / ML
- **MediaPipe Hands (v0.10+)** — pre-trained hand landmark detection (21 3D points per hand)
- **scikit-learn** — Random Forest classifier for gesture classification
- **OpenCV (cv2)** — image decoding/processing
- **NumPy** — numerical operations on landmark arrays

### Frontend
- **HTML5** — page structure
- **CSS3** — styling (custom, no framework)
- **Vanilla JavaScript** — webcam capture (`getUserMedia`), WebSocket client, canvas drawing

---

## 4. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      BROWSER (Frontend)                  │
│                                                          │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐ │
│  │   Webcam      │   │   Canvas     │   │  Text Panel  │ │
│  │ (getUserMedia)│   │ (landmarks)  │   │ (predictions)│ │
│  └──────┬───────┘   └──────▲───────┘   └──────▲───────┘ │
│         │                  │                   │         │
│         │ base64 frame     │ landmarks         │ letter  │
│         ▼                  │                   │         │
│  ┌─────────────────────────┴───────────────────┘       │ │
│  │              WebSocket Client (JS)                   │ │
│  └──────────────────────┬───────────────────────────────┘ │
└─────────────────────────┼────────────────────────────────┘
                          │ ws://localhost:8000/ws
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Backend                        │
│                                                          │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐ │
│  │  WebSocket    │   │  MediaPipe   │   │  Random      │ │
│  │  Endpoint     │──▶│  Hands       │──▶│  Forest      │ │
│  │  /ws          │   │  (landmarks) │   │  (classify)  │ │
│  └──────────────┘   └──────────────┘   └──────────────┘ │
│                                                          │
│  ┌──────────────┐   ┌──────────────┐                    │
│  │  GET /        │   │  Static      │                    │
│  │  (index.html) │   │  /static/*   │                    │
│  └──────────────┘   └──────────────┘                    │
└─────────────────────────────────────────────────────────┘
```

### Data Flow (per frame)
1. JS captures a video frame from webcam → converts to base64 JPEG
2. Sends base64 string via WebSocket to `/ws`
3. FastAPI decodes base64 → OpenCV image (numpy array)
4. MediaPipe processes image → extracts 21 hand landmarks (x, y, z) = 63 values
5. Landmarks fed into Random Forest → predicted letter + confidence score
6. Server sends back JSON: `{"letter": "A", "confidence": 0.95, "landmarks": [...]}`
7. JS draws landmarks on canvas overlay + displays predicted letter

---

## 5. Design / Color Scheme

Derived from the user's PowerPoint presentation:

| Role | Color | Hex |
|------|-------|-----|
| Primary (headings, labels) | Dark Navy | `#1E3A5F` |
| Accent (badges, highlights) | Teal | `#0F8B8D` |
| Body text | Dark Gray | `#2B3238` |
| Muted / secondary text | Gray | `#667178` |
| Background | White/Light | `#F5F7FA` |
| Cards/containers | White | `#FFFFFF` |
| Success/active | Green | `#27AE60` |
| Error/warning | Red | `#E74C3C` |

**Fonts:** Calibri (body), Cambria (headings) — from the presentation. For web, use system fonts or Google Fonts equivalents (Inter for body, serif for headings).

---

## 6. Project File Structure

```
d:\signsync\
│
├── agent.md                      # THIS FILE — project context & decisions
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation & setup guide
│
├── app.py                        # FastAPI main application entry point
│
├── src/                          # Core Python modules
│   ├── __init__.py
│   ├── hand_detector.py          # MediaPipe hand landmark detection wrapper
│   ├── gesture_classifier.py     # Model loading & inference logic
│   └── utils.py                  # Helpers (base64 decode, image processing)
│
├── data/                         # Dataset & data processing
│   ├── process_dataset.py        # Script to process Kaggle ASL images → landmarks CSV
│   └── asl_alphabet/             # USER DOWNLOADS THIS from Kaggle
│       └── asl_alphabet_train/   # Contains A/, B/, C/, ... Z/ folders with images
│
├── models/                       # Trained model artifacts
│   ├── train.py                  # Model training script
│   └── gesture_model.pkl         # Saved Random Forest model (generated after training)
│
├── static/                       # Frontend static assets
│   ├── css/
│   │   └── style.css             # All styles
│   └── js/
│       └── app.js                # WebSocket client, webcam, canvas, UI logic
│
└── templates/                    # Jinja2 HTML templates
    └── index.html                # Main (and only) page
```

---

## 7. Gesture Vocabulary (Initial Scope)

**Phase 1 (MVP):** 26 ASL alphabet letters (A–Z) — static hand poses only

**Phase 2 (Future):** Common words, dynamic gestures, numbers 0-9

---

## 8. Dataset Details

**Source:** [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

- 87,000 images (200×200 RGB)
- 29 classes: A-Z + SPACE + DELETE + NOTHING
- We use 26 letter classes (A-Z) initially
- User must download manually (requires free Kaggle account)
- Extract to: `d:\signsync\data\asl_alphabet\asl_alphabet_train\`

### Processing Pipeline
1. Load each image from the dataset
2. Run MediaPipe Hands on it → extract 21 landmarks (x, y, z)
3. Flatten to 63-value feature vector
4. Save as CSV: `landmark_0_x, landmark_0_y, landmark_0_z, ..., landmark_20_z, label`
5. This CSV is what the Random Forest trains on

---

## 9. Model Details

| Property | Value |
|----------|-------|
| Algorithm | Random Forest Classifier |
| Library | scikit-learn |
| Input features | 63 (21 landmarks × 3 coordinates) |
| Output classes | 26 (A-Z) |
| Expected accuracy | >90% on test set |
| Training time | < 5 minutes on CPU |
| Model file | `models/gesture_model.pkl` (saved with joblib) |
| Model size | ~10-50 MB |

---

## 10. API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Serve `index.html` (main page) |
| `WS` | `/ws` | WebSocket — receives video frames, returns predictions |
| `GET` | `/static/{path}` | Serve CSS/JS/image static files |
| `GET` | `/health` | Health check endpoint (returns model status) |

### WebSocket Protocol (`/ws`)

**Client → Server (per frame):**
```json
{
  "frame": "<base64-encoded-JPEG>"
}
```

**Server → Client (per frame):**
```json
{
  "success": true,
  "letter": "A",
  "confidence": 0.95,
  "landmarks": [[x, y, z], [x, y, z], ...],
  "hand_detected": true
}
```

**When no hand is detected:**
```json
{
  "success": true,
  "letter": null,
  "confidence": 0.0,
  "landmarks": [],
  "hand_detected": false
}
```

---

## 11. User's System

- **OS:** Windows
- **Python:** 3.11
- **Working directory:** `d:\signsync`
- **No GPU required** — everything runs on CPU
- **Browser:** Any modern browser (Chrome recommended for getUserMedia)

---

## 12. Important Notes

- MediaPipe is PRE-TRAINED — we do NOT train it. We only use it for landmark extraction.
- The only model WE train is the Random Forest on landmark data.
- The Kaggle dataset must be downloaded MANUALLY by the user.
- The app runs LOCALLY at `http://localhost:8000` — not deployed online.
- WebSocket is used for real-time frame-by-frame communication (not REST polling).
- Frontend accesses the webcam via JavaScript `navigator.mediaDevices.getUserMedia()`.
- Prediction smoothing: show a letter only if predicted consistently for 10+ consecutive frames to avoid flickering.
