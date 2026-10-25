/**
 * SignSync — Real-Time Sign Language Recognition Frontend Controller
 * Handles Webcam capture, WebSocket frame transmission, Canvas landmark visualization,
 * Prediction stabilization, and Text Assembly.
 */

// =========================================================
// APPLICATION STATE
// =========================================================
const state = {
    ws: null,
    stream: null,
    isCameraRunning: false,
    frameInterval: null,
    showSkeleton: true,
    formedText: "",
    lastSmoothedLetter: null,
    lastLetterAddedTime: 0,
    letterHoldThresholdMs: 1200, // Cooldown before repeating the same held letter
    fps: 0,
    frameCount: 0,
    lastFpsUpdateTime: performance.now(),
    waitingForServerResponse: false,
};

// =========================================================
// DOM ELEMENTS
// =========================================================
const elements = {
    webcam: document.getElementById("webcam"),
    canvas: document.getElementById("landmark-canvas"),
    placeholder: document.getElementById("camera-placeholder"),
    feedLiveIndicator: document.getElementById("feed-live-indicator"),
    detectionPill: document.getElementById("detection-pill"),
    btnStart: document.getElementById("btn-start"),
    btnStop: document.getElementById("btn-stop"),
    btnToggleSkeleton: document.getElementById("btn-toggle-skeleton"),
    skeletonToggleText: document.getElementById("skeleton-toggle-text"),

    badgeConnection: document.getElementById("badge-connection"),
    connectionStatusText: document.getElementById("connection-status-text"),
    badgeModel: document.getElementById("badge-model"),
    modelStatusText: document.getElementById("model-status-text"),

    predictedChar: document.getElementById("predicted-char"),
    confidenceBadge: document.getElementById("confidence-badge"),
    confidenceBar: document.getElementById("confidence-bar"),
    rawChar: document.getElementById("raw-char"),
    smoothedChar: document.getElementById("smoothed-char"),
    handStatus: document.getElementById("hand-status"),

    formedTextContent: document.getElementById("formed-text-content"),
    charCounter: document.getElementById("char-counter"),
    btnSpace: document.getElementById("btn-space"),
    btnBackspace: document.getElementById("btn-backspace"),
    btnClear: document.getElementById("btn-clear"),
    btnSpeak: document.getElementById("btn-speak"),
    btnCopy: document.getElementById("btn-copy"),

    footerConnectionVal: document.getElementById("footer-connection-val"),
    footerFpsVal: document.getElementById("footer-fps-val"),
    footerHandVal: document.getElementById("footer-hand-val"),
};

const ctx = elements.canvas.getContext("2d");

// Offscreen canvas used to grab frames from the video element and encode to JPEG
const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d");

// =========================================================
// MEDIAPIPE HAND TOPOLOGY (Connections)
// =========================================================
const HAND_CONNECTIONS = [
    // Palm base
    [0, 1], [0, 5], [5, 9], [9, 13], [13, 17], [0, 17],
    // Thumb
    [1, 2], [2, 3], [3, 4],
    // Index finger
    [5, 6], [6, 7], [7, 8],
    // Middle finger
    [9, 10], [10, 11], [11, 12],
    // Ring finger
    [13, 14], [14, 15], [15, 16],
    // Pinky
    [17, 18], [18, 19], [19, 20],
];

// =========================================================
// WEBSOCKET COMMUNICATION
// =========================================================
function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    updateConnectionStatus("connecting", "Connecting...");

    try {
        state.ws = new WebSocket(wsUrl);
    } catch (e) {
        console.error("[WS] Initialization error:", e);
        updateConnectionStatus("disconnected", "Offline");
        return;
    }

    state.ws.onopen = () => {
        updateConnectionStatus("connected", "Connected");
    };

    state.ws.onmessage = (event) => {
        state.waitingForServerResponse = false;
        try {
            const data = JSON.parse(event.data);
            handleServerPrediction(data);
        } catch (err) {
            console.error("[WS] Message parse error:", err);
        }
    };

    state.ws.onclose = () => {
        updateConnectionStatus("disconnected", "Disconnected");
        state.waitingForServerResponse = false;
        if (state.isCameraRunning) {
            setTimeout(connectWebSocket, 2000);
        }
    };

    state.ws.onerror = (err) => {
        console.error("[WS] Socket error:", err);
        updateConnectionStatus("disconnected", "Error");
    };
}

function updateConnectionStatus(status, text) {
    elements.badgeConnection.className = `status-pill ${status}`;
    elements.connectionStatusText.textContent = text;
    elements.footerConnectionVal.textContent = text;
}

// =========================================================
// CAMERA INITIALIZATION & FRAME CAPTURE
// =========================================================
async function startCamera() {
    if (state.isCameraRunning) return;

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user",
            },
            audio: false,
        });

        state.stream = stream;
        elements.webcam.srcObject = stream;

        elements.webcam.onloadedmetadata = () => {
            elements.webcam.play();
            const vw = elements.webcam.videoWidth || 640;
            const vh = elements.webcam.videoHeight || 480;

            // Sync canvas dimensions with video feed
            elements.canvas.width = vw;
            elements.canvas.height = vh;

            // High-resolution capture canvas for crisp hand joint recognition
            captureCanvas.width = 640;
            captureCanvas.height = 480;

            elements.placeholder.style.display = "none";
            elements.feedLiveIndicator.style.display = "flex";
            elements.btnStart.disabled = true;
            elements.btnStop.disabled = false;
            state.isCameraRunning = true;

            if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
                connectWebSocket();
            }

            // Stream frames at ~14 FPS (70ms interval) for smooth tracking & low latency
            state.frameInterval = setInterval(captureAndSendFrame, 70);
        };
    } catch (err) {
        console.error("[Camera] Access error:", err);
        alert(`Could not access webcam: ${err.message || err.name}. Please ensure camera permissions are allowed.`);
    }
}

function stopCamera() {
    if (!state.isCameraRunning) return;

    if (state.frameInterval) {
        clearInterval(state.frameInterval);
        state.frameInterval = null;
    }

    if (state.stream) {
        state.stream.getTracks().forEach((track) => track.stop());
        state.stream = null;
    }

    elements.webcam.srcObject = null;
    clearCanvas();

    elements.placeholder.style.display = "flex";
    elements.feedLiveIndicator.style.display = "none";
    elements.detectionPill.style.display = "none";
    elements.btnStart.disabled = false;
    elements.btnStop.disabled = true;
    state.isCameraRunning = false;

    resetPredictionDisplay();
}

function captureAndSendFrame() {
    if (!state.isCameraRunning || !state.ws || state.ws.readyState !== WebSocket.OPEN) {
        return;
    }

    // Skip frame if server is still processing to prevent network queue lag
    if (state.waitingForServerResponse) {
        return;
    }

    // Capture clear 640x480 frame for crisp finger detection
    captureCtx.drawImage(elements.webcam, 0, 0, captureCanvas.width, captureCanvas.height);

    // 0.82 quality gives crisp hand boundaries without artifacts
    const base64Data = captureCanvas.toDataURL("image/jpeg", 0.82);

    try {
        state.waitingForServerResponse = true;
        state.ws.send(JSON.stringify({ frame: base64Data }));
        updateFps();
    } catch (e) {
        state.waitingForServerResponse = false;
        console.error("[WS] Send error:", e);
    }
}

function updateFps() {
    state.frameCount++;
    const now = performance.now();
    const elapsed = now - state.lastFpsUpdateTime;

    if (elapsed >= 1000) {
        state.fps = Math.round((state.frameCount * 1000) / elapsed);
        elements.footerFpsVal.textContent = `${state.fps} fps`;
        state.frameCount = 0;
        state.lastFpsUpdateTime = now;
    }
}

// =========================================================
// PREDICTION & LANDMARK RENDERING
// =========================================================
function handleServerPrediction(data) {
    if (!data.success) {
        return;
    }

    clearCanvas();

    // 1. Hand presence & landmark rendering
    if (data.hand_detected && data.landmarks && data.landmarks.length > 0) {
        elements.footerHandVal.textContent = "Hand Tracked (21 points)";
        elements.handStatus.textContent = "Hand Active";
        elements.detectionPill.style.display = "block";

        if (state.showSkeleton) {
            drawLandmarks(data.landmarks);
        }
    } else {
        elements.footerHandVal.textContent = "No Hand Detected";
        elements.handStatus.textContent = "Awaiting Hand";
        elements.detectionPill.style.display = "none";
        resetPredictionDisplay();
        return;
    }

    // 2. Gesture classification values
    const rawLetter = data.letter;
    const confidence = data.confidence || 0.0;
    const smoothedLetter = data.smoothed_letter;

    // Raw display
    elements.rawChar.textContent = rawLetter || "—";
    const confPercent = Math.round(confidence * 100);
    elements.confidenceBadge.textContent = `${confPercent}% Confidence`;
    elements.confidenceBar.style.width = `${confPercent}%`;

    // Smoothed display
    if (smoothedLetter) {
        elements.smoothedChar.textContent = smoothedLetter;
        elements.predictedChar.textContent = smoothedLetter;
        elements.predictedChar.classList.add("updated");
        setTimeout(() => elements.predictedChar.classList.remove("updated"), 150);

        // 3. Assemble text automatically with anti-duplication cooldown
        processTextAccumulation(smoothedLetter);
    }
}

function processTextAccumulation(letter) {
    const now = Date.now();
    const isNewLetter = letter !== state.lastSmoothedLetter;
    const holdExpired = now - state.lastLetterAddedTime >= state.letterHoldThresholdMs;

    if (isNewLetter || holdExpired) {
        appendCharacter(letter);
        state.lastSmoothedLetter = letter;
        state.lastLetterAddedTime = now;
    }
}

function drawLandmarks(landmarks) {
    const w = elements.canvas.width;
    const h = elements.canvas.height;

    const points = landmarks.map((lm) => ({
        x: lm[0] * w,
        y: lm[1] * h,
    }));

    // 1. Draw connections
    ctx.lineWidth = 3.5;
    ctx.strokeStyle = "rgba(15, 139, 141, 0.9)"; // Teal stroke
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
        if (points[startIdx] && points[endIdx]) {
            ctx.beginPath();
            ctx.moveTo(points[startIdx].x, points[startIdx].y);
            ctx.lineTo(points[endIdx].x, points[endIdx].y);
            ctx.stroke();
        }
    }

    // 2. Draw landmark joints
    for (let i = 0; i < points.length; i++) {
        const pt = points[i];
        ctx.beginPath();
        const isFingertip = [4, 8, 12, 16, 20].includes(i);
        const radius = isFingertip ? 6 : 4;

        ctx.arc(pt.x, pt.y, radius, 0, 2 * Math.PI);
        ctx.fillStyle = isFingertip ? "#10B981" : "#FFFFFF"; // Green fingertips, white joints
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = "#1E3A5F"; // Navy border
        ctx.stroke();
    }
}

function clearCanvas() {
    ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
}

function resetPredictionDisplay() {
    elements.predictedChar.textContent = "—";
    elements.rawChar.textContent = "—";
    elements.smoothedChar.textContent = "—";
    elements.confidenceBadge.textContent = "0% Confidence";
    elements.confidenceBar.style.width = "0%";
}

// =========================================================
// FORMED TEXT ACTIONS
// =========================================================
function appendCharacter(char) {
    state.formedText += char;
    updateFormedTextUI();
}

function insertSpace() {
    state.formedText += " ";
    state.lastSmoothedLetter = " ";
    updateFormedTextUI();
}

function backspaceText() {
    state.formedText = state.formedText.slice(0, -1);
    state.lastSmoothedLetter = null;
    updateFormedTextUI();
}

function clearText() {
    state.formedText = "";
    state.lastSmoothedLetter = null;
    updateFormedTextUI();
}

function updateFormedTextUI() {
    elements.formedTextContent.textContent = state.formedText;
    elements.charCounter.textContent = `${state.formedText.length} chars`;
}

function speakText() {
    if (!state.formedText.trim()) return;

    if ("speechSynthesis" in window) {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(state.formedText);
        utterance.rate = 0.95;
        window.speechSynthesis.speak(utterance);
    } else {
        alert("Text-to-speech is not supported by your browser.");
    }
}

async function copyText() {
    if (!state.formedText) return;
    try {
        await navigator.clipboard.writeText(state.formedText);
        const origText = elements.btnCopy.querySelector("span").textContent;
        elements.btnCopy.querySelector("span").textContent = "Copied!";
        setTimeout(() => {
            elements.btnCopy.querySelector("span").textContent = origText;
        }, 1500);
    } catch (e) {
        console.error("Copy failed:", e);
    }
}

// =========================================================
// EVENT LISTENERS & INITIALIZATION
// =========================================================
elements.btnStart.addEventListener("click", startCamera);
elements.btnStop.addEventListener("click", stopCamera);

elements.btnToggleSkeleton.addEventListener("click", () => {
    state.showSkeleton = !state.showSkeleton;
    elements.skeletonToggleText.textContent = state.showSkeleton ? "Hide Skeleton" : "Show Skeleton";
    if (!state.showSkeleton) clearCanvas();
});

elements.btnSpace.addEventListener("click", insertSpace);
elements.btnBackspace.addEventListener("click", backspaceText);
elements.btnClear.addEventListener("click", clearText);
elements.btnSpeak.addEventListener("click", speakText);
elements.btnCopy.addEventListener("click", copyText);

// Check model status via health endpoint
async function checkHealth() {
    try {
        const res = await fetch("/health");
        const data = await res.json();
        if (data.model_loaded) {
            elements.badgeModel.className = "status-pill ready";
            elements.modelStatusText.textContent = `Model Ready (${data.vocabulary.length} Signs)`;
        } else {
            elements.badgeModel.className = "status-pill warning";
            elements.modelStatusText.textContent = "Model Not Loaded";
        }
    } catch (e) {
        console.warn("[Health] Failed to fetch health status:", e);
    }
}

document.addEventListener("DOMContentLoaded", () => {
    checkHealth();
});
