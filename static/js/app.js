/**
 * SignSync — Real-Time Sign Language Translator
 * Modern, Human-Centric Controller
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
    letterHoldThresholdMs: 1200,
    fps: 0,
    frameCount: 0,
    lastFpsUpdateTime: performance.now(),
    waitingForServerResponse: false,
};

// Quick gesture guide data
const ASL_GUIDE_DATA = [
    { letter: "A", hint: "Fist, thumb upright beside index" },
    { letter: "B", hint: "Flat hand, 4 fingers up, thumb in" },
    { letter: "C", hint: "Fingers curved like letter C" },
    { letter: "D", hint: "Index up, other fingers touch thumb" },
    { letter: "E", hint: "Fingers curled onto palm" },
    { letter: "F", hint: "Index & thumb touch, 3 fingers up" },
    { letter: "G", hint: "Index & thumb point horizontal" },
    { letter: "H", hint: "Index & middle fingers horizontal" },
    { letter: "I", hint: "Pinky straight up, others closed" },
    { letter: "J", hint: "Pinky traces J in air" },
    { letter: "K", hint: "Index up, middle forward, thumb in" },
    { letter: "L", hint: "L-shape: thumb & index at 90°" },
    { letter: "M", hint: "3 fingers over thumb" },
    { letter: "N", hint: "2 fingers over thumb" },
    { letter: "O", hint: "All fingers touch thumb in O-ring" },
    { letter: "P", hint: "K-shape pointing downward" },
    { letter: "Q", hint: "G-shape pointing downward" },
    { letter: "R", hint: "Index and middle fingers crossed" },
    { letter: "S", hint: "Fist with thumb across fingers" },
    { letter: "T", hint: "Thumb tucked under index finger" },
    { letter: "U", hint: "Index & middle together upright" },
    { letter: "V", hint: "Peace sign: Index & middle apart" },
    { letter: "W", hint: "3 fingers up: index, middle, ring" },
    { letter: "X", hint: "Index finger hooked like a hook" },
    { letter: "Y", hint: "Shaka sign: thumb & pinky out" },
    { letter: "Z", hint: "Index finger traces Z in air" },
];

// =========================================================
// DOM ELEMENTS
// =========================================================
const elements = {
    webcam: document.getElementById("webcam"),
    canvas: document.getElementById("landmark-canvas"),
    cameraPlaceholder: document.getElementById("camera-placeholder"),
    cameraHudTop: document.getElementById("camera-hud-top"),
    cameraHudBottom: document.getElementById("camera-hud-bottom"),
    hudChar: document.getElementById("hud-char"),
    hudConfidence: document.getElementById("hud-confidence"),
    
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
    handStatus: document.getElementById("hand-status"),

    formedTextBox: document.getElementById("formed-text-box"),
    formedTextContent: document.getElementById("formed-text-content"),
    emptyTranscriptHint: document.getElementById("empty-transcript-hint"),
    charCounter: document.getElementById("char-counter"),
    btnFontToggle: document.getElementById("btn-font-toggle"),

    btnSpace: document.getElementById("btn-space"),
    btnBackspace: document.getElementById("btn-backspace"),
    btnClear: document.getElementById("btn-clear"),
    btnSpeak: document.getElementById("btn-speak"),
    btnCopy: document.getElementById("btn-copy"),

    footerConnectionVal: document.getElementById("footer-connection-val"),
    footerFpsVal: document.getElementById("footer-fps-val"),
    footerHandVal: document.getElementById("footer-hand-val"),

    btnOpenGuide: document.getElementById("btn-open-guide"),
    btnCloseGuide: document.getElementById("btn-close-guide"),
    guideModal: document.getElementById("guide-modal"),
    aslGuideGrid: document.getElementById("asl-guide-grid"),
};

const ctx = elements.canvas.getContext("2d");

const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d");

const HAND_CONNECTIONS = [
    [0, 1], [0, 5], [5, 9], [9, 13], [13, 17], [0, 17],
    [1, 2], [2, 3], [3, 4],
    [5, 6], [6, 7], [7, 8],
    [9, 10], [10, 11], [11, 12],
    [13, 14], [14, 15], [15, 16],
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
    elements.badgeConnection.className = `status-indicator ${status}`;
    elements.connectionStatusText.textContent = text;
    elements.footerConnectionVal.textContent = text;
}

// =========================================================
// CAMERA INITIALIZATION & STREAMING
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

            elements.canvas.width = vw;
            elements.canvas.height = vh;

            captureCanvas.width = 640;
            captureCanvas.height = 480;

            // Transition UI from Hero to Active Live View
            elements.cameraPlaceholder.style.display = "none";
            elements.cameraHudTop.style.display = "flex";
            elements.cameraHudBottom.style.display = "flex";
            state.isCameraRunning = true;

            if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
                connectWebSocket();
            }

            state.frameInterval = setInterval(captureAndSendFrame, 70);
        };
    } catch (err) {
        console.error("[Camera] Access error:", err);
        alert(`Could not access camera: ${err.message || err.name}. Please ensure permissions are granted.`);
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

    elements.cameraPlaceholder.style.display = "flex";
    elements.cameraHudTop.style.display = "none";
    elements.cameraHudBottom.style.display = "none";
    state.isCameraRunning = false;

    resetPredictionDisplay();
}

function captureAndSendFrame() {
    if (!state.isCameraRunning || !state.ws || state.ws.readyState !== WebSocket.OPEN) {
        return;
    }

    if (state.waitingForServerResponse) {
        return;
    }

    captureCtx.drawImage(elements.webcam, 0, 0, captureCanvas.width, captureCanvas.height);
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
    if (!data.success) return;

    clearCanvas();

    if (data.hand_detected && data.landmarks && data.landmarks.length > 0) {
        elements.footerHandVal.textContent = "Hand detected";
        if (state.showSkeleton) {
            drawLandmarks(data.landmarks);
        }
    } else {
        elements.footerHandVal.textContent = "Waiting for hand";
        resetPredictionDisplay();
        return;
    }

    const rawLetter = data.letter;
    const confidence = data.confidence || 0.0;
    const smoothedLetter = data.smoothed_letter;
    const displayChar = smoothedLetter || rawLetter;
    const confPercent = Math.round(confidence * 100);

    // Update HUD
    elements.hudChar.textContent = displayChar || "—";
    elements.hudConfidence.textContent = `${confPercent}%`;

    // Update Hero Card
    elements.confidenceBadge.textContent = `${confPercent}% Match`;
    elements.confidenceBar.style.width = `${confPercent}%`;

    if (displayChar) {
        elements.predictedChar.textContent = displayChar;
        elements.predictedChar.classList.add("updated");
        setTimeout(() => elements.predictedChar.classList.remove("updated"), 150);
        elements.handStatus.textContent = `Signing: ${displayChar}`;
    }

    if (smoothedLetter) {
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

    // Natural warm ivory lines with subtle shadow for clarity on any skin tone
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(250, 248, 245, 0.85)";
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

    // Natural tactile joint markers (terracotta fingertips, warm ivory knuckles)
    for (let i = 0; i < points.length; i++) {
        const pt = points[i];
        ctx.beginPath();
        const isFingertip = [4, 8, 12, 16, 20].includes(i);
        const radius = isFingertip ? 5 : 3.5;

        ctx.arc(pt.x, pt.y, radius, 0, 2 * Math.PI);
        ctx.fillStyle = isFingertip ? "#C86D51" : "#FAF8F5";
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = "#231F20";
        ctx.stroke();
    }
}

function clearCanvas() {
    ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
}

function resetPredictionDisplay() {
    elements.predictedChar.textContent = "—";
    elements.hudChar.textContent = "—";
    elements.hudConfidence.textContent = "0%";
    elements.confidenceBadge.textContent = "0% Match";
    elements.confidenceBar.style.width = "0%";
    elements.handStatus.textContent = "Waiting for sign";
}

// =========================================================
// TEXT ASSEMBLE ACTIONS
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
    elements.charCounter.textContent = `${state.formedText.length} letters`;
    elements.emptyTranscriptHint.style.display = state.formedText.length > 0 ? "none" : "block";
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
        const originalText = elements.btnCopy.querySelector("span").textContent;
        elements.btnCopy.querySelector("span").textContent = "Copied!";
        setTimeout(() => {
            elements.btnCopy.querySelector("span").textContent = originalText;
        }, 1500);
    } catch (e) {
        console.error("Copy failed:", e);
    }
}

// =========================================================
// ASL GUIDE MODAL
// =========================================================
function populateGuide() {
    elements.aslGuideGrid.innerHTML = "";
    ASL_GUIDE_DATA.forEach((item) => {
        const card = document.createElement("div");
        card.className = "asl-card";
        card.innerHTML = `
            <div class="asl-card-letter">${item.letter}</div>
            <div class="asl-card-hint">${item.hint}</div>
        `;
        elements.aslGuideGrid.appendChild(card);
    });
}

function openGuide() {
    elements.guideModal.style.display = "flex";
}

function closeGuide() {
    elements.guideModal.style.display = "none";
}

// =========================================================
// EVENT LISTENERS & INITIALIZATION
// =========================================================
elements.btnStart.addEventListener("click", startCamera);
elements.btnStop.addEventListener("click", stopCamera);

elements.btnToggleSkeleton.addEventListener("click", () => {
    state.showSkeleton = !state.showSkeleton;
    elements.btnToggleSkeleton.classList.toggle("active", state.showSkeleton);
    elements.skeletonToggleText.textContent = state.showSkeleton ? "Skeleton" : "Hidden";
    if (!state.showSkeleton) clearCanvas();
});

elements.btnSpace.addEventListener("click", insertSpace);
elements.btnBackspace.addEventListener("click", backspaceText);
elements.btnClear.addEventListener("click", clearText);
elements.btnSpeak.addEventListener("click", speakText);
elements.btnCopy.addEventListener("click", copyText);

// Font size toggle for accessibility
elements.btnFontToggle.addEventListener("click", () => {
    elements.formedTextBox.classList.toggle("large-font");
});

// Guide modal
elements.btnOpenGuide.addEventListener("click", openGuide);
elements.btnCloseGuide.addEventListener("click", closeGuide);
elements.guideModal.addEventListener("click", (e) => {
    if (e.target === elements.guideModal) closeGuide();
});

// Model health check
async function checkHealth() {
    try {
        const res = await fetch("/health");
        const data = await res.json();
        if (data.model_loaded) {
            elements.badgeModel.className = "status-indicator ready";
            elements.modelStatusText.textContent = `${data.vocabulary.length} Signs Ready`;
        } else {
            elements.badgeModel.className = "status-indicator warning";
            elements.modelStatusText.textContent = "Model Offline";
        }
    } catch (e) {
        console.warn("[Health] Error checking server status:", e);
    }
}

document.addEventListener("DOMContentLoaded", () => {
    populateGuide();
    checkHealth();
});
