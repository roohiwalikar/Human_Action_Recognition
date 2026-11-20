import {
  FilesetResolver,
  PoseLandmarker,
  HandLandmarker,
  FaceLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// ============================================================
// DOM ELEMENTS
// ============================================================

const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const showLandmarksCheckbox = document.getElementById("showLandmarks");
const mirrorVideoCheckbox = document.getElementById("mirrorVideo");

const currentActivityEl = document.getElementById("currentActivity");
const activityConfidenceEl = document.getElementById("activityConfidence");
const frameCountEl = document.getElementById("frameCount");
const detectedStatesEl = document.getElementById("detectedStates");
const historyTableBody = document.getElementById("historyTableBody");

const systemStatus = document.getElementById("systemStatus");
const fpsBadge = document.getElementById("fpsBadge");
const poseStatus = document.getElementById("poseStatus");
const faceStatus = document.getElementById("faceStatus");
const handStatus = document.getElementById("handStatus");
const cameraStatus = document.getElementById("cameraStatus");

// ============================================================
// MEDIAPIPE MODELS
// ============================================================

let poseLandmarker = null;
let handLandmarker = null;
let faceLandmarker = null;

// ============================================================
// CAMERA / LOOP VARIABLES
// ============================================================

let stream = null;
let animationId = null;
let running = false;
let frameCounter = 0;
let lastVideoTime = -1;

let lastFpsUpdate = performance.now();
let fpsFrames = 0;

// ============================================================
// HISTORY
// ============================================================

const history = [];
const maxHistory = 12;

// ============================================================
// LSTM VARIABLES
// ============================================================

// The trained LSTM expects:
// 30 frames × 132 features
//
// 33 pose landmarks ×
// (x, y, z, visibility)
// = 132 features

const LSTM_SEQUENCE_LENGTH = 30;
const lstmBuffer = [];

let lstmPredictionInProgress = false;
let lastLSTMActivity = "Waiting...";
let lastLSTMConfidence = 0;

// ============================================================
// OLD MOTION BUFFER
// Used only for additional UI state information.
// It is NOT used for final activity prediction.
// ============================================================

const motionBuffer = [];
const maxBufferSize = 24;

// ============================================================
// STATUS
// ============================================================

function setStatus(text, type = "idle") {
  systemStatus.textContent = text;
  systemStatus.className = `badge ${type}`;
}

// ============================================================
// MODEL STATUS
// ============================================================

function updateModelStatus() {
  poseStatus.textContent = poseLandmarker ? "Ready" : "Not Loaded";
  faceStatus.textContent = faceLandmarker ? "Ready" : "Not Loaded";
  handStatus.textContent = handLandmarker ? "Ready" : "Not Loaded";
}

// ============================================================
// CREATE MEDIAPIPE LANDMARKERS
// ============================================================

async function createLandmarkers() {
  setStatus("Loading AI models...", "idle");

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
    },
    runningMode: "VIDEO",
    numPoses: 1
  });

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    },
    runningMode: "VIDEO",
    numHands: 2
  });

  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    },
    runningMode: "VIDEO",
    numFaces: 1
  });

  updateModelStatus();
  setStatus("System Ready", "live");
}

// ============================================================
// CANVAS
// ============================================================

function resizeCanvas() {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

// ============================================================
// START CAMERA
// ============================================================

async function startCamera() {
  if (running) return;

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: 1280,
        height: 720
      },
      audio: false
    });

    video.srcObject = stream;

    await new Promise((resolve) => {
      video.onloadedmetadata = () => resolve();
    });

    resizeCanvas();

    // Reset prediction buffers
    lstmBuffer.length = 0;
    motionBuffer.length = 0;

    frameCounter = 0;
    lastVideoTime = -1;

    lastLSTMActivity = "Collecting frames...";
    lastLSTMConfidence = 0;

    running = true;

    startBtn.disabled = true;
    stopBtn.disabled = false;

    cameraStatus.textContent = "On";

    setStatus("Camera Live", "live");

    requestAnimationFrame(processFrame);

  } catch (error) {
    console.error(error);

    setStatus("Camera Error", "error");
    cameraStatus.textContent = "Error";

    alert(
      "Could not access camera. Please allow camera permission and use localhost."
    );
  }
}

// ============================================================
// STOP CAMERA
// ============================================================

function stopCamera() {
  running = false;

  if (animationId) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }

  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }

  video.srcObject = null;

  ctx.clearRect(
    0,
    0,
    canvas.width,
    canvas.height
  );

  lstmBuffer.length = 0;
  motionBuffer.length = 0;

  lstmPredictionInProgress = false;

  startBtn.disabled = false;
  stopBtn.disabled = true;

  cameraStatus.textContent = "Off";

  currentActivityEl.textContent = "Waiting for camera...";
  activityConfidenceEl.textContent = "Confidence: 0%";

  setStatus("Stopped", "idle");
}

// ============================================================
// DRAWING FUNCTIONS
// ============================================================

function drawPoint(
  x,
  y,
  color = "#22c55e",
  radius = 4
) {
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, 2 * Math.PI);
  ctx.fillStyle = color;
  ctx.fill();
}

function drawLine(
  a,
  b,
  color = "#38bdf8",
  width = 2
) {
  ctx.beginPath();

  ctx.moveTo(
    a.x * canvas.width,
    a.y * canvas.height
  );

  ctx.lineTo(
    b.x * canvas.width,
    b.y * canvas.height
  );

  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}

function drawPoseLandmarks(landmarks) {
  const connections = [
    [11, 12],
    [11, 13],
    [13, 15],
    [12, 14],
    [14, 16],
    [11, 23],
    [12, 24],
    [23, 24],
    [23, 25],
    [25, 27],
    [24, 26],
    [26, 28],
    [27, 31],
    [28, 32]
  ];

  connections.forEach(([i, j]) => {
    if (landmarks[i] && landmarks[j]) {
      drawLine(
        landmarks[i],
        landmarks[j]
      );
    }
  });

  landmarks.forEach((lm) => {
    drawPoint(
      lm.x * canvas.width,
      lm.y * canvas.height,
      "#22c55e",
      4
    );
  });
}

function drawHandLandmarks(handLandmarksArray) {
  handLandmarksArray.forEach((landmarks) => {
    landmarks.forEach((lm) => {
      drawPoint(
        lm.x * canvas.width,
        lm.y * canvas.height,
        "#f59e0b",
        3
      );
    });
  });
}

function drawFaceLandmarks(faceLandmarksArray) {
  faceLandmarksArray.forEach((landmarks) => {

    [
      33,
      133,
      159,
      145,
      362,
      263,
      386,
      374
    ].forEach((index) => {

      const lm = landmarks[index];

      if (lm) {
        drawPoint(
          lm.x * canvas.width,
          lm.y * canvas.height,
          "#ef4444",
          2
        );
      }

    });
  });
}

// ============================================================
// UTILITY FUNCTIONS
// ============================================================

function distance(a, b) {
  return Math.hypot(
    a.x - b.x,
    a.y - b.y
  );
}

function average(arr) {
  if (!arr.length) return 0;

  return (
    arr.reduce(
      (sum, val) => sum + val,
      0
    ) / arr.length
  );
}

function formatTime(date) {
  return date.toLocaleTimeString();
}

// ============================================================
// HISTORY
// ============================================================

function addHistory(activity, confidence) {

  if (!activity || activity === "Waiting...") {
    return;
  }

  const last = history[0];

  if (
    last &&
    last.activity === activity
  ) {
    return;
  }

  history.unshift({
    time: formatTime(new Date()),
    activity: activity,
    confidence: `${Math.round(confidence)}%`
  });

  if (history.length > maxHistory) {
    history.pop();
  }

  renderHistory();
}

function renderHistory() {
  historyTableBody.innerHTML = "";

  history.forEach((item) => {

    const tr = document.createElement("tr");

    tr.innerHTML = `
      <td>${item.time}</td>
      <td>${item.activity}</td>
      <td>${item.confidence}</td>
    `;

    historyTableBody.appendChild(tr);
  });
}

// ============================================================
// DETECTED STATES
// ============================================================

function updateStates(states) {

  detectedStatesEl.innerHTML = "";

  if (!states.length) {

    detectedStatesEl.innerHTML =
      `<div class="state-item">No states detected</div>`;

    return;
  }

  states.forEach((state) => {

    const div = document.createElement("div");

    div.className = "state-item";

    div.textContent = state;

    detectedStatesEl.appendChild(div);
  });
}

// ============================================================
// EYE DETECTION
// ============================================================

function getEyeAspectRatio(
  top,
  bottom,
  left,
  right
) {
  const vertical = distance(
    top,
    bottom
  );

  const horizontal = distance(
    left,
    right
  );

  return horizontal === 0
    ? 0
    : vertical / horizontal;
}

// ============================================================
// NON-ML STATES
//
// These are only additional visual states.
// They do NOT determine the final activity.
// The final activity comes from LSTM.
// ============================================================

function detectStates(
  pose,
  hands,
  face
) {

  const states = [];

  if (!pose || pose.length < 33) {
    return states;
  }

  const leftShoulder = pose[11];
  const rightShoulder = pose[12];

  const leftWrist = pose[15];
  const rightWrist = pose[16];

  const leftHip = pose[23];
  const rightHip = pose[24];

  const leftKnee = pose[25];
  const rightKnee = pose[26];

  const leftAnkle = pose[27];
  const rightAnkle = pose[28];

  const nose = pose[0];

  // ----------------------------------------------------------
  // HAND STATES
  // ----------------------------------------------------------

  const leftHandRaised =
    leftWrist.y < leftShoulder.y;

  const rightHandRaised =
    rightWrist.y < rightShoulder.y;

  const bothHandsRaised =
    leftHandRaised &&
    rightHandRaised;

  if (leftHandRaised) {
    states.push("Left hand raised");
  }

  if (rightHandRaised) {
    states.push("Right hand raised");
  }

  if (bothHandsRaised) {
    states.push("Both hands raised");
  }

  // ----------------------------------------------------------
  // POSTURE STATES
  // ----------------------------------------------------------

  const shoulderMidY =
    (leftShoulder.y +
      rightShoulder.y) / 2;

  const hipMidY =
    (leftHip.y +
      rightHip.y) / 2;

  const ankleMidY =
    (leftAnkle.y +
      rightAnkle.y) / 2;

  const torsoLength =
    Math.abs(
      hipMidY -
      shoulderMidY
    );

  const bodyHeight =
    Math.abs(
      ankleMidY -
      nose.y
    );

  const sittingCondition =
    average([
      Math.abs(
        leftHip.y -
        leftKnee.y
      ),
      Math.abs(
        rightHip.y -
        rightKnee.y
      )
    ]) <
    torsoLength * 0.75;

  const standingCondition =
    Math.abs(
      leftHip.y -
      leftKnee.y
    ) >
      torsoLength * 0.7 &&
    Math.abs(
      rightHip.y -
      rightKnee.y
    ) >
      torsoLength * 0.7 &&
    bodyHeight > 0.45;

  if (sittingCondition) {
    states.push("Sitting posture");
  }

  if (standingCondition) {
    states.push("Standing posture");
  }

  // ----------------------------------------------------------
  // FACE / EYE STATE
  // ----------------------------------------------------------

  if (
    face &&
    face.length > 0
  ) {

    const f = face[0];

    const leftEAR =
      getEyeAspectRatio(
        f[159],
        f[145],
        f[33],
        f[133]
      );

    const rightEAR =
      getEyeAspectRatio(
        f[386],
        f[374],
        f[362],
        f[263]
      );

    const avgEAR =
      (leftEAR + rightEAR) / 2;

    if (avgEAR < 0.18) {
      states.push("Eyes closed");
    } else {
      states.push("Eyes open");
    }
  }

  // ----------------------------------------------------------
  // MOTION INFORMATION
  // ----------------------------------------------------------

  motionBuffer.push({
    timestamp: performance.now(),

    noseY: nose.y,

    leftWristX:
      leftWrist.x,

    rightWristX:
      rightWrist.x,

    leftAnkleX:
      leftAnkle.x,

    rightAnkleX:
      rightAnkle.x,

    hipY:
      hipMidY
  });

  if (
    motionBuffer.length >
    maxBufferSize
  ) {
    motionBuffer.shift();
  }

  if (motionBuffer.length > 8) {

    const noseYs =
      motionBuffer.map(
        (m) => m.noseY
      );

    const leftWristXs =
      motionBuffer.map(
        (m) => m.leftWristX
      );

    const rightWristXs =
      motionBuffer.map(
        (m) => m.rightWristX
      );

    const leftAnkleXs =
      motionBuffer.map(
        (m) => m.leftAnkleX
      );

    const rightAnkleXs =
      motionBuffer.map(
        (m) => m.rightAnkleX
      );

    const verticalRange =
      Math.max(...noseYs) -
      Math.min(...noseYs);

    const leftWristRange =
      Math.max(...leftWristXs) -
      Math.min(...leftWristXs);

    const rightWristRange =
      Math.max(...rightWristXs) -
      Math.min(...rightWristXs);

    const leftAnkleRange =
      Math.max(...leftAnkleXs) -
      Math.min(...leftAnkleXs);

    const rightAnkleRange =
      Math.max(...rightAnkleXs) -
      Math.min(...rightAnkleXs);

    if (verticalRange > 0.08) {
      states.push("Jump-like movement");
    }

    if (
      leftWristRange > 0.12 ||
      rightWristRange > 0.12
    ) {
      states.push("Hand movement");
    }

    if (
      leftAnkleRange > 0.05 ||
      rightAnkleRange > 0.05
    ) {
      states.push("Walking motion");
    }
  }

  return states;
}

// ============================================================
// EXTRACT 132 LSTM FEATURES
// ============================================================

function extractPoseFeatures(
  poseLandmarks
) {

  if (
    !poseLandmarks ||
    poseLandmarks.length !== 33
  ) {
    return null;
  }

  const features = [];

  for (
    const landmark of poseLandmarks
  ) {

    features.push(
      landmark.x,
      landmark.y,
      landmark.z,
      landmark.visibility ?? 0
    );

  }

  return features;
}

// ============================================================
// FORMAT LSTM CLASS NAME
// ============================================================

function formatActivityName(name) {

  if (!name) {
    return "Unknown";
  }

  const names = {

    dancing: "Dancing",

    hands_up: "Hands Up",

    jumping: "Jumping",

    lying_down: "Lying Down",

    reading: "Reading",

    sitting: "Sitting",

    standing: "Standing",

    walking: "Walking",

    waving: "Waving"
  };

  return (
    names[name] ||
    name
      .replaceAll("_", " ")
      .replace(
        /\b\w/g,
        (char) =>
          char.toUpperCase()
      )
  );
}

// ============================================================
// SEND 30-FRAME SEQUENCE TO FLASK LSTM SERVER
// ============================================================

async function predictWithLSTM(
  sequence
) {

  if (
    lstmPredictionInProgress
  ) {
    return;
  }

  if (
    !sequence ||
    sequence.length !==
      LSTM_SEQUENCE_LENGTH
  ) {
    return;
  }

  lstmPredictionInProgress = true;

  try {

    const response =
      await fetch(
        "http://127.0.0.1:5000/predict",
        {
          method: "POST",

          headers: {
            "Content-Type":
              "application/json"
          },

          body: JSON.stringify({
            sequence: sequence
          })
        }
      );

    if (!response.ok) {
      throw new Error(
        `LSTM server returned ${response.status}`
      );
    }

    const result =
      await response.json();

    if (result.error) {
      throw new Error(
        result.error
      );
    }

    // --------------------------------------------------------
    // Accept the prediction returned by Flask
    // --------------------------------------------------------

    let activityName = null;
    let confidence = 0;

    if (result.activity) {
      activityName =
        result.activity;
    } else if (result.prediction) {
      activityName =
        result.prediction;
    } else if (result.class_name) {
      activityName =
        result.class_name;
    }

    if (
      result.confidence !==
      undefined
    ) {
      confidence =
        Number(result.confidence);
    } else if (
      result.probability !==
      undefined
    ) {
      confidence =
        Number(result.probability);
    }

    // If server sends confidence as decimal
    // e.g. 0.82, convert to percentage.
    if (
      confidence > 0 &&
      confidence <= 1
    ) {
      confidence *= 100;
    }

    if (activityName) {

      const formattedActivity =
        formatActivityName(
          activityName
        );

      lastLSTMActivity =
        formattedActivity;

      lastLSTMConfidence =
        confidence;

      currentActivityEl.textContent =
        formattedActivity;

      activityConfidenceEl.textContent =
        `Confidence: ${Math.round(
          confidence
        )}%`;

      addHistory(
        formattedActivity,
        confidence
      );
    }

  } catch (error) {

    console.error(
      "LSTM prediction error:",
      error
    );

    // Do not replace the activity with
    // a fake rule-based prediction.
    // Show server connection status.

    if (
      lastLSTMActivity ===
      "Waiting..." ||
      lastLSTMActivity ===
      "Collecting frames..."
    ) {

      currentActivityEl.textContent =
        "LSTM Server Error";

      activityConfidenceEl.textContent =
        "Confidence: 0%";
    }

  } finally {

    lstmPredictionInProgress =
      false;
  }
}

// ============================================================
// PROCESS VIDEO FRAME
// ============================================================

function processFrame() {

  if (!running) {
    return;
  }

  const now =
    performance.now();

  // Only process a new video frame
  if (
    video.currentTime !==
    lastVideoTime
  ) {

    lastVideoTime =
      video.currentTime;

    frameCounter++;

    frameCountEl.textContent =
      `Frames Analyzed: ${frameCounter}`;

    // --------------------------------------------------------
    // FPS
    // --------------------------------------------------------

    fpsFrames++;

    const elapsed =
      now - lastFpsUpdate;

    if (elapsed >= 1000) {

      const fps =
        Math.round(
          (fpsFrames * 1000) /
            elapsed
        );

      fpsBadge.textContent =
        `FPS: ${fps}`;

      fpsFrames = 0;

      lastFpsUpdate = now;
    }

    // --------------------------------------------------------
    // CLEAR CANVAS
    // --------------------------------------------------------

    ctx.clearRect(
      0,
      0,
      canvas.width,
      canvas.height
    );

    // --------------------------------------------------------
    // MIRROR VIDEO
    // --------------------------------------------------------

    if (
      mirrorVideoCheckbox.checked
    ) {

      video.style.transform =
        "scaleX(-1)";

      canvas.style.transform =
        "scaleX(-1)";

    } else {

      video.style.transform =
        "scaleX(1)";

      canvas.style.transform =
        "scaleX(1)";
    }

    // --------------------------------------------------------
    // MEDIAPIPE DETECTION
    // --------------------------------------------------------

    const poseResults =
      poseLandmarker.detectForVideo(
        video,
        now
      );

    const handResults =
      handLandmarker.detectForVideo(
        video,
        now
      );

    const faceResults =
      faceLandmarker.detectForVideo(
        video,
        now
      );

    const poseLandmarks =
      poseResults.landmarks?.[0] ||
      null;

    const handLandmarks =
      handResults.landmarks || [];

    const faceLandmarks =
      faceResults.faceLandmarks || [];

    // --------------------------------------------------------
    // DRAW LANDMARKS
    // --------------------------------------------------------

    if (
      showLandmarksCheckbox.checked
    ) {

      if (poseLandmarks) {
        drawPoseLandmarks(
          poseLandmarks
        );
      }

      if (
        handLandmarks.length
      ) {
        drawHandLandmarks(
          handLandmarks
        );
      }

      if (
        faceLandmarks.length
      ) {
        drawFaceLandmarks(
          faceLandmarks
        );
      }
    }

    // --------------------------------------------------------
    // NO BODY
    // --------------------------------------------------------

    if (!poseLandmarks) {

      lstmBuffer.length = 0;

      currentActivityEl.textContent =
        "No body detected";

      activityConfidenceEl.textContent =
        "Confidence: 0%";

      updateStates([]);

    } else {

      // ------------------------------------------------------
      // ADD 132 FEATURES TO LSTM BUFFER
      // ------------------------------------------------------

      const frameFeatures =
        extractPoseFeatures(
          poseLandmarks
        );

      if (frameFeatures) {

        lstmBuffer.push(
          frameFeatures
        );

        // Keep exactly the latest 30 frames
        if (
          lstmBuffer.length >
          LSTM_SEQUENCE_LENGTH
        ) {

          lstmBuffer.shift();
        }

        // ----------------------------------------------------
        // PREDICT ONLY WHEN 30 FRAMES ARE READY
        // ----------------------------------------------------

        if (
          lstmBuffer.length ===
          LSTM_SEQUENCE_LENGTH
        ) {

          // Copy the sequence so that
          // the buffer can continue receiving frames.
          const sequence =
            lstmBuffer.map(
              (frame) =>
                [...frame]
            );

          predictWithLSTM(
            sequence
          );
        }
      }

      // ------------------------------------------------------
      // ADDITIONAL DETECTED STATES
      // ------------------------------------------------------

      const states =
        detectStates(
          poseLandmarks,
          handLandmarks,
          faceLandmarks
        );

      updateStates(states);
    }
  }

  animationId =
    requestAnimationFrame(
      processFrame
    );
}

// ============================================================
// BUTTON EVENTS
// ============================================================

startBtn.addEventListener(
  "click",
  startCamera
);

stopBtn.addEventListener(
  "click",
  stopCamera
);

// ============================================================
// WINDOW RESIZE
// ============================================================

window.addEventListener(
  "resize",
  () => {

    if (
      video.videoWidth &&
      video.videoHeight
    ) {
      resizeCanvas();
    }

  }
);

// ============================================================
// INITIALIZATION
// ============================================================

async function init() {

  try {

    await createLandmarkers();

  } catch (error) {

    console.error(error);

    setStatus(
      "Model Load Error",
      "error"
    );

    poseStatus.textContent =
      "Error";

    faceStatus.textContent =
      "Error";

    handStatus.textContent =
      "Error";
  }
}

init();