import cv2
import os
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# HUMAN ACTION RECOGNITION USING ML AND DL
# REAL-TIME RANDOM FOREST PREDICTION
# ==========================================

MODEL_PATH = os.path.join(
    "models",
    "random_forest_proper_model.pkl"
)

POSE_MODEL_PATH = "pose_landmarker_full.task"


# ==========================================
# CHECK FILES
# ==========================================

if not os.path.exists(MODEL_PATH):
    print("ERROR: Random Forest model not found.")
    print("Expected:", MODEL_PATH)
    raise SystemExit

if not os.path.exists(POSE_MODEL_PATH):
    print("ERROR: MediaPipe pose model not found.")
    print("Expected:", POSE_MODEL_PATH)
    raise SystemExit


# ==========================================
# LOAD RANDOM FOREST MODEL
# ==========================================

print("Loading Random Forest model...")

model = joblib.load(MODEL_PATH)

print("Random Forest model loaded successfully.")


# ==========================================
# MEDIAPIPE POSE LANDMARKER
# ==========================================

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
RunningMode = vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=POSE_MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_poses=1
)

landmarker = PoseLandmarker.create_from_options(options)


# ==========================================
# OPEN WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    landmarker.close()
    raise SystemExit

print()
print("==========================================")
print(" REAL-TIME RANDOM FOREST")
print("==========================================")
print("Webcam started.")
print("Press Q to quit.")


frame_timestamp = 0


# ==========================================
# REAL-TIME LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break

    # Flip frame for mirror view
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    frame_timestamp += 33

    # Detect pose landmarks
    result = landmarker.detect_for_video(
        mp_image,
        frame_timestamp
    )


    # ======================================
    # IF POSE IS DETECTED
    # ======================================

    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]

        features = []

        for landmark in landmarks:

            features.extend([
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility
            ])


        # Make sure there are exactly 132 features
        if len(features) == 132:

            # Use the same feature names as training
            if hasattr(model, "feature_names_in_"):

                feature_names = model.feature_names_in_

                input_data = pd.DataFrame(
                    [features],
                    columns=feature_names
                )

            else:

                input_data = np.array(
                    [features],
                    dtype=np.float32
                )


            # ==================================
            # PREDICT ACTION
            # ==================================

            prediction = model.predict(input_data)[0]


            # Prediction probability
            try:

                probabilities = model.predict_proba(
                    input_data
                )[0]

                confidence = np.max(
                    probabilities
                ) * 100

            except Exception:

                confidence = 0.0


            # ==================================
            # DISPLAY RESULT
            # ==================================

            text = f"Action: {prediction}"

            confidence_text = (
                f"Confidence: {confidence:.1f}%"
            )

            cv2.putText(
                frame,
                text,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                confidence_text,
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


        # ==================================
        # DRAW POSE LANDMARKS
        # ==================================

        for landmark in landmarks:

            x = int(
                landmark.x * frame.shape[1]
            )

            y = int(
                landmark.y * frame.shape[0]
            )

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )


    else:

        cv2.putText(
            frame,
            "No person detected",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )


    # ======================================
    # SHOW WEBCAM
    # ======================================

    cv2.imshow(
        "Human Action Recognition - Random Forest",
        frame
    )


    # ======================================
    # QUIT
    # ======================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()
cv2.destroyAllWindows()
landmarker.close()

print()
print("==========================================")
print(" REAL-TIME PREDICTION CLOSED")
print("==========================================")