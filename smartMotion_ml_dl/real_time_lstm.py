import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = "models/lstm_action_model_9class.keras"
CLASSES_PATH = "models/lstm_classes_9class.npy"
POSE_MODEL_PATH = "pose_landmarker_full.task"


# ============================================================
# LOAD 9-CLASS LSTM MODEL
# ============================================================

model = load_model(MODEL_PATH)
classes = np.load(CLASSES_PATH, allow_pickle=True)

print("=" * 60)
print("9-CLASS LSTM MODEL LOADED")
print("=" * 60)
print("Model:", MODEL_PATH)
print("Classes:")

for i, action in enumerate(classes):
    print(f"{i} -> {action}")

print("=" * 60)


# ============================================================
# MEDIAPIPE POSE LANDMARKER
# ============================================================

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
RunningMode = vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=POSE_MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO
)

landmarker = PoseLandmarker.create_from_options(options)


# ============================================================
# SEQUENCE SETTINGS
# ============================================================

SEQUENCE_LENGTH = 30

sequence = deque(maxlen=SEQUENCE_LENGTH)

timestamp = 0


# ============================================================
# START WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam started.")
print("Perform an action in front of the camera.")
print("Press Q to quit.")


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    # --------------------------------------------------------
    # Convert BGR → RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Create MediaPipe image
    # --------------------------------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # --------------------------------------------------------
    # Detect pose
    # --------------------------------------------------------

    timestamp += 33

    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )

    # ========================================================
    # POSE DETECTED
    # ========================================================

    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]

        # ----------------------------------------------------
        # Extract 33 landmarks × 4 features
        # = 132 features
        # ----------------------------------------------------

        features = []

        for landmark in landmarks:

            features.extend([
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility
            ])

        # ----------------------------------------------------
        # Add frame to sequence
        # ----------------------------------------------------

        sequence.append(features)

        # ----------------------------------------------------
        # Draw pose landmarks
        # ----------------------------------------------------

        for landmark in landmarks:

            x = int(
                landmark.x * frame.shape[1]
            )

            y = int(
                landmark.y * frame.shape[0]
            )

            if (
                0 <= x < frame.shape[1]
                and
                0 <= y < frame.shape[0]
            ):

                cv2.circle(
                    frame,
                    (x, y),
                    3,
                    (0, 255, 0),
                    -1
                )

        # ====================================================
        # PREDICTION
        # ====================================================

        if len(sequence) == SEQUENCE_LENGTH:

            # Convert sequence to NumPy array

            input_data = np.array(
                sequence,
                dtype=np.float32
            )

            # Shape:
            # (30, 132)
            #
            # LSTM expects:
            # (1, 30, 132)

            input_data = np.expand_dims(
                input_data,
                axis=0
            )

            # ------------------------------------------------
            # LSTM prediction
            # ------------------------------------------------

            prediction = model.predict(
                input_data,
                verbose=0
            )

            predicted_index = np.argmax(
                prediction[0]
            )

            predicted_action = classes[
                predicted_index
            ]

            confidence = float(
                prediction[0][predicted_index]
            )

            # =================================================
            # DISPLAY PREDICTION
            # =================================================

            cv2.putText(
                frame,
                f"Action: {predicted_action}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Confidence: {confidence * 100:.1f}%",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                f"Collecting frames: "
                f"{len(sequence)}/{SEQUENCE_LENGTH}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

    # ========================================================
    # NO POSE DETECTED
    # ========================================================

    else:

        cv2.putText(
            frame,
            "No pose detected",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # ========================================================
    # DISPLAY WEBCAM
    # ========================================================

    cv2.imshow(
        "Human Action Recognition - 9 Class LSTM",
        frame
    )


    # --------------------------------------------------------
    # Press Q to quit
    # --------------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()
landmarker.close()

print("Real-time 9-class LSTM prediction closed.")