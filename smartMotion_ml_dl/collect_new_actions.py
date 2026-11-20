import cv2
import mediapipe as mp
import csv
import os
import time
import urllib.request


# ==========================================
# SETTINGS
# ==========================================

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/1/"
    "pose_landmarker_full.task"
)

MODEL_FILE = "pose_landmarker_full.task"

TRAIN_FOLDER = "new_dataset_train"
TEST_FOLDER = "new_dataset_test"

ACTIONS = [
    "reading",
    "dancing",
    "lying_down"
]

TRAIN_FRAMES = 300
TEST_FRAMES = 300


# ==========================================
# CREATE FOLDERS
# ==========================================

os.makedirs(TRAIN_FOLDER, exist_ok=True)
os.makedirs(TEST_FOLDER, exist_ok=True)


# ==========================================
# DOWNLOAD MODEL IF NEEDED
# ==========================================

if not os.path.exists(MODEL_FILE):
    print("Downloading MediaPipe Pose model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    print("Pose model downloaded successfully.")


# ==========================================
# MEDIAPIPE POSE LANDMARKER
# ==========================================

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
RunningMode = vision.RunningMode


options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_FILE
    ),
    running_mode=RunningMode.VIDEO,
    num_poses=1
)

landmarker = PoseLandmarker.create_from_options(options)


# ==========================================
# WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    landmarker.close()
    exit()


# ==========================================
# FUNCTION TO COLLECT DATA
# ==========================================

def collect_action(action, folder, frames_to_collect, session_name):

    print()
    print("======================================")
    print("ACTION:", action.upper())
    print("SESSION:", session_name)
    print("======================================")

    input("Press ENTER when you are ready...")

    # Countdown
    for countdown in [3, 2, 1]:

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        cv2.putText(
            frame,
            action.upper(),
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            "Starting in: " + str(countdown),
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Dataset Collection", frame)
        cv2.waitKey(1000)


    filename = os.path.join(
        folder,
        action + ".csv"
    )

    header = ["label"]

    for i in range(33):
        header.extend([
            f"x_{i}",
            f"y_{i}",
            f"z_{i}",
            f"visibility_{i}"
        ])


    # Use append mode so 3 training sessions
    # can build up to 900 frames.

    file_exists = os.path.exists(filename)

    with open(
        filename,
        "a",
        newline=""
    ) as csvfile:

        writer = csv.writer(csvfile)

        if not file_exists:
            writer.writerow(header)

        frame_count = 0
        timestamp = int(time.time() * 1000)

        while frame_count < frames_to_collect:

            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp += 33

            result = landmarker.detect_for_video(
                mp_image,
                timestamp
            )

            if result.pose_landmarks:

                landmarks = result.pose_landmarks[0]

                row = [action]

                for landmark in landmarks:

                    row.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z,
                        landmark.visibility
                    ])

                writer.writerow(row)

                frame_count += 1

                # Draw landmarks
                for landmark in landmarks:

                    x = int(
                        landmark.x * frame.shape[1]
                    )

                    y = int(
                        landmark.y * frame.shape[0]
                    )

                    if (
                        0 <= x < frame.shape[1]
                        and 0 <= y < frame.shape[0]
                    ):
                        cv2.circle(
                            frame,
                            (x, y),
                            3,
                            (0, 255, 0),
                            -1
                        )


            cv2.putText(
                frame,
                "Action: " + action,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Frames: {frame_count}/{frames_to_collect}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Press Q to stop",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.imshow(
                "Dataset Collection",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


    print(
        action,
        session_name,
        "completed:",
        frame_count,
        "frames saved."
    )


# ==========================================
# TRAINING DATA
# 3 sessions × 300 = 900 frames/action
# ==========================================

for session in range(1, 4):

    print()
    print("######################################")
    print(" TRAINING SESSION", session)
    print("######################################")

    for action in ACTIONS:

        collect_action(
            action,
            TRAIN_FOLDER,
            TRAIN_FRAMES,
            f"Training {session}"
        )


# ==========================================
# TEST DATA
# One separate session = 300 frames/action
# ==========================================

print()
print("######################################")
print(" TEST DATA COLLECTION")
print("######################################")

print()
print("IMPORTANT:")
print("For TEST DATA, change your position")
print("or distance from the camera.")
print("Do not simply repeat the training setup.")
print()

for action in ACTIONS:

    collect_action(
        action,
        TEST_FOLDER,
        TEST_FRAMES,
        "Test"
    )


# ==========================================
# FINISH
# ==========================================

cap.release()
landmarker.close()
cv2.destroyAllWindows()

print()
print("======================================")
print(" NEW ACTION DATA COLLECTION COMPLETE")
print("======================================")
print()
print("Training folder:", TRAIN_FOLDER)
print("Testing folder:", TEST_FOLDER)
print()