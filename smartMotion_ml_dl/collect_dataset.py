import cv2
import mediapipe as mp
import csv
import os
import time
import urllib.request


# -----------------------------
# Settings
# -----------------------------

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/1/"
    "pose_landmarker_full.task"
)

MODEL_FILE = "pose_landmarker_full.task"

DATASET_FOLDER = "dataset"

ACTIONS = [
    "standing",
    "sitting",
    "hands_up",
    "walking",
    "jumping",
    "waving"
]

FRAMES_PER_ACTION = 600


# -----------------------------
# Create dataset folder
# -----------------------------

os.makedirs(DATASET_FOLDER, exist_ok=True)


# -----------------------------
# Download pose model
# -----------------------------

if not os.path.exists(MODEL_FILE):
    print("Downloading MediaPipe Pose model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    print("Pose model downloaded successfully.")


# -----------------------------
# MediaPipe Pose Landmarker
# -----------------------------

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


# -----------------------------
# Open webcam
# -----------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    landmarker.close()
    exit()


print()
print("======================================")
print(" HUMAN ACTION DATASET COLLECTION")
print("======================================")
print()
print("Actions to collect:")

for action in ACTIONS:
    print("-", action)

print()
print("You will collect 300 frames for each action.")
print("Follow the action shown on the screen.")
print("Press Q at any time to stop.")
print()


timestamp = 0


# -----------------------------
# Collect data
# -----------------------------

for action in ACTIONS:

    print()
    print("--------------------------------------")
    print("Get ready for:", action)
    print("--------------------------------------")

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

        cv2.imshow(
            "Dataset Collection",
            frame
        )

        cv2.waitKey(1000)


    filename = os.path.join(
        DATASET_FOLDER,
        action + ".csv"
    )


    # CSV header
    header = ["label"]

    for i in range(33):
        header.extend([
            f"x_{i}",
            f"y_{i}",
            f"z_{i}",
            f"visibility_{i}"
        ])


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

        while frame_count < FRAMES_PER_ACTION:

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
                f"Frames: {frame_count}/{FRAMES_PER_ACTION}",
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
        "completed:",
        frame_count,
        "frames saved."
    )


# -----------------------------
# Finish
# -----------------------------

cap.release()
landmarker.close()
cv2.destroyAllWindows()

print()
print("======================================")
print(" DATASET COLLECTION COMPLETED")
print("======================================")
print()