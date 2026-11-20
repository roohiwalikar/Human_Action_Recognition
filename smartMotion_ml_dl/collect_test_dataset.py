import cv2
import csv
import os
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Actions to collect
ACTIONS = [
    "standing",
    "sitting",
    "hands_up",
    "walking",
    "jumping",
    "waving"
]

FRAMES_PER_ACTION = 300

# MediaPipe model
MODEL_PATH = "pose_landmarker_full.task"

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
RunningMode = vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_poses=1
)

landmarker = PoseLandmarker.create_from_options(options)

# Create test dataset folder
os.makedirs("test_dataset", exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    landmarker.close()
    exit()

print("======================================")
print(" NEW TEST DATASET COLLECTION")
print("======================================")
print("You will collect 300 frames for each action.")
print("Perform each action naturally.")
print("Press Q to stop at any time.")
print("")

for action in ACTIONS:

    print("--------------------------------------")
    print("Get ready for:", action)
    print("Starting in 3 seconds...")
    print("--------------------------------------")

    start_time = cv2.getTickCount()

    while True:
        ret, frame = cap.read()

        if not ret:
            continue

        elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()

        remaining = 3 - int(elapsed)

        cv2.putText(
            frame,
            f"Action: {action} | Starting in {remaining}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        cv2.imshow("Test Dataset Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            landmarker.close()
            print("Collection stopped.")
            exit()

        if elapsed >= 3:
            break

    filename = os.path.join("test_dataset", f"{action}.csv")

    with open(filename, "w", newline="") as csvfile:

        writer = csv.writer(csvfile)

        # 33 landmarks × 4 values
        header = []

        for i in range(33):
            header.extend([
                f"x_{i}",
                f"y_{i}",
                f"z_{i}",
                f"visibility_{i}"
            ])

        writer.writerow(header)

        frame_count = 0

        while frame_count < FRAMES_PER_ACTION:

            ret, frame = cap.read()

            if not ret:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            timestamp_ms = int(
                cap.get(cv2.CAP_PROP_POS_MSEC)
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            if result.pose_landmarks:

                landmarks = result.pose_landmarks[0]

                row = []

                for landmark in landmarks:
                    row.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z,
                        landmark.visibility
                    ])

                writer.writerow(row)

                frame_count += 1

                cv2.putText(
                    frame,
                    f"{action}: {frame_count}/{FRAMES_PER_ACTION}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

                # Draw pose landmarks
                for landmark in landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])

                    cv2.circle(
                        frame,
                        (x, y),
                        3,
                        (0, 255, 0),
                        -1
                    )

            cv2.imshow("Test Dataset Collection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                landmarker.close()
                print("Collection stopped.")
                exit()

    print(f"{action} completed: {frame_count} frames saved.")

cap.release()
cv2.destroyAllWindows()
landmarker.close()

print("")
print("======================================")
print(" TEST DATASET COLLECTION COMPLETED")
print("======================================")
print("New test data saved in: test_dataset")