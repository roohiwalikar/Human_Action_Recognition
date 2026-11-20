import cv2
import mediapipe as mp
import urllib.request
import os

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/1/"
    "pose_landmarker_full.task"
)

MODEL_FILE = "pose_landmarker_full.task"


# Download the MediaPipe pose model if it is not already present
if not os.path.exists(MODEL_FILE):
    print("Downloading MediaPipe Pose model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    print("Pose model downloaded successfully.")


# Create MediaPipe Pose Landmarker
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_FILE
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1
)


landmarker = PoseLandmarker.create_from_options(options)


# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    landmarker.close()
    exit()


print("Webcam started.")
print("Press Q to close the camera.")

timestamp = 0


while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break

    # Mirror the webcam
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect pose
    timestamp += 33

    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )

    # Draw landmarks
    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]

        # Draw points
        for landmark in landmarks:

            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )

        # Draw simple skeleton connections
        connections = [
            (11, 12),
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),
            (11, 23),
            (12, 24),
            (23, 24),
            (23, 25),
            (25, 27),
            (24, 26),
            (26, 28)
        ]

        for start, end in connections:

            x1 = int(landmarks[start].x * frame.shape[1])
            y1 = int(landmarks[start].y * frame.shape[0])

            x2 = int(landmarks[end].x * frame.shape[1])
            y2 = int(landmarks[end].y * frame.shape[0])

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

    cv2.putText(
        frame,
        "Human Action Recognition - Pose Detection",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Human Action Recognition",
        frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
landmarker.close()
cv2.destroyAllWindows()

print("Webcam detector closed.")