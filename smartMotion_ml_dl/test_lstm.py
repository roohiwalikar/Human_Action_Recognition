import os
import numpy as np

from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# HUMAN ACTION RECOGNITION
# LSTM - NEW TEST DATA
# ==========================================

MODEL_PATH = os.path.join("models", "lstm_action_model.keras")
CLASSES_PATH = os.path.join("models", "lstm_classes.npy")
TEST_FOLDER = "test_dataset"

SEQUENCE_LENGTH = 30

ACTIONS = [
    "standing",
    "sitting",
    "hands_up",
    "walking",
    "jumping",
    "waving"
]

print("==========================================")
print(" LSTM - NEW TEST DATA")
print("==========================================")

# Load trained LSTM model
model = load_model(MODEL_PATH)

# Load class names
classes = np.load(CLASSES_PATH, allow_pickle=True)

print("LSTM model loaded successfully.")

# ==========================================
# CREATE TEST SEQUENCES
# ==========================================

X_test = []
y_test = []

for action in ACTIONS:

    filename = os.path.join(TEST_FOLDER, f"{action}.csv")

    if not os.path.exists(filename):
        print("ERROR: File not found:", filename)
        continue

    # Load CSV
    data = np.loadtxt(
        filename,
        delimiter=",",
        skiprows=1
    )

    # Create 30-frame sequences
    for i in range(len(data) - SEQUENCE_LENGTH + 1):

        sequence = data[i:i + SEQUENCE_LENGTH]

        X_test.append(sequence)
        y_test.append(action)

X_test = np.array(X_test)
y_test = np.array(y_test)

print()
print("Test sequences:", len(X_test))
print("Sequence shape:", X_test.shape)

# ==========================================
# PREDICTION
# ==========================================

print()
print("Predicting test sequences...")

predictions = model.predict(X_test, verbose=0)

predicted_indices = np.argmax(predictions, axis=1)

predicted_labels = classes[predicted_indices]

# ==========================================
# ACCURACY
# ==========================================

accuracy = accuracy_score(y_test, predicted_labels)

print()
print("==========================================")
print(" LSTM TEST RESULTS")
print("==========================================")

print(f"Accuracy: {accuracy * 100:.2f}%")

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print()
print("Classification Report:")
print("------------------------------------------")

print(
    classification_report(
        y_test,
        predicted_labels,
        labels=ACTIONS,
        zero_division=0
    )
)

# ==========================================
# CONFUSION MATRIX
# ==========================================

print()
print("Confusion Matrix:")
print("------------------------------------------")

print(
    confusion_matrix(
        y_test,
        predicted_labels,
        labels=ACTIONS
    )
)

print()
print("==========================================")
print(" LSTM TESTING COMPLETED")
print("==========================================")