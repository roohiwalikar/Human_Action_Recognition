import os
import csv
import numpy as np
import joblib

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Paths
MODEL_PATH = os.path.join("models", "random_forest_model.pkl")
TEST_FOLDER = "test_dataset"

# Load trained Random Forest model
model = joblib.load(MODEL_PATH)
print("======================================")
print(" RANDOM FOREST - NEW TEST DATA")
print("======================================")

X_test = []
y_test = []

# Same action names used during training
ACTIONS = [
    "standing",
    "sitting",
    "hands_up",
    "walking",
    "jumping",
    "waving"
]

# Read new test data
for action in ACTIONS:

    filename = os.path.join(TEST_FOLDER, f"{action}.csv")

    with open(filename, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)

        # Skip header
        next(reader)

        for row in reader:
            if len(row) == 132:
                X_test.append([float(value) for value in row])
                y_test.append(action)

X_test = np.array(X_test)
y_test = np.array(y_test)

print("Test samples:", len(X_test))
print("Features per frame:", X_test.shape[1])

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("")
print("======================================")
print(" RANDOM FOREST TEST RESULTS")
print("======================================")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("")
print("Classification Report:")
print(classification_report(y_test, y_pred, labels=ACTIONS))

print("")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=ACTIONS))

print("")
print("======================================")
print(" TESTING COMPLETED")
print("======================================")