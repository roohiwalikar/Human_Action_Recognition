import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ==========================================
# HUMAN ACTION RECOGNITION
# RANDOM FOREST ML CLASSIFIER
# ==========================================

DATASET_FOLDER = "dataset"
MODEL_FOLDER = "models"

# Create models folder if it does not exist
os.makedirs(MODEL_FOLDER, exist_ok=True)

# List of dataset files
files = [
    "standing.csv",
    "sitting.csv",
    "hands_up.csv",
    "walking.csv",
    "jumping.csv",
    "waving.csv"
]

print("==========================================")
print(" RANDOM FOREST TRAINING STARTED")
print("==========================================")

train_data = []
test_data = []

# Read each action separately
for file in files:

    path = os.path.join(DATASET_FOLDER, file)

    if not os.path.exists(path):
        print("ERROR: File not found:", path)
        continue

    df = pd.read_csv(path)

    # Use first 80% for training
    # Use last 20% for testing
    split_index = int(len(df) * 0.8)

    train_part = df.iloc[:split_index]
    test_part = df.iloc[split_index:]

    train_data.append(train_part)
    test_data.append(test_part)

    print(
        file,
        "->",
        len(train_part),
        "training frames,",
        len(test_part),
        "testing frames"
    )

# Combine all actions
train_df = pd.concat(train_data, ignore_index=True)
test_df = pd.concat(test_data, ignore_index=True)

# Feature columns
feature_columns = [
    column for column in train_df.columns
    if column != "label"
]

X_train = train_df[feature_columns]
y_train = train_df["label"]

X_test = test_df[feature_columns]
y_test = test_df["label"]

print()
print("Training samples :", len(X_train))
print("Testing samples  :", len(X_test))
print("Features per frame:", len(feature_columns))

# ==========================================
# CREATE RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

print()
print("Training Random Forest...")
model.fit(X_train, y_train)

print("Training completed.")

# ==========================================
# PREDICTION
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# ACCURACY
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print()
print("==========================================")
print(" RANDOM FOREST RESULTS")
print("==========================================")

print("Accuracy:", round(accuracy * 100, 2), "%")

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print()
print("Classification Report:")
print("------------------------------------------")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# ==========================================
# SAVE MODEL
# ==========================================

model_path = os.path.join(
    MODEL_FOLDER,
    "random_forest_model.pkl"
)

joblib.dump(model, model_path)

print("==========================================")
print(" MODEL SAVED SUCCESSFULLY")
print("==========================================")

print("Saved as:", model_path)

print()
print("Random Forest training finished.")