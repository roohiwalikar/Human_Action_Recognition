import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ==========================================
# HUMAN ACTION RECOGNITION
# COMBINED RANDOM FOREST ML CLASSIFIER
# ==========================================

DATASET_FOLDER = "combined_dataset"
MODEL_FOLDER = "models"

os.makedirs(MODEL_FOLDER, exist_ok=True)

files = [
    "standing.csv",
    "sitting.csv",
    "hands_up.csv",
    "walking.csv",
    "jumping.csv",
    "waving.csv"
]

print("==========================================")
print(" COMBINED RANDOM FOREST TRAINING")
print("==========================================")

all_data = []

# ==========================================
# READ COMBINED DATASET
# ==========================================

for file in files:

    path = os.path.join(DATASET_FOLDER, file)

    if not os.path.exists(path):
        print("ERROR: File not found:", path)
        continue

    df = pd.read_csv(path)

    # Feature columns
    feature_columns = [
        column for column in df.columns
        if column != "label"
    ]

    # Convert feature values to numbers
    df[feature_columns] = df[feature_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Replace missing feature values with 0
    df[feature_columns] = df[feature_columns].fillna(0)

    # Replace infinite values with 0
    df[feature_columns] = df[feature_columns].replace(
        [np.inf, -np.inf],
        0
    )

    all_data.append(df)

    print(file, "->", len(df), "frames loaded")

# ==========================================
# COMBINE ALL ACTIONS
# ==========================================

data = pd.concat(all_data, ignore_index=True)

feature_columns = [
    column for column in data.columns
    if column != "label"
]

# ==========================================
# CLEAN DATA BEFORE TRAINING
# ==========================================

# Remove rows where the action label is missing
data = data.dropna(subset=["label"])

X = data[feature_columns].copy()
y = data["label"].copy()

# Convert all features to numeric values
X = X.apply(pd.to_numeric, errors="coerce")

# Replace missing values with 0
X = X.fillna(0)

# Replace infinite values with 0
X = X.replace([np.inf, -np.inf], 0)

print()
print("Missing feature values after cleaning:", X.isna().sum().sum())
print("Missing labels after cleaning:", y.isna().sum())

print()
print("Total training samples :", len(X))
print("Features per frame     :", len(feature_columns))

# ==========================================
# TRAIN RANDOM FOREST
# ==========================================

print()
print("Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

print("Training completed.")

# ==========================================
# TRAINING ACCURACY
# ==========================================

y_pred = model.predict(X)

accuracy = accuracy_score(y, y_pred)

print()
print("==========================================")
print(" COMBINED RANDOM FOREST RESULTS")
print("==========================================")

print("Training Accuracy:", round(accuracy * 100, 2), "%")

print()
print("Classification Report:")
print("------------------------------------------")

print(
    classification_report(
        y,
        y_pred,
        zero_division=0
    )
)

# ==========================================
# SAVE MODEL
# ==========================================

model_path = os.path.join(
    MODEL_FOLDER,
    "random_forest_combined_model.pkl"
)

joblib.dump(model, model_path)

print("==========================================")
print(" MODEL SAVED SUCCESSFULLY")
print("==========================================")

print("Saved as:", model_path)

print()
print("Combined Random Forest training finished.")