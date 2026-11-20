import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ==========================================
# HUMAN ACTION RECOGNITION USING ML AND DL
# RANDOM FOREST - PROPER TRAIN / TEST
# ==========================================

DATASET_FOLDER = "combined_dataset"
MODEL_FOLDER = "models"

os.makedirs(MODEL_FOLDER, exist_ok=True)

ACTIONS = [
    "standing",
    "sitting",
    "hands_up",
    "walking",
    "jumping",
    "waving"
]

print("==========================================")
print(" RANDOM FOREST PROPER TRAINING & TESTING")
print("==========================================")

X_train_list = []
y_train_list = []

X_test_list = []
y_test_list = []

# ==========================================
# LOAD EACH ACTION
# ==========================================

for action in ACTIONS:

    file_path = os.path.join(
        DATASET_FOLDER,
        action + ".csv"
    )

    if not os.path.exists(file_path):
        print("ERROR: File not found:", file_path)
        continue

    df = pd.read_csv(file_path)

    # Find feature columns
    feature_columns = [
        column for column in df.columns
        if column != "label"
    ]

    # Convert features to numbers
    df[feature_columns] = df[feature_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Replace missing feature values with 0
    df[feature_columns] = df[feature_columns].fillna(0)

    # Remove rows where label is missing
    df = df.dropna(subset=["label"])

    total_frames = len(df)

    print()
    print(action, "->", total_frames, "usable frames")

    # Need at least 2 frames
    if total_frames < 2:
        print("ERROR:", action, "does not have enough usable frames.")
        continue

    # ======================================
    # SPLIT DATA 50% / 50%
    # ======================================

    split_point = total_frames // 2

    train_df = df.iloc[:split_point]
    test_df = df.iloc[split_point:]

    print(
        "   Training:",
        len(train_df),
        "frames"
    )

    print(
        "   Testing :",
        len(test_df),
        "frames"
    )

    # Training data
    X_train_list.append(
        train_df[feature_columns]
    )

    y_train_list.extend(
        train_df["label"].tolist()
    )

    # Testing data
    X_test_list.append(
        test_df[feature_columns]
    )

    y_test_list.extend(
        test_df["label"].tolist()
    )

# ==========================================
# CHECK DATA
# ==========================================

if len(X_train_list) == 0 or len(X_test_list) == 0:
    print()
    print("ERROR: No training/testing data available.")
    print("Please check the combined_dataset folder.")
    exit()

# ==========================================
# COMBINE ALL ACTIONS
# ==========================================

X_train = pd.concat(
    X_train_list,
    ignore_index=True
)

X_test = pd.concat(
    X_test_list,
    ignore_index=True
)

y_train = pd.Series(y_train_list)
y_test = pd.Series(y_test_list)

# Final cleaning
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# ==========================================
# DATASET SUMMARY
# ==========================================

print()
print("==========================================")
print(" DATASET SUMMARY")
print("==========================================")

print(
    "Training samples :",
    len(X_train)
)

print(
    "Testing samples  :",
    len(X_test)
)

print(
    "Features per frame:",
    len(feature_columns)
)

print(
    "Missing training values:",
    X_train.isna().sum().sum()
)

print(
    "Missing testing values:",
    X_test.isna().sum().sum()
)

# ==========================================
# TRAIN RANDOM FOREST
# ==========================================

print()
print("==========================================")
print(" TRAINING RANDOM FOREST")
print("==========================================")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("Training completed.")

# ==========================================
# TEST ON UNSEEN DATA
# ==========================================

print()
print("Testing on unseen data...")

y_pred = model.predict(X_test)

# ==========================================
# ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print()
print("==========================================")
print(" RANDOM FOREST TEST RESULTS")
print("==========================================")

print(
    "Test Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

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
# CONFUSION MATRIX
# ==========================================

print()
print("Confusion Matrix:")
print("------------------------------------------")

labels = ACTIONS

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print("Labels:")
print(labels)

print()
print(cm)

# ==========================================
# SAVE MODEL
# ==========================================

model_path = os.path.join(
    MODEL_FOLDER,
    "random_forest_proper_model.pkl"
)

joblib.dump(
    model,
    model_path
)

print()
print("==========================================")
print(" MODEL SAVED SUCCESSFULLY")
print("==========================================")

print(
    "Saved as:",
    model_path
)

print()
print("==========================================")
print(" RANDOM FOREST TESTING COMPLETED")
print("==========================================")