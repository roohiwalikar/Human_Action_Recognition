import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 9-CLASS MASTER RANDOM FOREST TESTING
# ============================================================

MODEL_FILE = os.path.join(
    "models",
    "master_random_forest_9class.pkl"
)

TEST_FOLDER = "master_dataset_test"

CLASSES = [
    "standing",
    "sitting",
    "hands_up",
    "walking",
    "jumping",
    "waving",
    "reading",
    "dancing",
    "lying_down"
]

print()
print("=" * 70)
print("9-CLASS MASTER RANDOM FOREST TESTING")
print("=" * 70)
print()

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

if not os.path.exists(MODEL_FILE):
    print("ERROR: 9-class model not found!")
    print(MODEL_FILE)
    exit()

with open(MODEL_FILE, "rb") as f:
    model = joblib.load("models/master_random_forest_9class.pkl")

print("9-class model loaded successfully.")
print()

# ------------------------------------------------------------
# LOAD TEST DATA
# ------------------------------------------------------------

X_test = []
y_test = []

for action in CLASSES:

    test_file = os.path.join(
        TEST_FOLDER,
        action + ".csv"
    )

    if not os.path.exists(test_file):
        print(f"WARNING: Missing {action}.csv")
        continue

    df = pd.read_csv(test_file)

    feature_cols = [
        c for c in df.columns
        if c != "label"
    ]

    X_test.append(
        df[feature_cols].values
    )

    y_test.extend(
        [action] * len(df)
    )

    print(
        f"{action:12s} -> {len(df):4d} test samples"
    )

# ------------------------------------------------------------
# COMBINE TEST DATA
# ------------------------------------------------------------

X_test = pd.concat(
    [
        pd.DataFrame(x)
        for x in X_test
    ],
    ignore_index=True
).values

print()
print("Total test samples:", len(y_test))
print("Features:", X_test.shape[1])
print()

# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

y_pred = model.predict(X_test)

# ------------------------------------------------------------
# ACCURACY
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("=" * 70)
print(f"9-CLASS RANDOM FOREST ACCURACY: {accuracy * 100:.2f}%")
print("=" * 70)
print()

# ------------------------------------------------------------
# CLASSIFICATION REPORT
# ------------------------------------------------------------

print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        labels=CLASSES,
        target_names=CLASSES,
        zero_division=0
    )
)

# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=CLASSES
)

print("CONFUSION MATRIX")
print("=" * 70)
print(cm)

print()
print("=" * 70)
print("9-CLASS RANDOM FOREST TESTING COMPLETE")
print("=" * 70)