import os
import pandas as pd
import numpy as np

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    VotingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib
import matplotlib.pyplot as plt


# ============================================================
# 9-CLASS ENSEMBLE RANDOM FOREST TRAINING
# ============================================================

TRAIN_FOLDER = "master_dataset_train"
TEST_FOLDER = "master_dataset_test"

MODEL_FOLDER = "models"
MODEL_FILE = os.path.join(
    MODEL_FOLDER,
    "master_random_forest_9class.pkl"
)

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

EXPECTED_FEATURES = 132

os.makedirs(MODEL_FOLDER, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(folder):

    X_parts = []
    y_parts = []

    print()
    print("=" * 70)
    print(f"LOADING DATA FROM: {folder}")
    print("=" * 70)

    for action in CLASSES:

        file_path = os.path.join(
            folder,
            action + ".csv"
        )

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Missing file: {file_path}"
            )

        df = pd.read_csv(file_path)

        # Remove label column if present
        if "label" in df.columns:
            X = df.drop(columns=["label"])
        else:
            X = df

        # Convert everything to numeric
        X = X.apply(
            pd.to_numeric,
            errors="coerce"
        )

        # Remove invalid values
        X = X.replace(
            [np.inf, -np.inf],
            0
        ).fillna(0)

        # Verify feature count
        if X.shape[1] != EXPECTED_FEATURES:
            raise ValueError(
                f"{file_path}: expected "
                f"{EXPECTED_FEATURES} features, "
                f"found {X.shape[1]}"
            )

        y = np.full(
            len(X),
            action
        )

        X_parts.append(X)
        y_parts.append(y)

        print(
            f"{action:12s} -> {len(X):5d} rows"
        )

    X_final = pd.concat(
        X_parts,
        ignore_index=True
    )

    y_final = np.concatenate(
        y_parts
    )

    return X_final, y_final


# ============================================================
# LOAD TRAINING DATA
# ============================================================

X_train, y_train = load_dataset(
    TRAIN_FOLDER
)

print()
print(
    f"TOTAL TRAINING SAMPLES : {len(X_train)}"
)
print(
    f"TRAINING FEATURES      : {X_train.shape[1]}"
)


# ============================================================
# LOAD TEST DATA
# ============================================================

X_test, y_test = load_dataset(
    TEST_FOLDER
)

print()
print(
    f"TOTAL TESTING SAMPLES  : {len(X_test)}"
)
print(
    f"TESTING FEATURES       : {X_test.shape[1]}"
)


# ============================================================
# MODEL 1 - RANDOM FOREST
# ============================================================

print()
print("=" * 70)
print("CREATING MODEL 1 - RANDOM FOREST")
print("=" * 70)

rf1 = RandomForestClassifier(
    n_estimators=700,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    class_weight=None,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# MODEL 2 - EXTRA TREES
# ============================================================

print()
print("=" * 70)
print("CREATING MODEL 2 - EXTRA TREES")
print("=" * 70)

extra = ExtraTreesClassifier(
    n_estimators=700,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=False,
    class_weight=None,
    random_state=123,
    n_jobs=-1
)


# ============================================================
# MODEL 3 - SECOND RANDOM FOREST
# ============================================================

print()
print("=" * 70)
print("CREATING MODEL 3 - RANDOM FOREST VARIANT")
print("=" * 70)

rf2 = RandomForestClassifier(
    n_estimators=700,
    criterion="entropy",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=0.5,
    bootstrap=True,
    class_weight=None,
    random_state=999,
    n_jobs=-1
)


# ============================================================
# MODEL 4 - KNN
# ============================================================

print()
print("=" * 70)
print("CREATING MODEL 4 - KNN")
print("=" * 70)

knn = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        ),
        (
            "knn",
            KNeighborsClassifier(
                n_neighbors=7,
                weights="distance",
                metric="minkowski",
                p=2,
                n_jobs=-1
            )
        )
    ]
)


# ============================================================
# ENSEMBLE
# ============================================================

print()
print("=" * 70)
print("CREATING ENSEMBLE MODEL")
print("=" * 70)
print()

model = VotingClassifier(
    estimators=[
        ("rf1", rf1),
        ("extra", extra),
        ("rf2", rf2),
        ("knn", knn)
    ],
    voting="soft",
    weights=[
        3,
        3,
        2,
        1
    ],
    n_jobs=-1
)


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 70)
print("TRAINING 9-CLASS ENSEMBLE")
print("=" * 70)
print()

print("This may take a few minutes...")
print()

model.fit(
    X_train,
    y_train
)

print()
print("ENSEMBLE TRAINING COMPLETE")


# ============================================================
# PREDICTION
# ============================================================

print()
print("=" * 70)
print("EVALUATING MODEL")
print("=" * 70)
print()

y_pred = model.predict(
    X_test
)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print()
print("=" * 70)
print("FINAL 9-CLASS ACCURACY")
print("=" * 70)
print()

print(
    f"TEST ACCURACY: {accuracy * 100:.2f}%"
)

print()


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)
print()

print(
    classification_report(
        y_test,
        y_pred,
        labels=CLASSES,
        target_names=CLASSES,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=CLASSES
)

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)
print()

print(cm)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)

print()
print("=" * 70)
print("MODEL SAVED")
print("=" * 70)
print()

print(
    f"Saved to: {MODEL_FILE}"
)


# ============================================================
# SAVE CONFUSION MATRIX IMAGE
# ============================================================

plt.figure(
    figsize=(11, 9)
)

plt.imshow(cm)

plt.title(
    "9-Class Ensemble Random Forest Confusion Matrix"
)

plt.xticks(
    range(len(CLASSES)),
    CLASSES,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(CLASSES)),
    CLASSES
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    "master_random_forest_confusion_matrix.png",
    dpi=300
)

plt.close()

print(
    "Confusion matrix saved as:"
)

print(
    "master_random_forest_confusion_matrix.png"
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("9-CLASS ENSEMBLE RANDOM FOREST COMPLETE")
print("=" * 70)
print()