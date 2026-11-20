import os
import pandas as pd
import numpy as np

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

TRAIN_FOLDER = "master_dataset_train"
TEST_FOLDER = "master_dataset_test"

print()
print("=" * 70)
print("MASTER DATASET DIAGNOSTIC CHECK")
print("=" * 70)

for action in CLASSES:

    train_file = os.path.join(
        TRAIN_FOLDER,
        action + ".csv"
    )

    test_file = os.path.join(
        TEST_FOLDER,
        action + ".csv"
    )

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    X_train = train_df.drop(columns=["label"])
    X_test = test_df.drop(columns=["label"])

    print()
    print("-" * 70)
    print(action.upper())
    print("-" * 70)

    print("Train shape :", X_train.shape)
    print("Test shape  :", X_test.shape)

    print(
        "Train mean  :",
        round(X_train.to_numpy().mean(), 6)
    )

    print(
        "Test mean   :",
        round(X_test.to_numpy().mean(), 6)
    )

    print(
        "Train std   :",
        round(X_train.to_numpy().std(), 6)
    )

    print(
        "Test std    :",
        round(X_test.to_numpy().std(), 6)
    )

    print(
        "Min         :",
        round(X_train.to_numpy().min(), 6)
    )

    print(
        "Max         :",
        round(X_train.to_numpy().max(), 6)
    )

print()
print("=" * 70)
print("CHECKING FEATURE COLUMN CONSISTENCY")
print("=" * 70)

reference_file = os.path.join(
    TRAIN_FOLDER,
    CLASSES[0] + ".csv"
)

reference_df = pd.read_csv(reference_file)

reference_columns = list(
    reference_df.drop(columns=["label"]).columns
)

print()
print("Expected feature count:", len(reference_columns))
print("First 10 features:")
print(reference_columns[:10])

print()

all_correct = True

for folder in [TRAIN_FOLDER, TEST_FOLDER]:

    for action in CLASSES:

        file_path = os.path.join(
            folder,
            action + ".csv"
        )

        df = pd.read_csv(file_path)

        columns = list(
            df.drop(columns=["label"]).columns
        )

        if columns != reference_columns:

            print(
                "❌ COLUMN MISMATCH:",
                folder,
                action
            )

            all_correct = False

print()

if all_correct:
    print("✅ ALL FEATURE COLUMNS MATCH EXACTLY")
else:
    print("❌ FEATURE COLUMN MISMATCH FOUND")

print()
print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)