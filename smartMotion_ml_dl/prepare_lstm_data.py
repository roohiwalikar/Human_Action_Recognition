import os
import numpy as np
import pandas as pd

# ==========================================
# HUMAN ACTION RECOGNITION
# 9-CLASS LSTM SEQUENCE DATA PREPARATION
# ==========================================

SEQUENCE_LENGTH = 30
OUTPUT_FOLDER = "lstm_data"

TRAIN_FOLDERS = [
    "dataset",
    "new_dataset_train"
]

TEST_FOLDERS = [
    "test_dataset",
    "new_dataset_test"
]

ACTIONS = [
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

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==========================================
# CREATE SEQUENCES
# ==========================================

def create_sequences(folders):

    all_sequences = []
    all_labels = []

    print()
    print("=" * 70)
    print("CREATING LSTM SEQUENCES")
    print("=" * 70)

    for folder in folders:

        print()
        print("FOLDER:", folder)

        for action in ACTIONS:

            file_path = os.path.join(
                folder,
                action + ".csv"
            )

            # Skip actions that do not exist in this folder
            if not os.path.exists(file_path):
                continue

            df = pd.read_csv(file_path)

            feature_columns = [
                column
                for column in df.columns
                if column != "label"
            ]

            data = df[feature_columns].values

            data = np.nan_to_num(
                data,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            print(
                f"{action:12s} -> {len(data):4d} frames"
            )

            # Create 30-frame sequences
            sequence_count = 0

            for start in range(
                0,
                len(data) - SEQUENCE_LENGTH + 1
            ):

                end = start + SEQUENCE_LENGTH

                sequence = data[start:end]

                all_sequences.append(sequence)
                all_labels.append(action)

                sequence_count += 1

            print(
                f"{'':12s}    {sequence_count:4d} sequences"
            )

    X = np.array(
        all_sequences,
        dtype=np.float32
    )

    y = np.array(
        all_labels
    )

    return X, y


# ==========================================
# TRAINING SEQUENCES
# ==========================================

print()
print("PREPARING TRAINING SEQUENCES")

X_train, y_train = create_sequences(
    TRAIN_FOLDERS
)

print()
print("TRAINING DATA SHAPE:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)


# ==========================================
# TEST SEQUENCES
# ==========================================

print()
print("PREPARING TEST SEQUENCES")

X_test, y_test = create_sequences(
    TEST_FOLDERS
)

print()
print("TEST DATA SHAPE:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ==========================================
# SAVE
# ==========================================

np.save(
    os.path.join(
        OUTPUT_FOLDER,
        "X_train_sequences.npy"
    ),
    X_train
)

np.save(
    os.path.join(
        OUTPUT_FOLDER,
        "y_train_labels.npy"
    ),
    y_train
)

np.save(
    os.path.join(
        OUTPUT_FOLDER,
        "X_test_sequences.npy"
    ),
    X_test
)

np.save(
    os.path.join(
        OUTPUT_FOLDER,
        "y_test_labels.npy"
    ),
    y_test
)


# ==========================================
# FINAL INFORMATION
# ==========================================

print()
print("=" * 70)
print("9-CLASS LSTM DATA PREPARATION COMPLETE")
print("=" * 70)

print()
print("Training sequences :", len(X_train))
print("Testing sequences  :", len(X_test))

print()
print("Sequence shape:", X_train.shape)

print()
print("Files saved:")
print("lstm_data/X_train_sequences.npy")
print("lstm_data/y_train_labels.npy")
print("lstm_data/X_test_sequences.npy")
print("lstm_data/y_test_labels.npy")

print()
print("=" * 70)