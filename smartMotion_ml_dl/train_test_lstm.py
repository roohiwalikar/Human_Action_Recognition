import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping


# ---------------------------------------
# Settings
# ---------------------------------------
DATASET_FOLDER = "combined_dataset"

ACTIONS = [
    "standing",
    "sitting",
    "hands_up",
    "walking",
    "jumping",
    "waving"
]

SEQUENCE_LENGTH = 30

# Use 75% for training and 25% for testing
TRAIN_RATIO = 0.75


# ---------------------------------------
# Load data
# ---------------------------------------
print("=" * 60)
print("LOADING COMBINED DATASET")
print("=" * 60)

all_sequences = []
all_labels = []

for action in ACTIONS:

    file_path = os.path.join(
        DATASET_FOLDER,
        action + ".csv"
    )

    df = pd.read_csv(file_path)

    # Remove label column if present
    feature_columns = [
        column for column in df.columns
        if column != "label"
    ]

    df[feature_columns] = df[feature_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df[feature_columns] = df[feature_columns].fillna(0)

    data = df[feature_columns].values.astype(
        np.float32
    )

    total_frames = len(data)

    train_frames = int(
        total_frames * TRAIN_RATIO
    )

    test_frames = total_frames - train_frames

    print(
        f"{action}: "
        f"{total_frames} frames -> "
        f"{train_frames} train + "
        f"{test_frames} test"
    )

    # ---------------------------------------
    # Create training sequences
    # ---------------------------------------
    train_data = data[:train_frames]

    for i in range(
        len(train_data) - SEQUENCE_LENGTH + 1
    ):

        sequence = train_data[
            i:i + SEQUENCE_LENGTH
        ]

        all_sequences.append(sequence)
        all_labels.append(action)


# ---------------------------------------
# Convert training data
# ---------------------------------------
X_train = np.array(
    all_sequences,
    dtype=np.float32
)

y_train_text = np.array(
    all_labels
)

print()
print("Training sequences:", X_train.shape)


# ---------------------------------------
# Load test sequences separately
# ---------------------------------------
test_sequences = []
test_labels = []

for action in ACTIONS:

    file_path = os.path.join(
        DATASET_FOLDER,
        action + ".csv"
    )

    df = pd.read_csv(file_path)

    feature_columns = [
        column for column in df.columns
        if column != "label"
    ]

    df[feature_columns] = df[feature_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df[feature_columns] = df[feature_columns].fillna(0)

    data = df[feature_columns].values.astype(
        np.float32
    )

    train_frames = int(
        len(data) * TRAIN_RATIO
    )

    test_data = data[train_frames:]

    for i in range(
        len(test_data) - SEQUENCE_LENGTH + 1
    ):

        sequence = test_data[
            i:i + SEQUENCE_LENGTH
        ]

        test_sequences.append(sequence)
        test_labels.append(action)


X_test = np.array(
    test_sequences,
    dtype=np.float32
)

y_test_text = np.array(
    test_labels
)

print("Testing sequences:", X_test.shape)


# ---------------------------------------
# Encode labels
# ---------------------------------------
label_encoder = LabelEncoder()

label_encoder.fit(ACTIONS)

y_train = label_encoder.transform(
    y_train_text
)

y_test = label_encoder.transform(
    y_test_text
)


# ---------------------------------------
# Build LSTM model
# ---------------------------------------
print()
print("=" * 60)
print("BUILDING LSTM MODEL")
print("=" * 60)

model = Sequential([

    LSTM(
        64,
        return_sequences=True,
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    ),

    Dropout(0.2),

    LSTM(32),

    Dropout(0.2),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        len(ACTIONS),
        activation="softmax"
    )
])


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# ---------------------------------------
# Train
# ---------------------------------------
print()
print("=" * 60)
print("TRAINING LSTM")
print("=" * 60)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.20,
    callbacks=[early_stopping],
    verbose=1
)


# ---------------------------------------
# Test model
# ---------------------------------------
print()
print("=" * 60)
print("LSTM TEST RESULTS")
print("=" * 60)

predictions = model.predict(
    X_test,
    verbose=0
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

accuracy = accuracy_score(
    y_test,
    predicted_classes
)

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)


# ---------------------------------------
# Classification report
# ---------------------------------------
print()
print("Classification Report:")
print()

print(
    classification_report(
        y_test,
        predicted_classes,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# ---------------------------------------
# Confusion matrix
# ---------------------------------------
print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predicted_classes
    )
)


# ---------------------------------------
# Save model
# ---------------------------------------
os.makedirs(
    "models",
    exist_ok=True
)

model.save(
    "models/lstm_proper_model.keras"
)

np.save(
    "models/lstm_proper_classes.npy",
    label_encoder.classes_
)


print()
print("=" * 60)
print("LSTM MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print(
    "Saved model  : models/lstm_proper_model.keras"
)

print(
    "Saved classes: models/lstm_proper_classes.npy"
)

print()
print("PROPER LSTM TRAINING AND TESTING COMPLETED")
print("=" * 60)