import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# HUMAN ACTION RECOGNITION
# LSTM DEEP LEARNING CLASSIFIER
# ==========================================

DATA_FOLDER = "lstm_data"
MODEL_FOLDER = "models"

os.makedirs(MODEL_FOLDER, exist_ok=True)

print("==========================================")
print(" LSTM TRAINING STARTED")
print("==========================================")

# ------------------------------------------
# LOAD SEQUENCE DATA
# ------------------------------------------

X = np.load(
    os.path.join(DATA_FOLDER, "X_sequences.npy")
)

y = np.load(
    os.path.join(DATA_FOLDER, "y_labels.npy")
)

print("Loaded X shape:", X.shape)
print("Loaded labels:", y.shape)

# Convert data to float32
X = X.astype("float32")

# ------------------------------------------
# ENCODE ACTION LABELS
# ------------------------------------------

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print()
print("Action classes:")
for index, action in enumerate(label_encoder.classes_):
    print(index, "->", action)

# ------------------------------------------
# TRAIN / TEST SPLIT
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print()
print("Training sequences:", len(X_train))
print("Testing sequences :", len(X_test))

# ------------------------------------------
# BUILD LSTM MODEL
# ------------------------------------------

model = Sequential([
    
    LSTM(
        64,
        return_sequences=True,
        input_shape=(X_train.shape[1], X_train.shape[2])
    ),

    Dropout(0.2),

    LSTM(32),

    Dropout(0.2),

    Dense(32, activation="relu"),

    Dense(
        len(label_encoder.classes_),
        activation="softmax"
    )
])

# ------------------------------------------
# COMPILE MODEL
# ------------------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print()
print("LSTM model created.")
print()

model.summary()

# ------------------------------------------
# EARLY STOPPING
# ------------------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# ------------------------------------------
# TRAIN MODEL
# ------------------------------------------

print()
print("Training LSTM...")
print()

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.20,
    callbacks=[early_stopping],
    verbose=1
)

# ------------------------------------------
# TEST MODEL
# ------------------------------------------

print()
print("Evaluating LSTM model...")

y_probability = model.predict(
    X_test,
    verbose=0
)

y_pred = np.argmax(
    y_probability,
    axis=1
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

# ------------------------------------------
# RESULTS
# ------------------------------------------

print()
print("==========================================")
print(" LSTM RESULTS")
print("==========================================")

print(
    "LSTM Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print()
print("Classification Report:")
print("------------------------------------------")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)

# ------------------------------------------
# SAVE MODEL
# ------------------------------------------

model_path = os.path.join(
    MODEL_FOLDER,
    "lstm_action_model.keras"
)

model.save(model_path)

# Save class names
classes_path = os.path.join(
    MODEL_FOLDER,
    "lstm_classes.npy"
)

np.save(
    classes_path,
    label_encoder.classes_
)

print()
print("==========================================")
print(" LSTM MODEL SAVED SUCCESSFULLY")
print("==========================================")

print("Model:", model_path)
print("Classes:", classes_path)

print()
print("LSTM training finished.")