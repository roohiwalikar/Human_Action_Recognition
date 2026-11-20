from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model

# ============================================================
# SMARTMOTION - 9 CLASS LSTM INFERENCE SERVER
# ============================================================

app = Flask(__name__)
CORS(app)

# ------------------------------------------------------------
# FILE PATHS
# ------------------------------------------------------------

MODEL_PATH = "models/lstm_action_model_9class.keras"
CLASSES_PATH = "models/lstm_classes_9class.npy"

# ------------------------------------------------------------
# LOAD MODEL ONCE
# ------------------------------------------------------------

print("=" * 60)
print("Loading SmartMotion 9-Class LSTM Model...")

model = load_model(MODEL_PATH)
classes = np.load(CLASSES_PATH, allow_pickle=True)

print("9-CLASS LSTM MODEL LOADED")
print("-" * 60)

for i, class_name in enumerate(classes):
    print(f"{i} -> {class_name}")

print("=" * 60)


# ------------------------------------------------------------
# HOME / STATUS
# ------------------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "model": "9-Class LSTM",
        "classes": [str(c) for c in classes]
    })


# ------------------------------------------------------------
# LSTM PREDICTION
# ------------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        if not data or "sequence" not in data:
            return jsonify({
                "error": "No sequence received"
            }), 400

        sequence = np.array(data["sequence"], dtype=np.float32)

        # Expected shape:
        # (30, 132)
        if sequence.shape != (30, 132):
            return jsonify({
                "error": "Invalid sequence shape",
                "received": list(sequence.shape),
                "expected": [30, 132]
            }), 400

        # Add batch dimension:
        # (30, 132) -> (1, 30, 132)
        sequence = np.expand_dims(sequence, axis=0)

        # LSTM prediction
        predictions = model.predict(sequence, verbose=0)

        probabilities = predictions[0]

        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])

        predicted_class = str(classes[predicted_index])

        return jsonify({
            "activity": predicted_class,
            "confidence": confidence,
            "class_index": predicted_index
        })

    except Exception as e:

        print("Prediction error:", e)

        return jsonify({
            "error": str(e)
        }), 500


# ------------------------------------------------------------
# START SERVER
# ------------------------------------------------------------

if __name__ == "__main__":

    print("\nSmartMotion LSTM Server")
    print("Running at: http://127.0.0.1:5000")
    print("Waiting for predictions...\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )