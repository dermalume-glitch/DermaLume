# app.py
# -----------------------------
# A tiny Flask app that serves an existing front-end (already in /static)
# and exposes a /predict API that uses a pre-trained ML model from /models.
#
# 🔧 What you need to know / change (if needed):
# - The app will try to load a model from:
#       models/model.pkl   (pickle)
#   then models/model.joblib (joblib)
#   You can keep both; the first one found will be used.
# - FEATURE_ORDER below lists the input features and their order expected by the model.
#   Update FEATURE_ORDER if your model expects different features or a different order.
#
# 📨 EXAMPLE REQUEST to POST /predict (send JSON):
# {
#   "dx": "benign or harmfull",
#   "age": "adolescence, adults, elderly",
#   "sex": "male or female",
#   "localization": "back, arm, face, chest, upper/lower extremity, scalp, abdomen, ear, truck"
# }
#
# ✅ EXAMPLE RESPONSE:
# {
#   "prediction": "will churn"
# }
#
# Notes:
# - This example assumes a classification model that returns a class label.
# - If your model returns a number, it will still be returned as a string for safety.
# - Errors return JSON with a message and a 400 status code, so the app won't crash.

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import pickle

# joblib is commonly used for scikit-learn models; import if available
try:
    import joblib
except Exception:  # keep the app simple; not fatal if joblib isn't installed
    joblib = None

# -------------------------------------------------
# Create the Flask app (must use this exact name)
# -------------------------------------------------
from flask import Flask
app = Flask(__name__, static_folder="static")

# (Optional but recommended) enable CORS so the browser JS can call /predict
from flask_cors import CORS
CORS(app)

# -------------------------------------------------
# Configuration: feature order expected by the model
# Adjust this list to match your trained model's inputs & order.
# -------------------------------------------------
FEATURE_ORDER = ["dx", "age", "sex", "localization"]

# -------------------------------------------------
# Load the saved model once when the server starts
# Tries models/model.pkl first, then models/model.joblib
# -------------------------------------------------
MODEL = None
MODEL_PATHS_TRY = [
    os.path.join("models", "model.pkl"),
    os.path.join("models", "model.joblib"),
]

def _load_model():
    """Attempt to load a trained model from the models/ folder."""
    global MODEL
    for path in MODEL_PATHS_TRY:
        if os.path.exists(path):
            try:
                if path.endswith(".pkl"):
                    with open(path, "rb") as f:
                        MODEL = pickle.load(f)
                elif path.endswith(".joblib") and joblib is not None:
                    MODEL = joblib.load(path)
                else:
                    # joblib file but joblib isn't installed
                    continue
                print(f"[INFO] Loaded model from: {path}")
                return
            except Exception as e:
                print(f"[WARN] Failed to load {path}: {e}")
    print("[ERROR] No model loaded. Ensure models/model.pkl or models/model.joblib exists.")

_load_model()

# -------------------------------------------------
# Routes
# -------------------------------------------------

# GET /  -> serves the existing index.html from /static
@app.route("/")
def home():
    # Sends the already-existing front-end file; do NOT modify the front-end
    return send_from_directory(app.static_folder, "index.html")


# POST /predict -> read JSON, extract features in order, call model, return JSON
@app.route("/predict", methods=["POST"])
def predict():
    # Make sure the model was loaded successfully
    if MODEL is None:
        return jsonify(error="Model not loaded. Please ensure a model file exists in /models."), 500

    # Ensure we got JSON
    if not request.is_json:
        return jsonify(error="Request must be JSON with Content-Type: application/json"), 400

    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify(error="Invalid or empty JSON body."), 400

        # Validate and collect features in the required order
        missing = [feat for feat in FEATURE_ORDER if feat not in data]
        if missing:
            return jsonify(error=f"Missing required fields: {missing}", required_order=FEATURE_ORDER), 400

        # Build the model input row in the correct order
        # Many sklearn models accept a list of lists: [[f1, f2, f3, ...]]
        row = [data[feat] for feat in FEATURE_ORDER]

        # Run prediction
        # For classification, sklearn's predict returns labels; we convert to string for safety.
        pred = MODEL.predict([row])[0]
        return jsonify(prediction=str(pred)), 200

    except Exception as e:
        # Catch any other error and return as JSON instead of crashing
        return jsonify(error=f"Prediction failed: {str(e)}"), 400


# Optional: serve other static assets by path (CSS/JS/images under /static)
# e.g., GET /static/style.css or /static/script.js will be served automatically by Flask,
# because we set static_folder="static" when creating the app.


# -------------------------------------------------
# Run the app
# -------------------------------------------------
if __name__ == "__main__":
    # debug=True is helpful during development; turn off in production
    app.run(debug=True)
