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
import numpy as np
from PIL import Image
import io
import base64
import json

# joblib is commonly used for scikit-learn models; import if available
try:
    import joblib
except Exception:  # keep the app simple; not fatal if joblib isn't installed
    joblib = None

# -------------------------------------------------
# Create the Flask app (must use this exact name)
# -------------------------------------------------
app = Flask(__name__, static_folder="../static")

# (Optional but recommended) enable CORS so the browser JS can call /predict
CORS(app)

# -------------------------------------------------
# Configuration: encoders for categorical features
# -------------------------------------------------
LABEL_ENCODERS = None
SEX_ENCODER = {"female": 0, "male": 1, "unknown": 2}
LOCATION_ENCODER = {
    "abdomen": 0, "acral": 1, "back": 2, "chest": 3, "ear": 4,
    "face": 5, "foot": 6, "genital": 7, "hand": 8, "lower extremity": 9,
    "neck": 10, "scalp": 11, "trunk": 12, "unknown": 13, "upper extremity": 14
}

# -------------------------------------------------
# Load the saved model once when the server starts
# Tries models/model.pkl first, then models/model.joblib
# -------------------------------------------------
MODEL = None
MODEL_PATHS_TRY = [
    os.path.join("..", "models", "skin_diagnosis_model.pkl"),
    os.path.join("models", "skin_diagnosis_model.pkl"),
]

LABEL_ENCODER_PATHS = [
    os.path.join("..", "models", "label_encoder.pkl"),
    os.path.join("models", "label_encoder.pkl"),
]

def _load_model():
    """Attempt to load a trained model from the models/ folder."""
    global MODEL
    for path in MODEL_PATHS_TRY:
        if os.path.exists(path):
            try:
                # Use joblib for all .pkl files (scikit-learn models)
                if joblib is not None:
                    MODEL = joblib.load(path)
                else:
                    with open(path, "rb") as f:
                        MODEL = pickle.load(f)
                print(f"[INFO] Loaded model from: {path}")
                return
            except Exception as e:
                print(f"[WARN] Failed to load {path}: {e}")
    print("[ERROR] No model loaded. Ensure models/skin_diagnosis_model.pkl exists.")

def _load_label_encoder():
    """Load the label encoder for decoding predictions."""
    global LABEL_ENCODERS
    for path in LABEL_ENCODER_PATHS:
        if os.path.exists(path):
            try:
                LABEL_ENCODERS = joblib.load(path)
                print(f"[INFO] Loaded label encoder from: {path}")
                return
            except Exception as e:
                print(f"[WARN] Failed to load label encoder {path}: {e}")
    
    # Load from JSON if pkl fails
    json_path = os.path.join("..", "data", "label_encoders.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            LABEL_ENCODERS = data.get("dx", [])
            print(f"[INFO] Loaded label encoders from JSON")

_load_model()
_load_label_encoder()

# -------------------------------------------------
# Routes
# -------------------------------------------------

# GET /  -> serves the existing index.html from project root
@app.route("/")
def home():
    # Sends the already-existing front-end file
    return send_from_directory("..", "index.html")


# POST /predict -> handle image upload and metadata
@app.route("/predict", methods=["POST"])
def predict():
    print("[INFO] Received prediction request")
    # Make sure the model was loaded successfully
    if MODEL is None:
        return jsonify(error="Model not loaded. Please ensure a model file exists in /models."), 500

    try:
        # Get form data
        age = request.form.get('age')
        gender = request.form.get('gender')
        location = request.form.get('location')
        
        # Get the image file
        if 'image' not in request.files:
            return jsonify(error="No image file provided"), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify(error="No image selected"), 400
        
        # Validate inputs
        if not age or not gender or not location:
            return jsonify(error="Missing required fields: age, gender, or location"), 400
        
        # Process age (normalize to 0-1 range like in training)
        try:
            age_value = float(age) / 85.0  # Normalize age (assuming max age ~85)
        except ValueError:
            return jsonify(error="Invalid age value"), 400
        
        # Encode gender
        gender_encoded = SEX_ENCODER.get(gender.lower(), 2)  # Default to 'unknown'
        
        # Encode location
        location_encoded = LOCATION_ENCODER.get(location.lower(), 13)  # Default to 'unknown'
        
        # Process image
        img = Image.open(image_file.stream).convert('RGB')
        img = img.resize((64, 64))
        img_array = np.array(img).flatten() / 255.0  # Normalize pixel values
        
        # Combine metadata and image features
        # Order: [age, sex, localization] + flattened image
        metadata_features = np.array([age_value, gender_encoded, location_encoded])
        features = np.hstack([metadata_features, img_array])
        
        # Make prediction
        pred_encoded = MODEL.predict([features])[0]
        print(f"[DEBUG] Predicted class: {pred_encoded}")
        
        # Decode prediction - direct mapping since we know the encoding
        dx_map = {0: "akiec", 1: "bcc", 2: "bkl", 3: "df", 4: "mel", 5: "nv", 6: "vasc"}
        prediction = dx_map.get(int(pred_encoded), str(pred_encoded))
        
        print(f"[DEBUG] Decoded prediction: {prediction}")
        
        # Get confidence (if available)
        confidence = 0.85  # Placeholder - you'd get this from model.predict_proba
        if hasattr(MODEL, 'predict_proba'):
            proba = MODEL.predict_proba([features])[0]
            confidence = float(max(proba))
        
        return jsonify({
            "prediction": str(prediction),
            "confidence": float(confidence),
            "dx_full": get_diagnosis_name(str(prediction))
        }), 200

    except Exception as e:
        # Catch any other error and return as JSON instead of crashing
        print(f"[ERROR] Prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify(error=f"Prediction failed: {str(e)}"), 400

def get_diagnosis_name(dx_code):
    """Convert dx code to full diagnosis name."""
    dx_names = {
        "akiec": "Actinic keratoses",
        "bcc": "Basal cell carcinoma", 
        "bkl": "Benign keratosis",
        "df": "Dermatofibroma",
        "mel": "Melanoma",
        "nv": "Melanocytic nevi",
        "vasc": "Vascular lesions"
    }
    return dx_names.get(dx_code, dx_code)


# Optional: serve other static assets by path (CSS/JS/images under /static)
# e.g., GET /static/style.css or /static/script.js will be served automatically by Flask,
# because we set static_folder="static" when creating the app.


# -------------------------------------------------
# Run the app
# -------------------------------------------------
if __name__ == "__main__":
    # debug=True is helpful during development; turn off in production
    app.run(debug=True)
