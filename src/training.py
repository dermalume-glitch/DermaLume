import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ----------------------------
# 1. Load the preprocessed dataset
# ----------------------------
data_path = "data/HAM10000_metadata_preprocessed.csv"
df = pd.read_csv(data_path)
print("✅ Loaded dataset:", df.shape)

# ----------------------------
# 2. Define features and target
# ----------------------------
# Target: dx (diagnosis)
# For simplicity, let’s assume "dx" column has encoded values:
# benign = 0, malignant = 1 (or similar)
X = df.drop(columns=["dx"])  # all other columns are inputs
y = df["dx"]

# Optionally drop image_path or non-numeric columns
if "image_path" in X.columns:
    X = X.drop(columns=["image_path"])

# ----------------------------
# 3. Scale numeric data
# ----------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print("✅ Scaled input features")

# ----------------------------
# 4. Split into train and test
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print("✅ Split data:", X_train.shape, X_test.shape)

# ----------------------------
# 5. Train model
# ----------------------------
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)
print("✅ Model trained")

# ----------------------------
# 6. Evaluate model
# ----------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Accuracy: {acc:.3f}")
print("📊 Classification report:\n", classification_report(y_test, y_pred))

# ----------------------------
# 7. Save model & scaler
# ----------------------------
joblib.dump(model, "models/dermalume_model.pkl")
joblib.dump(scaler, "models/dermalume_scaler.pkl")
print("✅ Model and scaler saved to /models folder")

# ----------------------------
# 8. Sample prediction
# ----------------------------
# ⚠️ Adjust this to match your real feature order
sample = X_test[0].reshape(1, -1)  # Take one random sample from test set
sample_pred = model.predict(sample)
print("🧪 Sample prediction:", sample_pred)
