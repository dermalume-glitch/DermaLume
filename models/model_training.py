# models/model_training.py
# Train a ML model using both metadata + images (preprocessed CSV, no extra preprocessing)
# Skips rows where the image file is missing

import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.preprocessing import LabelEncoder

# Paths
metadata_path = os.path.join("..", "data", "HAM10000_metadata_preprocessed.csv")

# 1️⃣ Load metadata
df = pd.read_csv(metadata_path)
print("✅ Metadata loaded:", df.shape)

# LIMIT TO 2000 ROWS FOR FASTER TRAINING (ensuring enough samples per class)
df = df.head(2000)
print("📊 Limited to", len(df), "rows for faster training")

# 2️⃣ Drop columns we don't want: dx_type, age_outlier_flag
df = df.drop(columns=["dx_type", "age_outlier_flag"])

# 3️⃣ Extract metadata features (keep numeric columns: age, sex, localization)
meta_cols = ["age", "sex", "localization"]
X_meta = df[meta_cols].values

# 4️⃣ Get the root folder (one level up from models/)
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 5️⃣ Load images, skipping missing files
X_images = []
valid_indices = []

print("⏳ Loading images...")
for idx, row in df.iterrows():
    image_path = os.path.join(root_folder, row["image_path"])
    if os.path.exists(image_path):
        img = Image.open(image_path).convert("RGB")
        img = img.resize((64, 64))
        X_images.append(np.array(img).flatten() / 255.0)
        valid_indices.append(idx)
    else:
        print(f"⚠️ Image not found, skipping: {image_path}")

X_images = np.array(X_images)
print("✅ Images loaded:", X_images.shape)

# Keep only rows with valid images
df = df.loc[valid_indices].reset_index(drop=True)
X_meta = df[meta_cols].values
y = df["dx"].values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 6️⃣ Combine metadata + image features
X = np.hstack([X_meta, X_images])
print("✅ Combined features shape:", X.shape)

# 7️⃣ Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"✅ Train/test split: {X_train.shape[0]} train, {X_test.shape[0]} test")

# 8️⃣ Train Random Forest
model = RandomForestClassifier(n_estimators=150, random_state=42)
print("⏳ Training model... (this may take a while)")
model.fit(X_train, y_train)
print("✅ Model trained")

# 9️⃣ Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🔹 Accuracy: {accuracy*100:.2f}%")
print("\nClassification Report:")
# Get unique labels in test set to handle cases where not all classes are present
unique_labels = np.unique(np.concatenate([y_test, y_pred]))
target_names = [str(label_encoder.inverse_transform([label])[0]) for label in unique_labels]
print(classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names))

# 🔟 Save model and label encoder
joblib.dump(model, "skin_diagnosis_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("💾 Model and label encoder was saved")

# 1️⃣1️⃣ Sample prediction (first test sample)
sample = X_test[0].reshape(1, -1)
pred_label = label_encoder.inverse_transform(model.predict(sample))[0]
print("\n🔍 Sample prediction (first test sample):", pred_label)
