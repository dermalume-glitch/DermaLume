import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

# -----------------------------
# 0) Paths (edit here if your layout changes)
# -----------------------------
META_PATH = os.path.join("data", "HAM10000_metadata.csv")
IMAGES_DIR = os.path.join("data", "HAM10000_images_part_1")  # <- you said images are here
OUTPUT_CSV = os.path.join("data", "HAM10000_metadata_preprocessed.csv")
ENCODER_MAP_JSON = os.path.join("data", "label_encoders.json")

# -----------------------------
# 1) Load metadata
# -----------------------------
df = pd.read_csv(META_PATH)
print("Initial dataset shape:", df.shape)

# Strip any accidental whitespace in string columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip()

# Expected HAM10000 columns (order may vary):
# lesion_id, image_id, dx, dx_type, age, sex, localization
expected_cols = {"lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization"}
missing = expected_cols - set(df.columns)
if missing:
    raise ValueError(f"Metadata is missing expected columns: {missing}")

print(df.head(3))

# -----------------------------
# 2) Handle missing values
# -----------------------------
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

print("\nMissing values BEFORE:")
print(df.isna().sum())

# Impute numeric with median (age is the main numeric)
if len(num_cols) > 0:
    num_imputer = SimpleImputer(strategy="median")
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Impute categoricals with most frequent
if len(cat_cols) > 0:
    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

print("\nMissing values AFTER:")
print(df.isna().sum())

# -----------------------------
# 3) Encode categorical columns
# -----------------------------
label_cols = ["sex", "localization", "dx", "dx_type"]
encoders = {}
for col in label_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = list(le.classes_)
        print(f"Encoded {col} → classes: {le.classes_[:8]}{' ...' if len(le.classes_)>8 else ''}")

# Save encoders mapping so you can reverse-map later if needed
with open(ENCODER_MAP_JSON, "w") as f:
    json.dump(encoders, f, indent=2)
print(f"\nSaved label encoders map → {ENCODER_MAP_JSON}")

# -----------------------------
# 4) Flag obvious outliers (age)
# -----------------------------
if "age" in df.columns:
    df["age_outlier_flag"] = np.where((df["age"] < 0) | (df["age"] > 120), 1, 0)
    print("Age outliers flagged:", int(df["age_outlier_flag"].sum()))

# -----------------------------
# 5) Normalize appropriate numeric columns
# -----------------------------
scaler = MinMaxScaler()
for col in ["age"]:  # add more numeric columns here if you create them later
    if col in df.columns:
        df[col] = scaler.fit_transform(df[[col]])
        print(f"Normalized {col} to [0,1]")

# -----------------------------
# 6) Link image paths
# -----------------------------
# Your images live in data/HAM10000_images_part_1 and are named like ISIC_0024306.jpg.
# image_id column contains strings like ISIC_0024306
def build_path(image_id: str) -> str:
    return os.path.join(IMAGES_DIR, f"{image_id}.jpg")

df["image_path"] = df["image_id"].astype(str).apply(build_path)

# Optionally, check how many paths exist (helps catch path mistakes)
exists_mask = df["image_path"].apply(os.path.exists)
missing_count = (~exists_mask).sum()
if missing_count > 0:
    print(f"WARNING: {missing_count} image files not found under {IMAGES_DIR}. "
          f"Check your image folder and file names.")
else:
    print("All image paths resolved successfully.")

# -----------------------------
# 7) (Optional) Quick distribution prints (no plotting, safe for any environment)
# -----------------------------
if "dx" in df.columns:
    print("\nEncoded diagnosis value counts (dx):")
    print(df["dx"].value_counts())

# -----------------------------
# 8) Save cleaned metadata
# -----------------------------
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved cleaned metadata to: {OUTPUT_CSV}")
