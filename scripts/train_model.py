"""
train_model.py
----------------
1. Loads the raw dataset
2. Cleans missing values / outliers
3. Splits into train/test and scales features
4. Trains 4 classifiers: Decision Tree, Random Forest, KNN, XGBoost
5. Evaluates each with confusion matrix / classification report / accuracy
6. Saves the best model + the fitted scaler to model/model.pkl and model/scaler.pkl

Run from the project root:  python scripts/train_model.py
"""

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier

RANDOM_STATE = 42

# --------------------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------------------
df = pd.read_csv("data/flood_dataset.csv")

# --------------------------------------------------------------------------
# 2. Data cleaning
# --------------------------------------------------------------------------
# Handle missing values -> median imputation (robust to outliers)
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# Outlier treatment via IQR capping on continuous features
continuous_cols = [
    "AnnualRainfall", "MonsoonRainfall", "CloudVisibility", "Humidity",
    "Temperature", "RiverWaterLevel", "SoilSaturation", "WindSpeed", "Elevation",
]
for col in continuous_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df[col] = df[col].clip(lower, upper)

# --------------------------------------------------------------------------
# 3. Feature / target split, scaling
# --------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "AnnualRainfall", "MonsoonRainfall", "CloudVisibility", "Humidity",
    "Temperature", "RiverWaterLevel", "SoilSaturation", "WindSpeed",
    "Elevation", "DrainageQuality",
]
TARGET_COLUMN = "FloodOccurred"

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------------------------------
# 4. Train & evaluate models
# --------------------------------------------------------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=RANDOM_STATE
    ),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "XGBoost": XGBClassifier(
        n_estimators=350,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    ),
}

results = {}
best_name, best_model, best_acc = None, None, -1

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    results[name] = {
        "accuracy": round(acc * 100, 2),
        "confusion_matrix": cm.tolist(),
    }

    print(f"\n{'=' * 55}\nModel: {name}")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y_test, preds))

    if acc > best_acc:
        best_acc = acc
        best_name = name
        best_model = model

print(f"\n{'=' * 55}")
print(f"BEST MODEL: {best_name}  |  Accuracy: {best_acc * 100:.2f}%")
print(f"{'=' * 55}")

# --------------------------------------------------------------------------
# 5. Persist best model + scaler + metadata
# --------------------------------------------------------------------------
joblib.dump(best_model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

metadata = {
    "best_model": best_name,
    "accuracy_percent": round(best_acc * 100, 2),
    "feature_columns": FEATURE_COLUMNS,
    "all_model_results": results,
}
with open("model/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nSaved:")
print(" - model/model.pkl")
print(" - model/scaler.pkl")
print(" - model/metadata.json")
