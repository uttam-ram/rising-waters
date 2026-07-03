"""
app.py
-------
Flask web application for the Rising Waters flood prediction system.

Routes:
  /            Home page (dashboard)
  /predict     Input form (GET) + prediction handling (POST)
  /history     Prediction history (past predictions, stored in-session/on-disk)

Run:
  python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

MODEL_PATH = "model/model.pkl"
SCALER_PATH = "model/scaler.pkl"
METADATA_PATH = "model/metadata.json"
HISTORY_PATH = "model/prediction_history.json"

FEATURE_COLUMNS = [
    "AnnualRainfall", "MonsoonRainfall", "CloudVisibility", "Humidity",
    "Temperature", "RiverWaterLevel", "SoilSaturation", "WindSpeed",
    "Elevation", "DrainageQuality",
]

FEATURE_LABELS = {
    "AnnualRainfall": ("Annual Rainfall (mm)", 300, 4500, 1800),
    "MonsoonRainfall": ("Monsoon/Seasonal Rainfall (mm)", 100, 3600, 1200),
    "CloudVisibility": ("Cloud Visibility (km)", 0.2, 15, 6),
    "Humidity": ("Humidity (%)", 20, 100, 70),
    "Temperature": ("Temperature (°C)", 10, 45, 28),
    "RiverWaterLevel": ("River Water Level (m)", 0.5, 15, 5),
    "SoilSaturation": ("Soil Saturation (%)", 5, 100, 55),
    "WindSpeed": ("Wind Speed (km/h)", 0, 60, 15),
    "Elevation": ("Elevation (m above sea level)", 0, 800, 150),
    "DrainageQuality": ("Drainage Quality (1=Poor .. 5=Excellent)", 1, 5, 3),
}


def load_model():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        return None, None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    metadata = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
    return model, scaler, metadata


model, scaler, metadata = load_model()


def load_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH) as f:
            return json.load(f)
    return []


def save_history(entry):
    history = load_history()
    history.insert(0, entry)
    history = history[:50]  # keep last 50
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


@app.route("/")
def home():
    return render_template("index.html", metadata=metadata)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html", fields=FEATURE_LABELS, values=None)

    # POST: read + validate form values
    try:
        values = {}
        for col in FEATURE_COLUMNS:
            raw = request.form.get(col)
            values[col] = float(raw)
    except (TypeError, ValueError):
        return render_template(
            "predict.html",
            fields=FEATURE_LABELS,
            values=request.form,
            error="Please enter valid numeric values for every field.",
        )

    if model is None or scaler is None:
        return render_template(
            "predict.html",
            fields=FEATURE_LABELS,
            values=request.form,
            error="Model not found. Please run scripts/train_model.py first.",
        )

    X = pd.DataFrame([[values[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    X_scaled = scaler.transform(X)

    prediction = int(model.predict(X_scaled)[0])
    try:
        proba = model.predict_proba(X_scaled)[0][prediction]
        confidence = round(float(proba) * 100, 2)
    except AttributeError:
        confidence = None

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": values,
        "prediction": prediction,
        "confidence": confidence,
    }
    save_history(entry)

    return render_template(
        "result.html",
        prediction=prediction,
        confidence=confidence,
        inputs=values,
        labels=FEATURE_LABELS,
    )


@app.route("/history")
def history():
    return render_template("history.html", history=load_history())


if __name__ == "__main__":
    app.run(debug=True)
