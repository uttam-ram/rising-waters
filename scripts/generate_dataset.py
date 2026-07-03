"""
generate_dataset.py
--------------------
Generates a realistic synthetic flood-prediction dataset.

In the real project you would replace this with a Kaggle dataset
(e.g. search "flood prediction dataset" on kaggle.com), but this
script lets the whole pipeline run end-to-end without an internet
download, and produces data with realistic relationships between
rainfall / cloud visibility / seasonal rainfall and flood risk.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 3000

# --- Core meteorological features -----------------------------------------
annual_rainfall = np.random.normal(1800, 500, N).clip(300, 4500)          # mm/year
monsoon_rainfall = annual_rainfall * np.random.uniform(0.55, 0.8, N)      # mm during monsoon
cloud_visibility = np.random.normal(6, 2.5, N).clip(0.2, 15)              # km (lower = worse visibility, more storm activity)
humidity = np.random.normal(70, 15, N).clip(20, 100)                      # %
temperature = np.random.normal(28, 5, N).clip(10, 45)                     # deg C
river_water_level = np.random.normal(5, 2, N).clip(0.5, 15)               # meters
soil_saturation = np.random.normal(55, 20, N).clip(5, 100)                # %
wind_speed = np.random.normal(15, 8, N).clip(0, 60)                       # km/h
elevation = np.random.normal(150, 100, N).clip(0, 800)                    # meters above sea level
drainage_quality = np.random.randint(1, 6, N)                             # 1 (poor) - 5 (excellent)

# --- Derive flood probability from a physically-plausible combination -----
risk_score = (
    0.30 * (annual_rainfall / 4500)
    + 0.25 * (monsoon_rainfall / 3600)
    + 0.15 * (1 - cloud_visibility / 15)      # poor visibility -> higher risk
    + 0.10 * (humidity / 100)
    + 0.10 * (river_water_level / 15)
    + 0.10 * (soil_saturation / 100)
    - 0.15 * (elevation / 800)
    - 0.10 * (drainage_quality / 5)
)

risk_score = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())
noise = np.random.normal(0, 0.08, N)
final_score = (risk_score + noise).clip(0, 1)

flood_occurred = (final_score > np.percentile(final_score, 55)).astype(int)

df = pd.DataFrame({
    "AnnualRainfall": annual_rainfall.round(1),
    "MonsoonRainfall": monsoon_rainfall.round(1),
    "CloudVisibility": cloud_visibility.round(2),
    "Humidity": humidity.round(1),
    "Temperature": temperature.round(1),
    "RiverWaterLevel": river_water_level.round(2),
    "SoilSaturation": soil_saturation.round(1),
    "WindSpeed": wind_speed.round(1),
    "Elevation": elevation.round(1),
    "DrainageQuality": drainage_quality,
    "FloodOccurred": flood_occurred,
})

# Inject a small number of missing values / light outliers so the
# preprocessing step in train_model.py has something real to clean.
for col in ["Humidity", "WindSpeed", "SoilSaturation"]:
    idx = np.random.choice(df.index, size=int(0.01 * N), replace=False)
    df.loc[idx, col] = np.nan

df.to_csv("data/flood_dataset.csv", index=False)
print(f"Saved data/flood_dataset.csv with shape {df.shape}")
print(df["FloodOccurred"].value_counts(normalize=True))
