"""
data_analysis.py
------------------
Exploratory Data Analysis (EDA) for the flood prediction dataset.
Produces:
  - Descriptive statistics (printed + saved to CSV)
  - Distribution plots (univariate)
  - Box plots (outlier detection)
  - Correlation heat map (multivariate)
  - Pairwise scatter of top correlated features vs target

Run from the project root:  python scripts/data_analysis.py
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("reports", exist_ok=True)
sns.set_style("whitegrid")

df = pd.read_csv("data/flood_dataset.csv")

print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# ---- Descriptive statistics ----
desc = df.describe(include="all")
desc.to_csv("reports/descriptive_statistics.csv")
print("\nDescriptive statistics saved to reports/descriptive_statistics.csv")

numeric_cols = [c for c in df.columns if c != "FloodOccurred"]

# ---- Univariate: distribution plots ----
fig, axes = plt.subplots(4, 3, figsize=(16, 14))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="#1f77b4")
    axes[i].set_title(f"Distribution of {col}")
for j in range(len(numeric_cols), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.savefig("reports/distributions.png", dpi=120)
plt.close()

# ---- Univariate: box plots (outlier detection) ----
fig, axes = plt.subplots(4, 3, figsize=(16, 14))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    sns.boxplot(y=df[col], ax=axes[i], color="#ff7f0e")
    axes[i].set_title(f"Box plot: {col}")
for j in range(len(numeric_cols), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.savefig("reports/boxplots.png", dpi=120)
plt.close()

# ---- Multivariate: correlation heat map ----
plt.figure(figsize=(11, 9))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("reports/correlation_heatmap.png", dpi=120)
plt.close()

# ---- Target balance ----
plt.figure(figsize=(5, 4))
sns.countplot(x="FloodOccurred", data=df, palette=["#2ca02c", "#d62728"])
plt.title("Flood vs No-Flood Class Balance")
plt.xlabel("Flood Occurred (0 = No, 1 = Yes)")
plt.tight_layout()
plt.savefig("reports/class_balance.png", dpi=120)
plt.close()

# ---- Multivariate: key features vs target ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["AnnualRainfall", "MonsoonRainfall", "CloudVisibility"]):
    sns.boxplot(x="FloodOccurred", y=col, data=df, ax=ax, palette=["#2ca02c", "#d62728"])
    ax.set_title(f"{col} by Flood Outcome")
plt.tight_layout()
plt.savefig("reports/key_features_vs_target.png", dpi=120)
plt.close()

print("\nAll EDA plots saved to the 'reports/' folder:")
for f in sorted(os.listdir("reports")):
    print(" -", f)
