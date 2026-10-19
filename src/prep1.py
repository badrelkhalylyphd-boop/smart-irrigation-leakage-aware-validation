"""Article 1 - preprocessing for leakage-aware smart-irrigation classification.
Usage: python prep1.py greenhouse_dataset.csv
Outputs: irrigation_clean.parquet
"""
import sys
import numpy as np
import pandas as pd

INPUT = sys.argv[1] if len(sys.argv) > 1 else "greenhouse_dataset.csv"
FEATURES = ["Soil Percent", "Soil Raw", "Suhu Tanah", "Suhu Udara", "Humidity"]

# The source CSV is semicolon-separated.
df = pd.read_csv(INPUT, sep=";")
print("Raw:", df.shape)
print("Label counts (including missing):", df["Label"].value_counts(dropna=False).to_dict())

# Article 1 is supervised: retain only rows with an observed irrigation label.
df = df[df["Label"].notna()].copy()
df["Label"] = df["Label"].astype(int)
print("Labeled:", len(df), df["Label"].value_counts().to_dict())

# Convert non-finite sensor measurements to missing.
df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
missing_sensor = int(df[FEATURES].isna().any(axis=1).sum())
print("Rows with missing/non-finite sensor values:", missing_sensor)

# Two timestamp formats occur in the file:
# 2026-04-23_04-51-49
# 2026-05-25 21:48:31
# Normalize both formats without using future information.
s = df["Timestamp"].astype(str).str.replace("_", " ", regex=False)
parts = s.str.split(" ", n=1, expand=True)
parts[1] = parts[1].str.replace("-", ":", regex=False)
df["Datetime"] = pd.to_datetime(parts[0] + " " + parts[1], errors="coerce")
invalid_time = int(df["Datetime"].isna().sum())
print("Rows with invalid timestamps:", invalid_time)

# Remove unusable supervised records (18 sensor-missing + 2 invalid timestamps in v1).
df = df.dropna(subset=FEATURES + ["Datetime"]).copy()
print("After unusable-row removal:", len(df))

# Exact duplicate feature-label vectors are removed before any validation split.
before = len(df)
df = df.drop_duplicates(subset=FEATURES + ["Label"]).copy()
print("Exact duplicate feature-label rows removed:", before - len(df))

# Safety check: the same feature vector must not carry contradictory labels.
amb = df.groupby(FEATURES, dropna=False)["Label"].nunique()
ambiguous = int((amb > 1).sum())
print("Contradictory feature vectors:", ambiguous)
if ambiguous:
    bad = amb[amb > 1].reset_index()[FEATURES]
    df = df.merge(bad.assign(_bad=1), on=FEATURES, how="left")
    df = df[df["_bad"].isna()].drop(columns="_bad")

# Calendar day is used only as a grouping variable for leakage-aware CV.
df["DateGroup"] = df["Datetime"].dt.strftime("%Y-%m-%d")
print("Final:", len(df), df["Label"].value_counts().to_dict())
print("Positive prevalence (%):", round(100 * df["Label"].mean(), 4))
print("Number of calendar-day groups:", df["DateGroup"].nunique())

df.to_parquet("irrigation_clean.parquet", index=False)
