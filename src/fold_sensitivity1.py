"""Article 1 - sensitivity of the date-grouped estimate to the fold assignment.
Usage: python fold_sensitivity1.py
Requires irrigation_clean.parquet produced by prep1.py.
Outputs: article1_fold_sensitivity.csv

With 17 calendar days and positives concentrated on a few of them, the way days
are packed into five folds changes the grouped score. This script re-runs the
grouped protocol for the deterministic split used in the paper and for 60
shuffled splits (seeds 0-59), keeping only distinct day-to-fold assignments.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, average_precision_score
import lightgbm as lgb

SEED = 42
FEATURES = ["Soil Percent", "Soil Raw", "Suhu Tanah", "Suhu Udara", "Humidity"]
df = pd.read_parquet("irrigation_clean.parquet")
X = df[FEATURES]
y = df["Label"].to_numpy(dtype=int)
groups = df["DateGroup"].to_numpy()

MODELS = {
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=SEED),
    "LightGBM": lambda: lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, class_weight="balanced",
        random_state=SEED, n_jobs=-1, verbose=-1),
}

def run(splits, make_model):
    pred = np.zeros(len(y), dtype=int)
    prob = np.zeros(len(y), dtype=float)
    for tr, va in splits:
        m = make_model().fit(X.iloc[tr], y[tr])
        pred[va] = m.predict(X.iloc[va])
        prob[va] = m.predict_proba(X.iloc[va])[:, 1]
    return f1_score(y, pred), matthews_corrcoef(y, pred), average_precision_score(y, prob)

configs = [("no-shuffle (paper)", StratifiedGroupKFold(n_splits=5))]
configs += [(f"shuffle seed {s}", StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=s))
            for s in range(60)]

rows, seen = [], set()
for label, cv in configs:
    splits = list(cv.split(X, y, groups))
    key = tuple(sorted(tuple(sorted(set(groups[va]))) for _, va in splits))
    if key in seen:  # identical day-to-fold packing already evaluated
        continue
    seen.add(key)
    for name, make in MODELS.items():
        f1, mcc, ap = run(splits, make)
        rows.append({"Split": label, "Model": name, "F1": f1, "MCC": mcc, "PRAUC": ap})
    print(label, "done", flush=True)

res = pd.DataFrame(rows)
res.to_csv("article1_fold_sensitivity.csv", index=False)
summary = (res.groupby("Model")[["F1", "MCC", "PRAUC"]]
              .agg(["mean", "std", "min", "max"]) * 100).round(2)
print("\nDistinct fold assignments evaluated:", len(seen))
print(summary.to_string())
