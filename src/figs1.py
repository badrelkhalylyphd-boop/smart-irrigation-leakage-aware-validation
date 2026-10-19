"""Article 1 - generate manuscript tables/figures from article1_results.csv.
Usage: python figs1.py
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

r = pd.read_csv("article1_results.csv")
metrics = ["Accuracy", "BalancedAccuracy", "Precision", "Recall", "F1", "MCC", "PRAUC"]
print("\nResults (%)")
print(r[["Protocol", "Model"] + metrics].assign(**{m: r[m] * 100 for m in metrics}).round(2).to_string(index=False))

# F1 comparison
p = r.pivot(index="Model", columns="Protocol", values="F1") * 100
ax = p.plot(kind="bar", figsize=(8, 4))
ax.set_ylabel("F1-score (%)")
ax.set_xlabel("")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("fig_f1_random_vs_grouped.png", dpi=300)
plt.close()

# PR-AUC comparison
p = r.pivot(index="Model", columns="Protocol", values="PRAUC") * 100
ax = p.plot(kind="bar", figsize=(8, 4))
ax.set_ylabel("PR-AUC (%)")
ax.set_xlabel("")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("fig_prauc_random_vs_grouped.png", dpi=300)
plt.close()

# Validation-gap table
q = r.pivot(index="Model", columns="Protocol", values="F1") * 100
q["F1_drop_points"] = q["Random stratified"] - q["Date-grouped"]
q.to_csv("table_f1_validation_gap.csv")
print("\nF1 degradation (percentage points)")
print(q[["F1_drop_points"]].round(2).to_string())
