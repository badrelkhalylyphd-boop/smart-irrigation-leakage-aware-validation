"""Article 1 - random vs date-grouped leakage-aware evaluation.
Usage: python eval1.py
Requires irrigation_clean.parquet produced by prep1.py.
Outputs: article1_results.csv, article1_oof_predictions.parquet
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, average_precision_score, confusion_matrix
)
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

SEED = 42
FEATURES = ["Soil Percent", "Soil Raw", "Suhu Tanah", "Suhu Udara", "Humidity"]
df = pd.read_parquet("irrigation_clean.parquet")
X = df[FEATURES]
y = df["Label"].to_numpy(dtype=int)
groups = df["DateGroup"].to_numpy()


def build_model(name, y_train):
    """Create a cost-sensitive classifier using training-fold information only."""
    if name == "Logistic Regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED),
        )
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=SEED
        )
    if name == "Extra Trees":
        return ExtraTreesClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=SEED
        )
    if name == "LightGBM":
        return lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, class_weight="balanced",
            random_state=SEED, n_jobs=-1, verbose=-1,
        )
    if name == "XGBoost":
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        return XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=n_neg / max(n_pos, 1), eval_metric="logloss",
            random_state=SEED, n_jobs=-1,
        )
    if name == "CatBoost":
        return CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05,
            auto_class_weights="Balanced", verbose=False, random_seed=SEED,
        )
    raise ValueError(name)


def score(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "BalancedAccuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "PRAUC": average_precision_score(y_true, y_prob),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }


def evaluate(protocol, splitter):
    rows = []
    oof_rows = []
    names = ["Logistic Regression", "Random Forest", "Extra Trees", "LightGBM", "XGBoost", "CatBoost"]
    split_iter = splitter.split(X, y, groups) if protocol == "Date-grouped" else splitter.split(X, y)
    splits = list(split_iter)

    # Group-leakage assertion for the grouped protocol.
    if protocol == "Date-grouped":
        for tr, va in splits:
            assert set(groups[tr]).isdisjoint(set(groups[va]))

    for name in names:
        pred = np.zeros(len(y), dtype=int)
        prob = np.zeros(len(y), dtype=float)
        fold_id = np.full(len(y), -1, dtype=int)

        for fold, (tr, va) in enumerate(splits):
            model = build_model(name, y[tr])
            model.fit(X.iloc[tr], y[tr])
            pred[va] = np.asarray(model.predict(X.iloc[va])).reshape(-1).astype(int)
            prob[va] = model.predict_proba(X.iloc[va])[:, 1]
            fold_id[va] = fold

        m = score(y, pred, prob)
        rows.append({"Protocol": protocol, "Model": name, **m})
        oof_rows.append(pd.DataFrame({
            "Protocol": protocol, "Model": name, "Row": np.arange(len(y)),
            "Fold": fold_id, "True": y, "Pred": pred, "Prob": prob,
            "DateGroup": groups,
        }))
        print(protocol, name, {k: round(100*v, 2) if k not in {"TP", "TN", "FP", "FN"} else v
                               for k, v in m.items()})
    return rows, oof_rows


random_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
group_cv = StratifiedGroupKFold(n_splits=5)  # no shuffle: deterministic, as used for the reported results

rows_a, oof_a = evaluate("Random stratified", random_cv)
rows_b, oof_b = evaluate("Date-grouped", group_cv)

res = pd.DataFrame(rows_a + rows_b)
res.to_csv("article1_results.csv", index=False)
pd.concat(oof_a + oof_b, ignore_index=True).to_parquet("article1_oof_predictions.parquet", index=False)
print("\nSaved article1_results.csv and article1_oof_predictions.parquet")
