# Leakage-aware evaluation of imbalanced smart-irrigation classifiers

Code for the paper:

> **Leakage-Aware and Cost-Sensitive Machine Learning for Highly Imbalanced Smart Irrigation Decision-Making**
> B. El Khalyly and Z. Ibn Batouta, 2026 (under review).

IoT greenhouse readings arrive seconds apart. If they are shuffled into random cross-validation folds, almost identical records end up in both training and validation, and the scores look far better than they are.
This repository measures that gap on a public dataset. Six cost-sensitive classifiers are scored under two protocols:
- stratified random folds;
- folds grouped by calendar day.

The grouped protocol is then repeated over 61 different assignments of days to folds, to show how much a single grouped number can move.

## What the pipeline does

1. **`src/prep1.py`** keeps the 53,209 labeled rows and parses both timestamp formats. It removes 18 rows with missing sensor values and 2 with malformed timestamps, then drops 4,547 exact duplicate feature–label vectors. That leaves 48,642 rows, 208 of them positive, over 17 days. It also checks that no feature vector carries two labels and builds `DateGroup`.
2. **`src/eval1.py`** runs logistic regression, Random Forest, Extra Trees, LightGBM, XGBoost and CatBoost under two protocols:

   | Protocol | Splitter |
   |---|---|
   | A | `StratifiedKFold(5, shuffle=True, random_state=42)` |
   | B | `StratifiedGroupKFold(5)` by date, no shuffling |

   Class weights and scaling are fitted inside each training fold, and protocol B asserts that no date is shared between training and validation.
3. **`src/figs1.py`** prints Tables II–IV and draws Figs. 1–2.
4. **`src/fold_sensitivity1.py`** repeats protocol B for the deterministic split and 60 shuffled ones, keeps the 61 distinct day-to-fold assignments, and reports F1/MCC/PR-AUC for Random Forest and LightGBM (Sec. IV-D).

## Run it in Google Colab or Jupyter

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<user>/smart-irrigation-leakage-aware-validation/blob/main/reproduce_colab.ipynb)

`reproduce_colab.ipynb` runs the whole experiment in one notebook. It installs the pinned library versions; on Colab the session restarts once, after which you choose *Run all* again. It then asks for `greenhouse_dataset.csv` (or picks it up from `data/` when run locally), and ends with a table that compares every number reported in the paper with the current run and marks each one ✅ or ❌.

Replace `<user>` in the badge link with your GitHub account once the repository is public.

## Quick start (command line)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python reproduce.py
```

Steps 1–3 take a few minutes. Step 4 trains 2 models × 5 folds × 61 splits and takes roughly 20 minutes on a laptop.
Remove it from `STEPS` in `reproduce.py` if you only need the main tables.

Outputs in `outputs/`:

| File | Content | Paper |
|---|---|---|
| `irrigation_clean.parquet` | cleaned data with `DateGroup` | Table I |
| `article1_results.csv` | all metrics and confusion counts, both protocols | Tables II–III |
| `article1_oof_predictions.parquet` | out-of-fold labels, predictions, probabilities, folds | – |
| `table_f1_validation_gap.csv` | F1 drop per model | Table IV |
| `fig_*.png` | F1 and PR-AUC under both protocols | Figs. 1–2 |
| `article1_fold_sensitivity.csv` | one row per (split, model) | Sec. IV-D |

## Expected results

Our outputs are in `results/` for comparison.

**F1 (%)**

| Model | Random folds | Date-grouped folds | Drop (points) |
|---|---|---|---|
| Logistic regression | 25.50 | 15.50 | 10.00 |
| Random Forest | 91.52 | 63.98 | 27.53 |
| Extra Trees | 92.07 | 60.82 | 31.26 |
| LightGBM | 93.43 | 48.28 | 45.15 |
| XGBoost | 93.24 | 52.78 | 40.46 |
| CatBoost | 89.50 | 45.90 | 43.60 |

Best grouped PR-AUC: CatBoost, 81.52%. An always-negative classifier would score 99.57% accuracy.

**Sensitivity of the grouped F1 to the fold assignment (61 distinct assignments)**

| Model | Mean | SD | Min | Max |
|---|---|---|---|---|
| LightGBM | 55.30 | 5.36 | 40.33 | 65.28 |
| Random Forest | 59.65 | 21.42 | 0.81 | 76.22 |

Random Forest falls below 30% F1 on 11 of the 61 assignments; LightGBM never goes below 40%.

## Reproducibility notes

- Verified on Windows 11, Python 3.11.9, with the versions in `requirements.txt`.
- The tables in the revised manuscript come from this exact environment. XGBoost and CatBoost scores shift by up to a few points with other library versions, Random Forest and Extra Trees by a few tenths. Logistic regression and LightGBM reproduce the values of the original submission exactly.
- Protocol B uses `StratifiedGroupKFold(n_splits=5)` **without shuffling**. An earlier version of the supplementary PDF listed `shuffle=True, random_state=42`; that was a documentation error, and Sec. IV-D shows why the choice matters.
- `catboost_info/` is written to `outputs/` by CatBoost and can be ignored.

## Data

See [`data/README.md`](data/README.md).

## Citation

Please cite the paper (reference to be updated on publication) and the dataset:

```
I. Keok, "IoT Greenhouse Irrigation Sensor Dataset," Mendeley Data, V1, 2026, doi: 10.17632/2tvpsht4s4.1.
```

## License

Code: MIT (see `LICENSE`). Data: CC BY 4.0, from the original authors.
