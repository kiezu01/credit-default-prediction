# Credit Default Risk Prediction – LightGBM vs. Logistic Regression vs. Stacked Ensemble

Comparative study of three credit scoring approaches on two credit datasets, with class-imbalance handling and SHAP explainability. Group coursework (Master's program in Information Systems).

➡️ **Full code, figures and detailed results are in [`github_repo/`](github_repo/README.md).**

## Approach

| # | Model | Library |
|---|---|---|
| 1 | Gradient Boosted Trees | LightGBM |
| 2 | Logistic Regression (class-weighted) | scikit-learn |
| 3 | Stacked Ensemble (LightGBM + LR → LR meta-learner) | scikit-learn |

- **SMOTE** oversampling for the imbalanced target (~8% default rate)
- **5-fold out-of-fold stacking** to avoid leakage
- Evaluation with **AUC-ROC, AUC-PR, F1 and Brier score** (calibration)
- **SHAP** feature importance for model explainability

## Key results

| Model | Dataset | AUC-ROC | AUC-PR | Brier |
|---|---|---|---|---|
| Logistic Regression | Give Me Some Credit | **0.884** | **0.489** | 0.148 |
| Stacked Ensemble | Give Me Some Credit | 0.847 | 0.368 | **0.087** |
| LightGBM | Give Me Some Credit | 0.780 | 0.245 | 0.255 |

**Takeaway:** on the harder, more imbalanced dataset, Logistic Regression ranks risk best (highest AUC-ROC and AUC-PR), while the stacked ensemble gives the best-calibrated probabilities (lowest Brier score), which matters when scores feed pricing or collection decisions.

> Note: the two datasets are simulated to match the schema and distributions of the Kaggle *Home Credit Default Risk* and *Give Me Some Credit* datasets. Results on the Home Credit–style dataset are near-perfect (AUC ≈ 0.999) and should not be read as real-world performance.

## Tech stack

Python · LightGBM · scikit-learn · imbalanced-learn (SMOTE) · SHAP · pandas · matplotlib

## How to run

```bash
cd github_repo
pip install -r requirements.txt
python src/credit_default_experiment.py
```
