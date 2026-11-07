# Credit Default Risk Prediction – LightGBM vs. Logistic Regression vs. Stacked Ensemble

**English** · [Tiếng Việt](README.vi.md)

[![CI](https://github.com/kiezu01/credit-default-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/kiezu01/credit-default-prediction/actions/workflows/ci.yml)

A comparison of three credit scoring approaches on two credit datasets, with class-imbalance handling, calibration analysis and SHAP explainability. Group coursework for a Master's program in Information Systems, framed around the analytics taxonomy of Lepenioti et al. (2020).

## Approach

| # | Model | Category (Lepenioti 2020) | Library |
|---|---|---|---|
| 1 | Gradient Boosted Trees | Machine Learning / Data Mining | LightGBM |
| 2 | Logistic Regression | Statistical Analysis | scikit-learn |
| 3 | Stacked Ensemble (LightGBM + LR → LR meta-learner) | Improvement | scikit-learn + imbalanced-learn |

- **80/20 stratified train/test split.** The test set is used only for the final metrics.
- **SMOTE** for the imbalanced target (7–8% default rate). It is the only imbalance correction; the models do not reweight classes on top of it.
- **Early stopping for LightGBM** on a 15% validation slice of the training data, never on the test set.
- **5-fold out-of-fold stacking**, with SMOTE running *inside* each fold so that synthetic points never reach a held-out fold, and the meta-learner trained on the real class mix.
- **Metrics:** AUC-ROC, AUC-PR (primary, because the classes are imbalanced), Gini, KS, F1 at a 0.5 threshold, and the Brier score for calibration.
- **SHAP** summary plots for the LightGBM models.

## Results

Test set, 2,000 applicants per dataset:

| Model | Dataset | AUC-ROC | AUC-PR | Gini | KS | F1 | Brier ↓ |
|---|---|---|---|---|---|---|---|
| LightGBM | Home Credit | 0.998 | 0.984 | 0.997 | 0.970 | 0.927 | 0.009 |
| Logistic Regression | Home Credit | **0.999** | **0.990** | **0.998** | **0.982** | 0.921 | 0.010 |
| Stacked Ensemble | Home Credit | **0.999** | **0.990** | **0.998** | 0.980 | **0.938** | **0.008** |
| LightGBM | Give Me Some Credit | 0.851 | 0.382 | 0.702 | 0.555 | **0.345** | 0.054 |
| Logistic Regression | Give Me Some Credit | **0.884** | **0.489** | **0.768** | 0.602 | 0.338 | 0.148 |
| Stacked Ensemble | Give Me Some Credit | **0.884** | 0.462 | 0.767 | **0.604** | 0.291 | **0.048** |

**Takeaways**

- On the harder Give Me Some Credit data, Logistic Regression and the stacked ensemble rank risk equally well (AUC 0.884, KS ≈ 0.60). LightGBM trails them.
- The stacked ensemble has by far the best calibration: its Brier score is 0.048 against 0.148 for LR. LR is trained on SMOTE-balanced data, so its probabilities sit far above the real 7% default rate. The meta-learner is fitted on the real class mix and corrects that. Calibration matters whenever the score feeds pricing, provisioning or collections decisions, not just a reject/accept cut-off.
- F1 at 0.5 is lowest for the ensemble *because* it is calibrated: few applicants get P(default) > 0.5 when the base rate is 7%. In practice the threshold should be set from the business cost of false accepts versus false rejects.

All numbers are in [`results/results_summary.json`](results/results_summary.json) and the figures in [`figures/`](figures/):

| Figure | Content |
|---|---|
| `fig1_roc_pr_curves.png` | ROC and precision-recall curves, all models × both datasets |
| `fig2_metrics_comparison.png` | Bar charts of AUC-ROC, AUC-PR, F1 and Brier |
| `fig3a/b_shap_*.png` | SHAP summary plots for LightGBM |
| `fig4_confusion_matrices.png` | Confusion matrices at threshold 0.5 |
| `fig5_stacking_architecture.png` | Stacked ensemble design |

> **About the data.** Both datasets (10,000 rows each) are *simulated* to match the schema and distributions of the Kaggle [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) and [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) datasets. The Home Credit–style data is close to linearly separable, so its near-perfect scores (AUC ≈ 0.999) say nothing about real-world performance. The Give Me Some Credit–style results are the meaningful comparison.

### Methodology fixes (November 2026)

The first version of this repo had three methodology bugs. They have been fixed, and the table above comes from the corrected pipeline:

1. **Test-set leakage.** LightGBM's early stopping monitored the test set. It now monitors a validation split carved from the training data.
2. **Double imbalance correction.** The data was SMOTE-balanced to 50/50 *and* the models were reweighted again (`scale_pos_weight` = 11–13, `class_weight='balanced'`). On Give Me Some Credit this made LightGBM's validation loss worsen after the first tree, so the reported LightGBM result came from a one-tree model (AUC 0.780, Brier 0.255).
3. **Leaky stacking.** The ensemble was cross-validated on SMOTE output, so synthetic points built from held-out neighbours leaked across folds.

Impact on Give Me Some Credit: LightGBM AUC 0.780 → 0.851 and Brier 0.255 → 0.054; stacked ensemble AUC 0.847 → 0.884 and Brier 0.087 → 0.048. The written reports in [`docs/`](docs/) describe the original version.

## Repository structure

```
├── data/
│   ├── dataset1_homecredit.csv     # Home Credit–style, 13 features + TARGET
│   └── dataset2_givemecredit.csv   # Give Me Some Credit–style, 10 features + SeriousDlqin2yrs
├── src/credit_default_experiment.py  # Load → train → evaluate → plot
├── tests/test_experiment.py          # pytest suite (~20 s)
├── figures/                          # Generated figures
├── results/results_summary.json      # Generated metrics
└── docs/                             # Course reports (Vietnamese, IEEE format)
```

## How to run

Requires Python 3.10+ (tested on 3.12).

```bash
pip install -r requirements.txt
python src/credit_default_experiment.py               # full run, under a minute
python src/credit_default_experiment.py --skip-shap   # skip the SHAP plots
python src/credit_default_experiment.py --data-dir path/to/csvs --out-dir runs/exp1
```

Tests and lint:

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check --select F src tests
```

## Tech stack

Python · LightGBM · scikit-learn · imbalanced-learn · SHAP · pandas · matplotlib · pytest · GitHub Actions

## Reference

Lepenioti, K., Bousdekis, A., Apostolou, D., & Mentzas, G. (2020). Prescriptive analytics: Literature review and research challenges. *International Journal of Information Management*, 50, 57–70. https://doi.org/10.1016/j.ijinfomgt.2019.04.003
