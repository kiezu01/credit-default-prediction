# Credit Default Risk Prediction — Experimental Study

> **Paper**: Comparative study of Gradient Boosted Tree (LightGBM), Logistic Regression,  
> and Stacked Ensemble for credit default prediction.  
> Based on: Lepenioti et al. (2020), *Prescriptive Analytics: Literature Review and Research Challenges*, IJIM 50, 57–70.

## Repository Structure

```
├── data/
│   ├── dataset1_homecredit.csv       # Home Credit Default Risk (simulated, Kaggle-aligned)
│   └── dataset2_givemecredit.csv     # Give Me Some Credit (simulated, Kaggle-aligned)
├── src/
│   └── credit_default_experiment.py  # Full pipeline: train / evaluate / visualize
├── figures/
│   ├── fig1_roc_pr_curves.png
│   ├── fig2_metrics_comparison.png
│   ├── fig3a_shap_Home_Credit.png
│   ├── fig3b_shap_Give_Me_Credit.png
│   ├── fig4_confusion_matrices.png
│   └── fig5_stacking_architecture.png
└── results/
    └── results_summary.json          # All metrics in machine-readable format
```

## Algorithms Implemented

| # | Algorithm | Category (Lepenioti 2020) | Library |
|---|-----------|--------------------------|---------|
| 1 | **Gradient Boosted Tree** | Machine Learning / Data Mining | LightGBM |
| 2 | **Logistic Regression** | Statistical Analysis | scikit-learn |
| 3 | **Stacked Ensemble** (Improvement) | GBT + LR → meta-LR | scikit-learn |

## Key Improvements Applied

- **SMOTE** oversampling to handle class imbalance (~8% default rate)
- **Probability calibration** via Brier Score evaluation
- **SHAP values** for model explainability (regulatory compliance)
- **5-fold cross-validation** out-of-fold stacking to prevent leakage

## Installation

```bash
pip install lightgbm scikit-learn pandas numpy matplotlib seaborn shap imbalanced-learn
```

## How to Run

```bash
# Generate datasets (included, but can regenerate)
# Run full experiment pipeline
python src/credit_default_experiment.py
```

## Results Summary

| Model | Dataset | AUC-ROC | AUC-PR | F1 | Brier |
|---|---|---|---|---|---|
| GBT (LightGBM) | DS1: Home Credit | 0.9983 | 0.9811 | 0.9231 | 0.0118 |
| Logistic Regression | DS1: Home Credit | **0.9991** | **0.9900** | 0.9213 | **0.0102** |
| **Stacked Ensemble** | DS1: Home Credit | 0.9990 | 0.9898 | **0.9240** | 0.0110 |
| GBT (LightGBM) | DS2: Give Me Credit | 0.7800 | 0.2453 | 0.1674 | 0.2552 |
| Logistic Regression | DS2: Give Me Credit | **0.8841** | **0.4888** | **0.3375** | 0.1479 |
| **Stacked Ensemble** | DS2: Give Me Credit | 0.8475 | 0.3681 | 0.3085 | **0.0868** |

> **Primary metric**: AUC-PR (most informative for imbalanced classification)  
> **Key finding**: Stacked Ensemble achieves best Brier Score on both datasets → superior probability calibration

## Reference

Lepenioti, K., Bousdekis, A., Apostolou, D., & Mentzas, G. (2020).  
*Prescriptive analytics: Literature review and research challenges.*  
International Journal of Information Management, 50, 57–70.  
https://doi.org/10.1016/j.ijinfomgt.2019.04.003
