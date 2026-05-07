"""
=============================================================================
Credit Default Risk Prediction — Full Experiment Pipeline
=============================================================================
Authors : Group (Lan, Kien, An)
Paper   : Lepenioti et al. (2020) — Prescriptive Analytics: Literature Review
          and Research Challenges, IJIM 50, 57–70.
Datasets: 
  - Dataset 1: Home Credit Default Risk (Kaggle) — simulated
  - Dataset 2: Give Me Some Credit (Kaggle) — simulated
Algorithms:
  - Algo 1 : Gradient Boosted Tree  [LightGBM] (ML/Data Mining group)
  - Algo 2 : Logistic Regression    (Statistical Analysis group)
  - Algo 3 : Stacked Ensemble (LightGBM + LR meta-learner) [Improvement]
=============================================================================
"""

# ─── Imports ────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, os, time, json
warnings.filterwarnings('ignore')

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    classification_report, roc_curve, precision_recall_curve,
    confusion_matrix, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
import shap

# Project-relative paths (portable across machines)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATASET1_PATH = os.path.join(PROJECT_DIR, 'data/dataset1_homecredit.csv')
DATASET2_PATH = os.path.join(PROJECT_DIR, 'data/dataset2_givemecredit.csv')
FIGURES_DIR = os.path.join(PROJECT_DIR, 'figures')
SUMMARY_PATH = os.path.join(PROJECT_DIR, 'results_summary.json')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── Color palette ──────────────────────────────────────────────────────────
C_GBT    = '#2E75B6'   # LightGBM
C_LR     = '#E74C3C'   # Logistic Regression
C_STACK  = '#27AE60'   # Stacked Ensemble
C_SHADE  = '#BDC3C7'
COLORS   = [C_GBT, C_LR, C_STACK]

# ═══════════════════════════════════════════════════════════════════════════
#  1. DATA LOADING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def load_dataset1(path=DATASET1_PATH):
    """Home Credit Default Risk — 13 features, target=TARGET"""
    df = pd.read_csv(path)
    X  = df.drop('TARGET', axis=1)
    y  = df['TARGET']
    # Feature engineering
    X['CREDIT_INCOME_RATIO'] = X['AMT_CREDIT'] / (X['AMT_INCOME_TOTAL'] + 1)
    X['ANNUITY_INCOME_RATIO'] = X['AMT_ANNUITY'] / (X['AMT_INCOME_TOTAL'] + 1)
    X['EXT_SOURCE_MEAN']  = X[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].mean(axis=1)
    X['EXT_SOURCE_STD']   = X[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].std(axis=1)
    X['DAYS_EMPLOY_RATIO'] = X['DAYS_EMPLOYED'] / (X['AGE'] * 365 + 1)
    print(f"[DS1] Shape: {X.shape} | Default rate: {y.mean():.3f} | "
          f"Imbalance ratio: 1:{int(1/y.mean())}")
    return X, y

def load_dataset2(path=DATASET2_PATH):
    """Give Me Some Credit — 10 features, target=SeriousDlqin2yrs"""
    df = pd.read_csv(path)
    y  = df['SeriousDlqin2yrs']
    X  = df.drop('SeriousDlqin2yrs', axis=1)
    # Feature engineering
    X['DELINQUENCY_SCORE'] = (
        X['NumberOfTime30-59DaysPastDueNotWorse'] +
        2 * X['NumberOfTime60-89DaysPastDueNotWorse'] +
        3 * X['NumberOfTimes90DaysLate']
    )
    X['INCOME_PER_DEPENDENT'] = X['MonthlyIncome'] / (X['NumberOfDependents'] + 1)
    print(f"[DS2] Shape: {X.shape} | Default rate: {y.mean():.3f} | "
          f"Imbalance ratio: 1:{int(1/y.mean())}")
    return X, y

def preprocess(X_train, X_test, y_train, apply_smote=True):
    """Scale + optional SMOTE for class imbalance"""
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    if apply_smote:
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_train_res, y_train_res = sm.fit_resample(X_train_sc, y_train)
        print(f"   After SMOTE: {X_train_res.shape} | "
              f"Default rate: {y_train_res.mean():.3f}")
    else:
        X_train_res, y_train_res = X_train_sc, y_train
    return X_train_sc, X_train_res, y_train_res, X_test_sc, scaler

# ═══════════════════════════════════════════════════════════════════════════
#  2. MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

def build_lgbm(scale_pos_weight=10):
    """Gradient Boosted Tree — LightGBM (Lepenioti: ML/Data Mining group)"""
    return lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=6,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=scale_pos_weight,   # handle class imbalance
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )

def build_lr():
    """Logistic Regression (Lepenioti: Statistical Analysis group)"""
    return LogisticRegression(
        C=0.1,
        max_iter=1000,
        class_weight='balanced',   # handle class imbalance
        solver='lbfgs',
        random_state=42,
        n_jobs=-1
    )

def build_stacked_ensemble(lgbm_model, lr_model):
    """
    Stacked Ensemble (Improvement):
    Level-0: LightGBM + Logistic Regression
    Level-1: Logistic Regression as meta-learner
    Uses cross-val predictions (out-of-fold) to train meta-learner → avoids leakage
    """
    return StackingClassifier(
        estimators=[
            ('lgbm', lgbm_model),
            ('lr',   lr_model),
        ],
        final_estimator=LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        ),
        cv=5,
        stack_method='predict_proba',
        passthrough=False,    # meta-learner sees only base model predictions
        n_jobs=-1
    )

# ═══════════════════════════════════════════════════════════════════════════
#  3. EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(model, X_test, y_test, model_name, threshold=0.5):
    """Comprehensive evaluation metrics"""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    auc_roc = roc_auc_score(y_test, y_prob)
    auc_pr  = average_precision_score(y_test, y_prob)
    f1      = f1_score(y_test, y_pred, zero_division=0)
    brier   = brier_score_loss(y_test, y_prob)
    report  = classification_report(y_test, y_pred, digits=4, zero_division=0)
    cm      = confusion_matrix(y_test, y_pred)

    print(f"\n{'─'*55}")
    print(f"  {model_name}")
    print(f"{'─'*55}")
    print(f"  AUC-ROC  : {auc_roc:.4f}")
    print(f"  AUC-PR   : {auc_pr:.4f}   ← primary metric (imbalanced)")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  Brier    : {brier:.4f}   ← calibration (lower=better)")
    print(f"\n{report}")

    return {
        'model_name': model_name,
        'AUC_ROC': auc_roc,
        'AUC_PR':  auc_pr,
        'F1':      f1,
        'Brier':   brier,
        'y_prob':  y_prob,
        'y_pred':  y_pred,
        'cm':      cm,
    }

# ═══════════════════════════════════════════════════════════════════════════
#  4. FULL EXPERIMENT ON ONE DATASET
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(X, y, ds_name, imbalance_ratio=10):
    print(f"\n{'═'*60}")
    print(f"  EXPERIMENT: {ds_name}")
    print(f"{'═'*60}")

    # Train/test split (stratified)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocessing
    X_tr_sc, X_tr_res, y_tr_res, X_te_sc, scaler = preprocess(
        X_tr, X_te, y_tr, apply_smote=True
    )

    results = {}

    # ── Algo 1: LightGBM ──────────────────────────────────────────────────
    print("\n[1] Training LightGBM (Gradient Boosted Tree)...")
    t0 = time.time()
    lgbm = build_lgbm(scale_pos_weight=imbalance_ratio)
    lgbm.fit(
        X_tr_res, y_tr_res,
        eval_set=[(X_te_sc, y_te)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
    )
    t_lgbm = time.time() - t0
    print(f"   Training time: {t_lgbm:.2f}s | Trees: {lgbm.best_iteration_}")
    results['LightGBM'] = evaluate(lgbm, X_te_sc, y_te, 'LightGBM (Gradient Boosted Tree)')

    # ── Algo 2: Logistic Regression ───────────────────────────────────────
    print("\n[2] Training Logistic Regression...")
    t0 = time.time()
    lr = build_lr()
    lr.fit(X_tr_res, y_tr_res)
    t_lr = time.time() - t0
    print(f"   Training time: {t_lr:.2f}s")
    results['LR'] = evaluate(lr, X_te_sc, y_te, 'Logistic Regression')

    # ── Algo 3: Stacked Ensemble (Improvement) ────────────────────────────
    print("\n[3] Training Stacked Ensemble (LightGBM + LR → meta-LR)...")
    t0 = time.time()
    stack = build_stacked_ensemble(
        build_lgbm(scale_pos_weight=imbalance_ratio),
        build_lr()
    )
    stack.fit(X_tr_res, y_tr_res)
    t_stack = time.time() - t0
    print(f"   Training time: {t_stack:.2f}s")
    results['Stacked'] = evaluate(stack, X_te_sc, y_te, 'Stacked Ensemble (Improvement)')

    # Store training times
    results['LightGBM']['train_time'] = t_lgbm
    results['LR']['train_time']       = t_lr
    results['Stacked']['train_time']  = t_stack

    return results, lgbm, X_te_sc, y_te, X_tr_res, scaler

# ═══════════════════════════════════════════════════════════════════════════
#  5. VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════

def plot_roc_pr(results_ds1, results_ds2, y_te1, y_te2):
    """ROC and PR curves for both datasets, all 3 models"""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle('Model Performance: ROC & Precision-Recall Curves',
                 fontsize=14, fontweight='bold', y=1.01)

    datasets   = [('DS1: Home Credit', results_ds1, y_te1),
                  ('DS2: Give Me Some Credit', results_ds2, y_te2)]
    model_keys = [('LightGBM', C_GBT, 'GBT (LightGBM)'),
                  ('LR',       C_LR,  'Logistic Regression'),
                  ('Stacked',  C_STACK,'Stacked Ensemble')]

    for col, (ds_label, results, y_te) in enumerate(datasets):
        # ROC
        ax_roc = axes[0][col]
        ax_roc.plot([0,1],[0,1],'--', color='gray', lw=1, label='Random')
        for key, color, label in model_keys:
            fpr, tpr, _ = roc_curve(y_te, results[key]['y_prob'])
            auc = results[key]['AUC_ROC']
            ax_roc.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC={auc:.3f})')
        ax_roc.set_title(f'ROC Curve — {ds_label}', fontweight='bold')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.legend(fontsize=8.5)
        ax_roc.grid(alpha=0.3)

        # PR
        ax_pr = axes[1][col]
        baseline = y_te.mean()
        ax_pr.axhline(baseline, color='gray', lw=1, ls='--', label=f'Baseline ({baseline:.3f})')
        for key, color, label in model_keys:
            prec, rec, _ = precision_recall_curve(y_te, results[key]['y_prob'])
            ap = results[key]['AUC_PR']
            ax_pr.plot(rec, prec, color=color, lw=2, label=f'{label} (AP={ap:.3f})')
        ax_pr.set_title(f'Precision-Recall Curve — {ds_label}', fontweight='bold')
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.legend(fontsize=8.5)
        ax_pr.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_roc_pr_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[Fig 1] ROC & PR curves saved.")

def plot_metrics_comparison(results_ds1, results_ds2):
    """Bar chart comparing AUC-ROC, AUC-PR, F1, Brier across all models & datasets"""
    metrics   = ['AUC_ROC', 'AUC_PR', 'F1', 'Brier']
    m_labels  = ['AUC-ROC', 'AUC-PR', 'F1-Score', 'Brier Score\n(↓ better)']
    models    = ['LightGBM', 'LR', 'Stacked']
    m_display = ['GBT (LightGBM)', 'Logistic Regression', 'Stacked Ensemble']
    datasets  = [('DS1: Home Credit', results_ds1),
                 ('DS2: Give Me Some Credit', results_ds2)]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Metric Comparison: All Models × Both Datasets',
                 fontsize=13, fontweight='bold')

    for row, (ds_label, results) in enumerate(datasets):
        for col, (metric, m_label) in enumerate(zip(metrics, m_labels)):
            ax = axes[row][col]
            vals   = [results[m][metric] for m in models]
            bars   = ax.bar(range(3), vals, color=COLORS, width=0.55, edgecolor='white', lw=1.2)
            ax.set_xticks(range(3))
            ax.set_xticklabels(['GBT', 'LR', 'Stack'], fontsize=9)
            ax.set_title(f'{m_label}\n{ds_label}', fontsize=9, fontweight='bold')
            ax.set_ylim(0, max(vals)*1.25 if max(vals) > 0 else 0.1)
            ax.grid(axis='y', alpha=0.3)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            # Highlight best (lowest for Brier, highest for others)
            if metric == 'Brier':
                best_idx = np.argmin(vals)
            else:
                best_idx = np.argmax(vals)
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(2.5)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_metrics_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[Fig 2] Metrics comparison saved.")

def plot_shap(lgbm_model, X_test_sc, feature_names, ds_name, fig_id):
    """SHAP summary plot for LightGBM"""
    print(f"  Computing SHAP values for {ds_name}...")
    explainer   = shap.TreeExplainer(lgbm_model)
    shap_values = explainer.shap_values(X_test_sc[:500])   # sample for speed
    if isinstance(shap_values, list):
        shap_values = shap_values[1]   # class=1 (default)

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test_sc[:500],
                      feature_names=feature_names,
                      plot_type='dot', show=False, max_display=10)
    plt.title(f'SHAP Feature Importance — {ds_name}', fontweight='bold', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'{fig_id}_shap_{ds_name.replace(" ","_")}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [{fig_id}] SHAP plot saved.")

def plot_confusion_matrices(results_ds1, results_ds2):
    """Confusion matrices for all models on both datasets"""
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle('Confusion Matrices — All Models × Both Datasets',
                 fontsize=13, fontweight='bold')
    model_keys   = ['LightGBM', 'LR', 'Stacked']
    model_labels = ['GBT (LightGBM)', 'Logistic Regression', 'Stacked Ensemble']
    ds_list      = [('DS1: Home Credit', results_ds1),
                    ('DS2: Give Me Some Credit', results_ds2)]

    for row, (ds_label, results) in enumerate(ds_list):
        for col, (key, label) in enumerate(zip(model_keys, model_labels)):
            ax  = axes[row][col]
            cm  = results[key]['cm']
            im  = ax.imshow(cm, cmap='Blues')
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(['Pred 0','Pred 1'])
            ax.set_yticklabels(['True 0','True 1'])
            ax.set_title(f'{label}\n{ds_label}', fontsize=9, fontweight='bold')
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                            fontsize=13, fontweight='bold',
                            color='white' if cm[i,j] > cm.max()/2 else 'black')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig4_confusion_matrices.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[Fig 4] Confusion matrices saved.")

def plot_stacking_architecture():
    """Diagram of Stacked Ensemble architecture"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('Stacked Ensemble Architecture (Improvement)',
                 fontsize=13, fontweight='bold', pad=15)

    # Level 0 models
    box_style = dict(boxstyle='round,pad=0.5', facecolor=C_GBT, edgecolor='white', alpha=0.9)
    ax.text(2, 4.5, 'Input Features\n(X_train)', ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ECF0F1', edgecolor='#7F8C8D'))
    # Arrow
    ax.annotate('', xy=(1.2, 3.2), xytext=(1.7, 4.1),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(2.8, 3.2), xytext=(2.3, 4.1),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    ax.text(1.0, 2.8, 'Level-0\nEstimator 1:\nGBT (LightGBM)', ha='center', va='center',
            fontsize=9, bbox=dict(boxstyle='round,pad=0.5', facecolor=C_GBT,
                                   edgecolor='white', alpha=0.9), color='white')
    ax.text(3.0, 2.8, 'Level-0\nEstimator 2:\nLogistic Reg.', ha='center', va='center',
            fontsize=9, bbox=dict(boxstyle='round,pad=0.5', facecolor=C_LR,
                                   edgecolor='white', alpha=0.9), color='white')

    ax.text(1.0, 1.8, '5-Fold CV\nOut-of-Fold\nPredictions', ha='center', va='center',
            fontsize=8, color='#555', style='italic')
    ax.text(3.0, 1.8, '5-Fold CV\nOut-of-Fold\nPredictions', ha='center', va='center',
            fontsize=8, color='#555', style='italic')

    ax.annotate('', xy=(2.0, 1.1), xytext=(1.0, 1.4),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(2.0, 1.1), xytext=(3.0, 1.4),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    ax.text(2.0, 0.7, 'Level-1 Meta-Learner:\nLogistic Regression', ha='center', va='center',
            fontsize=9, bbox=dict(boxstyle='round,pad=0.5', facecolor=C_STACK,
                                   edgecolor='white', alpha=0.9), color='white')
    ax.annotate('', xy=(2.0, 0.15), xytext=(2.0, 0.35),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.text(2.0, 0.05, 'Final Prediction (P(default))', ha='center', va='center',
            fontsize=9, fontweight='bold', color=C_STACK)

    # Annotations
    ax.text(5.5, 4.5, 'Key Advantages of Stacking:', fontsize=10, fontweight='bold')
    advantages = [
        '✓ Combines complementary models (GBT + LR)',
        '✓ 5-fold CV prevents data leakage',
        '✓ Meta-learner learns optimal weighting',
        '✓ Reduces variance vs. single model',
        '✓ Inherits LightGBM\'s feature importance',
    ]
    for i, adv in enumerate(advantages):
        ax.text(5.5, 3.8 - i*0.55, adv, fontsize=9, color='#333')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig5_stacking_architecture.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[Fig 5] Stacking architecture diagram saved.")

# ═══════════════════════════════════════════════════════════════════════════
#  6. SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════

def print_summary_table(results_ds1, results_ds2):
    models = ['LightGBM', 'LR', 'Stacked']
    labels = ['GBT (LightGBM)', 'Logistic Regression', 'Stacked Ensemble']
    print(f"\n{'═'*75}")
    print(f"{'FINAL SUMMARY TABLE':^75}")
    print(f"{'═'*75}")
    print(f"{'Model':<22} {'Dataset':<30} {'AUC-ROC':>8} {'AUC-PR':>8} {'F1':>7} {'Brier':>8}")
    print(f"{'─'*75}")
    all_rows = []
    for key, label in zip(models, labels):
        for ds_label, results in [('DS1: Home Credit', results_ds1),
                                   ('DS2: Give Me Credit', results_ds2)]:
            r = results[key]
            print(f"{label:<22} {ds_label:<30} {r['AUC_ROC']:>8.4f} {r['AUC_PR']:>8.4f} "
                  f"{r['F1']:>7.4f} {r['Brier']:>8.4f}")
            all_rows.append({
                'Model': label, 'Dataset': ds_label,
                'AUC_ROC': r['AUC_ROC'], 'AUC_PR': r['AUC_PR'],
                'F1': r['F1'], 'Brier': r['Brier'],
                'Train_Time_s': round(r['train_time'], 2)
            })
    print(f"{'═'*75}")
    # Save to JSON for report
    with open(SUMMARY_PATH, 'w') as f:
        json.dump(all_rows, f, indent=2)
    return all_rows

# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("="*60)
    print("  Credit Default Risk — Full Experiment Pipeline")
    print("="*60)

    # Load datasets
    X1, y1 = load_dataset1()
    X2, y2 = load_dataset2()

    # Run experiments
    results_ds1, lgbm1, X_te1, y_te1, X_tr1, sc1 = run_experiment(X1, y1, 'DS1: Home Credit', imbalance_ratio=11)
    results_ds2, lgbm2, X_te2, y_te2, X_tr2, sc2 = run_experiment(X2, y2, 'DS2: Give Me Some Credit', imbalance_ratio=13)

    # Summary
    summary = print_summary_table(results_ds1, results_ds2)

    # Plots
    print("\n[Generating figures...]")
    plot_roc_pr(results_ds1, results_ds2, y_te1, y_te2)
    plot_metrics_comparison(results_ds1, results_ds2)
    plot_shap(lgbm1, X_te1, list(X1.columns), 'Home_Credit', 'fig3a')
    plot_shap(lgbm2, X_te2, list(X2.columns), 'Give_Me_Credit', 'fig3b')
    plot_confusion_matrices(results_ds1, results_ds2)
    plot_stacking_architecture()

    print(f"\n[All experiments completed. Figures saved to {FIGURES_DIR}]")
    print(f"[Results saved to {SUMMARY_PATH}]")
