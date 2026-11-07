"""Tests for the credit default experiment pipeline.

The end-to-end tests run on a stratified 2,000-row sample of each dataset so
the whole suite finishes in well under a minute.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import credit_default_experiment as exp  # noqa: E402

DS1 = os.path.join(ROOT, 'data', 'dataset1_homecredit.csv')
DS2 = os.path.join(ROOT, 'data', 'dataset2_givemecredit.csv')


def _sample_csv(src, target, dst, n=2000):
    df = pd.read_csv(src)
    sample, _ = train_test_split(df, train_size=n, stratify=df[target], random_state=0)
    sample.to_csv(dst, index=False)


@pytest.fixture(scope='module')
def small_data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp('data')
    _sample_csv(DS1, 'TARGET', d / 'dataset1_homecredit.csv')
    _sample_csv(DS2, 'SeriousDlqin2yrs', d / 'dataset2_givemecredit.csv')
    return d


# ── Data loading ────────────────────────────────────────────────────────────

def test_load_dataset1_features_and_target():
    X, y = exp.load_dataset1(DS1)
    assert 'TARGET' not in X.columns
    assert len(X) == len(y) == 10_000
    for col in ['CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO', 'EXT_SOURCE_MEAN',
                'EXT_SOURCE_STD', 'DAYS_EMPLOY_RATIO']:
        assert col in X.columns
    assert set(y.unique()) == {0, 1}
    assert not X.isna().any().any()


def test_load_dataset2_features_and_target():
    X, y = exp.load_dataset2(DS2)
    assert 'SeriousDlqin2yrs' not in X.columns
    assert len(X) == len(y) == 10_000
    expected = (X['NumberOfTime30-59DaysPastDueNotWorse']
                + 2 * X['NumberOfTime60-89DaysPastDueNotWorse']
                + 3 * X['NumberOfTimes90DaysLate'])
    pd.testing.assert_series_equal(X['DELINQUENCY_SCORE'], expected, check_names=False)
    assert set(y.unique()) == {0, 1}


# ── Model definitions (regression tests for the imbalance fix) ─────────────

def test_models_do_not_reweight_classes_on_top_of_smote():
    lgbm = exp.build_lgbm()
    assert lgbm.get_params().get('scale_pos_weight', 1.0) == 1.0
    assert lgbm.get_params()['class_weight'] is None
    assert exp.build_lr().get_params()['class_weight'] is None


def test_stacking_runs_smote_inside_each_base_estimator():
    stack = exp.build_stacked_ensemble(exp.build_lgbm(n_estimators=10), exp.build_lr())
    for _, est in stack.estimators:
        assert isinstance(est, ImbPipeline)
        assert isinstance(est.steps[0][1], SMOTE)
    assert stack.cv == 5


# ── Metrics ─────────────────────────────────────────────────────────────────

class _FixedProba:
    def __init__(self, p):
        self.p = np.asarray(p, dtype=float)

    def predict_proba(self, X):
        return np.column_stack([1 - self.p, self.p])


def test_evaluate_perfect_ranking():
    y = np.array([0, 0, 0, 1, 1])
    r = exp.evaluate(_FixedProba([0.1, 0.2, 0.3, 0.8, 0.9]), None, y, 'perfect')
    assert r['AUC_ROC'] == pytest.approx(1.0)
    assert r['Gini'] == pytest.approx(1.0)
    assert r['KS'] == pytest.approx(1.0)
    assert r['F1'] == pytest.approx(1.0)
    assert r['cm'].tolist() == [[3, 0], [0, 2]]


def test_evaluate_gini_and_ks_are_consistent():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(0.3 * y + rng.uniform(0, 0.7, 500), 0, 1)
    r = exp.evaluate(_FixedProba(p), None, y, 'noisy')
    assert r['Gini'] == pytest.approx(2 * r['AUC_ROC'] - 1)
    # KS is the largest vertical gap between the ROC curve and the diagonal,
    # so it can never exceed Gini (twice the area above the diagonal).
    assert 0 < r['KS'] <= r['Gini'] + 1e-9


# ── End to end ──────────────────────────────────────────────────────────────

def test_run_experiment_small_sample(small_data_dir):
    X, y = exp.load_dataset2(str(small_data_dir / 'dataset2_givemecredit.csv'))
    results, lgbm, X_te, y_te, _, _ = exp.run_experiment(X, y, 'small DS2')
    assert set(results) == {'LightGBM', 'LR', 'Stacked'}
    for r in results.values():
        assert r['AUC_ROC'] > 0.6
        assert len(r['y_prob']) == len(y_te) == 400
        assert r['train_time'] >= 0
    # The stacked meta-learner is trained on the real class mix, so its mean
    # predicted probability should sit near the observed default rate rather
    # than near the 50% that SMOTE-trained models drift towards.
    assert abs(results['Stacked']['y_prob'].mean() - y_te.mean()) < 0.05
    assert lgbm.n_estimators >= 1


def test_main_writes_results_and_figures(small_data_dir, tmp_path):
    rows = exp.main(['--data-dir', str(small_data_dir),
                     '--out-dir', str(tmp_path), '--skip-shap'])
    assert len(rows) == 6

    with open(tmp_path / 'results' / 'results_summary.json', encoding='utf-8') as f:
        saved = json.load(f)
    assert saved == rows
    for row in saved:
        assert {'Model', 'Dataset', 'AUC_ROC', 'AUC_PR', 'Gini', 'KS',
                'F1', 'Brier', 'Train_Time_s'} <= set(row)

    figures = {p.name for p in (tmp_path / 'figures').iterdir()}
    assert {'fig1_roc_pr_curves.png', 'fig2_metrics_comparison.png',
            'fig4_confusion_matrices.png', 'fig5_stacking_architecture.png'} <= figures
