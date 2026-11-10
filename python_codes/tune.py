"""
================================================================================
TUNE.PY - ROUND 2: focused search on the winning config (no manual stats)
================================================================================

Round 1 (experiments.py) showed dropping the hand-rolled target-stat features
is the big win. This round tunes XGBoost around that config and benchmarks
LightGBM. Winner is chosen by CV only; the test set is evaluated once.
================================================================================
"""

import warnings
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import experiments as ex

warnings.filterwarnings('ignore')


def run_lgb(name, params, train_df, use_stats, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    num, cat = ex.feature_lists(use_stats)
    aucs = []
    for tr_idx, val_idx in skf.split(train_df, train_df['target']):
        fold_tr, fold_val = train_df.iloc[tr_idx], train_df.iloc[val_idx]
        feat_tr = ex.build_features(fold_tr, fold_tr, use_stats)
        feat_val = ex.build_features(fold_tr, fold_val, use_stats)
        pre = ex.ColumnTransformer([
            ('cb', ex.CatBoostEncoder(cols=cat, random_state=42), cat),
            ('scaler', ex.StandardScaler(), num)])
        m = ex.Pipeline([('preprocessor', pre), ('est', LGBMClassifier(random_state=42, verbose=-1, **params))])
        m.fit(feat_tr[num + cat], feat_tr['target'])
        aucs.append(roc_auc_score(feat_val['target'],
                                  m.predict_proba(feat_val[num + cat])[:, 1]))
    print(f"CV  {name:<36} {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    return np.mean(aucs)


def main():
    df = ex.base_prepare(ex.load_data())
    closed = df[df['deal_stage'].isin(['won', 'lost'])].copy()
    ref_date = pd.to_datetime(df[['engage_date', 'close_date']].max().max())
    closed = ex.temporal(closed)
    closed = ex.deal_age(closed, ref_date)
    y = closed['target']
    train, test = ex.train_test_split(closed, test_size=0.2, random_state=1, stratify=y)

    rng = np.random.RandomState(7)
    results = {}

    # Round-1 winner, as reference
    r1 = dict(n_estimators=500, learning_rate=0.03, max_depth=3, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0, reg_alpha=0.5)
    results['r1_winner'] = (ex.run_config("R1 winner (depth3)", 'xgb', r1, train, False), r1)

    # LightGBM benchmark
    lgb_params = dict(n_estimators=500, learning_rate=0.03, num_leaves=15,
                      subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0)
    results['lgb'] = (run_lgb("LGBM (leaves15)", lgb_params, train, False), lgb_params)

    # Small randomized search around the winner (all decided by CV)
    best_extra = None
    for i in range(8):
        p = dict(
            n_estimators=int(rng.choice([400, 600, 800])),
            learning_rate=float(rng.choice([0.02, 0.04, 0.06])),
            max_depth=int(rng.choice([2, 3, 4])),
            subsample=float(rng.choice([0.7, 0.8, 0.9])),
            colsample_bytree=float(rng.choice([0.5, 0.7, 0.9])),
            min_child_weight=int(rng.choice([3, 5, 10])),
            reg_lambda=float(rng.choice([1.0, 3.0, 5.0])),
            reg_alpha=float(rng.choice([0.0, 0.5, 1.0])),
        )
        auc = ex.run_config(f"search-{i} (d{p['max_depth']} lr{p['learning_rate']})",
                            'xgb', p, train, False)
        results[f'search-{i}'] = (auc, p)

    best_name = max(results, key=lambda k: results[k][0])
    best_auc, best_params = results[best_name]
    print(f"\nWinner by CV: {best_name} ({best_auc:.4f})")
    print(f"params: {best_params}\n")

    # The ONE test-set evaluation for the round-2 winner
    ex.final_test("round2 winner", 'xgb', best_params, train, test, False)


if __name__ == '__main__':
    import pandas as pd  # noqa: F401  (used via ex)
    main()
