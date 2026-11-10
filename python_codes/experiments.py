"""
================================================================================
EXPERIMENTS.PY - HONEST MODEL IMPROVEMENT HARNESS
================================================================================

Measures candidate models with STRATIFIED 5-FOLD CV on the training set only.
The held-out test set is touched exactly ONCE at the end, for the winner.

Key honesty rule (same fix as models.py): within each CV fold, the win-rate
statistics / agent speeds are computed on the FOLD-TRAIN rows only and mapped
onto the fold-validation rows - never the other way round.

Run:  python experiments.py
================================================================================
"""

import os
import pyodbc
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from category_encoders import CatBoostEncoder
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

EPS = 1e-6
CAT_COLS = ['sales_agent', 'product', 'account', 'sector', 'office_location',
            'manager', 'regional_office', 'company_size']

# col -> (win_rate feature, total_deals feature, win feature)
# EXACTLY the alias names models.py uses.
ALIAS = {
    'sales_agent': ('agent_win_rate', 'agent_total_deals', 'agent_win'),
    'account': ('account_win_rate', 'account_deal_count', 'account_total_win'),
    'sector': ('sector_win_rate', 'sector_deal_count', 'sector_total_win'),
    'office_location': ('office_win_rate', 'office_deal_count', 'office_total_win'),
    'regional_office': ('region_win_rate', 'region_deal_count', 'region_total_win'),
    'company_size': ('company_size_win_rate', 'company_size_deal_count', 'company_size_total_win'),
    'product': ('product_win_rate', 'product_deal_count', 'product_total_win'),
}


def load_data():
    conn = pyodbc.connect(
        "Driver={SQL Server};"
        f"Server={os.environ.get('DB_SERVER', 'ELICE99\\SQLEXPRESS')};"
        "Database=CRM_Sales_Opportunity;"
        "Trusted_Connection=yes;"
    )
    query = '''
    SELECT p.*, 
        a.sector, a.year_established, a.account_tier, a.employees, a.office_location,
        s.manager, s.regional_office
    FROM dbo.sales_pipeline p
    LEFT JOIN accounts a ON a.account = p.account
    LEFT JOIN sales_teams s ON p.sales_agent = s.sales_agent
    WHERE deal_stage NOT IN ('Prospecting')
    '''
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def base_prepare(df):
    """Standardisation + core features (same logic as models.py)."""
    df = df.copy()
    df['engage_date'] = pd.to_datetime(df['engage_date'], errors='coerce')
    df['close_date'] = pd.to_datetime(df['close_date'], errors='coerce')
    df['employees_log'] = np.log1p(df['employees'].fillna(0))
    bins = [0, 50, 250, 1000, 5000, 15000, np.inf]
    labels = ['micro', 'small', 'medium', 'large', 'enterprise', 'mega']
    df['company_size'] = pd.cut(df['employees'].fillna(0), bins=bins, labels=labels).astype(str)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna('').astype(str).str.lower().str.replace(' ', '_')
    df['target'] = (df['deal_stage'] == 'won').astype(int)
    return df


def temporal(df):
    df = df.copy()
    df['month_engaged'] = df['engage_date'].dt.month
    df['quarter_engaged'] = df['engage_date'].dt.quarter
    df['day_of_week_engaged'] = df['engage_date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week_engaged'].isin([5, 6])).astype(int)
    df['days_into_year'] = df['engage_date'].dt.dayofyear
    return df


def deal_age(df, ref_date):
    df = df.copy()
    closed = (df['close_date'] - df['engage_date']).dt.days
    active = (ref_date - df['engage_date']).dt.days
    df['deal_age'] = closed.fillna(active)
    df['deal_age'] = df['deal_age'].fillna(df['deal_age'].median()).astype(float)
    return df


def build_features(train, apply_df, use_stats=True):
    """Compute target stats on TRAIN only, map onto APPLY rows."""
    tr, ap = train.copy(), apply_df.copy()
    ap['agent_avg_days_to_close'] = tr['deal_age'].mean()  # global fallback

    if use_stats:
        for col in CAT_COLS:
            if col not in ap.columns:
                ap[col] = 'missing'
            stats = tr.groupby(col)['target'].agg(['mean', 'count', 'sum'])
            ap[f'{col}_win_rate'] = ap[col].map(stats['mean']).fillna(tr['target'].mean())
            ap[f'{col}_total_deals'] = ap[col].map(stats['count']).fillna(0)
            ap[f'{col}_win'] = ap[col].map(stats['sum']).fillna(0)
        for col, (wr, td, tw) in ALIAS.items():
            ap[wr] = ap.get(f'{col}_win_rate', 0.5)
            ap[td] = ap.get(f'{col}_total_deals', 0)
            ap[tw] = ap.get(f'{col}_win', 0)
    else:
        for wr, td, tw in ALIAS.values():
            ap[wr], ap[td], ap[tw] = 0.5, 0, 0

    # interactions (same formulas as models.py)
    ap['agent_product_synergy'] = ap['agent_win_rate'] * ap['product_win_rate']
    ap['agent_win_efficiency'] = (1 - ap['agent_win_rate']) * ap['agent_total_deals']
    ap['agent_vs_sector'] = ap['agent_win_rate'] - ap['sector_win_rate']
    ap['deal_complexity'] = ap['employees_log'] * ap['deal_age'] / (ap['agent_total_deals'] + 1)
    ap['office_load'] = ap['office_deal_count'] / (ap['office_total_win'] + 1)
    ap['product_sector_fit'] = ap['product_win_rate'] * ap['sector_win_rate']
    ap['region_size_match'] = ap['region_win_rate'] * ap['company_size_win_rate']
    ap['quarter_risk'] = ap['quarter_engaged'].map({1: 0.8, 2: 0.9, 3: 1.0, 4: 1.2}).fillna(1.0)
    ap['is_quarter_end'] = ap['month_engaged'].isin([3, 6, 9, 12]).astype(int)
    ap['sales_velocity_ratio'] = ap['deal_age'] / (ap['agent_avg_days_to_close'] + EPS)
    ap['pace_weighted_agent_score'] = ap['agent_win_rate'] / (ap['sales_velocity_ratio'] + EPS)
    ap['velocity_complexity_index'] = ap['sales_velocity_ratio'] * ap['employees_log']
    return ap


def feature_lists(use_stats):
    num = ['year_established', 'employees_log', 'deal_age', 'agent_avg_days_to_close',
           'agent_product_synergy', 'agent_win_efficiency', 'agent_vs_sector', 'deal_complexity',
           'office_load', 'product_sector_fit', 'region_size_match', 'quarter_risk',
           'is_quarter_end', 'sales_velocity_ratio', 'pace_weighted_agent_score',
           'velocity_complexity_index', 'month_engaged', 'quarter_engaged',
           'day_of_week_engaged', 'is_weekend', 'days_into_year']
    if use_stats:
        for wr, td, tw in ALIAS.values():
            num += [wr, td, tw]
    return num, list(CAT_COLS)


def make_pipeline(kind, params, num, cat):
    pre = ColumnTransformer([
        ('cb', CatBoostEncoder(cols=cat, random_state=42), cat),
        ('scaler', StandardScaler(), num)])
    if kind == 'xgb':
        est = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1, **params)
    elif kind == 'lr':
        est = LogisticRegression(max_iter=2000, random_state=42)
    return Pipeline([('preprocessor', pre), ('est', est)])


def run_config(name, kind, cv_params, train_df, use_stats, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    num, cat = feature_lists(use_stats)
    aucs = []
    for tr_idx, val_idx in skf.split(train_df, train_df['target']):
        fold_tr, fold_val = train_df.iloc[tr_idx], train_df.iloc[val_idx]
        feat_tr = build_features(fold_tr, fold_tr, use_stats)
        feat_val = build_features(fold_tr, fold_val, use_stats)
        m = make_pipeline(kind, cv_params, num, cat)
        m.fit(feat_tr[num + cat], feat_tr['target'])
        aucs.append(roc_auc_score(feat_val['target'],
                                  m.predict_proba(feat_val[num + cat])[:, 1]))
    print(f"CV  {name:<36} {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    return np.mean(aucs)


def final_test(name, kind, params, train, test, use_stats):
    feat_tr = build_features(train, train, use_stats)
    feat_te = build_features(feat_tr, test, use_stats)
    num, cat = feature_lists(use_stats)
    m = make_pipeline(kind, params, num, cat)
    m.fit(feat_tr[num + cat], feat_tr['target'])
    auc = roc_auc_score(feat_te['target'], m.predict_proba(feat_te[num + cat])[:, 1])
    print(f"TEST {name:<35} {auc:.4f}")
    return auc


def main():
    print("Loading data from SQL Server...")
    df = base_prepare(load_data())
    closed = df[df['deal_stage'].isin(['won', 'lost'])].copy()

    ref_date = pd.to_datetime(df[['engage_date', 'close_date']].max().max())
    closed = temporal(closed)
    closed = deal_age(closed, ref_date)

    y = closed['target']
    train, test = train_test_split(closed, test_size=0.2, random_state=1, stratify=y)
    print(f"Train {len(train)} / Test {len(test)} (stratified)\n")

    base_xgb = dict(n_estimators=300, learning_rate=0.05, max_depth=6,
                    subsample=0.8, colsample_bytree=0.8)
    tuned = dict(n_estimators=500, learning_rate=0.03, max_depth=3, subsample=0.8,
                 colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0, reg_alpha=0.5)
    reg = dict(n_estimators=800, learning_rate=0.02, max_depth=4, subsample=0.7,
               colsample_bytree=0.6, min_child_weight=10, reg_lambda=5.0, reg_alpha=1.0)

    results = {}
    results['baseline'] = run_config("A baseline (depth6, stats)", 'xgb', base_xgb, train, True)
    results['tuned'] = run_config("B tuned (depth3, stats)", 'xgb', tuned, train, True)
    results['nostats'] = run_config("C tuned (no manual stats)", 'xgb', tuned, train, False)
    results['reg'] = run_config("D tuned (depth4, heavy reg)", 'xgb', reg, train, True)
    results['lr'] = run_config("E logistic regression", 'lr', {}, train, True)

    best_name = max(results, key=results.get)
    cfgs = {'baseline': ('xgb', base_xgb, True), 'tuned': ('xgb', tuned, True),
            'nostats': ('xgb', tuned, False), 'reg': ('xgb', reg, True),
            'lr': ('lr', {}, True)}
    kind, params, use_stats = cfgs[best_name]
    print(f"\nWinner by CV: {best_name}")
    final_test(best_name, kind, params, train, test, use_stats)
    final_test("baseline (for comparison)", 'xgb', base_xgb, train, test, True)


if __name__ == '__main__':
    main()
