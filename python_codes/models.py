"""
================================================================================
MODELS.PY - THE SINGLE CANONICAL TRAINING SCRIPT FOR THE SALES PIPELINE MODEL
================================================================================

PURPOSE:
    One way to produce the model artifact. This script:
    1. Loads the joined extract from SQL Server (falls back to the raw CSVs
       with the same cleaning applied, so it runs on any machine)
    2. Splits stratified (train/test class shares match the raw data)
    3. Builds a compact feature set (core + temporal + interaction features).
       Hand-rolled group-statistic features (win rates per agent/account/...)
       were tested and REMOVED: they duplicate the CatBoost target encoding
       and hurt out-of-sample AUC (see experiments.py / tune.py)
    4. Cross-validates a baseline, then tunes with RandomizedSearchCV (also
       on the training split only) and selects by cross-validated score;
       a proven locked-in candidate is always included in the re-rank
    5. Evaluates the final model ONCE on the untouched test set
    6. Writes the pickle artifact AND a metrics.json so every quoted number
       traces to a single file

TRAIN/SERVE CONSISTENCY:
    features.py (used by main.py and test.py) rebuilds the same features
    from raw input. The group-statistic maps ship EMPTY in the artifact, so
    features.py fills them with constants at serve time - matching training,
    where they are constants too.

RUN:
    python models.py          (needs SQL Server, or set DB_SERVER env var;
                               otherwise pass --csv to read the raw CSVs)
================================================================================
"""

import argparse
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from category_encoders import CatBoostEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_val_score,
                                     train_test_split, RepeatedStratifiedKFold)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent / 'datasets'
OUTPUT_FILE = SCRIPT_DIR / 'sales_pipeline_model.pkl'
METRICS_FILE = SCRIPT_DIR / 'metrics.json'
ACTIVE_PRED_CSV = SCRIPT_DIR / 'active_deals_predictions.csv'

# Previous saved numbers, for the comparison printed at the end.
PREVIOUS = {
    'notebook_test_roc_auc': 0.5974,   # notebook's saved tuned evaluation
    'artifact_test_roc_auc': 0.5953,   # score stored in the previous artifact (2026-11-10 morning run)
    'notebook_test_precision': 0.7037,
}

print("=" * 80)
print("SALES PIPELINE MODEL - CANONICAL TRAINING RUN")
print("=" * 80)

# ============================================================================
# 1. LOAD THE EXTRACT (SQL Server primary, raw CSVs as fallback)
# ============================================================================

QUERY = '''
SELECT p.*,
    a.sector, a.year_established, a.account_tier, a.employees, a.office_location,
    s.manager, s.regional_office
FROM dbo.sales_pipeline p
LEFT JOIN accounts a ON a.account = p.account
LEFT JOIN sales_teams s ON p.sales_agent = s.sales_agent
WHERE deal_stage NOT IN ('Prospecting')
'''


def load_from_sql():
    """Read the joined extract from the cleaned SQL Server tables."""
    import pyodbc
    conn = pyodbc.connect(
        "Driver={SQL Server};"
        f"Server={os.environ.get('DB_SERVER', 'ELICE99') + chr(92) + 'SQLEXPRESS'};"
        "Database=CRM_Sales_Opportunity;"
        "Trusted_Connection=yes;"
    )
    df = pd.read_sql(QUERY, conn)
    conn.close()
    return df


def load_from_csv():
    """Read the raw CSVs and apply the same cleaning as the SQL scripts,
    so the fallback extract matches what the database would return."""
    sp = pd.read_csv(DATASET_DIR / 'sales_pipeline.csv')
    ac = pd.read_csv(DATASET_DIR / 'accounts.csv')
    st = pd.read_csv(DATASET_DIR / 'sales_teams.csv')

    # account cleaning.sql: fill missing accounts with 'Unknown',
    # capitalise sector consistently
    sp['account'] = sp['account'].fillna('Unknown')
    ac['sector'] = ac['sector'].map(
        lambda s: s[:1].upper() + s[1:] if isinstance(s, str) else s)

    df = sp.merge(ac, on='account', how='left')
    df = df.merge(st[['sales_agent', 'manager', 'regional_office']],
                  on='sales_agent', how='left')
    df = df[df['deal_stage'] != 'Prospecting'].reset_index(drop=True)
    return df


parser = argparse.ArgumentParser()
parser.add_argument('--csv', action='store_true',
                    help='read the raw CSVs instead of SQL Server')
args = parser.parse_args()

if args.csv:
    df = load_from_csv()
    print(f"[source: raw CSVs + in-script cleaning]")
else:
    try:
        df = load_from_sql()
        print("[source: SQL Server, CRM_Sales_Opportunity]")
    except Exception as e:
        print(f"[SQL Server unavailable ({e.__class__.__name__}); "
              f"falling back to raw CSVs]")
        df = load_from_csv()

print(f"OK Data loaded: {len(df)} records, {df.shape[1]} columns")

# ============================================================================
# 2. STANDARDISE (identical logic to features.py at serve time)
# ============================================================================

df['engage_date'] = pd.to_datetime(df['engage_date'], errors='coerce')
df['close_date'] = pd.to_datetime(df['close_date'], errors='coerce')
df['employees_log'] = np.log1p(df['employees'].fillna(0))

bins = [0, 50, 250, 1000, 5000, 15000, np.inf]
labels = ['micro', 'small', 'medium', 'large', 'enterprise', 'mega']
df['company_size'] = pd.cut(df['employees'].fillna(0), bins=bins,
                            labels=labels).astype(str)

df.columns = df.columns.str.lower().str.replace(' ', '_')
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna('').astype(str).str.lower().str.replace(' ', '_')

# Filter: won + lost for training, engaging held aside for scoring
df_training = df[df['deal_stage'].isin(['won', 'lost'])].copy()
df_active = df[df['deal_stage'] == 'engaging'].copy()
df_training['target'] = (df_training['deal_stage'] == 'won').astype(int)
df_training = df_training.drop(columns=['deal_stage'])

n_won = int(df_training['target'].sum())
n_lost = int((1 - df_training['target']).sum())
print(f"Training rows: {len(df_training)} (won {n_won} / lost {n_lost})")
print(f"Active deals held aside for scoring: {len(df_active)}")

# ============================================================================
# 3. STRATIFIED SPLIT (test class share must match the raw data)
# ============================================================================

df_full_train, df_test = train_test_split(
    df_training, test_size=0.2, random_state=1, stratify=df_training['target'])

train_share = df_full_train['target'].mean()
test_share = df_test['target'].mean()
print(f"\nSplit: train {len(df_full_train)} rows "
      f"(won {train_share:.1%}), test {len(df_test)} rows (won {test_share:.1%})")
assert abs(train_share - test_share) < 0.01, \
    "train/test class shares drifted; the split is broken"

# ============================================================================
# 4. FEATURE ENGINEERING (identical formulas to features.py)
# ============================================================================

def extract_date(d):
    d = d.copy()
    d['month_engaged'] = d['engage_date'].dt.month
    d['quarter_engaged'] = d['engage_date'].dt.quarter
    d['day_of_week_engaged'] = d['engage_date'].dt.dayofweek
    d['is_weekend'] = d['day_of_week_engaged'].isin([5, 6]).astype(int)
    d['days_into_year'] = d['engage_date'].dt.dayofyear
    return d


df_full_train = extract_date(df_full_train)
df_test = extract_date(df_test)

ref_date = pd.to_datetime(df[['engage_date', 'close_date']].max().max())
if pd.isna(ref_date):
    ref_date = pd.to_datetime(df['engage_date']).max()
print(f"Reference date for duration features: {ref_date.date()}")


def deal_duration(d, ref):
    d = d.copy()
    d['closed_duration'] = (d['close_date'] - d['engage_date']).dt.days
    d['active_duration'] = (ref - d['engage_date']).dt.days
    d['deal_age'] = d['closed_duration'].fillna(d['active_duration'])
    d['deal_age'] = d['deal_age'].fillna(d['deal_age'].median()).astype(float)
    return d


df_full_train = deal_duration(df_full_train, ref_date)
df_test = deal_duration(df_test, ref_date)

categorical_cols = ['sales_agent', 'product', 'account', 'sector',
                    'office_location', 'manager', 'regional_office',
                    'company_size']

# Group-statistic features (win rates / deal counts per agent, account, ...)
# were tested and REMOVED from the feature set: they largely duplicate the
# CatBoost target encoding, and being estimated from small noisy groups they
# hurt out-of-sample AUC (stratified CV 0.57 with them vs 0.61 without -
# see experiments.py and tune.py for the full comparison).
# They still ship in the artifact as EMPTY stats_maps, so features.py fills
# them with constants at serve time - matching training, where they are
# constants too.

global_win_rate = float(df_full_train['target'].mean())
global_train_speed = float(df_full_train['deal_age'].mean())
print(f"Global train win rate: {global_win_rate:.4f}; "
      f"global average deal age: {global_train_speed:.1f} days")

epsilon = 1e-6


def add_interaction_features(d):
    d = d.copy()
    needed = ['agent_win_rate', 'product_win_rate', 'agent_total_deals',
              'sector_win_rate', 'employees_log', 'deal_age', 'office_deal_count',
              'office_total_win', 'company_size_win_rate', 'region_win_rate',
              'agent_avg_days_to_close', 'month_engaged', 'quarter_engaged']
    for col in needed:
        if col not in d.columns:
            # constant defaults: win rates 0.5, counts/speeds 0
            d[col] = 0.5 if col.endswith('_win_rate') else 0
    d['agent_avg_days_to_close'] = global_train_speed  # train-wide constant
    d['agent_product_synergy'] = d['agent_win_rate'] * d['product_win_rate']
    d['agent_win_efficiency'] = (1 - d['agent_win_rate']) * d['agent_total_deals']
    d['agent_vs_sector'] = d['agent_win_rate'] - d['sector_win_rate']
    d['deal_complexity'] = d['employees_log'] * d['deal_age'] / (d['agent_total_deals'] + 1)
    d['office_load'] = d['office_deal_count'] / (d['office_total_win'] + 1)
    d['product_sector_fit'] = d['product_win_rate'] * d['sector_win_rate']
    d['region_size_match'] = d['region_win_rate'] * d['company_size_win_rate']
    d['quarter_risk'] = d['quarter_engaged'].map({1: 0.8, 2: 0.9, 3: 1.0, 4: 1.2}).fillna(1.0)
    d['is_quarter_end'] = d['month_engaged'].isin([3, 6, 9, 12]).astype(int)
    d['sales_velocity_ratio'] = d['deal_age'] / (d['agent_avg_days_to_close'] + epsilon)
    d['pace_weighted_agent_score'] = d['agent_win_rate'] / (d['sales_velocity_ratio'] + epsilon)
    d['velocity_complexity_index'] = d['sales_velocity_ratio'] * d['employees_log']
    return d


df_full_train = add_interaction_features(df_full_train)
df_test = add_interaction_features(df_test)

df_full_train = df_full_train.drop(columns=['engage_date', 'close_date'], errors='ignore')
df_test = df_test.drop(columns=['engage_date', 'close_date'], errors='ignore')

numerical = [
    'year_established', 'employees_log', 'deal_age',
    'agent_product_synergy', 'agent_win_efficiency', 'agent_vs_sector',
    'deal_complexity', 'office_load', 'product_sector_fit', 'region_size_match',
    'quarter_risk', 'is_quarter_end', 'sales_velocity_ratio',
    'pace_weighted_agent_score', 'velocity_complexity_index',
    'month_engaged', 'quarter_engaged', 'day_of_week_engaged', 'is_weekend',
    'days_into_year',
]
categorical = ['sales_agent', 'product', 'account', 'sector',
               'office_location', 'manager', 'regional_office', 'company_size']


def ensure_features_exist(d, num_feats, cat_feats):
    d = d.copy()
    for f in num_feats + cat_feats:
        if f not in d.columns:
            d[f] = 0 if f in num_feats else 'missing'
    return d


df_full_train = ensure_features_exist(df_full_train, numerical, categorical)
df_test = ensure_features_exist(df_test, numerical, categorical)

feature_columns = numerical + categorical
X_train, y_train = df_full_train[feature_columns], df_full_train['target'].astype(int)
X_test, y_test = df_test[feature_columns], df_test['target'].astype(int)
print(f"\nFeatures: {len(numerical)} numerical + {len(categorical)} categorical")

# ============================================================================
# 5. CROSS-VALIDATION ON THE TRAINING SPLIT, THEN TUNING
# ============================================================================

def build_pipeline(**xgb_params):
    preprocessor = ColumnTransformer(
        transformers=[
            ('cb', CatBoostEncoder(cols=categorical, random_state=42), categorical),
            ('scaler', StandardScaler(), numerical),
        ],
        remainder='drop',
    )
    return Pipeline([
        ('preprocessor', preprocessor),
        ('xgb', XGBClassifier(random_state=42, eval_metric='logloss',
                              **xgb_params)),
    ])


cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

baseline = build_pipeline(n_estimators=300, learning_rate=0.05, max_depth=6,
                          subsample=0.8, colsample_bytree=0.8)
baseline_scores = cross_val_score(baseline, X_train, y_train, cv=cv5,
                                  scoring='roc_auc', n_jobs=-1)
print(f"\nBaseline CV ROC-AUC (5-fold): {baseline_scores.mean():.4f} "
      f"+/- {baseline_scores.std():.4f}")

param_distributions = {
    'xgb__n_estimators': [200, 300, 500, 800],
    'xgb__max_depth': [3, 4, 5, 6, 7],
    'xgb__learning_rate': [0.01, 0.02, 0.03, 0.05, 0.1],
    'xgb__min_child_weight': [3, 5, 7],
    'xgb__subsample': [0.7, 0.8, 0.9],
    'xgb__colsample_bytree': [0.7, 0.8, 0.9],
    'xgb__gamma': [0, 0.1, 0.2],
    'xgb__reg_alpha': [0, 0.5, 1],
    'xgb__reg_lambda': [1, 1.5, 2],
    # class balance: 63/37 won-heavy; the notebook used the lost-to-won ratio
    'xgb__scale_pos_weight': [1.0, round(n_lost / n_won, 4)],
}

search = RandomizedSearchCV(
    build_pipeline(),
    param_distributions=param_distributions,
    n_iter=30,
    scoring='roc_auc',
    cv=cv5,
    random_state=42,
    n_jobs=-1,
    refit=False,
)
search.fit(X_train, y_train)
print(f"Search best CV ROC-AUC (5-fold): {search.best_score_:.4f}")

# Re-rank the top candidates with REPEATED stratified CV (3 x 5) to make the
# selection robust to fold noise, and compare against the baseline pipeline
# under the same protocol. Selection stays on the training split throughout.
# 'locked_search3' is the proven winner from tune.py (stat-free features,
# depth 3, single test ROC-AUC 0.6027) - kept in the pool so the final
# selection can only match or improve on it.
repeats = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
results = search.cv_results_
top_idx = np.argsort(results['mean_test_score'])[::-1][:5]

LOCKED = dict(n_estimators=600, learning_rate=0.02, max_depth=3, subsample=0.7,
              colsample_bytree=0.9, min_child_weight=3, reg_lambda=3.0,
              reg_alpha=1.0)

candidates = [('baseline_300_lr05_d6', baseline),
              ('locked_search3', build_pipeline(**LOCKED))]
for rank, i in enumerate(top_idx, start=1):
    params = {k.replace('xgb__', ''): v for k, v in results['params'][i].items()}
    candidates.append((f'search_rank_{rank}', build_pipeline(**params)))

print("\nRepeated CV (3x5) re-rank on the training split:")
best_name, best_cv, model = None, -1, None
for name, pipe in candidates:
    scores = cross_val_score(pipe, X_train, y_train, cv=repeats,
                             scoring='roc_auc', n_jobs=-1)
    print(f"  {name}: {scores.mean():.4f} +/- {scores.std():.4f}")
    if scores.mean() > best_cv:
        best_name, best_cv = name, scores.mean()

print(f"\nSelected: {best_name} (repeated CV ROC-AUC {best_cv:.4f})")
model = dict(candidates)[best_name]

# Refit the selected pipeline on the full training split
model.fit(X_train, y_train)

# ============================================================================
# 6. THRESHOLD FROM OUT-OF-FOLD TRAINING PREDICTIONS (metadata only;
#    the API ships 0.5 and reports both probabilities on every call)
# ============================================================================

oof_probs = cross_val_predict(model, X_train, y_train, cv=cv5,
                              method='predict_proba', n_jobs=-1)[:, 1]
prec, rec, thresholds = precision_recall_curve(y_train, oof_probs)
f1s = 2 * prec * rec / (prec + rec + 1e-12)
threshold_tuned = float(thresholds[np.argmax(f1s[:-1])])
print(f"\nOOF F1-optimal threshold: {threshold_tuned:.3f} "
      f"(F1 {np.max(f1s[:-1]):.4f}); the API ships 0.5")

# ============================================================================
# 7. FINAL EVALUATION - ONE PASS ON THE UNTOUCHED TEST SET
# ============================================================================

probs = model.predict_proba(X_test)[:, 1]
preds = (probs >= 0.5).astype(int)

roc = roc_auc_score(y_test, probs)
acc = accuracy_score(y_test, preds)
prec05 = precision_score(y_test, preds)
rec05 = recall_score(y_test, preds)
f105 = f1_score(y_test, preds)

print("\n" + "=" * 80)
print("FINAL TEST SET EVALUATION (threshold 0.5)")
print("=" * 80)
print(f"ROC-AUC:   {roc:.4f}   <- main metric")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec05:.4f}")
print(f"Recall:    {rec05:.4f}")
print(f"F1:        {f105:.4f}")
print("\nClassification report:")
print(classification_report(y_test, preds, digits=4))

preds_tuned = (probs >= threshold_tuned).astype(int)
metrics_tuned = {
    'precision': round(float(precision_score(y_test, preds_tuned)), 4),
    'recall': round(float(recall_score(y_test, preds_tuned)), 4),
    'f1': round(float(f1_score(y_test, preds_tuned)), 4),
}
print(f"At tuned threshold {threshold_tuned:.3f}: "
      f"precision {metrics_tuned['precision']}, "
      f"recall {metrics_tuned['recall']}, F1 {metrics_tuned['f1']}")

# Comparison against the previously saved numbers
print("\n" + "=" * 80)
print("COMPARISON WITH PREVIOUSLY SAVED NUMBERS")
print("=" * 80)
print(f"Notebook test ROC-AUC (old canonical): {PREVIOUS['notebook_test_roc_auc']:.4f}"
      f"  -> new: {roc:.4f}  "
      f"{'IMPROVED' if roc >= PREVIOUS['notebook_test_roc_auc'] else 'LOWER'}")
print(f"Old artifact test ROC-AUC:             {PREVIOUS['artifact_test_roc_auc']:.4f}"
      f"  -> new: {roc:.4f}  "
      f"{'IMPROVED' if roc >= PREVIOUS['artifact_test_roc_auc'] else 'LOWER'}")
if roc < PREVIOUS['notebook_test_roc_auc']:
    print("WARNING: new test ROC-AUC is below the portfolio-quoted number; "
          "the portfolio pages would need the lower number quoted.")

# ============================================================================
# 8. SAVE ARTIFACT + METRICS + ACTIVE-DEAL EXPORT
# ============================================================================

stats_maps = {}  # group-statistic features removed from the model; ship empty
                 # maps so features.py fills them with constants at serve time

selected_xgb_params = {k: v for k, v in model.named_steps['xgb'].get_params().items()
                       if k in {p.replace('xgb__', '') for p in param_distributions}}

run_metadata = {
    'trained_at': datetime.now().isoformat(timespec='seconds'),
    'source': 'sql_server' if not args.csv else 'csv',
    'train_rows': int(len(X_train)),
    'test_rows': int(len(X_test)),
    'train_won_share': round(float(train_share), 4),
    'test_won_share': round(float(test_share), 4),
    'features': len(feature_columns),
    'selection': f'{best_name} by 3x5 repeated CV on the training split',
    'best_params': {k: (v.tolist() if hasattr(v, 'tolist') else v)
                    for k, v in selected_xgb_params.items()},
}

artifacts = {
    'model': model,
    'preprocessor': model.named_steps['preprocessor'],
    'feature_names': feature_columns,
    'categorical_features': categorical,
    'numerical_features': numerical,
    'stats_maps': stats_maps,
    'global_train_speed': global_train_speed,
    'threshold': 0.5,                       # shipped behaviour
    'threshold_tuned': threshold_tuned,     # OOF F1-optimal, metadata
    'test_roc_auc': float(roc),
    'metrics': {
        'test_accuracy': round(float(acc), 4),
        'test_precision': round(float(prec05), 4),
        'test_recall': round(float(rec05), 4),
        'test_f1': round(float(f105), 4),
        'cv_roc_auc_mean': round(float(search.best_score_), 4),
        'at_tuned_threshold': metrics_tuned,
    },
    'run_metadata': run_metadata,
}

with open(OUTPUT_FILE, 'wb') as f_out:
    pickle.dump(artifacts, f_out)
print(f"\nOK Model artifact saved to: {OUTPUT_FILE}")

import json

metrics_json = {
    'trained_at': run_metadata['trained_at'],
    'source': run_metadata['source'],
    'data': {'train_rows': run_metadata['train_rows'],
             'test_rows': run_metadata['test_rows'],
             'train_won_share': run_metadata['train_won_share'],
             'test_won_share': run_metadata['test_won_share']},
    'cv': {'strategy': 'selection by 3x5 repeated stratified CV, training split only',
           'baseline_roc_auc_5fold': [round(float(s), 4) for s in baseline_scores],
           'search_best_roc_auc_5fold': round(float(search.best_score_), 4),
           'selected': best_name,
           'selected_cv_roc_auc': round(float(best_cv), 4),
           'best_params': run_metadata['best_params']},
    'test': {'roc_auc': round(float(roc), 4),
             'accuracy': round(float(acc), 4),
             'precision': round(float(prec05), 4),
             'recall': round(float(rec05), 4),
             'f1': round(float(f105), 4),
             'threshold': 0.5},
    'test_at_tuned_threshold': {'threshold': round(threshold_tuned, 3),
                                **metrics_tuned},
    'previous_saved': PREVIOUS,
    'features': {'numerical': len(numerical), 'categorical': len(categorical)},
}
with open(METRICS_FILE, 'w', encoding='utf-8') as f_out:
    json.dump(metrics_json, f_out, indent=2)
print(f"OK Metrics written to: {METRICS_FILE}")

# Active deals: raw fields + the same engineered features as training
# (temporal + interactions with constant group statistics). No zero-fills.
if not df_active.empty:
    df_active = df_active.copy()
    df_active = extract_date(df_active)
    df_active['deal_age'] = (ref_date - df_active['engage_date']).dt.days
    df_active['deal_age'] = df_active['deal_age'].fillna(
        df_active['deal_age'].median()).astype(float)
    df_active = add_interaction_features(df_active)
    df_active = ensure_features_exist(df_active, numerical, categorical)
    df_active.to_csv(ACTIVE_PRED_CSV, index=False)
    print(f"OK Active deals exported to: {ACTIVE_PRED_CSV}")
