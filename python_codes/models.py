import pandas as pd
import numpy as np
import os
import pyodbc
import pickle
import warnings


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from category_encoders import CatBoostEncoder
from xgboost import XGBClassifier

warnings.filterwarnings('ignore', category=UserWarning)

print("✓ All libraries imported successfully!\n")

OUTPUT_FILE = 'sales_pipeline_model.pkl'
ACTIVE_PRED_CSV = 'active_deals_predictions.csv'

print("=" * 80)
print("LOADING AND EXPLORING DATA")
print("=" * 80)

# Connect to database.
# The server name is read from the environment so the script works on any
# machine. Set it once with:  setx DB_SERVER "ELICE99\\SQLEXPRESS"
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

print(f"✓ Data loaded: {len(df)} records")
print(f"✓ Columns: {df.shape[1]}")

print("\n" + "=" * 80)
print(" EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# -------------------------------
# Convert date columns 
# -------------------------------
df['engage_date'] = pd.to_datetime(df.get('engage_date'), errors='coerce')
df['close_date'] = pd.to_datetime(df.get('close_date'), errors='coerce')

# numeric transform
df['employees_log'] = np.log1p(df['employees'].fillna(0))

# company size categorical
bins = [0, 50, 250, 1000, 5000, 15000, np.inf]
labels = ['micro', 'small', 'medium', 'large', 'enterprise', 'mega']
df['company_size'] = pd.cut(df['employees'].fillna(0), bins=bins, labels=labels).astype(str)

# Standardize column names and string values
df.columns = df.columns.str.lower().str.replace(' ', '_')
string_cols = df.select_dtypes(include='object').columns
for col in string_cols:
    # use fillna + astype(str) to avoid errors if column contains non-strings
    df[col] = df[col].fillna('').astype(str).str.lower().str.replace(' ', '_')

print("✓ Column names and values standardized\n")

# Filter: Keep only won and lost deals for training, save active deals separately
df_training = df[df['deal_stage'].isin(['won', 'lost'])].copy()
df_active = df[df['deal_stage'] == 'engaging'].copy()

print("\nAfter filtering:")
print(f"Training data (won + lost): {len(df_training)} rows")
print(f"Active deals (to predict): {len(df_active)} rows")

# Create binary target: 1 = Won, 0 = Lost
df_training['target'] = (df_training['deal_stage'] == 'won').astype(int)

print("\nTarget Distribution:")
print(df_training['target'].value_counts())
print(f"\nwon Rate: {df_training['target'].mean() * 100:.2f}%")

# Drop deal_stage column
df_training = df_training.drop(columns=['deal_stage'])

print("\n" + "=" * 80)
print(" TRAIN-TEST SPLIT")
print("=" * 80)

df_full_train, df_test = train_test_split(df_training, test_size=0.2, random_state=1)
print(f"Train rows: {len(df_full_train)}, Test rows: {len(df_test)}")

# -------- TEMPORAL FEATURES FROM ENGAGE_DATE --------
print("\nExtracting Temporal Features...")

def extract_date(df):
    df = df.copy()
    df['month_engaged'] = df['engage_date'].dt.month
    df['quarter_engaged'] = df['engage_date'].dt.quarter
    df['day_of_week_engaged'] = df['engage_date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week_engaged'].isin([5, 6])).astype(int)
    df['days_into_year'] = df['engage_date'].dt.dayofyear
    return df

df_full_train = extract_date(df_full_train)
df_test = extract_date(df_test)
print("✓ Temporal features created: month, quarter, day_of_week, is_weekend, days_into_year")

# -------- DEAL DURATION --------
print("\n Calculating Deal Duration...")

# Reference date: consider maximum available date in the full dataset (both engage and close)
ref_date = pd.to_datetime(df[['engage_date', 'close_date']].max().max())
if pd.isna(ref_date): # if NA fallback to the latest engage_date.
    ref_date = pd.to_datetime(df['engage_date']).max()   
print(f"Reference date used for duration calculations: {ref_date.date()}")


def deal_duration(df, ref_date):
    df = df.copy()
    # Ensure datetime types
    df['engage_date'] = pd.to_datetime(df['engage_date'], errors='coerce')
    df['close_date'] = pd.to_datetime(df['close_date'], errors='coerce')

    df['closed_duration'] = (df['close_date'] - df['engage_date']).dt.days
    df['active_duration'] = (ref_date - df['engage_date']).dt.days
    df['deal_age'] = df['closed_duration'].fillna(df['active_duration'])
    # If still NaN, fill with median or 0
    df['deal_age'] = df['deal_age'].fillna(df['deal_age'].median()).astype(float)

    return df

df_full_train = deal_duration(df_full_train, ref_date)
df_test = deal_duration(df_test, ref_date)
print("✓ Deal age calculated")

# -------- FEATURE ENGINEERING --------
print("\nCreating Features...")

def add_categorical_performance_stats(df, categorical_cols, target_col='target'):
    df = df.copy()
    stats_dict = {}
    for col in categorical_cols:
        
        # guard: if column missing, create placeholder
        if col not in df.columns:
            df[col] = 'missing'
            
        stats = df.groupby(col).agg({target_col: ['mean', 'count', 'sum']}).reset_index()
        stats.columns = [col, f'{col}_win_rate', f'{col}_total_deals', f'{col}_win']
        
        # Merge the stats back into the dataframe
        df = df.merge(stats, on=col, how='left')
        stats_dict[col] = stats
    return df, stats_dict

categorical_cols = ['sales_agent', 'product', 'account', 'sector', 'office_location', 'manager', 'regional_office', 'company_size']
df_full_train, full_train_stats = add_categorical_performance_stats(df_full_train, categorical_cols)

# FIXED (data leakage): the test set used to get its OWN win rates, computed
# from its own targets - i.e. the model was graded with the answers in hand,
# which inflated the test metrics. Instead, map the TRAINING statistics onto
# the test rows (unseen categories fall back to the global train win rate /
# zero counts), exactly the way the API does at prediction time.
global_win_rate = df_full_train['target'].mean()

def merge_train_stats(df, stats_dict, global_wr):
    df = df.copy()
    for col, stats in stats_dict.items():
        df = df.merge(stats, on=col, how='left')
        df[f'{col}_win_rate'] = df[f'{col}_win_rate'].fillna(global_wr)
        df[f'{col}_total_deals'] = df[f'{col}_total_deals'].fillna(0)
        df[f'{col}_win'] = df[f'{col}_win'].fillna(0)
    return df

df_test = merge_train_stats(df_test, full_train_stats, global_win_rate)

# Create aliases for easier reuse in feature functions (standardized names)
# e.g., sales_agent_win_rate -> agent_win_rate, sales_agent_total_deals -> agent_total_deals
alias_map = {
    'sales_agent_win_rate': 'agent_win_rate',
    'sales_agent_total_deals': 'agent_total_deals',
    'sales_agent_win': 'agent_win',
    'account_win_rate': 'account_win_rate',
    'account_total_deals': 'account_deal_count',
    'account_win': 'account_total_win',
    'sector_win_rate': 'sector_win_rate',
    'sector_total_deals': 'sector_deal_count',
    'sector_win': 'sector_total_win',
    'office_location_win_rate': 'office_win_rate',
    'office_location_total_deals': 'office_deal_count',
    'office_location_win': 'office_total_win',
    'regional_office_win_rate': 'region_win_rate',
    'regional_office_total_deals': 'region_deal_count',
    'regional_office_win': 'region_total_win',
    'company_size_win_rate': 'company_size_win_rate',
    'company_size_total_deals': 'company_size_deal_count',
    'company_size_win': 'company_size_total_win',
    'product_win_rate': 'product_win_rate',
    'product_total_deals': 'product_deal_count',
    'product_win': 'product_total_win'
}

def apply_aliases(df, alias_map):
    for long_name, short_name in alias_map.items():
        if long_name in df.columns and short_name not in df.columns:
            df[short_name] = df[long_name]
        elif short_name not in df.columns:
            # if neither exists, create a default fill
            df[short_name] = 0
    return df

df_full_train = apply_aliases(df_full_train, alias_map)
df_test = apply_aliases(df_test, alias_map)

def add_agent_speed(df, agent_col='sales_agent', speed_col='deal_age'):
    df = df.copy()
    if agent_col not in df.columns:
        df[agent_col] = 'missing'
    agent_speed = df.groupby(agent_col)[speed_col].mean().reset_index()
    agent_speed.columns = [agent_col, 'agent_avg_days_to_close']
    global_avg_speed = agent_speed['agent_avg_days_to_close'].mean()
    df = df.merge(agent_speed, on=agent_col, how='left')
    df['agent_avg_days_to_close'] = df['agent_avg_days_to_close'].fillna(global_avg_speed)
    return df, global_avg_speed

'''This function is designed to calculate the average deal speed (how long it takes to close a deal) 
for each sales agent and add it as a new feature to your dataset'''
df_full_train, global_train_speed = add_agent_speed(df_full_train)
df_test, global_test_speed = add_agent_speed(df_test)

def add_advanced_interaction_features(df):
    df = df.copy()
    epsilon = 1e-6

    # Ensure required columns exist 
    needed = [
        'agent_win_rate', 'product_win_rate', 'agent_total_deals', 'sector_win_rate',
        'employees_log', 'deal_age', 'office_deal_count', 'office_total_win',
        'company_size_win_rate', 'region_win_rate', 'agent_avg_days_to_close',
        'month_engaged', 'quarter_engaged'
    ]
    for col in needed:
        if col not in df.columns:
            df[col] = 0
    #Measures how well an agent and a product pair perform together
    df['agent_product_synergy'] = df['agent_win_rate'] * df['product_win_rate']
    #Captures potential or efficiency of the agent in closing deals they historically don’t win
    df['agent_win_efficiency'] = (1 - df['agent_win_rate']) * df['agent_total_deals']
    #Shows if an agent performs better or worse than their sector average
    df['agent_vs_sector'] = df['agent_win_rate'] - df['sector_win_rate']
    #Larger numbers → harder or more complex deals
    df['deal_complexity'] = (df['employees_log'] * df['deal_age'] / (df['agent_total_deals'] + 1))
    #Measures how busy an office is relative to successful deals, Helps capture potential bottlenecks
    df['office_load'] = df['office_deal_count'] / (df['office_total_win'] + 1)
    #Measures how suitable a product is for a sector
    df['product_sector_fit'] = df['product_win_rate'] * df['sector_win_rate']
    #Useful for understanding localized trends
    df['region_size_match'] = df['region_win_rate'] * df['company_size_win_rate']
    #Assigns a risk factor depending on the business quarter
    df['quarter_risk'] = df['quarter_engaged'].map({1: 0.8, 2: 0.9, 3: 1.0, 4: 1.2}).fillna(1.0)
    #Flags if the deal was engaged at the end of a quarter, which can affect urgency and closing rates
    df['is_quarter_end'] = df['month_engaged'].isin([3, 6, 9, 12]).astype(int)
    #Compares deal duration vs average agent closing speed
    df['sales_velocity_ratio'] = df['deal_age'] / (df['agent_avg_days_to_close'] + epsilon)
    #Combines win rate and speed, High values → agent wins often and quickly.
    df['pace_weighted_agent_score'] = df['agent_win_rate'] / (df['sales_velocity_ratio'] + epsilon)
    #Captures the combined effect of deal duration and company size, Larger numbers → slow deals with big companies.
    df['velocity_complexity_index'] = df['sales_velocity_ratio'] * df['employees_log']

    return df

df_full_train = add_advanced_interaction_features(df_full_train)
df_test = add_advanced_interaction_features(df_test)
print("✓ Created advanced interaction features")

# Remove engage_date 
df_full_train = df_full_train.drop(columns=['engage_date'], errors='ignore')
df_test = df_test.drop(columns=['engage_date'], errors='ignore')

# -----------------------------
# Feature lists (final)
# -----------------------------
numerical = [
    'year_established', 'employees_log', 'deal_age',
    'agent_win_rate', 'account_win_rate', 'sector_win_rate', 'office_win_rate', 'region_win_rate', 'company_size_win_rate',
    'product_win_rate', 'agent_avg_days_to_close',
    'agent_total_deals', 'agent_win', 'account_deal_count', 'account_total_win', 'sector_deal_count', 'sector_total_win',
    'office_deal_count', 'office_total_win', 'region_deal_count', 'region_total_win', 'company_size_deal_count',
    'company_size_total_win', 'product_deal_count', 'product_total_win',
    'agent_product_synergy', 'agent_win_efficiency', 'agent_vs_sector', 'deal_complexity', 'office_load', 'product_sector_fit',
    'region_size_match', 'quarter_risk', 'is_quarter_end', 'sales_velocity_ratio', 'pace_weighted_agent_score',
    'velocity_complexity_index',
    'month_engaged', 'quarter_engaged', 'day_of_week_engaged', 'is_weekend', 'days_into_year'
]

categorical = [
    'sales_agent', 'product', 'account', 'sector', 'office_location', 'manager', 'regional_office', 'company_size'
]

# Ensure features exist in dataframes, create defaults if not
def ensure_features_exist(df, num_feats, cat_feats):
    df = df.copy()
    for f in num_feats + cat_feats:
        if f not in df.columns:
            # numeric defaults -> 0, categorical defaults -> 'missing'
            df[f] = 0 if f in num_feats else 'missing'
    return df
'''This is a safeguard function to make sure all expected features exist in your datasets before modeling'''
df_full_train = ensure_features_exist(df_full_train, numerical, categorical)
df_test = ensure_features_exist(df_test, numerical, categorical)
df_active = ensure_features_exist(df_active, numerical, categorical)

# -----------------------------
# Prepare X and y
# -----------------------------
feature_columns = numerical + categorical

X_train = df_full_train[feature_columns]
y_train = df_full_train['target'].astype(int)

X_test = df_test[feature_columns]
y_test = df_test['target'].astype(int)

print("\nModel pipeline building...")

# -----------------------------
# Build pipeline with CatBoostEncoder for categorical
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('cb', CatBoostEncoder(cols=categorical, random_state=42), categorical),
        ('scaler', StandardScaler(), numerical)
    ],
    remainder='drop'
)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('xgb', XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    ))
])

print("✓ Pipeline ready. Starting training...\n")

# -----------------------------
# Train
# -----------------------------
model.fit(X_train, y_train)
print("✓ Training complete.")

# -----------------------------
# Evaluate
# -----------------------------
probs = model.predict_proba(X_test)[:, 1]
preds = model.predict(X_test)

roc = roc_auc_score(y_test, probs)
acc = accuracy_score(y_test, preds)

print("\nEVALUATION ON TEST SET")
print("-" * 40)
print(f"ROC AUC: {roc:.4f}")
print(f"Accuracy: {acc:.4f}")
print("\nClassification report:")
print(classification_report(y_test, preds, digits=4))

# -----------------------------
# Save artifacts
# -----------------------------
# Convert DataFrames to dicts for easier lookup

stats_maps = {}
for col, stats_df in full_train_stats.items():
    stats_df = stats_df.set_index(col)
    stats_maps[col] = {
        "win_rate": stats_df[f"{col}_win_rate"].to_dict(),
        "total_deals": stats_df[f"{col}_total_deals"].to_dict(),
        "total_win": stats_df[f"{col}_win"].to_dict()
    }

# -----------------------------
# Export active deals for testing
# -----------------------------
# test.py expects this file ("Generated from model.py") but it was never
# written. Export it with the TRAINING statistics mapped on - the same way
# the API does - so batch predictions are realistic instead of zero-filled.
if not df_active.empty:
    for col in categorical_cols:
        df_active[f'{col}_win_rate'] = df_active[col].map(stats_maps[col]['win_rate']).fillna(global_win_rate)
        df_active[f'{col}_total_deals'] = df_active[col].map(stats_maps[col]['total_deals']).fillna(0)
        df_active[f'{col}_win'] = df_active[col].map(stats_maps[col]['total_win']).fillna(0)
    df_active = apply_aliases(df_active, alias_map)
    df_active = ensure_features_exist(df_active, numerical, categorical)
    df_active.to_csv(ACTIVE_PRED_CSV, index=False)
    print(f"✓ Active deals exported to: {ACTIVE_PRED_CSV}")


artifacts = {
'model': model,
'preprocessor': preprocessor,
'feature_names': feature_columns,
'categorical_features': categorical,
'numerical_features': numerical,
'stats_maps': stats_maps,  # <-- include the maps
'global_train_speed': global_train_speed,  # for agent_avg_days_to_close fallback
'test_roc_auc': float(roc)
}

with open(OUTPUT_FILE, 'wb') as f_out:
    pickle.dump(artifacts, f_out)
print(f"\n✓ Model and artifacts saved to: {OUTPUT_FILE}")

