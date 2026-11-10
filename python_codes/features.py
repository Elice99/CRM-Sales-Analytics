"""
================================================================================
FEATURES.PY - SHARED FEATURE ENGINEERING
================================================================================

Single source of truth for the feature engineering that the trained model
expects. Imported by main.py (API) and test.py (batch scoring), so the two
can never drift apart. models.py performs the same logic at training time.

================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime


# Alias map: long stat column names (as built at training time) -> short
# feature names used by the model.
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


def ensure_features_exist(df, num_feats, cat_feats):
    """
    FUNCTION: ensure_features_exist(df, num_feats, cat_feats)

    PURPOSE: Ensure all required features exist in dataframe
             Fill missing ones with default values

    Used to prepare incoming data for the model.
    """
    df = df.copy()
    all_features = num_feats + cat_feats

    for f in all_features:
        if f not in df.columns:
            if f in num_feats:
                df[f] = 0  # Default numerical value
            else:
                df[f] = 'missing'  # Default categorical value

    return df


def compute_all_features(df, artifacts):
    """
    FUNCTION: compute_all_features(df, artifacts)

    PURPOSE: Replicate the entire feature engineering logic from models.py
        using saved artifacts (stats_maps, global_train_speed).
    """
    df = df.copy()
    stats_maps = artifacts.get('stats_maps', {})
    global_train_speed = artifacts.get('global_train_speed', 30.0)

    # STANDARDIZATION & CORE TRANSFORMS (models.py logic)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    string_cols = df.select_dtypes(include='object').columns
    for col in string_cols:
        df[col] = df[col].fillna('').astype(str).str.lower().str.replace(' ', '_')

    # employees_log & company_size creation
    df['employees_log'] = np.log1p(df['employees'].fillna(0))
    bins = [0, 50, 250, 1000, 5000, 15000, np.inf]
    labels = ['micro', 'small', 'medium', 'large', 'enterprise', 'mega']
    df['company_size'] = pd.cut(df['employees'].fillna(0), bins=bins, labels=labels).astype(str).fillna('missing')

    # TEMPORAL FEATURES (models.py logic)
    df['engage_date'] = pd.to_datetime(df['engage_date'], errors='coerce')

    # Calculate deal_age on the backend using current date
    ref_date = datetime.now()
    # Calculate active duration (since the deal is still "active" for prediction)
    df['deal_age'] = (ref_date - df['engage_date']).dt.days.fillna(0.0)

    df['month_engaged'] = df['engage_date'].dt.month.fillna(6).astype(int)
    df['quarter_engaged'] = df['engage_date'].dt.quarter.fillna(2).astype(int)
    df['day_of_week_engaged'] = df['engage_date'].dt.dayofweek.fillna(3).astype(int)
    df['is_weekend'] = (df['day_of_week_engaged'].isin([5, 6])).astype(int)
    df['days_into_year'] = df['engage_date'].dt.dayofyear.fillna(150).astype(int)

    # STATISTICAL FEATURES (Target Encoding using stats_maps)
    categorical_cols_for_stats = ['sales_agent', 'product', 'account', 'sector', 'office_location', 'manager', 'regional_office', 'company_size']

    for col in categorical_cols_for_stats:
        if col in df.columns:
            win_rate_col = f'{col}_win_rate'
            total_deals_col = f'{col}_total_deals'
            total_win_col = f'{col}_win'

            # Use safe fallbacks: 0.5 for rate, 1.0 for counts
            win_rate_map = stats_maps.get(col, {}).get('win_rate', {})
            total_deals_map = stats_maps.get(col, {}).get('total_deals', {})
            total_win_map = stats_maps.get(col, {}).get('total_win', {})

            df[win_rate_col] = df[col].map(win_rate_map).fillna(0.5)
            df[total_deals_col] = df[col].map(total_deals_map).fillna(1.0)
            df[total_win_col] = df[col].map(total_win_map).fillna(0.5)

    # ALIASES (Must replicate the mapping from models.py)
    for long_name, short_name in alias_map.items():
        if long_name in df.columns:
            df[short_name] = df[long_name]
        elif short_name not in df.columns:
            df[short_name] = 0

    # AGENT SPEED
    # Using the global average speed saved in artifacts as the only available data source
    df['agent_avg_days_to_close'] = global_train_speed

    # ADVANCED INTERACTION FEATURES (11 formulas from models.py)
    epsilon = 1e-6

    df['agent_product_synergy'] = df['agent_win_rate'] * df['product_win_rate']
    df['agent_win_efficiency'] = (1 - df['agent_win_rate']) * df['agent_total_deals']
    df['agent_vs_sector'] = df['agent_win_rate'] - df['sector_win_rate']
    df['deal_complexity'] = (df['employees_log'] * df['deal_age'] / (df['agent_total_deals'] + 1))
    df['office_load'] = df['office_deal_count'] / (df['office_total_win'] + 1)
    df['product_sector_fit'] = df['product_win_rate'] * df['sector_win_rate']
    df['region_size_match'] = df['region_win_rate'] * df['company_size_win_rate']
    df['quarter_risk'] = df['quarter_engaged'].map({1: 0.8, 2: 0.9, 3: 1.0, 4: 1.2}).fillna(1.0)
    df['is_quarter_end'] = df['month_engaged'].isin([3, 6, 9, 12]).astype(int)
    df['sales_velocity_ratio'] = df['deal_age'] / (df['agent_avg_days_to_close'] + epsilon)
    df['pace_weighted_agent_score'] = df['agent_win_rate'] / (df['sales_velocity_ratio'] + epsilon)
    df['velocity_complexity_index'] = df['sales_velocity_ratio'] * df['employees_log']

    # Remove raw date column as it's not a final feature
    df = df.drop(columns=['engage_date'], errors='ignore')

    return df
