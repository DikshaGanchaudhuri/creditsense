import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_raw_data():
    """Load raw data from PostgreSQL."""
    engine = create_engine('postgresql+psycopg2://localhost/creditsense')
    return pd.read_sql('SELECT * FROM credit_data', engine)

def build_features(raw_df):
    """
    Takes the raw dataframe, engineers all features,
    encodes categoricals, and returns X, y with clean column names.
    This function is the single source of truth — call it everywhere.
    """
    feat = raw_df.copy()

    # ── 12 engineered features ───────────────────────────────────
    feat['monthly_burden']       = feat['credit_amount'] / feat['duration']
    feat['is_overdrawn']         = (feat['checking_status'] == '<0').astype(int)
    feat['no_checking']          = (feat['checking_status'] == 'no checking').astype(int)
    feat['is_large_loan']        = (feat['credit_amount'] > feat['credit_amount'].quantile(0.75)).astype(int)
    feat['is_long_duration']     = (feat['duration'] > 24).astype(int)
    feat['is_young_borrower']    = (feat['age'] < 25).astype(int)

    savings_map = {
        'no known savings': 4, '<100': 3,
        '100<=X<500': 2, '500<=X<1000': 1, '>=1000': 0
    }
    feat['savings_risk_score']   = feat['savings_status'].map(savings_map).fillna(2)

    employment_map = {
        'unemployed': 0, '<1': 1, '1<=X<4': 2, '4<=X<7': 3, '>=7': 4
    }
    feat['employment_score']     = feat['employment'].map(employment_map).fillna(1)
    feat['debt_burden_index']    = feat['credit_amount'] * feat['installment_commitment'] / 100

    purpose_risk = {
        'vacation': 3, 'other': 3, 'retraining': 3,
        'repairs': 2, 'education': 2, 'business': 2,
        'domestic appliance': 1, 'radio/tv': 1, 'used car': 1,
        'furniture/equipment': 1, 'new car': 1, 'car': 1
    }
    feat['purpose_risk']         = feat['purpose'].map(purpose_risk).fillna(2)
    feat['has_multiple_credits'] = (feat['existing_credits'] > 1).astype(int)
    feat['has_guarantor']        = (feat['other_parties'] != 'none').astype(int)

    # ── Encode categoricals ──────────────────────────────────────
    categorical_cols = [
        'checking_status', 'credit_history', 'purpose', 'savings_status',
        'employment', 'personal_status', 'other_parties',
        'property_magnitude', 'other_payment_plans', 'housing',
        'job', 'own_telephone', 'foreign_worker'
    ]
    feat_encoded = pd.get_dummies(feat, columns=categorical_cols, drop_first=True)

    feat_encoded.columns = (
        feat_encoded.columns
        .str.replace('<=X<', '_to_', regex=False)
        .str.replace('>=',   'gte',  regex=False)
        .str.replace('<=',   'lte',  regex=False)
        .str.replace('<',    'lt',   regex=False)
        .str.replace('>',    'gt',   regex=False)
        .str.replace(' ',    '_',    regex=False)
        .str.replace('/',    '_',    regex=False)
    )

    # ── Build X and y ────────────────────────────────────────────
    drop_cols = [c for c in ['id', 'class', 'target'] if c in feat_encoded.columns]
    feat_encoded['target'] = (feat_encoded['class'] == 'bad').astype(int)

    X = feat_encoded.drop(columns=['id', 'class', 'target'], errors='ignore')
    y = feat_encoded['target']

    return X, y