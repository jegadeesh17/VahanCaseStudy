import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

EXCLUDED_LEAKAGE_COLUMNS = [
    # Post-event funnel ratios derived from FT — direct target leakage
    'attempted_pct',
    'attempt_to_connected_pct',
    'connect_to_interested_pct',
    'interested_to_ft_after_first_attempt_pct',
    'attempted_to_ft_after_upload_pct',
    'ft_after_first_attempt',
    # PII column
    'candidate_phone',
    'uploaded_leads',
    # Mid-to-late funnel events — not available at upload/scoring time
    # ob_after_upload requires lead to have already connected + expressed interest
    # ob_after_first_attempt requires the first call attempt to have occurred
    'ob_after_upload',
    'ob_after_first_attempt',
]

def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Transform clean raw lead DataFrame into feature matrix X and target y.

    Excluded columns:
    - Direct leakage: ratio/percentage columns derived from the target (ft_after_upload)
    - Mid-funnel post-events: ob_after_upload, ob_after_first_attempt — these are only
      observable after significant tele-calling progress and are NOT available when scoring
      a freshly uploaded lead. Keeping them would mean the model can only score leads that
      have already progressed deep into the funnel, defeating early-funnel scoring.
    - PII: candidate_phone (SHA-256 hashed, non-informative)
    """
    data = df.copy()
    
    # Extract date features
    upload_dt = pd.to_datetime(data['upload_date'])
    data['day_of_week'] = upload_dt.dt.dayofweek
    data['day_of_month'] = upload_dt.dt.day
    
    # Handle missing values in numerical columns
    data['attempt_per_lead'] = data['attempt_per_lead'].fillna(0.0)
    p50_median = data['upload_to_first_attempt_p50_hrs'].median()
    data['upload_to_first_attempt_p50_hrs'] = data['upload_to_first_attempt_p50_hrs'].fillna(p50_median)
    
    # Categorical encoding for lead_source
    le = LabelEncoder()
    data['lead_source_encoded'] = le.fit_transform(data['lead_source'].astype(str))
    
    encoders = {
        'lead_source_encoder': le,
        'p50_median': p50_median
    }
    
    # Select feature columns (drop target, leakage, raw strings, and original date)
    drop_cols = EXCLUDED_LEAKAGE_COLUMNS + ['ft_after_upload', 'upload_date', 'lead_source']
    feature_cols = [c for c in data.columns if c not in drop_cols]
    
    X = data[feature_cols].copy()
    y = data['ft_after_upload'].astype(int).copy()
    
    return X, y, encoders

def get_train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Perform stratified train-test split to preserve class imbalance ratios."""
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
