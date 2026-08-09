import pytest
import pandas as pd
from src.db_setup import get_engine, load_data
from src.feature_engineering import build_features, get_train_test_split, EXCLUDED_LEAKAGE_COLUMNS

@pytest.fixture(scope="module")
def sample_df():
    engine = get_engine()
    return load_data(engine)

def test_build_features(sample_df):
    X, y, encoders = build_features(sample_df)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(sample_df)
    assert len(y) == len(sample_df)
    
    # Assert none of the excluded leakage columns are present in X
    for col in EXCLUDED_LEAKAGE_COLUMNS:
        assert col not in X.columns
    assert 'ft_after_upload' not in X.columns
    
    # Check no null values remaining in feature matrix X
    assert X.isnull().sum().sum() == 0

def test_train_test_split_stratified(sample_df):
    X, y, _ = build_features(sample_df)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y, test_size=0.2, random_state=42)
    
    assert len(X_train) + len(X_test) == len(X)
    # Check stratification: positive rate in train and test should match y overall (~0.3%)
    overall_pos_rate = y.mean()
    train_pos_rate = y_train.mean()
    test_pos_rate = y_test.mean()
    
    assert abs(train_pos_rate - overall_pos_rate) < 0.001
    assert abs(test_pos_rate - overall_pos_rate) < 0.001
    assert y_test.sum() > 0  # At least 1 positive sample in test set
