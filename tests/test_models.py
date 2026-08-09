import pytest
import pandas as pd
from pathlib import Path
from src.db_setup import get_engine, load_data
from src.feature_engineering import build_features, get_train_test_split
from src.models import train_all_models, save_models, load_models

@pytest.fixture(scope="module")
def prepared_data():
    engine = get_engine()
    df = load_data(engine)
    X, y, _ = build_features(df)
    return get_train_test_split(X, y, test_size=0.2, random_state=42)

def test_train_and_save_models(prepared_data, tmp_path):
    X_train, X_test, y_train, y_test = prepared_data
    models = train_all_models(X_train, y_train)
    
    assert len(models) == 3
    assert 'Logistic Regression' in models
    assert 'Random Forest' in models
    assert 'XGBoost' in models
    
    # Save to temporary path and reload
    save_models(models, output_dir=tmp_path)
    reloaded = load_models(input_dir=tmp_path)
    
    assert len(reloaded) == 3
    assert 'Random Forest' in reloaded
