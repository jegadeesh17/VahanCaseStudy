import pytest
import pandas as pd
from src.db_setup import get_engine, load_data
from src.feature_engineering import build_features, get_train_test_split
from src.models import train_all_models
from src.evaluation import evaluate_model, summary_table, get_feature_importance_df

@pytest.fixture(scope="module")
def eval_setup():
    engine = get_engine()
    df = load_data(engine)
    X, y, _ = build_features(df)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y, test_size=0.2, random_state=42)
    models = train_all_models(X_train, y_train)
    return models, X_test, y_test, list(X.columns)

def test_evaluate_model(eval_setup):
    models, X_test, y_test, feature_names = eval_setup
    for name, model in models.items():
        res = evaluate_model(model, X_test, y_test, name)
        assert res['model_name'] == name
        assert res['roc_auc'] > 0.5  # Model is better than random guessing
        assert res['confusion_matrix'].shape == (2, 2)

def test_summary_table(eval_setup):
    models, X_test, y_test, _ = eval_setup
    st = summary_table(models, X_test, y_test)
    assert len(st) == 3
    assert 'ROC-AUC' in st.columns
    assert 'F1 (Weighted)' in st.columns

def test_get_feature_importance_df(eval_setup):
    models, _, _, feature_names = eval_setup
    for name, model in models.items():
        imp = get_feature_importance_df(model, feature_names)
        assert len(imp) == len(feature_names)
        assert 'feature' in imp.columns
        assert 'importance' in imp.columns
