from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_logistic_regression(X_train, y_train, random_state: int = 42) -> LogisticRegression:
    """Train Logistic Regression classifier with balanced class weights."""
    clf = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=random_state
    )
    clf.fit(X_train, y_train)
    return clf

def train_random_forest(X_train, y_train, n_estimators: int = 200, random_state: int = 42) -> RandomForestClassifier:
    """Train Random Forest classifier with balanced class weights."""
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    return clf

def train_xgboost(X_train, y_train, n_estimators: int = 200, random_state: int = 42) -> XGBClassifier:
    """Train XGBoost classifier with scale_pos_weight to handle class imbalance."""
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    clf = XGBClassifier(
        n_estimators=n_estimators,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric='logloss',
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    return clf

def train_all_models(X_train, y_train, random_state: int = 42) -> dict:
    """Train Logistic Regression, Random Forest, and XGBoost models on training set."""
    models = {
        'Logistic Regression': train_logistic_regression(X_train, y_train, random_state=random_state),
        'Random Forest': train_random_forest(X_train, y_train, random_state=random_state),
        'XGBoost': train_xgboost(X_train, y_train, random_state=random_state)
    }
    return models

def save_models(models_dict: dict, output_dir: str = None) -> None:
    """Persist trained models to disk using joblib."""
    if output_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        output_dir = project_root / "models"
    else:
        output_dir = Path(output_dir)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, model in models_dict.items():
        filename = name.lower().replace(' ', '_') + '.pkl'
        joblib.dump(model, output_dir / filename)

def load_models(input_dir: str = None) -> dict:
    """Load persisted models from disk."""
    if input_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        input_dir = project_root / "models"
    else:
        input_dir = Path(input_dir)
        
    models = {}
    mapping = {
        'logistic_regression.pkl': 'Logistic Regression',
        'random_forest.pkl': 'Random Forest',
        'xgboost.pkl': 'XGBoost'
    }
    for file, name in mapping.items():
        filepath = input_dir / file
        if filepath.exists():
            models[name] = joblib.load(filepath)
    return models
